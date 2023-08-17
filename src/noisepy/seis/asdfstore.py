import glob
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import obspy
import pyasdf
from datetimerange import DateTimeRange

from . import noise_module
from .constants import DONE_PATH, PROGRESS_DATATYPE
from .datatypes import Channel, ChannelData, ChannelType, Station
from .stores import (
    CrossCorrelationDataStore,
    RawDataStore,
    StackStore,
    parse_station_pair,
    parse_timespan,
    timespan_str,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ASDFDirectory(Generic[T]):
    """
    Utility class used byt ASDFRawDataStore and ASDFCCStore to provide easy access
    to a set of ASDF files in a directory that follow a specific naming convention.
    The files are named after a generic type T (e.g. a timestamp or a pair of stations)
    so the constructor takes two functions to map between the type T and the corresponding
    file name.
    """

    def __init__(
        self, directory: str, mode: str, get_filename: Callable[[T], str], parse_filename: Callable[[str], T]
    ) -> None:
        self.directory = directory
        self.mode = mode
        self.get_filename = get_filename
        self.parse_filename = parse_filename

    def __getitem__(self, key: T) -> pyasdf.ASDFDataSet:
        file_name = self.get_filename(key)
        file_path = os.path.join(self.directory, file_name)
        return _get_dataset(file_path, self.mode)

    def get_keys(self) -> List[T]:
        h5files = sorted(glob.glob(os.path.join(self.directory, "**/*.h5"), recursive=True))
        return list(map(self.parse_filename, h5files))

    def mark_done(self, key: T):
        self.add_aux_data(key, {}, PROGRESS_DATATYPE, DONE_PATH, np.zeros(0))

    def is_done(self, key: T):
        done = self.contains(key, PROGRESS_DATATYPE, DONE_PATH)
        if done:
            logger.info(f"{key} already computed")
        return done

    def contains(self, key: T, data_type: str, path: str = None):
        with self[key] as ccf_ds:
            if not ccf_ds:
                return False
            # source-receiver pair
            exists = data_type in ccf_ds.auxiliary_data
            if path is not None and exists:
                return path in ccf_ds.auxiliary_data[data_type]
            return exists

    def add_aux_data(self, key: T, params: Dict, data_type: str, path: str, data: np.ndarray):
        with self[key] as ccf_ds:
            ccf_ds.add_auxiliary_data(data=data, data_type=data_type, path=path, parameters=params)


class ASDFRawDataStore(RawDataStore):
    """
    A data store implementation to read from a directory of ASDF files. Each file is considered
    a timespan with the naming convention: 2019_02_01_00_00_00T2019_02_02_00_00_00.h5
    """

    def __init__(self, directory: str, mode: str = "r"):
        super().__init__()
        self.datasets = ASDFDirectory(directory, mode, _filename_from_timespan, parse_timespan)

    def get_channels(self, timespan: DateTimeRange) -> List[Channel]:
        with self.datasets[timespan] as ds:
            stations = [self._create_station(timespan, sta) for sta in ds.waveforms.list() if sta is not None]
            channels = [
                Channel(ChannelType(tag), sta) for sta in stations for tag in ds.waveforms[str(sta)].get_waveform_tags()
            ]
            return channels

    def get_timespans(self) -> List[DateTimeRange]:
        return self.datasets.get_keys()

    def read_data(self, timespan: DateTimeRange, chan: Channel) -> ChannelData:
        with self.datasets[timespan] as ds:
            stream = ds.waveforms[str(chan.station)][str(chan.type)]
        return ChannelData(stream)

    def get_inventory(self, timespan: DateTimeRange, station: Station) -> obspy.Inventory:
        with self.datasets[timespan] as ds:
            return ds.waveforms[str(station)]["StationXML"]

    def _create_station(self, timespan: DateTimeRange, name: str) -> Optional[Station]:
        # What should we do if there's no StationXML?
        try:
            with self.datasets[timespan] as ds:
                inventory = ds.waveforms[name]["StationXML"]
                sta, net, lon, lat, elv, loc = noise_module.sta_info_from_inv(inventory)
                return Station(net, sta, lat, lon, elv, loc)
        except Exception as e:
            logger.warning(f"Missing StationXML for station {name}. {e}")
            return None


class ASDFCCStore(CrossCorrelationDataStore):
    def __init__(self, directory: str, mode: str = "a") -> None:
        super().__init__()
        Path(directory).mkdir(exist_ok=True)
        self.datasets = ASDFDirectory(directory, mode, _filename_from_timespan, parse_timespan)

    # CrossCorrelationDataStore implementation
    def contains(self, timespan: DateTimeRange, src_chan: Channel, rec_chan: Channel) -> bool:
        station_pair = self._get_station_pair(src_chan.station, rec_chan.station)
        channel_pair = self._get_channel_pair(src_chan.type, rec_chan.type)
        logger.debug(f"station pair {station_pair} channel pair {channel_pair}")
        contains = self.datasets.contains(timespan, station_pair, channel_pair)
        if contains:
            logger.info(f"Cross-correlation {station_pair} and {channel_pair} already exists")
        return contains

    def append(
        self,
        timespan: DateTimeRange,
        src_chan: Channel,
        rec_chan: Channel,
        cc_params: Dict[str, Any],
        corr: np.ndarray,
    ):
        # source-receiver pair: e.g. CI.ARV_CI.BAK
        station_pair = self._get_station_pair(src_chan.station, rec_chan.station)
        # channels, e.g. bhn_bhn
        channels = self._get_channel_pair(src_chan.type, rec_chan.type)
        self.datasets.add_aux_data(timespan, cc_params, station_pair, channels, corr)

    def mark_done(self, timespan: DateTimeRange):
        self.datasets.mark_done(timespan)

    def is_done(self, timespan: DateTimeRange):
        return self.datasets.is_done(timespan)

    def get_timespans(self) -> List[DateTimeRange]:
        return self.datasets.get_keys()

    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        timespans = self.get_timespans()
        pairs_all = set()
        for timespan in timespans:
            with self.datasets[timespan] as ccf_ds:
                data = ccf_ds.auxiliary_data.list()
                pairs_all.update(parse_station_pair(p) for p in data if p != PROGRESS_DATATYPE)
        return list(pairs_all)

    def get_channeltype_pairs(
        self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station
    ) -> List[Tuple[Channel, Channel]]:
        with self.datasets[timespan] as ccf_ds:
            dtype = self._get_station_pair(src_sta, rec_sta)
            if dtype not in ccf_ds.auxiliary_data:
                logging.warning(f"No data available for {timespan}/{dtype}")
                return []
            ch_pairs = ccf_ds.auxiliary_data[dtype].list()
            return [tuple(map(ChannelType, ch.split("_"))) for ch in ch_pairs]

    def read(
        self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station, src_ch: ChannelType, rec_ch: ChannelType
    ) -> Tuple[Dict, np.ndarray]:
        dtype = self._get_station_pair(src_sta, rec_sta)
        path = self._get_channel_pair(src_ch, rec_ch)
        with self.datasets[timespan] as ds:
            if dtype not in ds.auxiliary_data or path not in ds.auxiliary_data[dtype]:
                return ({}, np.empty((0, 0)))
            stream = ds.auxiliary_data[dtype][path]
            return (stream.parameters, stream.data[:])


class ASDFStackStore(StackStore):
    def __init__(self, directory: str, mode: str = "a"):
        super().__init__()
        self.datasets = ASDFDirectory(directory, mode, _filename_from_stations, _parse_station_pair_h5file)

    def mark_done(self, src: Station, rec: Station):
        self.datasets.mark_done((src, rec))

    def is_done(self, src: Station, rec: Station):
        return self.datasets.is_done((src, rec))

    def append(
        self, src: Station, rec: Station, components: str, name: str, stack_params: Dict[str, Any], data: np.ndarray
    ):
        self.datasets.add_aux_data((src, rec), stack_params, name, components, data)

    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        return self.datasets.get_keys()

    def get_stack_names(self, src: Station, rec: Station) -> List[str]:
        with self.datasets[(src, rec)] as ds:
            return [name for name in ds.auxiliary_data.list() if name != PROGRESS_DATATYPE]

    def get_components(self, src: Station, rec: Station, name: str) -> List[str]:
        with self.datasets[(src, rec)] as ds:
            if name not in ds.auxiliary_data:
                logger.warning(f"Not data available for {src}_{rec}/{name}")
                return []
            return ds.auxiliary_data[name].list()

    def read(self, src: Station, rec: Station, component: str, name: str) -> Tuple[Dict[str, Any], np.ndarray]:
        with self.datasets[(src, rec)] as ds:
            if name not in ds.auxiliary_data or component not in ds.auxiliary_data[name]:
                logger.warning(f"Not data available for {src}_{rec}/{name}/{component}")
                return ({}, np.empty(0))
            stream = ds.auxiliary_data[name][component]
            return (stream.parameters, stream.data[:])


def _get_dataset(filename: str, mode: str) -> pyasdf.ASDFDataSet:
    logger.debug(f"Opening {filename}")
    if os.path.exists(filename):
        return pyasdf.ASDFDataSet(filename, mode=mode, mpi=False, compression=None)
    elif mode == "r":
        return None
    else:  # create new file
        Path(filename).parent.mkdir(exist_ok=True, parents=True)
        return pyasdf.ASDFDataSet(filename, mode=mode, mpi=False, compression=None)


def _filename_from_stations(pair: Tuple[Station, Station]) -> str:
    return f"{pair[0]}/{pair[0]}_{pair[1]}.h5"


def _filename_from_timespan(timespan: DateTimeRange) -> str:
    return f"{timespan_str(timespan)}.h5"


def _parse_station_pair_h5file(path: str) -> Tuple[Station, Station]:
    pair = Path(path).stem
    return parse_station_pair(pair)
