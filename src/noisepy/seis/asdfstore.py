import datetime
import glob
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import obspy
import pyasdf
from datetimerange import DateTimeRange

from . import noise_module
from .constants import DATE_FORMAT, DONE_PATH, PROGRESS_DATATYPE
from .datatypes import Channel, ChannelData, ChannelType, Station
from .stores import CrossCorrelationDataStore, RawDataStore

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
        return _get_dataset_cached(file_path, self.mode)

    def get_keys(self) -> List[T]:
        h5files = sorted(glob.glob(os.path.join(self.directory, "*.h5")))
        return list(map(self.parse_filename, h5files))

    def mark_done(self, key: T):
        self.add_aux_data(key, {}, PROGRESS_DATATYPE, DONE_PATH, np.zeros(0))

    def is_done(self, key: T):
        done = self.contains(key, PROGRESS_DATATYPE, DONE_PATH)
        if done:
            logger.info(f"{key} already computed")
        return done

    def contains(self, key: T, data_type: str, path: str = None):
        ccf_ds = self[key]
        if not ccf_ds:
            return False
        # source-receiver pair
        exists = data_type in ccf_ds.auxiliary_data
        if path is not None and exists:
            return path in ccf_ds.auxiliary_data[data_type]
        return exists

    def add_aux_data(self, key: T, params: Dict, data_type: str, path: str, data: np.ndarray):
        ccf_ds = self[key]
        ccf_ds.add_auxiliary_data(data=data, data_type=data_type, path=path, parameters=params)


class ASDFRawDataStore(RawDataStore):
    """
    A data store implementation to read from a directory of ASDF files. Each file is considered
    a timespan with the naming convention: 2019_02_01_00_00_00T2019_02_02_00_00_00.h5
    """

    def __init__(self, directory: str, mode: str = "r"):
        super().__init__()
        self.datasets = ASDFDirectory(directory, mode, _filename_from_timespan, _parse_timespan)

    def get_channels(self, timespan: DateTimeRange) -> List[Channel]:
        ds = self.datasets[timespan]
        stations = [self._create_station(timespan, sta) for sta in ds.waveforms.list() if sta is not None]
        channels = [
            Channel(ChannelType(tag), sta) for sta in stations for tag in ds.waveforms[str(sta)].get_waveform_tags()
        ]
        return channels

    def get_timespans(self) -> List[DateTimeRange]:
        return self.datasets.get_keys()

    def read_data(self, timespan: DateTimeRange, chan: Channel) -> np.ndarray:
        ds = self.datasets[timespan]
        stream = ds.waveforms[str(chan.station)][str(chan.type)]
        return ChannelData(stream)

    def get_inventory(self, timespan: DateTimeRange, station: Station) -> obspy.Inventory:
        ds = self.datasets[timespan]
        return ds.waveforms[str(station)]["StationXML"]

    def _create_station(self, timespan: DateTimeRange, name: str) -> Optional[Station]:
        # What should we do if there's no StationXML?
        try:
            inventory = self.datasets[timespan].waveforms[name]["StationXML"]
            sta, net, lon, lat, elv, loc = noise_module.sta_info_from_inv(inventory)
            return Station(net, sta, lat, lon, elv, loc)
        except Exception as e:
            logger.warning(f"Missing StationXML for station {name}. {e}")
            return None


class ASDFCCStore(CrossCorrelationDataStore):
    def __init__(self, directory: str, mode: str = "a") -> None:
        super().__init__()
        Path(directory).mkdir(exist_ok=True)
        self.datasets = ASDFDirectory(directory, mode, _filename_from_timespan, _parse_timespan)

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

    def get_station_pairs(self, timespan: DateTimeRange) -> List[Tuple[Station, Station]]:
        ccf_ds = self.datasets[timespan]
        data = ccf_ds.auxiliary_data.list()
        return [_parse_station_pair(p) for p in data if p != PROGRESS_DATATYPE]

    def get_channeltype_pairs(
        self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station
    ) -> List[Tuple[Channel, Channel]]:
        ccf_ds = self.datasets[timespan]
        dtype = self._get_station_pair(src_sta, rec_sta)
        ch_pairs = ccf_ds.auxiliary_data[dtype].list()
        return [tuple(map(ChannelType, ch.split("_"))) for ch in ch_pairs]

    def read(
        self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station, src_ch: ChannelType, rec_ch: ChannelType
    ) -> Tuple[Dict, np.ndarray]:
        dtype = self._get_station_pair(src_sta, rec_sta)
        path = self._get_channel_pair(src_ch, rec_ch)
        stream = self.datasets[timespan].auxiliary_data[dtype][path]
        return (stream.parameters, stream.data)

    # private helper methods

    def _get_station_pair(self, src_sta: Station, rec_sta: Station) -> str:
        return f"{src_sta}_{rec_sta}"

    def _get_channel_pair(self, src_chan: ChannelType, rec_chan: ChannelType) -> str:
        return f"{src_chan.name}_{rec_chan.name}"


@lru_cache
def _get_dataset_cached(filename: str, mode: str) -> pyasdf.ASDFDataSet:
    logger.info(f"ASDFCCStore - Opening {filename}")
    if os.path.exists(filename):
        return pyasdf.ASDFDataSet(filename, mode=mode, mpi=False, compression=None)
    elif mode == "r":
        return None
    else:  # create new file
        return pyasdf.ASDFDataSet(filename, mode=mode, mpi=False, compression=None)


def _parse_timespan(filename: str) -> DateTimeRange:
    parts = os.path.splitext(os.path.basename(filename))[0].split("T")
    dates = [obspy.UTCDateTime(p).datetime.replace(tzinfo=datetime.timezone.utc) for p in parts]
    return DateTimeRange(dates[0], dates[1])


def _filename_from_timespan(timespan: DateTimeRange) -> str:
    return f"{timespan.start_datetime.strftime(DATE_FORMAT)}T{timespan.end_datetime.strftime(DATE_FORMAT)}.h5"


def _filename_from_stations(pair: Tuple[Station, Station]) -> str:
    return f"{pair[0]}_{pair[1]}.h5"


def _parse_station_pair(pair: str) -> Tuple[Station, Station]:
    # Parse from:'CI.ARV_CI.BAK
    def station(sta: str) -> Station:
        net, name = sta.split(".")
        return Station(net, name)

    return tuple(map(station, pair.split("_")))
