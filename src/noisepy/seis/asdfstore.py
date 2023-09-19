import glob
import logging
import os
from pathlib import Path
from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np
import obspy
import pyasdf
from datetimerange import DateTimeRange

from . import noise_module
from .constants import PROGRESS_DATATYPE
from .datatypes import (
    Channel,
    ChannelData,
    ChannelType,
    CrossCorrelation,
    Stack,
    Station,
)
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
        if mode not in ["a", "r"]:
            raise ValueError(f"Invalid mode {mode}. Must be 'a' or 'r'")

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

    def contains(self, key: T, data_type: str, path: str = None):
        ccf_ds = self[key]

        if not ccf_ds:
            return False
        with ccf_ds:
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
    def contains(self, timespan: DateTimeRange, src: Station, rec: Station) -> bool:
        station_pair = self._get_station_pair(src, rec)
        contains = self.datasets.contains(timespan, station_pair)
        if contains:
            logger.info(f"Cross-correlation {station_pair} already exists")
        return contains

    def append(
        self,
        timespan: DateTimeRange,
        src: Station,
        rec: Station,
        ccs: List[CrossCorrelation],
    ):
        for cc in ccs:
            station_pair = self._get_station_pair(src, rec)
            # source-receiver pair: e.g. CI.ARV_CI.BAK
            # channels, e.g. bhn_bhn
            channels = self._get_channel_pair(cc.src, cc.rec)
            self.datasets.add_aux_data(timespan, cc.parameters, station_pair, channels, cc.data)

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

    def read_correlations(self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station) -> List[CrossCorrelation]:
        with self.datasets[timespan] as ccf_ds:
            dtype = self._get_station_pair(src_sta, rec_sta)
            if dtype not in ccf_ds.auxiliary_data:
                logging.warning(f"No data available for {timespan}/{dtype}")
                return []
            ccs = []
            ch_pair_paths = ccf_ds.auxiliary_data[dtype].list()
            for ch_pair_path in ch_pair_paths:
                src_ch, rec_ch = _parse_channel_path(ch_pair_path)
                stream = ccf_ds.auxiliary_data[dtype][ch_pair_path]
                ccs.append(CrossCorrelation(src_ch, rec_ch, stream.parameters, stream.data[:]))
            return ccs


class ASDFStackStore(StackStore):
    def __init__(self, directory: str, mode: str = "a"):
        super().__init__()
        self.datasets = ASDFDirectory(directory, mode, _filename_from_stations, _parse_station_pair_h5file)

    def append(self, src: Station, rec: Station, stacks: List[Stack]):
        for stack in stacks:
            self.datasets.add_aux_data((src, rec), stack.parameters, stack.name, stack.component, stack.data)

    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        return self.datasets.get_keys()

    def read_stacks(self, src: Station, rec: Station) -> List[Stack]:
        stacks = []
        with self.datasets[(src, rec)] as ds:
            for name in ds.auxiliary_data.list():
                for component in ds.auxiliary_data[name].list():
                    stream = ds.auxiliary_data[name][component]
                    stacks.append(Stack(component, name, stream.parameters, stream.data[:]))
        return stacks


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


def _parse_channel_path(path: str) -> Tuple[ChannelType, ChannelType]:
    parts = path.split("_")
    if len(parts) == 2:  # e.g. bhn_bhn
        return tuple(map(ChannelType, parts))
    elif len(parts) == 3:  # when we have one location code
        if parts[1].isdigit():  # e.g. bhn_00_bhn
            return tuple(map(ChannelType, ["_".join(parts[0:2]), parts[2]]))
        else:  # e.g. bhn_bhn_00
            return tuple(map(ChannelType, [parts[0], "_".join(parts[1:3])]))
    elif len(parts) == 4:  # when we have two location codes: e.g. bhn_00_bhn_00
        return tuple(map(ChannelType, ["_".join(parts[0:2]), "_".join(parts[2:4])]))
    else:
        raise ValueError(f"Invalid channel path {path}")
