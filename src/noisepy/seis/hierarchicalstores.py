import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datetimerange import DateTimeRange

from .datatypes import AnnotatedData, ChannelType, CrossCorrelation, Stack, Station
from .stores import CrossCorrelationDataStore, timespan_str
from .utils import TimeLogger, remove_nan_rows, remove_nans, unstack

CHANNELS_ATTR = "channels"
STACKS_ATTR = "stacks"
VERSION_ATTR = "version"

logger = logging.getLogger(__name__)


class ArrayStore(ABC):
    """
    An interface definition for reading and writing arrays with metadata
    """

    @abstractmethod
    def append(self, path: str, params: Dict[str, Any], data: np.ndarray):
        pass

    @abstractmethod
    def load_paths(self) -> List[str]:
        pass

    @abstractmethod
    def read(self, path: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        pass


class HierarchicalCCStoreBase(CrossCorrelationDataStore):
    """
    CrossCorrelationDataStore that uses hierarchical files for storage. The directory organization is as follows:
    /                           (root)
        station_pair
            timestamp

        The 'timestamp' files contain the cross-correlation data for all channels. The channel information is stored
        as part of the array metadata.
    """

    def __init__(self, helper: ArrayStore) -> None:
        super().__init__()
        self.helper = helper
        self._lock = threading.Lock()
        tlog = TimeLogger(logger, logging.INFO)
        self.path_cache = set(helper.load_paths())
        tlog.log(f"keys_cache initialized with {len(self.path_cache)} keys")

    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        del state["_lock"]
        return state

    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def contains(self, timespan: DateTimeRange, src: Station, rec: Station) -> bool:
        path = self._get_path(timespan, src, rec)
        with self._lock:
            return path in self.path_cache

    def append(
        self,
        timespan: DateTimeRange,
        src: Station,
        rec: Station,
        ccs: List[CrossCorrelation],
    ):
        path = self._get_path(timespan, src, rec)
        all_ccs, metadata = AnnotatedData.pack(ccs)
        tlog = TimeLogger(logger, logging.DEBUG)
        self.helper.append(path, {CHANNELS_ATTR: metadata, VERSION_ATTR: 1.0}, all_ccs)
        tlog.log(f"writing {len(ccs)} CCs to {path}")

        with self._lock:
            self.path_cache.add(path)

    @abstractmethod
    def get_timespans(self) -> List[DateTimeRange]:
        pass

    @abstractmethod
    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        pass

    def read_correlations(self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station) -> List[CrossCorrelation]:
        path = self._get_path(timespan, src_sta, rec_sta)

        tuple = self.helper.read(path)
        if not tuple:
            return []
        array, metadata = tuple

        channel_params = self._read_cc_metadata(metadata)
        cc_stack = array[:]
        ccs = unstack(cc_stack)

        return [
            CrossCorrelation(src, rec, params, remove_nan_rows(data))
            for (src, rec, params), data in zip(channel_params, ccs)
        ]

    def _read_cc_metadata(self, metadata: Dict[str, Any]):
        channels_dict = metadata[CHANNELS_ATTR]
        channel_params = [
            (ChannelType(src, src_loc), ChannelType(rec, rec_loc), params)
            for src, src_loc, rec, rec_loc, params in channels_dict
        ]

        return channel_params

    def _get_path(self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station) -> str:
        stations = self._get_station_pair(src_sta, rec_sta)
        times = timespan_str(timespan)
        return f"{stations}/{times}"


class HierarchicalStackStoreBase:
    """
    A class for reading and writing stack data files. Hierarchy is:
    /
        station_pair            (group)
            stack name          (group)
                component       (array)
    """

    def __init__(self, store: ArrayStore) -> None:
        super().__init__()
        self.helper = store

    def append(self, src: Station, rec: Station, stacks: List[Stack]):
        path = self._get_station_path(src, rec)
        all_stacks, metadata = AnnotatedData.pack(stacks)
        tlog = TimeLogger(logger, logging.DEBUG)
        self.helper.append(path, {STACKS_ATTR: metadata, VERSION_ATTR: 1.0}, all_stacks)
        tlog.log(f"writing {len(stacks)} stacks to {path}")

    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        return self.helper.get_station_pairs()

    def read_stacks(self, src: Station, rec: Station) -> List[Stack]:
        path = self._get_station_path(src, rec)
        tuple = self.helper.read(path)
        if not tuple:
            return []
        arrays, metadata = tuple
        stacks = unstack(arrays)
        stacks_params = metadata[STACKS_ATTR]
        return [
            Stack(comp, name, params, remove_nans(data)) for (comp, name, params), data in zip(stacks_params, stacks)
        ]

    def _get_station_path(self, src: Station, rec: Station) -> str:
        return f"{src}_{rec}"
