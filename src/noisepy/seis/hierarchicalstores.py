import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from datetimerange import DateTimeRange

from noisepy.seis.datatypes import ChannelType, CrossCorrelation, Station
from noisepy.seis.stores import CrossCorrelationDataStore, timespan_str
from noisepy.seis.utils import TimeLogger, remove_nan_rows, unstack

CHANNELS_ATTR = "channels"
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
            # The keys have the full path to the files so we do a prefix check
            # since the path is just the directory name
            return any(map(lambda k: path in k, self.path_cache))

    def append(
        self,
        timespan: DateTimeRange,
        src: Station,
        rec: Station,
        ccs: List[CrossCorrelation],
    ):
        path = self._get_path(timespan, src, rec)
        # Some channels may have different lengths, so pad them with NaNs for stacking
        max_rows = max(d.data.shape[0] for d in ccs)
        max_cols = max(d.data.shape[1] for d in ccs)
        padded = [
            np.pad(
                d.data,
                ((0, max_rows - d.data.shape[0]), (0, max_cols - d.data.shape[1])),
                mode="constant",
                constant_values=np.nan,
            )
            for d in ccs
        ]
        all_ccs = np.stack(padded, axis=0)
        json_params = [(p.src.name, p.src.location, p.rec.name, p.rec.location, p.parameters) for p in ccs]
        tlog = TimeLogger(logger, logging.DEBUG)
        self.helper.append(path, {CHANNELS_ATTR: json_params, VERSION_ATTR: 1.0}, all_ccs)
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

    def mark_done(self, src: Station, rec: Station):
        path = self._get_station_path(src, rec)
        return self.helper.mark_done(path)

    def is_done(self, src: Station, rec: Station):
        path = self._get_station_path(src, rec)
        return self.helper.is_done(path)

    def append(
        self, src: Station, rec: Station, component: str, name: str, stack_params: Dict[str, Any], data: np.ndarray
    ):
        path = self._get_path(src, rec, component, name)
        self.helper.append(path, stack_params, data)

    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        return self.helper.get_station_pairs()

    def get_stack_names(self, src: Station, rec: Station) -> List[str]:
        path = self._get_station_path(src, rec)
        return self._get_children(path)

    def get_components(self, src: Station, rec: Station, name: str) -> List[str]:
        path = self._get_station_path(src, rec) + "/" + name
        return self._get_children(path)

    def read(self, src: Station, rec: Station, component: str, name: str) -> Tuple[Dict[str, Any], np.ndarray]:
        path = self._get_path(src, rec, component, name)
        tuple = self.helper.read(path)
        if not tuple:
            return {}, np.zeros((0,))
        return tuple[1], tuple[0]

    def _get_children(self, path: str) -> List[str]:
        if path not in self.helper.root:
            return []
        return list(self.helper.root[path].array_keys()) + list(self.helper.root[path].group_keys())

    def _get_station_path(self, src: Station, rec: Station) -> str:
        return f"{src}_{rec}"

    def _get_path(self, src: Station, rec: Station, component: str, name: str) -> str:
        return f"{self._get_station_path(src,rec)}/{name}/{component}"
