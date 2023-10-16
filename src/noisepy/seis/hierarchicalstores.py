import logging
import threading
from abc import ABC, abstractmethod
from bisect import bisect
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import fsspec
import numpy as np
from datetimerange import DateTimeRange

from noisepy.seis.utils import io_retry

from .datatypes import AnnotatedData, Station
from .stores import timespan_str
from .utils import TimeLogger, fs_join, get_filesystem, unstack

META_ATTR = "metadata"
VERSION_ATTR = "version"
FAKE_STA = "FAKE_STATION"

logger = logging.getLogger(__name__)


class ArrayStore(ABC):
    """
    An interface definition for reading and writing arrays with metadata
    """

    def __init__(self, root_dir: str, storage_options={}) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.fs = get_filesystem(root_dir, storage_options=storage_options)

    @abstractmethod
    def append(self, path: str, params: Dict[str, Any], data: np.ndarray):
        pass

    @abstractmethod
    def read(self, path: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        pass

    @abstractmethod
    def parse_path(path: str) -> Optional[Tuple[str, DateTimeRange]]:
        """
        Parse a full file path into a receiving station and timespan tuple
        """

    def get_fs(self) -> fsspec.AbstractFileSystem:
        return self.fs

    def get_root_dir(self) -> str:
        return self.root_dir


class PairDirectoryCache:
    """
    Data structure to store the timespans for each station pair. The data is stored in a nested dictionary:
    First, stations are mapped to an integer index to save memory. The first dictionary is indexed by the source station
    and each entry holds a dictiory keyed by receiver station. The value of the second dictionary is a list of tuples.
    Each tuple is a pair of the time delta in the timespans and a numpy array of the start times of the timespans.
    E.g.
    .. code-block:: python
        stations_idx: {"CI.ACP": 0}
        idx_stations: ["CI.ACP"]
        items:
            {0: {0: [(86400, np.array([1625097600, 1625184000]))]}

    We keep the timespans in a sorted list so that we can use binary search to check if a timespan is contained. A
    python dictionary would be O(1) to check but use a lot more memory. Also the timestamp (np.uint32) is a lot
    more compact to store than the string representation of the timespan
    (e.g. "2023_07_01_00_00_00T2023_07_02_00_00_00").
    """

    def __init__(self) -> None:
        self.items: Dict[int, Dict[int, List[Tuple[int, np.ndarray]]]] = defaultdict(lambda: defaultdict(list))
        self.stations_idx: Dict[str, int] = {}
        self.idx_stations: List[str] = []
        self._lock: threading.Lock = threading.Lock()

    # We need to be able to pickle this across processes when using multiprocessing
    def __getstate__(self) -> object:
        state = self.__dict__.copy()
        del state["_lock"]
        del state["items"]
        return state

    def __setstate__(self, state: object) -> None:
        self.__dict__.update(state)
        self._lock = threading.Lock()
        self.items = defaultdict(lambda: defaultdict(list))

    def _sta_index(self, sta: str) -> int:
        idx = self.stations_idx.get(sta, -1)
        if idx == -1:
            with self._lock:
                # check again in case another thread added it before we got the lock
                idx = self.stations_idx.get(sta, -1)
                if idx == -1:
                    idx = len(self.stations_idx)
                    self.stations_idx[sta] = idx
                    self.idx_stations.append(sta)
        return idx

    def get_pairs(self) -> List[Tuple[str, str]]:
        return [
            (self.idx_stations[src], self.idx_stations[rec])
            for src in sorted(self.items.keys())
            for rec in sorted(self.items.get(src, {}).keys())
        ]

    def add(self, src: str, rec: str, timespans: List[DateTimeRange]):
        src_idx = self._sta_index(src)
        rec_idx = self._sta_index(rec)
        if len(timespans) == 0:
            if not self.items.get(src_idx, {}).get(rec_idx, None):
                self.items[src_idx][rec_idx] = []
            return
        grouped_timespans = defaultdict(list)
        for t in timespans:
            grouped_timespans[int(t.timedelta.total_seconds())].append(int(t.start_datetime.timestamp()))
        for delta, starts in grouped_timespans.items():
            self._add(src_idx, rec_idx, delta, starts)

    def _add(self, src_idx: int, rec_idx: int, delta: int, start_ts: List[DateTimeRange]):
        starts = np.array(sorted(start_ts), dtype=np.uint32)
        with self._lock:
            time_tuples = self.items[src_idx][rec_idx]
            for t in time_tuples:
                if t[0] == delta:
                    new_t = (delta, np.array(sorted(list(t[1]) + list(starts)), dtype=np.uint32))
                    # not generally safe to modify a list while iterating over it, but we return after this
                    time_tuples.remove(t)
                    time_tuples.append(new_t)
                    self.items[src_idx][rec_idx] = time_tuples
                    return
            # delta not found, so add it
            self.items[src_idx][rec_idx].append((delta, starts))

    def is_src_loaded(self, src: str) -> bool:
        return self._sta_index(src) in self.items

    def contains(self, src: str, rec: str, timespan: DateTimeRange) -> bool:
        time_tuples = self._get_tuples(src, rec)
        if time_tuples is None:
            return False

        delta = int(timespan.timedelta.total_seconds())
        start = int(timespan.start_datetime.timestamp())
        for t in time_tuples:
            if t[0] == delta:
                result = bisect(t[1], start)
                return result != 0 and t[1][result - 1] == start

        return False

    def get_timespans(self, src: str, rec: str) -> List[DateTimeRange]:
        time_tuples = self._get_tuples(src, rec)
        if time_tuples is None:
            return []

        timespans = []
        for delta, timestamps in time_tuples:
            for ts in timestamps:
                timespans.append(
                    DateTimeRange(
                        datetime.fromtimestamp(ts, timezone.utc),
                        datetime.fromtimestamp(ts + delta, timezone.utc),
                    )
                )
        return timespans

    def _get_tuples(self, src: str, rec: str) -> List[Tuple[int, np.ndarray]]:
        src_idx = self._sta_index(src)
        rec_idx = self._sta_index(rec)

        with self._lock:
            time_tuples = self.items.get(src_idx, {}).get(rec_idx, None)
            return time_tuples


T = TypeVar("T", bound=AnnotatedData)


class HierarchicalStoreBase(Generic[T]):
    """
    A CC and Stack store bases class that uses hierarchical files for storage. The directory organization is as follows:
    /
        src_sta/
                rec_sta/
                    timespan

        The specific file format stored at the timespan level is delegated to the ArrayStore helper class.
    """

    def __init__(
        self,
        helper: ArrayStore,
        loader_func: Callable[[List[Tuple[np.ndarray, Dict[str, Any]]]], List[T]],
    ) -> None:
        super().__init__()
        self.helper = helper
        self.dir_cache = PairDirectoryCache()
        self.loader_func = loader_func

    def contains(self, src_sta: Station, rec_sta: Station, timespan: DateTimeRange) -> bool:
        src = str(src_sta)
        rec = str(rec_sta)
        if self.dir_cache.contains(src, rec, timespan):
            return True
        self._load_src(src)
        return self.dir_cache.contains(src, rec, timespan)

    def _load_src(self, src: str):
        if self.dir_cache.is_src_loaded(src):
            return
        logger.info(f"Loading directory cache for {src} - ix: {self.dir_cache.stations_idx.get(src, -4)}")
        paths = io_retry(self._fs_find, src)

        grouped_paths = defaultdict(list)
        for rec_sta, timespan in [p for p in paths if p]:
            grouped_paths[rec_sta].append(timespan)
        for rec_sta, timespans in grouped_paths.items():
            self.dir_cache.add(src, rec_sta, sorted(timespans, key=lambda t: t.start_datetime.timestamp()))
        # if we didn't find any paths, add a fake entry so we don't try again and is_src_loaded returns True
        if len(grouped_paths) == 0:
            self.dir_cache.add(src, FAKE_STA, [])

    def _fs_find(self, src):
        paths = [self.helper.parse_path(p) for p in self.helper.get_fs().find(fs_join(self.helper.get_root_dir(), src))]
        return paths

    def append(self, timespan: DateTimeRange, src: Station, rec: Station, data: List[T]):
        path = self._get_path(src, rec, timespan)
        packed_data, metadata = AnnotatedData.pack(data)
        tlog = TimeLogger(logger, logging.DEBUG)
        self.helper.append(path, {META_ATTR: metadata, VERSION_ATTR: 1.0}, packed_data)
        tlog.log(f"writing {len(data)} arrays to {path}")
        self.dir_cache.add(str(src), str(rec), [timespan])

    def get_timespans(self, src: Station, rec: Station) -> List[DateTimeRange]:
        self._load_src(str(src))
        return self.dir_cache.get_timespans(str(src), str(rec))

    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        if not self.helper.get_fs().exists(self.helper.get_root_dir()):
            return []  # fsspec.ls throws if the directory doesn't exist

        def is_sta(path):
            return self.helper.get_fs().isdir(path) and Station.parse(Path(path).parts[-1])

        # ls gets us the first level
        sta = self.helper.get_fs().ls(self.helper.get_root_dir())
        # make sure we have stations
        sta = [s for s in sta if is_sta(s)]
        # ls the second level in parallel
        with ThreadPoolExecutor() as exec:
            sub_ls = list(exec.map(self.helper.get_fs().ls, sta))

        sub_ls = [item for sublist in sub_ls for item in sublist if is_sta(item)]
        pairs = [(Station.parse(Path(p).parts[-2]), Station.parse(Path(p).parts[-1])) for p in sub_ls]
        return [p for p in pairs if p[0] and p[1]]

    def read(self, timespan: DateTimeRange, src: Station, rec: Station) -> List[T]:
        path = self._get_path(src, rec, timespan)
        tuple = self.helper.read(path)
        if not tuple:
            return []
        array, metadata = tuple
        arrays = unstack(array)
        meta = metadata[META_ATTR]
        tuples = list(zip(arrays, meta))
        return self.loader_func(tuples)

    def _get_path(self, src: Station, rec: Station, timespan: DateTimeRange) -> str:
        return f"{src}/{rec}/{timespan_str(timespan)}"
