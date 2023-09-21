import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr
from datetimerange import DateTimeRange

from noisepy.seis.constants import DONE_PATH
from noisepy.seis.datatypes import Station
from noisepy.seis.hierarchicalstores import (
    ArrayStore,
    HierarchicalCCStoreBase,
    HierarchicalStackStoreBase,
)

from .stores import parse_station_pair, parse_timespan

logger = logging.getLogger(__name__)


class ZarrStoreHelper(ArrayStore):
    """
    Helper object for storing data and parameters into Zarr and track a "done" bit for subsets of the data
    'done' is a dummy array to track completed sets in its attribute dictionary.
    Args:
        root_dir: Storage location, can be a local or S3 path
        mode: "r" or "a" for read-only or writing mode
        storage_options: options to pass to fsspec
    """

    def __init__(self, root_dir: str, mode: str, storage_options={}) -> None:
        super().__init__()
        # We don't want to cache the data, but we do want to use the keys() cache
        CACHE_SIZE = 1
        logger.info(
            f"store creating at {root_dir}, mode={mode}, storage_options={storage_options}, cache_size={CACHE_SIZE}"
        )
        # TODO:
        storage_options["client_kwargs"] = {"region_name": "us-west-2"}
        store = zarr.storage.FSStore(root_dir, **storage_options)
        self.cache = zarr.LRUStoreCache(store, max_size=CACHE_SIZE)
        self.root = zarr.open_group(self.cache, mode=mode)
        self.clean_regex = re.compile("/.z.*")
        logger.info(f"store created at {root_dir}: {type(store)}. Zarr version {zarr.__version__}")

    def load_paths(self) -> List[str]:
        # the keys() returned contains the full file paths, so we remove suffixes like /.zgroup, /.zarray
        return [self.clean_regex.sub("", key) for key in self.cache.keys()]

    def append(self, path: str, params: Dict[str, Any], data: np.ndarray):
        logger.debug(f"Appending to {path}: {data.shape}")
        array = self.root.create_dataset(
            path,
            data=data,
            chunks=data.shape,
            dtype=data.dtype,
        )
        array.attrs.update(params)

    def read(self, path: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        if path not in self.root:
            return None
        array = self.root[path]
        metadata = {}
        metadata.update(array.attrs)
        return (array[:], metadata)


class ZarrCCStore(HierarchicalCCStoreBase):
    def __init__(self, root_dir: str, mode: str = "a", storage_options={}) -> None:
        helper = ZarrStoreHelper(root_dir, mode, storage_options=storage_options)
        super().__init__(helper)

    def get_timespans(self) -> List[DateTimeRange]:
        pairs = [k for k in self.helper.root.group_keys() if k != DONE_PATH]
        timespans = []
        for p in pairs:
            timespans.extend(k for k in self.helper.root[p].array_keys())
        return list(parse_timespan(t) for t in sorted(set(timespans)))

    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        return [parse_station_pair(k) for k in self.helper.root.group_keys() if k != DONE_PATH]


class ZarrStackStore(HierarchicalStackStoreBase):
    def __init__(self, root_dir: str, mode: str = "a", storage_options={}) -> None:
        helper = ZarrStoreHelper(root_dir, mode, storage_options=storage_options)
        super().__init__(helper)

    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        return [parse_station_pair(k) for k in self.helper.root.array_keys() if k != DONE_PATH]
