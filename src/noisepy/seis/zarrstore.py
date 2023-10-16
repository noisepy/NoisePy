import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import zarr
from datetimerange import DateTimeRange

from .datatypes import CrossCorrelation, Stack
from .hierarchicalstores import ArrayStore, HierarchicalStoreBase
from .stores import CrossCorrelationDataStore, StackStore, parse_timespan

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
        super().__init__(root_dir, storage_options)
        logger.info(f"store creating at {root_dir}, mode={mode}, storage_options={storage_options}")
        # TODO:
        storage_options["client_kwargs"] = {"region_name": "us-west-2"}
        store = zarr.storage.FSStore(root_dir, **storage_options)
        self.root = zarr.open_group(store, mode=mode)
        logger.info(f"store created at {root_dir}: {type(store)}. Zarr version {zarr.__version__}")

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

    def parse_path(self, path: str) -> Optional[Tuple[str, DateTimeRange]]:
        if not path.endswith("0.0"):
            return None
        parts = Path(path).parts
        if len(parts) < 3:
            return None
        ts = parse_timespan(parts[-2])
        if ts is None:
            return None
        return (parts[-3], ts)


class ZarrCCStore(HierarchicalStoreBase, CrossCorrelationDataStore):
    def __init__(self, root_dir: str, mode: str = "a", storage_options={}) -> None:
        helper = ZarrStoreHelper(root_dir, mode, storage_options=storage_options)
        super().__init__(helper, CrossCorrelation.load_instances)


class ZarrStackStore(HierarchicalStoreBase, StackStore):
    def __init__(self, root_dir: str, mode: str = "a", storage_options={}) -> None:
        helper = ZarrStoreHelper(root_dir, mode, storage_options=storage_options)
        super().__init__(helper, Stack.load_instances)
