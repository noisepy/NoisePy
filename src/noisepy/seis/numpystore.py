import io
import json
import logging
import tarfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from datetimerange import DateTimeRange

from .datatypes import CrossCorrelation, Stack, to_json_types
from .hierarchicalstores import ArrayStore, HierarchicalStoreBase
from .stores import CrossCorrelationDataStore, StackStore, parse_timespan
from .utils import fs_join

logger = logging.getLogger(__name__)

TAR_GZ_EXTENSION = ".tar.gz"
FILE_ARRAY_NPY = "array.npy"
FILE_PARAMS_JSON = "params.json"


class NumpyArrayStore(ArrayStore):
    def __init__(self, root_dir: str, mode: str, storage_options={}) -> None:
        super().__init__(root_dir, storage_options)
        logger.info(f"store creating at {root_dir}, mode={mode}, storage_options={storage_options}")
        # TODO: This needs to come in as part of the storage_options
        storage_options["client_kwargs"] = {"region_name": "us-west-2"}
        self.root_path = root_dir
        self.storage_options = storage_options
        logger.info(f"Numpy store created at {root_dir}")

    def append(self, path: str, params: Dict[str, Any], data: np.ndarray):
        logger.debug(f"Appending to {path}: {data.shape}")

        params = to_json_types(params)
        js = json.dumps(params)

        def add_file_bytes(tar, name, f):
            f.seek(0)
            ti = tarfile.TarInfo(name=name)
            ti.size = f.getbuffer().nbytes
            tar.addfile(ti, fileobj=f)

        dir = fs_join(self.root_path, str(Path(path).parent))
        self.get_fs().makedirs(dir, exist_ok=True)
        with self.get_fs().open(fs_join(self.root_path, path + TAR_GZ_EXTENSION), "wb") as f:
            with tarfile.open(fileobj=f, mode="w:gz") as tar:
                with io.BytesIO() as npyf:
                    np.save(npyf, data, allow_pickle=False)
                    with io.BytesIO() as jsf:
                        jsf.write(js.encode("utf-8"))

                        add_file_bytes(tar, FILE_ARRAY_NPY, npyf)
                        add_file_bytes(tar, FILE_PARAMS_JSON, jsf)

    def parse_path(self, path: str) -> Optional[Tuple[str, DateTimeRange]]:
        if not path.endswith(TAR_GZ_EXTENSION):
            return None
        path = path.removesuffix(TAR_GZ_EXTENSION)
        parts = Path(path).parts
        if len(parts) < 2:
            return None
        ts = parse_timespan(parts[-1])
        if ts is None:
            return None
        return (parts[-2], ts)

    def read(self, path: str) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        file = fs_join(self.root_path, path + TAR_GZ_EXTENSION)
        if not self.get_fs().exists(file):
            return None

        try:
            with self.get_fs().open(file, "rb") as f:
                with tarfile.open(fileobj=f, mode="r:gz") as tar:
                    npy_mem = tar.getmember(FILE_ARRAY_NPY)
                    with tar.extractfile(npy_mem) as f:
                        array_file = io.BytesIO()
                        array_file.write(f.read())
                        array_file.seek(0)
                        array = np.load(array_file, allow_pickle=False)
                    params_mem = tar.getmember(FILE_PARAMS_JSON)
                    with tar.extractfile(params_mem) as f:
                        params = json.load(f)
                    return (array, params)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            return None


class NumpyStackStore(HierarchicalStoreBase[Stack], StackStore):
    def __init__(self, root_dir: str, mode: str = "a", storage_options={}):
        super().__init__(NumpyArrayStore(root_dir, mode, storage_options=storage_options), Stack.load_instances)


class NumpyCCStore(HierarchicalStoreBase[CrossCorrelation], CrossCorrelationDataStore):
    def __init__(self, root_dir: str, mode: str = "a", storage_options={}):
        super().__init__(
            NumpyArrayStore(root_dir, mode, storage_options=storage_options), CrossCorrelation.load_instances
        )
