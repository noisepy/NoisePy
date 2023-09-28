import io
import json
import logging
import os
import sys
import tarfile
import tempfile
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyasdf
import zarr

from noisepy.seis.tiledb import _TileDBHelper
from noisepy.seis.utils import fs_join, get_filesystem

logger = logging.getLogger(__name__)
s3_path = "s3://<bucket>/write_speed/"

"""
A script to test the write speed of different storage options
"""


class Writer(ABC):
    def __init__(self, path: str) -> None:
        self.path = fs_join(path, type(self).__name__) + "/"
        logger.info(f"Creating writer at {self.path}")
        self.storage_options = {"client_kwargs": {"region_name": "us-west-2"}}
        self.fs = get_filesystem(path, storage_options=self.storage_options)
        self.data = np.random.rand(9, 1, 8001)
        with open("params.json", "r") as f:
            self.params = json.load(f)

    def initialize(self):
        pass

    def write(self, id: int) -> int:
        t = time.time()
        self.write_override(id)
        return time.time() - t

    @abstractmethod
    def write_override(self, id: int):
        pass


TAR_GZ_EXTENSION = ".tar.gz"
FILE_ARRAY_NPY = "array.npy"
FILE_PARAMS_JSON = "params.json"


class NumpyWriter(Writer):
    """
    Writes a single tar.gz file with the array (npy format) and params.json
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def write_override(self, id: int):
        js = json.dumps(self.params)

        def add_file_bytes(tar, name, f):
            f.seek(0)
            ti = tarfile.TarInfo(name=name)
            ti.size = f.getbuffer().nbytes
            tar.addfile(ti, fileobj=f)

        with self.fs.open(fs_join(self.path, f"{id}{TAR_GZ_EXTENSION}"), "wb") as f:
            with tarfile.open(fileobj=f, mode="w:gz") as tar:
                with io.BytesIO() as npyf:
                    np.save(npyf, self.data, allow_pickle=False)
                    with io.BytesIO() as jsf:
                        jsf.write(js.encode("utf-8"))
                        add_file_bytes(tar, FILE_ARRAY_NPY, npyf)
                        add_file_bytes(tar, FILE_PARAMS_JSON, jsf)


class ZarrMultiArrayWriter(Writer):
    """
    Writes a separate zarr array for each cross-correlation
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def initialize(self):
        mode = "a"
        logger.info(f"store creating at {self.path}, mode={mode}, storage_options={self.storage_options}")
        store = zarr.storage.FSStore(self.path, **self.storage_options)
        self.root = zarr.open_group(store, mode=mode)
        logger.info(f"store created at {self.path}: {type(store)}. Zarr version {zarr.__version__}")

    def write_override(self, id: int):
        array = self.root.create_dataset(
            f"{id}",
            data=self.data,
            chunks=self.data.shape,
            dtype=self.data.dtype,
        )
        array.attrs.update(self.params)


class ZarrSingleArrayWriter(Writer):
    """
    Writes a single zarr array with all cross-correlations (chunk size 1000 CCs)
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def initialize(self):
        mode = "a"
        logger.info(f"store creating at {self.path}, mode={mode}, storage_options={self.storage_options}")
        store = zarr.storage.FSStore(self.path, **self.storage_options)
        self.root = zarr.open_array(
            store, mode=mode, shape=(10000, 9, 1, 8001), chunks=(1000, 9, 1, 8001), dtype=self.data.dtype
        )
        logger.info(f"store created at {self.path}: {type(store)}. Zarr version {zarr.__version__}")

    def write_override(self, id: int):
        self.root[id] = self.data
        self.root.attrs.update({id: self.params})


class TileDBWriter(Writer):
    """
    Writes a single tiledb array with all cross-correlations (tile size: 1000 CCs)
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def initialize(self):
        self.helper = _TileDBHelper(self.path, "w", 1000, storage_options=self.storage_options)

    def write_override(self, id: int):
        self.helper.append(
            f"{id}",
            self.params,
            self.data,
        )


class TileDBOneWriter(TileDBWriter):
    """
    Writes a single tiledb array with all cross-correlations (tile size: 1 CC)
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def initialize(self):
        self.helper = _TileDBHelper(self.path, "w", 1, storage_options=self.storage_options)


class ASDFWriter(Writer):
    """
    Writes a single tiledb array with all cross-correlations (tile size: 1 CC)
    """

    def __init__(self, path: str) -> None:
        super().__init__(path)

    def initialize(self):
        self.temp = tempfile.gettempdir()

    def write_override(self, id: int):
        tmp = os.path.join(self.temp, f"{id}.h5")
        with pyasdf.ASDFDataSet(tmp, mode="w", mpi=False, compression=None) as ds:
            ds.add_auxiliary_data(data=self.data, parameters=self.params["channels"][0][4], data_type="CC", path="CC")

        self.fs.put(tmp, fs_join(self.path, f"{id}.h5"))
        os.remove(tmp)


def test(nwrites: int, writer: Writer):
    # warmup
    WARMUP = 10
    if writer.fs.exists(writer.path):
        writer.fs.rm(writer.path, recursive=True)
    writer.fs.makedirs(writer.path, exist_ok=True)
    writer.initialize()
    writer.write(0)
    executor = ThreadPoolExecutor()
    times = list(executor.map(writer.write, range(1, WARMUP)))

    t = time.time()
    times = list(executor.map(writer.write, range(WARMUP, WARMUP + nwrites)))
    total = time.time() - t
    logger.info(
        f"{type(writer).__name__} time stats: Mean: {np.mean(times):0.3f} Std: {np.std(times)} "
        f" Min: {np.min(times)} Max: {np.max(times)} Total: {total}"
    )
    with writer.fs.open(fs_join(writer.path, "times.txt"), "w") as f:
        f.write("\n".join([str(t) for t in times]))


writer_types = [TileDBOneWriter, ASDFWriter, NumpyWriter, ZarrMultiArrayWriter, ZarrSingleArrayWriter, TileDBWriter]
if __name__ == "__main__":
    writes = int(sys.argv[1])
    path = s3_path + str(writes)
    writers = [t(path) for t in writer_types]
    for writer in writers:
        test(writes, writer)
