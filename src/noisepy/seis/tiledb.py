import json
from typing import Any, Dict

import numpy as np
import tiledb

from noisepy.seis.utils import fs_join, get_filesystem
from noisepy.seis.zarrstore import logger


# Experimentation with TileDB
class _TileDBHelper:
    def __init__(self, root_dir: str, mode: str, pair_tile: int, storage_options={}) -> None:
        # if tiledb.default_ctx() is None:
        logger.info(f"Creating TileDB at '{root_dir}' with {storage_options}")

        cfg = tiledb.Ctx().config()
        cfg.update({"py.init_buffer_bytes": 1024**2 * 50})
        cfg.update({"vfs.s3.region": "us-west-2"})  # replace with bucket region
        self.ctx = tiledb.Ctx(cfg)

        self.fs = get_filesystem(root_dir, storage_options=storage_options)
        self.arr_path = fs_join(root_dir, "tile")
        if self.fs.exists(self.arr_path):
            logger.warning(f"Not creating {self.arr_path} because it already exists")
            return

        # Create the two dimensions: unit -> rows, time -> columns
        pair_dim = tiledb.Dim(name="pair", domain=(0, 128000), tile=pair_tile, dtype=np.int32)
        stack_dim = tiledb.Dim(name="stack", domain=(0, 0), tile=1, dtype=np.int32)
        chan_dim = tiledb.Dim(name="chan", domain=(0, 8), tile=9, dtype=np.int32)
        time_dim = tiledb.Dim(name="time", domain=(0, 8000), tile=8001, dtype=np.int32)

        # Create a domain using the two dimensions
        dom1 = tiledb.Domain(pair_dim, stack_dim, chan_dim, time_dim)
        attrib_cc = tiledb.Attr(name="cc", dtype=np.float32)
        schema = tiledb.ArraySchema(domain=dom1, sparse=False, attrs=[attrib_cc])
        tiledb.Array.create(self.arr_path, schema, ctx=self.ctx)

    def contains(self, path: str) -> bool:
        return False

    def append(
        self,
        path: str,
        params: Dict[str, Any],
        data: np.ndarray,
    ):
        index = int(path)
        logger.debug(f"Appending to {path}: {data.shape}, index = {index}")
        if data.shape[0] < 9:
            data = np.pad(data, ((0, 9 - data.shape[0]), (0, 0), (0, 0)), mode="constant", constant_values=np.nan)
        with tiledb.open(self.arr_path, "w", ctx=self.ctx) as array:
            array[index] = data
            array.meta[path] = json.dumps(params)
