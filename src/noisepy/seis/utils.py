import logging
import os
import posixpath
import time
from concurrent.futures import Future
from typing import Any, Iterable, List
from urllib.parse import urlparse

import fsspec
import numpy as np
import psutil
from tqdm.autonotebook import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from noisepy.seis.constants import AWS_EXECUTION_ENV

S3_SCHEME = "s3"
utils_logger = logging.getLogger(__name__)


def get_filesystem(path: str, storage_options: dict = {}) -> fsspec.AbstractFileSystem:
    """Construct an fsspec filesystem for the given path"""
    url = urlparse(path)
    # The storage_options coming from the ConfigParameters is keyed by protocol
    storage_options = storage_options.get(url.scheme, storage_options)
    return fsspec.filesystem(url.scheme, **storage_options)


def fs_join(path1: str, path2: str) -> str:
    """Helper for joining two paths that can handle both S3 URLs and local file system paths"""
    url = urlparse(path1)
    if url.scheme == S3_SCHEME:
        return posixpath.join(path1, path2)
    else:
        return os.path.join(path1, path2)


def get_fs_sep(path1: str) -> str:
    """Helper for getting a path separator that can handle both S3 URLs and local file system paths"""
    url = urlparse(path1)
    if url.scheme == S3_SCHEME:
        return posixpath.sep
    else:
        return os.sep


class TimeLogger:
    """
    A utility class to measure and log the time spent in code fragments. The basic usage is to call::

        tlog.log(message)

    This will measure the time between the last time checkpoint and the call to ``log()``. The
    checkpoint is updated when:
    - ``TimeLogger`` instance is created
    - ``log()`` is called
    - ``reset()`` is called

    Alternatively, an explicit reference start time may be passed when logging: ``log(message, start_time)``. E.g.::

        start_time = tlog.reset()
        ...
        tlog.log("step 1")
        ...
        tlog.log("step 2")
        ...
        tlog.log("overall", start_time)
    """

    enabled: bool = True

    def __init__(self, logger: logging.Logger = utils_logger, level: int = logging.DEBUG, prefix: str = None):
        """
        Create an instance that will use the given logger and logging level to log the times
        """
        self.logger = logger
        self.level = level
        self.prefix = "" if prefix is None else f" {prefix}"
        self.reset()

    def reset(self) -> float:
        self.time = time.time()
        return self.time

    def log(self, message: str = None, start: float = -1.0) -> float:
        stop = time.time()
        dt = stop - self.time if start <= 0 else stop - start
        self.reset()
        self.log_raw(message, dt)
        return self.time

    def log_raw(self, message: str, dt: float):
        if self.enabled:
            self.logger.log(self.level, f"TIMING{self.prefix}: {dt:6.4f} secs. for {message}")
        return self.time


def error_if(condition: bool, msg: str, error_type: type = RuntimeError):
    """
    Raise an error if the condition is True

    Args:
        condition (bool): Condition to evaluate
        msg (str): Error message
        error_type (type): Type of error to raise, e.g. ValueError
    """
    if condition:
        raise error_type(msg)


def get_results(futures: Iterable[Future], task_name: str = "", logger: logging.Logger = None) -> Iterable[Any]:
    # if running in AWS, use -1 for the position so it's logged properly
    position = -1 if AWS_EXECUTION_ENV in os.environ else None

    def pbar_update(_: Future):
        mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        pbar.update(1)
        pbar.set_description(f"{task_name}. Memory: {mem_mb:5.0f} MB")

    # Show a progress bar when processing futures
    with logging_redirect_tqdm():
        with tqdm(total=len(futures), desc=task_name, position=position) as pbar:
            for f in futures:
                f.add_done_callback(pbar_update)
            return [f.result() for f in futures]


def unstack(stack: np.ndarray, axis=0) -> List[np.ndarray]:
    """
    Split a stack along the given axis into a list of arrays
    """
    return [np.squeeze(a, axis=axis) for a in np.split(stack, stack.shape[axis], axis=axis)]


def remove_nans(a: np.ndarray) -> np.ndarray:
    """
    Remove NaN values from a 1D array
    """
    return a[~np.isnan(a)]


def remove_nan_rows(a: np.ndarray) -> np.ndarray:
    """
    Remove rows from a 2D array that contain NaN values
    """
    return a[~np.isnan(a).any(axis=1)]
