import logging
import os
import posixpath
import time
from concurrent.futures import Future
from typing import Iterable
from urllib.parse import urlparse

import fsspec

S3_SCHEME = "s3"
ANON_ARG = "anon"
utils_logger = logging.getLogger(__name__)


def get_filesystem(path: str, storage_options: dict = {}) -> fsspec.AbstractFileSystem:
    """Construct an fsspec filesystem for the given path"""
    url = urlparse(path)
    # default to anonymous access for S3 if the this is not already specified
    if url.scheme == S3_SCHEME and ANON_ARG not in storage_options:
        storage_options = {ANON_ARG: True}
    return fsspec.filesystem(url.scheme, **storage_options)


def fs_join(path1: str, path2: str) -> str:
    """Helper for joining two paths that can handle both S3 URLs and local file system paths"""
    url = urlparse(path1)
    if url.scheme == S3_SCHEME:
        return posixpath.join(path1, path2)
    else:
        return os.path.join(path1, path2)


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

    def __init__(self, logger: logging.Logger = utils_logger, level: int = logging.DEBUG):
        """
        Create an instance that will use the given logger and logging level to log the times
        """
        self.logger = logger
        self.level = level
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
            self.logger.log(self.level, f"TIMING: {dt:6.4f} for {message}")
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


def _get_results(futures: Iterable[Future]) -> Iterable[Future]:
    return [f.result() for f in futures]
