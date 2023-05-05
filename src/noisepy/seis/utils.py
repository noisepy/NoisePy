import os
import posixpath
from urllib.parse import urlparse

import fsspec

S3_SCHEME = "s3"
ANON_ARG = "anon"


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
