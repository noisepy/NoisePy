import os
import posixpath
from urllib.parse import urlparse

import fsspec

S3_SCHEME = "s3"


def get_filesystem(path: str) -> fsspec.AbstractFileSystem:
    url = urlparse(path)
    return fsspec.filesystem(S3_SCHEME, anon=True) if url.scheme == S3_SCHEME else fsspec.filesystem("file")


def fs_join(path1: str, path2: str) -> str:
    if path1.startswith(S3_SCHEME):
        return posixpath.join(path1, path2)
    else:
        return os.path.join(path1, path2)
