import os

import numpy as np
import pytest
from fsspec.implementations.local import LocalFileSystem
from s3fs import S3FileSystem

from noisepy.seis.utils import fs_join, get_filesystem, remove_nan_rows, unstack

SEP = os.path.sep
paths = [
    (f"{SEP}dir{SEP}", "file.csv", SEP.join(["", "dir", "file.csv"])),
    (f"..{SEP}relative", "file.json", SEP.join(["..", "relative", "file.json"])),
    ("s3://bucket/path", "file.xml", "s3://bucket/path/file.xml"),
]


@pytest.mark.parametrize("path1, path2, expected", paths)
def test_fs_join(path1: str, path2: str, expected: str):
    assert expected == fs_join(path1, path2)


fs_types = [
    ("s3://bucket/path", S3FileSystem),
    ("/some/file", LocalFileSystem),
    ("s3/local/file", LocalFileSystem),
]


@pytest.mark.parametrize("path, fs_type", fs_types)
def test_get_filesystem(path, fs_type):
    assert isinstance(get_filesystem(path), fs_type)


def test_unstack():
    array_list = [np.array([[1, 2, 3], [5, 6, 7]]), np.array([[7, 6, 5], [4, 3, 2]])]
    stacked = np.stack(array_list, axis=0)
    unstacked = unstack(stacked, axis=0)
    assert len(unstacked) == len(array_list)
    for i in range(len(array_list)):
        assert np.all(unstacked[i] == array_list[i])


def test_remove_nan_rows():
    a = np.array([[1, 2, 3], [4, 5, 6], [np.nan, np.nan, np.nan]])
    b = remove_nan_rows(a)
    assert np.all(b == np.array([[1, 2, 3], [4, 5, 6]]))
