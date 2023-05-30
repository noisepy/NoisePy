import pytest
from fsspec.implementations.local import LocalFileSystem
from s3fs import S3FileSystem

from noisepy.seis.utils import fs_join, get_filesystem

paths = [
    ("/dir/", "file.csv", "/dir/file.csv"),
    ("../relative", "file.json", "../relative/file.json"),
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
