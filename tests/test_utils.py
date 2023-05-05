import pytest

from noisepy.seis.utils import fs_join

paths = [
    ("/dir/", "file.txt", "/dir/file.txt"),
    ("../relative", "file.json", "../relative/file.json"),
    ("s3://bucket/path", "file.xml", "s3://bucket/path/file.xml"),
]


@pytest.mark.parametrize("path1, path2, expected", paths)
def test_fs_join(path1: str, path2: str, expected: str):
    assert expected == fs_join(path1, path2)
