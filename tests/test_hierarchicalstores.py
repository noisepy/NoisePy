from typing import Tuple
from unittest import mock

import pytest
from botocore.exceptions import ClientError
from datetimerange import DateTimeRange
from fsspec.implementations.local import LocalFileSystem  # noqa F401
from utils import date_range

from noisepy.seis.hierarchicalstores import (
    ERR_SLOWDOWN,
    FIND_RETRIES,
    PairDirectoryCache,
)
from noisepy.seis.numpystore import NumpyArrayStore, NumpyCCStore
from noisepy.seis.zarrstore import ZarrStoreHelper


def test_dircache():
    cache = PairDirectoryCache()

    ts1 = date_range(4, 1, 2)
    ts2 = date_range(4, 2, 3)
    ts3 = date_range(2, 1, 2)

    assert not cache.contains("src", "rec", ts1)
    assert cache.get_pairs() == []
    assert not cache.is_src_loaded("src")
    cache.add("src", "rec", [])
    assert cache.is_src_loaded("src")
    cache.add("src", "rec", [ts1, ts2])
    assert cache.contains("src", "rec", ts1)
    assert cache.contains("src", "rec", ts2)

    cache.add("src", "rec", [ts3])

    def check_1day():
        assert cache.contains("src", "rec", ts1)
        assert cache.contains("src", "rec", ts2)
        assert cache.contains("src", "rec", ts3)

        assert not cache.contains("src", "rec2", ts1)
        assert not cache.contains("src2", "rec", ts1)

    check_1day()
    assert cache.get_timespans("src", "rec") == [ts3, ts1, ts2]
    assert cache.get_timespans("src2", "rec2") == []

    tsh1 = date_range(4, 1, 1, 0, 1)
    assert not cache.contains("src", "rec", tsh1)
    cache.add("src", "rec", [tsh1])
    assert cache.contains("src", "rec", tsh1)
    check_1day()
    assert cache.get_timespans("src", "rec") == [ts3, ts1, ts2, tsh1]

    # add timespans with different lentghs
    cache.add("src", "rec", [ts1, tsh1])
    check_1day()
    assert cache.contains("src", "rec", tsh1)


numpy_paths = [
    (
        "some/path/CI.BAK/CI.ARV/2021_07_01_00_00_00T2021_07_02_00_00_00.tar.gz",
        ("CI.ARV", date_range(7, 1, 2)),
    ),
    ("some/path/CI.BAK/CI.BAK_CI.ARV/2021_07_01_00_00_00.tar.gz", None),
    ("some/path/CI.BAK/CI.BAK_CI.ARV/2021_07_01_00_00_00.tar.gz", None),
    ("path/2021_07_01_00_00_00.tar.gz", None),
    ("2021_07_01_00_00_00.tar.gz", None),
    ("some/path/CI.BAK/CI.BAK_CI.ARV/2021_07_01_00_00_00T2021_07_02_00_00_00.TXT", None),
]


@pytest.mark.parametrize("path,expected", numpy_paths)
def test_numpy_parse_path(path: str, expected: Tuple[str, DateTimeRange]):
    store = NumpyArrayStore("some/path", "r")
    assert store.parse_path(path) == expected


zarr_paths = [
    (
        "some/path/CI.BAK/CI.ARV/2021_07_01_00_00_00T2021_07_02_00_00_00/0.0.0",
        ("CI.ARV", date_range(7, 1, 2)),
    ),
    ("some/path/CI.BAK/CI.BAK_CI.ARV/2021_07_01_00_00_00/0.0.0", None),
    ("some/path/CI.BAK/CI.BAK/2021_07_01_00_00_00/.zgroup", None),
    ("path/non_ts/0.0.0", None),
    ("too_short/0.0.0", None),
]


@pytest.mark.parametrize("path,expected", zarr_paths)
def test_zarr_parse_path(tmp_path, path: str, expected: Tuple[str, DateTimeRange]):
    store = ZarrStoreHelper(str(tmp_path), "a")
    assert store.parse_path(path) == expected


@mock.patch("fsspec.implementations.local.LocalFileSystem.find")
def test_find(find_mock, tmp_path):
    store = NumpyCCStore(str(tmp_path), "r")
    find_mock.side_effect = ClientError({"Error": {"Code": ERR_SLOWDOWN}}, "ListObjectsV2")
    with pytest.raises(RuntimeError):
        store._find("foo")
    assert FIND_RETRIES == find_mock.call_count

    # if it's not a SlowDown error then we shouldn't retry
    find_mock.side_effect = ClientError({"Error": {"Code": "other error"}}, "ListObjectsV2")
    with pytest.raises(ClientError):
        store._find("foo")
    assert FIND_RETRIES + 1 == find_mock.call_count

    # same with other type of ClientError
    find_mock.side_effect = ClientError({}, "operation")
    with pytest.raises(ClientError):
        store._find("foo")
    assert FIND_RETRIES + 2 == find_mock.call_count

    # same with other type of exceptoins
    find_mock.side_effect = Exception()
    with pytest.raises(Exception):
        store._find("foo")
    assert FIND_RETRIES + 3 == find_mock.call_count
