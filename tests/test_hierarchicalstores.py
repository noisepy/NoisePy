from typing import Tuple

import pytest
from datetimerange import DateTimeRange
from utils import range

from noisepy.seis.hierarchicalstores import PairDirectoryCache
from noisepy.seis.numpystore import NumpyArrayStore
from noisepy.seis.zarrstore import ZarrStoreHelper


def test_dircache():
    cache = PairDirectoryCache()

    ts1 = range(4, 1, 2)
    ts2 = range(4, 2, 3)
    ts3 = range(2, 1, 2)

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

    tsh1 = range(4, 1, 1, 0, 1)
    assert not cache.contains("src", "rec", tsh1)
    cache.add("src", "rec", [tsh1])
    assert cache.contains("src", "rec", tsh1)
    check_1day()
    assert cache.get_timespans("src", "rec") == [ts3, ts1, ts2, tsh1]

    with pytest.raises(ValueError):
        cache.add("src", "rec", [ts1, tsh1])


numpy_paths = [
    (
        "some/path/CI.BAK/CI.ARV/2021_07_01_00_00_00T2021_07_02_00_00_00.tar.gz",
        ("CI.ARV", range(7, 1, 2)),
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
        ("CI.ARV", range(7, 1, 2)),
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
