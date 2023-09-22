import pytest
from datetimerange import DateTimeRange

from noisepy.seis.stack import validate_pairs


def test_validate_pairs():
    ts = DateTimeRange("2021-01-01", "2021-01-02")
    pair = "BAK_BAK"
    assert not validate_pairs(3, pair, 1, ts, 5)
    assert not validate_pairs(3, pair, 0, ts, 5)
    assert validate_pairs(3, pair, 1, ts, 6)
    assert validate_pairs(3, pair, 0, ts, 9)
    with pytest.raises(Exception):
        validate_pairs(3, pair, 0, ts, 10)
