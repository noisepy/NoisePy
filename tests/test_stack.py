from unittest.mock import Mock

import pytest
from datetimerange import DateTimeRange

from noisepy.seis.datatypes import ConfigParameters, Station
from noisepy.seis.stack import stack, validate_pairs


def test_validate_pairs():
    ts = DateTimeRange("2021-01-01", "2021-01-02")
    pair = "BAK_BAK"
    assert not validate_pairs(3, pair, 1, ts, 5)
    assert not validate_pairs(3, pair, 0, ts, 5)
    assert validate_pairs(3, pair, 1, ts, 6)
    assert validate_pairs(3, pair, 0, ts, 9)
    with pytest.raises(Exception):
        validate_pairs(3, pair, 0, ts, 10)


# since stacking using the ProcessPoolExecutor the stores need to be serializable
class SerializableMock(Mock):
    def __reduce__(self):
        return (Mock, ())


def test_stack_error():
    config = ConfigParameters()
    sta = Station("CI", "BAK")
    cc_store = SerializableMock()
    cc_store.get_timespans.return_value = [DateTimeRange("2021-01-01", "2021-01-02")]
    cc_store.get_station_pairs.return_value = [(sta, sta)]
    cc_store.read_correlations.return_value = []

    stack_store = SerializableMock()
    stack_store.get_station_pairs.return_value = []
    stack_store.contains.return_value = False
    with pytest.raises(RuntimeError) as e:
        stack(cc_store, stack_store, config)
        assert "CI.BAK" in str(e)
