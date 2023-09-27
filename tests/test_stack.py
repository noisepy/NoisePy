from unittest.mock import MagicMock

import numpy as np
import pytest
from datetimerange import DateTimeRange

from noisepy.seis.datatypes import (
    ChannelType,
    ConfigParameters,
    CrossCorrelation,
    Station,
)
from noisepy.seis.stack import stack, stack_pair, validate_pairs


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
class SerializableMock(MagicMock):
    def __reduce__(self):
        return (MagicMock, ())


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


def test_stack_pair():
    config = ConfigParameters()
    sta = Station("CI", "BAK")
    ts = DateTimeRange("2021-01-01", "2021-01-02")
    params = {
        "ngood": 4,
        "time": 1548979200.0,
    }
    cc_store = SerializableMock()

    data = np.random.rand(1, 8001)
    ch = [ChannelType(n) for n in ["BHE", "BHN", "BHZ"]]
    pairs = [(ch[0], ch[0]), (ch[0], ch[1]), (ch[0], ch[2]), (ch[1], ch[1]), (ch[1], ch[2]), (ch[2], ch[2])]

    ccs = [CrossCorrelation(p[0], p[1], params, data) for p in pairs]

    cc_store.read_correlations.return_value = ccs
    stacks = stack_pair(sta, sta, [ts, ts], cc_store, config)
    assert len(stacks) == 6
