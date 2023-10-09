from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from datetimerange import DateTimeRange
from utils import date_range

from noisepy.seis.datatypes import (
    ChannelType,
    ConfigParameters,
    CrossCorrelation,
    Station,
)
from noisepy.seis.stack import stack_cross_correlations, stack_pair, validate_pairs


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
    ts = date_range(1, 1, 2)
    config = ConfigParameters()
    config.start_date = ts.start_datetime
    config.end_date = ts.end_datetime
    sta = Station("CI", "BAK")
    cc_store = SerializableMock()
    cc_store.get_timespans.return_value = [ts]
    cc_store.get_station_pairs.return_value = [(sta, sta)]
    cc_store.read_correlations.return_value = []

    stack_store = SerializableMock()
    stack_store.get_station_pairs.return_value = []
    stack_store.contains.return_value = False
    # , new_callable=lambda: ThreadPoolExecutor(1)
    with patch("noisepy.seis.stack.ProcessPoolExecutor") as mock_executor:
        mock_executor.return_value = ThreadPoolExecutor(1)
        with pytest.raises(RuntimeError) as e:
            stack_cross_correlations(cc_store, stack_store, config)
    assert "CI.BAK" in str(e)


def test_stack_pair():
    config = ConfigParameters()
    sta = Station("CI", "BAK")
    ts = date_range(1, 1, 2)
    config.start_date = ts.start_datetime
    config.end_date = ts.end_datetime
    params = {
        "ngood": 4,
        "time": 1548979200.0,
    }
    cc_store = SerializableMock()

    data = np.random.rand(1, 8001)
    ch = [ChannelType(n) for n in ["BHE", "BHN", "BHZ"]]
    pairs = [(ch[0], ch[0]), (ch[0], ch[1]), (ch[0], ch[2]), (ch[1], ch[1]), (ch[1], ch[2]), (ch[2], ch[2])]

    ccs = [CrossCorrelation(p[0], p[1], params, data) for p in pairs]

    cc_store.read.return_value = ccs
    stacks = stack_pair(sta, sta, [ts, ts], cc_store, config)
    assert len(stacks) == 6
    ts2 = date_range(1, 20, 22)
    stacks = stacks = stack_pair(sta, sta, [ts2], cc_store, config)
    assert len(stacks) == 0
