from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import utils
from datetimerange import DateTimeRange
from utils import date_range

from noisepy.seis.noise_module import rotation, cc_parameters

from noisepy.seis.io.datatypes import (
    ChannelType,
    ConfigParameters,
    CrossCorrelation,
    StackMethod,
    Station,
    CCMethod
)
from noisepy.seis.stack import (
    stack_cross_correlations,
    stack_pair,
    stack_store_pair,
    validate_pairs,
)


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


def test_stack_error(caplog):
    ts = utils.date_range(1, 1, 2)
    config = ConfigParameters(start_date=ts.start_datetime, end_date=ts.end_datetime)
    sta = Station("CI", "BAK")
    cc_store = SerializableMock()
    cc_store.get_timespans.return_value = [ts]
    cc_store.get_station_pairs.return_value = [(sta, sta)]
    cc_store.read_correlations.return_value = []

    stack_store = SerializableMock()
    stack_store.get_station_pairs.return_value = []
    stack_store.contains.return_value = False
    with patch("noisepy.seis.stack.ProcessPoolExecutor") as mock_executor:
        mock_executor.return_value = ThreadPoolExecutor(1)
        stack_cross_correlations(cc_store, stack_store, config)
    assert any(str(sta) in rec.message for rec in caplog.records if rec.levelname == "ERROR")


def test_stack_contains():
    ts = utils.date_range(1, 1, 2)
    config = ConfigParameters(start_date=ts.start_datetime, end_date=ts.end_datetime)
    sta = Station("CI", "BAK")
    cc_store = SerializableMock()
    stack_store = SerializableMock()
    stack_store.contains.return_value = True
    # should not stack but succeed if stack_store contains the stack
    result = stack_store_pair(sta, sta, cc_store, stack_store, config)
    stack_store.append.assert_not_called()
    assert result


# ALL performs LINEAR + PWS + ROBUST
stackmethod = [
    StackMethod.LINEAR,
    StackMethod.PWS,
    StackMethod.ROBUST,
    StackMethod.AUTO_COVARIANCE,
    StackMethod.NROOT,
    StackMethod.SELECTIVE,
    StackMethod.ALL,
]


@pytest.mark.parametrize("stackmethod", stackmethod)
@pytest.mark.parametrize("substack", [True, False])
@pytest.mark.parametrize("rotation", [True])
def test_stack_pair(stackmethod, substack: bool, rotation: bool):
    ts = date_range(1, 1, 2)
    config = ConfigParameters(start_date=ts.start_datetime, end_date=ts.end_datetime)
    config.stack_method = stackmethod
    config.substack = substack
    config.rotation = rotation
    sta = Station("CI", "BAK")
    if substack:
        params = {"ngood": [1, 1], "time": [1548979200.0, 1548979300.0], "azi": 90.0, "baz": 270.0}
    else:
        params = {"ngood": 4, "time": 1548979200.0, "azi": 90.0, "baz": 270.0}
    cc_store = SerializableMock()

    data = np.random.rand(1, 8001)
    ch = [ChannelType(n) for n in ["BHE", "BHN", "BHZ"]]
    pairs = [(ch[0], ch[0]), (ch[0], ch[1]), (ch[0], ch[2]), (ch[1], ch[1]), (ch[1], ch[2]), (ch[2], ch[2])]

    ccs = [CrossCorrelation(p[0], p[1], params, data) for p in pairs]

    cc_store.read.return_value = ccs
    stacks = stack_pair(sta, sta, [ts, ts], cc_store, config)
    assert len(stacks) > 0
    ts2 = date_range(1, 20, 22)
    stacks = stack_pair(sta, sta, [ts2], cc_store, config)
    assert len(stacks) == 0


@pytest.mark.parametrize("bigstack", [np.random.rand(9, 8000), np.random.rand(8, 8000)])
@pytest.mark.parametrize("locs", [{}, {"station": ["CI.BAK", "CI.SVD"], "angle": [0., 1.]}])
def test_rotation(bigstack: np.ndarray, locs: dict):
    parameters = {"ngood": 4, "time": 1548979200.0, "azi": 90.0, "baz": 270.0, "station_source": "CI.BAK", "station_receiver": "CI.SVD"}
    rotated = rotation(bigstack, parameters, locs)
    if bigstack.shape[0] < 9:
        assert len(rotated) == 0
    else:
        assert rotated.shape == bigstack.shape
        