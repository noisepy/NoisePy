import json
from datetime import datetime, timedelta, timezone
from typing import Dict

import numpy as np
from datetimerange import DateTimeRange

from noisepy.seis.asdfstore import ASDFCCStore
from noisepy.seis.datatypes import (
    Channel,
    ChannelType,
    ConfigParameters,
    CrossCorrelation,
    Station,
    to_json_types,
)
from noisepy.seis.noise_module import cc_parameters
from noisepy.seis.numpystore import NumpyCCStore
from noisepy.seis.stores import CrossCorrelationDataStore
from noisepy.seis.zarrstore import ZarrCCStore


def make_1dts(dt: datetime):
    dt = dt.replace(tzinfo=timezone.utc, microsecond=0)
    return DateTimeRange(dt, dt + timedelta(days=1))


ts1 = make_1dts(datetime.now())
ts2 = make_1dts(ts1.end_datetime)
src = Channel(ChannelType("foo"), Station("nw", "sta1"))
rec = Channel(ChannelType("bar"), Station("nw", "sta2"))


def _ccstore_test_helper(ccstore: CrossCorrelationDataStore):
    data = np.random.random((10, 10))
    coor = {
        "lonS": 0,
        "latS": 0,
        "lonR": 0,
        "latR": 0,
    }
    conf = ConfigParameters()
    # test param serialization with real types
    params = cc_parameters(conf, coor, np.zeros(10, dtype=np.float32), np.zeros(10, dtype=np.int16), ["nn"])

    # assert empty state
    assert not ccstore.contains(src.station, rec.station, ts1)
    assert not ccstore.contains(src.station, rec.station, ts2)

    # add CC (src->rec) for ts1
    ccstore.append(ts1, src.station, rec.station, [CrossCorrelation(src.type, rec.type, params, data)])
    # assert ts1 is there, but not ts2
    assert ccstore.contains(src.station, rec.station, ts1)
    assert not ccstore.contains(src.station, rec.station, ts2)
    # also rec->src should not be there for ts1
    assert not ccstore.contains(rec.station, src.station, ts1)

    # now add CC for ts2
    ccstore.append(ts2, src.station, rec.station, [CrossCorrelation(src.type, rec.type, {}, data)])
    sta_pairs = check_populated_store(ccstore)
    ccs = ccstore.read(ts1, sta_pairs[0][0], sta_pairs[0][1])
    cha_pairs = [(c.src, c.rec) for c in ccs]
    assert cha_pairs == [(src.type, rec.type)]

    assert_dict_equal(params, ccs[0].parameters)
    assert np.all(data == ccs[0].data)

    wrong_ccs = ccstore.read(ts1, src.station, Station("nw", "wrong"))
    assert len(wrong_ccs) == 0


def assert_dict_equal(d1: Dict, d2: Dict):
    # use json to compare dicts with nested values such as numpy arrays
    d1_str = json.dumps(to_json_types(d1), sort_keys=True)
    d2_str = json.dumps(to_json_types(d2), sort_keys=True)
    assert d1_str == d2_str


def check_populated_store(ccstore):
    assert ccstore.contains(src.station, rec.station, ts2)

    timespans = ccstore.get_timespans(src.station, rec.station)
    assert timespans == [ts1, ts2]
    sta_pairs = ccstore.get_station_pairs()
    assert sta_pairs == [(src.station, rec.station)]
    return sta_pairs


# Use the built in tmp_path fixture: https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html
def test_asdfccstore(tmp_path):
    path = str(tmp_path)
    _ccstore_test_helper(ASDFCCStore(path))
    check_populated_store(ASDFCCStore(tmp_path))


def test_zarrccstore(tmp_path):
    path = str(tmp_path)
    _ccstore_test_helper(ZarrCCStore(path))
    check_populated_store(ZarrCCStore(path))


def test_numpyccstore(tmp_path):
    path = str(tmp_path)
    _ccstore_test_helper(NumpyCCStore(path))
    check_populated_store(NumpyCCStore(path))
