from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from datetimerange import DateTimeRange

from noisepy.seis.asdfstore import ASDFCCStore
from noisepy.seis.datatypes import Channel, ChannelType, CrossCorrelation, Station
from noisepy.seis.numpystore import NumpyCCStore
from noisepy.seis.stores import CrossCorrelationDataStore
from noisepy.seis.zarrstore import ZarrCCStore


# Use the built in tmp_path fixture: https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html
# to create CC stores
@pytest.fixture
def asdfstore(tmp_path: Path) -> ASDFCCStore:
    return ASDFCCStore(str(tmp_path))


@pytest.fixture
def zarrstore(tmp_path: Path) -> ZarrCCStore:
    return ZarrCCStore(str(tmp_path))


@pytest.fixture
def numpystore(tmp_path: Path) -> NumpyCCStore:
    return NumpyCCStore(str(tmp_path))


def _ccstore_test_helper(ccstore: CrossCorrelationDataStore):
    def make_1dts(dt: datetime):
        dt = dt.replace(tzinfo=timezone.utc, microsecond=0)
        return DateTimeRange(dt, dt + timedelta(days=1))

    data = np.random.random((10, 10))
    ts1 = make_1dts(datetime.now())
    ts2 = make_1dts(ts1.end_datetime)
    src = Channel(ChannelType("foo"), Station("nw", "sta1"))
    rec = Channel(ChannelType("bar"), Station("nw", "sta2"))
    params = {"key": "Value"}

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
    assert ccstore.contains(src.station, rec.station, ts2)

    timespans = ccstore.get_timespans(src.station, rec.station)
    assert timespans == [ts1, ts2]
    sta_pairs = ccstore.get_station_pairs()
    assert sta_pairs == [(src.station, rec.station)]
    ccs = ccstore.read(ts1, sta_pairs[0][0], sta_pairs[0][1])
    cha_pairs = [(c.src, c.rec) for c in ccs]
    assert cha_pairs == [(src.type, rec.type)]
    assert params == ccs[0].parameters
    assert np.all(data == ccs[0].data)

    wrong_ccs = ccstore.read(ts1, src.station, Station("nw", "wrong"))
    assert len(wrong_ccs) == 0


def test_asdfccstore(asdfstore: ASDFCCStore):
    _ccstore_test_helper(asdfstore)


def test_zarrccstore(zarrstore: ZarrCCStore):
    _ccstore_test_helper(zarrstore)


def test_numpyccstore(numpystore: NumpyCCStore):
    _ccstore_test_helper(numpystore)
