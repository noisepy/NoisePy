from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from datetimerange import DateTimeRange

from noisepy.seis.asdfstore import ASDFCCStore
from noisepy.seis.datatypes import Channel, ChannelType, Station
from noisepy.seis.stores import CrossCorrelationDataStore
from noisepy.seis.zarrstore import ZarrCCStore


@pytest.fixture
def ccstore(tmp_path: Path) -> CrossCorrelationDataStore:
    return CrossCorrelationDataStore(str(tmp_path))


def _ccstore_test_helper(ccstore: CrossCorrelationDataStore):
    def make_1dts(dt: datetime):
        dt = dt.replace(tzinfo=timezone.utc, microsecond=0)
        return DateTimeRange(dt, dt + timedelta(days=1))

    data = np.zeros(0)
    ts1 = make_1dts(datetime.now())
    ts2 = make_1dts(ts1.end_datetime)
    src = Channel(ChannelType("foo"), Station("nw", "sta1"))
    rec = Channel(ChannelType("bar"), Station("nw", "sta2"))

    # assert empty state
    assert not ccstore.is_done(ts1)
    assert not ccstore.is_done(ts2)
    assert not ccstore.contains(ts1, src, rec)
    assert not ccstore.contains(ts2, src, rec)

    # add CC (src->rec) for ts1
    ccstore.append(ts1, src, rec, {}, data)
    # assert ts1 is there, but not ts2
    assert ccstore.contains(ts1, src, rec)
    assert not ccstore.contains(ts2, src, rec)
    # also rec->src should not be there for ts1
    assert not ccstore.contains(ts1, rec, src)
    assert not ccstore.is_done(ts1)
    # now mark ts1 done and assert it
    ccstore.mark_done(ts1)
    assert ccstore.is_done(ts1)
    assert not ccstore.is_done(ts2)

    # now add CC for ts2
    ccstore.append(ts2, src, rec, {}, data)
    assert ccstore.contains(ts2, src, rec)
    assert not ccstore.is_done(ts2)
    ccstore.mark_done(ts2)
    assert ccstore.is_done(ts2)

    timespans = ccstore.get_timespans()
    assert timespans == [ts1, ts2]
    sta_pairs = ccstore.get_station_pairs(ts1)
    assert sta_pairs == [(src.station, rec.station)]
    cha_pairs = ccstore.get_channeltype_pairs(ts1, sta_pairs[0][0], sta_pairs[0][1])
    assert cha_pairs == [(src.type, rec.type)]


def test_asdfccstore(asdfstore: ASDFCCStore):
    _ccstore_test_helper(asdfstore)


def test_zarrccstore(zarrstore: ZarrCCStore):
    _ccstore_test_helper(zarrstore)
