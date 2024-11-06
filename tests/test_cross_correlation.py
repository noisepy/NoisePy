import os
from unittest.mock import Mock

import obspy
import pytest
from datetimerange import DateTimeRange

from noisepy.seis.constants import NO_DATA_MSG
from noisepy.seis.correlate import (
    _filter_channel_data,
    _safe_read_data,
    cross_correlate,
)
from noisepy.seis.io.datatypes import (
    CCMethod,
    Channel,
    ChannelData,
    ConfigParameters,
    RmResp,
    Station,
)
from noisepy.seis.io.s3store import SCEDCS3DataStore


def test_read_channels():
    CLOSEST_FREQ = 60
    sampling_rate = 40
    ch_data1 = []
    N = 5
    for f in [10, 39, CLOSEST_FREQ, 100]:
        cd = ChannelData.empty()
        cd.sampling_rate = f
        ch_data1.append(cd)

    cd = ChannelData.empty()
    cd.sampling_rate = 100
    ch_data2 = [cd]

    tuples = [(Channel("foo", Station("CI", "FOO")), cd) for cd in ch_data1] * N
    tuples += [(Channel("bar", Station("CI", "BAR")), cd) for cd in ch_data2] * N

    # we should pick the closest frequency that is >= to the target sampling_rate
    # 60 Hz in this case, for both stations
    # CI.FOO returns 60 Hz
    # CI.BAR returns nothing
    filtered = _filter_channel_data(tuples, sampling_rate, single_freq=True)
    assert N == len(filtered)
    assert [t[1].sampling_rate for t in filtered] == [CLOSEST_FREQ] * N

    # we should pick the closest frequency that is >= to the target sampling_rate
    # but might be different for each station
    # CI.FOO returns 60 Hz
    # CI.BAR returns 100 Hz
    filtered = _filter_channel_data(tuples, sampling_rate, single_freq=False)
    assert N * 2 == len(filtered)
    assert all([t[1].sampling_rate >= sampling_rate for t in filtered])


def test_safe_read_channels():
    store = Mock()
    store.read_data = Mock(side_effect=Exception("foo"))
    ch_data = _safe_read_data(store, "foo", "bar")
    assert ch_data.data.size == 0


def test_correlation_nodata():
    config = ConfigParameters()
    raw_store = Mock()
    raw_store.get_timespans.return_value = []
    cc_store = Mock()
    with pytest.raises(IOError) as excinfo:
        cross_correlate(raw_store, config, cc_store)
    assert NO_DATA_MSG in str(excinfo.value)


class MockCatalogMock:
    def get_full_channel(self, timespan: DateTimeRange, channel: Channel) -> Channel:
        return channel

    def get_inventory(self, timespan: DateTimeRange, station: Station) -> obspy.Inventory:
        net = station.network
        sta = station.name
        path = os.path.join(os.path.dirname(__file__), f"data/{net}/{net}_{sta}.xml")
        if os.path.exists(path):
            return obspy.read_inventory(path)
        else:
            return obspy.Inventory()


@pytest.mark.parametrize("rm_resp", [RmResp.NO, RmResp.INV])  # add tests for SPECTRUM, RESP and POLES_ZEROS
@pytest.mark.parametrize("cc_method", [CCMethod.XCORR, CCMethod.COHERENCY, CCMethod.DECONV])
@pytest.mark.parametrize("substack", [True, False])
@pytest.mark.parametrize("substack_windows", [1, 2])
@pytest.mark.parametrize("inc_hours", [0, 24])
@pytest.mark.parametrize("dpath", ["./data/cc", "./data/acc"])
def test_cross_correlation(
    rm_resp: RmResp, cc_method: CCMethod, substack: bool, substack_windows: int, inc_hours: int, dpath: str
):
    config = ConfigParameters()
    config.sampling_rate = 1.0
    config.rm_resp = rm_resp
    config.cc_method = cc_method
    config.inc_hours = inc_hours
    if substack:
        config.substack = substack
        config.substack_windows = substack_windows
    path = os.path.join(os.path.dirname(__file__), dpath)

    raw_store = SCEDCS3DataStore(path, MockCatalogMock())
    ts = raw_store.get_timespans()
    assert len(ts) == 1
    channels = raw_store.get_channels(ts[0])
    # fake the location so we don't have to download the inventory
    for c in channels:
        c.station.lat = 45
        c.station.lon = 45
        c.station.elevation = 45
    nsta = len(set([c.station.name for c in channels]))
    cc_store = Mock()
    cc_store.contains.return_value = False
    cross_correlate(raw_store, config, cc_store)
    expected_writes = nsta * (nsta + 1) / 2
    assert expected_writes == cc_store.append.call_count
