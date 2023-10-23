import os
from unittest.mock import Mock

import pytest
from test_channelcatalog import MockCatalog

from noisepy.seis.constants import NO_DATA_MSG
from noisepy.seis.correlate import (
    _filter_channel_data,
    _safe_read_data,
    cross_correlate,
)
from noisepy.seis.datatypes import Channel, ChannelData, ConfigParameters, Station
from noisepy.seis.scedc_s3store import SCEDCS3DataStore


def test_read_channels():
    CLOSEST_FREQ = 60
    samp_freq = 40
    freqs = [10, 39, CLOSEST_FREQ, 100]
    ch_data = []
    for f in freqs:
        cd = ChannelData.empty()
        cd.sampling_rate = f
        ch_data.append(cd)
    N = 5
    tuples = [(Channel("foo", Station("CI", "bar")), cd) for cd in ch_data] * N

    # we should pick the closest frequency that is >= to the target freq, 60 in this case
    filtered = _filter_channel_data(tuples, samp_freq, single_freq=True)
    assert N == len(filtered)
    assert [t[1].sampling_rate for t in filtered] == [CLOSEST_FREQ] * N

    # we should get all data at >= 40 Hz (60 and 100)
    filtered = _filter_channel_data(tuples, samp_freq, single_freq=False)
    assert N * 2 == len(filtered)
    assert all([t[1].sampling_rate >= samp_freq for t in filtered])


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


def test_correlation():
    config = ConfigParameters()
    config.samp_freq = 1.0
    config.rm_resp = "no"  # since we are using a mock catalog
    path = os.path.join(os.path.dirname(__file__), "./data/cc")
    raw_store = SCEDCS3DataStore(path, MockCatalog())
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
