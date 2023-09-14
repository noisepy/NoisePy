from datetime import datetime

import pytest

from noisepy.seis.asdfstore import ASDFRawDataStore, _parse_channel_path
from noisepy.seis.datatypes import ChannelType


@pytest.fixture
def store():
    import os

    return ASDFRawDataStore(os.path.join(os.path.dirname(__file__), "./data"))


def test_get_timespans(store: ASDFRawDataStore):
    ts = store.get_timespans()
    assert len(ts) == 1
    assert ts[0].start_datetime == datetime.fromisoformat("2019-02-01T00:00:00+00:00")
    assert ts[0].end_datetime == datetime.fromisoformat("2019-02-01T01:00:00+00:00")


def test_get_channels(store: ASDFRawDataStore):
    ts = store.get_timespans()[0]
    chans = store.get_channels(ts)
    assert len(chans) == 1
    assert str(chans[0].type) == "bhn_00"
    assert chans[0].station.name == "BAK"


def test_get_data(store: ASDFRawDataStore):
    ts = store.get_timespans()[0]
    chan = store.get_channels(ts)[0]
    chdata = store.read_data(ts, chan)
    assert chdata.data.size == 72001
    assert chdata.sampling_rate == 20.0
    assert chdata.start_timestamp == ts.start_datetime.timestamp()


@pytest.mark.parametrize(
    "path,ch1, ch2",
    [
        ("bhn_bhn", ChannelType("bhn"), ChannelType("bhn")),
        ("bhn_00_bhn_01", ChannelType("bhn", "00"), ChannelType("bhn", "01")),
        ("bhn_bhn_01", ChannelType("bhn"), ChannelType("bhn", "01")),
        ("bhn_00_bhn", ChannelType("bhn", "00"), ChannelType("bhn")),
    ],
)
def test_parse_channel_path(path, ch1, ch2):
    assert _parse_channel_path(path) == (ch1, ch2)
