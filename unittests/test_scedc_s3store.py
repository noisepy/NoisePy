from datetime import datetime

import pytest
from datetimerange import DateTimeRange
from obspy import Inventory

from noisepy.seis.scedc_s3store import SCEDCS3DataStore

timespan1 = DateTimeRange(datetime(2022, 1, 2), datetime(2022, 1, 3))
timespan2 = DateTimeRange(datetime(2021, 2, 3), datetime(2021, 2, 4))
files_dates = [
    ("CIGMR__LHN___2022002.ms", timespan1),
    ("CIGMR__LHN___2021034.ms", timespan2),
]


@pytest.mark.parametrize("file,expected", files_dates)
def test_parsefilename(file: str, expected: DateTimeRange):
    assert expected == SCEDCS3DataStore._parse_timespan(file)


@pytest.fixture
def store():
    import os

    key = SCEDCS3DataStore._get_cache_key([timespan1])
    # This avoids the call to the SCEDC service during unit testing
    # by populating an in-memory cache with an empty Inventory
    cache = {key: Inventory()}
    return SCEDCS3DataStore(os.path.join(os.path.dirname(__file__), "./data/s3scedc"), cache)


read_channels = [
    (SCEDCS3DataStore._parse_channel("BKTHIS_LHZ00_2022002.ms")),
    (SCEDCS3DataStore._parse_channel("CIFOX2_LHZ___2022002.ms")),
    (SCEDCS3DataStore._parse_channel("CINCH__LHZ___2022002.ms")),
]


@pytest.mark.parametrize("channel", read_channels)
def test_read(store: SCEDCS3DataStore, channel: str):
    data = store.read_data(timespan1, channel)
    assert data.size > 0


def test_timespan_channels(store: SCEDCS3DataStore):
    timespans = store.get_timespans()
    assert len(timespans) == 1
    assert timespans[0] == timespan1
    channels = store.get_channels(timespan1)
    assert len(channels) == 3
    channels = store.get_channels(timespan2)
    assert len(channels) == 0
