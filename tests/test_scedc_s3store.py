from datetime import datetime, timezone

import pytest
from datetimerange import DateTimeRange
from test_channelcatalog import MockCatalog

from noisepy.seis.scedc_s3store import SCEDCS3DataStore

timespan1 = DateTimeRange(datetime(2022, 1, 2, tzinfo=timezone.utc), datetime(2022, 1, 3, tzinfo=timezone.utc))
timespan2 = DateTimeRange(datetime(2021, 2, 3, tzinfo=timezone.utc), datetime(2021, 2, 4, tzinfo=timezone.utc))
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

    return SCEDCS3DataStore(os.path.join(os.path.dirname(__file__), "./data/s3scedc"), MockCatalog())


read_channels = [
    (SCEDCS3DataStore._parse_channel("BKTHIS_LHZ00_2022002.ms")),
    (SCEDCS3DataStore._parse_channel("CIFOX2_LHZ___2022002.ms")),
    (SCEDCS3DataStore._parse_channel("CINCH__LHZ___2022002.ms")),
]


@pytest.mark.parametrize("channel", read_channels)
def test_read(store: SCEDCS3DataStore, channel: str):
    chdata = store.read_data(timespan1, channel)
    assert chdata.sampling_rate == 1.0
    assert chdata.start_timestamp >= timespan1.start_datetime.timestamp()
    assert chdata.start_timestamp < timespan1.end_datetime.timestamp()
    assert chdata.data.size > 0


def test_timespan_channels(store: SCEDCS3DataStore):
    timespans = store.get_timespans()
    assert len(timespans) == 1
    assert timespans[0] == timespan1
    channels = store.get_channels(timespan1)
    assert len(channels) == 3
    channels = store.get_channels(timespan2)
    assert len(channels) == 0
