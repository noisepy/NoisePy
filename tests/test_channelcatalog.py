import os

import obspy
import pandas as pd
import pytest
from datetimerange import DateTimeRange
from obspy import UTCDateTime

from noisepy.seis.channelcatalog import ChannelCatalog, CSVChannelCatalog
from noisepy.seis.datatypes import Channel, ChannelType, Station
from noisepy.seis.noise_module import stats2inv_mseed

chan_data = [("ARV", "BHE", 35.1269, -118.83009, 258.0), ("BAK", "BHZ", 35.34444, -119.10445, 116.0)]

file = os.path.join(os.path.dirname(__file__), "./data/station.txt")


@pytest.mark.parametrize("stat,ch,lat,lon,elev", chan_data)
def test_csv(stat: str, ch: str, lat: float, lon: float, elev: float):
    cat = CSVChannelCatalog(file)
    chan = Channel(ChannelType(ch), Station("CI", stat, -1, -1, -1, -1))
    full_ch = cat.get_full_channel(DateTimeRange(), chan)
    assert full_ch.station.lat == lat
    assert full_ch.station.lon == lon
    assert full_ch.station.elevation == elev


class MockCatalog(ChannelCatalog):
    def get_full_channel(self, timespan: DateTimeRange, channel: Channel) -> Channel:
        pass

    def get_inventory(self, timespan: DateTimeRange, station: Station) -> obspy.Inventory:
        return obspy.Inventory()


@pytest.mark.parametrize("station,ch,lat,lon,elev", chan_data)
def test_frominventory(station: str, ch: str, lat: float, lon: float, elev: float):
    file = os.path.join(os.path.dirname(__file__), "./data/station.txt")
    df = pd.read_csv(file)

    class MockStat:
        station = ""
        starttime = UTCDateTime()
        channel = ch
        sampling_rate = 1.0
        location = "00"

    stat = MockStat()
    stat.station = station

    inv = stats2inv_mseed(stat, df)
    cat = MockCatalog()
    chan = Channel(ChannelType(ch), Station("CI", station, -1, -1, -1, -1))
    full_ch = cat.populate_from_inventory(inv, chan)
    assert full_ch.station.lat == lat
    assert full_ch.station.lon == lon
    assert full_ch.station.elevation == elev
