import os

from test_channelcatalog import MockCatalog
from test_scedc_s3store import timespan1

from noisepy.seis.channel_filter_store import LocationChannelFilterStore
from noisepy.seis.scedc_s3store import SCEDCS3DataStore


def test_location_filtering():
    # This folder has 4 channel .ms files, 2 of which are the same channel, different location
    path = os.path.join(os.path.dirname(__file__), "./data/s3scedc")
    store = SCEDCS3DataStore(path, MockCatalog())
    channels = store.get_channels(timespan1)
    assert len(channels) == 4
    filter_store = LocationChannelFilterStore(store)
    # This should filter out the BKTHIS_LHZ10 channel and leave the BKTHIS_LHZ00 channel
    channels = filter_store.get_channels(timespan1)
    assert len(channels) == 3
    bkthis_chan = next(filter(lambda c: c.station.network == "BK", channels))
    assert bkthis_chan.type.location == "00"
