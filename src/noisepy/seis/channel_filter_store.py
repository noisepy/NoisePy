from typing import List

import obspy
from datetimerange import DateTimeRange

from .datatypes import Channel, ChannelData, Station
from .stores import RawDataStore


class LocationChannelFilterStore(RawDataStore):
    """
    This 'store' simply wraps another store and filters out duplicate channels that differ only
    by location. It does this by keeping the channel with the 'lowest' (lexicographic) location code.
    """

    def __init__(self, store: RawDataStore):
        super().__init__()
        self.store = store

    def get_timespans(self) -> List[DateTimeRange]:
        return self.store.get_timespans()

    def read_data(self, timespan: DateTimeRange, chan: Channel) -> ChannelData:
        return self.store.read_data(timespan, chan)

    def get_inventory(self, timespan: DateTimeRange, station: Station) -> obspy.Inventory:
        return self.store.get_inventory(timespan, station)

    def get_channels(self, timespan: DateTimeRange) -> List[Channel]:
        channels = self.store.get_channels(timespan)
        min_chans = {}
        for ch in channels:
            key = f"{ch.station.network}_{ch.station.name}_{ch.type.name}"
            if key not in min_chans:
                min_chans[key] = ch
            # lexicographic comparison of location codes
            #  http://docs.python.org/reference/expressions.html
            elif ch.type.location < min_chans[key].type.location:
                min_chans[key] = ch
        return list(min_chans.values())
