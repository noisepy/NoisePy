import glob
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np
import obspy
from datetimerange import DateTimeRange
from obspy import Inventory, UTCDateTime
from obspy.clients.fdsn import Client

from noisepy.seis.stores import RawDataStore

from .datatypes import Channel, ChannelData, Station

logger = logging.getLogger(__name__)


class SCEDCS3DataStore(RawDataStore):
    """
    A data store implementation to read from a directory of miniSEED (.ms) files from the SCEDC S3 bucket.
    Every directory is a a day and each .ms file contains the data for a channel.
    """

    # TODO: Support reading directly from the S3 bucket

    # for checking the filename has the form: CIGMR__LHN___2022002.ms
    file_re = re.compile(r".*[0-9]{7}\.ms$", re.IGNORECASE)

    def __init__(self, directory: str, inventory_cache: Dict):
        super().__init__()
        self.directory = directory
        msfiles = [f for f in glob.glob(os.path.join(directory, "*.ms")) if self.file_re.match(f) is not None]
        # store a dict of {timerange: list of channels}
        self.channels = {}
        timespans = []
        for f in msfiles:
            timespan = SCEDCS3DataStore._parse_timespan(f)
            channel = SCEDCS3DataStore._parse_channel(os.path.basename(f))
            key = str(timespan)  # DataTimeFrame is not hashable
            if key not in self.channels:
                timespans.append(timespan)
                self.channels[key] = [channel]
            else:
                self.channels[key].append(channel)

        # some channel information is encoded in the file name, but for lat/lon/elevation we need to
        # query the obspy service
        inventory = self._get_inventory(timespans, inventory_cache)
        for ts, tmp_channels in self.channels.items():
            self.channels[ts] = list(
                map(lambda c: SCEDCS3DataStore._populate_from_inventory(inventory, c), tmp_channels)
            )

        logger.info(
            f"Init: {len(self.channels)} timespans and {sum(len(ch) for ch in  self.channels.values())} channels"
        )

    def get_channels(self, timespan: DateTimeRange) -> List[Channel]:
        return self.channels.get(str(timespan), [])

    def get_timespans(self) -> List[DateTimeRange]:
        return list([DateTimeRange.from_range_text(d) for d in sorted(self.channels.keys())])

    def read_data(self, timespan: DateTimeRange, chan: Channel) -> ChannelData:
        # reconstruct the file name from the channel parameters
        chan_str = (
            f"{chan.station.network}{chan.station.name.ljust(5, '_')}{chan.type}{chan.station.location.ljust(3, '_')}"
        )
        filename = os.path.join(self.directory, f"{chan_str}{timespan.start_datetime.strftime('%Y%j')}.ms")
        if not os.path.exists(filename):
            logger.warning(f"Could not find file {filename}")
            return np.empty
        stream = obspy.read(filename)[0]
        return ChannelData(stream.data, stream.stats.sampling_rate, stream.stats.starttime.timestamp)

    def _parse_timespan(filename: str) -> DateTimeRange:
        # The SCEDC S3 bucket stores files in the form: CIGMR__LHN___2022002.ms
        year = int(filename[-10:-6])
        day = int(filename[-5:-3])
        jan1 = datetime(year, 1, 1, tzinfo=timezone.utc)
        return DateTimeRange(jan1 + timedelta(days=day - 1), jan1 + timedelta(days=day))

    def _parse_channel(filename: str) -> Channel:
        # e.g.
        # CIGMR__LHN___2022002
        # CE13884HNZ10_2022002
        network = filename[:2]
        station = filename[2:7].rstrip("_")
        channel = filename[7:10]
        location = filename[10:12].strip("_")
        return Channel(
            channel,
            # lat/lon/elev will be populated later
            Station(network, station, -1, -1, -1, location),
        )

    def _populate_from_inventory(inv: Inventory, ch: Channel) -> Channel:
        filtered = inv.select(network=ch.station.network, station=ch.station.name, channel=ch.type)
        if (
            len(filtered) == 0
            or len(filtered.networks[0].stations) == 0
            or len(filtered.networks[0].stations[0].channels) == 0
        ):
            logger.warning(f"Could not find channel {ch} in the inventory")
            return ch

        inv_chan = filtered.networks[0].stations[0].channels[0]
        return Channel(
            ch.type,
            Station(
                network=ch.station.network,
                name=ch.station.name,
                lat=inv_chan.latitude,
                lon=inv_chan.longitude,
                elevation=inv_chan.elevation,
                location=ch.station.location,
            ),
        )

    def _get_cache_key(timespans: List[DateTimeRange]) -> str:
        return "_".join([str(t) for t in timespans])

    def _get_inventory(self, timespans: List[DateTimeRange], cache: Dict) -> Inventory:
        key = SCEDCS3DataStore._get_cache_key(timespans)
        inventory = cache.get(key, None)
        if inventory is None:
            logging.info(f"Inventory not found in cache for key: '{key}'. Fetching from SCEDC.")
            bulk_station_request = [
                ("*", "*", "*", "*", UTCDateTime(ts.start_datetime), UTCDateTime(ts.end_datetime)) for ts in timespans
            ]
            client = Client("SCEDC")
            inventory = client.get_stations_bulk(bulk_station_request, level="channel")
            cache[key] = inventory
        return inventory
