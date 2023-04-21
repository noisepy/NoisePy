import glob
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from typing import List

import numpy as np
import obspy
from datetimerange import DateTimeRange

from noisepy.seis.channelcatalog import ChannelCatalog
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

    def __init__(self, directory: str, chan_catalog: ChannelCatalog):
        super().__init__()
        self.directory = directory
        self.channel_catalog = chan_catalog
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
        logger.info(
            f"Init: {len(self.channels)} timespans and {sum(len(ch) for ch in  self.channels.values())} channels"
        )

    def get_channels(self, timespan: DateTimeRange) -> List[Channel]:
        tmp_channels = self.channels.get(str(timespan), [])
        return list(map(lambda c: self.channel_catalog.get_full_channel(timespan, c), tmp_channels))

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
