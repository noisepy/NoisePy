import io
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Callable, List, Tuple

import obspy
from datetimerange import DateTimeRange

from noisepy.seis.channelcatalog import ChannelCatalog
from noisepy.seis.stores import RawDataStore

from .datatypes import Channel, ChannelData, ChannelType, Station
from .utils import fs_join, get_filesystem

logger = logging.getLogger(__name__)


class PNWDataStore(RawDataStore):
    """
    A data store implementation to read from a SQLite DB of metadata and a directory of data files
    """

    def __init__(
        self,
        path: str,
        chan_catalog: ChannelCatalog,
        db_file: str,
        ch_filter: Callable[[Channel], bool] = None,
        date_range: DateTimeRange = None,
    ):
        """
        Parameters:
            path: path to look for ms files. Can be a local file directory or an s3://... url path
            chan_catalog: ChannelCatalog to retrieve inventory information for the channels
            db_file: path to the sqlite DB file
            channel_filter: Optional function to decide whether a channel should be used or not,
                            if None, all channels are used
            date_range: Optional date range to filter the data
        """
        super().__init__()
        self.fs = get_filesystem(path)
        self.channel_catalog = chan_catalog
        self.path = path
        self.db_file = db_file
        self.paths = {}
        # to store a dict of {timerange: list of channels}
        self.channels = {}
        if ch_filter is None:
            ch_filter = lambda s: True  # noqa: E731

        if date_range is None:
            self._load_channels(self.path, ch_filter)
        else:
            dt = date_range.end_datetime - date_range.start_datetime
            for d in range(0, dt.days):
                date = date_range.start_datetime + timedelta(days=d)
                date_path = str(date.year) + "/" + str(date.timetuple().tm_yday).zfill(3) + "/"
                full_path = fs_join(self.path, date_path)
                self._load_channels(full_path, ch_filter)

    def _load_channels(self, full_path: str, ch_filter: Callable[[Channel], bool]):
        # The path should look like: .../UW/2020/125/
        parts = full_path.split(os.path.sep)
        assert len(parts) >= 4
        net, year, doy = parts[-4:-1]

        rst = self._dbquery(
            f"SELECT network, station, channel, location, filename "
            f"FROM tsindex WHERE filename LIKE '%%/{net}/{year}/{doy}/%%'"
        )
        timespans = []
        for i in rst:
            timespan = PNWDataStore._parse_timespan(os.path.basename(i[4]))
            self.paths[timespan.start_datetime] = full_path
            channel = PNWDataStore._parse_channel(i)
            if not ch_filter(channel):
                continue
            key = str(timespan)
            if key not in self.channels:
                timespans.append(timespan)
                self.channels[key] = [channel]
            else:
                self.channels[key].append(channel)

    def get_channels(self, timespan: DateTimeRange) -> List[Channel]:
        tmp_channels = self.channels.get(str(timespan), [])
        return list(map(lambda c: self.channel_catalog.get_full_channel(timespan, c), tmp_channels))

    def get_timespans(self) -> List[DateTimeRange]:
        return list([DateTimeRange.from_range_text(d) for d in sorted(self.channels.keys())])

    def read_data(self, timespan: DateTimeRange, chan: Channel) -> ChannelData:
        assert timespan.start_datetime.year == timespan.end_datetime.year, "Did not expect timespans to cross years"
        year = timespan.start_datetime.year
        doy = str(timespan.start_datetime.timetuple().tm_yday).zfill(3)

        rst = self._dbquery(
            f"SELECT byteoffset, bytes "
            f"FROM tsindex WHERE network='{chan.station.network}' AND station='{chan.station.name}' "
            f"AND channel='{chan.type.name}' and location='{chan.station.location}' "
            f"AND filename LIKE '%%/{chan.station.network}/{year}/{doy}/%%'"
        )

        if len(rst) == 0:
            logger.warning(f"Could not find file {timespan}/{chan} in the database")
            return ChannelData.empty()
        byteoffset, bytes = rst[0]

        # reconstruct the file name from the channel parameters
        chan_str = f"{chan.station.name}.{chan.station.network}.{timespan.start_datetime.strftime('%Y.%j')}"
        filename = fs_join(self.paths[timespan.start_datetime].replace("__", chan.station.network), f"{chan_str}")
        if not self.fs.exists(filename):
            logger.warning(f"Could not find file {filename}")
            return ChannelData.empty()

        with self.fs.open(filename, "rb") as f:
            f.seek(byteoffset)
            buff = io.BytesIO(f.read(bytes))
            stream = obspy.read(buff)
        return ChannelData(stream)

    def get_inventory(self, timespan: DateTimeRange, station: Station) -> obspy.Inventory:
        return self.channel_catalog.get_inventory(timespan, station)

    def _parse_timespan(filename: str) -> DateTimeRange:
        # The PNWStore repository stores files in the form: STA.NET.YYYY.DOY
        # YA2.UW.2020.366
        year = int(filename.split(".")[2])
        day = int(filename.split(".")[3])
        jan1 = datetime(year, 1, 1, tzinfo=timezone.utc)
        return DateTimeRange(jan1 + timedelta(days=day - 1), jan1 + timedelta(days=day))

    def _parse_channel(record: tuple) -> Channel:
        # e.g.
        # YA2.UW.2020.366
        network = record[0]
        station = record[1]
        channel = record[2]
        location = record[3]
        c = Channel(
            ChannelType(channel, location),
            # lat/lon/elev will be populated later
            Station(network, station, location=location),
        )
        return c

    def _dbquery(self, query: str) -> List[Tuple]:
        db = sqlite3.connect(self.db_file)
        cursor = db.cursor()
        rst = cursor.execute(query)
        all = rst.fetchall()
        db.close()
        return all
