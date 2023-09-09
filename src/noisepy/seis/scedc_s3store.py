import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Callable, List

import obspy
from datetimerange import DateTimeRange

from noisepy.seis.channelcatalog import ChannelCatalog
from noisepy.seis.stores import RawDataStore

from .datatypes import Channel, ChannelData, ChannelType, Station
from .utils import TimeLogger, fs_join, get_filesystem

logger = logging.getLogger(__name__)


def channel_filter(stations: List[str], ch_prefixes: str) -> Callable[[Channel], bool]:
    """
    Helper function for creating a channel filter to be used in the constructor of the store.
    This filter uses a list of allowed station name along with a channel filter prefix.
    """
    sta_set = set(stations)

    def filter(ch: Channel) -> bool:
        if sta_set == {"*"}:
            return ch.type.name.lower().startswith(tuple(ch_prefixes.lower().split(",")))
        else:
            return ch.station.name in sta_set and ch.type.name.lower().startswith(tuple(ch_prefixes.lower().split(",")))

    return filter


class SCEDCS3DataStore(RawDataStore):
    """
    A data store implementation to read from a directory of miniSEED (.ms) files from the SCEDC S3 bucket.
    Every directory is a a day and each .ms file contains the data for a channel.
    """

    # TODO: Support reading directly from the S3 bucket

    # for checking the filename has the form: CIGMR__LHN___2022002.ms
    file_re = re.compile(r".*[0-9]{7}\.ms$", re.IGNORECASE)

    def __init__(
        self,
        path: str,
        chan_catalog: ChannelCatalog,
        ch_filter: Callable[[Channel], bool] = lambda s: True,  # noqa: E731
        date_range: DateTimeRange = None,
        storage_options: dict = {},
    ):
        """
        Parameters:
            path: path to look for ms files. Can be a local file directory or an s3://... url path
            chan_catalog: ChannelCatalog to retrieve inventory information for the channels
            channel_filter: Function to decide whether a channel should be used or not,
                            if None, all channels are used
        """
        super().__init__()
        self.fs = get_filesystem(path, storage_options=storage_options)
        self.channel_catalog = chan_catalog
        self.path = path
        self.paths = {}
        # to store a dict of {timerange: list of channels}
        self.channels = defaultdict(list)
        self.ch_filter = ch_filter
        if date_range is not None and date_range.start_datetime.tzinfo is None:
            start_datetime = date_range.start_datetime.replace(tzinfo=timezone.utc)
            end_datetime = date_range.end_datetime.replace(tzinfo=timezone.utc)
            date_range = DateTimeRange(start_datetime, end_datetime)

        self.date_range = date_range

        if date_range is None:
            self._load_channels(self.path, ch_filter)

    def _load_channels(self, full_path: str, ch_filter: Callable[[Channel], bool]):
        tlog = TimeLogger(logger=logger, level=logging.INFO)
        msfiles = [f for f in self.fs.glob(fs_join(full_path, "*.ms")) if self.file_re.match(f) is not None]
        tlog.log(f"Loading {len(msfiles)} files from {full_path}")
        for f in msfiles:
            timespan = SCEDCS3DataStore._parse_timespan(f)
            self.paths[timespan.start_datetime] = full_path
            channel = SCEDCS3DataStore._parse_channel(os.path.basename(f))
            if not ch_filter(channel):
                continue
            key = str(timespan)  # DataTimeFrame is not hashable
            self.channels[key].append(channel)
        tlog.log(f"Init: {len(self.channels)} timespans and {sum(len(ch) for ch in  self.channels.values())} channels")

    def _ensure_channels_loaded(self, date_range: DateTimeRange):
        key = str(date_range)
        if key not in self.channels or date_range.start_datetime not in self.paths:
            dt = date_range.end_datetime - date_range.start_datetime
            for d in range(0, dt.days):
                date = date_range.start_datetime + timedelta(days=d)
                if self.date_range is None or date not in self.date_range:
                    continue
                date_path = str(date.year) + "/" + str(date.year) + "_" + str(date.timetuple().tm_yday).zfill(3) + "/"
                full_path = fs_join(self.path, date_path)
                self._load_channels(full_path, self.ch_filter)

    def get_channels(self, date_range: DateTimeRange) -> List[Channel]:
        self._ensure_channels_loaded(date_range)
        tmp_channels = self.channels.get(str(date_range), [])
        executor = ThreadPoolExecutor()
        stations = set(map(lambda c: c.station, tmp_channels))
        _ = list(executor.map(lambda s: self.channel_catalog.get_inventory(date_range, s), stations))
        logger.info(f"Getting {len(tmp_channels)} channels for {date_range}: {tmp_channels}")
        return list(executor.map(lambda c: self.channel_catalog.get_full_channel(date_range, c), tmp_channels))

    def get_timespans(self) -> List[DateTimeRange]:
        if self.date_range is not None:
            days = (self.date_range.end_datetime - self.date_range.start_datetime).days
            return [
                DateTimeRange(
                    self.date_range.start_datetime.replace(tzinfo=timezone.utc) + timedelta(days=d),
                    self.date_range.start_datetime.replace(tzinfo=timezone.utc) + timedelta(days=d + 1),
                )
                for d in range(0, days)
            ]
        return list([DateTimeRange.from_range_text(d) for d in sorted(self.channels.keys())])

    def read_data(self, timespan: DateTimeRange, chan: Channel) -> ChannelData:
        self._ensure_channels_loaded(timespan)
        # reconstruct the file name from the channel parameters
        chan_str = (
            f"{chan.station.network}{chan.station.name.ljust(5, '_')}{chan.type.name}"
            f"{chan.station.location.ljust(3, '_')}"
        )
        filename = fs_join(
            self.paths[timespan.start_datetime], f"{chan_str}{timespan.start_datetime.strftime('%Y%j')}.ms"
        )
        if not self.fs.exists(filename):
            logger.warning(f"Could not find file {filename}")
            return ChannelData.empty()

        with self.fs.open(filename) as f:
            stream = obspy.read(f)
        data = ChannelData(stream)
        return data

    def get_inventory(self, timespan: DateTimeRange, station: Station) -> obspy.Inventory:
        return self.channel_catalog.get_inventory(timespan, station)

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
            ChannelType(channel, location),
            # lat/lon/elev will be populated later
            Station(network, station, location=location),
        )
