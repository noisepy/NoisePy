import logging
from abc import ABC, abstractmethod
from functools import lru_cache

import diskcache as dc
import obspy
import pandas as pd
from datetimerange import DateTimeRange
from obspy import UTCDateTime, read_inventory
from obspy.clients.fdsn import Client

from .datatypes import Channel, Station
from .utils import fs_join, get_filesystem

logger = logging.getLogger(__name__)


class ChannelCatalog(ABC):
    """
    An abstract catalog for getting full channel information (lat, lon, elev, resp)
    """

    def populate_from_inventory(self, inv: obspy.Inventory, ch: Channel) -> Channel:
        filtered = inv.select(network=ch.station.network, station=ch.station.name, channel=ch.type.name)
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

    def get_full_channel(self, timespan: DateTimeRange, channel: Channel) -> Channel:
        inv = self.get_inventory(timespan, channel.station)
        return self.populate_from_inventory(inv, channel)

    @abstractmethod
    def get_inventory(self, timespan: DateTimeRange, station: Station) -> obspy.Inventory:
        pass


class XMLStationChannelCatalog(ChannelCatalog):
    """
    A channel catalog that reads <station>.XML files from a directory or an s3://... bucket url path.
    """

    def __init__(self, xmlpath: str, path_format: str = "{network}_{name}.xml") -> None:
        """
        Constructs a XMLStationChannelCatalog

        Args:
            xmlpath (str): Base directory where to find the files
            path_format (str): Format string to construct the file name from a station.
                               The argument names are 'network' and 'name'.
        """
        super().__init__()
        self.xmlpath = xmlpath
        self.path_format = path_format
        self.fs = get_filesystem(xmlpath)
        if not self.fs.exists(self.xmlpath):
            raise Exception(f"The XML Station file directory '{xmlpath}' doesn't exist")

    def get_inventory(self, timespan: DateTimeRange, station: Station) -> obspy.Inventory:
        file_name = self.path_format.format(network=station.network, name=station.name)
        xmlfile = fs_join(self.xmlpath, file_name)
        return self._get_inventory_from_file(xmlfile)

    @lru_cache
    def _get_inventory_from_file(self, xmlfile):
        if not self.fs.exists(xmlfile):
            logger.warning(f"Could not find StationXML file {xmlfile}. Returning empty Inventory()")
            return obspy.Inventory()
        with self.fs.open(xmlfile) as f:
            return read_inventory(f)


class FDSNChannelCatalog(ChannelCatalog):
    """
    A channel catalog that queries the FDSN service
    FDSN ~ International Federation of Digital Seismograph Network
    """

    def __init__(
        self,
        url_key: str,
        cache_dir: str,
    ):
        super().__init__()
        self.url_key = url_key

        logger.info(f"Cache dir: ${cache_dir}")
        self.cache = dc.Cache(cache_dir)

    def get_full_channel(self, timespan: DateTimeRange, channel: Channel) -> Channel:
        inv = self._get_inventory(str(timespan))
        return self.populate_from_inventory(inv, channel)

    def get_inventory(self, timespan: DateTimeRange, station: Station) -> obspy.Inventory:
        return self._get_inventory(str(timespan))

    def _get_cache_key(self, ts_str: str) -> str:
        return f"{self.url_key}_{ts_str}"

    @lru_cache
    # pass the timestamp (DateTimeRange) as string so that the method is cacheable
    # since DateTimeRange is not hasheable
    def _get_inventory(self, ts_str: str) -> obspy.Inventory:
        ts = DateTimeRange.from_range_text(ts_str)
        key = self._get_cache_key(ts_str)
        inventory = self.cache.get(key, None)
        if inventory is None:
            logging.info(f"Inventory not found in cache for key: '{key}'. Fetching from {self.url_key}.")
            bulk_station_request = [("*", "*", "*", "*", UTCDateTime(ts.start_datetime), UTCDateTime(ts.end_datetime))]
            client = Client(self.url_key)
            inventory = client.get_stations_bulk(bulk_station_request, level="channel")
            self.cache[key] = inventory
        return inventory


class CSVChannelCatalog(ChannelCatalog):
    """
    A channel catalog implentations that reads the station csv file
    """

    def __init__(self, file: str):
        self.df = pd.read_csv(file)

    def get_full_channel(self, timespan: DateTimeRange, ch: Channel) -> Channel:
        ista = self.df[self.df["station"] == ch.station.name].index.values.astype("int64")[0]
        return Channel(
            ch.type,
            Station(
                network=ch.station.network,
                name=ch.station.name,
                lat=self.df.iloc[ista]["latitude"],
                lon=self.df.iloc[ista]["longitude"],
                elevation=self.df.iloc[ista]["elevation"],
                location=ch.station.location,
            ),
        )

    def get_inventory(self, timespan: DateTimeRange, station: Station) -> obspy.Inventory:
        return None


# TODO: A channel catalog that uses the files in the SCEDC S3 bucket: s3://scedc-pds/FDSNstationXML/
