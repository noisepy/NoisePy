import logging
from abc import ABC, abstractmethod

import diskcache as dc
import obspy
import pandas as pd
from datetimerange import DateTimeRange
from obspy import UTCDateTime

from .datatypes import Channel, Station

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

    @abstractmethod
    def get_full_channel(self, timespan: DateTimeRange, channel: Channel) -> Channel:
        pass


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
        inv = self._get_inventory(timespan)
        return self.populate_from_inventory(inv, channel)

    def _get_cache_key(self, timespan: DateTimeRange) -> str:
        return f"{self.url_key}_{timespan}"

    def _get_inventory(self, ts: DateTimeRange) -> obspy.Inventory:
        key = FDSNChannelCatalog._get_cache_key(ts)
        inventory = self.cache.get(key, None)
        if inventory is None:
            logging.info(f"Inventory not found in cache for key: '{key}'. Fetching from {self.url_key}.")
            bulk_station_request = [("*", "*", "*", "*", UTCDateTime(ts.start_datetime), UTCDateTime(ts.end_datetime))]
            client = obspy.Client(self.url_key)
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


# TODO: A channel catalog that uses the files in the SCEDC S3 bucket: s3://scedc-pds/FDSNstationXML/
