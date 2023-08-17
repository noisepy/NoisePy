import datetime
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import obspy
from datetimerange import DateTimeRange

from noisepy.seis.constants import DATE_FORMAT

from .datatypes import Channel, ChannelData, ChannelType, ConfigParameters, Station


class DataStore(ABC):
    """
    A base abstraction over a data source for seismic data
    """

    @abstractmethod
    def get_channels(self, timespan: DateTimeRange) -> List[Channel]:
        pass

    @abstractmethod
    def get_timespans(self) -> List[DateTimeRange]:
        pass


class RawDataStore(DataStore):
    """
    A class for reading the raw data for a given channel.
    TODO: write support?
    """

    @abstractmethod
    def read_data(self, timespan: DateTimeRange, chan: Channel) -> ChannelData:
        pass

    @abstractmethod
    def get_inventory(self, timespan: DateTimeRange, station: Station) -> obspy.Inventory:
        pass


class CrossCorrelationDataStore:
    @abstractmethod
    def contains(
        self, timespan: DateTimeRange, src_chan: Channel, rec_chan: Channel, parameters: ConfigParameters
    ) -> bool:
        pass

    @abstractmethod
    def append(
        self,
        timespan: DateTimeRange,
        src_chan: Channel,
        rec_chan: Channel,
        cc_params: Dict[str, Any],
        data: np.ndarray,
    ):
        pass

    @abstractmethod
    def is_done(self, timespan: DateTimeRange):
        pass

    @abstractmethod
    def mark_done(self, timespan: DateTimeRange):
        pass

    @abstractmethod
    def get_timespans(self) -> List[DateTimeRange]:
        pass

    @abstractmethod
    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        pass

    @abstractmethod
    def get_channeltype_pairs(
        self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station
    ) -> List[Tuple[ChannelType, ChannelType]]:
        pass

    @abstractmethod
    def read(
        self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station, src_ch: ChannelType, rec_ch: ChannelType
    ) -> Tuple[Dict, np.ndarray]:
        pass

    # private helpers
    def _get_station_pair(self, src_sta: Station, rec_sta: Station) -> str:
        return f"{src_sta}_{rec_sta}"

    def _get_channel_pair(self, src_chan: ChannelType, rec_chan: ChannelType) -> str:
        return f"{src_chan.name}_{rec_chan.name}"


class StackStore:
    """
    A class for reading and writing stack data
    """

    @abstractmethod
    def mark_done(self, src: Station, rec: Station):
        pass

    @abstractmethod
    def is_done(self, src: Station, rec: Station):
        pass

    @abstractmethod
    def append(
        self, src: Station, rec: Station, components: str, name: str, stack_params: Dict[str, Any], data: np.ndarray
    ):
        pass

    @abstractmethod
    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        pass

    @abstractmethod
    def get_stack_names(self, src: Station, rec: Station) -> List[str]:
        pass

    @abstractmethod
    def get_components(self, src: Station, rec: Station, name: str) -> List[str]:
        pass

    @abstractmethod
    def read(self, src: Station, rec: Station, component: str, name: str) -> Tuple[Dict[str, Any], np.ndarray]:
        pass


def timespan_str(timespan: DateTimeRange) -> str:
    return f"{timespan.start_datetime.strftime(DATE_FORMAT)}T{timespan.end_datetime.strftime(DATE_FORMAT)}"


def parse_station_pair(pair: str) -> Tuple[Station, Station]:
    # Parse from: CI.ARV_CI.BAK
    def station(sta: str) -> Station:
        net, name = sta.split(".")
        return Station(net, name)

    return tuple(map(station, pair.split("_")))


def parse_timespan(filename: str) -> DateTimeRange:
    parts = os.path.splitext(os.path.basename(filename))[0].split("T")
    dates = [obspy.UTCDateTime(p).datetime.replace(tzinfo=datetime.timezone.utc) for p in parts]
    return DateTimeRange(dates[0], dates[1])
