import datetime
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import obspy
from datetimerange import DateTimeRange

from noisepy.seis.constants import DATE_FORMAT

from .datatypes import (
    Channel,
    ChannelData,
    ChannelType,
    CrossCorrelation,
    Stack,
    Station,
)


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
    def contains(self, timespan: DateTimeRange, src: Station, rec: Station) -> bool:
        pass

    @abstractmethod
    def append(
        self,
        timespan: DateTimeRange,
        src: Station,
        rec: Station,
        ccs: List[CrossCorrelation],
    ):
        pass

    @abstractmethod
    def get_timespans(self) -> List[DateTimeRange]:
        pass

    @abstractmethod
    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        pass

    @abstractmethod
    def read_correlations(self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station) -> List[CrossCorrelation]:
        pass

    # private helpers
    def _get_station_pair(self, src_sta: Station, rec_sta: Station) -> str:
        return f"{src_sta}_{rec_sta}"

    def _get_channel_pair(self, src_chan: ChannelType, rec_chan: ChannelType) -> str:
        return f"{src_chan}_{rec_chan}"


class StackStore:
    """
    A class for reading and writing stack data
    """

    @abstractmethod
    def append(
        self, src: Station, rec: Station, components: str, name: str, stack_params: Dict[str, Any], data: np.ndarray
    ):
        pass

    @abstractmethod
    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        pass

    @abstractmethod
    def read_stacks(self, src: Station, rec: Station) -> List[Stack]:
        pass


def timespan_str(timespan: DateTimeRange) -> str:
    return f"{timespan.start_datetime.strftime(DATE_FORMAT)}T{timespan.end_datetime.strftime(DATE_FORMAT)}"


def parse_station_pair(pair: str) -> Optional[Tuple[Station, Station]]:
    # Parse from: CI.ARV_CI.BAK
    def station(sta: str) -> Optional[Station]:
        parts = sta.split(".")
        if len(parts) != 2:
            return None
        return Station(parts[0], parts[1])

    if re.match(r"([A-Z0-9]+)\.([A-Z0-9]+)_([A-Z0-9]+)\.([A-Z0-9]+)", pair, re.IGNORECASE) is None:
        return None

    tup = tuple(map(station, pair.split("_")))
    if None in tup:
        return None
    return tup


def parse_timespan(filename: str) -> Optional[DateTimeRange]:
    parts = os.path.splitext(os.path.basename(filename))[0].split("T")
    if parts is None or len(parts) != 2:
        return None
    dates = [obspy.UTCDateTime(p).datetime.replace(tzinfo=datetime.timezone.utc) for p in parts]
    return DateTimeRange(dates[0], dates[1])
