import datetime
import logging
import os
import re
from abc import ABC, abstractmethod
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Generic, List, Optional, Tuple, TypeVar

import obspy
from datetimerange import DateTimeRange

from noisepy.seis.constants import DATE_FORMAT
from noisepy.seis.utils import TimeLogger, get_results

from .datatypes import (
    AnnotatedData,
    Channel,
    ChannelData,
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


T = TypeVar("T", bound=AnnotatedData)


class ComputedDataStore(Generic[T]):
    """
    A class for reading and writing cross-correlation data
    """

    @abstractmethod
    def contains(self, src: Station, rec: Station, timespan: DateTimeRange) -> bool:
        pass

    @abstractmethod
    def append(
        self,
        timespan: DateTimeRange,
        src: Station,
        rec: Station,
        ccs: List[T],
    ):
        pass

    @abstractmethod
    def get_timespans(self, src_sta: Station, rec_sta: Station) -> List[DateTimeRange]:
        pass

    @abstractmethod
    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        pass

    @abstractmethod
    def read(self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station) -> List[T]:
        pass

    def read_bulk(
        self, timespan: DateTimeRange, pairs: List[Tuple[Station, Station]], executor: Executor = ThreadPoolExecutor()
    ) -> List[Tuple[Tuple[Station, Station], List[T]]]:
        """
        Reads the data for all the given station pairs (and timespan) in parallel.
        """
        tlog = TimeLogger(level=logging.INFO)
        futures = [executor.submit(self.read, timespan, p[0], p[1]) for p in pairs]
        results = get_results(futures)
        tlog.log(f"loading {len(pairs)} stacks")
        return list(zip(pairs, results))


class CrossCorrelationDataStore(ComputedDataStore[CrossCorrelation]):
    pass


class StackStore(ComputedDataStore[Stack]):
    """
    A class for reading and writing stack data
    """

    pass


def timespan_str(timespan: DateTimeRange) -> str:
    return f"{timespan.start_datetime.strftime(DATE_FORMAT)}T{timespan.end_datetime.strftime(DATE_FORMAT)}"


def parse_station_pair(pair: str) -> Optional[Tuple[Station, Station]]:
    if re.match(r"([A-Z0-9]+)\.([A-Z0-9]+)_([A-Z0-9]+)\.([A-Z0-9]+)", pair, re.IGNORECASE) is None:
        return None

    tup = tuple(map(Station.parse, pair.split("_")))
    if None in tup:
        return None
    return tup


def parse_timespan(filename: str) -> Optional[DateTimeRange]:
    parts = os.path.splitext(os.path.basename(filename))[0].split("T")
    if parts is None or len(parts) != 2:
        return None
    dates = [obspy.UTCDateTime(p).datetime.replace(tzinfo=datetime.timezone.utc) for p in parts]
    return DateTimeRange(dates[0], dates[1])
