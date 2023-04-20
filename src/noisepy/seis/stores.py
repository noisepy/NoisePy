from abc import ABC, abstractmethod
from datetime import datetime
from typing import List

import numpy as np
from datetimerange import DateTimeRange

from .datatypes import Channel, ChannelData, CrossCorrelationParameters, Station


class DataStore(ABC):
    """
    A base abstraction over a data source for seismic data
    """

    # TODO: Are these needed or is get_channels() enough?
    # @abstractmethod
    # def get_stations(self) -> List[Station]:
    #     pass

    # @abstractmethod
    # def get_channel_types(self) -> List[ChannelType]:
    #     pass

    @abstractmethod
    def get_channels(self, timespan: DateTimeRange) -> List[Channel]:
        pass

    @abstractmethod
    def get_timespans(self) -> List[DateTimeRange]:
        pass

    def get_all_stations(self) -> List[Station]:
        timespans = self.get_timespans()
        stations = set(ch.station for ts in timespans for ch in self.get_channels(ts))
        return list(stations)


class RawDataStore(DataStore):
    """
    A class for reading the raw data for a given channel.
    TODO: write support?
    """

    @abstractmethod
    def read_data(self, timespan: DateTimeRange, chan: Channel) -> ChannelData:
        pass


class CrossCorrelationDataStore(DataStore):
    @abstractmethod
    def contains(self, timespan: DateTimeRange, chan1: Channel, chan2: Channel):
        pass

    @abstractmethod
    def append(
        self,
        timespan: DateTimeRange,
        chan1: Channel,
        chan2: Channel,
        parameters: CrossCorrelationParameters,
        data: np.ndarray,
    ):
        pass

    @abstractmethod
    def read(self, chan1: Channel, chan2: Channel, start: datetime, end: datetime) -> np.ndarray:
        pass
