from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from datetimerange import DateTimeRange

from .datatypes import Channel, ChannelData, FFTParameters


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


class RawDataStore(DataStore):
    """
    A class for reading the raw data for a given channel.
    TODO: write support?
    """

    @abstractmethod
    def read_data(self, timespan: DateTimeRange, chan: Channel) -> ChannelData:
        pass


class CrossCorrelationDataStore:
    @abstractmethod
    def contains(self, timespan: DateTimeRange, chan1: Channel, chan2: Channel, parameters: FFTParameters) -> bool:
        pass

    @abstractmethod
    def save_parameters(self, parameters: FFTParameters):
        pass

    @abstractmethod
    def append(
        self,
        timespan: DateTimeRange,
        chan1: Channel,
        chan2: Channel,
        parameters: FFTParameters,
        cc_params: Dict[str, Any],
        data: np.ndarray,
    ):
        pass

    @abstractmethod
    def read(self, chan1: Channel, chan2: Channel, start: datetime, end: datetime) -> np.ndarray:
        pass