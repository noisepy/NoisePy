from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import obspy
from datetimerange import DateTimeRange

from .datatypes import Channel, ChannelData, ConfigParameters, Station


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
    def save_parameters(self, parameters: ConfigParameters):
        pass

    @abstractmethod
    def append(
        self,
        timespan: DateTimeRange,
        chan1: Channel,
        chan2: Channel,
        parameters: ConfigParameters,
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
    def read(self, chan1: Channel, chan2: Channel, start: datetime, end: datetime) -> np.ndarray:
        pass
