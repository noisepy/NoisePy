from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np
import obspy
from datetimerange import DateTimeRange

from .constants import DATE_FORMAT, DONE_PATH, PROGRESS_DATATYPE
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

    # TODO: Temporary method to get a list of stations to pass to the
    # stack() function. It can be removed once the stacking uses
    # the DataStores instead of direct file access
    def get_station_list(self) -> List[str]:
        ts = self.get_timespans()
        chs = self.get_channels(ts[0])
        stations = sorted(set(map(lambda c: str(c.station), chs)))
        return stations


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
    def get_timespans(self) -> List[DateTimeRange]:
        pass

    @abstractmethod
    def get_station_pairs(self, timespan: DateTimeRange) -> List[Tuple[Station, Station]]:
        #    ccf_ds = self.datasets[timespan]
        #   data = ccf_ds.auxiliary_data.list()
        #  return [_parse_station_pair(p) for p in data if p != PROGRESS_DATATYPE]

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

    def _parse_station_pair(pair: str) -> Tuple[Station, Station]:
        # Parse from:'CI.ARV_CI.BAK
        def station(sta: str) -> Station:
            net, name = sta.split(".")
            return Station(net, name)

        return tuple(map(station, pair.split("_")))
