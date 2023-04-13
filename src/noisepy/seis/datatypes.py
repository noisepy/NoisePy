from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List

import numpy as np
from datetimerange import DateTimeRange

from noisepy.seis import noise_module


@dataclass
class ChannelType:
    """
    A type of channel, e.g. BHN, but not associated with a particular station
    """

    name: str

    def __init__(self, name):
        assert len(name) == 3, "A channel type name should be length 3"
        self.name = name

    def get_orientation(self) -> str:
        return self.name[-1]


@dataclass
class Station:
    network: str  # 2 chars
    name: str  # 3-5 chars
    lat: float
    lon: float
    elevation: float
    location: str

    def __repr__(self) -> str:
        return f"{self.network}.{self.name}"


class CorrelationMethod(Enum):
    XCORR = 1
    DECONV = 2


@dataclass
class CrossCorrelationParameters:
    method: CorrelationMethod
    substack: bool
    # other parameters here


@dataclass
class Channel:
    """
    A channel instance belonging to a station. E.g. CI.ARV.BHN
    """

    type: ChannelType
    station: Station

    def __repr__(self) -> str:
        return f"{self.station}.{self.name}"


class DataSource(ABC):
    """
    A base abstraction over a data source for seismic data
    """

    @abstractmethod
    def get_stations() -> List[Station]:
        pass

    @abstractmethod
    def get_channel_types(self) -> List[ChannelType]:
        pass

    @abstractmethod
    def get_channels(self) -> List[Channel]:
        pass

    @abstractmethod
    def get_timespans(self) -> List[DateTimeRange]:
        pass


class RawDataSource(DataSource):
    """
    A class for reading the raw data for a given channel.
    TODO: write support
    """

    @abstractmethod
    def read_data(self, timespan: DateTimeRange, chan: Channel) -> np.ndarray:
        pass


class CrossCorrelationDataStore(DataSource):
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


# dummy
def fft(d):
    pass


def timespan_cc(ts: DateTimeRange, source: RawDataSource, store: CrossCorrelationDataStore):
    channels = source.get_channels()
    ffts = {}
    for chn in source.get_channels():
        data = source.read_data(ts, chn)
        ffts[chn] = fft(data)

    # compute cross correlations
    for ich1, chn1 in enumerate(source.get_channels()):
        for ich2 in range(ich1, len(channels)):
            chn2 = channels[ich2]
            if store.contains(ts, chn1, chn2):
                continue  # already done

            corr = noise_module.correlate(ffts[chn1], ffts[chn2])
            store.append(ts, chn1, chn2, CrossCorrelationParameters(), corr)


def cross_correlation_local(source: RawDataSource, store: CrossCorrelationDataStore):
    for ts in source.get_timespans():
        timespan_cc(ts, source, store)


def cross_correlation_mpi(source: RawDataSource, store: CrossCorrelationDataStore):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    timespans = source.get_timespans()

    timespans = comm.bcast(timespans, root=0)
    for i in range(rank, len(timespans), size):
        timespan_cc(timespans[i], source, store)

    comm.barrier()
