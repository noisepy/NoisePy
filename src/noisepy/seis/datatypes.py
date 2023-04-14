from dataclasses import dataclass
from enum import Enum


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
        return f"{self.station}.{self.type}"
