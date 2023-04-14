from dataclasses import dataclass
from enum import Enum


@dataclass
class ChannelType:
    """
    A type of channel, e.g. BHN, but not associated with a particular station
    """

    name: str

    def __post_init__(self):
        assert (
            len(self.name) == 3 or len(self.name) == 6
        ), "A channel type name should be length 3 (e.g. bhn) or 6 (e.g. bhn_00)"

    def get_orientation(self) -> str:
        if "_" in self.name:
            return self.name.split("_")[0][-1]
        else:
            assert len(self.name) == 3
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
