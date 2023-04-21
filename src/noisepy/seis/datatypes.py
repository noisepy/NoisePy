from dataclasses import dataclass
from enum import Enum

import numpy as np


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
        # comp = source[0].stats.channel
        # WIP: Fix to handle 'bhn_00'
        # comp = ch.type
        # if comp[-1] == "U":
        #     comp.replace("U", "Z")

    def get_basename(self):
        return self.name[0:3]

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

    def __hash__(self) -> int:
        return str(self).__hash__()


class CorrelationMethod(Enum):
    XCORR = 1
    DECONV = 2


@dataclass
class FFTParameters:
    dt: float = 1.0
    start_date: str = ""  # TODO: can we make this datetime?
    end_date: str = ""
    samp_freq: float = 20
    cc_len: int = 1800  # basic unit of data length for fft (sec)
    # pre-processing parameters
    step: int = 450  # overlapping between each cc_len (sec)
    freqmin: float = 0.05
    freqmax: float = 2.0
    freq_norm: str = "rma"
    time_norm: str = "no"  # 'no' for no normalization, or 'rma', 'one_bit' for normalization in time domain
    cc_method: str = "xcorr"  # 'xcorr' for pure cross correlation, 'deconv' for deconvolution;
    # FOR "COHERENCY" PLEASE set freq_norm to "rma", time_norm to "no" and cc_method to "xcorr"
    smooth_N: int = 10  # moving window length for time/freq domain normalization if selected (points)
    smoothspect_N: int = 10  # moving window length to smooth spectrum amplitude (points)
    # if substack=True, substack_len=2*cc_len, then you pre-stack every 2 correlation windows.
    # for instance: substack=True, substack_len=cc_len means that you keep ALL of the correlations
    substack: bool = True  # True = smaller stacks within the time chunk. False: it will stack over inc_hours
    substack_len: int = 1800  # how long to stack over (for monitoring purpose): need to be multiples of cc_len
    maxlag: int = 200  # lags of cross-correlation to save (sec)
    substack: bool = True
    inc_hours: int = 24
    # criteria for data selection
    max_over_std: int = 10  # threahold to remove window of bad signals: set it to 10*9 if prefer not to remove them
    ncomp: int = 3  # 1 or 3 component data (needed to decide whether do rotation)
    # station/instrument info for input_fmt=='sac' or 'mseed'
    stationxml: bool = False  # station.XML file used to remove instrument response for SAC/miniseed data
    rm_resp: str = "no"  # select 'no' to not remove response and use 'inv','spectrum',
    # some control parameters
    acorr_only: bool = False  # only perform auto-correlation
    xcorr_only: bool = True  # only perform cross-correlation or not

    # 'RESP', or 'polozeros' to remove response

    def __post_init__(self):
        self.dt = 1.0 / self.samp_freq
        assert self.substack_len % self.cc_len == 0

    # TODO: Remove once all uses of FFTParameters have been converted to use strongly typed access
    def __getitem__(self, key):
        return self.__dict__[key]


@dataclass
class Channel:
    """
    A channel instance belonging to a station. E.g. CI.ARV.BHN
    """

    type: ChannelType
    station: Station

    def __repr__(self) -> str:
        return f"{self.station}.{self.type}"


@dataclass
class ChannelData:
    """
    A 1D time series of channel data

    Attributes:
        data: series values
        sampling_rate: In HZ
        start_timestamp: Seconds since 01/01/1970
    """

    data: np.ndarray
    sampling_rate: float
    start_timestamp: float
