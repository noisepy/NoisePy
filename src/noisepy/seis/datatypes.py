from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import obspy


@dataclass
class ChannelType:
    """
    A type of channel, e.g. BHN, but not associated with a particular station
    """

    name: str
    location: str = ""

    def __post_init__(self):
        assert (
            len(self.name) == 3 or len(self.name) == 6
        ), "A channel type name should be length 3 (e.g. bhn) or 6 (e.g. bhn_00)"
        if "_" in self.name:
            parts = self.name.split("_")
            self.name = parts[0]
            self.location = parts[1]

        # Japanese channels use 'U' (up) for the vertical direction. Here we normalize to 'z'
        if self.name[-1] == "U":
            self.name = self.name.replace("U", "Z")

    def __repr__(self) -> str:
        if len(self.location) > 0:
            return f"{self.name}_{self.location}"
        else:
            return self.name

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
class ConfigParameters:
    dt: float = 0.05  # TODO: dt should be 1/sampling rate
    start_date: str = ""  # TODO: can we make this datetime?
    end_date: str = ""
    samp_freq: float = 20  # TODO: change this samp_freq for the obspy "sampling_rate"
    cc_len: float = 1800.0  # basic unit of data length for fft (sec)
    # download params.
    # Targeted region/station information: only needed when down_list is False
    lamin: float = 31
    lamax: float = 36
    lomin: float = -122
    lomax: float = -115
    down_list = False  # download stations from a pre-compiled list or not
    net_list = ["CI"]  # network list
    # pre-processing parameters
    step: float = 450.0  # overlapping between each cc_len (sec)
    freqmin: float = 0.05
    freqmax: float = 2.0
    freq_norm: str = "rma"  # choose between "rma" for a soft whitenning or "no" for no whitening
    #  TODO: change "no" for "None", and add "one_bit" as an option
    time_norm: str = "no"  # 'no' for no normalization, or 'rma', 'one_bit' for normalization in time domain,
    # TODO: change time_norm option from "no" to "None"
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
    rm_resp_out: str = "VEL"  # output location from response removal
    respdir: str = None  # response directory
    # some control parameters
    acorr_only: bool = False  # only perform auto-correlation
    xcorr_only: bool = True  # only perform cross-correlation or not

    # 'RESP', or 'polozeros' to remove response

    def __post_init__(self):
        self.dt = 1.0 / self.samp_freq
        assert self.substack_len % self.cc_len == 0

    # TODO: Remove once all uses of ConfigParameters have been converted to use strongly typed access
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


class ChannelData:
    """
    A 1D time series of channel data

    Attributes:
        data: series values
        sampling_rate: In HZ
        start_timestamp: Seconds since 01/01/1970
    """

    stream: obspy.Stream
    data: np.ndarray
    sampling_rate: int
    start_timestamp: float

    def empty() -> ChannelData:
        return ChannelData(obspy.Stream([obspy.Trace(np.empty(0))]))

    def __init__(self, stream: obspy.Stream):
        self.stream = stream
        self.data = stream[0].data
        self.sampling_rate = stream[0].stats.sampling_rate
        self.start_timestamp = stream[0].stats.starttime.timestamp
