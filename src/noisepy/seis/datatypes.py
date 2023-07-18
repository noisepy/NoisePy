from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional

import numpy as np
import obspy
from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_validators import model_validator
from pydantic_yaml import parse_yaml_raw_as, to_yaml_str

INVALID_COORD = -sys.float_info.max


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

    def __init__(
        self,
        network: str,
        name: str,
        lat: float = INVALID_COORD,
        lon: float = INVALID_COORD,
        elevation: float = INVALID_COORD,
        location: str = "",
    ):
        self.network = network
        self.name = name
        self.lat = lat
        self.lon = lon
        self.elevation = elevation
        self.location = location

    def valid(self) -> bool:
        return min(self.lat, self.lon, self.elevation) > INVALID_COORD

    def __repr__(self) -> str:
        return f"{self.network}.{self.name}"

    def __hash__(self) -> int:
        return str(self).__hash__()


class CorrelationMethod(Enum):
    XCORR = 1
    DECONV = 2


class StackMethod(Enum):
    LINEAR = "linear"
    PWS = "pws"
    ROBUST = "robust"
    AUTO_COVARIANCE = "auto_covariance"
    NROOT = "nroot"
    SELECTIVE = "selective"
    ALL = "all"


class ConfigParameters(BaseModel):
    model_config = ConfigDict(validate_default=True)

    client_url_key: str = "SCEDC"
    start_date: datetime = Field(default=datetime(2019, 1, 1))
    end_date: datetime = Field(default=datetime(2019, 1, 2))
    samp_freq: float = Field(default=20.0)  # TODO: change this samp_freq for the obspy "sampling_rate"
    cc_len: float = Field(default=1800.0, description="basic unit of data length for fft (sec)")
    # download params.
    # Targeted region/station information: only needed when down_list is False
    lamin: float = Field(default=31.0, description="Download: minimum latitude")
    lamax: float = Field(default=36.0, description="Download: maximum latitude")
    lomin: float = Field(default=-122.0, description="Download: minimum longitude")
    lomax: float = Field(default=-115.0, description="Download: maximum longitude")
    down_list: bool = Field(default=False, description="download stations from a pre-compiled list or not")
    net_list: List[str] = Field(default=["CI"], description="network list")
    stations: List[str] = Field(default=["*"], description="station list")
    channels: List[str] = Field(default=["BHE", "BHN", "BHZ"], description="channel list")
    # pre-processing parameters
    step: float = Field(default=450.0, description="overlapping between each cc_len (sec)")
    freqmin: float = Field(default=0.05)
    freqmax: float = Field(default=2.0)
    freq_norm: str = Field(
        default="rma", description="choose between 'rma' for a soft whitenning or 'no' for no whitening"
    )
    # TODO: change "no"for "None", and add "one_bit"as an option
    # TODO: change time_norm option from "no"to "None"
    time_norm: str = Field(
        default="no", description="'no' for no normalization, or 'rma', 'one_bit' for normalization in time domain,"
    )
    # FOR "COHERENCY"PLEASE set freq_norm to "rma", time_norm to "no"and cc_method to "xcorr"
    cc_method: str = Field(
        default="xcorr", description="'xcorr' for pure cross correlation, 'deconv' for deconvolution;"
    )
    smooth_N: int = Field(
        default=10, description="moving window length for time/freq domain normalization if selected (points)"
    )
    smoothspect_N: int = Field(default=10, description="moving window length to smooth spectrum amplitude (points)")
    # if substack=True, substack_len=2*cc_len, then you pre-stack every 2 correlation windows.
    # For instance: substack=True, substack_len=cc_len means that you keep ALL of the correlations"
    substack: bool = Field(
        default=True, description="True:  smaller stacks within the time chunk. False: it will stack over inc_hours"
    )
    substack_len: int = Field(
        default=1800, description="how long to stack over (for monitoring purpose): need to be multiples of cc_len"
    )
    maxlag: int = Field(default=200, description="lags of cross-correlation to save (sec)")
    substack: bool = Field(default=True)
    inc_hours: int = Field(default=24, description="Time increment size in hours")
    # criteria for data selection
    max_over_std: int = Field(
        default=10,
        description="threahold to remove window of bad signals: set it to 10*9 if prefer not to remove them",
    )
    ncomp: int = Field(default=3, description="1 or 3 component data (needed to decide whether to do rotation)")
    # station/instrument info for input_fmt=='sac' or 'mseed'
    stationxml: bool = Field(
        default=False, description="station.XML file used to remove instrument response for SAC/miniseed data"
    )
    rm_resp: str = Field(default="no", description="select 'no' to not remove response and use 'inv','spectrum',")
    rm_resp_out: str = Field(default="VEL", description="output location from response removal")
    respdir: Optional[str] = Field(default=None, description="response directory")
    # some control parameters
    acorr_only: bool = Field(default=False, description="only perform auto-correlation")
    xcorr_only: bool = Field(default=True, description="only perform cross-correlation or not")
    # Stacking parameters:
    stack_method: StackMethod = Field(default=StackMethod.LINEAR)
    keep_substack: bool = Field(default=False, description="keep all sub-stacks in final ASDF file")
    # new rotation para
    rotation: bool = Field(default=True, description="rotation from E-N-Z to R-T-Z")
    correction: bool = Field(default=False, description="angle correction due to mis-orientation")
    correction_csv: Optional[str] = Field(default=None, description="Path to e.g. meso_angles.csv")
    # 'RESP', or 'polozeros' to remove response

    @property
    def dt(self) -> float:
        return 1.0 / self.samp_freq

    @model_validator(mode="after")
    def validate(cls, m: ConfigParameters) -> ConfigParameters:
        if m.substack_len % m.cc_len != 0:
            raise ValueError(f"substack_len ({m.substack_len}) must be a multiple of cc_len ({m.cc_len})")
        return m

    # TODO: Remove once all uses of ConfigParameters have been converted to use strongly typed access
    def __getitem__(self, key):
        # Hack since pydantic model properties are nor part of the object's __dict__
        if key == "dt":
            return self.dt
        return self.__dict__[key]

    def save_yaml(self, filename: str):
        yaml_str = to_yaml_str(self)
        with open(filename, "w") as f:
            f.write(yaml_str)

    def load_yaml(filename: str) -> ConfigParameters:
        with open(filename, "r") as f:
            yaml_str = f.read()
            config = parse_yaml_raw_as(ConfigParameters, yaml_str)
            return config


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
        self.data = stream[0].data[:]
        self.sampling_rate = stream[0].stats.sampling_rate
        self.start_timestamp = stream[0].stats.starttime.timestamp


@dataclass
class NoiseFFT:
    """
    Data class to hold FFT and associated data
    """

    fft: np.ndarray
    std: np.ndarray
    fft_time: np.ndarray
    window_count: int
    length: int
