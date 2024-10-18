import json
import logging
import typing
from collections import defaultdict
from io import IOBase  # , StringIO
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Tuple, Union
from urllib.parse import urlparse

import fsspec
import numpy as np
import pydantic
from pydantic import BaseModel, ConfigDict, Field
from pydantic_yaml import to_yaml_str
from ruamel.yaml import YAML

from noisepy.seis.io.datatypes import ConfigParameters

S3_SCHEME = "s3"
HTTPS_SCHEME = "https"
logger = logging.getLogger(__name__)


class ConfigParameters_monitoring(BaseModel):
    model_config = ConfigDict(validate_default=True)

    # pre-defined group velocity to window direct and code waves
    vmin: float = Field(default=2.0, description="minimum velocity of the direct waves -> start of the coda window")
    lwin: float = Field(default=20, description="window length in sec for the coda waves")

    # basic parameters
    freq: List[float] = Field(
        default=[0.2, 0.4, 0.8, 2.0], description="targeted frequency band for waveform monitoring"
    )
    onelag: bool = Field(default=False, description="make measurement one one lag or two")
    norm_flag: bool = Field(default=True, description="whether to normalize the cross-correlation waveforms")
    do_stretch: bool = Field(default=True, description="use strecthing method or not")

    # parameters for stretching method
    epsilon: float = Field(default=0.02, description="limit for dv/v (in decimal)")
    nbtrial: int = Field(default=50, description="number of increment of dt [-epsilon,epsilon] for the streching")

    # parameters of coda window
    coda_tbeg: float = Field(
        default=2.0, description="begin time (sec) of the coda window for measuring velocity changes"
    )
    coda_tend: float = Field(
        default=8.0, description="end time (sec) of the coda window for measuring velocity changes"
    )
    ratio: float = Field(default=3.0, description="ratio for determining noise level by Mean absolute deviation (MAD)")

    # --- paramters for measuring attenuation
    single_station: bool = Field(default=True, description="make measurement on single statoin or multiple stations")
    smooth_winlen: float = Field(
        default=5.0, description="smoothing window length of the envelope waveforms for measuring attenuation"
    )
    cvel: float = Field(default=2.6, description="Rayleigh wave velocities over the freqency bands")
    atten_tbeg: float = Field(default=2.0, description="begin time (sec) of the coda window for measuring attenuation")
    atten_tend: float = Field(default=8.0, description="end time (sec) of the coda window for measuring attenuation")
    intb_interval_base: float = Field(
        default=0.01, description="interval base of intrinsic absorption parameter for a grid-searching process"
    )
    mfp_interval_base: float = Field(
        default=0.2, description="interval base of mean free path for a grid-searching process"
    )

    storage_options: DefaultDict[str, typing.MutableMapping] = Field(
        default=defaultdict(dict),
        description="Storage options to pass to fsspec, keyed by protocol (local files are ''))",
    )

    def get_storage_options(self, path: str) -> Dict[str, Any]:
        """The storage options for the given path"""
        url = urlparse(path)
        return self.storage_options.get(url.scheme, {})

    def save_yaml(self, filename: str):
        yaml_str = to_yaml_str(self)
        fs = get_filesystem(filename, storage_options=self.storage_options)
        with fs.open(filename, "w") as f:
            f.write(yaml_str)


#    def load_yaml(filename: str, storage_options={}) -> ConfigParameters_monitoring:
#        fs = get_filesystem(filename, storage_options=storage_options)
#        with fs.open(filename, "r") as f:
#            yaml_str = f.read()
#            config = parse_yaml_raw_as(ConfigParameters_monitoring, yaml_str)
#            return config


def _chk_model(model: Any) -> BaseModel:
    """Ensure the model passed is a Pydantic model."""
    if isinstance(model, BaseModel):
        return model
    raise TypeError(f"We can currently only write `pydantic.BaseModel`, but recieved: {model!r}")


def _write_yaml_model(stream: IOBase, model: BaseModel, **kwargs) -> None:
    """Write YAML model to the stream object.

    This uses JSON dumping as an intermediary.

    Parameters
    ----------
    stream : IOBase
        The stream to write to.
    model : BaseModel
        The model to convert.
    kwargs : Any
        Keyword arguments to pass `model.json()`. FIXME: Add explicit arguments.
    """
    model = _chk_model(model)
    if pydantic.version.VERSION < "2":
        json_val = model.json(**kwargs)  # type: ignore
    else:
        json_val = model.model_dump_json(**kwargs)  # type: ignore
    val = json.loads(json_val)
    writer = YAML(typ="safe", pure=True)
    # TODO: Configure writer
    # writer.default_flow_style = True or False or smth like that
    # writer.indent(...) for example
    writer.dump(val, stream)


# def to_yaml_str(model: BaseModel, **kwargs) -> str:
#    """Generate a YAML string representation of the model.
#
#    Parameters
#    ----------
#    model : BaseModel
#        The model to convert.
#    kwargs : Any
#        Keyword arguments to pass `model.json()`. FIXME: Add explicit arguments.
#
#    Notes
#    -----
#    This currently uses JSON dumping as an intermediary.
#    This means that you can use `json_encoders` in your model.
#    """
#    model = _chk_model(model)
#    stream = StringIO()
#    _write_yaml_model(stream, model, **kwargs)
#    stream.seek(0)
#    return stream.read()


def to_yaml_file(file: Union[Path, str, IOBase], model: BaseModel, **kwargs) -> None:
    """Write a YAML file representation of the model.

    Parameters
    ----------
    file : Path or str or IOBase
        The file path or stream to write to.
    model : BaseModel
        The model to convert.
    kwargs : Any
        Keyword arguments to pass `model.json()`. FIXME: Add explicit arguments.

    Notes
    -----
    This currently uses JSON dumping as an intermediary.
    This means that you can use `json_encoders` in your model.
    """
    model = _chk_model(model)
    if isinstance(file, IOBase):
        _write_yaml_model(file, model, **kwargs)
        return

    if isinstance(file, str):
        file = Path(file).resolve()
    elif isinstance(file, Path):
        file = file.resolve()
    else:
        raise TypeError(f"Expected Path, str, or stream, but got {file!r}")

    with file.open(mode="w") as f:
        _write_yaml_model(f, model, **kwargs)


def get_filesystem(path: str, storage_options: dict = {}) -> fsspec.AbstractFileSystem:
    """Construct an fsspec filesystem for the given path"""
    url = urlparse(path)
    # The storage_options coming from the ConfigParameters is keyed by protocol
    storage_options = storage_options.get(url.scheme, storage_options)
    if url.scheme == S3_SCHEME:
        return fsspec.filesystem(url.scheme, **storage_options)
    elif url.scheme == HTTPS_SCHEME:
        return fsspec.filesystem(url.scheme)
    else:
        return fsspec.filesystem("file", **storage_options)


def calc_segments(fft_params: ConfigParameters, num_chunk: int, MAX_MEM: int) -> Tuple[int, int]:
    if fft_params.substack:  # things are difference when do substack
        if fft_params.substack_len == fft_params.cc_len:
            num_segmts = int(np.floor((fft_params.inc_hours * 3600 - fft_params.cc_len) / fft_params.step))
        else:
            num_segmts = int(fft_params.inc_hours / (fft_params.substack_len / 3600))
    npts_segmt = int(2 * fft_params.maxlag * fft_params.sampling_rate) + 1
    memory_size = num_chunk * num_segmts * npts_segmt * 4 / 1024**3

    if memory_size > MAX_MEM:
        raise ValueError(
            "Require %5.3fG memory but only %5.3fG provided)! Cannot load cc data all once!" % (memory_size, MAX_MEM)
        )
    logger.debug("Good on memory (need %5.2f G and %s G provided)!" % (memory_size, MAX_MEM))
    return num_segmts, npts_segmt


def window_indx_def(npts_one_segmt: int, t1: float, t2: float, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    n2 = npts_one_segmt // 2  # halfway in the cross correlation functions
    pwin_indx = n2 + np.arange(int(t1 / dt), int(t2 / dt))  # coda indexes in the positive lags
    nwin_indx = n2 - np.arange(int(t1 / dt), int(t2 / dt))[::-1]  # coda indexes in the negative lags
    nwin_indx = nwin_indx[::-1]  # flip time axes
    return pwin_indx, nwin_indx
