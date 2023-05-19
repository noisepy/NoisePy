import datetime
import glob
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import obspy
import pyasdf
from datetimerange import DateTimeRange

from . import noise_module
from .constants import DATE_FORMAT, DONE_PATH, PROGRESS_DATATYPE
from .datatypes import Channel, ChannelData, ChannelType, ConfigParameters, Station
from .stores import CrossCorrelationDataStore, RawDataStore

logger = logging.getLogger(__name__)


class ASDFDirectory:
    """
    Utility class used byt ASDFRawDataStore and ASDFCCStore to provide easy access
    to a set of ASDF files in a directory that follow a specific naming convention
    """

    def __init__(self, directory: str, mode: str) -> None:
        self.directory = directory
        self.mode = mode

    def __getitem__(self, timespan: DateTimeRange) -> pyasdf.ASDFDataSet:
        file = _get_filename(self.directory, timespan)
        if os.path.isfile(file) or self.mode == "w":
            return _get_dataset_cached(file, self.mode)
        return None

    def get_timespans(self) -> List[DateTimeRange]:
        h5files = sorted(glob.glob(os.path.join(self.directory, "*.h5")))
        return list(map(_parse_timespans, h5files))


class ASDFRawDataStore(RawDataStore):
    """
    A data store implementation to read from a directory of ASDF files. Each file is considered
    a timespan with the naming convention: 2019_02_01_00_00_00T2019_02_02_00_00_00.h5
    """

    def __init__(self, directory: str, mode: str = "r"):
        super().__init__()
        self.datasets = ASDFDirectory(directory, mode)

    def get_channels(self, timespan: DateTimeRange) -> List[Channel]:
        ds = self.datasets[timespan]
        stations = [self._create_station(timespan, sta) for sta in ds.waveforms.list() if sta is not None]
        channels = [
            Channel(ChannelType(tag), sta) for sta in stations for tag in ds.waveforms[str(sta)].get_waveform_tags()
        ]
        return channels

    def get_timespans(self) -> List[DateTimeRange]:
        return self.datasets.get_timespans()

    def read_data(self, timespan: DateTimeRange, chan: Channel) -> np.ndarray:
        ds = self.datasets[timespan]
        stream = ds.waveforms[str(chan.station)][str(chan.type)]
        return ChannelData(stream)

    def get_inventory(self, timespan: DateTimeRange, station: Station) -> obspy.Inventory:
        ds = self.datasets[timespan]
        return ds.waveforms[str(station)]["StationXML"]

    def _create_station(self, timespan: DateTimeRange, name: str) -> Optional[Station]:
        # What should we do if there's no StationXML?
        try:
            inventory = self.datasets[timespan].waveforms[name]["StationXML"]
            sta, net, lon, lat, elv, loc = noise_module.sta_info_from_inv(inventory)
            return Station(net, sta, lat, lon, elv, loc)
        except Exception as e:
            logger.warning(f"Missing StationXML for station {name}. {e}")
            return None


class ASDFCCStore(CrossCorrelationDataStore):
    def __init__(self, directory: str, mode: str = "w") -> None:
        super().__init__()
        Path(directory).mkdir(exist_ok=True)
        self.datasets = ASDFDirectory(directory, mode)

    # CrossCorrelationDataStore implementation
    def contains(
        self, timespan: DateTimeRange, src_chan: Channel, rec_chan: Channel, parameters: ConfigParameters
    ) -> bool:
        station_pair = self._get_station_pair(src_chan.station, rec_chan.station)
        channel_pair = self._get_channel_pair(src_chan.type, rec_chan.type)
        logger.debug(f"station pair {station_pair} channel pair {channel_pair}")
        contains = self._contains(timespan, station_pair, channel_pair)
        if contains:
            logger.info(f"Cross-correlation {station_pair} and {channel_pair} already exists")
        return contains

    def save_parameters(self, parameters: ConfigParameters):
        fc_metadata = os.path.join(self.datasets.directory, "fft_cc_data.txt")

        fout = open(fc_metadata, "w")
        # WIP actually serialize this
        fout.write(str(parameters))
        fout.close()

    def append(
        self,
        timespan: DateTimeRange,
        src_chan: Channel,
        rec_chan: Channel,
        params: ConfigParameters,
        cc_params: Dict[str, Any],
        corr: np.ndarray,
    ):
        # source-receiver pair: e.g. CI.ARV_CI.BAK
        station_pair = self._get_station_pair(src_chan.station, rec_chan.station)
        # channels, e.g. bhn_bhn
        channels = self._get_channel_pair(src_chan.type, rec_chan.type)
        data = np.zeros(corr.shape, dtype=corr.dtype)
        data[:] = corr[:]
        self._add_aux_data(timespan, cc_params, station_pair, channels, data)

    def mark_done(self, timespan: DateTimeRange):
        self._add_aux_data(timespan, {}, PROGRESS_DATATYPE, DONE_PATH, np.zeros(0))

    def is_done(self, timespan: DateTimeRange):
        done = self._contains(timespan, PROGRESS_DATATYPE, DONE_PATH)
        if done:
            logger.info(f"Timespan {timespan} already computed")
        return done

    def get_timespans(self) -> List[DateTimeRange]:
        return self.datasets.get_timespans()

    def get_station_pairs(self, timespan: DateTimeRange) -> List[Tuple[Station, Station]]:
        ccf_ds = self.datasets[timespan]
        data = ccf_ds.auxiliary_data.list()
        return [_parse_station_pair(p) for p in data if p != PROGRESS_DATATYPE]

    def get_channeltype_pairs(
        self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station
    ) -> List[Tuple[Channel, Channel]]:
        ccf_ds = self.datasets[timespan]
        dtype = self._get_station_pair(src_sta, rec_sta)
        ch_pairs = ccf_ds.auxiliary_data[dtype].list()
        return [tuple(map(ChannelType, ch.split("_"))) for ch in ch_pairs]

    def read(
        self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station, src_ch: ChannelType, rec_ch: ChannelType
    ) -> Tuple[Dict, np.ndarray]:
        dtype = self._get_station_pair(src_sta, rec_sta)
        path = self._get_channel_pair(src_ch, rec_ch)
        stream = self.datasets[timespan].auxiliary_data[dtype][path]
        return (stream.parameters, stream.data)

    # private helper methods
    def _contains(self, timespan: DateTimeRange, data_type: str, path: str = None):
        ccf_ds = self.datasets[timespan]
        if not ccf_ds:
            return False
        # source-receiver pair
        exists = data_type in ccf_ds.auxiliary_data
        if path is not None and exists:
            return path in ccf_ds.auxiliary_data[data_type]
        return exists

    def _add_aux_data(self, timespan: DateTimeRange, params: Dict, data_type: str, path: str, data: np.ndarray):
        ccf_ds = self.datasets[timespan]
        ccf_ds.add_auxiliary_data(data=data, data_type=data_type, path=path, parameters=params)

    def _get_station_pair(self, src_sta: Station, rec_sta: Station) -> str:
        return f"{src_sta}_{rec_sta}"

    def _get_channel_pair(self, src_chan: ChannelType, rec_chan: ChannelType) -> str:
        return f"{src_chan.name}_{rec_chan.name}"


@lru_cache
def _get_dataset_cached(filename: str, mode: str) -> pyasdf.ASDFDataSet:
    logger.info(f"ASDFCCStore - Opening {filename}")
    return pyasdf.ASDFDataSet(filename, mode=mode, mpi=False, compression=None)


def _parse_timespans(filename: str) -> DateTimeRange:
    parts = os.path.splitext(os.path.basename(filename))[0].split("T")
    dates = [obspy.UTCDateTime(p).datetime.replace(tzinfo=datetime.timezone.utc) for p in parts]
    return DateTimeRange(dates[0], dates[1])


def _get_filename(directory: str, timespan: DateTimeRange) -> str:
    return os.path.join(
        directory,
        f"{timespan.start_datetime.strftime(DATE_FORMAT)}T{timespan.end_datetime.strftime(DATE_FORMAT)}.h5",
    )


def _parse_station_pair(pair: str) -> Tuple[Station, Station]:
    # Parse from:'CI.ARV_CI.BAK
    def station(sta: str) -> Station:
        net, name = sta.split(".")
        return Station(net, name)

    return tuple(map(station, pair.split("_")))
