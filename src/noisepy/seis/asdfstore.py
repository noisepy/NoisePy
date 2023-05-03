import datetime
import glob
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import obspy
import pyasdf
from datetimerange import DateTimeRange

from . import noise_module
from .constants import DATE_FORMAT, DONE_PATH, PROGRESS_DATATYPE
from .datatypes import Channel, ChannelData, ChannelType, ConfigParameters, Station
from .stores import CrossCorrelationDataStore, RawDataStore

logger = logging.getLogger(__name__)


class ASDFRawDataStore(RawDataStore):
    """
    A data store implementation to read from a directory of ASDF files. Each file is considered
    a timespan with the naming convention: 2019_02_01_00_00_00T2019_02_02_00_00_00.h5
    """

    def __init__(self, directory: str):
        super().__init__()
        self.directory = directory
        h5files = sorted(glob.glob(os.path.join(directory, "*.h5")))
        self.files = {str(self._parse_timespans(f)): f for f in h5files}
        logger.info(f"Initialized store with {len(self.files)}")

    def get_channels(self, timespan: DateTimeRange) -> List[Channel]:
        ds = self.get_dataset(str(timespan))
        stations = [self._create_station(timespan, sta) for sta in ds.waveforms.list() if sta is not None]
        channels = [
            Channel(ChannelType(tag), sta) for sta in stations for tag in ds.waveforms[str(sta)].get_waveform_tags()
        ]
        return channels

    def get_timespans(self) -> List[DateTimeRange]:
        return [DateTimeRange.from_range_text(d) for d in sorted(self.files.keys())]

    def read_data(self, timespan: DateTimeRange, chan: Channel) -> np.ndarray:
        ds = self.get_dataset(str(timespan))
        stream = ds.waveforms[str(chan.station)][str(chan.type)]
        return ChannelData(stream)

    def get_inventory(self, timespan: DateTimeRange, station: Station) -> obspy.Inventory:
        ds = self.get_dataset(str(timespan))
        return ds.waveforms[str(station)]["StationXML"]

    def _parse_timespans(self, filename: str) -> DateTimeRange:
        parts = os.path.splitext(os.path.basename(filename))[0].split("T")
        dates = [obspy.UTCDateTime(p).datetime.replace(tzinfo=datetime.timezone.utc) for p in parts]
        return DateTimeRange(dates[0], dates[1])

    def _create_station(self, timespan: DateTimeRange, name: str) -> Optional[Station]:
        # What should we do if there's not StationXML?
        try:
            inventory = self.get_dataset(str(timespan)).waveforms[name]["StationXML"]
            sta, net, lon, lat, elv, loc = noise_module.sta_info_from_inv(inventory)
            return Station(net, sta, lat, lon, elv, loc)
        except Exception as e:
            logger.warning(f"Missing StationXML for station {name}. {e}")
            return None

    @lru_cache
    def get_dataset(self, ts_str: str) -> pyasdf.ASDFDataSet:
        return pyasdf.ASDFDataSet(self.files[ts_str], mode="r", mpi=False)


class ASDFCCStore(CrossCorrelationDataStore):
    def __init__(self, directory: str) -> None:
        super().__init__()
        self.directory = directory
        Path(directory).mkdir(exist_ok=True)

    # CrossCorrelationDataStore implementation
    def contains(
        self, timespan: DateTimeRange, src_chan: Channel, rec_chan: Channel, parameters: ConfigParameters
    ) -> bool:
        station_pair = self._get_station_pair(src_chan, rec_chan)
        channel_pair = self._get_channel_pair(src_chan, rec_chan)
        logger.debug(f"station pair {station_pair} channel pair {channel_pair}")
        contains = self._contains(timespan, station_pair, channel_pair)
        if contains:
            logger.info(f"Cross-correlation {station_pair} and {channel_pair} already exists")
        return contains

    def save_parameters(self, parameters: ConfigParameters):
        fc_metadata = os.path.join(self.directory, "fft_cc_data.txt")

        fout = open(fc_metadata, "w")
        # WIP actually serialize this
        fout.write(str(parameters.__dict__))
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
        station_pair = self._get_station_pair(src_chan, rec_chan)
        # channels, e.g. bhn_bhn
        channels = f"{src_chan.type.name}_{rec_chan.type.name}"
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

    def read(self, chan1: Channel, chan2: Channel, start: datetime, end: datetime) -> np.ndarray:
        pass

    # private helper methods
    def _contains(self, timespan: DateTimeRange, data_type: str, path: str = None):
        filename = self._get_filename(timespan)
        if not os.path.isfile(filename):
            return False

        with pyasdf.ASDFDataSet(filename, mpi=False, mode="r") as ccf_ds:
            # source-receiver pair
            exists = data_type in ccf_ds.auxiliary_data
            if path is not None and exists:
                return path in ccf_ds.auxiliary_data[data_type]
            return exists

    def _add_aux_data(self, timespan: DateTimeRange, params: Dict, data_type: str, path: str, data: np.ndarray):
        filename = self._get_filename(timespan)
        with pyasdf.ASDFDataSet(filename, mpi=False) as ccf_ds:
            ccf_ds.add_auxiliary_data(data=data, data_type=data_type, path=path, parameters=params)

    def _get_station_pair(self, src_chan: Channel, rec_chan: Channel) -> str:
        return f"{src_chan.station}_{rec_chan.station}"

    def _get_channel_pair(self, src_chan: Channel, rec_chan: Channel) -> str:
        return f"{src_chan.type.name}_{rec_chan.type.name}"

    def _get_filename(self, timespan: DateTimeRange) -> str:
        return os.path.join(
            self.directory,
            f"{timespan.start_datetime.strftime(DATE_FORMAT)}T{timespan.end_datetime.strftime(DATE_FORMAT)}.h5",
        )
