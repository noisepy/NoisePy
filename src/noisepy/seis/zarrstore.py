import datetime
import glob
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import obspy
import zarr
from datetimerange import DateTimeRange

from . import noise_module
from .constants import DATE_FORMAT, DONE_PATH, PROGRESS_DATATYPE
from .datatypes import Channel, ChannelData, ChannelType, ConfigParameters, Station
from .stores import CrossCorrelationDataStore, RawDataStore

logger = logging.getLogger(__name__)


class ZarrCCStore(CrossCorrelationDataStore):
    def __init__(self, directory: str) -> None:
        super().__init__()
        self.directory = directory
        Path(directory).mkdir(exist_ok=True)

    # CrossCorrelationDataStore implementation
    def contains(
        self, timespan: DateTimeRange, src_chan: Channel, rec_chan: Channel, parameters: ConfigParameters
    ) -> bool:
        station_pair = CrossCorrelationDataStore._get_station_pair(self, src_chan, rec_chan)
        channel_pair = CrossCorrelationDataStore._get_channel_pair(self, src_chan, rec_chan)
        logger.debug(f"station pair {station_pair} channel pair {channel_pair}")
        contains = self._contains(timespan, station_pair, channel_pair)
        if contains:
            logger.info(f"Cross-correlation {station_pair} and {channel_pair} already exists")
        return contains

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
        station_pair = CrossCorrelationDataStore._get_station_pair(self, src_chan, rec_chan)
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

    def read(self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station, src_ch: ChannelType, rec_ch: ChannelType
    ) -> Tuple[Dict, np.ndarray]:
        dtype = CrossCorrelationDataStore._get_station_pair(self, src_sta, rec_sta)
        path = CrossCorrelationDataStore._get_channel_pair(self, src_ch, rec_ch)
        stream = self.datasets[timespan].auxiliary_data[dtype][path]
        return (stream.parameters, stream.data)

    # private helper methods
    def _contains(self, timespan: DateTimeRange, data_type: str, path: str = None):
        filename = self._get_filename(timespan)
        if not os.path.isfile(filename):
            return False

        store = zarr.DirectoryStore(filename)
        with zarr.open(store, mode="r") as ccf_ds:
            # source-receiver pair
            exists = data_type in ccf_ds.attrs["auxiliary_data"]
            if path is not None and exists:
                return path in ccf_ds.attrs["auxiliary_data"][data_type]
            return exists

    def _add_aux_data(self, timespan: DateTimeRange, params: Dict, data_type: str, path: str, data: np.ndarray):
        filename = self._get_filename(timespan)
        store = zarr.DirectoryStore(filename)
        with zarr.open(store, mode="a") as ccf_ds:
            ccf_ds.attrs.setdefault("auxiliary_data", {}).setdefault(data_type, {})
            ccf_ds.attrs["auxiliary_data"][data_type][path] = {"parameters": params}
            ccf_ds.create_dataset(f"auxiliary_data/{data_type}/{path}", data=data)

    def _get_filename(self, timespan: DateTimeRange) -> str:
        return os.path.join(
            self.directory,
            f"{timespan.start_datetime.strftime(DATE_FORMAT)}T{timespan.end_datetime.strftime(DATE_FORMAT)}.zarr",
        )
