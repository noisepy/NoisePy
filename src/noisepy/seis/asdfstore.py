import datetime
import glob
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import obspy
import pyasdf
from datetimerange import DateTimeRange

from noisepy.seis import noise_module
from noisepy.seis.stores import CrossCorrelationDataStore, RawDataStore

from .datatypes import Channel, ChannelData, ChannelType, FFTParameters, Station

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
        ds = pyasdf.ASDFDataSet(self.files[str(timespan)], mode="r")
        stations = [self._create_station(ds, sta) for sta in ds.waveforms.list() if sta is not None]
        channels = [
            Channel(ChannelType(tag), sta) for sta in stations for tag in ds.waveforms[str(sta)].get_waveform_tags()
        ]
        return channels

    def get_timespans(self) -> List[DateTimeRange]:
        return [DateTimeRange.from_range_text(d) for d in sorted(self.files.keys())]

    def read_data(self, timespan: DateTimeRange, chan: Channel) -> np.ndarray:
        ds = pyasdf.ASDFDataSet(self.files[str(timespan)], mode="r")
        stream = ds.waveforms[str(chan.station)][chan.type.name][0]
        return ChannelData(stream.data, stream.stats.sampling_rate, stream.stats.starttime.timestamp)

    def _parse_timespans(self, filename: str) -> DateTimeRange:
        parts = os.path.splitext(os.path.basename(filename))[0].split("T")
        dates = [obspy.UTCDateTime(p).datetime.replace(tzinfo=datetime.timezone.utc) for p in parts]
        return DateTimeRange(dates[0], dates[1])

    def _create_station(self, ds: pyasdf.ASDFDataSet, name: str) -> Optional[Station]:
        # What should we do if there's not StationXML?
        try:
            inventory = ds.waveforms[name]["StationXML"]
            sta, net, lon, lat, elv, loc = noise_module.sta_info_from_inv(inventory)
            return Station(net, sta, lat, lon, elv, loc)
        except Exception as e:
            logger.warning(f"Missing StationXML for station {name}. {e}")
            return None


class ASDFCCStore(CrossCorrelationDataStore):
    def __init__(self, directory: str) -> None:
        super().__init__()
        self.directory = directory

    def contains(self, timespan: DateTimeRange, chan1: Channel, chan2: Channel, parameters: FFTParameters) -> bool:
        pass

    def save_parameters(self, parameters: FFTParameters):
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
        params: FFTParameters,
        cc_params: Dict[str, Any],
        corr: np.ndarray,
    ):
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        filename = os.path.join(self.directory, f"{timespan}.h5")
        with pyasdf.ASDFDataSet(filename, mpi=False) as ccf_ds:
            # source-receiver pair
            data_type = f"{src_chan.station}_{rec_chan.station}"
            path = src_chan.type.get_basename() + "_" + rec_chan.type.get_basename()
            data = np.zeros(corr.shape, dtype=corr.dtype)
            data[:] = corr[:]
            ccf_ds.add_auxiliary_data(data=data, data_type=data_type, path=path, parameters=cc_params)

    def read(self, chan1: Channel, chan2: Channel, start: datetime, end: datetime) -> np.ndarray:
        pass
