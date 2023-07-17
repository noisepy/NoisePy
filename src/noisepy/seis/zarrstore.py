from typing import Any, Dict, List, Tuple

import numpy as np
import zarr
from datetimerange import DateTimeRange

from noisepy.seis.constants import DONE_PATH

from .datatypes import Channel, ChannelType, Station
from .stores import (
    CrossCorrelationDataStore,
    parse_station_pair,
    parse_timespan,
    timespan_str,
)


class ZarrCCStore(CrossCorrelationDataStore):
    """
    TODO:
    docs
    make chunking an argument
    update tutorials to use Zarr?
    update CLI to use Zarr?
    /                           (root)
        station_pair            (group)
            timestamp           (group)
                channel_pair    (array)
        'done'                  (array)
    """

    def __init__(self, root_dir: str, mode: str = "w") -> None:
        super().__init__()
        self.chunks = (1000, 1000)
        self.root = zarr.open(root_dir, mode=mode)

    def contains(self, timespan: DateTimeRange, src_chan: Channel, rec_chan: Channel) -> bool:
        path = self._get_path(timespan, src_chan, rec_chan)
        return path in self.root

    def append(
        self,
        timespan: DateTimeRange,
        src_chan: Channel,
        rec_chan: Channel,
        cc_params: Dict[str, Any],
        data: np.ndarray,
    ):
        path = self._get_path(timespan, src_chan, rec_chan)
        array = self.root.require_dataset(
            path,
            shape=data.shape,
            chunks=self.chunks,
            dtype=data.dtype,
        )
        array[:] = data
        for k, v in cc_params.items():
            array.attrs[k] = v

    def is_done(self, timespan: DateTimeRange):
        if DONE_PATH not in self.root.array_keys():
            return False
        done_array = self.root[DONE_PATH]
        return timespan_str(timespan) in done_array.attrs.keys()

    def mark_done(self, timespan: DateTimeRange):
        done_array = self.root.require_dataset(DONE_PATH, shape=(1, 1))
        done_array.attrs[timespan_str(timespan)] = True

    def get_timespans(self) -> List[DateTimeRange]:
        pairs = [k for k in self.root.group_keys() if k != DONE_PATH]
        timespans = []
        for p in pairs:
            timespans.extend(k for k in self.root[p].group_keys())
        return list(parse_timespan(t) for t in sorted(set(timespans)))

    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        pairs = [parse_station_pair(k) for k in self.root.group_keys() if k != DONE_PATH]
        return pairs

    def get_channeltype_pairs(
        self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station
    ) -> List[Tuple[ChannelType, ChannelType]]:
        path = self._get_station_path(timespan, src_sta, rec_sta)
        if path not in self.root:
            return []
        ch_pairs = self.root[path].array_keys()
        return [tuple(map(ChannelType, ch.split("_"))) for ch in ch_pairs]

    def read(
        self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station, src_ch: ChannelType, rec_ch: ChannelType
    ) -> Tuple[Dict, np.ndarray]:
        path = self._get_path(timespan, Channel(src_ch, src_sta), Channel(rec_ch, rec_sta))
        if path not in self.root:
            return ({}, np.empty)
        array = self.root[path]
        data = array[:]
        params = {k: array.attrs[k] for k in array.attrs.keys()}
        return (params, data)

    def _get_station_path(self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station) -> str:
        stations = self._get_station_pair(src_sta, rec_sta)
        times = timespan_str(timespan)
        return f"{stations}/{times}"

    def _get_path(self, timespan: DateTimeRange, src_chan: Channel, rec_chan: Channel) -> str:
        channels = self._get_channel_pair(src_chan.type, rec_chan.type)
        station_path = self._get_station_path(timespan, src_chan.station, rec_chan.station)
        return f"{station_path}/{channels}"
