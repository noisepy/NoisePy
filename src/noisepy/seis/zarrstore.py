import logging
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

logger = logging.getLogger(__name__)


class ZarrStoreHelper:
    """
    Helper object for storing data and parameters into Zarr and track a "done" bit for subsets of the data
    'done' is a dummy array to track completed sets in its attribute dictionary.
    Args:
        root_dir: Storage location
        mode: "r" or "w" for read-only or writing mode
        dims: number dimensions of the data

    """

    def __init__(self, root_dir: str, mode: str, dims: int) -> None:
        super().__init__()
        self.dims = dims
        self.root = zarr.open(root_dir, mode=mode)
        logger.info(f"store created at {root_dir}")

    def contains(self, path: str) -> bool:
        return path in self.root

    def append(
        self,
        path: str,
        params: Dict[str, Any],
        data: np.ndarray,
    ):
        logger.debug(f"Appending to {path}: {data.shape}")
        array = self.root.create_dataset(
            path,
            data=data,
            chunks=data.shape,
            dtype=data.dtype,
        )

        to_save = {k: _to_json(v) for k, v in params.items()}
        array.attrs.update(to_save)

    def is_done(self, key: str):
        if DONE_PATH not in self.root.array_keys():
            return False
        done_array = self.root[DONE_PATH]
        return key in done_array.attrs.keys()

    def mark_done(self, key: str):
        done_array = self.root.require_dataset(DONE_PATH, shape=(1, 1))
        done_array.attrs[key] = True

    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        pairs = [parse_station_pair(k) for k in self.root.group_keys() if k != DONE_PATH]
        return pairs

    def read(self, path: str) -> Tuple[Dict, np.ndarray]:
        if path not in self.root:
            # return empty data with the same dimensions as the chunks
            return ({}, np.empty(tuple(0 for i in range(self.dims))))
        array = self.root[path]
        data = array[:]
        params = dict(array.attrs.items())
        return (params, data)


class ZarrCCStore(CrossCorrelationDataStore):
    """
    CrossCorrelationDataStore that uses hierarchical Zarr files for storage. The directory organization is as follows:
    /                           (root)
        station_pair            (group)
            timestamp           (group)
                channel_pair    (array)
    """

    def __init__(self, root_dir: str, mode: str = "a") -> None:
        super().__init__()
        self.helper = ZarrStoreHelper(root_dir, mode, dims=2)

    def contains(self, timespan: DateTimeRange, src_chan: Channel, rec_chan: Channel) -> bool:
        path = self._get_path(timespan, src_chan, rec_chan)
        return self.helper.contains(path)

    def append(
        self,
        timespan: DateTimeRange,
        src_chan: Channel,
        rec_chan: Channel,
        cc_params: Dict[str, Any],
        data: np.ndarray,
    ):
        path = self._get_path(timespan, src_chan, rec_chan)
        return self.helper.append(path, cc_params, data)

    def is_done(self, timespan: DateTimeRange):
        return self.helper.is_done(timespan_str(timespan))

    def mark_done(self, timespan: DateTimeRange):
        self.helper.mark_done(timespan_str(timespan))

    def get_timespans(self) -> List[DateTimeRange]:
        pairs = [k for k in self.helper.root.group_keys() if k != DONE_PATH]
        timespans = []
        for p in pairs:
            timespans.extend(k for k in self.helper.root[p].group_keys())
        return list(parse_timespan(t) for t in sorted(set(timespans)))

    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        return self.helper.get_station_pairs()

    def get_channeltype_pairs(
        self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station
    ) -> List[Tuple[ChannelType, ChannelType]]:
        path = self._get_station_path(timespan, src_sta, rec_sta)
        if path not in self.helper.root:
            return []
        ch_pairs = self.helper.root[path].array_keys()
        return [tuple(map(ChannelType, ch.split("_"))) for ch in ch_pairs]

    def read(
        self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station, src_ch: ChannelType, rec_ch: ChannelType
    ) -> Tuple[Dict, np.ndarray]:
        path = self._get_path(timespan, Channel(src_ch, src_sta), Channel(rec_ch, rec_sta))
        return self.helper.read(path)

    def _get_station_path(self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station) -> str:
        stations = self._get_station_pair(src_sta, rec_sta)
        times = timespan_str(timespan)
        return f"{stations}/{times}"

    def _get_path(self, timespan: DateTimeRange, src_chan: Channel, rec_chan: Channel) -> str:
        channels = self._get_channel_pair(src_chan.type, rec_chan.type)
        station_path = self._get_station_path(timespan, src_chan.station, rec_chan.station)
        return f"{station_path}/{channels}"


class ZarrStackStore:
    """
    A class for reading and writing stack data using Zarr format. Hierarchy is:
    /
        station_pair            (group)
            stack name          (group)
                component       (array)
    """

    def __init__(self, root_dir: str, mode: str = "a") -> None:
        super().__init__()
        self.helper = ZarrStoreHelper(root_dir, mode, dims=1)

    def mark_done(self, src: Station, rec: Station):
        path = self._get_station_path(src, rec)
        return self.helper.mark_done(path)

    def is_done(self, src: Station, rec: Station):
        path = self._get_station_path(src, rec)
        return self.helper.is_done(path)

    def append(
        self, src: Station, rec: Station, component: str, name: str, stack_params: Dict[str, Any], data: np.ndarray
    ):
        path = self._get_path(src, rec, component, name)
        self.helper.append(path, stack_params, data)

    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        return self.helper.get_station_pairs()

    def get_stack_names(self, src: Station, rec: Station) -> List[str]:
        path = self._get_station_path(src, rec)
        return self._get_children(path)

    def get_components(self, src: Station, rec: Station, name: str) -> List[str]:
        path = self._get_station_path(src, rec) + "/" + name
        return self._get_children(path)

    def read(self, src: Station, rec: Station, component: str, name: str) -> Tuple[Dict[str, Any], np.ndarray]:
        path = self._get_path(src, rec, component, name)
        return self.helper.read(path)

    def _get_children(self, path: str) -> List[str]:
        if path not in self.helper.root:
            return []
        return list(self.helper.root[path].array_keys()) + list(self.helper.root[path].group_keys())

    def _get_station_path(self, src: Station, rec: Station) -> str:
        return f"{src}_{rec}"

    def _get_path(self, src: Station, rec: Station, component: str, name: str) -> str:
        return f"{self._get_station_path(src,rec)}/{name}/{component}"


def _to_json(value: Any) -> Any:
    if type(value) == np.ndarray:
        return value.tolist()
    return value
