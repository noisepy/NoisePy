import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import zarr
from datetimerange import DateTimeRange

from noisepy.seis.constants import DONE_PATH
from noisepy.seis.utils import TimeLogger, remove_nan_rows, unstack

from .datatypes import Channel, ChannelType, CrossCorrelation, Station
from .stores import (
    CrossCorrelationDataStore,
    parse_station_pair,
    parse_timespan,
    timespan_str,
)

CHANNELS_ATTR = "channels"
VERSION_ATTR = "version"

logger = logging.getLogger(__name__)


class ZarrStoreHelper:
    """
    Helper object for storing data and parameters into Zarr and track a "done" bit for subsets of the data
    'done' is a dummy array to track completed sets in its attribute dictionary.
    Args:
        root_dir: Storage location, can be a local or S3 path
        mode: "r" or "a" for read-only or writing mode
        storage_options: options to pass to fsspec
    """

    def __init__(self, root_dir: str, mode: str, storage_options={}) -> None:
        super().__init__()
        logger.info(f"store creating at {root_dir}, mode={mode}, storage_options={storage_options}")
        self.root = zarr.open_group(root_dir, mode=mode, storage_options=storage_options)
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
        array.attrs.update(params)

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

    def read(self, path: str) -> Optional[zarr.Array]:
        if path not in self.root:
            return None
        array = self.root[path]
        return array


class ZarrCCStore(CrossCorrelationDataStore):
    """
    CrossCorrelationDataStore that uses hierarchical Zarr files for storage. The directory organization is as follows:
    /                           (root)
        station_pair            (group)
            timestamp           (array)

        The 'timestamp' arrays contain the cross-correlation data for all channels. The channel information is stored
        as part of the array attributes.
    """

    def __init__(self, root_dir: str, mode: str = "a", storage_options={}) -> None:
        super().__init__()
        self.helper = ZarrStoreHelper(root_dir, mode, storage_options=storage_options)

    def contains(self, timespan: DateTimeRange, src_chan: Channel, rec_chan: Channel) -> bool:
        path = self._get_station_path(timespan, src_chan.station, rec_chan.station)
        array = self.helper.read(path)
        if not array:
            return False
        tuples = [(src, rec) for src, rec, _ in self._read_cc_metadata(array)]
        return (src_chan.type, rec_chan.type) in tuples

    def append(
        self,
        timespan: DateTimeRange,
        src: Station,
        rec: Station,
        ccs: List[CrossCorrelation],
    ):
        path = self._get_station_path(timespan, src, rec)
        # Some channels may have different lengths, so pad them with NaNs for stacking
        max_rows = max(d.data.shape[0] for d in ccs)
        max_cols = max(d.data.shape[1] for d in ccs)
        padded = [
            np.pad(
                d.data,
                ((0, max_rows - d.data.shape[0]), (0, max_cols - d.data.shape[1])),
                mode="constant",
                constant_values=np.nan,
            )
            for d in ccs
        ]
        all_ccs = np.stack(padded, axis=0)
        json_params = [(p.src.name, p.src.location, p.rec.name, p.rec.location, p.parameters) for p in ccs]
        tlog = TimeLogger(logger, logging.DEBUG)
        self.helper.append(path, {CHANNELS_ATTR: json_params, VERSION_ATTR: 1.0}, all_ccs)
        tlog.log(f"writing {len(ccs)} CCs to {path}")

    def is_done(self, timespan: DateTimeRange):
        return self.helper.is_done(timespan_str(timespan))

    def mark_done(self, timespan: DateTimeRange):
        self.helper.mark_done(timespan_str(timespan))

    def get_timespans(self) -> List[DateTimeRange]:
        pairs = [k for k in self.helper.root.group_keys() if k != DONE_PATH]
        timespans = []
        for p in pairs:
            timespans.extend(k for k in self.helper.root[p].array_keys())
        return list(parse_timespan(t) for t in sorted(set(timespans)))

    def get_station_pairs(self) -> List[Tuple[Station, Station]]:
        return self.helper.get_station_pairs()

    def read_correlations(self, timespan: DateTimeRange, src_sta: Station, rec_sta: Station) -> List[CrossCorrelation]:
        path = self._get_station_path(timespan, src_sta, rec_sta)

        array = self.helper.read(path)
        if not array:
            return []

        channel_params = self._read_cc_metadata(array)
        cc_stack = array[:]
        ccs = unstack(cc_stack)

        return [
            CrossCorrelation(src, rec, params, remove_nan_rows(data))
            for (src, rec, params), data in zip(channel_params, ccs)
        ]

    def _read_cc_metadata(self, array):
        channels_dict = array.attrs[CHANNELS_ATTR]
        channel_params = [
            (ChannelType(src, src_loc), ChannelType(rec, rec_loc), params)
            for src, src_loc, rec, rec_loc, params in channels_dict
        ]

        return channel_params

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

    def __init__(self, root_dir: str, mode: str = "a", storage_options={}) -> None:
        super().__init__()
        self.helper = ZarrStoreHelper(root_dir, mode, storage_options=storage_options)

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
        array = self.helper.read(path)
        if not array:
            return {}, np.zeros((0,))
        return array.attrs, array[:]

    def _get_children(self, path: str) -> List[str]:
        if path not in self.helper.root:
            return []
        return list(self.helper.root[path].array_keys()) + list(self.helper.root[path].group_keys())

    def _get_station_path(self, src: Station, rec: Station) -> str:
        return f"{src}_{rec}"

    def _get_path(self, src: Station, rec: Station, component: str, name: str) -> str:
        return f"{self._get_station_path(src,rec)}/{name}/{component}"
