"""
The obspy-free pre-processing path must produce EXACTLY what the obspy path
produces — same samples, same dtype, same start time.

``preprocess_raw_np`` replaces ``preprocess_raw`` for ``rm_resp=NO``, which is
the default configuration for cloud runs. "Close enough" is not the bar: these
tests assert bit-identity (``np.array_equal``), because any drift here silently
changes published cross-correlations.

The fixtures matter. ``CI.CCA``/``CI.HEC`` 2022-002 day files start at
00:00:00.0195, i.e. 0.218 samples off the requested window at 40 Hz, and their
start time is not a whole microsecond. That exercises:
  - the nearest-sample trim/pad at the end of the chain,
  - the Fourier resample branch (40 -> 20 and 40 -> 1 Hz),
  - the sub-sample ``segment_interpolate`` alignment inside it, which reads
    ``fric`` off a microsecond-rounded start time.
"""

import os

import numpy as np
import obspy
import pytest

from noisepy.seis import noise_module
from noisepy.seis.io.datatypes import ChannelData, ConfigParameters, RmResp
from noisepy.seis.io.s3store import SCEDCS3DataStore


class MockCatalog:
    def get_inventory(self, timespan, station):
        return obspy.Inventory()

    def get_full_channel(self, timespan, channel):
        return channel


def _config(sampling_rate: float) -> ConfigParameters:
    config = ConfigParameters()
    config.rm_resp = RmResp.NO
    config.sampling_rate = sampling_rate
    config.freqmin = 0.05
    config.freqmax = 0.4 * sampling_rate
    return config


def _read(dpath: str):
    path = os.path.join(os.path.dirname(__file__), dpath)
    store = SCEDCS3DataStore(path, MockCatalog())
    timespan = store.get_timespans()[0]
    return store, timespan, store.get_channels(timespan)


@pytest.mark.parametrize("dpath", ["./data/cc", "./data/acc"])
# 40 Hz is the native rate (no resample branch); 20 and 1 Hz resample, and the
# fixtures' sub-second start makes the sub-sample alignment fire
@pytest.mark.parametrize("sampling_rate", [40.0, 20.0, 1.0])
def test_numpy_path_is_bit_identical_to_obspy_path(dpath: str, sampling_rate: float):
    store, timespan, channels = _read(dpath)
    config = _config(sampling_rate)

    for chan in channels:
        ch_data = store.read_data(timespan, chan)

        expected = noise_module.preprocess_raw(
            ch_data.stream.copy(),
            obspy.Inventory(),
            config,
            obspy.UTCDateTime(timespan.start_datetime),
            obspy.UTCDateTime(timespan.end_datetime),
        )
        actual = noise_module.preprocess_raw_np(
            ch_data,
            config,
            timespan.start_datetime.timestamp(),
            timespan.end_datetime.timestamp(),
        )

        assert actual.data.dtype == expected[0].data.dtype
        assert actual.sampling_rate == expected[0].stats.sampling_rate
        assert actual.start_timestamp == expected[0].stats.starttime.timestamp
        np.testing.assert_array_equal(actual.data, expected[0].data)


def test_window_is_not_sample_aligned():
    """Guard the guard: if the fixture ever became aligned, the test above
    would stop covering the case the alignment logic exists for."""
    store, timespan, channels = _read("./data/cc")
    ch_data = store.read_data(timespan, channels[0])
    dt = 1.0 / ch_data.sampling_rate
    offset_samples = ((timespan.start_datetime.timestamp() - ch_data.start_timestamp) % dt) / dt
    assert 0.01 < offset_samples < 0.99


def test_empty_channel_data_returns_empty():
    config = _config(20.0)
    out = noise_module.preprocess_raw_np(ChannelData.empty(), config, 0.0, 86400.0)
    assert out.data.size == 0


def test_taper_matches_obspy_hann_entry_point():
    """noise_module.taper used obspy's ('taper', 'hann') entry point; it now
    calls scipy directly. Assert the two windows are the same function."""
    from obspy.core.util.base import _get_function_from_entry_point
    from scipy.signal.windows import hann

    obspy_hann = _get_function_from_entry_point("taper", "hann")
    for n in (21, 40, 41, 4001):
        np.testing.assert_array_equal(hann(n), obspy_hann(n))


def test_cc_parameters_distance_matches_obspy():
    """cc_parameters writes dist/azi/baz into every saved CCF."""
    from obspy.geodetics.base import gps2dist_azimuth
    from seisfetch.contrib.obspy_ports import gps2dist_azimuth_np

    pairs = [
        (34.1, -118.1, 37.9, -122.3),
        (33.61, -116.46, 35.82, -120.34),
        (45.0, 45.0, 45.0, 45.0),
    ]
    for lat1, lon1, lat2, lon2 in pairs:
        assert gps2dist_azimuth_np(lat1, lon1, lat2, lon2) == gps2dist_azimuth(lat1, lon1, lat2, lon2)
