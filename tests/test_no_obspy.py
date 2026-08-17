"""
obspy is an optional extra, not a requirement.

These tests run in a subprocess with obspy (and pyasdf, which pulls obspy in)
made unimportable, then exercise the default cloud configuration: read raw
miniSEED through seisfetch, pre-process at rm_resp=NO, FFT, cross-correlate.
If anything on that path grows an obspy import again, these fail.

Blocking the module — rather than checking ``sys.modules`` afterwards — is
deliberate: a lazy ``import obspy`` inside a rarely-taken branch would pass a
"was it imported?" check while still making obspy a hard requirement for the
user who takes that branch.
"""

import subprocess
import sys
import textwrap

BLOCKER = """
import sys

class _Blocked:
    def find_module(self, name, path=None):
        if name.split(".")[0] in ("obspy", "pyasdf"):
            raise ImportError(f"{name} is blocked in this test")
        return None

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("obspy", "pyasdf"):
            raise ImportError(f"{name} is blocked in this test")
        return None

sys.meta_path.insert(0, _Blocked())
"""


def _run(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", BLOCKER + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_obspy_is_actually_blocked():
    """Sanity check on the blocker itself, so a green suite means something."""
    result = _run(
        """
        try:
            import obspy
        except ImportError:
            print("BLOCKED")
        else:
            print("NOT BLOCKED")
        """
    )
    assert "BLOCKED" in result.stdout, result.stdout + result.stderr
    assert "NOT BLOCKED" not in result.stdout


def test_import_noisepy_without_obspy():
    result = _run(
        """
        import noisepy.seis
        from noisepy.seis import cross_correlate, stack_cross_correlations
        from noisepy.seis import noise_module
        from noisepy.seis.io.datatypes import ChannelData, ConfigParameters
        from noisepy.seis.io.seisfetchstore import SeisfetchRawDataStore
        from noisepy.seis.io.numpystore import NumpyCCStore
        print("OK")
        """
    )
    assert "OK" in result.stdout, result.stdout + result.stderr


def test_preprocess_and_correlate_without_obspy():
    """The full rm_resp=NO compute path on synthetic array-backed data."""
    result = _run(
        """
        import numpy as np
        from datetimerange import DateTimeRange
        from datetime import datetime, timezone

        from noisepy.seis import noise_module
        from noisepy.seis.correlate import compute_fft, cross_corr
        from noisepy.seis.io.datatypes import (
            Channel, ChannelData, ChannelType, ConfigParameters, RmResp, Station, TraceSegment,
        )

        sr = 40.0
        npts = int(sr * 3600)
        rng = np.random.default_rng(0)
        t0 = datetime(2022, 1, 2, tzinfo=timezone.utc).timestamp()
        ts = DateTimeRange(
            datetime(2022, 1, 2, tzinfo=timezone.utc),
            datetime(2022, 1, 2, 1, tzinfo=timezone.utc),
        )

        config = ConfigParameters()
        config.rm_resp = RmResp.NO
        config.sampling_rate = 20.0
        config.freqmin = 0.05
        config.freqmax = 2.0
        config.cc_len = 1800
        config.step = 900
        config.inc_hours = 1

        chans = []
        for name in ("AAA", "BBB"):
            data = rng.standard_normal(npts).astype(np.float32) * 1000
            # a sub-second, non-whole-microsecond start, like a real day file
            cd = ChannelData.from_segments(
                [TraceSegment(data=data, sampling_rate=sr, start_timestamp=t0 + 0.019537920, id=f"CI.{name}..BHZ")]
            )
            pre = noise_module.preprocess_raw_np(cd, config, ts.start_datetime.timestamp(), ts.end_datetime.timestamp())
            assert pre.data.size > 0, "pre-processing produced nothing"
            chans.append((Channel(ChannelType("BHZ"), Station("CI", name, 34.0, -118.0, 100.0, "")), pre))

        ffts = [compute_fft(config, cd) for _, cd in chans]
        assert all(f.window_count > 0 for f in ffts), "no FFT windows"

        src_fft = ffts[0]
        sfft1 = np.conj(src_fft.fft).reshape(src_fft.window_count, src_fft.length // 2)
        sou_ind = np.where((src_fft.std < config.max_over_std) & (src_fft.std > 0))[0]
        result = cross_corr(
            config, chans[0][0], chans[1][0], sfft1, sou_ind, ffts[1], src_fft.length, is_autocorr=False
        )
        assert result is not None, "cross-correlation returned nothing"
        _, _, parameters, corr = result
        assert corr.size > 0
        # dist/azi/baz come from the geodetics port, not obspy
        assert parameters["dist"] >= 0
        print("OK")
        """
    )
    assert "OK" in result.stdout, result.stdout + result.stderr
