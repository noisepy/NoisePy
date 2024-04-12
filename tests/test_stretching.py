import time

import numpy as np
import pytest
from obspy.signal.invsim import cosine_taper

from noisepy.monitoring.monitoring_methods import (
    mwcs_dvv,
    stretching,
    stretching_vect,
    wcc_dvv,
)

# This short script is intended as a test for the stretching routine
# it takes a generic sine curve with known stretching factor and ensures
# that the stretching routines in noise_module always recover this factor
# Note: The script has to be called from src/ directory, like
# (in directory noisepy/src:)
# python ../test/test_routines/test_stretching.py


def test_stretching():
    t = np.linspace(0.0, 9.95, 2500)  # 0.5 % perturbation
    original_signal = np.sin(t * 10.0) * cosine_taper(2500, p=0.75)

    t_stretch = np.linspace(0.0, 10.0, 2500)
    stretched_signal = np.interp(t, t_stretch, original_signal)

    para = {}
    para["dt"] = 1.0 / 250.0
    para["twin"] = [0.0, 10.0]
    para["freq"] = [9.9, 10.1]

    dvv, error, cc, cdp = stretching(ref=original_signal, cur=stretched_signal, dv_range=0.05, nbtrial=100, para=para)

    assert pytest.approx(cc) == 1.0
    assert dvv + 0.5 < para["dt"]  # assert result is -0.5%


def test_stretching_vect():
    t = np.linspace(0.0, 9.95, 2500)  # 0.5 % perturbation
    original_signal = np.sin(t * 10.0) * cosine_taper(2500, p=0.75)

    t_stretch = np.linspace(0.0, 10.0, 2500)
    stretched_signal = np.interp(t, t_stretch, original_signal)

    para = {}
    para["dt"] = 1.0 / 250.0
    para["twin"] = [0.0, 10.0]
    para["freq"] = [9.9, 10.1]

    dvv, error, cc, cdp = stretching_vect(
        ref=original_signal, cur=stretched_signal, dv_range=0.05, nbtrial=100, para=para
    )

    assert pytest.approx(cc) == 1.0
    assert dvv + 0.5 < para["dt"]  # assert result is -0.5%


def test_wcc_dvv():
    t = np.linspace(0.0, 9.95, 2500)  # 0.5 % perturbation
    original_signal = np.sin(t * 10.0) * cosine_taper(2500, p=0.75)

    t_stretch = np.linspace(0.0, 10.0, 2500)
    stretched_signal = np.interp(t, t_stretch, original_signal)

    para = {}
    para["dt"] = 1.0 / 250.0
    para["twin"] = [0.0, 10.0]
    para["freq"] = [9.9, 10.1]

    mwl = 2.0  # moving window length in sec
    ss = 1.0  # sliding step in seconds
    # ref, cur, moving_window_length, slide_step, para
    dvv, error = wcc_dvv(ref=original_signal, cur=stretched_signal, moving_window_length=mwl, slide_step=ss, para=para)

    assert abs(dvv + 0.5) < 20 * para["dt"]  # assert result is -0.5%
    # WCC is extremely low accuracy


def test_mwcs_dvv():
    t = np.linspace(0.0, 9.95, 2500)  # 0.5 % perturbation
    original_signal = np.sin(t * 10.0) * cosine_taper(2500, p=0.75)

    t_stretch = np.linspace(0.0, 10.0, 2500)
    stretched_signal = np.interp(t, t_stretch, original_signal)

    para = {}
    para["dt"] = 1.0 / 250.0
    para["twin"] = [0.0, 10.0]
    para["freq"] = [0.2, 10.0]

    mwl = 5.0  # moving window length in sec
    ss = 0.5  # sliding step in seconds
    # ref, cur, moving_window_length, slide_step, para
    # ref, cur, moving_window_length, slide_step, para, smoothing_half_win=5
    dvv, error = mwcs_dvv(
        ref=original_signal,
        cur=stretched_signal,
        moving_window_length=mwl,
        slide_step=ss,
        para=para,
        smoothing_half_win=5,
    )

    assert abs(dvv + 0.5) < 0.2  # assert result is -0.5%


# wts_allfreq?
# wxs_allfreq?
# WCC_dvv
# mwcs_dvv
# dtw_dvv
# stretching_vect: faster version of stretching?


if __name__ == "__main__":
    print("Running stretching...")
    t = time.time()
    for i in range(100):
        test_stretching()
    print("Done stretching, no errors, %4.2fs." % (time.time() - t))

    print("Running stretching using numpy...")
    t = time.time()
    for i in range(100):
        test_stretching_vect()
    print("Done stretching, no errors, %4.2fs." % (time.time() - t))

    t = time.time()
    for i in range(100):
        test_mwcs_dvv()
    print("Done stretching, no errors, %4.2fs." % (time.time() - t))

    t = time.time()
    for i in range(100):
        test_wcc_dvv()
    print("Done stretching, no errors, %4.2fs." % (time.time() - t))
