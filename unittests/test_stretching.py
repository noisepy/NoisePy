import time

import numpy as np
import pytest
from obspy.signal.invsim import cosine_taper

from noisepy.seis.noise_module import stretching, stretching_vect

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


if __name__ == "__main__":
    print("Running stretching...")
    t = time.time()
    for i in range(300):
        test_stretching()
    print("Done stretching, no errors, %4.2fs." % (time.time() - t))

    print("Running stretching using numpy...")
    t = time.time()
    for i in range(300):
        test_stretching_vect()
    print("Done stretching, no errors, %4.2fs." % (time.time() - t))
