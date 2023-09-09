import time

import numpy as np

from noisepy.seis.noise_module import nroot_stack, pws, robust_stack

"""
check the performance of all different stacking method for noise cross-correlations.

Chengxin Jiang @Harvard Oct/29/2019
Updated @May/2020 to include nth-root stacking, selective stacking and tf-PWS stacking
modified in 9/2023 to turn this into a test. Tests are bad, they just check for the dimensions.
"""


def test_pws():
    np.random.seed(12)
    ndata = np.ones((3, 3))
    dt = 1

    stack = pws(ndata, int(1 / dt))

    assert np.all(stack == np.mean(ndata, axis=-1))
    # assert len(stack) == np.max(ndata.shape)
    # assert dvv + 0.5 < para["dt"]  # assert result is -0.5%


def test_robust_stack():
    np.random.seed(12)
    ndata = np.ones((3, 3))

    stack = robust_stack(ndata, 0.001)

    # assert np.all(stack == np.mean(ndata, axis=-1))
    assert len(stack) == ndata.shape[0]


def test_nroot_stack():
    np.random.seed(12)
    ndata = np.ones((3, 3))

    stack = nroot_stack(ndata, 1)

    assert np.all(stack == np.mean(ndata, axis=-1))
    # assert len(stack) == ndata.shape[0]


# srobust, ww, nstep = noise_module.robust_stack(ndata, 0.001)
# sACF = noise_module.adaptive_filter(ndata, 1)
# nroot = noise_module.nroot_stack(ndata, 2)
# sstack, nstep = noise_module.selective_stack(ndata, 0.001, 0.01)

if __name__ == "__main__":
    print("Running stretching...")
    t = time.time()
    for i in range(100):
        test_pws()
    print("Done stretching, no errors, %4.2fs." % (time.time() - t))

    print("Running robust stacking...")
    t = time.time()
    for i in range(100):
        test_robust_stack()
    print("Done stretching, no errors, %4.2fs." % (time.time() - t))

    print("Running robust stacking...")
    t = time.time()
    for i in range(100):
        test_nroot_stack()
    print("Done stretching, no errors, %4.2fs." % (time.time() - t))
