import numpy as np
import pytest

from noisepy.seis.noise_module import pws

"""
check the performance of all different stacking method for noise cross-correlations.

Chengxin Jiang @Harvard Oct/29/2019
Updated @May/2020 to include nth-root stacking, selective stacking and tf-PWS stacking
modified in 9/2023 to turn this into a test.
"""


def test_pws():
    np.random.seed(12)
    ndata = np.ones((3, 3))
    dt = 1

    spws = pws(ndata, int(1 / dt))

    assert len(spws) == ndata.shape[0]
    # assert dvv + 0.5 < para["dt"]  # assert result is -0.5%


# srobust, ww, nstep = noise_module.robust_stack(ndata, 0.001)
# sACF = noise_module.adaptive_filter(ndata, 1)
# nroot = noise_module.nroot_stack(ndata, 2)
# sstack, nstep = noise_module.selective_stack(ndata, 0.001, 0.01)
