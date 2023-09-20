import math
from typing import Tuple

import numpy as np
import pyasdf

### -----
# These scripts are aim to perform the 2-D radiative transfer equation
# for scalar waves (Shang and Gao 1988; Sato 1993),
# assuming of isotropic scattering and source radiation in infinite medium
# to calculate synthesized energy densities Esyn.
### -----


### -----
def read_pyasdf(sfile: str, ccomp: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    # useful parameters from each asdf file
    with pyasdf.ASDFDataSet(sfile, mode="r") as ds:
        alist = ds.auxiliary_data.list()
        try:
            dt = ds.auxiliary_data[alist[0]][ccomp].parameters["dt"]
            dist = ds.auxiliary_data[alist[0]][ccomp].parameters["dist"]
            print("working on %s (comp: %s) that is %5.2fkm apart. dt: %.3f " % (sfile, ccomp, dist, dt))
            # read stacked data
            sdata = ds.auxiliary_data[alist[0]][ccomp].data[:]

            # time domain variables
            npts = sdata.size
            tvec = np.arange(-npts // 2 + 1, npts // 2 + 1) * dt
            return dist, dt, tvec, sdata

        except Exception:
            print("continue! no %s component exist" % ccomp)
            return None


### -----
# Function that Calculate Mean Square
def msValue(arr: np.ndarray, n: int) -> np.ndarray:
    square = 0.0
    mean = 0.0
    square = np.sum(np.square(arr))

    # Calculate Mean
    mean = square / (float)(n)

    return mean


### -----
# Dirac delta function
def impulse(x: float) -> Tuple[float]:
    return 1 * (x == 0)


# Heaviside function (step function)
def step(x: float) -> Tuple[float]:
    return 1 * (x > 0)


# --- for single station (ignoring the first term)
# Esyn -->The 2-D radiative transfer equation for scalar waves
"""
# mean_free is the scattering mean free paths
# tm is the timepoint
# c is the Rayleigh wave velocity
# r is the distance between the source and receiver
# step is the Heaviside function
# ---
"""


def ESYN_RadiaTrans_onesta(mean_free: float, tm: float, r: float, c: float) -> Tuple[float]:
    s0 = c**2 * tm**2 - r**2
    if s0 > 0:
        # second term
        ind2 = mean_free ** (-1) * (math.sqrt(s0) - c * tm)
        a2up = math.exp(ind2)
        a2bot = 2 * math.pi * mean_free * math.sqrt(s0)
        second = (a2up / a2bot) * step(tm - r / c)
        Esyn = second

        return Esyn


# --- for inter-station
# Esyn -->The 2-D radiative transfer equation for scalar waves
"""
# mean_free is the scattering mean free paths
# tm is the timepoint
# c is the Rayleigh wave velocity
# r is the distance between the source and receiver
# step is the Heaviside function
# impulse is the Dirac delta function
# ---
"""


def ESYN_RadiaTrans_intersta(mean_free: float, tm: float, r: float, c: float) -> Tuple[float]:
    s0 = c**2 * tm**2 - r**2
    if s0 > 0:
        # first term
        a1up = math.exp(-1 * c * tm * (mean_free ** (-1)))
        a1bot = 2 * math.pi * c * r
        first = (a1up / a1bot) * impulse(tm - r / c)

        # second term
        ind2 = mean_free ** (-1) * (math.sqrt(s0) - c * tm)
        a2up = math.exp(ind2)
        a2bot = 2 * math.pi * mean_free * math.sqrt(s0)
        second = (a2up / a2bot) * step(tm - r / c)

        Esyn = first + second

        return Esyn


### -----
def convertTuple(tup: str) -> Tuple[str]:
    # initialize an empty string
    str = "".join(tup)
    return str
