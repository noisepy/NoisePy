import logging
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

logger = logging.getLogger(__name__)


### -----
def read_pyasdf(sfile: str, ccomp: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    # useful parameters from each asdf file
    with pyasdf.ASDFDataSet(sfile, mode="r") as ds:
        alist = ds.auxiliary_data.list()
        try:
            dt = ds.auxiliary_data[alist[0]][ccomp].parameters["dt"]
            dist = ds.auxiliary_data[alist[0]][ccomp].parameters["dist"]
            logger.info(f"working on {sfile} (comp: {ccomp}) that is {dist} km apart. dt: {dt}")
            # read stacked data
            sdata = ds.auxiliary_data[alist[0]][ccomp].data[:]

            # time domain variables
            npts = sdata.size
            tvec = np.arange(-npts // 2 + 1, npts // 2 + 1) * dt
            return dist, dt, tvec, sdata

        except Exception:
            logger.warning(f"continue! no {ccomp} component exist")
            return None


### -----
# Function that Calculate Mean Square
def msValue(arr: np.ndarray) -> np.ndarray:
    """
    # Mean-squared value calculation
    ----------------------------------------------
    Input:
        arr:  Input time series
    Return:
        mean: Mean-squared value
    ----------------------------------------------
    """
    square = 0.0
    mean = 0.0
    square = np.sum(np.square(arr))

    # Calculate Mean
    mean = square / (float)(len(arr))

    return mean


### -----
# Dirac delta function
def impulse(x: float) -> float:
    return 1 * (x == 0)


# Heaviside function (step function)
def step(x: float) -> float:
    return 1 * (x > 0)


# --- for single station (ignoring the first term)
def ESYN_RadiaTrans_onesta(mean_free: float, tm: float, r: float, c: float) -> float:
    """
    # Esyn of single-station case based on the 2-D radiative transfer equation for scalar waves
    ----------------------------------------------
    Parameters:
        mean_free: The scattering mean free paths
        tm:        The timepoint
        c :        The Rayleigh wave velocity
        r :        The distance between the source and receiver
    ----------------------------------------------
    step:      The Heaviside function
    ----------------------------------------------
    Return:
        Esyn:   The synthetic energy density
    ----------------------------------------------
    """
    s0 = c**2 * tm**2 - r**2
    check_s0(s0)

    const = 0.00000001
    r = r + const  # to avoid the a2bot becomes zero

    # second term
    ind2 = mean_free ** (-1) * (math.sqrt(s0) - c * tm)
    a2up = math.exp(ind2)
    a2bot = 2 * math.pi * mean_free * math.sqrt(s0)
    second = (a2up / a2bot) * step(tm - r / c)
    Esyn = second

    return Esyn


# --- for inter-station
def ESYN_RadiaTrans_intersta(mean_free: float, tm: float, r: float, c: float) -> float:
    """
    # Esyn of inter-station case based on the 2-D radiative transfer equation for scalar waves
    ----------------------------------------------
    Parameters:
        mean_free: The scattering mean free paths
        tm:        The timepoint
        c :        The Rayleigh wave velocity
        r :        The distance between the source and receiver
    ----------------------------------------------
    step:      The Heaviside function
    impulse:   The Dirac delta function
    ----------------------------------------------
    Return:
        Esyn:   The synthetic energy density
    ----------------------------------------------
    """
    s0 = c**2 * tm**2 - r**2
    check_s0(s0)

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
def convertTuple(tup: str) -> str:
    # initialize an empty string
    str = "".join(tup)
    return str


### -----
def check_s0(x: float) -> None:
    if not (x > 0):
        raise ValueError(
            f"Invalid x: {x}. Considering the 2-D radiative transfer equation, \
                it is not sensible for the case of c**2 * tm**2 - r**2  == {x} <=0. "
        )


### -----
def get_SSR(fnum: int, para) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    #  Calculate the sum of squared residuals (SSR) between Eobs and Esyn
    #  for different combination of mean free path and intrinsic absorption parameter b
    ----------------------------------------------
    Parameters:
        fb: the number of used frequency band
        dt: data sampling interval
        c:  Rayleigh wave velocity
        npts:   Total sample number
        vdist:  Inter-station distance
        mfpx:   The mean free path array
        intby:  The intrinsic absorption parameter b array
        twinbe: Window begin time and end time array
        fmsv_mean: Observed mean-squared value
    ----------------------------------------------
    Return:
        SSR_final: The sum of squared residuals between Eobs and Esyn
        mfpx:      The searching range of mean free path array
        intby:     The searching range of intrinsic absorption parameter b array
    ----------------------------------------------
    """
    fb = para["fb"]
    dt = para["dt"]
    c = para["cvel"]
    vdist = para["vdist"]
    mfpx = para["mfp"]
    intby = para["intb"]
    twinbe = para["twin"]
    npts = para["npts"]
    fmsv_mean = para["fmsv"]

    Esyn_temp = np.ndarray((len(mfpx), len(intby), npts // 2 + 1))
    Eobs_temp = np.ndarray((len(mfpx), len(intby), npts // 2 + 1))
    SSR_final = np.ndarray((len(mfpx), len(intby)))
    SSR_final[:][:] = 0.0
    for aa in range(fnum):
        r = float(vdist[aa])
        twindow = []
        twindow = range(int(twinbe[aa][fb][0]), int(twinbe[aa][fb][1]), 1)
        SSR_temppp = np.ndarray((len(mfpx), len(intby), len(twindow)))

        # grid search in combination of mean_free_path and intrinsic_b
        Esyn_temp[:][:][:] = 0.0
        Eobs_temp[:][:][:] = 0.0

        for nfree in range(len(mfpx)):
            mean_free = 0.4 + 0.2 * nfree
            mfpx[nfree] = mean_free
            for nb in range(len(intby)):
                intrinsic_b = 0.01 * (nb + 1)
                intby[nb] = intrinsic_b

                # calculate the Esyn and SSR for combination of mean_free_path and intrinsic_b
                for twn in range(npts // 2 + 1):
                    tm = dt * twn
                    Eobs_temp[nfree][nb][twn] = fmsv_mean[aa][fb + 1][twn]

                    s0 = c**2 * tm**2 - r**2
                    if s0 < 0:
                        # logger.warning(f"s0 {s0} <0")
                        continue

                    tmp = ESYN_RadiaTrans_onesta(mean_free, tm, r, c)
                    Esyn_temp[nfree][nb][twn] = tmp * math.exp(-1 * intrinsic_b * tm)
                # using scalar factor for further fitting processes --> shape matters more than amplitude

                #### specific window --> find the scaling factor in the specific window
                for tsn in range(len(twindow)):
                    tsb = int(twindow[tsn] // dt)
                    SSR_temppp[nfree][nb][tsn] = 0.0
                    SSR_temppp[nfree][nb][tsn] = math.log10(Eobs_temp[nfree][nb][tsb]) - math.log10(
                        Esyn_temp[nfree][nb][tsb]
                    )

                crap = np.mean(SSR_temppp[nfree][nb])
                Esyn_temp[nfree][nb] *= 10**crap  # scale the Esyn

                #### specific window
                #### Calculate the SSR in the specific window
                for tsn in range(len(twindow)):
                    tsb = int(twindow[tsn] // dt)
                    tse = int((twindow[tsn] + 1) // dt)
                    SSR_temp = 0.0
                    for twn in range(tsb, tse):
                        SSR_temp += (math.log10(Eobs_temp[nfree][nb][twn]) - math.log10(Esyn_temp[nfree][nb][twn])) ** 2
                SSR_final[nfree][nb] += SSR_temp
            # --- time comsuming for plotting out individual fitting curves
            # plot_fitting_curves(mean_free,y,fmsv_mean[aa][0][:],Eobs_temp[nfree],Esyn_temp[nfree],fname[aa],vdist[aa],twindow)
        # logger.info(f"mean_free: {mean_free}, intrinsic_b {intrinsic_b}, SSR:{SSR_temp}")
        SSR_final = SSR_final / (np.min(SSR_final[:][:]))

    return SSR_final, mfpx, intby


def get_optimal(fnum: int, para) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    # Getting the optimal value from the grid searching results (the SSR output from the get_SSR)
    # Return with the optimal value of mean free path, intrinsic absorption parameter
    # and the optimal fit of synthetic energy density function
    ----------------------------------------------
    Parameters:
        fb: the number of used frequency band
        dt: data sampling interval
        c:  Rayleigh wave velocity
        npts:   Total sample number
        vdist:  Inter-station distance
        mfpx:   The mean free path array
        intby:  The intrinsic absorption parameter b array
        twinbe: Window begin time and end time array
        fmsv_mean: Observed mean-squared value
        fmin:   Lower bound of measured frequency band
        fmax:   Upper bound of measured frequency band
        SSR:    The sum of squared residuals over combination of mean free path
                and intrinsic absorption parameter b
        sta_pair: Station pair
        aa:     The file number of measurment (here in this test is 0)
    ----------------------------------------------
    Return:
        result_intb: The optimal value of intrinsic absorption parameter b
        result_mfp:  The optimal value of mean free path
        Eobs:   The observed energy density function
        Esyn:   The optimal fit of synthetic energy density function
    ----------------------------------------------
    """
    fb = para["fb"]
    dt = para["dt"]
    c = para["cvel"]
    vdist = para["vdist"]
    mfpx = para["mfp"]
    intby = para["intb"]
    twinbe = para["twin"]
    npts = para["npts"]
    fmsv_mean = para["fmsv"]

    fmin = para["fmin"]
    fmax = para["fmax"]
    SSR = para["SSR"]
    sta_pair = para["sta"]
    aa = para["filenum"]
    r = float(vdist[aa])

    loc = np.where(SSR[fb].T == np.amin(SSR[fb].T))
    ymin = intby[loc[0]]
    xmin = mfpx[loc[1]]
    logger.info(f" Station Pair: {sta_pair}, frequency band {fmin}-{fmax}Hz, intrinsic_b {ymin}, mean_free: {xmin}")
    result_intb = np.take(ymin, 0)
    result_mfp = np.take(xmin, 0)

    twindow = []
    twindow = range(int(twinbe[aa][fb][0]), int(twinbe[aa][fb][1]), 1)

    Eobs = np.ndarray((npts // 2 + 1))
    Esyn = np.ndarray((npts // 2 + 1))
    temppp = np.ndarray((len(twindow)))
    for twn in range(npts // 2 + 1):
        tm = dt * twn
        s0 = c**2 * tm**2 - r**2
        if s0 <= 0:
            continue
        Eobs[twn] = fmsv_mean[aa][fb + 1][twn]
        tmp = ESYN_RadiaTrans_onesta(result_mfp, tm, r, c)
        Esyn[twn] = tmp * math.exp(-1 * result_intb * tm)

    for tsn in range(len(twindow)):
        tsb = int(twindow[tsn] // dt)
        temppp[tsn] = 0.0
        temppp[tsn] = math.log10(Eobs[tsb]) - math.log10(Esyn[tsb])

    crap = np.mean(temppp)
    Esyn *= 10**crap

    return result_intb, result_mfp, Eobs, Esyn


# -----
def get_symmetric(msv: np.ndarray, indx: int) -> np.ndarray:
    """
    Calculating symmetric waveforms and returning positive side only
    ----------------------------------------------
    Input:
        msv:    The original time seires
        indx:   The half-side point number
    ----------------------------------------------
    Return:
        sym:    The symmetric waveform
    ----------------------------------------------
    """
    sym = 0.5 * msv[indx:] + 0.5 * np.flip(msv[: indx + 1], axis=0)
    return sym


# -----
def get_smooth(data: np.ndarray, para) -> np.ndarray:
    """
    ----------------------------------------------
    Input:
        data:    The original time seires

    Parameter:
        winlen: The smoothing window length
        dt:     Samping interval of the data
        npts:   Data length
    ----------------------------------------------
    Return:
        msv:    The mean-squared waveforms
    ----------------------------------------------
    """
    winlen = para["winlen"]
    dt = para["dt"]
    npts = para["npts"]

    msv = np.ndarray((npts))
    # small window smoothing
    npt = int(winlen / dt) + 1
    half_npt = int(npt / 2)
    arr = np.zeros(int(npt))
    for jj in range(0, (npts)):
        if jj < half_npt:
            arr = data[jj : jj + half_npt]
        elif jj > (npts) - half_npt:
            arr = data[jj:npts]
        else:
            arr = data[jj - half_npt : jj + half_npt]
        msv[jj] = msValue(arr)
    return msv


# -----
