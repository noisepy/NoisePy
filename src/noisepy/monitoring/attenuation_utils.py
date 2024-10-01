import logging
import math
import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from noisepy.monitoring.monitoring_utils import ConfigParameters_monitoring

### -----
# These scripts are aim to perform the 2-D radiative transfer equation
# for scalar waves (Shang and Gao 1988; Sato 1993),
# assuming of isotropic scattering and source radiation in infinite medium
# to calculate synthesized energy densities Esyn.
### -----

logger = logging.getLogger(__name__)


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
def window_determine(tbeg: float, tend: float, nwindows=6) -> Tuple[np.ndarray]:
    """
    # Determine the window begin and end time
    ----------------------------------------------
    Parameters:
        tbeg: The begin time of the window
        tend: The end time of the window
        dt:   The sampling interval
    ----------------------------------------------
    Return:
        twinbe: The window begin and end time array
    ----------------------------------------------
    """

    twinbe = np.ndarray((nwindows, 2))
    coda_length = tend - tbeg

    for i in range(3):
        twinbe[i][0] = tbeg + i * coda_length / 10
        twinbe[i][1] = tbeg + (i + 1) * coda_length / 10

        twinbe[i + 3][0] = tbeg + (i + 5) * coda_length / 10
        twinbe[i + 3][1] = tbeg + (i + 5 + 1) * coda_length / 10

    return twinbe


### -----
def get_energy_density(
    fmsv_single: np.ndarray, dt: float, cvel: float, r: float, npts: int, mean_free_path: float, intrinsic_b: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    # Calculate the energy density function
    ----------------------------------------------
    Parameters:
        fmsv_single: The mean-squared value
        dt:          The sampling interval
        cvel:        The Rayleigh wave velocity
        r:           The distance between the source and receiver
        npts:        The total sample number
        mean_free_path: The mean free path
        intrinsic_b: The intrinsic absorption parameter b
    ----------------------------------------------
    Return:
        Esyn_temp: The synthetic energy density function
        Eobs_temp: The observed energy density function
    ----------------------------------------------
    """

    monito_config = ConfigParameters_monitoring()
    ONESTATION = monito_config.single_station

    # initialize the temporary arrays
    Esyn_temp = np.zeros((npts // 2 + 1))
    Eobs_temp = np.zeros((npts // 2 + 1))

    Eobs_temp = fmsv_single
    # calculate the Esyn and SSR for combination of mean_free_path and intrinsic_b
    for twn in range(npts // 2):
        tm = dt * twn
        s0 = cvel**2 * tm**2 - r**2
        if s0 <= 0:
            # logger.warning(f"s0 {s0} <0")
            continue

        if ONESTATION:
            tmp = ESYN_RadiaTrans_onesta(mean_free_path, tm, r, cvel)
        else:
            tmp = ESYN_RadiaTrans_intersta(mean_free_path, tm, r, cvel)

        Esyn_temp[twn] = tmp * math.exp(-1 * intrinsic_b * tm)

    return Esyn_temp, Eobs_temp


# ---
def scaling_Esyn(Esyn_temp: np.ndarray, Eobs_temp: np.ndarray, twindow: np.ndarray, dt: float) -> np.ndarray:
    """
    # Scaling the Esyn in the specific window
    ----------------------------------------------
    Parameters:
        Esyn_temp: The synthetic energy density function
        Eobs_temp: The observed energy density function
        twindow:   The window begin and end time array
        dt:        The sampling interval
    ----------------------------------------------
    Return:
        Esyn_temp: The scaled synthetic energy density function
    ----------------------------------------------
    """

    # find the scaling factor in the specific window
    SSR_temppp = np.zeros((len(twindow)))
    for tsn in range(len(twindow)):
        tsb = int(twindow[tsn] // dt)
        SSR_temppp[tsn] = math.log10(Eobs_temp[tsb]) - math.log10(Esyn_temp[tsb])
    crap = np.mean(SSR_temppp)
    Esyn_temp *= 10**crap  # scale the Esyn
    # using scalar factor for further fitting processes --> shape matters more than amplitude

    return Esyn_temp, 10**crap


# ### -----
def windowing_SSR(
    coda_window: np.array,
    mean_free_path: float,
    intrinsic_b: float,
    fmsv: np.ndarray,
    dist: np.ndarray,
    npts: int,
    dt: float,
    cvel: float,
    ncoda: int,
    PLOT_CHECK: bool = False,
) -> float:
    """
    # Calculate the sum of squared residuals (SSR) in the specific window
    ----------------------------------------------
    Parameters:
        Eobs: The observed energy density function
        Esyn: The synthetic energy density function
        coda_window: The window begin and end time array
        mean_free_path: The mean free path
        intrinsic_b: The intrinsic absorption parameter b
        npts: The total sample number
    ----------------------------------------------
    Return:
        window_SSR: The sum of squared residuals in the specific window
    """
    Esyn_temp = np.zeros((npts // 2))
    Eobs_temp = np.zeros((npts // 2))

    npair = fmsv.shape[0]
    window_SSR = 0.0
    if PLOT_CHECK:
        fig, ax = plt.subplots(npair, figsize=(6, 40))

    for aa in range(npair):
        twindow = np.arange((coda_window[aa, 0]), (coda_window[aa, 1]), dt)
        r = float(dist[aa])

        # Get the energy density
        Esyn_temp, Eobs_temp = get_energy_density(fmsv[aa], dt, cvel, r, npts, mean_free_path, intrinsic_b)

        # Scale the Esyn
        scaled_Esyn, scaling_amplitude = scaling_Esyn(Esyn_temp, Eobs_temp, twindow, dt)
        # logger.info(f"pair {aa}, (mfp, intb)=({mean_free_path:.1f}, {intrinsic_b:.2f}) \
        # -- scaling amp: {scaling_amplitude:.2f}")

        if PLOT_CHECK:
            ax[aa].plot(Eobs_temp, color="black", ls="-", label="Eobs")
            ax[aa].plot(scaled_Esyn, color="blue", ls="-", label="Esyn")
            ax[aa].plot([twindow[0] // dt, twindow[0] // dt], [0, 1], color="red", ls="--", label="window")
            ax[aa].plot([twindow[-1] // dt, twindow[-1] // dt], [0, 1], color="red", ls="--")
            ax[aa].set_title(
                f"pair {aa} on (mfp, intb)=({mean_free_path:.1f}, {intrinsic_b:.2f}) \
                    -- scaling amp: {scaling_amplitude:.2f}"
            )
            ax[aa].set_yscale("log")
            ax[aa].set_xlabel("Npts")
            ax[aa].set_xlim(0, npts // 8)
            ax[aa].set_ylim(10**-6, 10)

        # Calculate the SSR in the rescale Esyn and Eobs
        tsb = int(twindow[0] // dt)
        tse = int((twindow[-1]) // dt) + 1
        SSR_temp = 0.0
        for twn in range(tsb, tse):
            SSR_temp = SSR_temp + ((Eobs_temp[twn]) - (scaled_Esyn[twn])) ** 2
        window_SSR += SSR_temp

    if PLOT_CHECK:
        os.system("mkdir figure_checking")
        ax[0].legend(loc="upper right")
        plt.tight_layout()
        fname = f"figure_checking/Scaled_density_ncoda{ncoda}_mfp{mean_free_path:.1f}_intb{intrinsic_b:.2f}.png"
        plt.savefig(fname)
        plt.close(fig)

    return window_SSR


### -----
def get_SSR_codawindows(para) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    #  Calculate the sum of squared residuals (SSR) between Eobs and Esyn
    #  for different combination of mean free path and intrinsic absorption parameter b
    ----------------------------------------------
    Parameters:
        dt: data sampling interval
        cvel:  Rayleigh wave velocity
        npts:   Total sample number
        vdist:  Inter-station distance
        mfpx:   The mean free path array
        intby:  The intrinsic absorption parameter b array
        twin_select: Window begin time and end time array
        fmsv_mean: Observed mean-squared value
    ----------------------------------------------
    Return:
        SSR_final: The sum of squared residuals between Eobs and Esyn
        mfpx:      The searching range of mean free path array
        intby:     The searching range of intrinsic absorption parameter b array
    ----------------------------------------------
    """
    # default config parameters which can be customized
    monito_config = ConfigParameters_monitoring()

    dt = para["dt"]
    cvel = para["cvel"]
    vdist = para["vdist"]
    npts = para["npts"]
    mfpx = para["mfp"]
    intby = para["intb"]
    twin_select = para["twin"]
    fmsv_mean = para["fmsv"]
    intb_interval_base = monito_config.intb_interval_base
    mfp_interval_base = monito_config.mfp_interval_base
    nwindows = twin_select.shape[1]

    SSR_final = np.zeros((len(mfpx), len(intby)))
    # grid search in combination of mean_free_path and intrinsic_b
    for nfree in range(len(mfpx)):
        mean_free = 0.2 + mfp_interval_base * nfree
        mfpx[nfree] = mean_free
        for nb in range(len(intby)):
            intrinsic_b = intb_interval_base * (nb + 1)
            intby[nb] = intrinsic_b

            SSR_temp = 0.0
            #### specific window for all pairs
            for ncoda in range(nwindows):
                twindow = twin_select[:, ncoda, :]
                single_window_SSR = windowing_SSR(
                    twindow, mean_free, intrinsic_b, fmsv_mean, vdist, npts, dt, cvel, ncoda
                )

                SSR_temp = SSR_temp + single_window_SSR
                # logger.info(f"window {ncoda}, (mfp, intb)=({mean_free:.1f}, {intrinsic_b:.2f}) \
                # -- single SSR: {single_window_SSR:.2f}, final SSR: {SSR_temp:.2f}")
            SSR_final[nfree][nb] = SSR_temp

        # logger.info(f"mean_free: {mean_free}, intrinsic_b {intrinsic_b}, SSR:{SSR_temp}")
    SSR_final = SSR_final / (np.min(SSR_final[:][:]))

    return SSR_final, mfpx, intby


def get_optimal_Esyn(para) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    # Getting the optimal value from the grid searching results (the SSR output from the get_SSR)
    # Return with the optimal value of mean free path, intrinsic absorption parameter
    # and the optimal fit of synthetic energy density function
    ----------------------------------------------
    Parameters:
        dt: data sampling interval
        cvel:  Rayleigh wave velocity
        npts:   Total sample number
        vdist:  Inter-station distance
        mfpx:   The mean free path array
        intby:  The intrinsic absorption parameter b array
        twin_select: Window begin time and end time array
        fmsv: Observed mean-squared value
        SSR:    The sum of squared residuals over combination of mean free path
                and intrinsic absorption parameter b
        sta_pair: Station pair
    ----------------------------------------------
    Return:
        result_intb: The optimal value of intrinsic absorption parameter b
        result_mfp:  The optimal value of mean free path
        Eobs:   The observed energy density function
        Esyn:   The optimal fit of synthetic energy density function
        scaling_amplitude: The scaling amplitude on synthetic energy density function
    ----------------------------------------------
    """

    dt = para["dt"]
    cvel = para["cvel"]
    vdist = para["vdist"]
    mfpx = para["mfp"]
    intby = para["intb"]
    twin_select = para["twin"]
    npts = para["npts"]
    fmsv = para["fmsv"]

    SSR = para["SSR"]
    sta_pair = para["sta"]

    nwindows = twin_select.shape[1]
    npair = fmsv.shape[0]

    loc = np.where(SSR.T == np.amin(SSR.T))
    if len(loc[0]) > 1:
        loc = (loc[0][0], loc[1][0])

    intrinsic_b = intby[loc[0]]
    mean_free_path = mfpx[loc[1]]
    logger.info(f"Station Pair: {sta_pair}, intrinsic_b {intrinsic_b}, mean_free: {mean_free_path}")

    result_intb = np.take(intrinsic_b, 0)
    result_mfp = np.take(mean_free_path, 0)

    Eobs = np.zeros((npair, nwindows, npts // 2 + 1))
    Esyn = np.zeros((npair, nwindows, npts // 2 + 1))
    scaling_amplitude = np.zeros((npair, nwindows))

    for ncoda in range(nwindows):
        for aa in range(npair):
            twindow = twin_select[aa, ncoda, :]
            # Get the energy density
            Esyn[aa, ncoda], Eobs[aa, ncoda] = get_energy_density(
                fmsv[aa], dt, cvel, vdist, npts, result_mfp, result_intb
            )

            # Scale the Esyn
            scaled_Esyn, scaling_temp = scaling_Esyn(Esyn[aa, ncoda], Eobs[aa, ncoda], twindow, dt)
            logger.info(
                f"nwindow {ncoda}, pair {aa}, (mfp, intb)=({result_mfp:.1f},"
                f"{result_intb:.2f}) -- scaling amp: {scaling_temp:.2f}"
            )
            Esyn[aa, ncoda] = scaled_Esyn
            scaling_amplitude[aa, ncoda] = scaling_temp

    return result_intb, result_mfp, Eobs, Esyn, scaling_amplitude


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
