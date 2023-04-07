import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import pyasdf
import scipy
from obspy.signal.filter import bandpass
from scipy.fftpack import next_fast_len
from scipy.signal import hilbert

"""
check the performance of all different stacking method for noise cross-correlations.

Chengxin Jiang @Harvard Oct/29/2019
Updated @May/2020 to include nth-root stacking, selective stacking and tf-PWS stacking
"""


def pws(cc_array, sampling_rate, power=2, pws_timegate=5.0):
    """
    Performs phase-weighted stack on array of time series.
    Follows methods of Schimmel and Paulssen, 1997.
    If s(t) is time series data (seismogram, or cross-correlation),
    S(t) = s(t) + i*H(s(t)), where H(s(t)) is Hilbert transform of s(t)
    S(t) = s(t) + i*H(s(t)) = A(t)*exp(i*phi(t)), where
    A(t) is envelope of s(t) and phi(t) is phase of s(t)
    Phase-weighted stack, g(t), is then:
    g(t) = 1/N sum j = 1:N s_j(t) * | 1/N sum k = 1:N exp[i * phi_k(t)]|^v
    where N is number of traces used, v is sharpness of phase-weighted stack

    PARAMETERS:
    ---------------------
    arr: N length array of time series data (numpy.ndarray)
    sampling_rate: sampling rate of time series arr (int)
    power: exponent for phase stack (int)
    pws_timegate: number of seconds to smooth phase stack (float)

    RETURNS:
    ---------------------
    weighted: Phase weighted stack of time series data (numpy.ndarray)

    Originally written by Tim Clements
    Modified by Chengxin Jiang @Harvard
    """

    if cc_array.ndim == 1:
        print("2D matrix is needed for pws")
        return cc_array
    N, M = cc_array.shape

    # construct analytical signal
    analytic = hilbert(cc_array, axis=1, N=next_fast_len(M))[:, :M]
    phase = np.angle(analytic)
    phase_stack = np.mean(np.exp(1j * phase), axis=0)
    phase_stack = np.abs(phase_stack) ** (power)

    # weighted is the final waveforms
    weighted = np.multiply(cc_array, phase_stack)
    return np.mean(weighted, axis=0)


def adaptive_filter(cc_array, g):
    """
    the adaptive covariance filter to enhance coherent signals. Fellows the method of
    Nakata et al., 2015 (Appendix B)

    the filtered signal [x1] is given by x1 = ifft(P*x1(w)) where x1 is the ffted spectra
    and P is the filter. P is constructed by using the temporal covariance matrix.

    PARAMETERS:
    ----------------------
    cc_array: numpy.ndarray contains the 2D traces of daily/hourly cross-correlation functions
    g: a positive number to adjust the filter harshness
    RETURNS:
    ----------------------
    narr: numpy vector contains the stacked cross correlation function

    Written by Chengxin Jiang @Harvard (Oct2019)
    """
    if cc_array.ndim == 1:
        print("2D matrix is needed for adaptive filtering")
        return cc_array
    N, M = cc_array.shape
    Nfft = next_fast_len(M)

    # fft the 2D array
    spec = scipy.fftpack.fft(cc_array, axis=1, n=Nfft)[:, :M]

    # make cross-spectrm matrix
    cspec = np.zeros(shape=(N * N, M), dtype=np.complex64)
    for ii in range(N):
        for jj in range(N):
            kk = ii * N + jj
            cspec[kk] = spec[ii] * np.conjugate(spec[jj])

    S1 = np.zeros(M, dtype=np.complex64)
    S2 = np.zeros(M, dtype=np.complex64)
    # construct the filter P
    for ii in range(N):
        mm = ii * N + ii
        S2 += cspec[mm]
        for jj in range(N):
            kk = ii * N + jj
            S1 += cspec[kk]

    p = np.power((S1 - S2) / (S2 * (N - 1)), g)

    # make ifft
    narr = np.real(scipy.fftpack.ifft(np.multiply(p, spec), Nfft, axis=1)[:, :M])
    return np.mean(narr, axis=0)


def robust_stack(cc_array, epsilon):
    """
    this is a robust stacking algorithm described in Palvis and Vernon 2010

    PARAMETERS:
    ----------------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    epsilon: residual threhold to quit the iteration
    RETURNS:
    ----------------------
    newstack: numpy vector contains the stacked cross correlation

    Written by Marine Denolle
    """
    res = 9e9  # residuals
    w = np.ones(cc_array.shape[0])
    nstep = 0
    newstack = np.median(cc_array, axis=0)
    while res > epsilon:
        stack = newstack
        for i in range(cc_array.shape[0]):
            crap = np.multiply(stack, cc_array[i, :].T)
            crap_dot = np.sum(crap)
            di_norm = np.linalg.norm(cc_array[i, :])
            ri = cc_array[i, :] - crap_dot * stack
            ri_norm = np.linalg.norm(ri)
            w[i] = np.abs(crap_dot) / di_norm / ri_norm  # /len(cc_array[:,1])
        # print(w)
        w = w / np.sum(w)
        newstack = np.sum((w * cc_array.T).T, axis=0)  # /len(cc_array[:,1])
        res = np.linalg.norm(newstack - stack, ord=1) / np.linalg.norm(newstack) / len(cc_array[:, 1])
        nstep += 1
        if nstep > 10:
            return newstack, w, nstep
    return newstack, w, nstep


def nroot_stack(cc_array, power):
    """
    this is nth-root stacking algorithm translated based on the matlab function
    from https://github.com/xtyangpsp/SeisStack (by Xiaotao Yang; follows the
    reference of Millet, F et al., 2019 JGR)

    Parameters:
    ------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    power: np.int, nth root for the stacking

    Returns:
    ------------
    nstack: np.ndarray, final stacked waveforms

    Written by Chengxin Jiang @ANU (May2020)
    """
    if cc_array.ndim == 1:
        print("2D matrix is needed for nroot_stack")
        return cc_array
    N, M = cc_array.shape
    dout = np.zeros(M, dtype=np.float32)

    # construct y
    for ii in range(N):
        dat = cc_array[ii, :]
        dout += np.sign(dat) * np.abs(dat) ** (1 / power)
    dout /= N

    # the final stacked waveform
    nstack = np.sign(dout) * np.abs(dout) ** (power - 1)

    return nstack


def selective_stack(cc_array, epsilon, cc_th):
    """
    this is a selective stacking algorithm developed by Jared Bryan/Kurama Okubo.

    PARAMETERS:
    ----------------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    epsilon: residual threhold to quit the iteration
    cc_th: numpy.float, threshold of correlation coefficient to be selected

    RETURNS:
    ----------------------
    newstack: numpy vector contains the stacked cross correlation
    nstep: np.int, total number of iterations for the stacking

    Originally ritten by Marine Denolle
    Modified by Chengxin Jiang @Harvard (Oct2020)
    """
    if cc_array.ndim == 1:
        print("2D matrix is needed for nroot_stack")
        return cc_array
    N, M = cc_array.shape

    res = 9e9  # residuals
    cof = np.zeros(N, dtype=np.float32)
    newstack = np.mean(cc_array, axis=0)

    nstep = 0
    # start iteration
    while res > epsilon:
        for ii in range(N):
            cof[ii] = np.corrcoef(newstack, cc_array[ii, :])[0, 1]

        # find good waveforms
        indx = np.where(cof >= cc_th)[0]
        if not len(indx):
            raise ValueError("cannot find good waveforms inside selective stacking")
        oldstack = newstack
        newstack = np.mean(cc_array[indx], axis=0)
        res = np.linalg.norm(newstack - oldstack) / (np.linalg.norm(newstack) * M)
        nstep += 1

    return newstack, nstep


###############################
####### main function #########
###############################

# input files
sfiles = sorted(glob.glob("/Users/chengxin/Documents/ANU/STACK_ALL/ME04_ME36.h5"))

# common parameters
ccomp = "ZZ"
norm_freq = False
lag = 60

write_file = False
do_filter = True
fqmin = 0.3
fqmax = 0.5

# loop through each station-pair
for sfile in sfiles:
    # useful parameters from each asdf file
    with pyasdf.ASDFDataSet(sfile, mode="r") as ds:
        alist = ds.auxiliary_data.list()
        try:
            dt = ds.auxiliary_data[alist[0]][ccomp].parameters["dt"]
            dist = ds.auxiliary_data[alist[0]][ccomp].parameters["dist"]
        except Exception:
            print("continue! no %s component exist" % ccomp)
            continue
        print("working on %s that is %5.2fkm apart" % (sfile, dist))

        # stacked data and filter it
        sdata = ds.auxiliary_data[alist[0]][ccomp].data[:]
        para = ds.auxiliary_data[alist[0]][ccomp].parameters

        # time domain variables
        nwin = len(alist[1:])
        npts = sdata.size
        tvec = np.arange(-npts // 2 + 1, npts // 2 + 1) * dt
        indx = np.where(np.abs(tvec) <= lag)[0]
        npts = len(indx)
        tvec = np.arange(-npts // 2 + 1, npts // 2 + 1) * dt
        ndata = np.zeros(shape=(nwin, npts), dtype=np.float32)

        # freq domain variables
        nfft = int(next_fast_len(npts))
        nfreq = scipy.fftpack.fftfreq(nfft, d=dt)[: nfft // 2]
        timestamp = np.empty(nwin, dtype="datetime64[s]")

        #################################
        ####### load data matrix ########
        #################################
        ii = 0
        for ilist in alist[2:]:
            try:
                ndata[ii] = ds.auxiliary_data[ilist][ccomp].data[indx]
                # timestamp[ii] = obspy.UTCDateTime(int(ilist.split('T')[-1]))
                ii += 1
            except Exception:
                continue

    # remove empty data
    nwin = ii
    ndata = ndata[:nwin]
    # timestamp = timestamp[:nwin]

    # do stacking to see their waveforms
    t0 = time.time()
    slinear = np.mean(ndata, axis=0)
    t1 = time.time()
    spws = pws(ndata, int(1 / dt))
    t2 = time.time()
    srobust, ww, nstep = robust_stack(ndata, 0.001)
    t3 = time.time()
    sACF = adaptive_filter(ndata, 1)
    t4 = time.time()
    nroot = nroot_stack(ndata, 2)
    t5 = time.time()
    sstack, nstep = selective_stack(ndata, 0.001, 0.01)
    t6 = time.time()
    tbase = t1 - t0

    print(
        "pws,robust, acf,nth-root and selective-stacking are %d %d %d %d and %d times relative to linear"
        % (
            int((t2 - t1) / tbase),
            int((t3 - t2) / tbase),
            int((t4 - t3) / tbase),
            int((t5 - t4) / tbase),
            int((t6 - t5) / tbase),
        )
    )

    # do filtering if needed
    if do_filter:
        slinear = bandpass(slinear, fqmin, fqmax, int(1 / dt), corners=4, zerophase=True)
        spws = bandpass(spws, fqmin, fqmax, int(1 / dt), corners=4, zerophase=True)
        srobust = bandpass(srobust, fqmin, fqmax, int(1 / dt), corners=4, zerophase=True)
        sACF = bandpass(sACF, fqmin, fqmax, int(1 / dt), corners=4, zerophase=True)
        nroot = bandpass(nroot, fqmin, fqmax, int(1 / dt), corners=4, zerophase=True)
        sstack = bandpass(sstack, fqmin, fqmax, int(1 / dt), corners=4, zerophase=True)

    # variations of correlation coefficient relative to the linear stacks
    corr = np.zeros(nwin, dtype=np.float32)
    for ii in range(nwin):
        corr[ii] = np.corrcoef(slinear, ndata[ii])[0, 1]

    # plot the 2D background matrix and the stacked data
    fig, ax = plt.subplots(3, figsize=(8, 12), sharex=False)

    # 2D raw matrix
    ax[0].matshow(ndata, cmap="seismic", extent=[-lag, lag, nwin, 0], aspect="auto")
    ax[0].set_xlabel("time [s]")
    ax[0].set_yticks(np.arange(0, nwin, step=180))
    ax[0].set_yticklabels(np.arange(0, nwin // 3, step=60))
    ax[0].xaxis.set_ticks_position("bottom")
    ax[0].set_title("%s %d km" % (sfile.split("/")[-1], dist))

    # stacked waveforms
    ax[1].plot(tvec, slinear / np.max(slinear))
    ax[1].plot(tvec, srobust / np.max(srobust) + 2)
    ax[1].plot(tvec, spws / np.max(spws) + 4)
    ax[1].plot(tvec, sACF / np.max(sACF) + 6)
    ax[1].plot(tvec, nroot / np.max(nroot) + 8)
    ax[1].plot(tvec, sstack / np.max(sstack) + 10)
    ax[1].legend(["linear", "robust", "pws", "ACF", "nroot", "selective"], loc="lower left")
    ax[1].set_xlabel("time [s]")

    # cc coeff variations
    ax[2].plot(corr, "b-")
    ax[2].grid(True)
    ax[2].set_xlabel("# windows")
    ax[2].set_ylabel("cc coeff.")
    ax[2].legend(["cc coeff"], loc="upper left")
    axx = ax[2].twinx()
    axx.plot(ww, "r-")
    axx.set_ylabel("robust stack weight")
    axx.legend(["stack weight"], loc="upper right")
    fig.tight_layout()
    plt.show()
