# noise_module.py from the Jiang and Denolle 2019
# This file contains old 2019 functions that are not used in the new version of NoisePy

import glob
import os

import logger
import numpy as np
import obspy
import pandas as pd
import pws
import scipy

from noisepy.seis.noise_module import (
    moving_ave,
    robust_stack,
    stats2inv_mseed,
    stats2inv_sac,
    stats2inv_staxml,
)

from .datatypes import StackMethod


def make_timestamps(prepro_para):
    """
    this function prepares the timestamps of both the starting and ending time of each mseed/sac file that
    is stored on local machine. this time info is used to search all stations in specific time chunck
    when preparing noise data in ASDF format. it creates a csv file containing all timestamp info if the
    file does not exist (used in S0B)f
    PARAMETERS:
    -----------------------
    prepro_para: a dic containing all pre-processing parameters used in S0B
    RETURNS:
    -----------------------
    all_stimes: numpy float array containing startting and ending time for all SAC/mseed files
    """
    # load parameters from para dic
    wiki_file = prepro_para["wiki_file"]
    messydata = prepro_para["messydata"]
    RAWDATA = prepro_para["RAWDATA"]
    allfiles_path = prepro_para["allfiles_path"]

    if os.path.isfile(wiki_file):
        tmp = pd.read_csv(wiki_file)
        allfiles = tmp["names"]
        all_stimes = np.zeros(shape=(len(allfiles), 2), dtype=np.float)
        all_stimes[:, 0] = tmp["starttime"]
        all_stimes[:, 1] = tmp["endtime"]

    # have to read each sac/mseed data one by one
    else:
        allfiles = glob.glob(allfiles_path)
        nfiles = len(allfiles)
        if not nfiles:
            raise ValueError("Abort! no data found in subdirectory of %s" % RAWDATA)
        all_stimes = np.zeros(shape=(nfiles, 2), dtype=np.float)

        if messydata:
            # get VERY precise trace-time from the header
            for ii in range(nfiles):
                try:
                    tr = obspy.read(allfiles[ii])
                    all_stimes[ii, 0] = tr[0].stats.starttime - obspy.UTCDateTime(1970, 1, 1)
                    all_stimes[ii, 1] = tr[0].stats.endtime - obspy.UTCDateTime(1970, 1, 1)
                except Exception as e:
                    logger.error(e)
                    continue
        else:
            # get rough estimates of the time based on the folder: need modified to accommodate your data
            for ii in range(nfiles):
                year = int(allfiles[ii].split("/")[-2].split("_")[1])
                # julia = int(allfiles[ii].split('/')[-2].split('_')[2])
                # all_stimes[ii,0] = obspy.UTCDateTime(year=year,julday=julia)
                # -obspy.UTCDateTime(year=1970,month=1,day=1)
                month = int(allfiles[ii].split("/")[-2].split("_")[2])
                day = int(allfiles[ii].split("/")[-2].split("_")[3])
                all_stimes[ii, 0] = obspy.UTCDateTime(year=year, month=month, day=day) - obspy.UTCDateTime(
                    year=1970, month=1, day=1
                )
                all_stimes[ii, 1] = all_stimes[ii, 0] + 86400

        # save name and time info for later use if the file not exist
        if not os.path.isfile(wiki_file):
            wiki_info = {
                "names": allfiles,
                "starttime": all_stimes[:, 0],
                "endtime": all_stimes[:, 1],
            }
            df = pd.DataFrame(wiki_info, columns=["names", "starttime", "endtime"])
            df.to_csv(wiki_file)
    return all_stimes


def stats2inv(stats, prepro_para, locs=None):
    """
    this function creates inventory given the stats parameters in an obspy stream or a station list.
    (used in S0B)
    PARAMETERS:
    ------------------------
    stats: obspy trace stats object containing all station header info
    prepro_para: dict containing fft parameters, such as frequency bands and selection for instrument
    response removal etc.
    locs:  panda data frame of the station list. it is needed for convering miniseed files into ASDF
    RETURNS:
    ------------------------
    inv: obspy inventory object of all station info to be used later
    """
    staxml = prepro_para["stationxml"]
    respdir = prepro_para["respdir"]
    input_fmt = prepro_para["input_fmt"]
    if staxml:
        return stats2inv_staxml(stats, respdir)
    if input_fmt == "sac":
        return stats2inv_sac(stats)
    elif input_fmt == "mseed":
        return stats2inv_mseed(stats, locs)


def correlate_nonlinear_stack(fft1_smoothed_abs, fft2, D, Nfft, dataS_t):
    """
    this function does the cross-correlation in freq domain and has the option to keep sub-stacks of
    the cross-correlation if needed. it takes advantage of the linear relationship of ifft, so that
    stacking is performed in spectrum domain first to reduce the total number of ifft. (used in S1)
    PARAMETERS:
    ---------------------
    fft1_smoothed_abs: smoothed power spectral density of the FFT for the source station
    fft2: raw FFT spectrum of the receiver station
    D: dictionary containing following parameters:
        maxlag:  maximum lags to keep in the cross correlation
        dt:      sampling rate (in s)
        nwin:    number of segments in the 2D matrix
        method:  cross-correlation methods selected by the user
        freqmin: minimum frequency (Hz)
        freqmax: maximum frequency (Hz)
    Nfft:    number of frequency points for ifft
    dataS_t: matrix of datetime object.
    RETURNS:
    ---------------------
    s_corr: 1D or 2D matrix of the averaged or sub-stacks of cross-correlation functions in time domain
    t_corr: timestamp for each sub-stack or averaged function
    n_corr: number of included segments for each sub-stack or averaged function
    """
    # ----load paramters----
    dt = D["dt"]
    maxlag = D["maxlag"]
    method = D["cc_method"]
    cc_len = D["cc_len"]
    substack = D["substack"]
    stack_method = D["stack_method"]
    substack_len = D["substack_len"]
    smoothspect_N = D["smoothspect_N"]

    nwin = fft1_smoothed_abs.shape[0]
    Nfft2 = fft1_smoothed_abs.shape[1]

    # ------convert all 2D arrays into 1D to speed up--------
    corr = np.zeros(nwin * Nfft2, dtype=np.complex64)
    corr = fft1_smoothed_abs.reshape(
        fft1_smoothed_abs.size,
    ) * fft2.reshape(
        fft2.size,
    )

    # normalize by receiver spectral for coherency
    if method == "coherency":
        temp = moving_ave(
            np.abs(
                fft2.reshape(
                    fft2.size,
                )
            ),
            smoothspect_N,
        )
        corr /= temp
    corr = corr.reshape(nwin, Nfft2)

    # transform back to time domain waveforms
    s_corr = np.zeros(shape=(nwin, Nfft), dtype=np.float32)  # stacked correlation
    ampmax = np.zeros(nwin, dtype=np.float32)
    n_corr = np.zeros(nwin, dtype=np.int16)  # number of correlations for each substack
    t_corr = dataS_t  # timestamp
    crap = np.zeros(Nfft, dtype=np.complex64)
    for i in range(nwin):
        n_corr[i] = 1
        crap[:Nfft2] = corr[i, :]
        crap[:Nfft2] = crap[:Nfft2] - np.mean(crap[:Nfft2])  # remove the mean in freq domain (spike at t=0)
        crap[-(Nfft2) + 1 :] = np.flip(np.conj(crap[1:(Nfft2)]), axis=0)
        crap[0] = complex(0, 0)
        s_corr[i, :] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

    ns_corr = s_corr
    for iii in range(ns_corr.shape[0]):
        ns_corr[iii] /= np.max(np.abs(ns_corr[iii]))

    if substack:
        if substack_len == cc_len:
            # remove abnormal data
            ampmax = np.max(s_corr, axis=1)
            tindx = np.where((ampmax < 20 * np.median(ampmax)) & (ampmax > 0))[0]
            s_corr = s_corr[tindx, :]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

        else:
            # get time information
            Ttotal = dataS_t[-1] - dataS_t[0]  # total duration of what we have now
            tstart = dataS_t[0]

            nstack = int(np.round(Ttotal / substack_len))
            ampmax = np.zeros(nstack, dtype=np.float32)
            s_corr = np.zeros(shape=(nstack, Nfft), dtype=np.float32)
            n_corr = np.zeros(nstack, dtype=np.int)
            t_corr = np.zeros(nstack, dtype=np.float)
            crap = np.zeros(Nfft, dtype=np.complex64)

            for istack in range(nstack):
                # find the indexes of all of the windows that start or end within
                itime = np.where((dataS_t >= tstart) & (dataS_t < tstart + substack_len))[0]
                if len(itime) == 0:
                    tstart += substack_len
                    continue

                crap[:Nfft2] = np.mean(corr[itime, :], axis=0)  # linear average of the correlation
                crap[:Nfft2] = crap[:Nfft2] - np.mean(crap[:Nfft2])  # remove the mean in freq domain (spike at t=0)
                crap[-(Nfft2) + 1 :] = np.flip(np.conj(crap[1:(Nfft2)]), axis=0)
                crap[0] = complex(0, 0)
                s_corr[istack, :] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))
                n_corr[istack] = len(itime)  # number of windows stacks
                t_corr[istack] = tstart  # save the time stamps
                tstart += substack_len
                # print('correlation done and stacked at time %s' % str(t_corr[istack]))

            # remove abnormal data
            ampmax = np.max(s_corr, axis=1)
            tindx = np.where((ampmax < 20 * np.median(ampmax)) & (ampmax > 0))[0]
            s_corr = s_corr[tindx, :]
            t_corr = t_corr[tindx]
            n_corr = n_corr[tindx]

    else:
        # average daily cross correlation functions
        if stack_method == StackMethod.LINEAR:
            ampmax = np.max(s_corr, axis=1)
            tindx = np.where((ampmax < 20 * np.median(ampmax)) & (ampmax > 0))[0]
            s_corr = np.mean(s_corr[tindx], axis=0)
            t_corr = dataS_t[0]
            n_corr = len(tindx)
        elif stack_method == StackMethod.ROBUST:
            logger.info("do robust substacking")
            s_corr = robust_stack(s_corr, 0.001)
            t_corr = dataS_t[0]
            n_corr = nwin
    #  elif stack_method == 'selective':
    #      print('do selective substacking')
    #      s_corr = selective_stack(s_corr,0.001)
    #      t_corr = dataS_t[0]
    #      n_corr = nwin

    # trim the CCFs in [-maxlag maxlag]
    t = np.arange(-Nfft2 + 1, Nfft2) * dt
    ind = np.where(np.abs(t) <= maxlag)[0]
    if s_corr.ndim == 1:
        s_corr = s_corr[ind]
    elif s_corr.ndim == 2:
        s_corr = s_corr[:, ind]
    return s_corr, t_corr, n_corr, ns_corr[:, ind]


def stacking_rma(cc_array, cc_time, cc_ngood, stack_para):
    """
    this function stacks the cross correlation data according to the user-defined substack_len parameter
    PARAMETERS:
    ----------------------
    cc_array: 2D numpy float32 matrix containing all segmented cross-correlation data
    cc_time:  1D numpy array of timestamps for each segment of cc_array
    cc_ngood: 1D numpy int16 matrix showing the number of segments for each sub-stack and/or full stack
    stack_para: a dict containing all stacking parameters
    RETURNS:
    ----------------------
    cc_array, cc_ngood, cc_time: same to the input parameters but with abnormal cross-correaltions removed
    allstacks1: 1D matrix of stacked cross-correlation functions over all the segments
    nstacks:    number of overall segments for the final stacks
    """
    # load useful parameters from dict
    sampling_rate = stack_para["sampling_rate"]
    smethod = stack_para["stack_method"]
    rma_substack = stack_para["rma_substack"]
    rma_step = stack_para["rma_step"]
    start_date = stack_para["start_date"]
    end_date = stack_para["end_date"]
    npts = cc_array.shape[1]

    # remove abnormal data
    ampmax = np.max(cc_array, axis=1)
    tindx = np.where((ampmax < 20 * np.median(ampmax)) & (ampmax > 0))[0]
    if not len(tindx):
        allstacks1 = []
        allstacks2 = []
        nstacks = 0
        cc_array = []
        cc_ngood = []
        cc_time = []
        return cc_array, cc_ngood, cc_time, allstacks1, allstacks2, nstacks
    else:
        # remove ones with bad amplitude
        cc_array = cc_array[tindx, :]
        cc_time = cc_time[tindx]
        cc_ngood = cc_ngood[tindx]

        # do substacks
        if rma_substack:
            tstart = obspy.UTCDateTime(start_date) - obspy.UTCDateTime(1970, 1, 1)
            tend = obspy.UTCDateTime(end_date) - obspy.UTCDateTime(1970, 1, 1)
            ttime = tstart
            nstack = int(np.round((tend - tstart) / (rma_step * 3600)))
            ncc_array = np.zeros(shape=(nstack, npts), dtype=np.float32)
            ncc_time = np.zeros(nstack, dtype=np.float)
            ncc_ngood = np.zeros(nstack, dtype=np.int)

            # loop through each time
            for ii in range(nstack):
                sindx = np.where((cc_time >= ttime) & (cc_time < ttime + rma_substack * 3600))[0]

                # when there are data in the time window
                if len(sindx):
                    ncc_array[ii] = np.mean(cc_array[sindx], axis=0)
                    ncc_time[ii] = ttime
                    ncc_ngood[ii] = np.sum(cc_ngood[sindx], axis=0)
                ttime += rma_step * 3600

            # remove bad ones
            tindx = np.where(ncc_ngood > 0)[0]
            ncc_array = ncc_array[tindx]
            ncc_time = ncc_time[tindx]
            ncc_ngood = ncc_ngood[tindx]

        # do stacking
        allstacks1 = np.zeros(npts, dtype=np.float32)
        allstacks2 = np.zeros(npts, dtype=np.float32)
        allstacks3 = np.zeros(npts, dtype=np.float32)
        allstacks4 = np.zeros(npts, dtype=np.float32)

        if smethod == StackMethod.LINEAR:
            allstacks1 = np.mean(cc_array, axis=0)
        elif smethod == StackMethod.PWS:
            allstacks1 = pws(cc_array, sampling_rate)
        elif smethod == StackMethod.ROBUST:
            (
                allstacks1,
                w,
            ) = robust_stack(cc_array, 0.001)
        elif smethod == StackMethod.SELECTIVE:
            allstacks1 = selective_stack(cc_array, 0.001)
        elif smethod == StackMethod.ALL:
            allstacks1 = np.mean(cc_array, axis=0)
            allstacks2 = pws(cc_array, sampling_rate)
            allstacks3 = robust_stack(cc_array, 0.001)
            allstacks4 = selective_stack(cc_array, 0.001)
        nstacks = np.sum(cc_ngood)

    # replace the array for substacks
    if rma_substack:
        cc_array = ncc_array
        cc_time = ncc_time
        cc_ngood = ncc_ngood

    # good to return
    return (
        cc_array,
        cc_ngood,
        cc_time,
        allstacks1,
        allstacks2,
        allstacks3,
        allstacks4,
        nstacks,
    )


def selective_stack(cc_array, epsilon):
    """
    this is a selective stacking algorithm developed by Jared Bryan.

    PARAMETERS:
    ----------------------
    cc_array: numpy.ndarray contains the 2D cross correlation matrix
    epsilon: residual threhold to quit the iteration
    RETURNS:
    ----------------------
    newstack: numpy vector contains the stacked cross correlation

    Written by Marine Denolle
    """
    cc = np.ones(cc_array.shape[0])
    newstack = np.mean(cc_array, axis=0)
    for i in range(cc_array.shape[0]):
        cc[i] = np.sum(np.multiply(newstack, cc_array[i, :].T))
    ik = np.where(cc >= epsilon)[0]
    newstack = np.mean(cc_array[ik, :], axis=0)

    return newstack, cc


def selective_stack(cc_array, epsilon, cc_th):  # noqa: F811
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
        logger.debug("2D matrix is needed for nroot_stack")
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


def get_cc(s1, s_ref):
    # returns the correlation coefficient between waveforms in s1 against reference
    # waveform s_ref.
    #
    cc = np.zeros(s1.shape[0])
    s_ref_norm = np.linalg.norm(s_ref)
    for i in range(s1.shape[0]):
        cc[i] = np.sum(np.multiply(s1[i, :], s_ref)) / np.linalg.norm(s1[i, :]) / s_ref_norm
    return cc


# function to extract the dispersion from the image
def extract_dispersion(amp, per, vel):
    """
    this function takes the dispersion image from CWT as input, tracks the global maxinum on
    the wavelet spectrum amplitude and extract the sections with continous and high quality data

    PARAMETERS:
    ----------------
    amp: 2D amplitude matrix of the wavelet spectrum
    phase: 2D phase matrix of the wavelet spectrum
    per:  period vector for the 2D matrix
    vel:  vel vector of the 2D matrix
    RETURNS:
    ----------------
    per:  central frequency of each wavelet scale with good data
    gv:   group velocity vector at each frequency
    """
    maxgap = 5
    nper = amp.shape[0]
    gv = np.zeros(nper, dtype=np.float32)
    dvel = vel[1] - vel[0]

    # find global maximum
    for ii in range(nper):
        maxvalue = np.max(amp[ii], axis=0)
        indx = list(amp[ii]).index(maxvalue)
        gv[ii] = vel[indx]

    # check the continuous of the dispersion
    for ii in range(1, nper - 15):
        # 15 is the minumum length needed for output
        for jj in range(15):
            if np.abs(gv[ii + jj] - gv[ii + 1 + jj]) > maxgap * dvel:
                gv[ii] = 0
                break

    # remove the bad ones
    indx = np.where(gv > 0)[0]

    return per[indx], gv[indx]
