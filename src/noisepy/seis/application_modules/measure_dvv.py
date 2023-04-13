import os

import matplotlib.pyplot as plt
import numpy as np
import obspy
import pyasdf
from obspy.signal.filter import bandpass
from pandas.plotting import register_matplotlib_converters

from . import noise_module

# register datetime converter

register_matplotlib_converters()

"""
this application script of NoisePy is to perform dv/v analysis on the
resulted cross-correlation functions from S2. Note that, to use this script,
the `keep_substack` parameter in S2 has to be turned `True` when
running S2. So the sub-stacked waveforms can be saved and
further to be compared with the all-stacked waveforms to measure dv/v.

Authors: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
         Marine Denolle (mdenolle@fas.harvard.edu)

NOTE:
    0) this script is only showing an example of how dv/v can be
    measured on the resulted file from S2, and the users need to
    expand/modify this script in order to apply for regional studies;
    1) See Yuan et al., (2019) for more details on the comparison
    of different methods for mesuring dv/v as well as the numerical validation.
"""

############################################
############ PAMAETER SECTION ##############
############################################

# input data and targeted component
rootpath = os.path.join(
    os.path.expanduser("~"), "Documents/NoisePy_example/SCAL/"
)  # root path for this data processing
sfile = os.path.join(rootpath, "STACK_month/CI.BLC/CI.BLC_CI.MPI.h5")  # ASDF file containing stacked data
outdir = os.path.join(rootpath, "figures/monitoring")  # dir where to output dispersive image and extracted dispersion
if not os.path.isdir(outdir):
    os.mkdir(outdir)

# targeted component
stack_method = "linear"  # which stacked data to measure dispersion info
ccomp = "ZZ"  # cross component

# pre-defined group velocity to window direct and code waves
vmin = 0.8  # minimum velocity of the direct waves -> start of the coda window
lwin = 150  # window length in sec for the coda waves

# basic parameters
freq = [0.1, 0.2, 0.3, 0.5]  # targeted frequency band for waveform monitoring
nfreq = len(freq) - 1
onelag = False  # make measurement one one lag or two
norm_flag = True  # whether to normalize the cross-correlation waveforms
do_stretch = True  # use strecthing method or not
do_dtw = False  # use dynamic time warping method or not
do_mwcs = True  # use moving-window cross spectrum method or not
do_mwcc = False  # use moving-window cross correlation method or not
do_wts = True  # use wavelet streching method or not
do_wxs = True  # use wavelet cross spectrum method or not

# parameters for stretching method
epsilon = 2 / 100  # limit for dv/v (in decimal)
nbtrial = 50  # number of increment of dt [-epsilon,epsilon] for the streching

# parameters for DTW
mlag = 50  # maxmum points to move (times dt gives the maximum time shifts)
b = 5  # strain limit (to be tested)
direct = 1  # direction to accumulate errors (1=forward, -1=backward)

# parameters for MWCS & MWCC
move_win_sec = 1.2 * int(1 / np.min(freq))  # moving window length (in sec)
step_sec = 0.3 * move_win_sec  # step for moving window sliding (in sec)

# parameters for wavelet domain methods
dj = 1 / 12  # Spacing between discrete scales. Default value is 1/12.
s0 = -1  # Smallest scale of the wavelet. Default value is 2*dt.
J = -1  # Number of scales less one.
wvn = "morlet"  # wavelet class

##############################################
############ LOAD WAVEFORM DATA ##############
##############################################

# load stacked and sub-stacked waveforms
with pyasdf.ASDFDataSet(sfile, mode="r") as ds:
    dtype = "Allstack0" + stack_method
    substacks = ds.auxiliary_data.list()
    nwin = len(substacks) - 2
    try:
        dt = ds.auxiliary_data[dtype][ccomp].parameters["dt"]
        dist = ds.auxiliary_data[dtype][ccomp].parameters["dist"]
        maxlag = ds.auxiliary_data[dtype][ccomp].parameters["maxlag"]
        tdata = ds.auxiliary_data[dtype][ccomp].data[:]
    except Exception:
        raise ValueError("cannot open %s to read" % sfile)

# make conda window based on vmin
twin = [int(dist / vmin), int(dist / vmin) + lwin]
if twin[1] > maxlag:
    raise ValueError("proposed window exceeds limit! reduce %d" % lwin)

# ref and tvec
ref = tdata
tvec_all = np.arange(-maxlag, maxlag + dt, dt)
# add 20 s to the coda window for plotting purpose
disp_indx = np.where(np.abs(tvec_all) <= np.max(twin) + 20)[0]
# casual and acasual coda window
pwin_indx = np.where((tvec_all >= np.min(twin)) & (tvec_all < np.max(twin)))[0]
nwin_indx = np.where((tvec_all <= -np.min(twin)) & (tvec_all >= -np.max(twin)))[0]
tvec_disp = tvec_all[disp_indx]
# npts for the win and raw
npts_all = len(tvec_all)
npts_win = len(pwin_indx)

# save parameters as a dictionary
para = {
    "twin": twin,
    "freq": freq,
    "dt": dt,
    "ccomp": ccomp,
    "onelag": onelag,
    "norm_flag": norm_flag,
    "npts_all": npts_all,
    "npts_win": npts_win,
}

# allocate matrix for cur and ref waveforms and corr coefficient
cur = np.zeros(shape=(nwin, npts_all), dtype=np.float32)
tcur = np.zeros(shape=(nwin, npts_all), dtype=np.float32)
pcor_cc = np.zeros(shape=(nwin), dtype=np.float32)
ncor_cc = np.zeros(shape=(nwin), dtype=np.float32)
timestamp = np.empty(nwin, dtype="datetime64[s]")

# tick inc for plotting
if nwin > 100:
    tick_inc = int(nwin / 10)
elif nwin > 10:
    tick_inc = int(nwin / 5)
else:
    tick_inc = 2

# load all current waveforms and get corr-coeff
with pyasdf.ASDFDataSet(sfile, mode="r") as ds:
    # loop through each freq band
    for ifreq in range(nfreq):
        # freq parameters
        freq1 = freq[ifreq]
        freq2 = freq[ifreq + 1]
        para["freq"] = [freq1, freq2]
        move_win_sec = 1.2 * int(1 / freq1)

        # reference waveform
        tref = bandpass(ref, freq1, freq2, int(1 / dt), corners=4, zerophase=True)
        if norm_flag:
            tref = tref / np.max(np.abs(tref))

        # loop through each cur waveforms and do filtering
        igood = 0
        for ii in range(nwin):
            try:
                cur[igood] = ds.auxiliary_data[substacks[ii + 2]][ccomp].data[:]
            except Exception:
                continue
            timestamp[igood] = obspy.UTCDateTime(np.float(substacks[ii + 2][1:]))
            tcur[igood] = bandpass(cur[igood], freq1, freq2, int(1 / dt), corners=4, zerophase=True)
            if norm_flag:
                tcur[igood] /= np.max(np.abs(tcur[igood]))

            # get cc coeffient
            pcor_cc[igood] = np.corrcoef(tref[pwin_indx], tcur[igood, pwin_indx])[0, 1]
            ncor_cc[igood] = np.corrcoef(tref[nwin_indx], tcur[igood, nwin_indx])[0, 1]
            igood += 1
        nwin = igood

        ############ PLOT WAVEFORM DATA AND CC ##############
        # plot the raw waveform and the correlation coefficient
        plt.figure(figsize=(11, 12))
        ax0 = plt.subplot(311)
        # 2D waveform matrix
        ax0.matshow(
            tcur[:igood, disp_indx],
            cmap="seismic",
            extent=[tvec_disp[0], tvec_disp[-1], nwin, 0],
            aspect="auto",
        )
        ax0.plot([0, 0], [0, nwin], "k--", linewidth=2)
        ax0.set_title("%s, dist:%5.2fkm, filter @%4.2f-%4.2fHz" % (sfile.split("/")[-1], dist, freq1, freq2))
        ax0.set_xlabel("time [s]")
        ax0.set_ylabel("wavefroms")
        ax0.set_yticks(np.arange(0, nwin, step=tick_inc))
        # shade the coda part
        ax0.fill(
            np.concatenate((tvec_all[nwin_indx], np.flip(tvec_all[nwin_indx], axis=0)), axis=0),
            np.concatenate((np.ones(len(nwin_indx)) * 0, np.ones(len(nwin_indx)) * nwin), axis=0),
            "c",
            alpha=0.3,
            linewidth=1,
        )
        ax0.fill(
            np.concatenate((tvec_all[pwin_indx], np.flip(tvec_all[pwin_indx], axis=0)), axis=0),
            np.concatenate((np.ones(len(nwin_indx)) * 0, np.ones(len(nwin_indx)) * nwin), axis=0),
            "y",
            alpha=0.3,
        )
        ax0.xaxis.set_ticks_position("bottom")
        # reference waveform
        ax1 = plt.subplot(613)
        ax1.plot(tvec_disp, tref[disp_indx], "k-", linewidth=1)
        ax1.autoscale(enable=True, axis="x", tight=True)
        ax1.grid(True)
        ax1.legend(["reference"], loc="upper right")
        # the cross-correlation coefficient
        ax2 = plt.subplot(614)
        ax2.plot(timestamp[:igood], pcor_cc[:igood], "yo-", markersize=2, linewidth=1)
        ax2.plot(timestamp[:igood], ncor_cc[:igood], "co-", markersize=2, linewidth=1)
        ax2.set_xticks(timestamp[0:nwin:tick_inc])
        ax2.set_ylabel("cc coeff")
        ax2.legend(["positive", "negative"], loc="upper right")

        ###############################################
        ############ MONITORING PROCESSES #############
        ###############################################

        # allocate matrix for dvv and its unc
        dvv_stretch = np.zeros(shape=(nwin, 4), dtype=np.float32)
        dvv_dtw = np.zeros(shape=(nwin, 4), dtype=np.float32)
        dvv_mwcs = np.zeros(shape=(nwin, 4), dtype=np.float32)
        dvv_wcc = np.zeros(shape=(nwin, 4), dtype=np.float32)
        dvv_wts = np.zeros(shape=(nwin, 4), dtype=np.float32)
        dvv_wxs = np.zeros(shape=(nwin, 4), dtype=np.float32)

        # loop through each win again
        for ii in range(nwin):
            # casual and acasual lags for both ref and cur waveforms
            pcur = tcur[ii, pwin_indx]
            ncur = tcur[ii, nwin_indx]
            pref = tref[pwin_indx]
            nref = tref[nwin_indx]

            # functions working in time domain
            if do_stretch:
                (
                    dvv_stretch[ii, 0],
                    dvv_stretch[ii, 1],
                    cc,
                    cdp,
                ) = noise_module.stretching(pref, pcur, epsilon, nbtrial, para)
                (
                    dvv_stretch[ii, 2],
                    dvv_stretch[ii, 3],
                    cc,
                    cdp,
                ) = noise_module.stretching(nref, ncur, epsilon, nbtrial, para)
            if do_dtw:
                dvv_dtw[ii, 0], dvv_dtw[ii, 1], dist = noise_module.dtw_dvv(pref, pcur, para, mlag, b, direct)
                dvv_dtw[ii, 2], dvv_dtw[ii, 3], dist = noise_module.dtw_dvv(nref, ncur, para, mlag, b, direct)

            # check parameters for mwcs
            if move_win_sec > 0.5 * (np.max(twin) - np.min(twin)):
                raise IOError("twin too small for MWCS")

            # functions with moving window
            if do_mwcs:
                dvv_mwcs[ii, 0], dvv_mwcs[ii, 1] = noise_module.mwcs_dvv(pref, pcur, move_win_sec, step_sec, para)
                dvv_mwcs[ii, 2], dvv_mwcs[ii, 3] = noise_module.mwcs_dvv(nref, ncur, move_win_sec, step_sec, para)
            if do_mwcc:
                dvv_wcc[ii, 0], dvv_wcc[ii, 1] = noise_module.WCC_dvv(pref, pcur, move_win_sec, step_sec, para)
                dvv_wcc[ii, 2], dvv_wcc[ii, 3] = noise_module.WCC_dvv(pref, pcur, move_win_sec, step_sec, para)

            allfreq = False  # average dv/v over the frequency band for wts and wxs
            if do_wts:
                dvv_wts[ii, 0], dvv_wts[ii, 1] = noise_module.wts_allfreq(
                    pref, pcur, allfreq, para, epsilon, nbtrial, dj, s0, J, wvn
                )
                dvv_wts[ii, 2], dvv_wts[ii, 3] = noise_module.wts_allfreq(
                    nref, ncur, allfreq, para, epsilon, nbtrial, dj, s0, J, wvn
                )
            if do_wxs:
                dvv_wxs[ii, 0], dvv_wxs[ii, 1] = noise_module.wxs_allfreq(pref, pcur, allfreq, para, dj, s0, J)
                dvv_wxs[ii, 2], dvv_wxs[ii, 3] = noise_module.wxs_allfreq(nref, ncur, allfreq, para, dj, s0, J)

            """
            allfreq = True     # look at all frequency range
            para['freq'] = freq

            # functions in wavelet domain to compute dvv for all frequncy
            if do_wts:
                dfreq,dv_wts1,unc1 = noise_module.wts_allfreq(ref[pwin_indx],
                cur[pwin_indx],allfreq,para,epsilon,nbtrial,dj,s0,J,wvn)
                dfreq,dv_wts2,unc2 = noise_module.wts_allfreq(ref[nwin_indx],
                cur[nwin_indx],allfreq,para,epsilon,nbtrial,dj,s0,J,wvn)
            if do_wxs:
                dfreq,dv_wxs1,unc1 = noise_module.wxs_allfreq(ref[pwin_indx],
                cur[pwin_indx],allfreq,para,dj,s0,J)
                dfreq,dv_wxs2,unc2 = noise_module.wxs_allfreq(ref[nwin_indx],
                cur[nwin_indx],allfreq,para,dj,s0,J)
            """

        ###############################################
        ############ PLOTTING SECTION #################
        ###############################################

        # dv/v at each filtered frequency band
        ax3 = plt.subplot(313)
        legend_mark = []
        if do_stretch:
            ax3.plot(timestamp[:igood], dvv_stretch[:, 0], "yo-", markersize=6, linewidth=0.5)
            ax3.plot(timestamp[:igood], dvv_stretch[:, 2], "co-", markersize=6, linewidth=0.5)
            legend_mark.append("str+")
            legend_mark.append("str-")
        if do_dtw:
            ax3.plot(timestamp[:igood], dvv_dtw[:, 0], "yv-", markersize=6, linewidth=0.5)
            ax3.plot(timestamp[:igood], dvv_dtw[:, 2], "cv-", markersize=6, linewidth=0.5)
            legend_mark.append("dtw+")
            legend_mark.append("dtw-")
        if do_mwcs:
            ax3.plot(timestamp[:igood], dvv_mwcs[:, 0], "ys-", markersize=6, linewidth=0.5)
            ax3.plot(timestamp[:igood], dvv_mwcs[:, 2], "cs-", markersize=6, linewidth=0.5)
            legend_mark.append("mwcs+")
            legend_mark.append("mwcs-")
        if do_mwcc:
            ax3.plot(timestamp[:igood], dvv_wcc[:, 0], "y*-", markersize=6, linewidth=0.5)
            ax3.plot(timestamp[:igood], dvv_wcc[:, 2], "c*-", markersize=6, linewidth=0.5)
            legend_mark.append("wcc+")
            legend_mark.append("wcc-")
        if do_wts:
            ax3.plot(timestamp[:igood], dvv_wts[:, 0], "yx-", markersize=6, linewidth=0.5)
            ax3.plot(timestamp[:igood], dvv_wts[:, 2], "cx-", markersize=6, linewidth=0.5)
            legend_mark.append("wts+")
            legend_mark.append("wts-")
        if do_wxs:
            ax3.plot(timestamp[:igood], dvv_wxs[:, 0], "yp-", markersize=6, linewidth=0.5)
            ax3.plot(timestamp[:igood], dvv_wxs[:, 2], "cp-", markersize=6, linewidth=0.5)
            legend_mark.append("wxs+")
            legend_mark.append("wxs-")
        ax3.legend(legend_mark, loc="upper right")
        # ax3.grid('true')
        ax3.set_ylabel("dv/v [%]")

        # save figure or just show
        outfname = outdir + "/{0:s}_{1:4.2f}_{2:4.2f}Hz.pdf".format(sfile.split("/")[-1], freq1, freq2)
        plt.savefig(outfname, format="pdf", dpi=400)
        plt.close()
