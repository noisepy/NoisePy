import os

import matplotlib.pyplot as plt
import numpy as np
import pyasdf
import pycwt
import scipy

from . import noise_module

"""
this application script of NoisePy is to measure group velocity on the resulted cross-correlation
functions from S2. It uses the wavelet transform to trace the wave energy on multiple frequencies.
Based on our tests, it generates very similar results to those from Frequency-Time Analysis (FTAN).

Authors: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)

NOTE:
    According to Bensen et al., (2007), the central frequency of
    each narrowband filters (equivalent to wavelet tranformed signal
    at each scale) would be different from the instaneous frequency calculated
    using instaneous phase due to spectral linkage. We do not
    correct this effect in this script. Phase velocity is not calculated here,
    but could be expaneded using the phase info of wavelet transformed signal.
"""

############################################
############ PAMAETER SECTION ##############
############################################

# input file info
rootpath = os.path.join(os.path.expanduser("~"), "Documents/NoisePy_example/SCAL")  # root path for this data processing
sfile = os.path.join(rootpath, "STACK_month/CI.BLC/CI.BLC_CI.BTP.h5")  # ASDF file containing stacked data
outdir = os.path.join(rootpath, "figures/dispersion")  # dir where to output dispersive image and extracted dispersion

# data type and cross-component
stack_method = "linear"  # which stacked data to measure dispersion info
lag_type = "sym"  # options to do measurements on the 'neg', 'pos' or 'sym' lag (average of neg and pos)
ncomp = 3
if ncomp == 1:
    rtz_system = ["ZZ"]
else:
    rtz_system = ["ZR", "ZT", "ZZ", "RR", "RT", "RZ", "TR", "TT", "TZ"]
# index for plotting the figures
post1 = [0, 0, 0, 1, 1, 1, 2, 2, 2]
post2 = [0, 1, 2, 0, 1, 2, 0, 1, 2]

# targeted freq bands for dispersion analysis
fmin = 0.03
fmax = 1
per = np.arange(int(1 / fmax), int(1 / fmin), 0.02)

# set time window for disperion analysis
vmin = 0.5
vmax = 4.5
vel = np.arange(vmin, vmax, 0.02)

# basic parameters for wavelet transform
dj = 1 / 12
s0 = -1
J = -1
wvn = "morlet"

# get station-pair name ready for output
tmp = sfile.split("/")[-1].split("_")
spair = tmp[0] + "_" + tmp[1][:-3]

# load basic data information including dt, dist and maxlag
with pyasdf.ASDFDataSet(sfile, mode="r") as ds:
    dtype = "Allstack0" + stack_method
    try:
        maxlag = ds.auxiliary_data[dtype]["ZZ"].parameters["maxlag"]
        dist = ds.auxiliary_data[dtype]["ZZ"].parameters["dist"]
        dt = ds.auxiliary_data[dtype]["ZZ"].parameters["dt"]
    except Exception as e:
        raise ValueError(e)

# initialize the plotting procedure
if ncomp == 3:
    fig, ax = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
else:
    plt.figure(figsize=(4, 3))

##################################################
############ MEASURE GROUP VELOCITY ##############
##################################################

# loop through each component
for comp in rtz_system:
    cindx = rtz_system.index(comp)
    pos1 = post1[cindx]
    pos2 = post2[cindx]

    # load cross-correlation functions
    with pyasdf.ASDFDataSet(sfile, mode="r") as ds:
        try:
            tdata = ds.auxiliary_data[dtype][comp].data[:]
        except Exception as e:
            raise ValueError(e)

    # stack positive and negative lags
    npts = int(1 / dt) * 2 * maxlag + 1
    indx = npts // 2

    if lag_type == "neg":
        data = tdata[: indx + 1]
    elif lag_type == "pos":
        data = tdata[indx:]
    elif lag_type == "sym":
        data = 0.5 * tdata[indx:] + 0.5 * np.flip(tdata[: indx + 1], axis=0)
    else:
        raise ValueError("parameter of lag_type (L35) is not right! please double check")

    # trim the data according to vel window
    pt1 = int(dist / vmax / dt)
    pt2 = int(dist / vmin / dt)
    if pt1 == 0:
        pt1 = 10
    if pt2 > (npts // 2):
        pt2 = npts // 2
    indx = np.arange(pt1, pt2)
    tvec = indx * dt
    data = data[indx]

    # wavelet transformation
    cwt, sj, freq, coi, _, _ = pycwt.cwt(data, dt, dj, s0, J, wvn)

    # do filtering here
    if (fmax > np.max(freq)) | (fmax <= fmin):
        raise ValueError("Abort: frequency out of limits!")
    freq_ind = np.where((freq >= fmin) & (freq <= fmax))[0]
    cwt = cwt[freq_ind]
    freq = freq[freq_ind]

    # use amplitude of the cwt
    period = 1 / freq
    rcwt, pcwt = np.abs(cwt) ** 2, np.angle(cwt)

    # interpolation to grids of freq-vel
    fc = scipy.interpolate.interp2d(dist / tvec, period, rcwt)
    rcwt_new = fc(vel, per)

    # do normalization for each frequency
    for ii in range(len(per)):
        rcwt_new[ii] /= np.max(rcwt_new[ii])

    # extract dispersion curves for ZZ, RR and TT
    if comp == "ZZ" or comp == "RR" or comp == "TT":
        nper, gv = noise_module.extract_dispersion(rcwt_new, per, vel)
        fphase = open(os.path.join(outdir, spair + "_group_" + comp + ".csv"), "w")
        for iii in range(len(nper)):
            fphase.write("%5.1f %5.2f\n" % (nper[iii], gv[iii]))
        fphase.close()

    # plot wavelet spectrum
    if ncomp == 3:
        # dispersive image
        im = ax[pos1, pos2].imshow(
            np.transpose(rcwt_new),
            cmap="jet",
            extent=[per[0], per[-1], vel[0], vel[-1]],
            aspect="auto",
            origin="lower",
        )
        # extracted dispersion curves
        if comp == "ZZ" or comp == "RR" or comp == "TT":
            ax[pos1, pos2].plot(nper, gv, "w--")
        ax[pos1, pos2].set_xlabel("Period [s]")
        ax[pos1, pos2].set_ylabel("U [km/s]")
        if cindx == 1:
            ax[pos1, pos2].set_title("%s %5.2fkm linear" % (spair, dist))
        ax[pos1, pos2].xaxis.set_ticks_position("bottom")
        cbar = fig.colorbar(im, ax=ax[pos1, pos2])
        font = {"family": "serif", "color": "green", "weight": "bold", "size": 16}
        ax[pos1, pos2].text(int(per[-1] * 0.85), vel[-1] - 0.5, comp, fontdict=font)
    else:
        plt.imshow(
            np.transpose(rcwt_new),
            cmap="jet",
            extent=[per[0], per[-1], vel[0], vel[-1]],
            aspect="auto",
            origin="lower",
        )
        # extracted disperison curves
        plt.plot(nper, gv, "w--")
        plt.xlabel("Period [s]")
        plt.ylabel("U [km/s]")
        plt.title("%s %5.2fkm linear" % (spair, dist))
        font = {"family": "serif", "color": "green", "weight": "bold", "size": 16}
        plt.text(int(per[-1] * 0.85), vel[-1] - 0.5, comp, fontdict=font)
        plt.tight_layout()

# save figures
outfname = outdir + "/{0:s}_{1:s}.pdf".format(spair, lag_type)
if ncomp == 3:
    fig.tight_layout()
    fig.savefig(outfname, format="pdf", dpi=400)
    plt.close()
else:
    plt.savefig(outfname, format="pdf", dpi=400)
    plt.close()
