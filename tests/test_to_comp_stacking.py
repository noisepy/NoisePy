import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import pyasdf
import scipy
from obspy.signal.filter import bandpass
from scipy.fftpack import next_fast_len

from ..src.noisepy.seis import noise_module

# from scipy.signal import hilbert

"""
check the performance of all different stacking method for noise cross-correlations.

Chengxin Jiang @Harvard Oct/29/2019
Updated @May/2020 to include nth-root stacking, selective stacking and tf-PWS stacking
"""


####### PLEASE TURN THIS TO A PROPER TEST AND NOT A TUTORIAL
###########
#########

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
    spws = noise_module.pws(ndata, int(1 / dt))
    t2 = time.time()
    srobust, ww, nstep = noise_module.robust_stack(ndata, 0.001)
    t3 = time.time()
    sACF = noise_module.adaptive_filter(ndata, 1)
    t4 = time.time()
    nroot = noise_module.nroot_stack(ndata, 2)
    t5 = time.time()
    sstack, nstep = noise_module.selective_stack(ndata, 0.001, 0.01)
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
