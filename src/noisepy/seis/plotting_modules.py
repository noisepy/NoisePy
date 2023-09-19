import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import obspy
import pyasdf
import scipy
from datetimerange import DateTimeRange
from obspy.signal.filter import bandpass
from scipy.fftpack import next_fast_len

from noisepy.seis.stores import CrossCorrelationDataStore, StackStore

logging.getLogger("matplotlib.font_manager").disabled = True
logger = logging.getLogger(__name__)

"""
Ensembles of plotting functions to display intermediate/final waveforms from the NoisePy package.
by Chengxin Jiang @Harvard (May.04.2019)

Specifically, this plotting module includes functions of:
    1) plot_waveform     -> display the downloaded waveform for specific station
    2) plot_substack_cc  -> plot 2D matrix of the CC functions for one time-chunck (e.g., 2 days)
    3) plot_substack_all -> plot 2D matrix of the CC functions for all time-chunck (e.g., every 1 day in 1 year)
    4) plot_all_moveout  -> plot the moveout of the stacked CC functions for all time-chunk
"""


#############################################################################
# #############PLOTTING FUNCTIONS FOR FILES FROM S0##########################
#############################################################################
def plot_waveform(sfile, net, sta, freqmin, freqmax, savefig=False, sdir=None):
    """
    display the downloaded waveform for station A

    PARAMETERS:
    -----------------------
    sfile: containing all wavefrom data for a time-chunck in ASDF format
    net,sta,comp: network, station name and component
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered

    USAGE:
    -----------------------
    plot_waveform('temp.h5','CI','BLC',0.01,0.5)
    """
    # open pyasdf file to read
    try:
        ds = pyasdf.ASDFDataSet(sfile, mode="r")
        sta_list = ds.waveforms.list()
    except Exception:
        raise Exception("exit! cannot open %s to read" % sfile)

    # check whether station exists
    tsta = net + "." + sta
    if tsta not in sta_list:
        raise ValueError("no data for %s in %s" % (tsta, sfile))

    tcomp = ds.waveforms[tsta].get_waveform_tags()
    ncomp = len(tcomp)
    if ncomp == 1:
        tr = ds.waveforms[tsta][tcomp[0]]
        dt = tr[0].stats.delta
        npts = tr[0].stats.npts
        tt = np.arange(0, npts) * dt
        data = tr[0].data
        data = bandpass(data, freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
        plt.figure(figsize=(9, 3))
        plt.plot(tt, data, "k-", linewidth=1)
        plt.title(
            "T\u2080:%s   %s.%s.%s   @%5.3f-%5.2f Hz"
            % (
                tr[0].stats.starttime,
                net,
                sta,
                tcomp[0].split("_")[0].upper(),
                freqmin,
                freqmax,
            )
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show()
    elif ncomp == 3:
        tr = ds.waveforms[tsta][tcomp[0]]
        dt = tr[0].stats.delta
        npts = tr[0].stats.npts
        tt = np.arange(0, npts) * dt
        data = np.zeros(shape=(ncomp, npts), dtype=np.float32)
        for ii in range(ncomp):
            data[ii] = ds.waveforms[tsta][tcomp[ii]][0].data
            data[ii] = bandpass(data[ii], freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
        plt.figure(figsize=(9, 6))
        plt.subplot(311)
        plt.plot(tt, data[0], "k-", linewidth=1)
        plt.title("T\u2080:%s   %s.%s   @%5.3f-%5.2f Hz" % (tr[0].stats.starttime, net, sta, freqmin, freqmax))
        plt.legend([tcomp[0].split("_")[0].upper()], loc="upper left")
        plt.subplot(312)
        plt.plot(tt, data[1], "k-", linewidth=1)
        plt.legend([tcomp[1].split("_")[0].upper()], loc="upper left")
        plt.subplot(313)
        plt.plot(tt, data[2], "k-", linewidth=1)
        plt.legend([tcomp[2].split("_")[0].upper()], loc="upper left")
        plt.xlabel("Time [s]")
        plt.tight_layout()

        if savefig:
            if not os.path.isdir(sdir):
                os.mkdir(sdir)
            outfname = sdir + "/{0:s}_{1:s}.{2:s}.pdf".format(sfile.split(".")[0], net, sta)
            plt.savefig(outfname, format="pdf", dpi=400)
            plt.close()
        else:
            plt.show()


#############################################################################
# #############PLOTTING FUNCTIONS FOR FILES FROM S1##########################
#############################################################################


def plot_substack_cc(
    cc_store: CrossCorrelationDataStore,
    ts: DateTimeRange,
    freqmin,
    freqmax,
    disp_lag=None,
    savefig=True,
    sdir="./",
):
    """
    display the 2D matrix of the cross-correlation functions for a certain time-chunck.

    PARAMETERS:
    --------------------------
    cc_store: Store to read CC data from
    ts: Timespan to ploe
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    disp_lag: time ranges for display
    savefg: Whether to save the figures as a PDF on disk
    sdir: Save directory

    Note: IMPORTANT!!!! this script only works for cross-correlation with sub-stacks being set to True in S1.
    """
    # open data for read
    if savefig:
        if sdir is None:
            raise ValueError("sdir argument must be provided if savefig=True")

    sta_pairs = cc_store.get_station_pairs()
    if len(sta_pairs) == 0:
        logger.error("No data available for plotting")
        return

    ccs = cc_store.read_correlations(ts, sta_pairs[0][0], sta_pairs[0][1])
    if len(ccs) == 0:
        logger.error(f"No data available for plotting in {ts}/{sta_pairs[0]}")
        return

    # Read some common arguments from the first available data set:
    params = ccs[0].parameters
    substack_flag, dt, maxlag = (params[p] for p in ["substack", "dt", "maxlag"])

    # only works for cross-correlation with substacks generated
    if not substack_flag:
        raise ValueError("seems no substacks have been done! not suitable for this plotting function")

    # lags for display
    if not disp_lag:
        disp_lag = maxlag
    if disp_lag > maxlag:
        raise ValueError("lag excceds maxlag!")

    # t is the time labels for plotting
    t = np.arange(-int(disp_lag), int(disp_lag) + dt, step=int(2 * int(disp_lag) / 4))
    # windowing the data
    indx1 = int((maxlag - disp_lag) / dt)
    indx2 = indx1 + 2 * int(disp_lag / dt) + 1

    # for spair in spairs:
    for src_sta, rec_sta in sta_pairs:
        for cc in ccs:
            src_cha, rec_cha, params, all_data = cc.src, cc.rec, cc.parameters, cc.data
            try:
                dist, ngood, ttime = (params[p] for p in ["dist", "ngood", "time"])
                timestamp = np.empty(len(ttime), dtype="datetime64[s]")
            except Exception as e:
                logger.warning(f"continue! something wrong with {src_sta}_{rec_sta}/{src_cha}_{rec_cha}: {e}")
                continue

            # cc matrix
            data = all_data[:, indx1:indx2]
            nwin = data.shape[0]
            amax = np.zeros(nwin, dtype=np.float32)
            if nwin == 0 or len(ngood) == 1:
                print("continue! no enough substacks!")
                continue

            tmarks = []
            # load cc for each station-pair
            for ii in range(nwin):
                data[ii] = bandpass(data[ii], freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
                amax[ii] = max(data[ii])
                data[ii] /= amax[ii]
                timestamp[ii] = obspy.UTCDateTime(ttime[ii])
                tmarks.append(obspy.UTCDateTime(ttime[ii]).strftime("%H:%M:%S"))

            # plotting
            if nwin > 10:
                tick_inc = int(nwin / 5)
            else:
                tick_inc = 2
            fig, (
                ax1,
                ax2,
            ) = plt.subplots(2, 1, gridspec_kw={"height_ratios": [1, 3]}, figsize=(10, 6))
            ax1.matshow(
                data,
                cmap="seismic",
                extent=[-disp_lag, disp_lag, nwin, 0],
                aspect="auto",
            )
            ax1.set_title(f"{src_sta}.{src_cha} {rec_sta}.{rec_cha}  dist:{dist:5.2f}km")
            ax1.set_xlabel("time [s]")
            ax1.set_xticks(t)
            ax1.set_yticks(np.arange(0, nwin, step=tick_inc))
            ax1.set_yticklabels(timestamp[0::tick_inc])
            ax1.xaxis.set_ticks_position("bottom")
            ax2.set_title("stacked and filtered at %4.2f-%4.2f Hz" % (freqmin, freqmax))
            ax2.plot(
                np.arange(-disp_lag, disp_lag + dt, dt),
                np.mean(data, axis=0),
                "k-",
                linewidth=1,
            )
            ax2.set_xticks(t)
            fig.tight_layout()

            # save figure or just show
            if savefig:
                if not os.path.isdir(sdir):
                    os.mkdir(sdir)
                outfname = os.path.join(sdir, f"{src_sta}.{src_cha}_{rec_sta}.{rec_cha}.pdf")
                fig.savefig(outfname, format="pdf", dpi=400)
                plt.close()
            else:
                plt.show()


def plot_substack_cc_spect(sfile, freqmin, freqmax, disp_lag=None, savefig=True, sdir="./"):
    """
    display the 2D matrix of the cross-correlation functions for a time-chunck.

    PARAMETERS:
    -----------------------
    sfile: cross-correlation functions outputed by S1
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    disp_lag: time ranges for display

    USAGE:
    -----------------------
    plot_substack_cc('temp.h5',0.1,1,200,True,'./')

    Note: IMPORTANT!!!! this script only works for the cross-correlation with sub-stacks in S1.
    """
    # open data for read
    if savefig:
        if sdir is None:
            print("no path selected! save figures in the default path")

    try:
        ds = pyasdf.ASDFDataSet(sfile, mode="r")
        # extract common variables
        spairs = ds.auxiliary_data.list()
        path_lists = ds.auxiliary_data[spairs[0]].list()
        flag = ds.auxiliary_data[spairs[0]][path_lists[0]].parameters["substack"]
        dt = ds.auxiliary_data[spairs[0]][path_lists[0]].parameters["dt"]
        maxlag = ds.auxiliary_data[spairs[0]][path_lists[0]].parameters["maxlag"]
    except Exception:
        raise Exception("exit! cannot open %s to read" % sfile)

    # only works for cross-correlation with substacks generated
    if not flag:
        raise ValueError("seems no substacks have been done! not suitable for this plotting function")

    # lags for display
    if not disp_lag:
        disp_lag = maxlag
    if disp_lag > maxlag:
        raise ValueError("lag excceds maxlag!")
    t = np.arange(-int(disp_lag), int(disp_lag) + dt, step=int(2 * int(disp_lag) / 4))
    indx1 = int((maxlag - disp_lag) / dt)
    indx2 = indx1 + 2 * int(disp_lag / dt) + 1
    nfft = int(next_fast_len(indx2 - indx1))
    freq = scipy.fftpack.fftfreq(nfft, d=dt)[: nfft // 2]

    for spair in spairs:
        ttr = spair.split("_")
        net1, sta1 = ttr[0].split(".")
        net2, sta2 = ttr[1].split(".")
        for ipath in path_lists:
            chan1, chan2 = ipath.split("_")
            try:
                dist = ds.auxiliary_data[spair][ipath].parameters["dist"]
                ngood = ds.auxiliary_data[spair][ipath].parameters["ngood"]
                ttime = ds.auxiliary_data[spair][ipath].parameters["time"]
                timestamp = np.empty(ttime.size, dtype="datetime64[s]")
            except Exception:
                print("continue! something wrong with %s %s" % (spair, ipath))
                continue

            # cc matrix
            data = ds.auxiliary_data[spair][ipath].data[:, indx1:indx2]
            nwin = data.shape[0]
            amax = np.zeros(nwin, dtype=np.float32)
            spec = np.zeros(shape=(nwin, nfft // 2), dtype=np.complex64)
            if nwin == 0 or len(ngood) == 1:
                print("continue! no enough substacks!")
                continue

            # load cc for each station-pair
            for ii in range(nwin):
                spec[ii] = scipy.fftpack.fft(data[ii], nfft, axis=0)[: nfft // 2]
                spec[ii] /= np.max(np.abs(spec[ii]), axis=0)
                data[ii] = bandpass(data[ii], freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
                amax[ii] = max(data[ii])
                data[ii] /= amax[ii]
                timestamp[ii] = obspy.UTCDateTime(ttime[ii])

            # plotting
            if nwin > 10:
                tick_inc = int(nwin / 5)
            else:
                tick_inc = 2
            fig, ax = plt.subplots(3, sharex=False)
            ax[0].matshow(
                data,
                cmap="seismic",
                extent=[-disp_lag, disp_lag, nwin, 0],
                aspect="auto",
            )
            ax[0].set_title("%s.%s.%s  %s.%s.%s  dist:%5.2f km" % (net1, sta1, chan1, net2, sta2, chan2, dist))
            ax[0].set_xlabel("time [s]")
            ax[0].set_xticks(t)
            ax[0].set_yticks(np.arange(0, nwin, step=tick_inc))
            ax[0].set_yticklabels(timestamp[0::tick_inc])
            ax[0].xaxis.set_ticks_position("bottom")
            ax[1].matshow(
                np.abs(spec),
                cmap="seismic",
                extent=[freq[0], freq[-1], nwin, 0],
                aspect="auto",
            )
            ax[1].set_xlabel("freq [Hz]")
            ax[1].set_ylabel("amplitudes")
            ax[1].set_yticks(np.arange(0, nwin, step=tick_inc))
            ax[1].xaxis.set_ticks_position("bottom")
            ax[2].plot(amax / min(amax), "r-")
            ax[2].plot(ngood, "b-")
            ax[2].set_xlabel("waveform number")
            # ax[1].set_xticks(np.arange(0,nwin,int(nwin/5)))
            ax[2].legend(["relative amp", "ngood"], loc="upper right")
            fig.tight_layout()

            # save figure or just show
            if savefig:
                if sdir is None:
                    sdir = sfile.split(".")[0]
                if not os.path.isdir(sdir):
                    os.mkdir(sdir)
                outfname = sdir + "/{0:s}.{1:s}.{2:s}_{3:s}.{4:s}.{5:s}.pdf".format(
                    net1, sta1, chan1, net2, sta2, chan2
                )
                fig.savefig(outfname, format="pdf", dpi=400)
                plt.close()
            else:
                plt.show()


#############################################################################
# #############PLOTTING FUNCTIONS FOR FILES FROM S2##########################
#############################################################################


def plot_substack_all(
    sfile,
    freqmin,
    freqmax,
    ccomp,
    disp_lag=None,
    savefig=False,
    sdir=None,
    figsize=(14, 14),
):
    """
    display the 2D matrix of the cross-correlation functions stacked for all time windows.

    PARAMETERS:
    ---------------------
    sfile: cross-correlation functions outputed by S2
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    disp_lag: time ranges for display
    ccomp: cross component of the targeted cc functions

    USAGE:
    ----------------------
    plot_substack_all('temp.h5',0.1,1,'ZZ',50,True,'./')
    """
    # open data for read
    if savefig:
        if sdir is None:
            print("no path selected! save figures in the default path")

    paths = ccomp
    try:
        ds = pyasdf.ASDFDataSet(sfile, mode="r")
        # extract common variables
        dtype_lists = ds.auxiliary_data.list()
        dt = ds.auxiliary_data[dtype_lists[0]][paths].parameters["dt"]
        dist = ds.auxiliary_data[dtype_lists[0]][paths].parameters["dist"]
        maxlag = ds.auxiliary_data[dtype_lists[0]][paths].parameters["maxlag"]
    except Exception:
        raise Exception("exit! cannot open %s to read" % sfile)

    if len(dtype_lists) == 1:
        raise ValueError("Abort! seems no substacks have been done")

    # lags for display
    if not disp_lag:
        disp_lag = maxlag
    if disp_lag > maxlag:
        raise ValueError("lag excceds maxlag!")
    t = np.arange(-int(disp_lag), int(disp_lag) + dt, step=int(2 * int(disp_lag) / 4))
    indx1 = int((maxlag - disp_lag) / dt)
    indx2 = indx1 + 2 * int(disp_lag / dt) + 1

    # other parameters to keep
    num_stacks = len([itype for itype in dtype_lists if "stack" in itype])
    nwin = len(dtype_lists) - num_stacks
    data = np.zeros(shape=(nwin, indx2 - indx1), dtype=np.float32)
    ngood = np.zeros(nwin, dtype=np.int16)
    ttime = np.zeros(nwin, dtype=np.int)
    timestamp = np.empty(ttime.size, dtype="datetime64[s]")
    amax = np.zeros(nwin, dtype=np.float32)

    for ii, itype in enumerate(dtype_lists[num_stacks:]):
        if "Allstack" in itype:
            continue
        timestamp[ii] = obspy.UTCDateTime(np.float(itype[1:]))
        try:
            ngood[ii] = ds.auxiliary_data[itype][paths].parameters["ngood"]
            ttime[ii] = ds.auxiliary_data[itype][paths].parameters["time"]
            # timestamp[ii] = obspy.UTCDateTime(ttime[ii])
            # cc matrix
            data[ii] = ds.auxiliary_data[itype][paths].data[indx1:indx2]
            data[ii] = bandpass(data[ii], freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
            amax[ii] = np.max(data[ii])
            data[ii] /= amax[ii]
        except Exception as e:
            print(e)
            continue

        if len(ngood) == 1:
            raise ValueError("seems no substacks have been done! not suitable for this plotting function")

    # plotting
    if nwin > 100:
        tick_inc = int(nwin / 10)
    elif nwin > 10:
        tick_inc = int(nwin / 5)
    else:
        tick_inc = 2
    fig, ax = plt.subplots(2, sharex=False, figsize=figsize)
    ax[0].matshow(data, cmap="seismic", extent=[-disp_lag, disp_lag, nwin, 0], aspect="auto")
    ax[0].set_title("%s dist:%5.2f km filtered at %4.2f-%4.2fHz" % (sfile.split("/")[-1], dist, freqmin, freqmax))
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("wavefroms")
    ax[0].set_xticks(t)
    ax[0].set_yticks(np.arange(0, nwin, step=tick_inc))
    ax[0].set_yticklabels(timestamp[0:nwin:tick_inc])
    ax[0].xaxis.set_ticks_position("bottom")
    ax[1].plot(amax / max(amax), "r-")
    ax2 = ax[1].twinx()
    ax2.plot(ngood, "b-")
    ax2.set_ylabel("ngood", color="b")
    ax[1].set_ylabel("relative amp", color="r")
    ax[1].set_xlabel("waveform number")
    ax[1].set_xticks(np.arange(0, nwin, nwin // 5))
    ax[1].legend(["relative amp", "ngood"], loc="upper right")
    # save figure or just show
    if savefig:
        if sdir is None:
            sdir = sfile.split(".")[0]
        if not os.path.isdir(sdir):
            os.mkdir(sdir)
        outfname = sdir + "/{0:s}_{1:4.2f}_{2:4.2f}Hz.pdf".format(sfile.split("/")[-1], freqmin, freqmax)
        fig.savefig(outfname, format="pdf", dpi=400)
        plt.close()
    else:
        plt.show()


def plot_substack_all_spect(
    sfile,
    freqmin,
    freqmax,
    ccomp,
    disp_lag=None,
    savefig=False,
    sdir=None,
    figsize=(14, 14),
):
    """
    display the 2D matrix of the cross-correlation functions stacked for all time windows.

    PARAMETERS:
    -----------------------
    sfile: cross-correlation functions outputed by S2
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    disp_lag: time ranges for display
    ccomp: cross component of the targeted cc functions

    USAGE:
    -----------------------
    plot_substack_all('temp.h5',0.1,1,'ZZ',50,True,'./')
    """
    # open data for read
    if savefig:
        if sdir is None:
            print("no path selected! save figures in the default path")

    paths = ccomp
    try:
        ds = pyasdf.ASDFDataSet(sfile, mode="r")
        # extract common variables
        dtype_lists = ds.auxiliary_data.list()
        dt = ds.auxiliary_data[dtype_lists[0]][paths].parameters["dt"]
        dist = ds.auxiliary_data[dtype_lists[0]][paths].parameters["dist"]
        maxlag = ds.auxiliary_data[dtype_lists[0]][paths].parameters["maxlag"]
    except Exception:
        raise Exception("exit! cannot open %s to read" % sfile)

    if len(dtype_lists) == 1:
        raise ValueError("Abort! seems no substacks have been done")

    # lags for display
    if not disp_lag:
        disp_lag = maxlag
    if disp_lag > maxlag:
        raise ValueError("lag excceds maxlag!")
    t = np.arange(-int(disp_lag), int(disp_lag) + dt, step=int(2 * int(disp_lag) / 4))
    indx1 = int((maxlag - disp_lag) / dt)
    indx2 = indx1 + 2 * int(disp_lag / dt) + 1
    nfft = int(next_fast_len(indx2 - indx1))
    freq = scipy.fftpack.fftfreq(nfft, d=dt)[: nfft // 2]

    # other parameters to keep
    num_stacks = len([itype for itype in dtype_lists if "stack" in itype])
    nwin = len(dtype_lists) - num_stacks
    data = np.zeros(shape=(nwin, indx2 - indx1), dtype=np.float32)
    spec = np.zeros(shape=(nwin, nfft // 2), dtype=np.complex64)
    ngood = np.zeros(nwin, dtype=np.int16)
    ttime = np.zeros(nwin, dtype=np.int)
    timestamp = np.empty(ttime.size, dtype="datetime64[s]")
    amax = np.zeros(nwin, dtype=np.float32)

    for ii, itype in enumerate(dtype_lists[num_stacks:]):
        if "stack" in itype:
            continue
        timestamp[ii] = obspy.UTCDateTime(np.float(itype[1:]))
        try:
            ngood[ii] = ds.auxiliary_data[itype][paths].parameters["ngood"]
            ttime[ii] = ds.auxiliary_data[itype][paths].parameters["time"]
            # timestamp[ii] = obspy.UTCDateTime(ttime[ii])
            # cc matrix
            tdata = ds.auxiliary_data[itype][paths].data[indx1:indx2]
            spec[ii] = scipy.fftpack.fft(tdata, nfft, axis=0)[: nfft // 2]
            spec[ii] /= np.max(np.abs(spec[ii]))
            data[ii] = bandpass(tdata, freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
            amax[ii] = np.max(data[ii])
            data[ii] /= amax[ii]
        except Exception as e:
            print(e)
            continue

        if len(ngood) == 1:
            raise ValueError("seems no substacks have been done! not suitable for this plotting function")

    # plotting
    tick_inc = 50
    fig, ax = plt.subplots(3, sharex=False, figsize=figsize)
    ax[0].matshow(data, cmap="seismic", extent=[-disp_lag, disp_lag, nwin, 0], aspect="auto")
    ax[0].set_title("%s dist:%5.2f km" % (sfile.split("/")[-1], dist))
    ax[0].set_xlabel("time [s]")
    ax[0].set_ylabel("wavefroms")
    ax[0].set_xticks(t)
    ax[0].set_yticks(np.arange(0, nwin, step=tick_inc))
    ax[0].set_yticklabels(timestamp[0:nwin:tick_inc])
    ax[0].xaxis.set_ticks_position("bottom")
    ax[1].matshow(np.abs(spec), cmap="seismic", extent=[freq[0], freq[-1], nwin, 0], aspect="auto")
    ax[1].set_xlabel("freq [Hz]")
    ax[1].set_ylabel("amplitudes")
    ax[1].set_yticks(np.arange(0, nwin, step=tick_inc))
    ax[1].set_yticklabels(timestamp[0:nwin:tick_inc])
    ax[1].xaxis.set_ticks_position("bottom")
    ax[2].plot(amax / max(amax), "r-")
    ax[2].plot(ngood, "b-")
    ax[2].set_xlabel("waveform number")
    ax[2].set_xticks(np.arange(0, nwin, nwin // 15))
    ax[2].legend(["relative amp", "ngood"], loc="upper right")
    # save figure or just show
    if savefig:
        if sdir is None:
            sdir = sfile.split(".")[0]
        if not os.path.isdir(sdir):
            os.mkdir(sdir)
        outfname = sdir + "/{0:s}.pdf".format(sfile.split("/")[-1])
        fig.savefig(outfname, format="pdf", dpi=400)
        plt.close()
    else:
        plt.show()


def plot_all_moveout(
    store: StackStore,
    stack_name,
    freqmin,
    freqmax,
    ccomp,
    dist_inc,
    disp_lag=None,
    savefig=False,
    sdir=None,
):
    """
    display the moveout (2D matrix) of the cross-correlation functions stacked for all time chuncks.

    PARAMETERS:
    ---------------------
    store: StackStore to read stacked data
    stack_name: datatype either 'Allstack0pws' or 'Allstack_linear'
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    ccomp:   cross component
    dist_inc: distance bins to stack over
    disp_lag: lag times for displaying
    savefig: set True to save the figures (in pdf format)
    sdir: diresied directory to save the figure (if not provided, save to default dir)

    USAGE:
    ----------------------
    plot_substack_moveout(store,'Allstack_pws',0.1,0.2,1,'ZZ',200,True,'./temp')
    """
    # open data for read
    if savefig:
        if sdir is None:
            raise ValueError("sdir argument must be provided if savefig=True")

    sta_pairs = store.get_station_pairs()
    if len(sta_pairs) == 0:
        logger.error("No data available for plotting")
        return

    # Read some common arguments from the first available data set:
    stacks = store.read_stacks(sta_pairs[0][0], sta_pairs[0][1])
    dtmp = stacks[0].data
    params = stacks[0].parameters
    if len(params) == 0 or dtmp.size == 0:
        logger.error(f"No data available for plotting {stack_name}/{ccomp}")
        return

    dt, maxlag = (params[p] for p in ["dt", "maxlag"])
    stack_method = stack_name.split("0")[-1]
    print(stack_name, stack_method)

    # lags for display
    if not disp_lag:
        disp_lag = maxlag
    if disp_lag > maxlag:
        raise ValueError("lag excceds maxlag!")
    t = np.arange(-int(disp_lag), int(disp_lag) + dt, step=(int(2 * int(disp_lag) / 4)))
    indx1 = int((maxlag - disp_lag) / dt)
    indx2 = indx1 + 2 * int(disp_lag / dt) + 1

    # cc matrix
    nwin = len(sta_pairs)
    data = np.zeros(shape=(nwin, indx2 - indx1), dtype=np.float32)
    dist = np.zeros(nwin, dtype=np.float32)
    ngood = np.zeros(nwin, dtype=np.int16)

    # load cc and parameter matrix
    for ii, (src, rec) in enumerate(sta_pairs):
        stacks = store.read_stacks(src, rec)
        stacks = list(filter(lambda x: x.name == stack_name and x.component == ccomp, stacks))
        if len(stacks) == 0:
            logger.warning(f"No data available for {src}_{rec}/{stack_name}/{ccomp}")
            continue
        params, all_data = stacks[0].parameters, stacks[0].data
        dist[ii] = params["dist"]
        ngood[ii] = params["ngood"]
        crap = bandpass(all_data, freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
        data[ii] = crap[indx1:indx2]

    # average cc
    ntrace = int(np.round(np.max(dist) + 0.51) / dist_inc)
    ndata = np.zeros(shape=(ntrace, indx2 - indx1), dtype=np.float32)
    ndist = np.zeros(ntrace, dtype=np.float32)
    for td in range(0, ntrace - 1):
        tindx = np.where((dist >= td * dist_inc) & (dist < (td + 1) * dist_inc))[0]
        if len(tindx):
            ndata[td] = np.mean(data[tindx], axis=0)
            ndist[td] = (td + 0.5) * dist_inc

    # normalize waveforms
    indx = np.where(ndist > 0)[0]
    ndata = ndata[indx]
    ndist = ndist[indx]
    for ii in range(ndata.shape[0]):
        ndata[ii] /= np.max(np.abs(ndata[ii]))

    if ndata.shape[0] >= 10:
        # plotting figures
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(ndata, cmap="RdBu", extent=[-disp_lag, disp_lag, ndist[0], ndist[-1]], aspect="auto", origin="lower")
        ax.set_title("stacked %s (%5.3f-%5.2f Hz)" % (stack_method, freqmin, freqmax))
        ax.set_xlabel("time [s]")
        ax.set_ylabel("distance [km]")
        ax.set_xticks(t)
        ax.xaxis.set_ticks_position("bottom")
        # ax.text(np.ones(len(ndist))*(disp_lag-5),dist[ndist],ngood[ndist],fontsize=8)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        print(disp_lag, data.shape[1])
        tt = 2 * np.linspace(0, disp_lag, data.shape[1]) - disp_lag
        for ii in range(len(data)):
            ax.plot(tt, data[ii] / np.max(np.abs(data[ii])) * 10 + dist[ii], "k")
            ax.set_title("stacked %s (%5.3f-%5.2f Hz)" % (stack_method, freqmin, freqmax))
            ax.set_xlabel("time [s]")
            ax.set_ylabel("distance [km]")
            ax.set_xticks(t)
            ax.xaxis.set_ticks_position("bottom")

    # save figure or show
    if savefig:
        outfname = sdir + "/moveout_stack_" + str(stack_method) + "_" + str(dist_inc) + "kmbin.pdf"
        fig.savefig(outfname, format="pdf", dpi=400)
        plt.close()
    else:
        plt.show()


def plot_all_moveout_1D_1comp(
    sfiles,
    sta,
    dtype,
    freqmin,
    freqmax,
    ccomp,
    disp_lag=None,
    savefig=False,
    sdir=None,
    figsize=(14, 11),
):
    """
    display the moveout waveforms of the cross-correlation functions stacked for all time chuncks.

    PARAMETERS:
    ---------------------
    sfile: cross-correlation functions outputed by S2
    sta: source station name
    dtype: datatype either 'Allstack0pws' or 'Allstack0linear'
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    ccomp:   cross component
    disp_lag: lag times for displaying
    savefig: set True to save the figures (in pdf format)
    sdir: diresied directory to save the figure (if not provided, save to default dir)

    USAGE:
    ----------------------
    plot_substack_moveout('temp.h5','Allstack0pws',0.1,0.2,'ZZ',200,True,'./temp')
    """
    # open data for read
    if savefig:
        if sdir is None:
            print("no path selected! save figures in the default path")

    receiver = sta + ".h5"
    stack_method = dtype.split("_")[-1]

    # extract common variables
    try:
        ds = pyasdf.ASDFDataSet(sfiles[0], mode="r")
        dt = ds.auxiliary_data[dtype][ccomp].parameters["dt"]
        maxlag = ds.auxiliary_data[dtype][ccomp].parameters["maxlag"]
    except Exception:
        raise Exception("exit! cannot open %s to read" % sfiles[0])

    # lags for display
    if not disp_lag:
        disp_lag = maxlag
    if disp_lag > maxlag:
        raise ValueError("lag excceds maxlag!")
    tt = np.arange(-int(disp_lag), int(disp_lag) + dt, dt)
    indx1 = int((maxlag - disp_lag) / dt)
    indx2 = indx1 + 2 * int(disp_lag / dt) + 1

    # load cc and parameter matrix
    mdist = 0
    if not figsize:
        plt.figure()
    else:
        plt.figure(figsize=figsize)
    for ii in range(len(sfiles)):
        sfile = sfiles[ii]
        iflip = 0
        treceiver = sfile.split("_")[-1]
        if treceiver == receiver:
            iflip = 1

        ds = pyasdf.ASDFDataSet(sfile, mode="r")
        try:
            # load data to variables
            dist = ds.auxiliary_data[dtype][ccomp].parameters["dist"]
            tdata = ds.auxiliary_data[dtype][ccomp].data[indx1:indx2]

        except Exception:
            print("continue! cannot read %s " % sfile)
            continue

        tdata = bandpass(tdata, freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
        tdata /= np.max(tdata, axis=0)

        if iflip:
            plt.plot(tt, np.flip(tdata, axis=0) + dist, "k", linewidth=0.8)
        else:
            plt.plot(tt, tdata + dist, "k", linewidth=0.8)
        plt.title("%s %s filtered @%4.1f-%4.1f Hz" % (sta, ccomp, freqmin, freqmax))
        plt.xlabel("time (s)")
        plt.ylabel("offset (km)")
        plt.text(disp_lag * 0.9, dist + 0.5, receiver, fontsize=6)

        # ----use to plot o times------
        if mdist < dist:
            mdist = dist
    plt.plot([0, 0], [0, mdist], "r--", linewidth=1)

    # save figure or show
    if savefig:
        outfname = sdir + "/moveout_" + sta + "_1D_" + str(stack_method) + ".pdf"
        plt.savefig(outfname, format="pdf", dpi=400)
        plt.close()
    else:
        plt.show()


def plot_all_moveout_1D_9comp(
    sfiles,
    sta,
    dtype,
    freqmin,
    freqmax,
    mdist,
    disp_lag=None,
    savefig=False,
    sdir=None,
    figsize=(14, 11),
):
    """
    display the moveout waveforms of the cross-correlation functions stacked for all time chuncks.

    PARAMETERS:
    ---------------------
    sfile: cross-correlation functions outputed by S2
    sta: source station name
    dtype: datatype either 'Allstack0pws' or 'Allstack0linear'
    freqmin: min frequency to be filtered
    freqmax: max frequency to be filtered
    mdist: maximum inter-station distance to show on plot
    disp_lag: lag times for displaying
    savefig: set True to save the figures (in pdf format)
    sdir: diresied directory to save the figure (if not provided, save to default dir)

    USAGE:
    ----------------------
    plot_substack_moveout('temp.h5','Allstack0pws',0.1,0.2,'ZZ',200,True,'./temp')
    """
    # open data for read
    if savefig:
        if sdir is None:
            print("no path selected! save figures in the default path")

    receiver = sta + ".h5"
    stack_method = dtype.split("_")[-1]
    ccomp = ["ZR", "ZT", "ZZ", "RR", "RT", "RZ", "TR", "TT", "TZ"]

    # extract common variables
    try:
        ds = pyasdf.ASDFDataSet(sfiles[0], mode="r")
        dt = ds.auxiliary_data[dtype][ccomp[0]].parameters["dt"]
        maxlag = ds.auxiliary_data[dtype][ccomp[0]].parameters["maxlag"]
    except Exception:
        raise Exception("exit! cannot open %s to read" % sfiles[0])

    # lags for display
    if not disp_lag:
        disp_lag = maxlag
    if disp_lag > maxlag:
        raise ValueError("lag excceds maxlag!")
    tt = np.arange(-int(disp_lag), int(disp_lag) + dt, dt)
    indx1 = int((maxlag - disp_lag) / dt)
    indx2 = indx1 + 2 * int(disp_lag / dt) + 1

    # load cc and parameter matrix
    if not figsize:
        plt.figure()
    else:
        plt.figure(figsize=figsize)
    for ic in range(len(ccomp)):
        comp = ccomp[ic]
        tmp = "33" + str(ic + 1)
        plt.subplot(tmp)

        for ii in range(len(sfiles)):
            sfile = sfiles[ii]
            iflip = 0
            treceiver = sfile.split("_")[-1]
            if treceiver == receiver:
                iflip = 1

            ds = pyasdf.ASDFDataSet(sfile, mode="r")
            try:
                # load data to variables
                dist = ds.auxiliary_data[dtype][comp].parameters["dist"]
                tdata = ds.auxiliary_data[dtype][comp].data[indx1:indx2]

            except Exception:
                print("continue! cannot read %s " % sfile)
                continue

            if dist > mdist:
                continue
            tdata = bandpass(tdata, freqmin, freqmax, int(1 / dt), corners=4, zerophase=True)
            tdata /= np.max(tdata, axis=0)

            if iflip:
                plt.plot(tt, np.flip(tdata, axis=0) + dist, "k", linewidth=0.8)
            else:
                plt.plot(tt, tdata + dist, "k", linewidth=0.8)
            if ic == 1:
                plt.title("%s filtered @%4.1f-%4.1f Hz" % (sta, freqmin, freqmax))
            plt.xlabel("time (s)")
            plt.ylabel("offset (km)")
            if ic == 0:
                plt.plot([0, 2 * mdist], [0, mdist], "r--", linewidth=0.2)
                plt.plot([0, mdist], [0, mdist], "g--", linewidth=0.2)
            plt.text(disp_lag * 1.1, dist + 0.5, treceiver, fontsize=6)

        plt.plot([0, 0], [0, mdist], "b--", linewidth=1)
        font = {"family": "serif", "color": "red", "weight": "bold", "size": 16}
        plt.text(disp_lag * 0.65, 0.9 * mdist, comp, fontdict=font)
    plt.tight_layout()

    # save figure or show
    if savefig:
        outfname = sdir + "/moveout_" + sta + "_1D_" + str(stack_method) + ".pdf"
        plt.savefig(outfname, format="pdf", dpi=300)
        plt.close()
    else:
        plt.show()
