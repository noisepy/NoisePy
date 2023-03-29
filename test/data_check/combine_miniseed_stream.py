import datetime
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import obspy
import scipy.signal
from numba import jit
from obspy.signal.filter import bandpass, lowpass
from obspy.signal.util import _npts2nfft

import noise_module

sys.path.insert(1, "../../src")
import noise_module

"""
a test script for pre-processing data

modify the parameters and data-path at the end of this script
"""


def preprocess_raw(
    st, downsamp_freq, clean_time=True, pre_filt=None, resp=False, respdir=None
):
    """
    pre-process daily stream of data from IRIS server, including:

        - check sample rate is matching (from original process_raw)
        - remove small traces (from original process_raw)
        - remove trend and mean of each trace
        - interpolated to ensure all samples are at interger times of the sampling rate
        - low pass and downsample the data  (from original process_raw)
        - remove instrument response according to the option of resp_option.
            "inv" -> using inventory information and obspy function of remove_response;
            "spectrum" -> use downloaded response spectrum and interpolate if necessary
            "polezeros" -> use the pole zeros for a crude correction of response
        - trim data to a day-long sequence and interpolate it to ensure starting at 00:00:00.000
    """

    # ----remove the ones with too many segments and gaps------
    if len(st) > 100 or portion_gaps(st) > 0.5:
        print("Too many traces or gaps in Stream: Continue!")
        st = []
        return st

    # ----check sampling rate and trace length----
    st = check_sample(st)

    if len(st) == 0:
        print("No traces in Stream: Continue!")
        return st

    # -----remove mean and trend for each trace before merge------
    for ii in range(len(st)):
        st[ii].data = np.float32(st[ii].data)
        st[ii].data = scipy.signal.detrend(st[ii].data, type="constant")
        st[ii].data = scipy.signal.detrend(st[ii].data, type="linear")

    st.merge(method=1, fill_value=0)
    sps = st[0].stats.sampling_rate

    if abs(downsamp_freq - sps) > 1e-4:
        # -----low pass filter with corner frequency = 0.9*Nyquist frequency----
        # st[0].data = lowpass(st[0].data,freq=0.4*downsamp_freq,df=sps,corners=4,zerophase=True)
        st[0].data = bandpass(
            st[0].data, 0.005, 0.4 * downsamp_freq, df=sps, corners=4, zerophase=True
        )

        # ----make downsampling------
        st.interpolate(downsamp_freq, method="weighted_average_slopes")

        delta = st[0].stats.delta
        # -------when starttimes are between sampling points-------
        fric = st[0].stats.starttime.microsecond % (delta * 1e6)
        if fric > 1e-4:
            st[0].data = segment_interpolate(
                np.float32(st[0].data), float(fric / delta * 1e6)
            )
            # --reset the time to remove the discrepancy---
            st[0].stats.starttime -= fric * 1e-6

    station = st[0].stats.station

    # -----check whether file folder exists-------
    if resp is not False:
        if resp != "inv":
            if (respdir is None) or (not os.path.isdir(respdir)):
                raise ValueError("response file folder not found! abort!")

        if resp == "inv":
            # ----check whether inventory is attached----
            if not st[0].stats.response:
                raise ValueError("no response found in the inventory! abort!")
            else:
                print("removing response using inv")
                st.remove_response(output="VEL", pre_filt=pre_filt, water_level=60)

        elif resp == "spectrum":
            print("remove response using spectrum")
            specfile = glob.glob(os.path.join(respdir, "*" + station + "*"))
            if len(specfile) == 0:
                raise ValueError("no response sepctrum found for %s" % station)
            st = resp_spectrum(st[0], specfile[0], downsamp_freq)

        elif resp == "RESP_files":
            print("using RESP files")
            seedresp = glob.glob(os.path.join(respdir, "RESP." + station + "*"))
            if len(seedresp) == 0:
                raise ValueError("no RESP files found for %s" % station)
            st.simulate(paz_remove=None, pre_filt=pre_filt, seedresp=seedresp)

        elif resp == "polozeros":
            print("using polos and zeros")
            paz_sts = glob.glob(os.path.join(respdir, "*" + station + "*"))
            if len(paz_sts) == 0:
                raise ValueError("no polozeros found for %s" % station)
            st.simulate(paz_remove=paz_sts, pre_filt=pre_filt)

        else:
            raise ValueError(
                "no such option of resp in preprocess_raw! please double check!"
            )

    # -----fill gaps, trim data and interpolate to ensure all starts at 00:00:00.0------
    if clean_time:
        st = clean_daily_segments(st)

    return st


def portion_gaps(stream):
    """
    get the accumulated gaps (npts) by looking at the accumulated difference between starttime and endtime,
    instead of using the get_gaps function of obspy object of stream. remove the trace if gap length is
    more than 30% of the trace size. remove the ones with sampling rate not consistent with max(freq)
    """
    # -----check the consistency of sampling rate----

    pgaps = 0
    npts = (stream[-1].stats.endtime - stream[0].stats.starttime) * stream[
        0
    ].stats.sampling_rate

    if len(stream) == 0:
        return pgaps
    else:
        # ----loop through all trace to find gaps----
        for ii in range(len(stream) - 1):
            pgaps += (
                stream[ii + 1].stats.starttime - stream[ii].stats.endtime
            ) * stream[ii].stats.sampling_rate

    return pgaps / npts


@jit("float32[:](float32[:],float32)")
def segment_interpolate(sig1, nfric):
    """
    a sub-function of clean_daily_segments:

    interpolate the data according to fric to ensure all points located on interger times of the
    sampling rate (e.g., starttime = 00:00:00.015, delta = 0.05.)

    input parameters:
    sig1:  float32 -> seismic recordings in a 1D array
    nfric: float32 -> the amount of time difference between the point and the adjacent assumed samples
    """
    npts = len(sig1)
    sig2 = np.zeros(npts, dtype=np.float32)

    # ----instead of shifting, do a interpolation------
    for ii in range(npts):
        # ----deal with edges-----
        if ii == 0 or ii == npts:
            sig2[ii] = sig1[ii]
        else:
            # ------interpolate using a hat function------
            sig2[ii] = (1 - nfric) * sig1[ii + 1] + nfric * sig1[ii]

    return sig2


def resp_spectrum(source, resp_file, downsamp_freq):
    """
    remove the instrument response with response spectrum from evalresp.
    the response spectrum is evaluated based on RESP/PZ files and then
    inverted using obspy function of invert_spectrum.
    """
    # --------resp_file is the inverted spectrum response---------
    respz = np.load(resp_file)
    nrespz = respz[1][:]
    spec_freq = max(respz[0])

    # -------on current trace----------
    nfft = _npts2nfft(source.stats.npts)
    sps = source.stats.sample_rate

    # ---------do the interpolation if needed--------
    if spec_freq < 0.5 * sps:
        raise ValueError("spectrum file has peak freq smaller than the data, abort!")
    else:
        indx = np.where(respz[0] <= 0.5 * sps)
        nfreq = np.linspace(0, 0.5 * sps, nfft)
        nrespz = np.interp(nfreq, respz[0][indx], respz[1][indx])

    # ----do interpolation if necessary-----
    source_spect = np.fft.rfft(source.data, n=nfft)

    # -----nrespz is inversed (water-leveled) spectrum-----
    source_spect *= nrespz
    source.data = np.fft.irfft(source_spect)[0 : source.stats.npts]

    return source


def clean_daily_segments(tr):
    """
    subfunction to clean the tr recordings. only the traces with at least 0.5-day long
    sequence (respect to 00:00:00.0 of the day) is kept. note that the trace here could
    be of several days recordings, so this function helps to break continuous chunck
    into a day-long segment from 00:00:00.0 to 24:00:00.0.

    tr: obspy stream object
    """
    # -----all potential-useful time information-----
    stream_time = tr[0].stats.starttime
    time0 = obspy.UTCDateTime(
        stream_time.year, stream_time.month, stream_time.day, 0, 0, 0
    )
    time1 = obspy.UTCDateTime(
        stream_time.year, stream_time.month, stream_time.day, 12, 0, 0
    )
    time2 = time1 + datetime.timedelta(hours=12)

    # --only keep days with > 0.5-day recordings-
    if stream_time <= time1:
        starttime = time0
    else:
        starttime = time2

    # -----ndays represents how many days from starttime to endtime----
    ndays = round((tr[0].stats.endtime - starttime) / (time2 - time0))
    if ndays == 0:
        tr = []
        return tr

    else:
        # -----make a new stream------
        ntr = obspy.Stream()
        ttr = tr[0].copy()
        # ----trim a continous segment into day-long sequences----
        for ii in range(ndays):
            tr[0] = ttr.copy()
            endtime = starttime + datetime.timedelta(days=1)
            tr[0].trim(starttime=starttime, endtime=endtime, pad=True, fill_value=0)

            ntr.append(tr[0])
            starttime = endtime

    return ntr


def check_sample(stream):
    """
    Returns sampling rate of traces in stream.

    :type stream:`~obspy.core.stream.Stream` object.
    :param stream: Stream containing one or more day-long trace
    :return: List of sampling rates in stream

    """
    if len(stream) == 0:
        return stream
    else:
        freqs = []
        for tr in stream:
            freqs.append(tr.stats.sampling_rate)

    freq = max(freqs)
    for tr in stream:
        if tr.stats.sampling_rate != freq:
            stream.remove(tr)

    return stream


###############################################################
# ---------open a new stream for the mini-seed files------------
source = obspy.Stream()

sacfiles = glob.glob(
    "/Users/chengxin/Documents/Harvard/JAKARTA/JKA20miniSEED/*13101*.CHZ"
)
for ii in range(len(sacfiles)):
    temp = obspy.read(sacfiles[ii])
    source.append(temp[0])

# ---some control parameters
checkt = True
NewFreq = 20
pre_filt = [0.04, 0.05, 4, 5]
resp = "spectrum"  # not removing response here
respdir = "/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/instrument/resp_4types/resp_spect_20Hz"

# ----do the pre-processing------
print("before pre-processing:", source)
osource = source.copy()

nsource = preprocess_raw(osource, NewFreq, checkt, pre_filt, resp, respdir)
print("after processing:", nsource)

# ---note the order in source is a mass----
iday_old = []
iday_new = []
for ii in range(len(source)):
    iday_old.append(source[ii].stats.starttime.day)
    iday_new.append(nsource[ii].stats.starttime.day)

print(iday_old, iday_new)

# ------check the waveforms now-------
for ii in range(len(iday_new)):
    indx = iday_old.index(iday_new[ii])
    plt.subplot(211)
    plt.plot(source[indx].data)
    plt.legend(["before pre-process"], loc="upper right")
    plt.title("Note starttimg, dt and freq are all different")
    plt.subplot(212)
    plt.plot(nsource[ii].data)
    plt.legend(["after pre-process"], loc="upper right")
    plt.show()
