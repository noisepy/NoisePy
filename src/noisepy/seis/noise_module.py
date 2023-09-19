# noisemodule.py
# This file is excluded from this check because it contains two modules with
# the same name but different inputs
# noqa: F811
# import pyasdf
import datetime
import glob
import logging
import os

import numpy as np
import obspy
import pandas as pd

# import pycwt
import scipy
from numba import jit
from obspy.core.inventory import Channel, Inventory, Network, Site, Station
from obspy.core.util.base import _get_function_from_entry_point
from obspy.signal.filter import bandpass

# from obspy.signal.invsim import cosine_taper
# from obspy.signal.regression import linear_regression
from obspy.signal.util import _npts2nfft
from scipy.fftpack import next_fast_len
from scipy.signal import hilbert

from .datatypes import ChannelData, ConfigParameters, StackMethod

logger = logging.getLogger(__name__)
"""
This VERY LONG noise module file is necessary to keep the NoisePy working properly. In general,
the modules are organized based on their functionality in the following way. it includes:

1) core functions called directly by the main NoisePy scripts;
2) utility functions used by the core functions;
3) monitoring functions representing different methods to measure dv/v;
4) monitoring utility functions used by the monitoring functions.

by: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
    Marine Denolle (mdenolle@uw.edu)

several utility functions are modified based on https://github.com/tclements/noise
"""

####################################################
############## CORE FUNCTIONS ######################
####################################################


def get_event_list(d1: datetime.datetime, d2: datetime.datetime, inc_hours: int):
    """
    this function calculates the event list between times d1 and d2 by increment of inc_hours
    PARAMETERS:
    ----------------
    str1: string of the starting time -> 2010_01_01_0_0
    str2: string of the ending time -> 2010_10_11_0_0
    inc_hours: integer of incremental hours
    RETURNS:
    ----------------
    event: a numpy character list
    """
    dt = datetime.timedelta(hours=inc_hours)

    event = []
    while d1 < d2:
        event.append(d1.strftime("%Y_%m_%d_%H_%M_%S"))
        d1 += dt
    event.append(d2.strftime("%Y_%m_%d_%H_%M_%S"))

    return event


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


def preprocess_raw(
    st: obspy.Stream,
    inv: obspy.Inventory,
    prepro_para: ConfigParameters,
    starttime: obspy.UTCDateTime,
    endtime: obspy.UTCDateTime,
):
    """
    this function pre-processes the raw data stream by:
        1) check samping rate and gaps in the data;
        2) remove sigularity, trend and mean of each trace
        3) filter and correct the time if integer time are between sampling points
        4) remove instrument responses with selected methods including:
            "inv"   -> using inventory information to remove_response;
            "spectrum"   -> use the inverse of response spectrum.
            (a script is provided in additional_module to estimate response spectrum from RESP files)
            "RESP_files" -> use the raw download RESP files
            "polezeros"  -> use pole/zero info for a crude correction of response
        5) trim data to a day-long sequence and interpolate it to ensure starting at 00:00:00.000
    (used in S0A & S0B)
    PARAMETERS:
    -----------------------
    st:  obspy stream object, containing noise data to be processed
    inv: obspy inventory object, containing stations info
    prepro_para: dict containing fft parameters, such as frequency bands and
    selection for instrument response removal etc.
    date_info:   dict of start and end time of the stream data
    RETURNS:
    -----------------------
    ntr: obspy stream object of cleaned, merged and filtered noise data
    """
    # load paramters from fft dict
    rm_resp = prepro_para["rm_resp"]
    rm_resp_out = prepro_para["rm_resp_out"]
    respdir = prepro_para["respdir"]
    freqmin = prepro_para["freqmin"]
    freqmax = prepro_para["freqmax"]
    samp_freq = prepro_para["samp_freq"]

    # parameters for butterworth filter
    f1 = 0.9 * freqmin
    f2 = freqmin
    if 1.1 * freqmax > 0.45 * samp_freq:
        f3 = 0.4 * samp_freq
        f4 = 0.45 * samp_freq
    else:
        f3 = freqmax
        f4 = 1.1 * freqmax
    pre_filt = [f1, f2, f3, f4]

    # check sampling rate and trace length
    st = check_sample_gaps(st, starttime, endtime)
    if len(st) == 0:
        logger.warning("No traces in Stream: Continue!")
        return st
    sps = int(st[0].stats.sampling_rate)
    station = st[0].stats.station

    # remove nan/inf, mean and trend of each trace before merging
    for ii in range(len(st)):
        # -----set nan/inf values to zeros (it does happens!)-----
        tttindx = np.where(np.isnan(st[ii].data))
        if len(tttindx) > 0:
            st[ii].data[tttindx] = 0
        tttindx = np.where(np.isinf(st[ii].data))
        if len(tttindx) > 0:
            st[ii].data[tttindx] = 0

        st[ii].data = np.float32(st[ii].data)
        st[ii].data = scipy.signal.detrend(st[ii].data, type="constant")
        st[ii].data = scipy.signal.detrend(st[ii].data, type="linear")

    # merge, taper and filter the data
    if len(st) > 1:
        st.merge(method=1, fill_value=0)
    st[0].taper(max_percentage=0.05, max_length=50)  # taper window
    st[0].data = np.float32(bandpass(st[0].data, pre_filt[0], pre_filt[-1], df=sps, corners=4, zerophase=True))

    # make downsampling if needed
    if abs(samp_freq - sps) > 1e-4:
        # downsampling here
        st.interpolate(samp_freq, method="weighted_average_slopes")
        delta = st[0].stats.delta

        # when starttimes are between sampling points
        fric = st[0].stats.starttime.microsecond % (delta * 1e6)
        if fric > 1e-4:
            st[0].data = segment_interpolate(np.float32(st[0].data), float(fric / (delta * 1e6)))
            # --reset the time to remove the discrepancy---
            st[0].stats.starttime -= fric * 1e-6

    # remove traces of too small length

    # options to remove instrument response
    if rm_resp != "no":
        if rm_resp != "inv":
            if (respdir is None) or (not os.path.isdir(respdir)):
                raise ValueError("response file folder not found! abort!")

        if rm_resp == "inv":
            # ----check whether inventory is attached----
            if not inv[0][0][0].response:
                raise ValueError("no response found in the inventory! abort!")
            elif inv[0][0][0].response == obspy.core.inventory.response.Response():
                raise ValueError("The response found in the inventory is empty (no stages)! abort!  %s" % st[0])
            else:
                try:
                    logger.info("removing response for %s using inv" % st[0])
                    st[0].attach_response(inv)
                    st[0].remove_response(output=rm_resp_out, pre_filt=pre_filt, water_level=60)
                except Exception as e:
                    logger.warning("Failed to remove response from %s. Returning empty stream. %s" % (st[0], e))
                    st = []
                    return st

        elif rm_resp == "spectrum":
            logger.info("remove response using spectrum")
            specfile = glob.glob(os.path.join(respdir, "*" + station + "*"))
            if len(specfile) == 0:
                raise ValueError("no response sepctrum found for %s" % station)
            st = resp_spectrum(st, specfile[0], samp_freq, pre_filt)

        elif rm_resp == "RESP":
            logger.info("remove response using RESP files")
            resp = glob.glob(os.path.join(respdir, "RESP." + station + "*"))
            if len(resp) == 0:
                raise ValueError("no RESP files found for %s" % station)
            seedresp = {
                "filename": resp[0],
                "date": starttime,
                "units": "DIS",
            }
            st.simulate(paz_remove=None, pre_filt=pre_filt, seedresp=seedresp)

        elif rm_resp == "poleszeros":
            logger.info("remove response using poles and zeros")
            paz_sts = glob.glob(os.path.join(respdir, "*" + station + "*"))
            if len(paz_sts) == 0:
                raise ValueError("no poleszeros found for %s" % station)
            st.simulate(paz_remove=paz_sts[0], pre_filt=pre_filt)

        else:
            raise ValueError("no such option for rm_resp! please double check!")

    ntr = obspy.Stream()
    # trim a continous segment into user-defined sequences
    st[0].trim(
        starttime=starttime,
        endtime=endtime,
        pad=True,
        fill_value=0,
    )
    ntr.append(st[0])

    return ntr


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
        return stats2Inv_staxml(stats, respdir)
    if input_fmt == "sac":
        return stats2inv_sac(stats)
    elif input_fmt == "mseed":
        return stats2inv_mseed(stats, locs)


def stats2Inv_staxml(stats, respdir) -> Inventory:
    if not respdir:
        raise ValueError("Abort! staxml is selected but no directory is given to access the files")
    else:
        invfilelist = glob.glob(os.path.join(respdir, "*" + stats.station + "*"))
        if len(invfilelist) > 0:
            invfile = invfilelist[0]
            if len(invfilelist) > 1:
                logger.warning(
                    (
                        "Warning! More than one StationXML file was found for station %s."
                        + "Keeping the first file in list."
                    )
                    % stats.station
                )
            if os.path.isfile(str(invfile)):
                inv = obspy.read_inventory(invfile)
                return inv
        else:
            raise ValueError("Could not find a StationXML file for station: %s." % stats.station)


def stats2inv_sac(stats):
    inv = Inventory(networks=[], source="homegrown")
    net = Network(
        # This is the network code according to the SEED standard.
        code=stats.network,
        stations=[],
        description="created from SAC and resp files",
        start_date=stats.starttime,
    )

    sta = Station(
        # This is the station code according to the SEED standard.
        code=stats.station,
        latitude=stats.sac["stla"],
        longitude=stats.sac["stlo"],
        elevation=stats.sac["stel"],
        creation_date=stats.starttime,
        site=Site(name="First station"),
    )

    cha = Channel(
        # This is the channel code according to the SEED standard.
        code=stats.channel,
        # This is the location code according to the SEED standard.
        location_code=stats.location,
        # Note that these coordinates can differ from the station coordinates.
        latitude=stats.sac["stla"],
        longitude=stats.sac["stlo"],
        elevation=stats.sac["stel"],
        depth=-stats.sac["stel"],
        azimuth=stats.sac["cmpaz"],
        dip=stats.sac["cmpinc"],
        sample_rate=stats.sampling_rate,
    )
    response = obspy.core.inventory.response.Response()

    # Now tie it all together.
    cha.response = response
    sta.channels.append(cha)
    net.stations.append(sta)
    inv.networks.append(net)

    return inv


def stats2inv_mseed(stats, locs: pd.DataFrame) -> Inventory:
    inv = Inventory(networks=[], source="homegrown")
    ista = locs[locs["station"] == stats.station].index.values.astype("int64")[0]

    net = Network(
        # This is the network code according to the SEED standard.
        code=locs.iloc[ista]["network"],
        stations=[],
        description="created from SAC and resp files",
        start_date=stats.starttime,
    )

    sta = Station(
        # This is the station code according to the SEED standard.
        code=locs.iloc[ista]["station"],
        latitude=locs.iloc[ista]["latitude"],
        longitude=locs.iloc[ista]["longitude"],
        elevation=locs.iloc[ista]["elevation"],
        creation_date=stats.starttime,
        site=Site(name="First station"),
    )

    cha = Channel(
        code=stats.channel,
        location_code=stats.location,
        latitude=locs.iloc[ista]["latitude"],
        longitude=locs.iloc[ista]["longitude"],
        elevation=locs.iloc[ista]["elevation"],
        depth=-locs.iloc[ista]["elevation"],
        azimuth=0,
        dip=0,
        sample_rate=stats.sampling_rate,
    )

    response = obspy.core.inventory.response.Response()

    # Now tie it all together.
    cha.response = response
    sta.channels.append(cha)
    net.stations.append(sta)
    inv.networks.append(net)

    return inv


def sta_info_from_inv(inv: obspy.core.inventory.inventory.Inventory):
    """
    this function outputs station info from the obspy inventory object
    (used in S0B)
    PARAMETERS:
    ----------------------
    inv: obspy inventory object
    RETURNS:
    ----------------------
    sta: station name
    net: netowrk name
    lon: longitude of the station
    lat: latitude of the station
    elv: elevation of the station
    location: location code of the station
    """
    # load from station inventory
    sta = inv[0][0].code
    net = inv[0].code
    lon = inv[0][0].longitude
    lat = inv[0][0].latitude
    if inv[0][0].elevation:
        elv = inv[0][0].elevation
    else:
        elv = 0.0

    if inv[0][0][0].location_code:
        location = inv[0][0][0].location_code
    else:
        location = "00"

    return sta, net, lon, lat, elv, location


def cut_trace_make_stat(fc_para: ConfigParameters, ch_data: ChannelData):
    """
    this function cuts continous noise data into user-defined segments, estimate the statistics of
    each segment and keep timestamp of each segment for later use. (used in S1)
    PARAMETERS:
    ----------------------
    fft_para: A dictionary containing all fft and cc parameters.
    source: obspy stream object
    RETURNS:
    ----------------------
    trace_stdS: standard deviation of the noise amplitude of each segment
    dataS_t:    timestamps of each segment
    dataS:      2D matrix of the segmented data
    """
    # define return variables first
    source_params = []
    dataS_t = []
    dataS = []

    # useful parameters for trace sliding
    nseg = int(np.floor((fc_para.inc_hours / 24 * 86400 - fc_para.cc_len) / fc_para.step))
    sps = int(ch_data.sampling_rate)
    starttime = ch_data.start_timestamp
    # copy data into array
    data = ch_data.data

    # if the data is shorter than the tim chunck, return zero values
    if data.size < sps * fc_para.inc_hours * 3600:
        logger.warning(
            f"The data ({data.size}) is shorter than the time chunk ({sps*fc_para.inc_hours*3600})"
            ", returning zero values."
        )
        return source_params, dataS_t, dataS

    # statistic to detect segments that may be associated with earthquakes
    all_madS = mad(data)  # median absolute deviation over all noise window
    all_stdS = np.std(data)  # standard deviation over all noise window
    if all_madS == 0 or all_stdS == 0 or np.isnan(all_madS) or np.isnan(all_stdS):
        logger.debug("continue! madS or stdS equals to 0 for %s")
        return source_params, dataS_t, dataS

    # initialize variables
    npts = int(fc_para.cc_len * sps)
    # trace_madS = np.zeros(nseg,dtype=np.float32)
    trace_stdS = np.zeros(nseg, dtype=np.float32)
    dataS = np.zeros(shape=(int(nseg), int(npts)), dtype=np.float32)
    dataS_t = np.zeros(nseg, dtype=np.float32)

    indx1 = 0
    for iseg in range(nseg):
        indx2 = indx1 + npts
        dataS[iseg] = data[indx1:indx2]
        # trace_madS[iseg] = (np.max(np.abs(dataS[iseg]))/all_madS)
        trace_stdS[iseg] = np.max(np.abs(dataS[iseg])) / all_stdS
        dataS_t[iseg] = starttime + fc_para.step * iseg
        indx1 = indx1 + int(fc_para.step) * sps

    # 2D array processing
    dataS = demean(dataS)
    dataS = detrend(dataS)
    dataS = taper(dataS)

    return trace_stdS, dataS_t, dataS


def noise_processing(fft_para: ConfigParameters, dataS):
    """
    this function performs time domain and frequency domain normalization if needed. in real case, we prefer use include
    the normalization in the cross-correaltion steps by selecting coherency or decon
    (Prieto et al, 2008, 2009; Denolle et al, 2013)
    PARMAETERS:
    ------------------------
    fft_para: ConfigParameters class containing all useful variables used for fft and cc
    dataS: 2D matrix of all segmented noise data
    # OUTPUT VARIABLES:
    source_white: 2D matrix of data spectra
    """
    # ------to normalize in time or not------
    if fft_para.time_norm != "no":
        if fft_para.time_norm == "one_bit":  # sign normalization
            white = np.sign(dataS)
        elif fft_para.time_norm == "rma":  # running mean: normalization over smoothed absolute average
            white = np.zeros(shape=dataS.shape, dtype=dataS.dtype)
            for kkk in range(dataS.shape[0]):
                white[kkk, :] = dataS[kkk, :] / moving_ave(np.abs(dataS[kkk, :]), fft_para.smooth_N)

    else:  # don't normalize
        white = dataS

    # -----to whiten or not------
    if fft_para.freq_norm != "no":
        source_white = whiten(white, fft_para)  # whiten and return FFT
    else:
        Nfft = int(next_fast_len(int(dataS.shape[1])))
        source_white = scipy.fftpack.fft(white, Nfft, axis=1)  # return FFT

    return source_white


def smooth_source_spect(cc_para, fft1):
    """
    this function smoothes amplitude spectrum of the 2D spectral matrix. (used in S1)
    PARAMETERS:
    ---------------------
    cc_para: dictionary containing useful cc parameters
    fft1:    source spectrum matrix

    RETURNS:
    ---------------------
    sfft1: complex numpy array with normalized spectrum
    """
    cc_method = cc_para["cc_method"]
    smoothspect_N = cc_para["smoothspect_N"]

    if cc_method == "deconv":
        # -----normalize single-station cc to z component-----
        temp = moving_ave(np.abs(fft1), smoothspect_N)
        try:
            sfft1 = np.conj(fft1) / temp**2
        except Exception:
            raise ValueError("smoothed spectrum has zero values")

    elif cc_method == "coherency":
        temp = moving_ave(np.abs(fft1), smoothspect_N)
        try:
            sfft1 = np.conj(fft1) / temp
        except Exception:
            raise ValueError("smoothed spectrum has zero values")

    elif cc_method == "xcorr":
        sfft1 = np.conj(fft1)

    else:
        raise ValueError("no correction correlation method is selected at L59")

    return sfft1


def correlate(fft1_smoothed_abs, fft2, D, Nfft, dataS_t):
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

    MODIFICATIONS:
    ---------------------
    output the linear stack of each time chunk even when substack is selected (by Chengxin @Aug2020)
    """
    # ----load paramters----
    dt = D["dt"]
    maxlag = D["maxlag"]
    method = D["cc_method"]
    cc_len = D["cc_len"]
    substack = D["substack"]
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

    if substack:
        if substack_len == cc_len:
            # choose to keep all fft data for a day
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
            n_corr = np.zeros(nstack, dtype=np.int16)
            t_corr = np.zeros(nstack, dtype=np.float32)
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
        ampmax = np.max(corr, axis=1)
        tindx = np.where((ampmax < 20 * np.median(ampmax)) & (ampmax > 0))[0]
        n_corr = nwin
        s_corr = np.zeros(Nfft, dtype=np.float32)
        t_corr = dataS_t[0]
        crap = np.zeros(Nfft, dtype=np.complex64)
        crap[:Nfft2] = np.mean(corr[tindx], axis=0)
        crap[:Nfft2] = crap[:Nfft2] - np.mean(crap[:Nfft2], axis=0)
        crap[-(Nfft2) + 1 :] = np.flip(np.conj(crap[1:(Nfft2)]), axis=0)
        s_corr = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

    # trim the CCFs in [-maxlag maxlag]
    t = np.arange(-Nfft2 + 1, Nfft2) * dt
    ind = np.where(np.abs(t) <= maxlag)[0]
    if s_corr.ndim == 1:
        # Expand dims to ensure we always return a 2D array
        s_corr = np.expand_dims(s_corr[ind], axis=0)
    elif s_corr.ndim == 2:
        s_corr = s_corr[:, ind]
    return s_corr, t_corr, n_corr


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


def cc_parameters(cc_para, coor, tcorr, ncorr, comp):
    """
    this function assembles the parameters for the cc function, which is used
    when writing them into ASDF files
    PARAMETERS:
    ---------------------
    cc_para: dict containing parameters used in the fft_cc step
    coor:    dict containing coordinates info of the source and receiver stations
    tcorr:   timestamp matrix
    ncorr:   matrix of number of good segments for each sub-stack/final stack
    comp:    2 character strings for the cross correlation component
    RETURNS:
    ------------------
    parameters: dict containing above info used for later stacking/plotting
    """
    latS = coor["latS"]
    lonS = coor["lonS"]
    latR = coor["latR"]
    lonR = coor["lonR"]
    dt = cc_para["dt"]
    maxlag = cc_para["maxlag"]
    substack = cc_para["substack"]
    cc_method = cc_para["cc_method"]

    dist, azi, baz = obspy.geodetics.base.gps2dist_azimuth(latS, lonS, latR, lonR)
    parameters = {
        "dt": dt,
        "maxlag": int(maxlag),
        "dist": np.float32(dist / 1000),
        "azi": np.float32(azi),
        "baz": np.float32(baz),
        "lonS": np.float32(lonS),
        "latS": np.float32(latS),
        "lonR": np.float32(lonR),
        "latR": np.float32(latR),
        "ngood": ncorr,
        "cc_method": cc_method,
        "time": tcorr,
        "substack": substack,
        "comp": comp,
    }
    return parameters


def stacking(cc_array, cc_time, cc_ngood, stack_para):
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
    samp_freq = stack_para["samp_freq"]
    smethod = stack_para["stack_method"]
    npts = cc_array.shape[1]

    # remove abnormal data
    ampmax = np.max(cc_array, axis=1)
    tindx = np.where((ampmax < 20 * np.median(ampmax)) & (ampmax > 0))[0]
    if not len(tindx):
        allstacks1 = []
        allstacks2 = []
        allstacks3 = []
        nstacks = 0
        cc_array = []
        cc_ngood = []
        cc_time = []
        return cc_array, cc_ngood, cc_time, allstacks1, allstacks2, allstacks3, nstacks
    else:
        # remove ones with bad amplitude
        cc_array = cc_array[tindx, :]
        cc_time = cc_time[tindx]
        cc_ngood = cc_ngood[tindx]

        # do stacking
        allstacks1 = np.zeros(npts, dtype=np.float32)
        allstacks2 = np.zeros(npts, dtype=np.float32)
        allstacks3 = np.zeros(npts, dtype=np.float32)

        if smethod == StackMethod.LINEAR:
            allstacks1 = np.mean(cc_array, axis=0)
        elif smethod == StackMethod.PWS:
            allstacks1 = pws(cc_array, samp_freq)
        elif smethod == StackMethod.ROBUST:
            allstacks1, w, nstep = robust_stack(cc_array, 0.001)
        elif smethod == StackMethod.AUTO_COVARIANCE:
            allstacks1 = adaptive_filter(cc_array, 1)
        elif smethod == StackMethod.NROOT:
            allstacks1 = nroot_stack(cc_array, 2)
        elif smethod == StackMethod.ALL:
            allstacks1 = np.mean(cc_array, axis=0)
            allstacks2 = pws(cc_array, samp_freq)
            allstacks3, w, nstep = robust_stack(cc_array, 0.001)
        nstacks = np.sum(cc_ngood)

    # good to return
    return cc_array, cc_ngood, cc_time, allstacks1, allstacks2, allstacks3, nstacks


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
    samp_freq = stack_para["samp_freq"]
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
            allstacks1 = pws(cc_array, samp_freq)
        elif smethod == StackMethod.ROBUST:
            (
                allstacks1,
                w,
            ) = robust_stack(cc_array, 0.001)
        elif smethod == StackMethod.SELECTIVE:
            allstacks1 = selective_stack(cc_array, 0.001)
        elif smethod == StackMethod.ALL:
            allstacks1 = np.mean(cc_array, axis=0)
            allstacks2 = pws(cc_array, samp_freq)
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


def rotation(bigstack, parameters, locs):
    """
    this function transfers the Green's tensor from a E-N-Z system into a R-T-Z one

    PARAMETERS:
    -------------------
    bigstack:   9 component Green's tensor in E-N-Z system
    parameters: dict containing all parameters saved in ASDF file
    locs:       dict containing station angle info for correction purpose
    RETURNS:
    -------------------
    tcorr: 9 component Green's tensor in R-T-Z system
    """
    # load parameter dic
    pi = np.pi
    azi = parameters["azi"]
    baz = parameters["baz"]
    ncomp, npts = bigstack.shape
    if ncomp < 9:
        logger.debug("crap did not get enough components")
        tcorr = []
        return tcorr
    staS = parameters["station_source"]
    staR = parameters["station_receiver"]

    if len(locs):
        sta_list = list(locs["station"])
        angles = list(locs["angle"])
        # get station info from the name of ASDF file
        ind = sta_list.index(staS)
        acorr = angles[ind]
        ind = sta_list.index(staR)
        bcorr = angles[ind]

    # ---angles to be corrected----
    if len(locs):
        cosa = np.cos((azi + acorr) * pi / 180)
        sina = np.sin((azi + acorr) * pi / 180)
        cosb = np.cos((baz + bcorr) * pi / 180)
        sinb = np.sin((baz + bcorr) * pi / 180)
    else:
        cosa = np.cos(azi * pi / 180)
        sina = np.sin(azi * pi / 180)
        cosb = np.cos(baz * pi / 180)
        sinb = np.sin(baz * pi / 180)

    # rtz_components = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
    tcorr = np.zeros(shape=(9, npts), dtype=np.float32)
    tcorr[0] = -cosb * bigstack[7] - sinb * bigstack[6]
    tcorr[1] = sinb * bigstack[7] - cosb * bigstack[6]
    tcorr[2] = bigstack[8]
    tcorr[3] = (
        -cosa * cosb * bigstack[4] - cosa * sinb * bigstack[3] - sina * cosb * bigstack[1] - sina * sinb * bigstack[0]
    )
    tcorr[4] = (
        cosa * sinb * bigstack[4] - cosa * cosb * bigstack[3] + sina * sinb * bigstack[1] - sina * cosb * bigstack[0]
    )
    tcorr[5] = cosa * bigstack[5] + sina * bigstack[2]
    tcorr[6] = (
        sina * cosb * bigstack[4] + sina * sinb * bigstack[3] - cosa * cosb * bigstack[1] - cosa * sinb * bigstack[0]
    )
    tcorr[7] = (
        -sina * sinb * bigstack[4] + sina * cosb * bigstack[3] + cosa * sinb * bigstack[1] - cosa * cosb * bigstack[0]
    )
    tcorr[8] = -sina * bigstack[5] + cosa * bigstack[2]

    return tcorr


####################################################
############## UTILITY FUNCTIONS ###################
####################################################


def check_sample_gaps(stream: obspy.Stream, starttime: obspy.UTCDateTime, endtime: obspy.UTCDateTime):
    """
    this function checks sampling rate and find gaps of all traces in stream.
    PARAMETERS:
    -----------------
    stream: obspy stream object.
    date_info: dict of starting and ending time of the stream

    RETURENS:
    -----------------
    stream: List of good traces in the stream
    """
    # remove empty/big traces
    if len(stream) == 0 or len(stream) > 100:
        stream = []
        return stream

    # remove traces with big gaps
    if portion_gaps(stream, starttime, endtime) > 0.3:
        stream = []
        return stream

    freqs = []
    for tr in stream:
        freqs.append(int(tr.stats.sampling_rate))
    freq = max(freqs)
    for tr in stream:
        if int(tr.stats.sampling_rate) != freq:
            stream.remove(tr)
        if tr.stats.npts < 10:
            stream.remove(tr)

    return stream


def portion_gaps(stream, starttime: obspy.UTCDateTime, endtime: obspy.UTCDateTime):
    """
    this function tracks the gaps (npts) from the accumulated difference between starttime and endtime
    of each stream trace. it removes trace with gap length > 30% of trace size.
    PARAMETERS:
    -------------------
    stream: obspy stream object
    date_info: dict of starting and ending time of the stream

    RETURNS:
    -----------------
    pgaps: proportion of gaps/all_pts in stream
    """
    # ideal duration of data
    npts = (endtime - starttime) * stream[0].stats.sampling_rate

    pgaps = 0
    # loop through all trace to accumulate gaps
    for ii in range(len(stream) - 1):
        pgaps += (stream[ii + 1].stats.starttime - stream[ii].stats.endtime) * stream[ii].stats.sampling_rate
    if npts != 0:
        pgaps = pgaps / npts
    if npts == 0:
        pgaps = 1
    return pgaps


@jit("float32[:](float32[:],float32)")
def segment_interpolate(sig1, nfric):
    """
    this function interpolates the data to ensure all points located on interger times of the
    sampling rate (e.g., starttime = 00:00:00.015, delta = 0.05.)
    PARAMETERS:
    ----------------------
    sig1:  seismic recordings in a 1D array
    nfric: the amount of time difference between the point and the adjacent assumed samples
    RETURNS:
    ----------------------
    sig2:  interpolated seismic recordings on the sampling points
    """
    npts = len(sig1)
    sig2 = np.zeros(npts, dtype=np.float32)

    # ----instead of shifting, do a interpolation------
    for ii in range(npts):
        # ----deal with edges-----
        if ii == 0 or ii == npts - 1:
            sig2[ii] = sig1[ii]
        else:
            # ------interpolate using a hat function------
            sig2[ii] = (1 - nfric) * sig1[ii + 1] + nfric * sig1[ii]

    return sig2


def resp_spectrum(source, resp_file, downsamp_freq, pre_filt=None):
    """
    this function removes the instrument response using response spectrum from evalresp.
    the response spectrum is evaluated based on RESP/PZ files before inverted using the obspy
    function of invert_spectrum. a module of create_resp.py is provided in directory of 'additional_modules'
    to create the response spectrum
    PARAMETERS:
    ----------------------
    source: obspy stream object of targeted noise data
    resp_file: numpy data file of response spectrum
    downsamp_freq: sampling rate of the source data
    pre_filt: pre-defined filter parameters
    RETURNS:
    ----------------------
    source: obspy stream object of noise data with instrument response removed
    """
    # --------resp_file is the inverted spectrum response---------
    respz = np.load(resp_file)
    nrespz = respz[1][:]
    spec_freq = max(respz[0])

    # -------on current trace----------
    nfft = _npts2nfft(source[0].stats.npts)
    sps = int(source[0].stats.sampling_rate)

    # ---------do the interpolation if needed--------
    if spec_freq < 0.5 * sps:
        raise ValueError("spectrum file has peak freq smaller than the data, abort!")
    else:
        indx = np.where(respz[0] <= 0.5 * sps)
        nfreq = np.linspace(0, 0.5 * sps, nfft // 2 + 1)
        nrespz = np.interp(nfreq, np.real(respz[0][indx]), respz[1][indx])

    # ----do interpolation if necessary-----
    source_spect = np.fft.rfft(source[0].data, n=nfft)

    # -----nrespz is inversed (water-leveled) spectrum-----
    source_spect *= nrespz
    source[0].data = np.fft.irfft(source_spect)[0 : source[0].stats.npts]

    if pre_filt is not None:
        source[0].data = np.float32(
            bandpass(
                source[0].data,
                pre_filt[0],
                pre_filt[-1],
                df=sps,
                corners=4,
                zerophase=True,
            )
        )

    return source


def mad(arr):
    """
    Median Absolute Deviation: MAD = median(|Xi- median(X)|)
    PARAMETERS:
    -------------------
    arr: numpy.ndarray, seismic trace data array
    RETURNS:
    data: Median Absolute Deviation of data
    """
    if not np.ma.is_masked(arr):
        med = np.median(arr)
        data = np.median(np.abs(arr - med))
    else:
        med = np.ma.median(arr)
        data = np.ma.median(np.ma.abs(arr - med))
    return data


def detrend(data):
    """
    this function removes the signal trend based on QR decomposion
    NOTE: QR is a lot faster than the least square inversion used by
    scipy (also in obspy).
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with trend removed
    """
    # ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        npts = data.shape[0]
        X = np.ones((npts, 2))
        X[:, 0] = np.arange(0, npts) / npts
        Q, R = np.linalg.qr(X)
        rq = np.dot(np.linalg.inv(R), Q.transpose())
        coeff = np.dot(rq, data)
        data = data - np.dot(X, coeff)
    elif data.ndim == 2:
        npts = data.shape[1]
        X = np.ones((npts, 2))
        X[:, 0] = np.arange(0, npts) / npts
        Q, R = np.linalg.qr(X)
        rq = np.dot(np.linalg.inv(R), Q.transpose())
        for ii in range(data.shape[0]):
            coeff = np.dot(rq, data[ii])
            data[ii] = data[ii] - np.dot(X, coeff)
    return data


def demean(data):
    """
    this function remove the mean of the signal
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with mean removed
    """
    # ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        data = data - np.mean(data)
    elif data.ndim == 2:
        for ii in range(data.shape[0]):
            data[ii] = data[ii] - np.mean(data[ii])
    return data


def taper(data):
    """
    this function applies a cosine taper using obspy functions
    PARAMETERS:
    ---------------------
    data: input data matrix
    RETURNS:
    ---------------------
    data: data matrix with taper applied
    """
    # ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        npts = data.shape[0]
        # window length
        if npts * 0.05 > 20:
            wlen = 20
        else:
            wlen = npts * 0.05
        # taper values
        func = _get_function_from_entry_point("taper", "hann")
        if 2 * wlen == npts:
            taper_sides = func(2 * wlen)
        else:
            taper_sides = func(2 * wlen + 1)
        # taper window
        win = np.hstack(
            (
                taper_sides[:wlen],
                np.ones(npts - 2 * wlen),
                taper_sides[len(taper_sides) - wlen :],
            )
        )
        data *= win
    elif data.ndim == 2:
        npts = data.shape[1]
        # window length
        if npts * 0.05 > 20:
            wlen = 20
        else:
            wlen = npts * 0.05
        # taper values
        func = _get_function_from_entry_point("taper", "hann")
        if 2 * wlen == npts:
            taper_sides = func(2 * wlen)
        else:
            taper_sides = func(2 * wlen + 1)
        # taper window
        win = np.hstack(
            (
                taper_sides[:wlen],
                np.ones(npts - 2 * wlen),
                taper_sides[len(taper_sides) - wlen :],
            )
        )
        for ii in range(data.shape[0]):
            data[ii] *= win
    return data


# @jit(nopython = True)


# change the moving average calculation to take as input N the full window length to smooth
def moving_ave(A, N):
    """
    Alternative function for moving average for an array.
    PARAMETERS:
    ---------------------
    A: 1-D array of data to be smoothed
    N: integer, it defines the full!! window length to smooth
    RETURNS:
    ---------------------
    B: 1-D array with smoothed data
    """
    # defines an array with N extra samples at either side
    temp = np.zeros(len(A) + 2 * N)
    # set the central portion of the array to A
    temp[N:-N] = A
    # leading samples: equal to first sample of actual array
    temp[0:N] = temp[N]
    # trailing samples: Equal to last sample of actual array
    temp[-N:] = temp[-N - 1]
    # convolve with a boxcar and normalize, and use only central portion of the result
    # with length equal to the original array, discarding the added leading and trailing samples
    B = np.convolve(temp, np.ones(N) / N, mode="same")[N:-N]
    return B


# change the moving average calculation to take as input N the full window length to smooth
def moving_ave_2D(A, N):
    """
    Alternative function for moving average for an array.
    PARAMETERS:
    ---------------------
    A: 2-D array of data to be smoothed
    N: integer, it defines the full!! window length to smooth
    RETURNS:
    ---------------------
    B: 2-D array with smoothed data
    """
    ntc, nspt = A.shape
    # defines an array with N extra samples at either side
    temp = np.zeros([ntc, nspt + 2 * N])
    # set the central portion of the array to A
    temp[:, N:-N] = A
    # leading samples: equal to first sample of actual array
    temp[:, 0:N] = np.repeat(np.expand_dims(temp[:, N], axis=-1), N, axis=-1)
    # trailing samples: Equal to last sample of actual array
    temp[:, -N:] = np.repeat(np.expand_dims(temp[:, -N - 1], axis=-1), N, axis=-1)
    # convolve with a boxcar and normalize, and use only central portion of the result
    # with length equal to the original array, discarding the added leading and trailing samples
    B = scipy.signal.convolve2d(temp, np.expand_dims(np.ones(N) / N, axis=0), mode="same")[:, N:-N]
    return B


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


def whiten_1D(timeseries, fft_para: ConfigParameters, n_taper):
    """
    This function takes a 1-dimensional timeseries array, transforms to frequency domain using fft,
    whitens the amplitude of the spectrum in frequency domain between *freqmin* and *freqmax*
    and returns the whitened fft.
    PARAMETERS:
    ----------------------
    data: numpy.ndarray contains the 1D time series to whiten
    fft_para: ConfigParameters class containing all fft_cc parameters such as
        dt: The sampling space of the `data`
        freqmin: The lower frequency bound
        freqmax: The upper frequency bound
        smooth_N: integer, it defines the half window length to smooth
        n_taper, optional: integer, define the width of the taper in samples
    RETURNS:
    ----------------------
    FFTRawSign: numpy.ndarray contains the FFT of the whitened input trace between the frequency bounds
    """
    nfft = next_fast_len(len(timeseries))
    spec = np.fft.fft(timeseries, nfft)
    freq = np.fft.fftfreq(nfft, d=fft_para.dt)

    ix0 = np.argmin(np.abs(freq - fft_para.freqmin))
    ix1 = np.argmin(np.abs(freq - fft_para.freqmax))

    if ix1 + n_taper > nfft:
        ix11 = nfft
    else:
        ix11 = ix1 + n_taper

    if ix0 - n_taper < 0:
        ix00 = 0
    else:
        ix00 = ix0 - n_taper

    spec_out = spec.copy()
    spec_out[0:ix00] = 0.0 + 0.0j
    spec_out[ix11:] = 0.0 + 0.0j

    if fft_para.smooth_N <= 1:
        spec_out[ix00:ix11] = np.exp(1.0j * np.angle(spec_out[ix00:ix11]))
    else:
        spec_out[ix00:ix11] /= moving_ave(np.abs(spec_out[ix00:ix11]), fft_para.smooth_N)

    x = np.linspace(np.pi / 2.0, np.pi, ix0 - ix00)
    spec_out[ix00:ix0] *= np.cos(x) ** 2

    x = np.linspace(0.0, np.pi / 2.0, ix11 - ix1)
    spec_out[ix1:ix11] *= np.cos(x) ** 2

    return spec_out


def whiten_2D(timeseries, fft_para: ConfigParameters, n_taper):
    """
    This function takes a 2-dimensional timeseries array, transforms to frequency domain using fft,
    whitens the amplitude of the spectrum in frequency domain between *freqmin* and *freqmax*
    and returns the whitened fft.
    PARAMETERS:
    ----------------------
    data: numpy.ndarray contains the 1D time series to whiten
    fft_para: ConfigParameters class containing all fft_cc parameters such as
        dt: The sampling space of the `data`
        freqmin: The lower frequency bound
        freqmax: The upper frequency bound
        smooth_N: integer, it defines the half window length to smooth
        n_taper, optional: integer, define the width of the taper in samples
    RETURNS:
    ----------------------
    FFTRawSign: numpy.ndarray contains the FFT of the whitened input trace between the frequency bounds
    """
    nfft = next_fast_len(timeseries.shape[1])
    spec = np.fft.fftn(timeseries, s=[nfft])
    freq = np.fft.fftfreq(nfft, d=fft_para.dt)

    ix0 = np.argmin(np.abs(freq - fft_para.freqmin))
    ix1 = np.argmin(np.abs(freq - fft_para.freqmax))

    if ix1 + n_taper > nfft:
        ix11 = nfft
    else:
        ix11 = ix1 + n_taper

    if ix0 - n_taper < 0:
        ix00 = 0
    else:
        ix00 = ix0 - n_taper

    spec_out = spec.copy()  # may be inconvenient due to higher memory usage
    spec_out[:, 0:ix00] = 0.0 + 0.0j
    spec_out[:, ix11:] = 0.0 + 0.0j

    if fft_para.smooth_N <= 1:
        spec_out[:, ix00:ix11] = np.exp(1.0j * np.angle(spec_out[:, ix00:ix11]))
    else:
        spec_out[:, ix00:ix11] /= moving_ave_2D(np.abs(spec_out[:, ix00:ix11]), fft_para.smooth_N)

    x = np.linspace(np.pi / 2.0, np.pi, ix0 - ix00)
    spec_out[:, ix00:ix0] *= np.cos(x) ** 2

    x = np.linspace(0.0, np.pi / 2.0, ix11 - ix1)
    spec_out[:, ix1:ix11] *= np.cos(x) ** 2

    return spec_out


def whiten(data, fft_para: ConfigParameters, n_taper=100):
    """
    This function takes a timeseries array, transforms to frequency domain using fft,
    whitens the amplitude of the spectrum in frequency domain between *freqmin* and *freqmax*
    and returns the whitened fft.
    PARAMETERS:
    ----------------------
    data: numpy.ndarray contains the 1D time series to whiten
    fft_para: ConfigParameters class containing all fft_cc parameters such as
        dt: The sampling space of the `data`
        freqmin: The lower frequency bound
        freqmax: The upper frequency bound
        smooth_N: integer, it defines the half window length to smooth
        freq_norm: whitening method between 'one-bit' and 'RMA'
    RETURNS:
    ----------------------
    FFTRawSign: numpy.ndarray contains the FFT of the whitened input trace between the frequency bounds
    """

    # Speed up FFT by padding to optimal size for FFTPACK
    if data.ndim == 1:
        FFTRawSign = whiten_1D(data, fft_para, n_taper)
        # ARR_OUT: Only for consistency with noisepy approach of holding the full
        # spectrum (not just 0 and positive freq. part)
        arr_out = np.zeros((FFTRawSign.shape[0] - 1) * 2 + 1, dtype=complex)
        arr_out[0 : FFTRawSign.shape[0]] = FFTRawSign
        arr_out[FFTRawSign.shape[0] :] = FFTRawSign[1:].conjugate()[::-1]

    elif data.ndim == 2:
        FFTRawSign = whiten_2D(data, fft_para, n_taper)
        arr_out = np.zeros((FFTRawSign.shape[0], (FFTRawSign.shape[1] - 1) * 2 + 1), dtype=complex)
        arr_out[:, FFTRawSign.shape[1] :] = FFTRawSign[:, 1:].conjugate()[::-1]
    return FFTRawSign


def adaptive_filter(arr, g):
    """
    the adaptive covariance filter to enhance coherent signals. Fellows the method of
    Nakata et al., 2015 (Appendix B)

    the filtered signal [x1] is given by x1 = ifft(P*x1(w)) where x1 is the ffted spectra
    and P is the filter. P is constructed by using the temporal covariance matrix.

    PARAMETERS:
    ----------------------
    arr: numpy.ndarray contains the 2D traces of daily/hourly cross-correlation functions
    g: a positive number to adjust the filter harshness
    RETURNS:
    ----------------------
    narr: numpy vector contains the stacked cross correlation function
    """
    if arr.ndim == 1:
        return arr
    N, M = arr.shape
    Nfft = next_fast_len(M)

    # fft the 2D array
    spec = scipy.fftpack.fft(arr, axis=1, n=Nfft)[:, :M]

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


def pws(arr, sampling_rate, power=2, pws_timegate=5.0):
    """
    Performs phase-weighted stack on array of time series. Modified on the noise function by Tim Climents.
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
    """

    if arr.ndim == 1:
        return arr
    N, M = arr.shape
    analytic = hilbert(arr, axis=1, N=next_fast_len(M))[:, :M]
    phase = np.angle(analytic)
    phase_stack = np.mean(np.exp(1j * phase), axis=0)
    phase_stack = np.abs(phase_stack) ** (power)

    # smoothing
    # timegate_samples = int(pws_timegate * sampling_rate)
    # phase_stack = moving_ave(phase_stack,timegate_samples)
    weighted = np.multiply(arr, phase_stack)
    return np.mean(weighted, axis=0)


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
        logger.debug("2D matrix is needed for nroot_stack")
        return cc_array
    N, M = cc_array.shape
    dout = np.zeros(M, dtype=np.float32)

    # construct y
    for ii in range(N):
        dat = cc_array[ii, :]
        dout += np.sign(dat) * np.abs(dat) ** (1 / power)
    dout /= N

    # the final stacked waveform
    nstack = dout * np.abs(dout) ** (power - 1)

    return nstack


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


################################################################
################ DISPERSION EXTRACTION FUNCTIONS ###############
################################################################


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
