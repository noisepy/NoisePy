import gc
import logging
import sys
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import obspy
import ray
from datetimerange import DateTimeRange
from scipy.fftpack.helper import next_fast_len

from noisepy.seis.datatypes import Channel, ChannelData, ConfigParameters, NoiseFFT

from . import noise_module
from .stores import CrossCorrelationDataStore, RawDataStore
from .utils import TimeLogger

logger = logging.getLogger(__name__)
# ignore warnings
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

"""
This main script of NoisePy:
    1) read the saved noise data in user-defined chunk of inc_hours, cut them into
    smaller length segments, do general pre-processing
    (trend, normalization) and then do FFT;
    2) save all FFT data of the same time chunk in memory;
    3) performs cross-correlation for all station pairs in the same time chunk and
    output the sub-stacked (if selected) into ASDF format;
Authors: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
         Marine Denolle (mdenolle@fas.harvard.edu)

NOTE:
    0. MOST occasions you just need to change parameters followed with detailed
    explanations to run the script.
    1. To read SAC/mseed files, we assume the users have sorted the data
    by the time chunk they prefer (e.g., 1day)
        and store them in folders named after the time chunk (e.g, 2010_10_1).
        modify L135 to find your local data;
    2. A script of S0B_to_ASDF.py is provided to help clean messy SAC/MSEED data and
    convert them into ASDF format.
        the script takes minor time compared to that for cross-correlation.
        so we recommend to use S0B script for
        better NoisePy performance. the downside is that it duplicates the
        continuous noise data on your machine;
    3. When "coherency" is preferred, please set "freq_norm" to "rma" and "time_norm" to "no"
    for better performance.
"""


def cross_correlate(
    raw_store: RawDataStore,
    fft_params: ConfigParameters,
    cc_store: CrossCorrelationDataStore,
):
    """
    Perform the cross-correlation analysis

        Parameters:
                raw_store: Store to load data from
                fft_params: Parameters for the FFT calculations
                cc_store: Store for saving cross correlations

    """

    tlog = TimeLogger(logger, logging.INFO)
    t_s1_total = tlog.reset()

    if not ray.is_initialized():
        context = ray.init(ignore_reinit_error=True)
        tlog.log("Ray init")
        logger.info(context.dashboard_url)

    # save metadata
    cc_store.save_parameters(fft_params)

    # set variables to broadcast
    timespans = raw_store.get_timespans()
    splits = len(timespans)
    if splits == 0:
        raise IOError("Abort! no available seismic files for FFT")

    for ts in timespans:
        if cc_store.is_done(ts):
            continue

        """
        LOADING NOISE DATA AND DO FFT
        """
        nnfft = int(next_fast_len(int(fft_params.cc_len * fft_params.samp_freq)))  # samp_freq should be sampling_rate

        t_chunk = tlog.reset()  # for tracking overall chunk processing time
        all_channels = raw_store.get_channels(ts)
        tlog.log("get channels")
        ch_data_tuples = _read_channels(ts, raw_store, all_channels, fft_params.samp_freq)
        # only the channels we are using
        channels = list(zip(*ch_data_tuples))[0]
        tlog.log("read channel data")

        ch_data_tuples = preprocess(ch_data_tuples, raw_store, fft_params, ts)
        tlog.log("preprocess")

        nchannels = len(ch_data_tuples)
        nseg_chunk = check_memory(fft_params, nchannels)
        # Dictionary to store all the FFTs, keyed by channel index
        ffts: Dict[int, NoiseFFT] = OrderedDict()

        logger.debug(f"nseg_chunk: {nseg_chunk}, nnfft: {nnfft}")
        # loop through all channels
        tlog.reset()

        fft_refs = [compute_fft_ray.remote(fft_params, chd[1]) for chd in ch_data_tuples]
        fft_datas = ray.get(fft_refs)
        for ix_ch, fft_data in enumerate(fft_datas):
            if fft_data.fft.size > 0:
                ffts[ix_ch] = fft_data
            else:
                logger.warning(f"No data available for channel '{channels[ix_ch]}', skipped")
        Nfft = fft_data.length
        t_tasks = tlog.log("Compute FFTs")

        if len(ffts) != nchannels:
            logger.warning("it seems some stations miss data in download step, but it is OKAY!")

        tasks = []

        # Put the FFTs into Ray's shared memory since they will be used by all tasks
        ffts_ref = ray.put(ffts)
        tlog.log("ray.put(ffts)")
        # # ###########PERFORM CROSS-CORRELATION##################
        for iiS in ffts.keys():  # looping over the channel source
            # We parallelize over the channel sources. This is less than ideal because for each subsequent
            # source there are fewer receiving channels to correlate with (ie. its a diagonal) matrix.
            # This means the first task is the longest and the last one if very short. However,
            # parallelizing over the full set of pairs results in too many tiny tasks and the parallelization
            # overhead outweighs the benefits.
            task = source_cross_correlation.remote(fft_params, channels, ffts_ref, Nfft, iiS)
            tasks.append(task)
        tlog.log(f"Created {len(tasks)} CC tasks", t_tasks)
        while len(tasks) > 0:
            # Use partial waits so we can start savign results to the store
            # while other computations are still running
            ready, tasks = ray.wait(tasks, num_returns=min(len(tasks), 4), timeout=0.250)
            results = ray.get(ready)
            results = [r for subresult in results for r in subresult]
            for src_chan, rec_chan, parameters, corr in results:
                cc_store.append(ts, src_chan, rec_chan, fft_params, parameters, corr)

        ffts.clear()
        gc.collect()

        tlog.log(f"Process the chunk of {ts}", t_chunk)
        cc_store.mark_done(ts)

    tlog.log("Step 1 in total", t_s1_total)


@ray.remote
def source_cross_correlation(
    fft_params: ConfigParameters,
    channels: List[Channel],
    ffts: Dict[int, NoiseFFT],
    Nfft: int,
    iiS: int,
) -> List[Tuple[Channel, Channel, dict, np.ndarray]]:
    src_chan = channels[iiS]  # this is the name of the source channel
    src_fft = ffts[iiS]
    src_std = src_fft.std
    # this finds the windows of "good" noise
    sou_ind = np.where((src_std < fft_params.max_over_std) & (src_std > 0) & (np.isnan(src_std) == 0))[0]
    if len(sou_ind) == 0:
        return None

    # in the case of pure deconvolution, we recommend smoothing anyway.
    if fft_params.cc_method == "deconv":
        # -----------get the smoothed source spectrum for decon later----------
        sfft1 = noise_module.smooth_source_spect(fft_params, src_fft.fft)
        sfft1 = sfft1.reshape(src_fft.window_count, src_fft.length // 2)
    else:
        sfft1 = np.conj(src_fft.fft).reshape(src_fft.window_count, src_fft.length // 2)

        # get index right for auto/cross correlation
    istart = iiS  # start at the channel source / only fills the upper right triangle matrix of channel pairs
    iend = len(channels)
    # -----------now loop III for each receiver B----------
    results = []
    for iiR in range(istart, iend):
        rec_chan = channels[iiR]
        if fft_params.acorr_only:
            if src_chan.station != rec_chan.station:
                continue
        if iiR not in ffts:
            continue
        result = cross_corr(fft_params, src_chan, rec_chan, sfft1, sou_ind, ffts[iiR], Nfft)
        results.append(result)
    del sfft1
    return results


def preprocess(
    ch_data: List[Tuple[Channel, ChannelData]], raw_store: RawDataStore, fft_params: ConfigParameters, ts: DateTimeRange
) -> List[Tuple[Channel, ChannelData]]:
    stream_refs = [preprocess_ray.remote(raw_store, t[0], t[1], fft_params, ts) for t in ch_data]
    new_streams = ray.get(stream_refs)
    channels = list(zip(*ch_data))[0]
    return list(zip(channels, [ChannelData(st) for st in new_streams]))


def cross_corr(
    fft_params: ConfigParameters,
    src_chan: Channel,
    rec_chan: Channel,
    sfft1: np.ndarray,
    sou_ind: np.ndarray,
    rec_fft: NoiseFFT,
    Nfft: int,
) -> Tuple[Channel, Channel, dict, np.ndarray]:
    logger.debug(f"receiver: {rec_chan}")
    # read the receiver data
    sfft2 = rec_fft.fft.reshape(rec_fft.window_count, rec_fft.length // 2)
    rec_std = rec_fft.std

    # ---------- check the existence of earthquakes or spikes ----------
    rec_ind = np.where((rec_std < fft_params.max_over_std) & (rec_std > 0) & (np.isnan(rec_std) == 0))[0]
    bb = np.intersect1d(sou_ind, rec_ind)
    if len(bb) == 0:
        return

    # ----------- GAME TIME: cross correlation step ---------------
    corr, tcorr, ncorr = noise_module.correlate(sfft1[bb, :], sfft2[bb, :], fft_params, Nfft, rec_fft.fft_time[bb])

    del sfft2
    # ---------- OUTPUT: store metadata and data into file ------------
    coor = {
        "lonS": src_chan.station.lon,
        "latS": src_chan.station.lat,
        "lonR": rec_chan.station.lon,
        "latR": rec_chan.station.lat,
    }
    comp = src_chan.type.get_orientation() + rec_chan.type.get_orientation()
    parameters = noise_module.cc_parameters(fft_params, coor, tcorr, ncorr, comp)
    return (src_chan, rec_chan, parameters, corr)


@ray.remote
def preprocess_ray(
    raw_store: RawDataStore, ch: Channel, ch_data: ChannelData, fft_params: ConfigParameters, ts: DateTimeRange
) -> obspy.Stream:
    return noise_module.preprocess_raw(
        ch_data.stream.copy(),  # If we don't copy it's not writeable
        raw_store.get_inventory(ts, ch.station),
        fft_params,
        obspy.UTCDateTime(ts.start_datetime),
        obspy.UTCDateTime(ts.end_datetime),
    )


@ray.remote
def compute_fft_ray(fft_params: ConfigParameters, ch_data: ChannelData) -> NoiseFFT:
    if ch_data.data.size == 0:
        return NoiseFFT(np.empty(0), np.empty(0), np.empty(0), 0, 0)

    # cut daily-long data into smaller segments (dataS always in 2D)
    trace_stdS, dataS_t, dataS = noise_module.cut_trace_make_stat(
        fft_params, ch_data
    )  # optimized version:3-4 times faster
    if not len(dataS):
        return NoiseFFT(np.empty(0), np.empty(0), np.empty(0), 0, 0)

    N = dataS.shape[0]

    # do normalization if needed
    source_white = noise_module.noise_processing(fft_params, dataS)
    Nfft = source_white.shape[1]
    Nfft2 = Nfft // 2
    logger.debug(f"N and Nfft are {N}, {Nfft}")

    # load fft data in memory for cross-correlations
    data = source_white[:, :Nfft2]
    fft = data.reshape(data.size)
    std = trace_stdS
    fft_time = dataS_t
    del trace_stdS, dataS_t, dataS, source_white, data
    return NoiseFFT(fft, std, fft_time, N, Nfft)


def _read_channels(
    ts: DateTimeRange, store: RawDataStore, channels: List[Channel], samp_freq: int
) -> List[Tuple[Channel, ChannelData]]:
    ch_data_refs = [read_data_ray.remote(store, ts, ch) for ch in channels]
    ch_data = ray.get(ch_data_refs)
    tuples = list(zip(channels, ch_data))
    return _filter_channel_data(tuples, samp_freq)


@ray.remote
def read_data_ray(store: RawDataStore, ts: DateTimeRange, ch: Channel) -> ChannelData:
    return store.read_data(ts, ch)


def _filter_channel_data(
    tuples: List[Tuple[Channel, ChannelData]], samp_freq: int
) -> List[Tuple[Channel, ChannelData]]:
    frequencies = set(t[1].sampling_rate for t in tuples)
    closest_freq = min(
        filter(lambda f: f >= samp_freq, frequencies),
        key=lambda f: max(f - samp_freq, 0),
    )
    filtered_tuples = list(filter(lambda tup: tup[1].sampling_rate == closest_freq, tuples))
    logger.info(
        f"Picked {closest_freq} as the closest sampling frequence to {samp_freq}. "
        f"Filtered to {len(filtered_tuples)}/{len(tuples)} channels"
    )
    return filtered_tuples


def check_memory(params: ConfigParameters, nsta: int) -> int:
    # maximum memory allowed per core in GB
    MAX_MEM = 4.0
    # crude estimation on memory needs (assume float32)
    nsec_chunk = params.inc_hours / 24 * 86400
    nseg_chunk = int(np.floor((nsec_chunk - params.cc_len) / params.step))
    npts_chunk = int(nseg_chunk * params.cc_len * params.samp_freq)
    memory_size = nsta * npts_chunk * 4 / 1024**3
    if memory_size > MAX_MEM:
        raise ValueError(
            "Require %5.3fG memory but only %5.3fG provided)! Reduce inc_hours to avoid this issue!"
            % (memory_size, MAX_MEM)
        )
    return nseg_chunk


# Point people to new entry point:
if __name__ == "__main__":
    print("Please see:\n\npython noisepy.py cross_correlate --help\n")
