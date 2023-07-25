import gc
import logging
import sys
import time
from collections import OrderedDict
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import obspy
from datetimerange import DateTimeRange
from scipy.fftpack.helper import next_fast_len

from . import noise_module
from .datatypes import Channel, ChannelData, ConfigParameters, NoiseFFT
from .scheduler import Scheduler, SingleNodeScheduler
from .stores import CrossCorrelationDataStore, RawDataStore
from .utils import TimeLogger, _get_results, error_if

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
    1. A script of S0B_to_ASDF.py is provided to help clean messy SAC/MSEED data and
    convert them into ASDF format.
        the script takes minor time compared to that for cross-correlation.
        so we recommend to use S0B script for
        better NoisePy performance. the downside is that it duplicates the
        continuous noise data on your machine;
    2. When "coherency" is preferred, please set "freq_norm" to "rma" and "time_norm" to "no"
    for better performance.
"""


def cross_correlate(
    raw_store: RawDataStore,
    fft_params: ConfigParameters,
    cc_store: CrossCorrelationDataStore,
    scheduler: Scheduler = SingleNodeScheduler(),
):
    """
    Perform the cross-correlation analysis

    Args:
        raw_store: Store to load data from
        fft_params: Parameters for the FFT calculations
        cc_store: Store for saving cross correlations
    """

    executor = ThreadPoolExecutor()
    tlog = TimeLogger(logger, logging.INFO)
    t_s1_total = tlog.reset()

    def init() -> List:
        # set variables to broadcast
        timespans = raw_store.get_timespans()
        if len(timespans) == 0:
            raise IOError("Abort! no available seismic files for FFT")
        return [timespans]

    [timespans] = scheduler.initialize(init, 1)

    for its in scheduler.get_indices(timespans):
        ts = timespans[its]
        if cc_store.is_done(ts):
            logger.info(f"{ts} already processed, skipped")
            continue

        """
        LOADING NOISE DATA AND DO FFT
        """
        nnfft = int(next_fast_len(int(fft_params.cc_len * fft_params.samp_freq)))  # samp_freq should be sampling_rate

        t_chunk = tlog.reset()  # for tracking overall chunk processing time
        all_channels = raw_store.get_channels(ts)
        error_if(
            not all(map(lambda c: c.station.valid(), all_channels)),
            "The stations don't have their lat/lon/elev properties populated. Problem with the ChannelCatalog used?",
        )

        tlog.log("get channels")
        ch_data_tuples = _read_channels(executor, ts, raw_store, all_channels, fft_params.samp_freq)
        # only the channels we are using

        if len(ch_data_tuples) == 0:
            logger.warning(f"No data available for {ts}")
            continue

        channels = list(zip(*ch_data_tuples))[0]
        tlog.log("read channel data")

        ch_data_tuples = preprocess_all(executor, ch_data_tuples, raw_store, fft_params, ts)
        tlog.log("preprocess")

        nchannels = len(ch_data_tuples)
        nseg_chunk = check_memory(fft_params, nchannels)
        # Dictionary to store all the FFTs, keyed by channel index
        ffts: Dict[int, NoiseFFT] = OrderedDict()

        logger.debug(f"nseg_chunk: {nseg_chunk}, nnfft: {nnfft}")
        # loop through all channels
        tlog.reset()

        fft_refs = [executor.submit(compute_fft, fft_params, chd[1]) for chd in ch_data_tuples]
        fft_datas = _get_results(fft_refs)
        for ix_ch, fft_data in enumerate(fft_datas):
            if fft_data.fft.size > 0:
                ffts[ix_ch] = fft_data
            else:
                logger.warning(f"No data available for channel '{channels[ix_ch]}', skipped")
        Nfft = fft_data.length
        tlog.log("Compute FFTs")

        if len(ffts) != nchannels:
            logger.warning("it seems some stations miss data in download step, but it is OKAY!")

        tasks = []

        # # ###########PERFORM CROSS-CORRELATION##################
        for iiS in range(nchannels):
            for iiR in range(iiS, nchannels):
                src_chan = channels[iiS]
                rec_chan = channels[iiR]
                if fft_params.acorr_only:
                    if src_chan.station != rec_chan.station:
                        continue
                if iiR not in ffts:
                    continue
                t = executor.submit(cross_correlation, fft_params, iiS, iiR, channels, ffts, Nfft)
                tasks.append(t)

        t_append = 0
        for t in as_completed(tasks):
            # Use as_completed so we can start saving results to the store
            # while other computations are still running
            src_chan, rec_chan, parameters, corr = t.result()
            t_start = time.time()
            cc_store.append(ts, src_chan, rec_chan, parameters, corr)
            t_append += time.time() - t_start
        tlog.log_raw("Append to store", t_append)

        ffts.clear()
        gc.collect()

        tlog.log(f"Process the chunk of {ts}", t_chunk)
        cc_store.mark_done(ts)

    tlog.log("Step 1 in total", t_s1_total)
    executor.shutdown()


def cross_correlation(
    fft_params: ConfigParameters,
    iiS: int,
    iiR: int,
    channels: List[Channel],
    ffts: Dict[int, NoiseFFT],
    Nfft: int,
) -> Tuple[Channel, Channel, dict, np.ndarray]:
    src_chan = channels[iiS]  # this is the name of the source channel
    rec_chan = channels[iiR]
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

    result = cross_corr(fft_params, src_chan, rec_chan, sfft1, sou_ind, ffts[iiR], Nfft)
    return result


def preprocess_all(
    executor: Executor,
    ch_data: List[Tuple[Channel, ChannelData]],
    raw_store: RawDataStore,
    fft_params: ConfigParameters,
    ts: DateTimeRange,
) -> List[Tuple[Channel, ChannelData]]:
    stream_refs = [executor.submit(preprocess, raw_store, t[0], t[1], fft_params, ts) for t in ch_data]
    new_streams = _get_results(stream_refs)
    # Log if any streams were removed during pre-processing
    for ch, st in zip(ch_data, new_streams):
        if len(st) == 0:
            logging.warning(
                f"Empty stream for {ts}/{ch[0]} after pre-processing. "
                f"Before pre-processing data.shape: {ch[1].data.shape}, len(stream): {len(ch[1].stream)}"
            )

    # Filter to only non-empty streams
    channels = list(zip(*ch_data))[0]
    non_empty = filter(lambda tup: len(tup[1]) > 0, zip(channels, new_streams))
    return [(ch, ChannelData(st)) for ch, st in non_empty]


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


def preprocess(
    raw_store: RawDataStore, ch: Channel, ch_data: ChannelData, fft_params: ConfigParameters, ts: DateTimeRange
) -> obspy.Stream:
    inv = raw_store.get_inventory(ts, ch.station)
    ch_inv = inv.select(channel=ch.type.name, time=ts.start_datetime)
    # if we don't find an inventory when filtering by time, back off and try
    # without this constraint
    if len(ch_inv) < 1:
        ch_inv = inv.select(channel=ch.type.name)

    return noise_module.preprocess_raw(
        ch_data.stream.copy(),  # If we don't copy it's not writeable
        ch_inv,
        fft_params,
        obspy.UTCDateTime(ts.start_datetime),
        obspy.UTCDateTime(ts.end_datetime),
    )


def compute_fft(fft_params: ConfigParameters, ch_data: ChannelData) -> NoiseFFT:
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
    executor: Executor, ts: DateTimeRange, store: RawDataStore, channels: List[Channel], samp_freq: int
) -> List[Tuple[Channel, ChannelData]]:
    ch_data_refs = [executor.submit(store.read_data, ts, ch) for ch in channels]
    ch_data = _get_results(ch_data_refs)
    tuples = list(zip(channels, ch_data))
    return _filter_channel_data(tuples, samp_freq)


def _filter_channel_data(
    tuples: List[Tuple[Channel, ChannelData]], samp_freq: int
) -> List[Tuple[Channel, ChannelData]]:
    frequencies = set(t[1].sampling_rate for t in tuples)
    frequencies = list(filter(lambda f: f >= samp_freq, frequencies))
    if len(frequencies) == 0:
        logging.warning(f"No data available with sampling frequency >= {samp_freq}")
        return []
    closest_freq = min(
        frequencies,
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
