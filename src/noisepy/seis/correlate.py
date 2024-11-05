import gc
import logging
import os
import sys
from collections import OrderedDict, defaultdict
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import obspy
from datetimerange import DateTimeRange
from scipy.fftpack.helper import next_fast_len

from noisepy.seis.io.datatypes import (
    CCMethod,
    Channel,
    ChannelData,
    ConfigParameters,
    CrossCorrelation,
    NoiseFFT,
    Station,
)
from noisepy.seis.io.stores import CrossCorrelationDataStore, RawDataStore
from noisepy.seis.io.utils import TimeLogger, get_results

from . import noise_module
from .constants import NO_DATA_MSG
from .scheduler import Scheduler, SingleNodeScheduler

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
    pair_filter: Callable[[Channel, Channel], bool] = lambda src, rec: True,
):
    """
    Perform the cross-correlation analysis

    Args:
        raw_store: Store to load data from
        fft_params: Parameters for the FFT calculations
        cc_store: Store for saving cross correlations
        scheduler: Scheduler to use for parallelization
        pair_filter: Function to decide whether a pair of channels should be used or not. E.g.

        .. code-block:: python

            def filter_by_lat(s: Channel, d: Channel) -> bool:
                return abs(s.station.lat - d.station.lat) > 0.1

            cross_correlate(..., pair_filter=filter_by_lat)
    """
    # Force config validation
    fft_params = ConfigParameters.model_validate(dict(fft_params), strict=True)

    tlog = TimeLogger(logger, logging.INFO, prefix="CC Main")
    t_s1_total = tlog.reset()
    logger.info(f"Starting Cross-Correlation with {os.cpu_count()} cores")

    def init() -> List:
        # set variables to broadcast
        timespans = raw_store.get_timespans()
        if len(timespans) == 0:
            raise IOError(NO_DATA_MSG)
        return [timespans]

    [timespans] = scheduler.initialize(init, 1)
    failed = []
    for its in scheduler.get_indices(timespans):
        ts = timespans[its]
        failed_pairs = cc_timespan(raw_store, fft_params, cc_store, ts, pair_filter)
        if len(failed_pairs) > 0:
            failed.extend((ts, failed_pairs))

    tlog.log(f"Step 1 in total with {os.cpu_count()} cores", t_s1_total)
    if len(failed):
        failed_str = "\n".join(map(str, failed))
        logger.error(
            "Errors occurred during cross-correlation. Check logs for details. "
            f"The following pairs failed:\n{failed_str}"
        )


def cc_timespan(
    raw_store: RawDataStore,
    fft_params: ConfigParameters,
    cc_store: CrossCorrelationDataStore,
    ts: DateTimeRange,
    pair_filter: Callable[[Channel, Channel], bool] = lambda src, rec: True,
) -> List[Tuple[Station, Station]]:
    executor = ThreadPoolExecutor()
    tlog = TimeLogger(logger, logging.INFO, prefix="CC Main")
    """
    LOADING NOISE DATA AND DO FFT
    """
    nnfft = int(next_fast_len(int(fft_params.cc_len * fft_params.sampling_rate)))

    t_chunk = tlog.reset()  # for tracking overall chunk processing time
    all_channels = raw_store.get_channels(ts)
    all_channel_count = len(all_channels)
    tlog.log(f"get {all_channel_count} channels")
    all_channels = list(filter(lambda c: c.station.valid(), all_channels))
    all_stations = set([c.station for c in all_channels])
    if all_channel_count > len(all_channels):
        logger.warning(
            f"Some stations were filtered due to missing catalog information (lat/lon/elen). "
            f"Using {len(all_channels)}/{all_channel_count}"
        )
    tlog.reset()
    station_pairs = list(create_pairs(pair_filter, all_channels, fft_params.acorr_only).keys())
    # Check for stations that are already done, do this in parallel
    logger.info(f"Checking for stations already done: {len(station_pairs)} pairs")

    stations = set([station for pair in station_pairs for station in pair])
    _ = list(executor.map(lambda s: cc_store.contains(s, s, ts), stations))
    tlog.log(f"check for {len(stations)} stations already done (warm up cache)")
    station_pair_dones = list(executor.map(lambda p: cc_store.contains(p[0], p[1], ts), station_pairs))

    missing_pairs = [pair for pair, done in zip(station_pairs, station_pair_dones) if not done]
    # get a set of unique stations from the list of pairs
    missing_stations = set([station for pair in missing_pairs for station in pair])
    # Filter the channels to only the missing stations
    missing_channels = list(filter(lambda c: c.station in missing_stations, all_channels))
    tlog.log("check for stations already done")

    logger.info(
        f"Still need to process: {len(missing_stations)}/{len(all_stations)} stations, "
        f"{len(missing_channels)}/{len(all_channels)} channels, "
        f"{len(missing_pairs)}/{len(station_pairs)} pairs "
        f"for {ts}"
    )
    if len(missing_channels) == 0:
        logger.warning(f"{ts} already completed")
        return []

    ch_data_tuples = _read_channels(
        executor, ts, raw_store, missing_channels, fft_params.sampling_rate, fft_params.single_freq
    )
    # only the channels we are using

    if len(ch_data_tuples) == 0:
        logger.warning(f"No data available for {ts}")
        return missing_pairs

    tlog.log(f"Read channel data: {len(ch_data_tuples)} channels")
    ch_data_tuples_pre = preprocess_all(executor, ch_data_tuples, raw_store, fft_params, ts)
    del ch_data_tuples
    tlog.log(f"Preprocess: {len(ch_data_tuples_pre)} channels")
    if len(ch_data_tuples_pre) == 0:
        logger.warning(f"No data available for {ts} after preprocessing")
        return missing_pairs

    nchannels = len(ch_data_tuples_pre)
    nseg_chunk = check_memory(fft_params, nchannels)
    # Dictionary to store all the FFTs, keyed by channel index
    ffts: Dict[int, NoiseFFT] = OrderedDict()

    logger.debug(f"nseg_chunk: {nseg_chunk}, nnfft: {nnfft}")
    # loop through all channels
    tlog.reset()

    fft_refs = [executor.submit(compute_fft, fft_params, chd[1]) for chd in ch_data_tuples_pre]
    # Important: get the list of channels at this point and not before because some
    # tuples could have been removed during pre-processing
    channels = list(zip(*ch_data_tuples_pre))[0]
    # Done with the raw data, clear it out
    ch_data_tuples_pre.clear()
    del ch_data_tuples_pre
    gc.collect()
    fft_datas = get_results(fft_refs, "Compute ffts")
    for ix_ch, fft_data in enumerate(fft_datas):
        if fft_data.fft.size > 0:
            ffts[ix_ch] = fft_data
        else:
            logger.warning(f"No data available for channel '{channels[ix_ch]}', skipped")
    tlog.log(f"Compute FFTs: {len(ffts)} channels")
    Nfft = max(map(lambda d: d.length, fft_datas))
    if Nfft == 0:
        logger.error(f"No FFT data available for any channel in {ts}, skipping")
        return missing_pairs

    if len(ffts) != nchannels:
        logger.warning("it seems some stations miss data in download step, but it is OKAY!")

    tasks = []

    station_pairs = create_pairs(pair_filter, channels, fft_params.acorr_only, ffts)
    tlog.reset()

    save_exec = ThreadPoolExecutor()
    work_items = list(station_pairs.items())
    work_items = sorted(work_items, key=lambda t: t[0][0].name + t[0][1].name)
    logger.info(f"Starting CC with {len(work_items)} station pairs")
    for station_pair, ch_pairs in work_items:
        t = executor.submit(
            stations_cross_correlation,
            ts,
            fft_params,
            station_pair[0],
            station_pair[1],
            channels,
            ch_pairs,
            ffts,
            Nfft,
            cc_store,
            save_exec,
        )
        tasks.append(t)
    compute_results = get_results(tasks, "Cross correlation")
    _, save_tasks = zip(*compute_results)
    save_tasks = [t for t in save_tasks if t]
    _ = get_results(save_tasks, "Save correlations")
    failed_pairs = [
        pair[0]
        for pair, (comp_res, save_task) in zip(work_items, compute_results)
        if not (comp_res and (save_task is None or save_task.result()))
    ]

    save_exec.shutdown()
    tlog.log("Correlate and write to store")

    ffts.clear()
    gc.collect()

    tlog.log(f"Process the chunk of {ts}", t_chunk)
    executor.shutdown()
    return failed_pairs


def create_pairs(
    pair_filter: Callable[[Channel, Channel], bool],
    channels: List[Channel],
    acorr_only: bool,
    ffts: Optional[Dict[int, NoiseFFT]] = None,
) -> Dict[Tuple[Station, Station], List[Tuple[int, int]]]:
    station_pairs = defaultdict(list)
    nchannels = len(channels)
    for iiS in range(nchannels):
        for iiR in range(iiS, nchannels):
            src_chan = channels[iiS]
            rec_chan = channels[iiR]
            if not pair_filter(src_chan, rec_chan):
                continue
            if acorr_only:
                if src_chan.station != rec_chan.station:
                    continue
            if ffts and iiS not in ffts:
                logger.warning(f"No FFT data available for src channel '{src_chan}', skipped")
                continue
            if ffts and iiR not in ffts:
                logger.warning(f"No FFT data available for rec channel '{rec_chan}', skipped")
                continue

            station_pairs[(src_chan.station, rec_chan.station)].append((iiS, iiR))
    return station_pairs


def stations_cross_correlation(
    ts: DateTimeRange,
    fft_params: ConfigParameters,
    src: Station,
    rec: Station,
    channels: List[Channel],
    channel_pairs: List[Tuple[int, int]],
    ffts: Dict[int, NoiseFFT],
    Nfft: int,
    cc_store: CrossCorrelationDataStore,
    executor: Executor,
) -> Tuple[bool, Future]:
    tlog = TimeLogger(logger, logging.DEBUG)
    datas = []
    try:
        if cc_store.contains(src, rec, ts):
            logger.info(f"Skipping {src}_{rec} for {ts} since it's already done")
            return True, None

        # TODO: Are there any potential gains to parallelliing this? It could make a difference if
        # num station pairs < num cores since we are already parallelizing at the station pair level
        for src_chan, rec_chan in channel_pairs:
            assert channels[src_chan].station == src
            assert channels[rec_chan].station == rec
            result = cross_correlation(fft_params, src_chan, rec_chan, channels, ffts, Nfft)
            if result is not None:
                data = CrossCorrelation(result[0].type, result[1].type, result[2], result[3])
                datas.append(data)
        tlog.log(f"Cross-correlated {len(datas)} pairs for {src} and {rec} for {ts}")
        save_future = executor.submit(save, cc_store, ts, src, rec, datas)
        return True, save_future
    except Exception as e:
        logger.error(f"Error processing {src} and {rec} for {ts}: {e}")
        return False, None


def save(
    store: CrossCorrelationDataStore, ts: DateTimeRange, src: Station, rec: Station, datas: List[CrossCorrelation]
) -> bool:
    try:
        store.append(ts, src, rec, datas)
        return True
    except Exception as e:
        logger.error(f"Error saving {src} and {rec} for {ts}: {e}")
        return False


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
        logger.warning(f"no good data for source: {src_chan}")
        return None

    # in the case of pure deconvolution, we recommend smoothing anyway.
    if fft_params.cc_method == CCMethod.DECONV:
        # -----------get the smoothed source spectrum for decon later----------
        sfft1 = noise_module.smooth_source_spect(fft_params, src_fft.fft)
        sfft1 = sfft1.reshape(
            src_fft.window_count, src_fft.length // 2
        )  # conjugate already included in smooth_source_spect
    else:
        sfft1 = np.conj(src_fft.fft).reshape(src_fft.window_count, src_fft.length // 2)

    # note here that sfft1 and ffts have gone through noise_processing already:
    # if FreqNorm is not None, then they have been whitened already.

    result = cross_corr(fft_params, src_chan, rec_chan, sfft1, sou_ind, ffts[iiR], Nfft)
    return result


def preprocess_all(
    executor: Executor,
    ch_data: List[Tuple[Channel, ChannelData]],
    raw_store: RawDataStore,
    fft_params: ConfigParameters,
    ts: DateTimeRange,
) -> List[Tuple[Channel, ChannelData]]:
    channels = list(zip(*ch_data))[0]
    stream_refs = [executor.submit(preprocess, raw_store, t[0], t[1], fft_params, ts) for t in ch_data]
    del ch_data
    new_streams = get_results(stream_refs, "Pre-process")
    # Log if any streams were removed during pre-processing
    for ch, st in zip(channels, new_streams):
        if len(st) == 0:
            logging.warning(f"Empty stream for {ts}/{ch} after pre-processing. ")

    # Filter to only non-empty streams
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

    # load fft data in memory for cross-correlations
    data = source_white[:, :Nfft2]
    fft = data.reshape(data.size)
    std = trace_stdS
    fft_time = dataS_t
    del trace_stdS, dataS_t, dataS, source_white, data
    return NoiseFFT(fft, std, fft_time, N, Nfft)


def _read_channels(
    executor: Executor,
    ts: DateTimeRange,
    store: RawDataStore,
    channels: List[Channel],
    sampling_rate: int,
    single_freq: bool = True,
) -> List[Tuple[Channel, ChannelData]]:
    ch_data_refs = [executor.submit(_safe_read_data, store, ts, ch) for ch in channels]
    ch_data = get_results(ch_data_refs, "Read channel data")
    tuples = list(filter(lambda tup: tup[1].data.size > 0, zip(channels, ch_data)))

    return _filter_channel_data(tuples, sampling_rate, single_freq)


def _safe_read_data(store: RawDataStore, ts: DateTimeRange, ch: Channel) -> ChannelData:
    try:
        return store.read_data(ts, ch)
    except Exception as e:
        logger.warning(f"Error reading data for {ch} in {ts}: {e}")
        return ChannelData.empty()


def _filter_channel_data(
    tuples: List[Tuple[Channel, ChannelData]], sampling_rate: int, single_freq: bool = True
) -> List[Tuple[Channel, ChannelData]]:
    frequencies = set(t[1].sampling_rate for t in tuples)
    frequencies = list(filter(lambda f: f >= sampling_rate, frequencies))
    if len(frequencies) == 0:
        logging.warning(f"No data available with sampling rate >= {sampling_rate}")
        return []
    if single_freq:
        closest_freq = _get_closest_freq(frequencies, sampling_rate)
        logger.info(f"Picked {closest_freq} as the closest sampling_rate to {sampling_rate}. ")
        filtered_tuples = list(filter(lambda tup: tup[1].sampling_rate == closest_freq, tuples))
        logger.info(f"Filtered to {len(filtered_tuples)}/{len(tuples)} channels with sampling rate == {closest_freq}")
    else:
        filtered_tuples = list(filter(lambda tup: tup[1].sampling_rate >= sampling_rate, tuples))
        # for each station, pick the closest >= to sampling_rate
        tmp = list(
            map(
                lambda s: [t for t in filtered_tuples if t[0].station == s],
                set([t[0].station for t in filtered_tuples]),
            )
        )
        filtered_tuples = sum(list(map(lambda t: _filt_single_station(t, sampling_rate), tmp)), [])
        logger.info(f"Filtered to {len(filtered_tuples)}/{len(tuples)} channels with sampling rate >= {sampling_rate}")

    return filtered_tuples


def _get_closest_freq(frequencies, sampling_rate: int):
    return min(frequencies, key=lambda f: max(f - sampling_rate, 0))


def _filt_single_station(tuples: List[Tuple[Channel, ChannelData]], sampling_rate: int):
    frequencies = set(t[1].sampling_rate for t in tuples)
    closest_freq = _get_closest_freq(frequencies, sampling_rate)
    return [t for t in tuples if t[1].sampling_rate == closest_freq]


def check_memory(params: ConfigParameters, nsta: int) -> int:
    # maximum memory allowed
    # TODO: Is this needed? Should it be configurable?
    MAX_MEM = 96.0
    # crude estimation on memory needs (assume float32)
    nsec_chunk = params.inc_hours / 24 * 86400
    nseg_chunk = int(np.floor((nsec_chunk - params.cc_len) / params.step))
    npts_chunk = int(nseg_chunk * params.cc_len * params.sampling_rate)
    memory_size = nsta * npts_chunk * 4 / 1024**3
    if memory_size > MAX_MEM:
        raise ValueError(
            "Require %5.3fG memory but only %5.3fG provided)! Reduce inc_hours to avoid this issue!"
            % (memory_size, MAX_MEM)
        )
    logger.info(f"Require {memory_size:5.2f}gb memory for cross correlations")
    return nseg_chunk
