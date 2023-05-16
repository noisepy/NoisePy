import gc
import logging
import sys
from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import obspy
from datetimerange import DateTimeRange
from mpi4py import MPI
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
    # --------MPI---------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # save metadata
        cc_store.save_parameters(fft_params)

        # set variables to broadcast
        timespans = raw_store.get_timespans()
        splits = len(timespans)
        if splits == 0:
            raise IOError("Abort! no available seismic files for FFT")
    else:
        splits, timespans = (None, None)

    # broadcast the variables
    splits = comm.bcast(splits, root=0)
    timespans = comm.bcast(timespans, root=0)

    # MPI loop: loop through each user-defined time chunk
    for ick in range(rank, splits, size):
        ts = timespans[ick]
        if cc_store.is_done(ts):
            continue

        """
        LOADING NOISE DATA AND DO FFT
        """
        t_chunk = tlog.reset()  # for tracking overall chunk processing time
        channels = raw_store.get_channels(ts)
        tlog.log("get channels")
        ch_data_tuples = _read_channels(ts, raw_store, channels, fft_params.samp_freq)
        tlog.log("read channel data")
        ch_data_tuples = preprocess(ch_data_tuples, raw_store, fft_params, ts)
        tlog.log("preprocess")

        nchannels = len(ch_data_tuples)
        nseg_chunk = check_memory(fft_params, nchannels)
        nnfft = int(next_fast_len(int(fft_params.cc_len * fft_params.samp_freq)))  # samp_freq should be sampling_rate
        # Dictionary to store all the FFTs, keyed by channel index
        ffts: OrderedDict[int, NoiseFFT] = OrderedDict()

        logger.debug(f"nseg_chunk: {nseg_chunk}, nnfft: {nnfft}")
        # loop through all channels
        tlog.reset()
        for ix_ch, (ch, ch_data) in enumerate(ch_data_tuples):
            # TODO: Below the last values for N and Nfft are used?
            fft_data = compute_fft(fft_params, ch_data)
            if fft_data.fft.size > 0:
                ffts[ix_ch] = fft_data
            else:
                logger.warning(f"No data available for channel '{ch}', skipped")
        Nfft = fft_data.Length
        tlog.log("Compute FFTs")

        if len(ffts) != nchannels:
            logger.warning("it seems some stations miss data in download step, but it is OKAY!")

        # ###########PERFORM CROSS-CORRELATION##################
        for iiS, src_fft in ffts.items():  # looping over the channel source
            src_chan = channels[iiS]  # this is the name of the source channel
            src_std = src_fft.std

            # this finds the windows of "good" noise
            sou_ind = np.where((src_std < fft_params.max_over_std) & (src_std > 0) & (np.isnan(src_std) == 0))[0]
            if not len(sou_ind):
                continue
            # in the case of pure deconvolution, we recommend smoothing anyway.
            if fft_params.cc_method == "deconv":
                tlog.reset()
                # -----------get the smoothed source spectrum for decon later----------
                sfft1 = noise_module.smooth_source_spect(fft_params, src_fft.fft)
                sfft1 = sfft1.reshape(src_fft.SegmentCount, src_fft.Length // 2)
                tlog.log("smoothing source")
            else:
                sfft1 = np.conj(src_fft.fft).reshape(src_fft.SegmentCount, src_fft.Length // 2)

            # get index right for auto/cross correlation
            istart = iiS  # start at the channel source / only fills the upper right triangle matrix of channel pairs
            iend = nchannels

            # -----------now loop III for each receiver B----------
            for iiR in range(istart, iend):
                rec_chan = channels[iiR]
                if fft_params.acorr_only:
                    if src_chan.station != rec_chan.station:
                        continue
                logger.debug(f"receiver: {rec_chan}")
                if iiR not in ffts:
                    continue
                if cc_store.contains(ts, src_chan, rec_chan, fft_params):
                    continue

                # read the receiver data
                rec_fft = ffts[iiR]
                sfft2 = rec_fft.fft.reshape(rec_fft.SegmentCount, rec_fft.Length // 2)
                rec_std = rec_fft.std

                # ---------- check the existence of earthquakes or spikes ----------
                rec_ind = np.where((rec_std < fft_params.max_over_std) & (rec_std > 0) & (np.isnan(rec_std) == 0))[0]
                bb = np.intersect1d(sou_ind, rec_ind)
                if len(bb) == 0:
                    continue

                # ----------- GAME TIME: cross correlation step ---------------
                tlog.reset()
                corr, tcorr, ncorr = noise_module.correlate(
                    sfft1[bb, :], sfft2[bb, :], fft_params, Nfft, rec_fft.fft_time[bb]
                )
                tlog.log("cross-correlate")

                # ---------- OUTPUT: store metadata and data into file ------------
                coor = {
                    "lonS": src_chan.station.lon,
                    "latS": src_chan.station.lat,
                    "lonR": rec_chan.station.lon,
                    "latR": rec_chan.station.lat,
                }
                comp = src_chan.type.get_orientation() + rec_chan.type.get_orientation()
                parameters = noise_module.cc_parameters(fft_params, coor, tcorr, ncorr, comp)
                cc_store.append(ts, src_chan, rec_chan, fft_params, parameters, corr)
                tlog.log("write cc")

                del sfft2
            del sfft1

        ffts.clear()
        gc.collect()

        tlog.log(f"Process the chunk of {ts}", t_chunk)
        cc_store.mark_done(ts)

    tlog.log("Step 1 in total", t_s1_total)
    comm.barrier()


def preprocess(
    ch_data: List[Tuple[Channel, ChannelData]], raw_store: RawDataStore, fft_params: ConfigParameters, ts: DateTimeRange
) -> List[Tuple[Channel, ChannelData]]:
    new_streams = [
        noise_module.preprocess_raw(
            tup[1].stream,
            raw_store.get_inventory(ts, tup[0].station),
            fft_params,
            obspy.UTCDateTime(ts.start_datetime),
            obspy.UTCDateTime(ts.end_datetime),
        )
        for tup in ch_data
    ]
    channels = list(zip(*ch_data))[0]
    return list(zip(channels, [ChannelData(st) for st in new_streams]))


def compute_fft(fft_params: ConfigParameters, ch_data: ChannelData) -> NoiseFFT:
    if ch_data.data.size == 0:
        return NoiseFFT(np.empty, np.empty, np.empty, 0, 0)

    # cut daily-long data into smaller segments (dataS always in 2D)
    trace_stdS, dataS_t, dataS = noise_module.cut_trace_make_stat(
        fft_params, ch_data
    )  # optimized version:3-4 times faster
    if not len(dataS):
        return NoiseFFT(np.empty, np.empty, np.empty, 0, 0)

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
    ch_data = [store.read_data(ts, ch) for ch in channels]
    tuples = list(zip(channels, ch_data))
    return _filter_channel_data(tuples, samp_freq)


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
