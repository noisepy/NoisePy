import gc
import logging
import sys
import time
from typing import List, Tuple

import numpy as np
import obspy
from datetimerange import DateTimeRange
from mpi4py import MPI
from scipy.fftpack.helper import next_fast_len

from noisepy.seis.datatypes import Channel, ChannelData, ConfigParameters

from . import noise_module
from .stores import CrossCorrelationDataStore, RawDataStore

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

    #######################################
    # #########PROCESSING SECTION##########
    #######################################

    tt0 = time.time()
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
        t10 = time.time()
        ts = timespans[ick]
        if cc_store.is_done(ts):
            continue

        # ###########LOADING NOISE DATA AND DO FFT##################
        channels = raw_store.get_channels(ts)
        ch_data_tuples = _read_channels(ts, raw_store, channels, fft_params.samp_freq)
        ch_data_tuples = preprocess(ch_data_tuples, raw_store, fft_params, ts)

        nchannels = len(ch_data_tuples)
        nseg_chunk = check_memory(fft_params, nchannels)
        nnfft = int(next_fast_len(int(fft_params.cc_len * fft_params.samp_freq)))  # samp_freq should be sampling_rate
        # open array to store fft data/info in memory
        fft_array = np.zeros((nchannels, nseg_chunk * (nnfft // 2)), dtype=np.complex64)
        fft_std = np.zeros((nchannels, nseg_chunk), dtype=np.float32)
        fft_flag = np.zeros(nchannels, dtype=np.int16)
        fft_time = np.zeros((nchannels, nseg_chunk), dtype=np.float64)

        logger.debug(f"nseg_chunk: {nseg_chunk}, nnfft: {nnfft}")
        # loop through all channels
        N: int
        Nfft2: int
        for ix_ch, (ch, ch_data) in enumerate(ch_data_tuples):
            # TODO: Below the last values for N and Nfft are used?
            fft_array[ix_ch], fft_std[ix_ch], fft_time[ix_ch], N, Nfft2 = compute_fft(fft_params, ch_data)
            fft_flag[ix_ch] = fft_array[ix_ch].size > 0
            if not fft_flag[ix_ch]:
                logger.warning(f"No data available for channel '{ch}', skipped")
        Nfft = Nfft2 * 2

        # check whether array size is enough
        if np.sum(fft_flag) != nchannels:
            logger.warning("it seems some stations miss data in download step, but it is OKAY!")

        # ###########PERFORM CROSS-CORRELATION##################
        for iiS in range(nchannels):  # looping over the channel source
            src_chan = channels[iiS]
            fft1 = fft_array[iiS]
            source_std = fft_std[iiS]
            sou_ind = np.where((source_std < fft_params.max_over_std) & (source_std > 0) & (np.isnan(source_std) == 0))[
                0
            ]
            if not fft_flag[iiS] or not len(sou_ind):
                continue
            t0 = time.time()
            # -----------get the smoothed source spectrum for decon later----------
            sfft1 = noise_module.smooth_source_spect(fft_params, fft1)
            sfft1 = sfft1.reshape(N, Nfft2)
            t1 = time.time()
            logger.debug("smoothing source takes %6.4fs" % (t1 - t0))

            # get index right for auto/cross correlation
            istart = iiS  # start at the channel source / only fills the upper right triangle matrix of channel pairs
            iend = nchannels
            #             if ncomp==1:
            #                 iend=np.minimum(iiS+ncomp,iii)
            #             else:
            #                 if (channel[iiS][-1]=='Z'): # THIS IS NOT GENERALIZABLE. WE need to
            #                                               change this to the order there are
            #                                               bugs that shifts the components
            #                     iend=np.minimum(iiS+1,iii)
            #                 elif (channel[iiS][-1]=='N'):
            #                     iend=np.minimum(iiS+2,iii)
            #                 else:
            #                     iend=np.minimum(iiS+ncomp,iii)

            #         if fft_params.xcorr_only:
            #             if ncomp==1:
            #                 istart=np.minimum(iiS+ncomp,iii)
            #             else:
            #                 if (channel[iiS][-1]=='Z'):
            #                     istart=np.minimum(iiS+1,iii)
            #                 elif (channel[iiS][-1]=='N'):
            #                     istart=np.minimum(iiS+2,iii)
            #                 else:
            #                     istart=np.minimum(iiS+ncomp,iii)

            # -----------now loop III for each receiver B----------
            for iiR in range(istart, iend):
                rec_chan = channels[iiR]
                if fft_params.acorr_only:
                    if src_chan.station != rec_chan.station:
                        continue
                logger.debug(f"receiver: {rec_chan}")
                if not fft_flag[iiR]:
                    continue
                if cc_store.contains(ts, src_chan, rec_chan, fft_params):
                    continue

                fft2 = fft_array[iiR]
                sfft2 = fft2.reshape(N, Nfft2)
                receiver_std = fft_std[iiR]

                # ---------- check the existence of earthquakes ----------
                rec_ind = np.where(
                    (receiver_std < fft_params.max_over_std) & (receiver_std > 0) & (np.isnan(receiver_std) == 0)
                )[0]
                bb = np.intersect1d(sou_ind, rec_ind)
                if len(bb) == 0:
                    continue

                t2 = time.time()
                corr, tcorr, ncorr = noise_module.correlate(
                    sfft1[bb, :], sfft2[bb, :], fft_params, Nfft, fft_time[iiR][bb]
                )
                t3 = time.time()
                coor = {
                    "lonS": src_chan.station.lon,
                    "latS": src_chan.station.lat,
                    "lonR": rec_chan.station.lon,
                    "latR": rec_chan.station.lat,
                }
                comp = src_chan.type.get_orientation() + rec_chan.type.get_orientation()
                parameters = noise_module.cc_parameters(fft_params, coor, tcorr, ncorr, comp)
                cc_store.append(ts, src_chan, rec_chan, fft_params, parameters, corr)
                t4 = time.time()
                logger.debug("read S %6.4fs, cc %6.4fs, write cc %6.4fs" % ((t1 - t0), (t3 - t2), (t4 - t3)))

                del fft2, sfft2, receiver_std
            del fft1, sfft1, source_std

        fft_array = []
        fft_std = []
        fft_flag = []
        fft_time = []
        n = gc.collect()
        print("unreadable garbarge", n)

        t11 = time.time()
        print("it takes %6.2fs to process the chunk of %s" % (t11 - t10, ts))
        cc_store.mark_done(ts)

    tt1 = time.time()
    print("it takes %6.2fs to process step 1 in total" % (tt1 - tt0))
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


def compute_fft(
    fft_params: ConfigParameters, ch_data: ChannelData
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    if ch_data.data.size == 0:
        return (np.empty, np.empty, np.empty, 0, 0)

    # cut daily-long data into smaller segments (dataS always in 2D)
    trace_stdS, dataS_t, dataS = noise_module.cut_trace_make_stat(
        fft_params, ch_data
    )  # optimized version:3-4 times faster
    if not len(dataS):
        return (np.empty, np.empty, np.empty, 0, 0)

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
    return fft, std, fft_time, N, Nfft2


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
