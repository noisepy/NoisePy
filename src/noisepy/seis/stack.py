import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from datetimerange import DateTimeRange

from . import noise_module
from .datatypes import ConfigParameters, Stack, StackMethod, Station
from .scheduler import Scheduler, SingleNodeScheduler
from .stores import CrossCorrelationDataStore, StackStore
from .utils import TimeLogger, get_results

logger = logging.getLogger(__name__)
if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")

"""
Stacking script of NoisePy to:
    1) load cross-correlation data for sub-stacking (if needed) and all-time average;
    2) stack data with either linear or phase weighted stacking (pws) methods (or both);
    3) save outputs in ASDF or SAC format depend on user's choice (for latter option, find the script of write_sac
       in the folder of application_modules;
    4) rotate from a E-N-Z to R-T-Z system if needed.

Authors: Chengxin Jiang (chengxin_jiang@fas.harvard.edu)
         Marine Denolle (mdenolle@uw.edu)

NOTE:
    0. MOST occasions you just need to change parameters followed with detailed explanations to run the script.
    1. assuming 3 components are E-N-Z
    2. auto-correlation is not kept in the stacking due to the fact that it has only 6 cross-component.
    this tends to mess up the orders of matrix that stores the CCFs data
"""


def stack(
    cc_store: CrossCorrelationDataStore,
    stack_store: StackStore,
    fft_params: ConfigParameters,
    scheduler: Scheduler = SingleNodeScheduler(),
):
    # Use 'spawn' to avoid issues with multiprocessing on linux and 'fork'
    executor = ProcessPoolExecutor(mp_context=get_context("spawn"))
    tlog = TimeLogger(logger=logger, level=logging.INFO)
    t_tot = tlog.reset()

    def initializer():
        timespans = cc_store.get_timespans()
        pairs_all = cc_store.get_station_pairs()
        logger.info(
            f"Station pairs: {len(pairs_all)}, timespans:{len(timespans)}. From: {timespans[0]} to {timespans[-1]}"
        )

        if len(timespans) == 0 or len(pairs_all) == 0:
            raise IOError("Abort! no available CCF data for stacking")
        return timespans, pairs_all

    timespans, pairs_all = scheduler.initialize(initializer, 2)

    # Get the pairs that need to be processed by this node
    pairs_node = [pairs_all[i] for i in scheduler.get_indices(pairs_all)]

    done_pairs = set(stack_store.get_station_pairs())
    missing_pairs = []
    for p in pairs_node:
        if (p[0], p[1]) in done_pairs:
            logger.info(f"Stack already exists for {p[0]}-{p[1]}")
            continue
        missing_pairs.append(p)
    tasks = [
        executor.submit(stack_store_pair, p[0], p[1], timespans, cc_store, stack_store, fft_params)
        for p in missing_pairs
    ]
    _ = get_results(tasks, "Stacking Pairs")

    scheduler.synchronize()
    tlog.log("step 2 in total", t_tot)


def stack_store_pair(
    src_sta: Station,
    rec_sta: Station,
    timespans: List[DateTimeRange],
    cc_store: CrossCorrelationDataStore,
    stack_store: StackStore,
    fft_params: ConfigParameters,
):
    try:
        stacks = stack_pair(src_sta, rec_sta, timespans, cc_store, fft_params)
        tlog = TimeLogger(logger=logger, level=logging.INFO)
        stack_store.append(src_sta, rec_sta, stacks)
        tlog.log(f"writing stack pair {(src_sta, rec_sta)}")
    except Exception as e:
        logger.error(f"Error stacking pair {(src_sta, rec_sta)}: {e}")


def stack_pair(
    src_sta: Station,
    rec_sta: Station,
    timespans: List[DateTimeRange],
    cc_store: CrossCorrelationDataStore,
    fft_params: ConfigParameters,
) -> List[Stack]:
    tlog = TimeLogger(logger=logger, level=logging.INFO)
    # check if it is auto-correlation
    if src_sta == rec_sta:
        fauto = 1
    else:
        fauto = 0
    nccomp = fft_params.ncomp * fft_params.ncomp
    if fft_params.rotation and fft_params.correction:
        if not fft_params.correction_csv:
            logger.warning("Missing correction_csv parameter but rotation=True and correction=True")
        else:
            locs = pd.read_csv(fft_params.correction_csv)
    else:
        locs = []

    # cross component info
    if fft_params.ncomp == 1:
        enz_system = ["ZZ"]
    else:
        enz_system = ["EE", "EN", "EZ", "NE", "NN", "NZ", "ZE", "ZN", "ZZ"]

    # ZZ_R component used to avoid a collision with ZZ component above
    rtz_components = ["ZR", "ZT", "ZZ_R", "RR", "RT", "RZ", "TR", "TT", "TZ"]
    num_chunk = len(timespans) * nccomp
    num_segmts = 1
    # crude estimation on memory needs (assume float32)
    num_segmts, npts_segmt = calc_segments(fft_params, num_chunk)
    # allocate array to store fft data/info
    cc_array = np.zeros((num_chunk * num_segmts, npts_segmt), dtype=np.float32)
    cc_time = np.zeros(num_chunk * num_segmts, dtype=np.float32)
    cc_ngood = np.zeros(num_chunk * num_segmts, dtype=np.int16)
    cc_comp = np.chararray(num_chunk * num_segmts, itemsize=2, unicode=True)

    # loop through all time-chuncks
    iseg = 0
    for ts in timespans:
        # load the data from daily compilation
        cross_correlations = cc_store.read_correlations(ts, src_sta, rec_sta)

        logger.debug(f"path_list for {src_sta}-{rec_sta}: {cross_correlations}")
        # seperate auto and cross-correlation
        if not validate_pairs(fft_params.ncomp, str((src_sta, rec_sta)), fauto, ts, len(cross_correlations)):
            continue

        # load the 9-component data
        for cc in cross_correlations:
            src_chan, rec_chan = cc.src, cc.rec
            tparameters, tdata = cc.parameters, cc.data
            tcmp1 = src_chan.get_orientation()
            tcmp2 = rec_chan.get_orientation()

            # read data and parameter matrix
            ttime = tparameters["time"]
            tgood = tparameters["ngood"]
            if fft_params.substack:
                for ii in range(tdata.shape[0]):
                    cc_array[iseg] = tdata[ii]
                    cc_time[iseg] = ttime[ii]
                    cc_ngood[iseg] = tgood[ii]
                    cc_comp[iseg] = tcmp1 + tcmp2
                    iseg += 1
            else:
                cc_array[iseg] = tdata
                cc_time[iseg] = ttime
                cc_ngood[iseg] = tgood
                cc_comp[iseg] = tcmp1 + tcmp2
                iseg += 1

    t_load = tlog.log("loading CCF data")

    # continue when there is no data or for auto-correlation
    if iseg <= 1 and fauto == 1:
        return

    # matrix used for rotation
    if fft_params.rotation:
        bigstack = np.zeros(shape=(9, npts_segmt), dtype=np.float32)
    if fft_params.stack_method == StackMethod.ALL:
        bigstack1 = np.zeros(shape=(9, npts_segmt), dtype=np.float32)
        bigstack2 = np.zeros(shape=(9, npts_segmt), dtype=np.float32)

    stack_results: List[Stack] = []

    def append_stacks(comp: str, tparameters: Dict[str, Any], stack_data: List[Tuple[StackMethod, np.ndarray]]):
        for method, data in stack_data:
            stack_results.append(Stack(comp, f"Allstack_{method.value}", tparameters, data))

    # loop through cross-component for stacking
    iflag = 1
    for icomp in range(nccomp):
        comp = enz_system[icomp]
        indx = np.where(cc_comp.lower() == comp.lower())[0]
        logger.debug(f"index to find the comp: {indx}")

        # jump if there are not enough data
        if len(indx) < 2:
            iflag = 0
            continue

        # output stacked data
        (
            cc_final,
            ngood_final,
            stamps_final,
            allstacks1,
            allstacks2,
            allstacks3,
            nstacks,
        ) = noise_module.stacking(cc_array[indx], cc_time[indx], cc_ngood[indx], fft_params)
        logger.debug(f"after stacking nstacks: {nstacks}")
        if not len(allstacks1):
            continue
        if fft_params.rotation:
            bigstack[icomp] = allstacks1
            if fft_params.stack_method == StackMethod.ALL:
                bigstack1[icomp] = allstacks2
                bigstack2[icomp] = allstacks3

            tparameters["time"] = stamps_final[0]
            tparameters["ngood"] = nstacks
            if fft_params.stack_method != StackMethod.ALL:
                to_write = [(fft_params.stack_method, allstacks1)]
            else:
                to_write = [
                    (StackMethod.LINEAR, allstacks1),
                    (StackMethod.PWS, allstacks2),
                    (StackMethod.ROBUST, allstacks3),
                ]
            append_stacks(comp, tparameters, to_write)

        # keep a track of all sub-stacked data from S1
        if fft_params.keep_substack:
            for ii in range(cc_final.shape[0]):
                tparameters["time"] = stamps_final[ii]
                tparameters["ngood"] = ngood_final[ii]
                stack_name = "T" + str(int(stamps_final[ii]))
                append_stacks(comp, tparameters, [(stack_name, cc_final[ii])])

    # do rotation if needed
    if fft_params.rotation and iflag:
        if np.all(bigstack == 0):
            return stack_results
        tparameters["station_source"] = src_sta.name
        tparameters["station_receiver"] = rec_sta.name
        if fft_params.stack_method != StackMethod.ALL:
            bigstack_rotated = noise_module.rotation(bigstack, tparameters, locs)

            # write to file
            for icomp in range(nccomp):
                comp = rtz_components[icomp]
                tparameters["time"] = stamps_final[0]
                tparameters["ngood"] = nstacks
                append_stacks(comp, tparameters, [(fft_params.stack_method, bigstack_rotated[icomp])])
        else:
            bigstack_rotated = noise_module.rotation(bigstack, tparameters, locs)
            bigstack_rotated1 = noise_module.rotation(bigstack1, tparameters, locs)
            bigstack_rotated2 = noise_module.rotation(bigstack2, tparameters, locs)

            # write to file
            for icomp in range(nccomp):
                comp = rtz_components[icomp]
                tparameters["time"] = stamps_final[0]
                tparameters["ngood"] = nstacks
                stacks = [
                    (StackMethod.LINEAR, bigstack_rotated[icomp]),
                    (StackMethod.PWS, bigstack_rotated1[icomp]),
                    (StackMethod.ROBUST, bigstack_rotated2[icomp]),
                ]
                append_stacks(comp, tparameters, stacks)
    tlog.log(f"stack/rotate all station pairs {(src_sta,rec_sta)}", t_load)
    return stack_results


def validate_pairs(ncomp: int, sta_pair: str, fauto: int, ts: DateTimeRange, n_pairs: int) -> bool:
    if fauto == 1:
        if ncomp == 3 and n_pairs < 6:
            logger.warning("continue! not enough cross components for auto-correlation %s in %s" % (sta_pair, ts))
            return False
    else:
        if ncomp == 3 and n_pairs < 9:
            logger.warning("continue! not enough cross components for cross-correlation %s in %s" % (sta_pair, ts))
            return False

    if n_pairs > 9:
        raise ValueError("more than 9 cross-component exists for %s %s! please double check" % (ts, sta_pair))
    return True


def calc_segments(fft_params: ConfigParameters, num_chunk: int) -> Tuple[int, int]:
    num_segmts = 1
    if fft_params.substack:  # things are difference when do substack
        if fft_params.substack_len == fft_params.cc_len:
            num_segmts = int(np.floor((fft_params.inc_hours * 3600 - fft_params.cc_len) / fft_params.step))
        else:
            num_segmts = int(fft_params.inc_hours / (fft_params.substack_len / 3600))
    npts_segmt = int(2 * fft_params.maxlag * fft_params.samp_freq) + 1
    memory_size = num_chunk * num_segmts * npts_segmt * 4 / 1024**3

    logger.debug("Good on memory (need %5.2f G )!" % (memory_size))
    return num_segmts, npts_segmt