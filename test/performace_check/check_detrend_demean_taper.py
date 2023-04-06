import glob
import time

import numpy as np
import obspy
from obspy.core.util.base import _get_function_from_entry_point
from scipy import signal

"""
check efficiency of detrend, demean
"""


def detrend(data):
    """
    remove the trend of the signal based on QR decomposion
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
    remove the mean of the signal
    """
    # ndata = np.zeros(shape=data.shape,dtype=data.dtype)
    if data.ndim == 1:
        data = data - np.mean(data)
    elif data.ndim == 2:
        for ii in range(data.shape[0]):
            data[ii] = data[ii] - np.mean(data[ii])
    return data


def taper1(data):
    """
    apply a cosine taper using tukey window
    """
    ndata = np.zeros(shape=data.shape, dtype=data.dtype)
    if data.ndim == 1:
        npts = data.shape[0]
        win = signal.tukey(npts, alpha=0.05)
        ndata = data * win
    elif data.ndim == 2:
        npts = data.shape[1]
        win = signal.tukey(npts, alpha=0.05)
        for ii in range(data.shape[0]):
            ndata[ii] = data[ii] * win
    return ndata


def taper(data):
    """
    apply a cosine taper using obspy functions
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


def test_1d(sacfile):
    """
    performance check with 1d data
    """
    tr = obspy.read(sacfile)
    tdata = tr[0].data

    # detrend, demean using obspy functions
    t0 = time.time()
    tr[0].detrend(type="constant")
    t1 = time.time()
    tr[0].detrend(type="linear")
    t2 = time.time()
    tr[0].taper(max_percentage=0.05, max_length=20)
    t3 = time.time()
    print("1D: it takes %6.3f in total with %6.3f %6.3f and %6.3f for obspy" % (t3 - t0, t1 - t0, t2 - t1, t3 - t2))

    # detrend, demean using newly defined function
    t0 = time.time()
    tdata = demean(tdata)
    t1 = time.time()
    tdata = detrend(tdata)
    t2 = time.time()
    tdata = taper(tdata)
    t3 = time.time()
    print("1D: it takes %6.3f in total with %6.3f %6.3f and %6.3f for new" % (t3 - t0, t1 - t0, t2 - t1, t3 - t2))


def test_2d(sacfile):
    """ """
    # parameters for obspy function
    cc_len = 3600
    step = 900

    # read data
    tr = obspy.read(sacfile)
    tdata = tr[0].data

    # sliding
    t0 = time.time()
    for ii, win in enumerate(tr[0].slide(window_length=cc_len, step=step)):
        win.detrend(type="constant")  # remove mean
        win.detrend(type="linear")  # remove trend
        win.taper(max_percentage=0.05, max_length=20)  # taper window
    t1 = time.time()

    print("2D: it takes %6.3f (%d traces) in total with obspy" % (t1 - t0, ii))

    # define parameters for new
    nseg = int(np.floor((86400 - cc_len) / step))
    sps = int(tr[0].stats.sampling_rate)
    npts = cc_len * sps
    dataS = np.zeros(shape=(nseg, npts), dtype=np.float32)

    indx1 = 0
    for iseg in range(nseg):
        indx2 = indx1 + npts
        dataS[iseg] = tdata[indx1:indx2]
        indx1 = indx1 + step * sps

    t2 = time.time()
    dataS = demean(dataS)
    dataS = detrend(dataS)
    dataS = taper(dataS)
    t3 = time.time()

    print("2D: it takes %6.3f (%d traces) in total with new" % (t3 - t2, dataS.shape[0]))


def main():
    sfiles = glob.glob("/Users/chengxin/Documents/NoisePy_example/Kanto/CLEAN_DATA//Event_2010_352/*.sac")

    for sacfile in sfiles:
        # test_1d(sacfile)
        test_2d(sacfile)


if __name__ == "__main__":
    main()
