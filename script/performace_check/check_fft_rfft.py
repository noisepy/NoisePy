import time

import matplotlib.pyplot as plt
import numpy as np
import pyasdf
import scipy
from obspy.signal.util import _npts2nfft
from scipy.fftpack.helper import next_fast_len

"""
this script compares the computational efficiency of the two numpy function
to do the fft, which are rfft and fft respectively
"""

hfile = "/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/CCF_C3/2010_01_11.h5"
with pyasdf.ASDFDataSet(hfile, mode="r") as ds:
    data_types = ds.auxiliary_data.list()

    # -----take the first data_type-----
    data_type = data_types[2]
    paths = ds.auxiliary_data[data_type].list()
    path = paths[2]

    data = ds.auxiliary_data[data_type][path].data[:]
    npts = len(data)

    # ----do fft----
    nfft2 = _npts2nfft(npts)
    nfft1 = int(next_fast_len(npts))
    nfft3 = int(next_fast_len(npts * 2 + 1))
    print("nfft1 and nfft2 %d %d" % (nfft1, nfft2))

    t0 = time.time()
    spec1 = scipy.fftpack.fft(data, nfft1)
    wave1 = scipy.fftpack.ifft(spec1, nfft1)
    t1 = time.time()
    spec2 = np.fft.rfft(data, nfft2)
    wave2 = np.fft.irfft(spec2, nfft2)
    t2 = time.time()
    spec3 = scipy.fftpack.fft(data, nfft3)
    wave3 = scipy.fftpack.ifft(spec3, nfft3)
    t3 = time.time()
    print("fft and rfft takes %f s, %f s and %f s" % (t1 - t0, t2 - t1, t3 - t2))
    print("length of spec1 and spec2 are %d %d %d" % (len(spec1), len(spec2), len(spec3)))

    freq = np.linspace(0, 10, nfft1)
    plt.subplot(311)
    plt.loglog(freq, np.abs(spec1))
    freq = np.linspace(0, 10, nfft2 // 2 + 1)
    plt.subplot(312)
    plt.loglog(freq, np.abs(spec2))
    plt.subplot(313)
    freq = np.linspace(0, 10, nfft3)
    plt.loglog(freq, np.abs(spec3))
    plt.show()

    plt.subplot(311)
    plt.plot(wave1)
    plt.subplot(312)
    plt.plot(wave2)
    plt.subplot(313)
    plt.plot(wave3)
    plt.show()

    print(spec1[-10:], spec2[-10:], spec3[-10:])
