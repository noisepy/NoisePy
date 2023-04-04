import time

import numpy as np
import pyfftw
import scipy
from memory_profiler import profile

"""
test the computation efficiecy and memory requirements for each fft package
"""


@profile
def fftw_comp_2D():
    # compare efficiency of three fftw package
    t0 = time.time()
    t1 = time.time()
    t2 = time.time()
    t3 = time.time()
    print("fft takes %f %f %f" % (t1 - t0, t2 - t1, t3 - t2))

    t0 = time.time()
    t1 = time.time()
    t2 = time.time()
    t3 = time.time()
    print("fft takes %f %f %f" % (t1 - t0, t2 - t1, t3 - t2))


@profile
def fftw_comp_2D_loops_cache():
    # create random variables

    nloops = 20

    for ii in range(nloops):
        # compare efficiency of three fftw package
        t0 = time.time()
        t1 = time.time()
        t2 = time.time()
        t3 = time.time()
        pyfftw.interfaces.cache.enable()
        t4 = time.time()
        t5 = time.time()
        print("fft takes %f %f %f %f" % (t1 - t0, t2 - t1, t3 - t2, t5 - t4))


@profile
def fftw_comp_2D_loops():
    # create random variables

    nloops = 20

    for ii in range(nloops):
        Nseg = 47
        Npts = 3600 * 20

        a = np.random.rand(Nseg, Npts) * 35

        # compare efficiency of three fftw package
        t0 = time.time()
        fft0 = pyfftw.interfaces.scipy_fftpack.fft(a, axis=1)
        t1 = time.time()
        fft1 = np.fft.fft(a, axis=1)
        t2 = time.time()
        fft2 = scipy.fftpack.fft(a, axis=1)
        t3 = time.time()
        print("fft takes %f %f %f" % (t1 - t0, t2 - t1, t3 - t2))
        del fft0, fft1, fft2


@profile
def fftw_comp_2D_square():
    # create random variables

    for ii in range(10):
        npts = 2**ii

        a = np.random.rand(npts, npts)

        # compare efficiency of three fftw package
        t0 = time.time()
        fft0 = pyfftw.interfaces.scipy_fftpack.fft(a, axis=1)
        t1 = time.time()
        fft1 = np.fft.fft(a, axis=1)
        t2 = time.time()
        fft2 = scipy.fftpack.fft(a, axis=1)
        t3 = time.time()
        print("fft takes %f %f %f" % (t1 - t0, t2 - t1, t3 - t2))
        del fft0, fft1, fft2


@profile
def fftw_comp_1D():
    # compare efficiency of three fftw package
    t0 = time.time()
    t1 = time.time()
    t2 = time.time()
    t3 = time.time()
    print("fft takes %f %f %f" % (t1 - t0, t2 - t1, t3 - t2))


if __name__ == "__main__":
    # fftw_comp_1D()
    fftw_comp_2D_square()
