import matplotlib.pyplot as plt
from memory_profiler import profile
import numpy as np
import pyfftw
import scipy
import time

@profile
def fftw_comp_2D():
    # create random variables
    Nseg = 47
    Npts = 3600*20

    a = np.random.rand(Nseg,Npts)*35
    b = np.random.rand(Nseg,Npts)*20
    c = a+1j*b

    # compare efficiency of three fftw package
    t0 = time.time()
    fft0 = pyfftw.interfaces.numpy_fft.fft(c)
    t1 = time.time()
    fft1 = np.fft.fft(c)
    t2 = time.time()
    fft2 = scipy.fftpack.fft(c)
    t3 = time.time()
    print('fft takes %f %f %f' % (t1-t0,t2-t1,t3-t2))

@profile
def fftw_comp_1D():
    # create random variables
    Nseg = 1
    Npts = 3600*200

    a = np.random.rand(Nseg,Npts)*35
    b = np.random.rand(Nseg,Npts)*20
    c = a+1j*b

    # compare efficiency of three fftw package
    t0 = time.time()
    fft0 = pyfftw.interfaces.numpy_fft.fft(c)
    t1 = time.time()
    fft1 = np.fft.fft(c)
    t2 = time.time()
    fft2 = scipy.fftpack.fft(c)
    t3 = time.time()
    print('fft takes %f %f %f' % (t1-t0,t2-t1,t3-t2))

if __name__ == '__main__':
    fftw_comp_1D()
    fftw_comp_2D()
