import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
from numba import jit


@jit("float32[:](float32[:],int16)")
def moving_ave(A, N):
    """
    Numba compiled function to do running smooth average.
    N is the the half window length to smooth
    A and B are both 1-D arrays (which runs faster compared to 2-D operations)
    """
    A = np.r_[A[:N], A, A[-N:]]
    B = np.zeros(A.shape, A.dtype)

    tmp = 0.0
    for pos in range(N, A.size - N):
        # do summing only once
        if pos == N:
            for i in range(-N, N + 1):
                tmp += A[pos + i]
        else:
            tmp = tmp - A[pos - N - 1] + A[pos + N]
        B[pos] = tmp / (2 * N + 1)
        if B[pos] == 0:
            B[pos] = 1
    return B[N:-N]


source1 = "/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/FFT/hdf5/N.AC2H.h5"
source2 = "/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/FFT/hdf5/N.CHHH.h5"

ds_s = h5py.File(source1, "r")
list_s1 = "fft_N_AC2H_EHZ_2010_01_01.real"
list_s2 = "fft_N_AC2H_EHZ_2010_01_01.imag"

ds_r = h5py.File(source2, "r")
list_r1 = "fft_N_CHHH_EHZ_2010_01_01.real"
list_r2 = "fft_N_CHHH_EHZ_2010_01_01.imag"

Nfft = ds_s[list_s1].attrs["nfft"]
Nseg = ds_s[list_s1].attrs["nseg"]

fft1 = np.zeros(Nfft // 2, dtype=np.complex64)
fft1 = ds_s[list_s1][0, : Nfft // 2] + 1j * ds_s[list_s2][0, : Nfft // 2]

fft2 = np.zeros(Nfft // 2, dtype=np.complex64)
fft2 = ds_r[list_r1][0, : Nfft // 2] + 1j * ds_r[list_r2][0, : Nfft // 2]

# do cross correlations
temp = moving_ave(np.abs(fft1), 10)
fft1 = np.conj(fft1) * fft2 / temp**2
fft1[0] = complex(0, 0)
ncorr = np.real(np.fft.ifftshift(scipy.fftpack.ifft(fft1, Nfft, axis=0)))

temp = np.load(
    "/Users/chengxin/Documents/Harvard/Kanto_basin/code/KANTO/check_again/CCF_hourly/2010_01_01.h5.npy"
)
tcorr = temp[2, :]

plt.subplot(211)
plt.plot(ncorr)
plt.subplot(212)
plt.plot(tcorr)
plt.show()
