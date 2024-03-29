import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.fftpack import next_fast_len

from noisepy.seis.io.datatypes import FreqNorm
from noisepy.seis.noise_module import moving_ave, whiten


def whiten_original(data, fft_para):
    """
    This function takes 1-dimensional timeseries array, transforms to frequency domain using fft,
    whitens the amplitude of the spectrum in frequency domain between *freqmin* and *freqmax*
    and returns the whitened fft.
    PARAMETERS:
    ----------------------
    data: numpy.ndarray contains the 1D time series to whiten
    fft_para: dict containing all fft_cc parameters such as
        dt: The sampling space of the `data`
        freqmin: The lower frequency bound
        freqmax: The upper frequency bound
        smooth_N: integer, it defines the half window length to smooth
        freq_norm: whitening method between 'one-bit' and 'RMA'
    RETURNS:
    ----------------------
    FFTRawSign: numpy.ndarray contains the FFT of the whitened input trace between the frequency bounds
    """

    # load parameters
    delta = fft_para["dt"]
    freqmin = fft_para["freqmin"]
    freqmax = fft_para["freqmax"]
    smooth_N = fft_para["smooth_N"]
    freq_norm = fft_para["freq_norm"]

    # Speed up FFT by padding to optimal size for FFTPACK
    if data.ndim == 1:
        axis = 0
    elif data.ndim == 2:
        axis = 1

    Nfft = int(next_fast_len(int(data.shape[axis])))

    Napod = 100
    Nfft = int(Nfft)
    freqVec = scipy.fftpack.fftfreq(Nfft, d=delta)[: Nfft // 2]
    J = np.where((freqVec >= freqmin) & (freqVec <= freqmax))[0]
    low = J[0] - Napod
    if low <= 0:
        low = 1

    left = J[0]
    right = J[-1]
    high = J[-1] + Napod
    if high > Nfft / 2:
        high = int(Nfft // 2)

    FFTRawSign = scipy.fftpack.fft(data, Nfft, axis=axis)
    # Left tapering:
    if axis == 1:
        FFTRawSign[:, 0:low] *= 0
        FFTRawSign[:, low:left] = np.cos(np.linspace(np.pi / 2.0, np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:, low:left])
        )
        # Pass band:
        if freq_norm == FreqNorm.PHASE_ONLY:
            FFTRawSign[:, left:right] = np.exp(1j * np.angle(FFTRawSign[:, left:right]))
        elif freq_norm == FreqNorm.RMA:
            for ii in range(data.shape[0]):
                tave = moving_ave(np.abs(FFTRawSign[ii, left:right]), smooth_N)
                FFTRawSign[ii, left:right] = FFTRawSign[ii, left:right] / tave
        # Right tapering:
        FFTRawSign[:, right:high] = np.cos(np.linspace(0.0, np.pi / 2.0, high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[:, right:high])
        )
        FFTRawSign[:, high : Nfft // 2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[:, -(Nfft // 2) + 1 :] = np.flip(np.conj(FFTRawSign[:, 1 : (Nfft // 2)]), axis=axis)
    else:
        FFTRawSign[0:low] *= 0
        FFTRawSign[low:left] = np.cos(np.linspace(np.pi / 2.0, np.pi, left - low)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[low:left])
        )
        # Pass band:
        if freq_norm == FreqNorm.PHASE_ONLY:
            FFTRawSign[left:right] = np.exp(1j * np.angle(FFTRawSign[left:right]))
        elif freq_norm == FreqNorm.RMA:
            tave = moving_ave(np.abs(FFTRawSign[left:right]), smooth_N)
            FFTRawSign[left:right] = FFTRawSign[left:right] / tave
        # Right tapering:
        FFTRawSign[right:high] = np.cos(np.linspace(0.0, np.pi / 2.0, high - right)) ** 2 * np.exp(
            1j * np.angle(FFTRawSign[right:high])
        )
        FFTRawSign[high : Nfft // 2] *= 0

        # Hermitian symmetry (because the input is real)
        FFTRawSign[-(Nfft // 2) + 1 :] = FFTRawSign[1 : (Nfft // 2)].conjugate()[::-1]

    return FFTRawSign


# test case:
# the non-smoothed version of whitening needs to return the same as the original version.
# it is not expected that the smoothed version returns the same, so currently no test for that
# (would be good to add one based on some expected outcome)

fft_para = {
    "dt": 1.0,
    "freqmin": 0.01,
    "freqmax": 0.2,
    "smooth_N": 1,
    "freq_norm": FreqNorm.PHASE_ONLY,
}

# 1 D case
data = np.random.random(1000)
white_original = whiten_original(data, fft_para)
white_new = whiten(data, fft_para)
plt.plot(white_original[0:501].real)
plt.plot(white_new.real)
plt.show()
plt.plot(white_original[100:500].real - white_new[100:500].real)
plt.show()
plt.plot(white_original[100:500].imag - white_new[100:500].imag)
plt.show()

# A strict test does not work because the
assert np.sqrt(np.sum((white_original[0:500] - white_new[0:500]) ** 2) / 500.0) < 0.01 * white_new.max()
print("1D ok")

# 2 D case
data = np.random.random((5, 1000))
white_original = whiten_original(data, fft_para)
white_new = whiten(data, fft_para)

for i in range(5):
    plt.plot(white_original[i, 0:501].real)
    plt.plot(white_new[i, :].real)
plt.show()
for i in range(5):
    plt.plot(white_original[i, 100:500].real - white_new[i, 100:500].real)
plt.show()
for i in range(5):
    plt.plot(white_original[i, 100:500].imag - white_new[i, 100:500].imag)
plt.show()
for i in range(5):
    assert np.sqrt(np.sum((white_original[i, 0:500] - white_new[i, 0:500]) ** 2) / 500.0) < 0.01 * white_new[i, :].max()
print("2D ok")
