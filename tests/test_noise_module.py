import numpy as np
import pytest

from noisepy.seis.noise_module import mad, demean, detrend, taper, noise_processing, smooth_source_spect
from noisepy.seis.io.datatypes import ConfigParameters, TimeNorm, FreqNorm, CCMethod

data = [np.random.random(100), np.random.random(1000), np.random.random([2, 100]), np.random.random([2, 1000])]
data2 = [np.random.random(1000) + np.arange(1000) * 1e-3, np.random.random([2, 1000]) + np.arange(1000) * 1e-3]


@pytest.mark.parametrize("data", data)
def test_taper(data: np.ndarray):
    data_taper = taper(data)
    if data_taper.ndim == 1:
        assert np.isclose(data_taper[0], 0)
        assert np.isclose(data_taper[-1], 0)
    else:
        assert np.isclose(np.linalg.norm(data_taper[:, 0]), 0)
        assert np.isclose(np.linalg.norm(data_taper[:, -1]), 0)

@pytest.mark.parametrize("mask", [True, False])
def test_mad(mask: bool):
    data = np.random.random(500)
    if mask:
        ma = np.ma.masked_array(data, mask)
    else:
        ma = data
    mad(ma)


@pytest.mark.parametrize("data", data)
def test_demean(data: np.ndarray):
    data_demean = demean(data)
    if data_demean.ndim == 1:
        assert np.isclose(np.mean(data_demean), 0)
    else:
        assert np.isclose(np.linalg.norm(np.mean(data_demean, axis=1)), 0)


@pytest.mark.parametrize("data", data2)
def test_detrend(data: np.ndarray):
    data_detrend = detrend(data)
    if data_detrend.ndim == 1:
        npts = data_detrend.shape[0]
    else:
        npts = data_detrend.shape[1]

    X = np.ones((npts, 2))
    X[:, 0] = np.arange(0, npts) / npts
    Q, R = np.linalg.qr(X)
    rq = np.dot(np.linalg.inv(R), Q.transpose())
    coeff = np.dot(rq, data_detrend.T)
    assert np.isclose(np.linalg.norm(coeff), 0)


@pytest.mark.parametrize("freq_norm", [FreqNorm.NO, FreqNorm.RMA])
@pytest.mark.parametrize("time_norm", [TimeNorm.ONE_BIT, TimeNorm.RMA])
def test_noise_processing(time_norm: TimeNorm, freq_norm: FreqNorm):
    config = ConfigParameters()
    config.time_norm = time_norm
    config.freq_norm = freq_norm
    dataS = np.random.random([2, 500])
    noise_processing(config, dataS)


@pytest.mark.parametrize("cc_method", [CCMethod.COHERENCY, CCMethod.DECONV, CCMethod.XCORR])
def test_smooth_source_spect(cc_method: CCMethod):
    config = ConfigParameters()
    config.cc_method = cc_method
    fft1 = np.random.random(500)
    smooth_source_spect(config, fft1)

