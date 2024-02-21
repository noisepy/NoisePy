import numpy as np
import pytest

from noisepy.seis.noise_module import demean, detrend, taper

data = [np.random.random(100), np.random.random(1000), np.random.random([2, 100]), np.random.random([2, 1000])]
data2 = [np.random.random(1000) + np.arange(1000) * 1e-3, np.random.random([2, 1000]) + np.arange(1000) * 1e-3]


@pytest.mark.parametrize("data", data)
def test_taper(data: np.ndarray):
    data_taper = taper(data)
    if data_taper.ndim == 1:
        assert np.isclose(data_taper[0], 0)
        assert np.isclose(data_taper[-1], 0)
    elif data_taper.ndim == 2:
        assert np.isclose(np.linalg.norm(data_taper[:, 0]), 0)
        assert np.isclose(np.linalg.norm(data_taper[:, -1]), 0)


@pytest.mark.parametrize("data", data)
def test_demean(data: np.ndarray):
    data_demean = demean(data)
    if data_demean.ndim == 1:
        assert np.isclose(np.mean(data_demean), 0)
    elif data_demean.ndim == 2:
        assert np.isclose(np.linalg.norm(np.mean(data_demean, axis=1)), 0)


@pytest.mark.parametrize("data", data2)
def test_detrend(data: np.ndarray):
    data_detrend = detrend(data)
    if data_detrend.ndim == 1:
        npts = data_detrend.shape[0]
    elif data_detrend.ndim == 2:
        npts = data_detrend.shape[1]

    X = np.ones((npts, 2))
    X[:, 0] = np.arange(0, npts) / npts
    Q, R = np.linalg.qr(X)
    rq = np.dot(np.linalg.inv(R), Q.transpose())
    coeff = np.dot(rq, data_detrend.T)
    assert np.isclose(np.linalg.norm(coeff), 0)
