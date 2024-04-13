import numpy as np
import pytest

from noisepy.monitoring.monitoring_methods import (
    dtw_dvv,
    mwcs_dvv,
    stretching,
    stretching_vect,
    wtdtw_allfreq,
    wtdtw_dvv,
    wts_allfreq,
    wts_dvv,
    wxs_dvv,
)

data = [
    (
        np.sin(np.arange(100) / 10),
        np.sin(np.arange(100) / 10.1),
        {"twin": [0, 100], "freq": [0.01, 0.49], "dt": 1},
    )
]

data2 = [
    (
        np.sin(np.arange(100) / 10),
        np.sin(np.arange(100) / 10.1),
        {"twin": [0, 99], "t": np.array([0, 100]), "freq": [0.01, 0.49], "dt": 1},
    )
]


@pytest.mark.parametrize("d1,d2,param", data)
def test_stretching(d1: np.ndarray, d2: np.ndarray, param: dict):
    dv_range = 0.05
    nbtrial = 50
    dv, error, cc, _ = stretching(d1, d2, dv_range, nbtrial, param)
    assert np.isclose(dv, -1, rtol=0.2)
    assert np.isclose(error, 0.0, atol=1e-2)
    assert np.isclose(cc, 1.0, atol=1e-1)


@pytest.mark.parametrize("d1,d2,param", data)
def test_stretching_vect(d1: np.ndarray, d2: np.ndarray, param: dict):
    dv_range = 0.05
    nbtrial = 50
    dv, error, cc, _ = stretching_vect(d1, d2, dv_range, nbtrial, param)
    assert np.isclose(dv, -1, rtol=0.2)
    assert np.isclose(error, 0.0, atol=1e-2)
    assert np.isclose(cc, 1.0, atol=1e-1)


@pytest.mark.parametrize("d1,d2,param", data)
def test_dtw_dvv(d1: np.ndarray, d2: np.ndarray, param: dict):
    maxlag = 5
    b = 10
    direction = 1
    dv, _, _ = dtw_dvv(d1, d2, param, maxlag, b, direction)
    assert np.isclose(dv, -1, rtol=0.2)


@pytest.mark.parametrize("d1,d2,param", data)
def test_mwcs_dvv(d1: np.ndarray, d2: np.ndarray, param: dict):
    moving_window_length = 10.0
    slide_step = 1.0
    dv, _ = mwcs_dvv(d1, d2, moving_window_length, slide_step, param)
    assert dv < 0


@pytest.mark.parametrize("d1,d2,param", data)
@pytest.mark.parametrize("allfreq", [True, False])
def test_wxs_dvv(d1: np.ndarray, d2: np.ndarray, param: dict, allfreq: bool):
    if allfreq:
        _, dvv, err = wxs_dvv(d1, d2, allfreq, param)
    else:
        dvv, err = wxs_dvv(d1, d2, allfreq, param)
    assert np.all(np.isclose(err, 0, atol=1e-1))
    assert np.all(dvv < 0)


@pytest.mark.parametrize("d1,d2,param", data)
@pytest.mark.parametrize("allfreq", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_wts_dvv(d1: np.ndarray, d2: np.ndarray, param: dict, allfreq: bool, normalize: bool):
    dv_range = 0.05
    nbtrial = 50
    if allfreq:
        _, dvv, err = wts_dvv(d1, d2, allfreq, param, dv_range, nbtrial, normalize=normalize)
    else:
        dvv, err = wts_dvv(d1, d2, allfreq, param, dv_range, nbtrial, normalize=normalize)
    assert np.all(np.isclose(err, 0, atol=1e-1))
    assert np.all(dvv < 0)


@pytest.mark.parametrize("d1,d2,param", data)
@pytest.mark.parametrize("allfreq", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_wts_allfreq(d1: np.ndarray, d2: np.ndarray, param: dict, allfreq: bool, normalize: bool):
    dv_range = 0.05
    nbtrial = 50
    if allfreq:
        _, dvv, err = wts_allfreq(d1, d2, allfreq, param, dv_range, nbtrial, normalize=normalize)
    else:
        dvv, err = wts_allfreq(d1, d2, allfreq, param, dv_range, nbtrial, normalize=normalize)
    assert np.all(np.isclose(err, 0, atol=1e-1))
    assert np.all(dvv < 0)


@pytest.mark.parametrize("d1,d2,param", data)
@pytest.mark.parametrize("allfreq", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_wtdtw_allfreq(d1: np.ndarray, d2: np.ndarray, param: dict, allfreq: bool, normalize: bool):
    maxlag = 5
    b = 10
    direction = 1
    if allfreq:
        _, dvv, err = wtdtw_allfreq(d1, d2, allfreq, param, maxlag, b, direction, normalize=normalize)
    else:
        dvv, err = wtdtw_allfreq(d1, d2, allfreq, param, maxlag, b, direction, normalize=normalize)
    assert np.all(np.isclose(err, 0, atol=2e-1))
    assert np.all(np.isclose(dvv, 0, atol=2e0))


@pytest.mark.parametrize("d1,d2,param", data2)
@pytest.mark.parametrize("allfreq", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_wtdtw_dvv(d1: np.ndarray, d2: np.ndarray, param: dict, allfreq: bool, normalize: bool):
    maxlag = 5
    b = 10
    direction = 1
    if allfreq:
        _, dvv, err = wtdtw_dvv(d1, d2, allfreq, param, maxlag, b, direction, normalize=normalize)
    else:
        dvv, err = wtdtw_dvv(d1, d2, allfreq, param, maxlag, b, direction, normalize=normalize)
    assert np.all(np.isclose(err, 0, atol=2e-1))
    assert np.all(np.isclose(dvv, 0, atol=2e0))
