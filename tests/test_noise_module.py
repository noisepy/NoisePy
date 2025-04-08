import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from noisepy.seis.io.datatypes import CCMethod, ConfigParameters, FreqNorm, TimeNorm
from noisepy.seis.noise_module import (
    check_sample_gaps,
    demean,
    detrend,
    mad,
    noise_processing,
    smooth_source_spect,
    taper,
)

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
@pytest.mark.parametrize("smoothspect_N", [10, 1])
def test_noise_processing(time_norm: TimeNorm, freq_norm: FreqNorm, smoothspect_N: int):
    config = ConfigParameters()
    config.time_norm = time_norm
    config.freq_norm = freq_norm
    config.smoothspect_N = smoothspect_N
    dataS = np.random.random([2, 500])
    noise_processing(config, dataS)


@pytest.mark.parametrize("cc_method", [CCMethod.COHERENCY, CCMethod.DECONV, CCMethod.XCORR])
def test_smooth_source_spect(cc_method: CCMethod):
    config = ConfigParameters()
    config.cc_method = cc_method
    fft1 = np.random.random(500)
    smooth_source_spect(config, fft1)


def test_check_sample_gaps():
    start_date_st1 = UTCDateTime("2021-01-01T00:00:00.0Z")
    start_date_st2 = UTCDateTime("2021-01-01T00:00:00.5Z")
    end_date_st = UTCDateTime("2021-01-01T00:02:00.0Z")

    st = Stream()
    st_checked = check_sample_gaps(st.copy(), start_date_st1, end_date_st)
    assert len(st_checked) == 0  # empty stream

    st = Stream([Trace(np.zeros(5), header={"starttime": start_date_st1, "sampling_rate": 10})])
    st_checked = check_sample_gaps(st.copy(), start_date_st1, end_date_st)
    assert len(st_checked) == 0  # too short

    st += Trace(np.zeros(600), header={"starttime": start_date_st2, "sampling_rate": 10.0001})
    st_checked = check_sample_gaps(st.copy(), start_date_st1, end_date_st)
    assert [t.stats.sampling_rate for t in st_checked] == [10.0]

    st[1].stats.starttime += 1000
    end_date_st += 1000
    st_checked = check_sample_gaps(st.copy(), start_date_st1, end_date_st)
    assert len(st_checked) == 0  # gap too big
