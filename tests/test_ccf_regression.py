"""
A/B regression test for the 2026-08 performance changes to the cross-correlation
pipeline (float32 waveform windows, batched hermitian irfft in correlate(),
rfft-based whiten_2D, uniform_filter1d-based moving_ave_2D, vectorized detrend).

The reference ("legacy") implementations below are verbatim copies of the code
paths prior to those changes: float64 waveform windows, per-window complex
scipy.fftpack.ifft calls, np.fft.fftn whitening and convolve2d moving average.

The test computes full-pipeline CCFs on a seeded synthetic day with both paths
and asserts that
  1) the normalized CCF waveforms correlate at > 0.99999,
  2) a stretching grid search over +/-5% dv/v gives the same answer,
  3) the substack timestamps are identical (and exactly regular, issue #352).
"""

import numpy as np
import obspy
import scipy.fftpack
import scipy.signal
from scipy.fftpack import next_fast_len

from noisepy.monitoring.monitoring_methods import stretching
from noisepy.seis.io.datatypes import (
    CCMethod,
    ChannelData,
    ConfigParameters,
    FreqNorm,
    TimeNorm,
)
from noisepy.seis.noise_module import (
    correlate,
    cut_trace_make_stat,
    demean,
    detrend,
    moving_ave,
    noise_processing,
    taper,
)

# ----------------------------------------------------------------------------
# legacy reference implementations (pre-change code, float64 path)
# ----------------------------------------------------------------------------


def legacy_moving_ave_2D(A, N):
    ntc, nspt = A.shape
    temp = np.zeros([ntc, nspt + 2 * N])
    temp[:, N:-N] = A
    temp[:, 0:N] = np.repeat(np.expand_dims(temp[:, N], axis=-1), N, axis=-1)
    temp[:, -N:] = np.repeat(np.expand_dims(temp[:, -N - 1], axis=-1), N, axis=-1)
    B = scipy.signal.convolve2d(temp, np.expand_dims(np.ones(N) / N, axis=0), mode="same")[:, N:-N]
    return B


def legacy_whiten_2D(timeseries, fft_para: ConfigParameters, n_taper):
    nfft = next_fast_len(timeseries.shape[1])
    spec = np.fft.fftn(timeseries, s=[nfft])
    freq = np.fft.fftfreq(nfft, d=fft_para.dt)

    ix0 = np.argmin(np.abs(freq - fft_para.freqmin))
    ix1 = np.argmin(np.abs(freq - fft_para.freqmax))

    ix11 = nfft if ix1 + n_taper > nfft else ix1 + n_taper
    ix00 = 0 if ix0 - n_taper < 0 else ix0 - n_taper

    spec_out = spec.copy()
    spec_out[:, 0:ix00] = 0.0 + 0.0j
    spec_out[:, ix11:] = 0.0 + 0.0j

    if fft_para.smoothspect_N <= 1:
        spec_out[:, ix00:ix11] = np.exp(1.0j * np.angle(spec_out[:, ix00:ix11]))
    else:
        spec_out[:, ix00:ix11] /= legacy_moving_ave_2D(np.abs(spec_out[:, ix00:ix11]), fft_para.smoothspect_N)

    x = np.linspace(np.pi / 2.0, np.pi, ix0 - ix00)
    spec_out[:, ix00:ix0] *= np.cos(x) ** 2

    x = np.linspace(0.0, np.pi / 2.0, ix11 - ix1)
    spec_out[:, ix1:ix11] *= np.cos(x) ** 2

    return spec_out


def legacy_cut_trace(fc_para: ConfigParameters, ch_data: ChannelData):
    """window cutting as before the change: float64 waveform matrix"""
    sps = int(ch_data.sampling_rate)
    starttime = ch_data.start_timestamp
    data = ch_data.data
    nseg = int(np.floor((fc_para.inc_hours / 24 * 86400 - fc_para.cc_len) / fc_para.step))
    npts = int(fc_para.cc_len * sps)
    dataS = np.zeros(shape=(nseg, npts), dtype=np.float64)
    dataS_t = np.zeros(nseg, dtype=np.float64)
    indx1 = 0
    for iseg in range(nseg):
        dataS[iseg] = data[indx1 : indx1 + npts]
        dataS_t[iseg] = starttime + fc_para.step * iseg
        indx1 += int(fc_para.step) * sps
    dataS = demean(dataS)
    dataS = detrend(dataS)
    dataS = taper(dataS)
    return dataS_t, dataS


def legacy_noise_processing(fft_para: ConfigParameters, dataS):
    """RMA time normalization + fftn-based whitening, as before the change"""
    white = np.zeros(shape=dataS.shape, dtype=dataS.dtype)
    for kkk in range(dataS.shape[0]):
        white[kkk, :] = dataS[kkk, :] / moving_ave(np.abs(dataS[kkk, :]), fft_para.smooth_N)
    return legacy_whiten_2D(white, fft_para, 100)


def legacy_correlate(fft1_smoothed_abs, fft2, D, Nfft, dataS_t):
    """substack (substack_len == cc_len) branch of correlate() as before the change"""
    dt = D["dt"]
    maxlag = D["maxlag"]
    nwin = fft1_smoothed_abs.shape[0]
    Nfft2 = fft1_smoothed_abs.shape[1]

    corr = fft1_smoothed_abs.reshape(fft1_smoothed_abs.size) * fft2.reshape(fft2.size)
    x_corr = not np.all(np.abs(fft1_smoothed_abs) == np.abs(fft2))
    corr = corr.reshape(nwin, Nfft2)

    s_corr = np.zeros(shape=(nwin, Nfft), dtype=np.float32)
    n_corr = np.zeros(nwin, dtype=np.int16)
    t_corr = dataS_t
    crap = np.zeros(Nfft, dtype=np.complex64)
    for i in range(nwin):
        n_corr[i] = 1
        crap[:Nfft2] = corr[i, :]
        crap[:Nfft2] = crap[:Nfft2] - np.mean(crap[:Nfft2])
        crap[-(Nfft2) + 1 :] = np.flip(np.conj(crap[1:(Nfft2)]), axis=0)
        if x_corr:
            crap[0] = complex(0, 0)
        s_corr[i, :] = np.real(np.fft.ifftshift(scipy.fftpack.ifft(crap, Nfft, axis=0)))

    ampmax = np.max(s_corr, axis=1)
    tindx = np.where((ampmax < 20 * np.median(ampmax)) & (ampmax > 0))[0]
    s_corr = s_corr[tindx, :]
    t_corr = t_corr[tindx]
    n_corr = n_corr[tindx]

    t = np.arange(-Nfft2 + 1, Nfft2) * dt
    ind = np.where(np.abs(t) <= maxlag)[0]
    return s_corr[:, ind], t_corr, n_corr


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def make_config() -> ConfigParameters:
    config = ConfigParameters()
    config.sampling_rate = 20.0
    config.cc_len = 300
    config.step = 150.0
    config.inc_hours = 1
    config.maxlag = 50
    config.freqmin = 0.1
    config.freqmax = 8.0
    config.smooth_N = 10
    config.smoothspect_N = 10
    config.time_norm = TimeNorm.RMA
    config.freq_norm = FreqNorm.RMA
    config.cc_method = CCMethod.XCORR
    config.substack = True
    config.substack_windows = 1  # substack_len == cc_len: keep every window
    return config


def make_synthetic_day(config: ConfigParameters):
    """two channels sharing a delayed common signal plus independent noise"""
    fs = int(config.sampling_rate)
    n = fs * 3600 * config.inc_hours
    rng = np.random.default_rng(1234)
    delay = 2 * fs  # 2 s delay between the two channels
    common = rng.standard_normal(n + delay)
    ch1 = common[delay:] + 0.5 * rng.standard_normal(n)
    ch2 = common[:n] + 0.5 * rng.standard_normal(n)
    start = obspy.UTCDateTime("2025-06-01T00:00:00.0Z")
    header = {"sampling_rate": config.sampling_rate, "starttime": start}
    cd1 = ChannelData(obspy.Stream([obspy.Trace(ch1, header=header)]))
    cd2 = ChannelData(obspy.Stream([obspy.Trace(ch2, header=header)]))
    return cd1, cd2


def new_pipeline_ccf(config, cd1, cd2):
    """CCFs through the current production functions"""
    _, t1, win1 = cut_trace_make_stat(config, cd1)
    _, t2, win2 = cut_trace_make_stat(config, cd2)
    white1 = noise_processing(config, win1)
    white2 = noise_processing(config, win2)
    Nfft = white1.shape[1]
    fft1 = white1[:, : Nfft // 2]
    fft2 = white2[:, : Nfft // 2]
    sfft1 = np.conj(fft1)  # XCORR
    return correlate(sfft1, fft2, config, Nfft, t1, is_autocorr=False)


def legacy_pipeline_ccf(config, cd1, cd2):
    """CCFs through the legacy float64 reference implementations"""
    t1, win1 = legacy_cut_trace(config, cd1)
    _, win2 = legacy_cut_trace(config, cd2)
    white1 = legacy_noise_processing(config, win1)
    white2 = legacy_noise_processing(config, win2)
    Nfft = white1.shape[1]
    fft1 = white1[:, : Nfft // 2]
    fft2 = white2[:, : Nfft // 2]
    sfft1 = np.conj(fft1)  # XCORR
    return legacy_correlate(sfft1, fft2, config, Nfft, t1)


def normalized_correlation(a, b):
    a = (a - np.mean(a)) / np.std(a)
    b = (b - np.mean(b)) / np.std(b)
    return float(np.mean(a * b))


def measure_dvv(ref, cur, config):
    npts = len(ref)
    # same para convention as tests/test_stretching.py: t has npts+1 samples so
    # that the measurement window covers the full waveform. Time is expressed in
    # samples (dt=1) to avoid float truncation in the window index computation;
    # the dv/v estimate is unitless so this does not affect the result.
    para = {
        "t": np.arange(npts + 1, dtype=np.float64),
        "dt": 1.0,
        "twin": [0.0, float(npts)],
        "freq": [config.freqmin, config.freqmax],
    }
    dvv, error, cc, cdp = stretching(ref=ref, cur=cur, dv_range=0.05, nbtrial=101, para=para)
    return dvv, cc


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------


def test_ccf_ab_regression():
    config = make_config()
    cd1, cd2 = make_synthetic_day(config)

    ccf_new, t_new, n_new = new_pipeline_ccf(config, cd1, cd2)
    ccf_ref, t_ref, n_ref = legacy_pipeline_ccf(config, cd1, cd2)

    assert ccf_new.shape == ccf_ref.shape
    assert ccf_new.shape[0] > 1

    # timestamps identical between paths, and exactly regular (issue #352)
    assert np.array_equal(t_new, t_ref)
    assert np.all(np.diff(t_new) == config.step)

    # per-window and stacked CCF waveforms must be essentially identical
    for i in range(ccf_new.shape[0]):
        r = normalized_correlation(ccf_new[i], ccf_ref[i])
        assert r > 0.99999, f"window {i}: correlation {r}"

    stack_new = np.mean(ccf_new.astype(np.float64), axis=0)
    stack_ref = np.mean(ccf_ref.astype(np.float64), axis=0)
    r = normalized_correlation(stack_new, stack_ref)
    assert r > 0.99999, f"stacked correlation {r}"

    # a stretching grid search over +/-5% dv/v between the two stacks finds no
    # velocity change (to within the refined grid resolution of ~0.004%) and
    # near-perfect correlation
    dvv, cc = measure_dvv(stack_ref, stack_new, config)
    assert abs(dvv) < 0.005  # in %
    # cc is computed after resampling onto the refined stretching grid, which
    # does not contain exactly 1.0, so it is limited by interpolation error;
    # waveform fidelity itself is asserted at > 0.99999 above
    assert cc > 0.999

    # measuring a real, synthetically stretched waveform against either stack
    # gives the same dv/v
    npts = len(stack_ref)
    tax = np.linspace(0.0, 1.0, npts)
    stretched = np.interp(tax, np.linspace(0.0, 1.0 + 0.005, npts), stack_ref)
    dvv_ref, _ = measure_dvv(stack_ref, stretched, config)
    dvv_new, _ = measure_dvv(stack_new, stretched, config)
    assert abs(dvv_ref - dvv_new) < 0.005  # in %, within the refined grid resolution
    assert abs(dvv_ref) > 0.1  # the synthetic stretch is actually detected


def test_ccf_ab_regression_no_substack():
    config = make_config()
    config.substack = False
    cd1, cd2 = make_synthetic_day(config)

    ccf_new, t_new, _ = new_pipeline_ccf(config, cd1, cd2)
    assert ccf_new.ndim == 2 and ccf_new.shape[0] == 1

    config.substack = True
    ccf_ref, t_ref, _ = legacy_pipeline_ccf(config, cd1, cd2)
    stack_ref = np.mean(ccf_ref.astype(np.float64), axis=0)

    # daily average of the substacks matches the no-substack daily CCF
    r = normalized_correlation(ccf_new[0], stack_ref)
    assert r > 0.9999
