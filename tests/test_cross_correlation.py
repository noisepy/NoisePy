from noisepy.seis.datatypes import Channel, ChannelData, Station
from noisepy.seis.S1_fft_cc_MPI import _filter_channel_data


def test_read_channels():
    # we should pick the closest frequency that is >= to the target freq, 60 in this case
    CLOSEST_FREQ = 60
    samp_freq = 40
    freqs = [10, 39, CLOSEST_FREQ, 100]
    ch_data = []
    for f in freqs:
        cd = ChannelData.empty()
        cd.sampling_rate = f
        ch_data.append(cd)
    N = 5
    tuples = [(Channel("foo", Station("CI", "bar")), cd) for cd in ch_data] * N
    filtered = _filter_channel_data(tuples, samp_freq)
    assert N == len(filtered)
    assert [t[1].sampling_rate for t in filtered] == [CLOSEST_FREQ] * N
