from noisepy.seis.correlate import _filter_channel_data
from noisepy.seis.datatypes import Channel, ChannelData, Station


def test_read_channels():
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

    # we should pick the closest frequency that is >= to the target freq, 60 in this case
    filtered = _filter_channel_data(tuples, samp_freq, single_freq=True)
    assert N == len(filtered)
    assert [t[1].sampling_rate for t in filtered] == [CLOSEST_FREQ] * N

    # we should get all data at >= 40 Hz (60 and 100)
    filtered = _filter_channel_data(tuples, samp_freq, single_freq=False)
    assert N * 2 == len(filtered)
    assert all([t[1].sampling_rate >= samp_freq for t in filtered])
