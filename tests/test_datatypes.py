import pytest

from noisepy.seis.datatypes import ChannelType


def test_channeltype():
    with pytest.raises(Exception):
        ChannelType("toolong")


@pytest.mark.parametrize("ch,orien", [("bhe", "e"), ("bhe_00", "e")])
def test_orientation(ch, orien):
    ch = ChannelType(ch)
    assert ch.get_orientation() == orien
