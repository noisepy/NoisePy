from pathlib import Path

import pytest

from noisepy.seis.datatypes import ChannelType, ConfigParameters, StackMethod


def test_channeltype():
    with pytest.raises(Exception):
        ChannelType("toolong")


@pytest.mark.parametrize("ch,orien", [("bhe", "e"), ("bhe_00", "e")])
def test_orientation(ch, orien):
    ch = ChannelType(ch)
    assert ch.get_orientation() == orien


def test_config_yaml(tmp_path: Path):
    file = str(tmp_path.joinpath("config.yaml"))
    c1 = ConfigParameters()
    # change a couple of properties
    c1.step = 800
    c1.stack_method = StackMethod.ROBUST
    c1.save_yaml(file)
    c2 = ConfigParameters.parse_file(file)
    assert c1 == c2
