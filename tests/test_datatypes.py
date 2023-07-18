from pathlib import Path

import pytest

from noisepy.seis.datatypes import ChannelType, ConfigParameters, StackMethod, Station


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
    c2 = ConfigParameters.load_yaml(file)
    assert c1 == c2


def test_station_valid():
    s = Station("CI", "BAK")
    assert not s.valid()
    s = Station("CI", "BAK", 110.0, 120.1, 15.0)
    assert s.valid()
    s = Station("CI", "BAK", -110.0, 120.1, 15.0)
    assert s.valid()
