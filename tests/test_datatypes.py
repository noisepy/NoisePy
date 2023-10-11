from pathlib import Path

import dateutil
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
    ConfigParameters.validate(c1)
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


def test_storage_options():
    c = ConfigParameters()
    assert c.storage_options == {}
    # make sure missing keys default to {}
    assert c.storage_options["some_key"] == {}
    c.storage_options["some_key"]["some_other_key"] = 6
    assert c.storage_options["some_key"]["some_other_key"] == 6

    c.storage_options["s3"] = {"profile": "my_profile"}
    assert c.get_storage_options("s3://my_bucket/my_file") == {"profile": "my_profile"}

    # scheme is '' for local files
    c.storage_options[""]["foo"] = "bar"
    assert c.get_storage_options("/local/file") == {"foo": "bar"}


def test_config_dates():
    c = ConfigParameters()
    # defaults should be valid
    c = ConfigParameters.model_validate(dict(c), strict=True)
    c.start_date = dateutil.parser.isoparse("2021-01-01")  # no timezone
    with pytest.raises(Exception):
        c = ConfigParameters.model_validate(dict(c), strict=True)
    c.start_date = dateutil.parser.isoparse("2021-01-01T09:00:00+09:00")  # not utc
    with pytest.raises(Exception):
        c = ConfigParameters.model_validate(dict(c), strict=True)
    c.start_date = dateutil.parser.isoparse("2021-01-01T09:00:00+00:00")  # utc
    c = ConfigParameters.model_validate(dict(c), strict=True)
