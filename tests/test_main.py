from datetime import datetime, timezone
from typing import List
from unittest import mock

import obspy
import pytest

from noisepy.seis.constants import NO_CCF_DATA_MSG
from noisepy.seis.main import (
    Command,
    _valid_config_file,
    initialize_params,
    main,
    parse_args,
)


def test_parse_args():
    cmd = ["download"]
    # Test one of each of the data types we use
    cmd += ["--start_date", "2020-01-02T03:04:05Z"]
    cmd += ["--step", "5"]
    cmd += ["--samp_freq", "160.4"]
    cmd += ["--stations", "BAK,ARV"]
    cmd += ["--cc_method", "foobar"]
    cmd += ["--substack", "true"]
    cmd += ["--correction", "False"]
    args = parse_args(cmd)
    assert args.cmd == Command.DOWNLOAD

    cfg = initialize_params(args, None)
    assert cfg.start_date == datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert cfg.step == 5
    assert cfg.samp_freq == 160.4
    assert cfg.stations == ["BAK", "ARV"]
    assert cfg.cc_method == "foobar"
    assert cfg.substack is True
    assert cfg.correction is False


def empty(path: str, tmp_path: str) -> str:
    return f"--{path}_path={str(tmp_path)}"


def run_cmd_with_empty_dirs(cmd: Command, args: List[str]):
    args = parse_args([cmd.name.lower()] + args)
    main(args)


def test_main_cc(tmp_path):
    tmp = str(tmp_path)
    run_cmd_with_empty_dirs(Command.CROSS_CORRELATE, [empty("raw_data", tmp), empty("xml", tmp)])


def test_main_stack(tmp_path):
    tmp = str(tmp_path)
    with pytest.raises(IOError) as excinfo:
        run_cmd_with_empty_dirs(Command.STACK, [empty("ccf", tmp), "--format=asdf"])
    assert NO_CCF_DATA_MSG in str(excinfo.value)


def test_main_download(tmp_path):
    tmp = str(tmp_path)
    with pytest.raises(obspy.clients.fdsn.header.FDSNNoDataException):
        run_cmd_with_empty_dirs(
            Command.DOWNLOAD,
            [
                empty("raw_data", tmp),
                "--start_date=2020-01-01",
                "--end_date=2020-01-01",
                "--stations=''",
                "--net_list=''",
                "--channels=''",
            ],
        )


def test_valid_config(tmp_path):
    cfgfile = tmp_path.joinpath("config.yaml")
    parser = mock.Mock()
    assert not _valid_config_file(parser, str(cfgfile))
    parser.error.assert_called_once()

    parser = mock.Mock()
    cfgfile.write_text("")  # creates the file
    assert _valid_config_file(parser, str(cfgfile))
    parser.error.assert_not_called()
