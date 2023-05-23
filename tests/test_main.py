from datetime import datetime, timezone

from noisepy.seis.main import Command, initialize_params, parse_args


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
