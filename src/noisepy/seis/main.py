import argparse
import os
import typing
from enum import Enum
from typing import Any, Callable, List

import obspy

from .asdfstore import ASDFCCStore, ASDFRawDataStore
from .channelcatalog import CSVChannelCatalog, XMLStationChannelCatalog
from .constants import DATE_FORMAT_HELP, STATION_FILE
from .datatypes import Channel, ConfigParameters
from .S0A_download_ASDF_MPI import download
from .S1_fft_cc_MPI import cross_correlate
from .S2_stacking import stack
from .scedc_s3store import SCEDCS3DataStore
from .utils import fs_join, get_filesystem

# Utility running the different steps from the command line. Defines the arguments for each step

default_start_date = "2016_07_01_0_0_0"  # start date of download
default_end_date = "2016_07_02_0_0_0"  # end date of download
default_data_path = "Documents/SCAL"


class Step(Enum):
    DOWNLOAD = 1
    CROSS_CORRELATE = 2
    STACK = 3
    ALL = 4


def valid_date(d: str) -> str:
    _ = obspy.UTCDateTime(d)
    return d


def initialize_fft_params(raw_dir: str) -> ConfigParameters:
    params = ConfigParameters()
    dfile = fs_join(raw_dir, "download_info.txt")
    if os.path.isfile(dfile):
        down_info = eval(open(dfile).read())  # TODO: do proper json/yaml serialization
        params.samp_freq = down_info["samp_freq"]
        params.freqmin = down_info["freqmin"]
        params.freqmax = down_info["freqmax"]
        params.start_date = down_info["start_date"]
        params.end_date = down_info["end_date"]
        params.inc_hours = down_info["inc_hours"]
        params.ncomp = down_info["ncomp"]
    return params


def get_channel_filter(sta_list: List[str]) -> Callable[[Channel], bool]:
    if len(sta_list) == 1 and sta_list[0] == "*":
        return lambda ch: True
    else:
        stations = set(sta_list)
        return lambda ch: ch.station.name in stations


def create_raw_store(raw_dir: str, sta_list: List[str], xml_path: str):
    fs = get_filesystem(raw_dir)

    def count(pat):
        return len(fs.glob(fs_join(raw_dir, pat)))

    # Use heuristics around which store to use by the files present
    if count("*.h5") > 0:
        return ASDFRawDataStore(raw_dir)
    else:
        assert count("*.ms") > 0 or count("*.sac") > 0, f"Can not find any .h5, .ms or .sac files in {raw_dir}"
        if xml_path is not None:
            catalog = XMLStationChannelCatalog(xml_path)
        elif os.path.isfile(os.path.join(raw_dir, STATION_FILE)):
            catalog = CSVChannelCatalog(raw_dir)
        else:
            raise ValueError(f"Either an --xml_path argument or a {STATION_FILE} must be provided")

        return SCEDCS3DataStore(raw_dir, catalog, get_channel_filter(sta_list))


def main(args: typing.Any):
    def run_cross_correlation():
        raw_dir = args.raw_data_path
        ccf_dir = args.ccf_path
        fft_params = initialize_fft_params(raw_dir)
        fft_params.freq_norm = args.freq_norm
        cc_store = ASDFCCStore(ccf_dir)
        raw_store = create_raw_store(raw_dir, args.stations, args.xml_path)
        cross_correlate(raw_store, fft_params, cc_store)

    def run_download():
        params = ConfigParameters()
        params.start_date = args.start
        params.end_date = args.end
        params.inc_hours = args.inc_hours
        download(args.raw_data_path, args.channels, args.stations, params)

    if args.step == Step.DOWNLOAD:
        run_download()
    if args.step == Step.CROSS_CORRELATE:
        run_cross_correlation()
    if args.step == Step.STACK:
        stack(args.raw_data_path, args.ccf_path, args.stack_path, args.method)
    if args.step == Step.ALL:
        run_download()
        run_cross_correlation()
        stack(args.raw_data_path, args.ccf_path, args.stack_path, args.method)


def add_download_args(down_parser: argparse.ArgumentParser):
    down_parser.add_argument(
        "--start",
        type=valid_date,
        required=True,
        help="Start date in the format: " + DATE_FORMAT_HELP,
        default=default_start_date,
    )
    down_parser.add_argument(
        "--end",
        type=valid_date,
        required=True,
        help="End date in the format: " + DATE_FORMAT_HELP,
        default=default_end_date,
    )
    down_parser.add_argument(
        "--channels",
        type=lambda s: s.split(","),
        help="Comma separated list of channels",
        default="BHE,BHN,BHZ",
    )
    down_parser.add_argument("--inc_hours", type=int, default=24, help="Time increment size (hrs)")


def add_stations_arg(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--stations",
        type=lambda s: s.split(","),
        help="Comma separated list of stations or '*' for all",
        default="*",
    )


def add_path(parser, prefix: str):
    parser.add_argument(
        f"--{prefix}_path",
        type=str,
        default=os.path.join(os.path.join(os.path.expanduser("~"), default_data_path), prefix.upper()),
        help=f"Directory for {prefix} data",
    )


def add_paths(parser, types: List[str]):
    for t in types:
        add_path(parser, t)


def add_cc_args(parser):
    parser.add_argument("--freq_norm", choices=["rma", "no", "phase_only"], default="rma")
    parser.add_argument("--xml_path", required=False, default=None)


def add_stack_args(parser):
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=[
            "linear",
            "pws",
            "robust",
            "nroot",
            "selective",
            "auto_covariance",
            "all",
        ],
        help="Stacking method",
    )


def make_step_parser(subparsers: Any, step: Step, parser_config_funcs: List[Callable[[argparse.ArgumentParser], None]]):
    parser = subparsers.add_parser(
        step.name.lower(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for config_fn in parser_config_funcs:
        config_fn(parser)


def main_cli():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="step", required=True)
    make_step_parser(
        subparsers, Step.DOWNLOAD, [lambda p: add_paths(p, ["raw_data"]), add_download_args, add_stations_arg]
    )
    make_step_parser(
        subparsers, Step.CROSS_CORRELATE, [lambda p: add_paths(p, ["raw_data", "ccf"]), add_cc_args, add_stations_arg]
    )
    make_step_parser(subparsers, Step.STACK, [lambda p: add_paths(p, ["raw_data", "stack", "ccf"]), add_stack_args])
    make_step_parser(
        subparsers,
        Step.ALL,
        [
            lambda p: add_paths(p, ["raw_data", "ccf", "stack"]),
            add_download_args,
            add_cc_args,
            add_stack_args,
            add_stations_arg,
        ],
    )

    args = parser.parse_args()
    args.step = Step[args.step.upper()]
    main(args)


if __name__ == "__main__":
    main_cli()
