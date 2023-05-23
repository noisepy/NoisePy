import argparse
import logging
import os
import typing

# from datetime import datetime
from enum import Enum
from typing import Any, Callable, List

import obspy
from datetimerange import DateTimeRange

from .asdfstore import ASDFCCStore, ASDFRawDataStore
from .channelcatalog import CSVChannelCatalog, XMLStationChannelCatalog
from .constants import STATION_FILE
from .datatypes import Channel, ConfigParameters
from .S0A_download_ASDF_MPI import download
from .S1_fft_cc_MPI import cross_correlate
from .S2_stacking import stack
from .scedc_s3store import SCEDCS3DataStore
from .utils import fs_join, get_filesystem

# Utility running the different steps from the command line. Defines the arguments for each step

default_data_path = "Documents/SCAL"


class Command(Enum):
    DOWNLOAD = 1
    CROSS_CORRELATE = 2
    STACK = 3
    ALL = 4


def valid_date(d: str) -> str:
    _ = obspy.UTCDateTime(d)
    return d


def list_str(values: str) -> List[str]:
    return values.split(",")


def _valid_config_file(parser, f: str) -> str:
    if os.path.isfile(f):
        return f
    parser.error(f"'{f}' is not a valid config file")


# TODO: DATETIMES
# TODO: Save config file each step?
# TODO: Comments to Fields in ConfigParameters
# TODO: Unit tests for Config arg parsing


def get_arg_type(arg_type):
    if arg_type == list:
        return list_str
    # if arg_type == datetime
    return arg_type


def add_model(parser: argparse.ArgumentParser, model: ConfigParameters):
    # Add config model to the parser
    fields = model.__fields__
    for name, field in fields.items():
        parser.add_argument(
            f"--{name}",
            dest=name,
            type=get_arg_type(field.type_),
            default=field.default,
            help=field.field_info.description,
        )


def initialize_params(args) -> ConfigParameters:
    if args.config is None:
        params = ConfigParameters()
    else:
        params = ConfigParameters.load_yaml(args.config)
    cpy = params.copy(update=vars(args))
    return cpy


def get_channel_filter(sta_list: List[str]) -> Callable[[Channel], bool]:
    if len(sta_list) == 1 and sta_list[0] == "*":
        return lambda ch: True
    else:
        stations = set(sta_list)
        return lambda ch: ch.station.name in stations


def get_date_range(args) -> DateTimeRange:
    if args.start is None or args.end is None:
        return None
    return DateTimeRange(obspy.UTCDateTime(args.start).datetime, obspy.UTCDateTime(args.end).datetime)


def create_raw_store(args, params: ConfigParameters):
    xml_path = args.xml_path if "xml_path" in args else None
    raw_dir = args.raw_data_path

    fs = get_filesystem(raw_dir)

    def count(pat):
        return len(fs.glob(fs_join(raw_dir, pat)))

    # Use heuristics around which store to use by the files present
    if count("*.h5") > 0:
        return ASDFRawDataStore(raw_dir)
    else:
        # assert count("*.ms") > 0 or count("*.sac") > 0, f"Can not find any .h5, .ms or .sac files in {raw_dir}"
        if xml_path is not None:
            catalog = XMLStationChannelCatalog(xml_path)
        elif os.path.isfile(os.path.join(raw_dir, STATION_FILE)):
            catalog = CSVChannelCatalog(raw_dir)
        else:
            raise ValueError(f"Either an --xml_path argument or a {STATION_FILE} must be provided")

        date_range = get_date_range(args)
        return SCEDCS3DataStore(raw_dir, catalog, get_channel_filter(params.stations), date_range)


def main(args: typing.Any):
    logger = logging.getLogger(__package__)
    logger.setLevel(args.loglevel.upper())
    params = initialize_params(args)

    def run_cross_correlation():
        ccf_dir = args.ccf_path
        cc_store = ASDFCCStore(ccf_dir)
        raw_store = create_raw_store(args, params)
        cross_correlate(raw_store, params, cc_store)

    def run_download():
        download(args.raw_data_path, params)

    def run_stack():
        cc_store = ASDFCCStore(args.ccf_path, "r")
        stack(cc_store, args.stack_path, params)

    if args.cmd == Command.DOWNLOAD:
        run_download()
    if args.cmd == Command.CROSS_CORRELATE:
        run_cross_correlation()
    if args.cmd == Command.STACK:
        run_stack()
    if args.cmd == Command.ALL:
        run_download()
        run_cross_correlation()
        run_stack()


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


def make_step_parser(subparsers: Any, cmd: Command, paths: List[str]):
    parser = subparsers.add_parser(
        cmd.name.lower(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_paths(parser, paths)
    parser.add_argument(
        "-log",
        "--loglevel",
        type=str.lower,
        default="info",
        choices=["notset", "debug", "info", "warning", "error", "critical"],
    )
    parser.add_argument(
        "-c", "--config", type=lambda f: _valid_config_file(parser, f), required=False, help="Configuration YAML file"
    )
    add_model(parser, ConfigParameters())


def main_cli():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    make_step_parser(subparsers, Command.DOWNLOAD, ["raw_data"])
    make_step_parser(subparsers, Command.CROSS_CORRELATE, ["raw_data", "ccf", "xml"])
    make_step_parser(subparsers, Command.STACK, ["raw_data", "stack", "ccf"])
    make_step_parser(subparsers, Command.ALL, ["raw_data", "ccf", "stack", "xml"])

    args, config_args = parser.parse_known_args()
    print(config_args)
    # return

    args.cmd = Command[args.cmd.upper()]
    main(args)


if __name__ == "__main__":
    main_cli()
