import argparse
import logging
import os
import sys
import typing
from datetime import datetime

# from datetime import datetime
from enum import Enum
from typing import Any, Callable, Iterable, List

import dateutil.parser
import obspy
from datetimerange import DateTimeRange

from .asdfstore import ASDFCCStore, ASDFRawDataStore, ASDFStackStore
from .channelcatalog import CSVChannelCatalog, XMLStationChannelCatalog
from .constants import CONFIG_FILE, STATION_FILE
from .datatypes import Channel, ConfigParameters
from .S0A_download_ASDF_MPI import download
from .S1_fft_cc_MPI import cross_correlate
from .S2_stacking import stack
from .scedc_s3store import SCEDCS3DataStore
from .scheduler import MPIScheduler, SingleNodeScheduler
from .utils import fs_join, get_filesystem
from .zarrstore import ZarrCCStore, ZarrStackStore

logger = logging.getLogger(__name__)
# Utility running the different steps from the command line. Defines the arguments for each step

default_data_path = "Documents/SCAL"


class Command(Enum):
    DOWNLOAD = 1
    CROSS_CORRELATE = 2
    STACK = 3
    ALL = 4


class DataFormat(Enum):
    ZARR = "zarr"
    ASDF = "asdf"


def valid_date(d: str) -> str:
    _ = obspy.UTCDateTime(d)
    return d


def list_str(values: str) -> List[str]:
    return values.split(",")


def _valid_config_file(parser, f: str) -> str:
    if os.path.isfile(f):
        return f
    parser.error(f"'{f}' is not a valid config file")


def parse_bool(bstr: str) -> bool:
    if bstr.upper() == "TRUE":
        return True
    elif bstr.upper() == "FALSE":
        return False
    raise ValueError(f"Invalid boolean value: '{bstr}'")


def get_arg_type(arg_type):
    if arg_type == List[str]:
        return list_str
    if arg_type == datetime:
        return dateutil.parser.isoparse
    if arg_type == bool:
        return parse_bool
    return arg_type


def add_model(parser: argparse.ArgumentParser, model: ConfigParameters):
    # Add config model to the parser
    fields = model.model_fields
    for name, field in fields.items():
        parser.add_argument(
            f"--{name}",
            dest=name,
            type=get_arg_type(field.annotation),
            default=argparse.SUPPRESS,
            help=field.description,
        )


def initialize_params(args, data_dir: str) -> ConfigParameters:
    """
    Loads initial parameters from 3 options:
    - --config_path option
    - <data_dir>/config.yaml
    - Default parameters

    Then overrides with values passed in the command line
    """
    config_path = args.config
    if config_path is None and data_dir is not None:
        config_path = fs_join(data_dir, CONFIG_FILE)
    if config_path is not None and os.path.isfile(config_path):
        logger.info(f"Loading parameters from {config_path}")
        params = ConfigParameters.load_yaml(config_path)
    else:
        logger.warning(f"Config file {config_path if config_path else ''} not found. Using default parameters.")
        params = ConfigParameters()
    cpy = params.model_copy(update={k: v for (k, v) in vars(args).items() if k in params.__fields__})
    return cpy


def get_channel_filter(sta_list: List[str]) -> Callable[[Channel], bool]:
    if len(sta_list) == 1 and sta_list[0] == "*":
        return lambda ch: True
    else:
        stations = set(sta_list)
        return lambda ch: ch.station.name in stations


def get_date_range(args) -> DateTimeRange:
    if "start_date" not in args or args.start_date is None or "end_date" not in args or args.end_date is None:
        return None
    return DateTimeRange(obspy.UTCDateTime(args.start_date).datetime, obspy.UTCDateTime(args.end_date).datetime)


def create_raw_store(args, params: ConfigParameters):
    raw_dir = args.raw_data_path

    fs = get_filesystem(raw_dir)

    def count(pat):
        return len(fs.glob(fs_join(raw_dir, pat)))

    # Use heuristics around which store to use by the files present
    if count("*.h5") > 0:
        return ASDFRawDataStore(raw_dir)
    else:
        # assert count("*.ms") > 0 or count("*.sac") > 0, f"Can not find any .h5, .ms or .sac files in {raw_dir}"
        if args.xml_path is not None:
            catalog = XMLStationChannelCatalog(args.xml_path)
        elif os.path.isfile(os.path.join(raw_dir, STATION_FILE)):
            catalog = CSVChannelCatalog(raw_dir)
        else:
            raise ValueError(f"Either an --xml_path argument or a {STATION_FILE} must be provided")

        date_range = get_date_range(args)
        return SCEDCS3DataStore(raw_dir, catalog, get_channel_filter(params.stations), date_range)


def main(args: typing.Any):
    logger = logging.getLogger(__package__)
    logger.setLevel(args.loglevel.upper())

    def run_download():
        params = initialize_params(args, None)
        download(args.raw_data_path, params)
        params.save_yaml(fs_join(args.raw_data_path, CONFIG_FILE))

    def get_cc_store(args, mode="a"):
        return (
            ZarrCCStore(args.ccf_path, mode=mode)
            if args.format == DataFormat.ZARR.value
            else ASDFCCStore(args.ccf_path, mode=mode)
        )

    def get_stack_store(args):
        return (
            ZarrStackStore(args.stack_path, mode="a")
            if args.format == DataFormat.ZARR.value
            else ASDFStackStore(args.stack_path, "a")
        )

    def run_cross_correlation():
        ccf_dir = args.ccf_path
        cc_store = get_cc_store(args)
        params = initialize_params(args, args.raw_data_path)
        raw_store = create_raw_store(args, params)
        scheduler = MPIScheduler(0) if args.mpi else SingleNodeScheduler()
        cross_correlate(raw_store, params, cc_store, scheduler)
        params.save_yaml(fs_join(ccf_dir, CONFIG_FILE))

    def run_stack():
        cc_store = get_cc_store(args, mode="r")
        stack_store = get_stack_store(args)
        params = initialize_params(args, args.ccf_path)
        scheduler = MPIScheduler(0) if args.mpi else SingleNodeScheduler()
        stack(cc_store, stack_store, params, scheduler)
        params.save_yaml(fs_join(args.stack_path, CONFIG_FILE))

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


def make_step_parser(subparsers: Any, cmd: Command, paths: List[str]) -> Any:
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

    if cmd != Command.DOWNLOAD:
        parser.add_argument(
            "--format",
            default=DataFormat.ZARR.value,
            choices=[f.value for f in DataFormat],
            help="Format of the raw data files",
        )
    return parser


def add_mpi(parser: Any):
    parser.add_argument("-m", "--mpi", action="store_true")


def main_cli():
    args = parse_args(sys.argv[1:])
    main(args)


def parse_args(arguments: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    make_step_parser(subparsers, Command.DOWNLOAD, ["raw_data"])
    add_mpi(make_step_parser(subparsers, Command.CROSS_CORRELATE, ["raw_data", "ccf", "xml"]))
    add_mpi(make_step_parser(subparsers, Command.STACK, ["raw_data", "stack", "ccf"]))
    add_mpi(make_step_parser(subparsers, Command.ALL, ["raw_data", "ccf", "stack", "xml"]))

    args = parser.parse_args(arguments)

    args.cmd = Command[args.cmd.upper()]
    return args


if __name__ == "__main__":
    main_cli()
