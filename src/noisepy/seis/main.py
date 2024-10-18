import argparse
import logging
import os
import random
import sys
import typing
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterable, List, Optional

import dateutil.parser
from datetimerange import DateTimeRange

from noisepy.seis.io.asdfstore import ASDFCCStore, ASDFRawDataStore, ASDFStackStore
from noisepy.seis.io.channel_filter_store import (
    LocationChannelFilterStore,
    channel_filter,
)
from noisepy.seis.io.channelcatalog import CSVChannelCatalog, XMLStationChannelCatalog
from noisepy.seis.io.datatypes import ConfigParameters
from noisepy.seis.io.numpystore import NumpyCCStore, NumpyStackStore
from noisepy.seis.io.s3store import SCEDCS3DataStore
from noisepy.seis.io.utils import fs_join, get_filesystem, io_retry
from noisepy.seis.io.zarrstore import ZarrCCStore, ZarrStackStore

from . import __version__
from .constants import CONFIG_FILE, STATION_FILE
from .correlate import cross_correlate
from .fdsn_download import download
from .scheduler import (
    AWSBatchArrayScheduler,
    MPIScheduler,
    Scheduler,
    SingleNodeScheduler,
)
from .stack import stack_cross_correlations

logger = logging.getLogger(__name__)
# Utility running the different steps from the command line. Defines the arguments for each step

default_data_path = "noisepy_data"


class Command(Enum):
    DOWNLOAD = 1
    CROSS_CORRELATE = 2
    STACK = 3


class DataFormat(Enum):
    ZARR = "zarr"
    ASDF = "asdf"
    NUMPY = "numpy"


def list_str(values: str) -> List[str]:
    return values.split(",")


def _valid_config_file(parser, f: str) -> str:
    fs = get_filesystem(f, storage_options={})
    if fs.exists(f):
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
        return parse_utc
    if arg_type == bool:
        return parse_bool
    return arg_type


def parse_utc(dstr: str) -> datetime:
    d = dateutil.parser.isoparse(dstr)
    if d.timetz().tzinfo is None:
        return d.replace(tzinfo=timezone.utc)
    return d


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
    params = ConfigParameters()
    config_path = args.config
    if config_path is None and data_dir is not None:
        config_path = fs_join(data_dir, CONFIG_FILE)
    if config_path is None:
        logger.warning("No config file specified. Using default parameters.")
    else:
        fs = get_filesystem(config_path, storage_options={})
        if fs.exists(config_path):
            logger.info(f"Loading parameters from {config_path}")
            params = ConfigParameters.load_yaml(config_path)
        else:
            logger.warning(f"Config file '{config_path}' not found. Using default parameters.")
    cpy = params.model_copy(update={k: v for (k, v) in vars(args).items() if k in params.model_fields})
    return cpy


def create_raw_store(args, params: ConfigParameters):
    raw_dir = args.raw_data_path

    fs = get_filesystem(raw_dir, storage_options=params.storage_options)

    def count(pat):
        return len(fs.glob(fs_join(raw_dir, pat)))

    # Use heuristics around which store to use by the files present
    if count("*.h5") > 0:
        return ASDFRawDataStore(raw_dir)
    else:
        # assert count("*.ms") > 0 or count("*.sac") > 0, f"Can not find any .h5, .ms or .sac files in {raw_dir}"
        if args.xml_path is not None:
            catalog = XMLStationChannelCatalog(args.xml_path, storage_options=params.storage_options)
        elif os.path.isfile(os.path.join(raw_dir, STATION_FILE)):
            catalog = CSVChannelCatalog(raw_dir)
        else:
            raise ValueError(f"Either an --xml_path argument or a {STATION_FILE} must be provided")

        store = SCEDCS3DataStore(
            raw_dir,
            catalog,
            channel_filter(params.networks, params.stations, params.channels),
            DateTimeRange(params.start_date, params.end_date),
            params.storage_options,
        )
        # Some SCEDC channels have duplicates differing only by location, so filter them out
        return LocationChannelFilterStore(store)


def save_log(data_dir: str, log_file: Optional[str], storage_options: dict = {}):
    if log_file is None:
        return
    fs = get_filesystem(data_dir, storage_options=storage_options)
    fs.makedirs(data_dir, exist_ok=True)
    # Add a random suffix to make sure we don't override logs from previous runs
    # or from other nodes (in array jobs)
    unique_suffix = f".{random.randint(0, 10000000):06d}"
    fs.put(log_file, fs_join(data_dir, os.path.basename(log_file) + unique_suffix))


def get_scheduler(args) -> Scheduler:
    if args.mpi:
        return MPIScheduler(0)
    elif AWSBatchArrayScheduler.is_array_job():
        return AWSBatchArrayScheduler()
    else:
        return SingleNodeScheduler()


def makedir(dir: str, storage_options: dict = {}):
    fs = get_filesystem(dir, storage_options=storage_options)
    fs.makedirs(dir, exist_ok=True)


class ErrorStopHandler(logging.Handler):
    def handle(self, record):
        if record.levelno == logging.ERROR:
            raise RuntimeError(record.getMessage())
        return record


def main(args: typing.Any):
    logger = logging.getLogger(__package__)
    logger.setLevel(args.loglevel.upper())

    if args.logfile is not None:
        fh = logging.FileHandler(
            args.logfile,
        )
        fh.setFormatter(logging.Formatter("%(asctime)s\t%(levelname)s\t%(module)s.%(funcName)s():\t%(message)s"))
        logging.getLogger("").addHandler(fh)

    logger.info(f"NoisePy version: {__version__}")
    # _enable_s3fs_debug_logs()

    def get_cc_store(args, params: ConfigParameters, mode="a"):
        if args.format == DataFormat.ZARR.value:
            return ZarrCCStore(args.ccf_path, mode=mode, storage_options=params.get_storage_options(args.ccf_path))
        elif args.format == DataFormat.NUMPY.value:
            return NumpyCCStore(args.ccf_path, mode=mode, storage_options=params.get_storage_options(args.ccf_path))
        else:
            return ASDFCCStore(args.ccf_path, mode=mode)

    def get_stack_store(args, params: ConfigParameters):
        if args.format == DataFormat.ZARR.value:
            return ZarrStackStore(
                args.stack_path, mode="a", storage_options=params.get_storage_options(args.stack_path)
            )
        elif args.format == DataFormat.NUMPY.value:
            return NumpyStackStore(
                args.stack_path, mode="a", storage_options=params.get_storage_options(args.stack_path)
            )
        else:
            return ASDFStackStore(args.stack_path, "a")

    def cmd_wrapper(cmd: Callable[[ConfigParameters], None], src_dir: str, tgt_dir: str):
        storage_options = {}
        try:
            params = initialize_params(args, src_dir)
            storage_options = params.storage_options
            makedir(tgt_dir, storage_options)
            logger.info(f"Running {args.cmd.name}. Start: {params.start_date}. End: {params.end_date}")
            io_retry(params.save_yaml, fs_join(tgt_dir, CONFIG_FILE))
            cmd(params)
        except Exception as e:
            logger.exception(e)
            logging.shutdown()
            raise e
        finally:
            save_log(tgt_dir, args.logfile, storage_options)

    def run_download(params: ConfigParameters):
        download(args.raw_data_path, params)

    def run_cross_correlation(params: ConfigParameters):
        cc_store = get_cc_store(args, params)
        raw_store = create_raw_store(args, params)
        scheduler = get_scheduler(args)
        cross_correlate(raw_store, params, cc_store, scheduler)

    def run_stack(params: ConfigParameters):
        cc_store = get_cc_store(args, params, mode="r")
        stack_store = get_stack_store(args, params)
        scheduler = get_scheduler(args)
        stack_cross_correlations(cc_store, stack_store, params, scheduler)

    err_handler = ErrorStopHandler()
    if args.stop_on_error:
        logger.addHandler(err_handler)
    try:
        if args.cmd == Command.DOWNLOAD:
            cmd_wrapper(run_download, None, args.raw_data_path)
        if args.cmd == Command.CROSS_CORRELATE:
            cmd_wrapper(run_cross_correlation, args.raw_data_path, args.ccf_path)
        if args.cmd == Command.STACK:
            cmd_wrapper(run_stack, args.ccf_path, args.stack_path)
    finally:
        if args.stop_on_error:
            logger.removeHandler(err_handler)


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
    parser.add_argument("--logfile", type=str, default="log.txt", help="Log file")
    parser.add_argument(
        "-c", "--config", type=lambda f: _valid_config_file(parser, f), required=False, help="Configuration YAML file"
    )
    parser.add_argument("--stop_on_error", action="store_true", default=False, help="Stop on any errors")

    add_model(parser, ConfigParameters())

    if cmd != Command.DOWNLOAD:
        parser.add_argument(
            "--format",
            default=DataFormat.NUMPY.value,
            choices=[f.value for f in DataFormat],
            help="Format of the raw data files",
        )
    return parser


def add_mpi(parser: Any):
    parser.add_argument("-m", "--mpi", action="store_true")


def main_cli():
    try:
        args = parse_args(sys.argv[1:])
        main(args)
    finally:
        sys.stdout.flush()
        sys.stderr.flush()


def parse_args(arguments: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    make_step_parser(subparsers, Command.DOWNLOAD, ["raw_data"])
    add_mpi(make_step_parser(subparsers, Command.CROSS_CORRELATE, ["raw_data", "ccf", "xml"]))
    add_mpi(make_step_parser(subparsers, Command.STACK, ["raw_data", "stack", "ccf"]))

    args = parser.parse_args(arguments)

    args.cmd = Command[args.cmd.upper()]
    return args


# def _enable_s3fs_debug_logs():
#     os.environ["S3FS_LOGGING_LEVEL"] = "DEBUG"
#     for pkg in ["urllib3", "s3fs", "zarr"]:
#         logger.info("Enable debug log for %s", pkg)
#         lgr = logging.getLogger(pkg)
#         lgr.setLevel(logging.DEBUG)
#         lgr.propagate = True


if __name__ == "__main__":
    main_cli()
