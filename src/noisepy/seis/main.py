import argparse
import os
import typing
from enum import Enum

import obspy

from .S0A_download_ASDF_MPI import download
from .S1_fft_cc_MPI import cross_correlate
from .S2_stacking import stack

# Utility running the different steps from the command line. Defines the arguments for each step

DATE_FORMAT = "%%Y_%%m_%%d_%%H_%%M_%%S"
default_start_date = "2016_07_01_0_0_0"  # start date of download
default_end_date = "2016_07_02_0_0_0"  # end date of download
default_data_path = "Documents/SCAL"


class Step(Enum):
    DOWNLOAD = 1
    CROSS_CORRELATE = 2
    STACK = 3


def valid_date(d: str) -> str:
    _ = obspy.UTCDateTime(d)
    return d


def main(args: typing.Any):
    if args.step == Step.DOWNLOAD:
        download(
            args.path,
            args.channels,
            args.stations,
            [args.start],
            [args.end],
            args.inc_hours,
        )
    if args.step == Step.CROSS_CORRELATE:
        cross_correlate(args.path, args.freq_norm)
    if args.step == Step.STACK:
        stack(args.path, args.method)


def main_cli():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="step", required=True)

    # Download arguments
    down_parser = subparsers.add_parser(
        Step.DOWNLOAD.name.lower(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    down_parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(os.path.expanduser("~"), default_data_path),
        help="Directory for downloading files",
    )
    down_parser.add_argument(
        "--start",
        type=valid_date,
        required=True,
        help="Start date in the format: " + DATE_FORMAT,
        default=default_start_date,
    )
    down_parser.add_argument(
        "--end",
        type=valid_date,
        required=True,
        help="End date in the format: " + DATE_FORMAT,
        default=default_end_date,
    )
    down_parser.add_argument(
        "--stations",
        type=lambda s: s.split(","),
        help="Comma separated list of stations or '*' for all",
        default="*",
    )
    down_parser.add_argument(
        "--channels",
        type=lambda s: s.split(","),
        help="Comma separated list of channels",
        default="BHE,BHN,BHZ",
    )
    down_parser.add_argument("--inc_hours", type=int, default=24, help="Time increment size (hrs)")
    # Cross_correlate arguments
    cc_parser = subparsers.add_parser(
        Step.CROSS_CORRELATE.name.lower(),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cc_parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(os.path.expanduser("~"), default_data_path),
        help="Directory to look for input files",
    )
    cc_parser.add_argument("--freq_norm", choices=["rma", "no", "phase_only"], default="rma")
    # Stack arguments
    stack_parser = subparsers.add_parser(
        Step.STACK.name.lower(), formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    stack_parser.add_argument(
        "--path",
        type=str,
        default=os.path.join(os.path.expanduser("~"), default_data_path),
        help="Directory to look for input files",
    )
    stack_parser.add_argument(
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

    args = parser.parse_args()
    args.step = Step[args.step.upper()]
    main(args)


if __name__ == "__main__":
    main_cli()
