"""
   isort:skip_file
"""
# no-qa F401
import logging

from ._version import __version__  # noqa: F401

from .fdsn_download import download  # noqa: F401
from .correlate import cross_correlate  # noqa: F401
from .stack import stack_cross_correlations  # noqa: F401

"""
NoisePy is a Python package designed for fast and easy computation of ambient noise cross-correlation functions.
It provides additional functionality for noise monitoring and surface wave dispersion analysis.

The main functions exported by the package are:
- download:         download continous noise data based on obspy's core functions of
                    https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_stations.html and
                    https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_waveforms.html
- cross_correlate:  This is the core function of NoisePy, which performs Fourier transform to all noise data first
                    and loads them into the memory before they are further cross-correlated
- stack_cross_correlations:
                    Used to assemble and/or stack all cross-correlation functions computed for the staion
                    pairs in the cross_correlate step
- noise_module:     Collection of functions used in the cross_correlate and stacking steps
"""

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(thread)d %(levelname)s %(module)s.%(funcName)s(): %(message)s"
)
logger = logging.getLogger(__name__)
