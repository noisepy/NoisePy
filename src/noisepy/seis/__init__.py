# no-qa F401
import logging

from ._version import __version__  # noqa: F401
from .S0A_download_ASDF_MPI import download  # noqa: F401
from .S1_fft_cc_MPI import cross_correlate  # noqa: F401
from .S2_stacking import stack  # noqa: F401

"""
NoisePy is a Python package designed for fast and easy computation of ambient noise cross-correlation functions.
It provides additional functionality for noise monitoring and surface wave dispersion analysis.

The main functions exported by the package are:
- download:         download continous noise data based on obspy's core functions of
                    https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_stations.html and
                    https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_waveforms.html
- cross_correlate:  This is the core function of NoisePy, which performs Fourier transform to all noise data first
                    and loads them into the memory before they are further cross-correlated
- stack:            Used to assemble and/or stack all cross-correlation functions computed for the staion pairs in
                    the cross_correlate step
- noise_module:     Collection of functions used in the cross_correlate and stacking steps
- plotting_modules: Utility functions for plotting the data
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(module)s.%(funcName)s(): %(message)s")
logger = logging.getLogger(__name__)
