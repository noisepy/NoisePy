# About NoisePy
NoisePy is a Python package designed for fast and easy computation of ambient noise cross-correlation functions. It provides additional functionality for noise monitoring and surface wave dispersion analysis.

Disclaimer: this code should not be used "as-is" and not run like a blackbox. The user is expected to change local paths and parameters. Submit an issue to github with information such as the scripts+error messages to debug.

Detailed documentation can be found at https://noisepy.readthedocs.io/en/latest/

[![Documentation Status](https://readthedocs.org/projects/noisepy/badge/?version=latest)](https://noisepy.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/chengxinjiang/NoisePy.svg?branch=master)](https://travis-ci.com/github/chengxinjiang/NoisePy)
[![Codecov](https://codecov.io/gh/chengxinjiang/NoisePy/branch/master/graph/badge.svg)](https://codecov.io/gh/chengxinjiang/NoisePy)

<img src="https://raw.githubusercontent.com/mdenolle/NoisePy/master/docs/figures/logo.png" width="800" height="400">

## Major updates coming
NoisePy is going through a major refactoring to make this package easier to use and contribute.

# Installation
The nature of NoisePy being composed of python scripts allows flexible package installation, which is essentially to build dependent libraries the scripts and related functions live upon. We recommend using [conda](https://docs.conda.io/en/latest/) or [pip](https://pypi.org/project/pip/) to install the library due to their convenience. Below are command lines we have tested to create a python environment to run NoisePy. Note that the test is performed on `macOS Big Sur (11.7)`, so it could be slightly different for other OS.


### Note the order of the command lines below matters ###

# With Conda:
```bash
conda create -n noisepy python=3.8 pip
conda activate noisepy
conda install -c conda-forge openmpi
pip install noisepy-seis
```

# With virtual environment:
An MPI installation is required. E.g. for macOS using [brew](https://brew.sh/) :
```bash
brew install open-mpi
```

```bash
python -m venv noisepy
source noisepy/bin/activate
pip install noisepy-seis
```
To run the code on a single core, open the terminal and activate the noisepy environment before run following command. To run on institutional clusters, see installation notes for individual packages on the module list of the cluster. Examples of installation on Frontera are below.


# Deploy using Docker
We use I/O on disk, so users need root access to the file system. To install rootless docker, see instructions [here](https://docs.docker.com/engine/security/rootless/#install).
```bash
docker pull  ghcr.io/mdenolle/noisepy:latest
docker run -v ~/tmp:/tmp cross_correlate --path /tmp
```

# Functionality
Here is a list of features of the package:
* download continous noise data based on obspy's core functions of [get_station](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_stations.html) and [get_waveforms](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_waveforms.html)
* save seismic data in [ASDF](https://asdf-definition.readthedocs.io/en/latest/) format, which convinently assembles meta, wavefrom and auxililary data into one single file ([Tutorials](https://github.com/SeismicData/pyasdf/blob/master/doc/tutorial.rst) on reading/writing ASDF files)
* offers high flexibility to handle messy SAC/miniSEED data stored on your local machine and convert them into ASDF format data that could easily be pluged into NoisePy
* performs fast and easy cross-correlation with functionality to run in parallel through [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)
* includes a series of monitoring functions to measure dv/v on the resulted cross-correlation functions using some recently developed new methods (see our papers for more details<sup>**</sup>)

# Short tutorial
A short tutorial on how to use NoisePy-seis can be is available as a [Jupyter notebook](https://github.com/mdenolle/NoisePy/blob/master/Jupyter_notebook/tutorial.ipynb) and can be
[run directly in Colab](https://colab.research.google.com/github/mdenolle/NoisePy/blob/master/Jupyter_notebook/tutorial.ipynb).


This tutorial presents one simple example of how NoisePy might work! We strongly encourage you to download the NoisePy package and play it on your own! If you have any  comments and/or suggestions during running the codes, please do not hesitate to contact us through email or open an issue in this github page!

Chengxin Jiang (chengxinjiang@gmail.com)
Marine Denolle (mdenolle@uw.edu).

#### Peer-Reviewed Publications Referenced
Seats, K. J., Jesse F. L., and German A. P. "Improved ambient noise correlation functions using Welch′ s method." _Geophysical Journal International_ 188, no. 2 (2012): 513-523.
*Jiang, C. and Denolle, M. "NoisePy: a new high-performance python tool for seismic ambient noise seismology." _Seismological Research Letter_ 91, no. 3 (2020): 1853–1866..
** Yuan, C., Bryan, J. T., and Denolle, M. "Numerical comparison of time-, frequency- and wavelet-domain methods for coda wave interferometry." _Geophysical Journal International_ 226, no. 2 (2021): 828-846.


### Some taxonomy of the NoisePy variables.

* ``station`` refers to the site that has the seismic instruments that records ground shaking.
* `` channel`` refers to the direction of ground motion investigated for 3 component seismometers. For DAS project, it may refers to the single channel sensors.
* ``ista`` is the index name for looping over stations

* ``cc_len`` correlation length, basic window length in seconds
* ``step`` is the window that get skipped when sliding windows in seconds
* ``smooth_N`` number of points for smoothing the  time or frequency domain discrete arrays.
* ``maxlag`` maximum length in seconds saved in files in each side of the correlation (save on storage)
* ``substack,substack_len`` boolean, window length over which to substack the correlation (to save storage or do monitoring), it has to be a multiple of ``cc_len``.
* ``time_chunk, nchunk`` refers to the time unit that defined a single job. for instace, ``cc_len`` is the correlation length (e.g., 1 hour, 30 min), the overall duration of the experiment is the total length (1 month, 1 year, ...). The time chunk could be 1 day: the code would loop through each cc_len window in a for loop. But each day will be sent as a thread.
