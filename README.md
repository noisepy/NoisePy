# About NoisePy
NoisePy is a Python package designed for fast and easy computation of ambient noise cross-correlation functions. It provides additional functionality for noise monitoring and surface wave dispersion analysis.

Disclaimer: this code should not be used "as-is" and not run like a blackbox. The user is expected to change local paths and parameters. Submit an issue to github with information such as the scripts+error messages to debug.

Detailed documentation can be found at https://noisepy.readthedocs.io/en/latest/

[![Documentation Status](https://readthedocs.org/projects/noisepy/badge/?version=latest)](https://noisepy.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/chengxinjiang/NoisePy.svg?branch=master)](https://travis-ci.com/github/chengxinjiang/NoisePy)
[![Codecov](https://codecov.io/gh/chengxinjiang/NoisePy/branch/master/graph/badge.svg)](https://codecov.io/gh/chengxinjiang/NoisePy)

<img src="https://raw.githubusercontent.com/mdenolle/NoisePy/master/docs/figures/logo.png" width="800" height="400">

## Major updates coming
NoisePy is going through a major refactoring to make this package easier to develop and deploy. Submit an issue, fork the repository and create pull requests to contribute.

# Installation
The nature of NoisePy being composed of python scripts allows flexible package installation, which is essentially to build dependent libraries the scripts and related functions live upon. We recommend using [conda](https://docs.conda.io/en/latest/) or [pip](https://pypi.org/project/pip/) to install.

### Note the order of the command lines below matters ###

## With Conda and pip:
```bash
conda create -n noisepy python=3.8 pip
conda activate noisepy
conda install -c conda-forge openmpi
pip install noisepy-seis
```

## With virtual environment:
An MPI installation is required. E.g. for macOS using [brew](https://brew.sh/) :
```bash
brew install open-mpi
```

```bash
python -m venv noisepy
source noisepy/bin/activate
pip install noisepy-seis
```


# Functionality
Here is a list of features of the package:
* download continous noise data based:
   + on webservices using obspy's core functions of [get_station](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_stations.html) and [get_waveforms](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_waveforms.html)
   + on AWS S3 bucket calls, with a test on the SCEDC AWS Open Dataset.
* save seismic data in [ASDF](https://asdf-definition.readthedocs.io/en/latest/) format, which convinently assembles meta, wavefrom and auxililary data into one single file ([Tutorials](https://github.com/SeismicData/pyasdf/blob/master/doc/tutorial.rst) on reading/writing ASDF files)
* offers scripts to precondition data sets before cross correlations. This involves working with gappy data from various formats (SAC/miniSEED) and storing it on local in ASDF.

* performs fast and easy cross-correlation with functionality to run in parallel through [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)
* **Applications module**:
   + *Ambient noise monitoring*: measure dv/v using a wide variety of techniques in time, fourier, and wavelet domain (Yuan et al., 2021)
   + *Surface wave dispersion*: construct dispersion images using conventional techniques. 
   


# Usage

To run the code on a single core, open the terminal and activate the noisepy environment before run following commands. To run on institutional clusters, see installation notes for individual packages on the module list of the cluster. 

## Deploy using Docker
We use I/O on disk, so users need root access to the file system. To install rootless docker, see instructions [here](https://docs.docker.com/engine/security/rootless/#install).
```bash
docker pull  ghcr.io/mdenolle/noisepy:latest
docker run -v ~/tmp:/tmp cross_correlate --path /tmp
```

# Tutorials
A short tutorial on how to use NoisePy-seis can be is available as a [Jupyter notebook](https://github.com/mdenolle/NoisePy/blob/master/tutorials/get_started.ipynb) and can be
[run directly in Colab](https://colab.research.google.com/github/mdenolle/NoisePy/blob/master/tutorials/get_started.ipynb).


This tutorial presents one simple example of how NoisePy might work! We strongly encourage you to download the NoisePy package and play it on your own! If you have any  comments and/or suggestions during running the codes, please do not hesitate to contact us through email or open an issue in this github page!

Chengxin Jiang (chengxinjiang@gmail.com)
Marine Denolle (mdenolle@uw.edu).

## Use this reference when publishing on your work with noisepy

Main code:
* Jiang, C. and Denolle, M. [NoisePy: a new high-performance python tool for seismic ambient noise seismology.](https://doi.org/10.1785/0220190364) _Seismological Research Letter_ 91, no. 3 (2020): 1853–1866. https://doi.org/10.1785/0220190364

Algorithms used:
* (data pre-processing) Seats, K. J., Jesse F. L., and German A. P. [Improved ambient noise correlation functions using Welch′ s method.](https://doi.org/10.1111/j.1365-246X.2011.05263.x) _Geophysical Journal International_ 188, no. 2 (2012): 513-523. https://doi.org/10.1111/j.1365-246X.2011.05263.x

* (dv/v in wavelet domain) Yuan, C., Bryan, J. T., and Denolle, M. [Numerical comparison of time-, frequency- and wavelet-domain methods for coda wave interferometry.](https://doi.org/10.1093/gji/ggab140) _Geophysical Journal International_ 226, no. 2 (2021): 828-846. https://doi.org/10.1093/gji/ggab140

* (optimal stacking) Yang X, Bryan J, Okubo K, Jiang C, Clements T, Denolle MA. [Optimal stacking of noise cross-correlation functions/](https://doi.org/10.1093/gji/ggac410) _Geophysical Journal International_. 2023 Mar;232(3):1600-18. https://doi.org/10.1093/gji/ggac410



## Parameters to configure

The parameters of the workflow are saved into an object called ``ConfigParameter``. The parameters

 *   dt: float = 0.05  # TODO: dt should be 1/sampling rate
 *   start_date: str = ""  # TODO: can we make this datetime?
 *   end_date: str = ""
 *   samp_freq: float = 20  # TODO: change this samp_freq for the obspy "sampling_rate"
*  cc_len: float = 1800.0  # basic unit of data length for fft (sec)
 *   # download params.
 *   # Targeted region/station information: only needed when down_list is False
 *  lamin: float = 31
 *   lamax: float = 36
 *   lomin: float = -122
 *   lomax: float = -115
 *   down_list = False  # download stations from a pre-compiled list or no 
 *   net_list = ["CI"]  # network list
    # pre-processing parameters
 *   step: float = 450.0  # overlapping between each cc_len (sec)
 *  freqmin: float = 0.05
 *   freqmax: float = 2.0
 *   freq_norm: str = "rma"  # choose between "rma" for a soft whitenning or "no" for no whitening
 *   #  TODO: change "no" for "None", and add "one_bit" as an option
 *   time_norm: str = "no"  # 'no' for no normalization, or 'rma', 'one_bit' for normalization in time domain,
 *   # TODO: change time_norm option from "no" to "None"
 *   cc_method: str = "xcorr"  # 'xcorr' for pure cross correlation, 'deconv' for deconvolution;
 *   # FOR "COHERENCY" PLEASE set freq_norm to "rma", time_norm to "no" and cc_method to "xcorr"
 *   smooth_N: int = 10  # moving window length for time/freq domain normalization if selected (points)
 *   smoothspect_N: int = 10  # moving window length to smooth spectrum amplitude (points)
 *   # if substack=True, substack_len=2*cc_len, then you pre-stack every 2 correlation windows.
 *   # for instance: substack=True, substack_len=cc_len means that you keep ALL of the correlations
 *   substack: bool = True  # True = smaller stacks within the time chunk. False: it will stack over inc_hours
 *   substack_len: int = 1800  # how long to stack over (for monitoring purpose): need to be multiples of cc_len
 *   maxlag: int = 200  # lags of cross-correlation to save (sec)
 *   substack: bool = True
 *   inc_hours: int = 24
 *  # criteria for data selection
 *  max_over_std: int = 10  # threahold to remove window of bad signals: set it to 10*9 if prefer not to remove them
 *   ncomp: int = 3  # 1 or 3 component data (needed to decide whether do rotation)
 *   # station/instrument info for input_fmt=='sac' or 'mseed'
 *   stationxml: bool = False  # station.XML file used to remove instrument response for SAC/miniseed data
 *   rm_resp: str = "no"  # select 'no' to not remove response and use 'inv','spectrum',
 *   rm_resp_out: str = "VEL"  # output location from response removal
 *   respdir: str = None  # response directory
 *   # some control parameters
 *   acorr_only: bool = False  # only perform auto-correlation
 *   xcorr_only: bool = True  # only perform cross-correlation or not


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
