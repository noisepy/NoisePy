# About NoisePy
NoisePy is a Python package designed for fast and easy computation of ambient noise cross-correlation functions. It provides additional functionality for noise monitoring and surface wave dispersion analysis.

[![Documentation Status](https://github.com/noisepy/NoisePy/actions/workflows/notebooks.yml/badge.svg)](https://noisepy.github.io/NoisePy/)
[![Build Status](https://github.com/noisepy/NoisePy/actions/workflows/test.yaml/badge.svg)](https://github.com/noisepy/NoisePy/actions/workflows/test.yaml)
[![Codecov](https://codecov.io/gh/noisepy/NoisePy/branch/main/graph/badge.svg)](https://codecov.io/gh/noisepy/NoisePy)
[![DOI](https://zenodo.org/badge/157871462.svg)](https://zenodo.org/badge/latestdoi/157871462)

<img src="https://raw.githubusercontent.com/noisepy/NoisePy/main/docs_old/figures/logo.png" width="800" height="400">

## Major updates coming
NoisePy is going through a major refactoring to make this package easier to develop and deploy. Submit an issue, fork the repository and create pull requests to [contribute](CONTRIBUTING.md).

## Installation
The nature of NoisePy being composed of python scripts allows flexible package installation, which is essentially to build dependent libraries the scripts and related functions live upon. We recommend using [conda](https://docs.conda.io/en/latest/) or [pip](https://pypi.org/project/pip/) to install.

**Note the order of the command lines below matters**

### With Conda and pip
```bash
conda create -n noisepy -y python=3.10 pip
conda activate noisepy
pip install noisepy-seis
```

To add jupyter dependencies, install them
```
pip install ipykernel notebook
python -m ipykernel install --user --name noisepy
```

### With Conda and pip and MPI support
```bash
conda create -n noisepy -y python=3.10 pip mpi4py
conda activate noisepy
pip install noisepy-seis[mpi]
```

### With virtual environment
```bash
python -m venv noisepy
source noisepy/bin/activate
pip install noisepy-seis
```

### With virtual environment and MPI support
An MPI installation is required. E.g. for macOS using [brew](https://brew.sh/) :
```bash
brew install open-mpi
```

```bash
python -m venv noisepy
source noisepy/bin/activate
pip install noisepy-seis[mpi]
```

## Functionality
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

## Usage

To run the code on a single core, open the terminal and activate the noisepy environment before run following commands. To run on institutional clusters, see installation notes for individual packages on the module list of the cluster.

### Deploy using Docker
We use I/O on disk, so users need root access to the file system. To install rootless docker, see instructions [here](https://docs.docker.com/engine/security/rootless/#install).
```bash
docker pull  ghcr.io/noisepy/noisepy:latest
docker run -v ~/tmp:/tmp ghcr.io/noisepy/noisepy:latest cross_correlate --path /tmp
```

## Tutorials
Short tutorials on how to use NoisePy can be is available [here](https://noisepy.github.io/NoisePy/) and can be run directly in Colab. These tutorials present simple examples of how NoisePy might work. We strongly encourage you to download the NoisePy package and play it on your own! If you have any comments and/or suggestions during running the codes, please do not hesitate to contact us through email or open an issue in this github page!

Chengxin Jiang (chengxinjiang@gmail.com)
Marine Denolle (mdenolle@uw.edu)
Yiyu Ni (niyiyu@uw.edu)

### Taxonomy
Taxonomy of the NoisePy variables.

* ``station`` refers to the site that has the seismic instruments that records ground shaking.
* ``channel`` refers to the direction of ground motion investigated for 3 component seismometers. For DAS project, it may refers to the single channel sensors.
* ``ista`` is the index name for looping over stations
* ``cc_len`` correlation length, basic window length in seconds
* ``step`` is the window that get skipped when sliding windows in seconds
* ``smooth_N`` number of points for smoothing the  time or frequency domain discrete arrays.
* ``maxlag`` maximum length in seconds saved in files in each side of the correlation (save on storage)
* ``substack, substack_windows`` boolean, number of window over which to substack the correlation (to save storage or do monitoring).
* ``time_chunk, nchunk`` refers to the time unit that defined a single job. for instace, ``cc_len`` is the correlation length (e.g., 1 hour, 30 min), the overall duration of the experiment is the total length (1 month, 1 year, ...). The time chunk could be 1 day: the code would loop through each cc_len window in a for loop. But each day will be sent as a thread.

## Acknowledgements
Thanks to our contributors so far!

[![Contributors](https://contrib.rocks/image?repo=noisepy/NoisePy)](https://github.com/noisepy/NoisePy/graphs/contributors)

### Use this reference when publishing on your work with noisepy

Main code:

* Zenodo DOI: [noisepy/NoisePy](https://zenodo.org/badge/latestdoi/157871462)
* Jiang, C. and Denolle, M. [NoisePy: a new high-performance python tool for seismic ambient noise seismology.](https://doi.org/10.1785/0220190364) _Seismological Research Letter_ 91, no. 3 (2020): 1853–1866. https://doi.org/10.1785/0220190364

Algorithms used:
* (data pre-processing) Seats, K. J., Jesse F. L., and German A. P. [Improved ambient noise correlation functions using Welch′ s method.](https://doi.org/10.1111/j.1365-246X.2011.05263.x) _Geophysical Journal International_ 188, no. 2 (2012): 513-523. https://doi.org/10.1111/j.1365-246X.2011.05263.x

* (dv/v in wavelet domain) Yuan, C., Bryan, J. T., and Denolle, M. [Numerical comparison of time-, frequency- and wavelet-domain methods for coda wave interferometry.](https://doi.org/10.1093/gji/ggab140) _Geophysical Journal International_ 226, no. 2 (2021): 828-846. https://doi.org/10.1093/gji/ggab140

* (optimal stacking) Yang X, Bryan J, Okubo K, Jiang C, Clements T, Denolle MA. [Optimal stacking of noise cross-correlation functions/](https://doi.org/10.1093/gji/ggac410) _Geophysical Journal International_. 2023 Mar;232(3):1600-18. https://doi.org/10.1093/gji/ggac410

This research received software engineering support from the University of Washington’s Scientific Software Engineering Center ([SSEC](https://escience.washington.edu/software-engineering/ssec/)) supported by Schmidt Futures, as part of the Virtual Institute for Scientific Software (VISS). We would like to acknowledge [Carlos Garcia Jurado Suarez](https://github.com/carlosgjs) and [Nicholas Rich](https://github.com/nrich20) for their collaboration and contributions to the software.
