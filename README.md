# About NoisePy
NoisePy is a Python package designed for fast and easy computation of ambient noise cross-correlation functions. It provides additional functionality for noise monitoring and surface wave dispersion analysis.

Disclaimer: this code should not be used "as-is" and not run like a blackbox. The user is expected to change local paths and parameters. Submit an issue to github with information such as the scripts+error messages to debug.

Detailed documentation can be found at https://noisepy.readthedocs.io/en/latest/

[![Documentation Status](https://readthedocs.org/projects/noisepy/badge/?version=latest)](https://noisepy.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/chengxinjiang/NoisePy.svg?branch=master)](https://travis-ci.com/github/chengxinjiang/NoisePy)
[![Codecov](https://codecov.io/gh/chengxinjiang/NoisePy/branch/master/graph/badge.svg)](https://codecov.io/gh/chengxinjiang/NoisePy)

<img src="https://raw.githubusercontent.com/mdenolle/NoisePy/master/docs/figures/logo.png" width="800" height="400">

# Citation:
Please cite the following reference if you use the code for your publication:
Jiang, C. and Denolle, M. "NoisePy: a new high-performance python tool for seismic ambient noise seismology." Seismological Research Letter 91 (3): 1853–1866.

## Major updates include
* adding options for several stacking methods such as nth-root, robust-stacking, auto-covariance and selective. A script is added to the folder of application_modules to cross-compare the effects of different stacking method (note that `substack` parameter in S2 has to be `True` in order to use it)
* adding a jupter notebook for tutorials on performing seismic monitoring analysis using NoisePy
* adding a jupter notebook for generating response spectrum for a nodal array (to be done)

# Installation
The nature of NoisePy being composed of python scripts allows flexible package installation, which is essentially to build dependent libraries the scripts and related functions live upon. We recommend using [conda](https://docs.conda.io/en/latest/) or [pip](https://pypi.org/project/pip/) to install the library due to their convenience. Below are command lines we have tested to create a python environment to run NoisePy. Note that the test is performed on `macOS Mojave (10.14.5)`, so it could be slightly different for other OS.


### Note the order of the command lines below matters ###

# With Conda:
```bash
$ conda create -n noisepy python=3.8 pip
$ conda activate noisepy
$ conda install -c conda-forge openmpi
$ pip install noisepy-seis
```

# With virtual environment:
An MPI installation is required. E.g. for macOS using [brew](https://brew.sh/) :
```sh
$ brew install open-mpi
```

```sh
$ python -m venv noisepy
$ source noisepy/bin/activate
$ pip install noisepy-seis
```
To run the code on a single core, open the terminal and activate the noisepy environment before run following command. To run on institutional clusters, see installation notes for individual packages on the module list of the cluster. Examples of installation on Frontera are below.

# Functionality
* download continous noise data based on obspy's core functions of [get_station](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_stations.html) and [get_waveforms](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_waveforms.html)
* save seismic data in [ASDF](https://asdf-definition.readthedocs.io/en/latest/) format, which convinently assembles meta, wavefrom and auxililary data into one single file ([Tutorials](https://github.com/SeismicData/pyasdf/blob/master/doc/tutorial.rst) on reading/writing ASDF files)
* offers high flexibility to handle messy SAC/miniSEED data stored on your local machine and convert them into ASDF format data that could easily be pluged into NoisePy
* performs fast and easy cross-correlation with functionality to run in parallel through [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface)
* includes a series of monitoring functions to measure dv/v on the resulted cross-correlation functions using some recently developed new methods (see our papers for more details<sup>**</sup>)

# Short tutorial

### 0A. Downloading seismic noise data
This command allows to download all available broadband CI stations `(BH?)` located in a certain region and operated during 1/Jul/2016-2/Jul/2016 through the SCEC data center.

In the script, short summary is provided for all input parameters that can be changed according to the user's needs. In the current form of the script, we set `inc_hours=24` to download day-long continous noise data as well as the meta info and store them into a single ASDF file. To increase the signal-to-noise (SNR) of the final cross-correlation functions (see Seats et al.,2012 for more details), we break the day-long sequence into smaller segments, each of `cc_len` (s) long with some overlapping defined by `step`. You may wanto to set `flag` to be `True` if intermediate outputs/operational time is preferred during the downloading process.

```sh
$ noisepy download
```
The data to be downloaded can be customized via command line arguments. See `noisepy download --help` for details.

If you want to use multiple cores (e.g, 4), run the script with the following command using [mpi4py](https://mpi4py.readthedocs.io/en/stable/).
```sh
$ mpirun -n 4 noisepy download
```

The outputted files from the download include ASDF files containing daily-long (24h) continous noise data, a parameter file recording all used parameters in the download and a CSV file of all station information (more details on reading the ASDF files with downloaded data can be found in docs/src/ASDF.md). The continous waveforms data stored in the ASDF file can be displayed using the plotting modules named as `plotting_modules` in the directory of `src` as shown below.

```python
from noisepy.seis import plotting_modules
sfile = '/Users/chengxin/Documents/SCAL/RAW_DATA/2016_07_01_00_00_00T2016_07_02_00_00_00.h5'
plotting_modules.plot_waveform(sfile,'CI','BLC',0.01,0.4)
```
<img src="https://raw.githubusercontent.com/mdenolle/NoisePy/master/docs/figures/waveform3.png" width="600" height="400">

Note that the script also offers the option to download data from an existing station list in a format same to the outputed CSV file. In this case, `down_list` should be set to `True` at L53. In reality, the downloading speed is dependent on many factors such as the original sampling rate of targeted data, the networks, the data center where it is hosted and the general structure you want to store on your machine etc. We tested a bunch of the parameters to evaluate their performance and the readers are referred to our paper for more details (Jiang et al., 2020).


### 0B. DEAL with local SAC/miniseed files using `S0B_to_ASDF.py`
If you want to use the NoisePy to handel local data in SAC/miniseed format stored on your own disk, this is the script you need. Most of the variables are the same as those for the download step and thus should be pretty straighforward to follow and change. In this script, it preprocesses the data by merging, detrending, demeaning, downsampling and then trimming before saving them into ASDF format for later NoisePy processing. In particular, we expect the script to deal with very messydata, by which we mean that, seismic data is broken into small pieces and of messy time info such as overlapping time. REMEMBER to set `messydata` at L62 to `True` when you have messy data! (Tutorial on removing instrument response)



### 1. Perform cross correlations
This is the core function of NoisePy, which performs [Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform) to all noise data first and loads them into the memory before they are further cross-correlated. This means that we are performing [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation) in the frequency domain. In the script, we provide several options to calculate the cross correlation, including `raw`, `coherency` and `deconv` (see our paper<sup>*</sup> for detailed definition). We choose `coherency` as an example here. After running the script, it will create a new folder named `CCF`, in which new ASDF files containing all cross-correlation functions between different station pairs are located. It also creates a parameter file of `fft_cc_data.txt` that records all useful parameters used in this script.


```sh
$ noisepy cross_correlate
```
If you downloaded the data to a custom location, specify the `--path` argument. See `noisepy cross_correlate --help` for details.

Once you get the cross-correlation file, you can show the daily temporal variation between all station-pair by calling `plot_substack_cc` function in `plotting_modules` as follows. NOTE that to make this plot, the parameter of `substack` has to be set to `True` in S1.


```python
from noisepy.seis import plotting_modules
sfile = '/Users/chengxin/Documents/SCAL/CCF/2016_07_01_00_00_00T2016_07_02_00_00_00.h5'
plotting_modules.plot_substack_cc(sfile,0.1,0.2,200,True,'/Users/chengxin/Documents/SCAL/CCF/figures')
```
<img src="https://raw.githubusercontent.com/mdenolle/NoisePy/master/docs/figures/substack_cc_NN.png" width="400" height="190"><img src="https://raw.githubusercontent.com/mdenolle/NoisePy/master/docs/figures/substack_cc_ZZ.png" width="400" height="190">


### 2. Do stacking
This script is used to assemble and/or stack all cross-correlation functions computed for the staion pairs in the `cross_correlate` step and save them into ASDF files for future analysis (e.g., temporal variation and/or dispersion extraction). In particular, there are two options for the stacking process, including linear and phase weighted stacking (pws). See ```noisepy stack --help```

```sh
$ noisepy stack --method linear
$ noisepy stack --method pws
```

In general, the pws produces waveforms with high SNR, and the snapshot below shows the waveform comparison from the two stacking methods. We use the folloing commend lines to make the move-out plot.

```python
from noisepy.seis import plotting_modules
import glob
sfiles = glob.glob('/Users/chengxin/Documents/SCAL/STACK/*/*.h5')
plotting_modules.plot_all_moveout(sfiles,'Allstack_linear'0.1,0.2,'ZZ',1,300,True,'/Users/chengxin/Documents/SCAL/STACK') #(move-out for linear stacking)
plotting_modules.plot_all_moveout(sfiles,'Allstack_pws'0.1,0.2,'ZZ',1,300,True,'/Users/chengxin/Documents/SCAL/STACK')    #(move-out for pws)
```
<img src="https://raw.githubusercontent.com/mdenolle/NoisePy/master/docs/figures/linear_stack1.png" width="400" height="300"><img src="https://raw.githubusercontent.com/mdenolle/NoisePy/master/docs/figures/pws_stack1.png" width="400" height="300">

Anyway, here just presents one simple example of how NoisePy might work! We strongly encourage you to download the NoisePy package and play it on your own! If you have any  comments and/or suggestions during running the codes, please do not hesitate to contact us through email or open an issue in this github page!

Chengxin Jiang (chengxinjiang@gmail.com)
Marine Denolle (mdenolle@uw.edu).

#### Reference
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

# Contributing

After cloning the repo and creating a virtual environment with either **pip** o **conda**:

Do an editable installation to get the dependencies (from the project root):
```sh
$ pip install -e ".[dev]"
```

Install the `pre-commit` hook:
```sh
$ pre-commit install
```

This will run the linting and formatting checks configured in the project before every commit.

## Using VS Code

The following extensions are recommended:

- [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort)
- [black](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
- [flake8](https://marketplace.visualstudio.com/items?itemName=ms-python.flake8)
