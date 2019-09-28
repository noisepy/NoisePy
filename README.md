# About NoisePy
NoisePy is a Python package designed for fast and easy computation of ambient noise cross-correlation functions. It provides additional functionality for noise monitoring and surface wave dispersion analysis. 

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://kura-okubo.github.io/SeisDownload.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://kura-okubo.github.io/SeisDownload.jl/dev)
[![Build Status](https://travis-ci.com/kura-okubo/SeisDownload.jl.svg?branch=master)](https://travis-ci.com/kura-okubo/SeisDownload.jl)
[![Codecov](https://codecov.io/gh/kura-okubo/SeisDownload.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kura-okubo/SeisDownload.jl)

<img src="/docs/src/logo.png" width="800" height="400">
 
# Installation
The nature of NoisePy being composed of python scripts allows flexiable package installation, which is essentially to build dependented libraries the scripts and related functions live upon. We recommand to use [conda](https://docs.conda.io/en/latest/) and [pip](https://pypi.org/project/pip/) to install the library considering their convinence. Below are command lines we have tested that would create a python environment called noisepy where all required libraries to run NoisePy are located. Note that the test is performed on `macOS Mojave (10.14.5)`, so it could be slightly different for other OS. 

```python
conda create -n noisepy -c conda-forge python=3.7.3 numpy=1.16.2 numba pandas pycwt mpi4py=3.0.1
conda activate noisepy
pip install obspy pyasdf 
```

# Functionality
* download continous noise data based on obspy's core functions of [get_station](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_stations.html) and [get_waveforms](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_waveforms.html)
* save seismic data in [ASDF](https://asdf-definition.readthedocs.io/en/latest/) format, which convinently assembles meta, wavefrom and auxililary data into one single file
* offers high flexibility to handle messy SAC/miniSEED data stored on your local machine and convert them into ASDF format data that could easily be pluged into NoisePy
* performs fast and easy cross-correlation with functionality to run in parallel through [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) 
* includes a series of monitoring functions to measure dv/v on the resulted cross-correlation functions using some recently developed new methods (see our papers for more details<sup>**</sup>) 

# Short tutorial
**0A. Downloading seismic noise data by using `S0A_download_ASDF_MPI.py`**\
This script (located in the directory of `src`) and its existing parameters allows to download all available broadband CI stations (BH?) located in a certain region and operated during 1/Jul/2016 - 2/Jul/2016 through the SCEC data center. 

In the script, short summary is provided for all input parameters that could be changed according to the user's need. In the default form of the script as an example, we set `inc_hours=24` to download continous noise data of every 24-h, and store the daily data from all stations into one ASDF file. In order to increase the signal-to-noise (SNR) of the final cross-correlation functions (see Seats et al.,2012 for more details), we further break the 24-h long data into smaller segments of `cc_len` (s) long with some overlapping in time defined by `step`. We set `down_list` to `False` because no prior station info is used, and the script relies on the geographic information at L63 to find targeted station info. `flag` should be set to `True` if intermediate outputs/operational time is needed during the downloading process. To run the code on a single core, open the terminal and activate the noisepy environment before run following command. (NOTE that things may go completely different if you want to run NoisePy on a cluster. Better check it out first!!) 

```python
python S0_download_ASDF_MPI.py
```  

If you want to use multiple cores (e.g, 4), run the script with the following command using [mpi4py](https://mpi4py.readthedocs.io/en/stable/). 
```python
mpirun -n 4 python S0_download_ASDF_MPI.py
```

The outputted files from S0A includes ASDF files containing daily-long (24h) continous noise data for all available stations, a parameter file recording all used parameters in the script of S0A and a CSV file showing the station information (more details on reading the ASDF files with downloaded data can be found in docs/src/ASDF.md). The continous waveforms data stored in the ASDF file can be displayed using the plotting modules named as `plot_modules` in the directory of `src` as shown below.

```python
import plotting_modules (cd to your source file directory first before loading this module)
sfile = '/Users/chengxin/Documents/SCAL/RAW_DATA/2016_07_01_00_00_00T2016_07_02_00_00_00.h5'
plotting_modules.plot_waveform(sfile,'CI','BLC',0.01,0.4)                                                          
```
<img src="/docs/src/waveform3.png" width="800" height="530">

Note that the script also offers the flexibility to download data from an existing station list with a format same to the outputed CSV file. In this case, the variable of `down_list` should be set to `True` at Lxx. We want to NOTE that the downloading speed is dependent on many factors such as the original sampling rate of the data, the networks you are requesting, the data center the data is hosted upon and the general structure to store on your machine et. We tested a bunch of the parameters to evaluate their performance and the readers are referred to our paper for some details. 

**0B. DEAL with SAC or miniseed files stored on your local machine using `S0B_sacMSEED_to_ASDF.py`**\
If you want to use the NoisePy to handel local data in SAC/miniseed format stored on your own disk, this is the script you need. Most of the variables are the same as those for S0A and they should be pretty straighforward to understand and change. In this script, it preprocess the data by merging, detrending, demeaning, downsampling and then trimming before saving them into ASDF format for later NoisePy processing. REMEMBER to set `messydata` at L62 to `True` when you have messy data! From our test, this script could handles seismic data that is broken into small pieces and of messy format such as many overlapping time. (Tutorial on removing instrument response)

**2. Perform cross correlations using `S1_fft_cc_MPI.py`**\
This is the core script of NoisePy, which performs [Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform) to all noise data first and loads them into the memory before they are further cross-correlated. This means that we are performing [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation) in the frequency domain. In the script, we provide several options to calculate the cross correlation, including `raw`, `coherency` and `deconv` (see our paper<sup>*</sup> for detailed definition). We choose `coherency` as an example when running the script, which creates a new folder of `CCF` along with a new ASDF file of the same name as the downloaded file. This new ASFD file in the `CCF` folder contains all cross-correlation functions between all small time segments of `cc_len` long for all station pairs, and you can show their temporal variation by using the following command lines that calls `plot_substack_cc` function in `plot_modules`. 

```python
import plotting_modules
sfile = '/Users/chengxin/Documents/SCAL/CCF/2016_07_01_00_00_00T2016_07_02_00_00_00.h5'
plot_modules.plot_substack_cc(sfile,0.1,0.2,200,True,'/Users/chengxin/Documents/SCAL/CCF/figures')     
```
<img src="/docs/src/substack_cc_NN.png" width="400" height="190"><img src="/docs/src/substack_cc_ZZ.png" width="400" height="190">

**3. Do stacking with `S2_stacking.py`**\
This script is used to assemble and/or stack all cross-correlation functions computed for the staion pairs in S1 and save them into ASDF files for future analysis (e.g., temporal variation and/or dispersion extraction). In particular, there are two options for the stacking process, including linear and phase weighted stacking (pws). In general, the pws produces waveforms with high SNR, and the snapshot below shows the waveform comparison from the two stacking methods. We use the folloing commend lines to make the move-out plot.

```python
import plotting_modules,glob
sfiles = glob.glob('/Users/chengxin/Documents/SCAL/STACK/*/*.h5')
plot_modules.plot_all_moveout(sfiles,'Allstack0linear'0.1,0.2,'ZZ',1,300,True,'/Users/chengxin/Documents/SCAL/STACK')
```
<img src="/docs/src/linear_stack1.png" width="400" height="300"><img src="/docs/src/pws_stack1.png" width="400" height="300">

Anyway, here just presents one simple example of how NoisePy might work! We strongly encourage you to download the NoisePy package and play it on your own! If you have any  comments and/or suggestions during running the codes, please do not hesitate to contact us through email or open an issue in this github page!  

Chengxin Jiang (chengxin_jiang@fas.harvard.edu)  
Marine Denolle (mdenolle@fas.harvard.edu).

**Reference**\
Seats, K. J., Jesse F. L., and German A. P. "Improved ambient noise correlation functions using Welch′ s method." _Geophysical Journal International_ 188, no. 2 (2012): 513-523.  
*Jiang, C., Yuan, C., and Denolle, M. "NoisePy: a new high-performance python tool for seismic ambient noise seismology." In prep for _Seismological Research Letter_.  
**Yuan, C., et al.