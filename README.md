# About NoisePy
NoisePy is a Python package designed for fast and easy computation of ambient noise cross-correlation functions. It provides additional functionality for noise monitoring and surface wave dispersion analysis. 

Detailed documentation can be found at https://noise-python.readthedocs.io/en/latest/

[![Documentation Status](https://readthedocs.org/projects/noise-python/badge/?version=latest)](https://noise-python.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.com/chengxinjiang/Noise_python.svg?token=jimdAXcrUt4ngV6Dy3s7&branch=master)](https://travis-ci.com/chengxinjiang/Noise_python)
[![Codecov](https://codecov.io/gh/chengxinjiang/Noise_python/branch/master/graph/badge.svg)](https://codecov.io/gh/chengxinjiang/Noise_python)

<img src="/docs/figures/logo.png" width="800" height="400">
 
# Installation
The nature of NoisePy being composed of python scripts allows flexiable package installation, which is essentially to build dependented libraries the scripts and related functions live upon. We recommand to use [conda](https://docs.conda.io/en/latest/) and [pip](https://pypi.org/project/pip/) to install the library due to their convinence. Below are command lines we have tested that would create a python environment to run NoisePy. Note that the test is performed on `macOS Mojave (10.14.5)`, so it could be slightly different for other OS. 

```python
conda create -n noisepy -c conda-forge python=3.7.3 numpy=1.16.2 numba pandas pycwt mpi4py=3.0.1
conda activate noisepy
pip install obspy pyasdf 
```

# Functionality
* download continous noise data based on obspy's core functions of [get_station](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_stations.html) and [get_waveforms](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_waveforms.html)
* save seismic data in [ASDF](https://asdf-definition.readthedocs.io/en/latest/) format, which convinently assembles meta, wavefrom and auxililary data into one single file ([Turtorials](https://github.com/SeismicData/pyasdf/blob/master/doc/tutorial.rst) on reading/writing ASDF files)
* offers high flexibility to handle messy SAC/miniSEED data stored on your local machine and convert them into ASDF format data that could easily be pluged into NoisePy
* performs fast and easy cross-correlation with functionality to run in parallel through [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) 
* includes a series of monitoring functions to measure dv/v on the resulted cross-correlation functions using some recently developed new methods (see our papers for more details<sup>**</sup>) 

# Short tutorial

### 0A. Downloading seismic noise data by using `S0A_download_ASDF_MPI.py`
This script (located in the directory of `src`) and its existing parameters allows to download all available broadband CI stations `(BH?)` located in a certain region and operated during 1/Jul/2016-2/Jul/2016 through the SCEC data center. 

In the script, short summary is provided for all input parameters that can be changed according to the user's needs. In the current form of the script, we set `inc_hours=24` to download day-long continous noise data as well as the meta info and store them into a single ASDF file. To increase the signal-to-noise (SNR) of the final cross-correlation functions (see Seats et al.,2012 for more details), we break the day-long sequence into smaller segments, each of `cc_len` (s) long with some overlapping defined by `step`. You may wanto to set `flag` to be `True` if intermediate outputs/operational time is preferred during the downloading process. To run the code on a single core, open the terminal and activate the noisepy environment before run following command. (NOTE that things may go completely different if you want to run NoisePy on a cluster. Better check it out first!!) 

```python
python S0_download_ASDF_MPI.py
```  

If you want to use multiple cores (e.g, 4), run the script with the following command using [mpi4py](https://mpi4py.readthedocs.io/en/stable/). 
```python
mpirun -n 4 python S0_download_ASDF_MPI.py
```

The outputted files from S0A include ASDF files containing daily-long (24h) continous noise data, a parameter file recording all used parameters in the script of S0A and a CSV file of all station information (more details on reading the ASDF files with downloaded data can be found in docs/src/ASDF.md). The continous waveforms data stored in the ASDF file can be displayed using the plotting modules named as `plotting_modules` in the directory of `src` as shown below.

```python
import plotting_modules #(cd to your source file directory first before loading this module)
sfile = '/Users/chengxin/Documents/SCAL/RAW_DATA/2016_07_01_00_00_00T2016_07_02_00_00_00.h5'
plotting_modules.plot_waveform(sfile,'CI','BLC',0.01,0.4)                                                          
```
<img src="/docs/figures/waveform3.png" width="600" height="400">

Note that the script also offers the option to download data from an existing station list in a format same to the outputed CSV file. In this case, `down_list` should be set to `True` at L53. In reality, the downloading speed is dependent on many factors such as the original sampling rate of targeted data, the networks, the data center where it is hosted and the general structure you want to store on your machine etc. We tested a bunch of the parameters to evaluate their performance and the readers are referred to our paper for more details (Jiang et al., 2019). 


### 0B. DEAL with local SAC/miniseed files using `S0B_sacMSEED_to_ASDF.py`
If you want to use the NoisePy to handel local data in SAC/miniseed format stored on your own disk, this is the script you need. Most of the variables are the same as those for S0A and thus should be pretty straighforward to follow and change. In this script, it preprocesses the data by merging, detrending, demeaning, downsampling and then trimming before saving them into ASDF format for later NoisePy processing. In particular, we expect the script to deal with very messydata, by which we mean that, seismic data is broken into small pieces and of messy time info such as overlapping time. REMEMBER to set `messydata` at L62 to `True` when you have messy data! (Tutorial on removing instrument response)



### 1. Perform cross correlations using `S1_fft_cc_MPI.py`\
This is the core script of NoisePy, which performs [Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform) to all noise data first and loads them into the memory before they are further cross-correlated. This means that we are performing [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation) in the frequency domain. In the script, we provide several options to calculate the cross correlation, including `raw`, `coherency` and `deconv` (see our paper<sup>*</sup> for detailed definition). We choose `coherency` as an example here. After running the script, it will create a new folder named `CCF`, in which new ASDF files containing all cross-correlation functions between different station pairs are located. It also creates a parameter file of `fft_cc_data.txt` that records all useful parameters used in this script. Once you get the cross-correlation file, you can show the daily temporal variation between all station-pair by calling `plot_substack_cc` function in `plotting_modules` as follows. 

```python
import plotting_modules
sfile = '/Users/chengxin/Documents/SCAL/CCF/2016_07_01_00_00_00T2016_07_02_00_00_00.h5'
plot_modules.plot_substack_cc(sfile,0.1,0.2,200,True,'/Users/chengxin/Documents/SCAL/CCF/figures')     
```
<img src="/docs/figures/substack_cc_NN.png" width="400" height="190"><img src="/docs/figures/substack_cc_ZZ.png" width="400" height="190">


### 2. Do stacking with `S2_stacking.py`\
This script is used to assemble and/or stack all cross-correlation functions computed for the staion pairs in S1 and save them into ASDF files for future analysis (e.g., temporal variation and/or dispersion extraction). In particular, there are two options for the stacking process, including linear and phase weighted stacking (pws). In general, the pws produces waveforms with high SNR, and the snapshot below shows the waveform comparison from the two stacking methods. We use the folloing commend lines to make the move-out plot.

```python
import plotting_modules,glob
sfiles = glob.glob('/Users/chengxin/Documents/SCAL/STACK/*/*.h5')
plot_modules.plot_all_moveout(sfiles,'Allstack0linear'0.1,0.2,'ZZ',1,300,True,'/Users/chengxin/Documents/SCAL/STACK') #(move-out for linear stacking)
plot_modules.plot_all_moveout(sfiles,'Allstack0pws'0.1,0.2,'ZZ',1,300,True,'/Users/chengxin/Documents/SCAL/STACK')    #(move-out for pws)
```
<img src="/docs/figures/linear_stack1.png" width="400" height="300"><img src="/docs/figures/pws_stack1.png" width="400" height="300">

Anyway, here just presents one simple example of how NoisePy might work! We strongly encourage you to download the NoisePy package and play it on your own! If you have any  comments and/or suggestions during running the codes, please do not hesitate to contact us through email or open an issue in this github page!  

Chengxin Jiang (chengxin_jiang@fas.harvard.edu)  
Marine Denolle (mdenolle@fas.harvard.edu).

#### Reference
Seats, K. J., Jesse F. L., and German A. P. "Improved ambient noise correlation functions using Welch′ s method." _Geophysical Journal International_ 188, no. 2 (2012): 513-523.  
*Jiang, C., Yuan, C., and Denolle, M. "NoisePy: a new high-performance python tool for seismic ambient noise seismology." In prep for _Seismological Research Letter_.  
**Yuan, C., et al.
