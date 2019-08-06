# About NoisePy
NoisePy is a Python package designed for fast and easy computation of ambient noise cross-correlation functions, with a particular emphasise on noise monitoring application. 

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.come/mdenolle/NoisPy/latest) [![Build Status](https://travis-ci.org/mdenolle/Noise.jl.svg?branch=master)](https://travis-ci.org/mdenolle/NoisePy) [![Coverage Status](https://coveralls.io/repos/github/mdenolle/Noise.jl/badge.svg?branch=master)](https://coveralls.io/github/mdenolle/NoisePy?branch=master)

<img src="/docs/src/logo.png" width="800" height="400">
 
# Installation
This package contains 4 main python scripts with 2 dependent modules (`core_functions` and `noise_module`) and 1 plotting module ( `plot_modules`). We prefer the script style over the function style mainly because we want to implement the MPI into the package. The scripts are depended on some common python libraries with a detailed list shown below. We recommend installing them using [conda](https://docs.conda.io/en/latest/) or [pip](https://pypi.org/project/pip/). Due to the availablility of multiple version of dependent libraries, we did not exclusively tested their performance on our package. But the information provided below works well on `macOS Mojave (10.14.5)`. 

|  **library**  |  **version**  |
|:-------------:|:-------------:|
|[numpy](https://numpy.org/)|  >= 1.16.3|
|[scipy](https://www.scipy.org/) | >= 1.3.0|
|[numba](https://devblogs.nvidia.com/numba-python-cuda-acceleration/) | >= 0.44.1|
|[obspy](https://github.com/obspy/obspy/wiki) | >= 1.1.1|
|[pandas](https://pandas.pydata.org/) | >= 0.24.2|
|[pyasdf](http://seismicdata.github.io/pyasdf/) | >= 0.4.0|
|[python](https://www.python.org/) | >= 3.7.3|
|[mpi4py](https://mpi4py.readthedocs.io/en/stable/) | >= 3.0.1|


# Functionality
* download continous noise data based on obspy's core functions of [get_station](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_stations.html) and [get_waveforms](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.client.Client.get_waveforms.html) and save data in [ASDF](https://asdf-definition.readthedocs.io/en/latest/) format, which convinently assemble meta, wavefrom and auxililary data into one single file
* offers great flexibility to handle messy SAC/miniSEED data stored on local machine and options to convert them into ASDF format
* perform fast and easy cross-correlation 
* options to do and save substacking of the cross-correlation functions 
* coded with [MPI](https://en.wikipedia.org/wiki/Message_Passing_Interface) functionality to run in parallel
* a series of monitoring functions including some recently developed methods (see our papers for more details<sup>**</sup>) for ambient noise monitoring applications

# Short tutorial
**1A. Downloading seismic noise data (`S0A_download_ASDF_MPI.py`)**\
The current settings of the script (as located in `src`) allows the users to download all available broadband CI stations operated at 4/Jul/2016 through SCEC data center. 

In the script, short summary is provided for the input parameters so they should be straightforward to understand. For example, we set `inc_hours=24` to download continous noise data of every 24-h long, and store the daily data from all stations into one ASDF file. In order to increase the signal-to-noise (SNR) of the final cross-correlation functions (see Seats et al.,2012 for more details), we further break the 24-h long data into smaller segments of `cc_len` long with some overlapping defined by `step`. `down_list` is set to `False` because no prior station info is used, and the script relies on the geographic information at L63 to find targeted station info. `flag` should be `True` if intermediate outputs/operational time is needed during the downloading process. To run the code on a single core, go to your terminal with a python environment of required libraries and run following command. (Things may go completely different if you want to run NoisePy on a cluster. Better check it out first!!) 

```python
python S0_download_ASDF_MPI.py
```  

If you want to use multiple cores (e.g, 4), run the script with the following command instead (with [mpi4py](https://mpi4py.readthedocs.io/en/stable/) installed of course). 
```python
mpirun -n 4 python S0_download_ASDF_MPI.py
```

<img src="/docs/src/downloaded.png" width="800" height="40">

The snapshot above shows the output files from S0, including a ASDF file containing the 24-h continous noise data for all stations, a parameter file of all used parameters in the script and a CSV file containing station information (more details on reading the ASDF files with downloaded data can be found in docs/src/ASDF.md). The continous waveforms data stored in the ASDF file can be displayed using the plotting functions in the `plot_modules` as shown below.

```python
import plot_modules
sfile = '/Users/chengxin/Documents/SCAL/RAW_DATA/2016_07_04_00_00_00T2016_07_05_00_00_00.h5'
plot_modules.plot_waveform(sfile,'CI','USC',0.01,0.4)                                                          
```
<img src="/docs/src/waveform.png" width="800" height="250">

Note that the script also offers the flexibility to download data from an existing station list with a format same to the outputed CSV file. In this case, the variable of `down_list` should be set to `True` in the script. 

**1B. DEAL SAC or miniseed files on your local disk (`S0A_download_MPI.py`)**\
If you just want to deal with the local data stored as SAC/miniseed format on your own disk, you should use this script. Most of the variables are pretty straighforward to understand. What this script essentially do is to prepare the SAC/miniseed files for the NoisePy package. In this script, it preprocess the data to cut, trim and downsample it into your targeted range. (choose when you have messy data! Trust me, this script is a better choice compared to the `S1_fft_cc_MPI.py`). Based on our test, this script could handles seismic data that is broken into small pieces and have messy format such as overlapping time for the broken pieces etc.   

**2. Perform cross correlations (`S1_fft_cc_MPI.py`)**\
This is the core script of NoisePy, which performs [Fourier transform](https://en.wikipedia.org/wiki/Fourier_transform) to all noise data first before they are further cross-correlated. This means that we are performing [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation) in the frequency domain. In the script, several options are provided to calculate the cross correlation in different ways, including `raw`, `coherency` and `deconv` (see our paper<sup>*</sup> for detailed definition). We choose `decon` as an example when running the script, and it will create a new folder called `CCF` along with a new ASDF file of the same name as the downloaded file. The new ASFD file in the `CCF` folder contains cross-correlation functions between all small time segments of `cc_len` long for all station pairs, and you can show their temporal variation by using the following command lines that calls `plot_substack_cc` function in `plot_modules`. 

```python
import plot_modules
sfile = '/Users/chengxin/Documents/SCAL/CCF/2016_07_04_00_00_00T2016_07_05_00_00_00.h5'
plot_modules.plot_substack_cc(sfile,0.1,0.2,200,True,'/Users/chengxin/Documents/SCAL/CCF/figures')     
```
<img src="/docs/src/substack_cc.png" width="800" height="250">

**3. Do stacking (`S2_stacking.py`)**\
This script is designed to assemble all computed cross-correlation functions of one staion-pair (of different time) from S1 into one single file, and keeps final stacking (including substacking) of them for future application purpose (e.g., temporal variation and dispersion extraction). In particular, we provide two options of linear and phase weighted stacking (pws) methods for the stacking process, and the figure below shows the waveform comparison resulted from the two methods. To make the move-out plot shown blow, we use the folloing commend lines.

```python
import plot_modules,glob
sfiles = glob.glob('/Users/chengxin/Documents/SCAL/STACK/*/linear*.h5')
plot_modules.plot_all_moveout1(sfiles,0.1,0.2,'ZZ',1,200,True,'/Users/chengxin/Documents/SCAL/STACK')
```
<img src="/docs/src/linear_stack.png" width="400" height="300"><img src="/docs/src/pws_stack.png" width="400" height="300">

Note that, although here we only show the process of downloading/dealing of one component data, the scripts are able to handle 3-component data. We encourage you to download the NoisePy package and play it on your own! If you have any thoughts, comments and suggestions, please do not hesitate to contact  
Chengxin Jiang (chengxin_jiang@fas.harvard.edu)  
Marine Denolle (mdenolle@fas.harvard.edu).

**Reference**\
Seats, K. J., Jesse F. L., and German A. P. "Improved ambient noise correlation functions using Welch′ s method." _Geophysical Journal International_ 188, no. 2 (2012): 513-523.  
*Jiang, C., Toghramadjian, N., Yuan, C., Clements, T., and Denolle, M. "NoisePy: a new high-performance python tool for seismic ambient noise seismology." In prep for _Seismological Research Letter_.  
**Yuan, C., et al.