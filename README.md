# About NoisePy
NoisePy is a Python package designed for fast and easy ambient noise cross-correlation, with a particular emphasise on ambient noise monitoring application. 

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.come/mdenolle/NoisPy/latest) [![Build Status](https://travis-ci.org/mdenolle/Noise.jl.svg?branch=master)](https://travis-ci.org/mdenolle/NoisePy) [![Coverage Status](https://coveralls.io/repos/github/mdenolle/Noise.jl/badge.svg?branch=master)](https://coveralls.io/github/mdenolle/NoisePy?branch=master)

<img src="/docs/src/logo.png" width="800" height="400">
 
# Installation
This package contains 3 main python scripts with 1 dependent module (`noise_module`) and 1 plotting module ( `plot_modules`). The scripts are depended on some common python libraries as listed below, and we recommend to install them using [conda](https://docs.conda.io/en/latest/) or [pip](https://pypi.org/project/pip/). Due to the availablility of multiple version of dependent libraries, we did not exclusively tested their performance on our package. But the information provided below works well on `macOS Mojave (10.14.5)`. 

|  **library**  |  **version**  |
|:-------------:|:-------------:|
|[numpy](https://numpy.org/)|  1.16.3|
|[scipy](https://www.scipy.org/) | 1.3.0|
|[numba](https://devblogs.nvidia.com/numba-python-cuda-acceleration/) | 0.44.1|
|[obspy](https://github.com/obspy/obspy/wiki) |1.1.1|
|[pandas](https://pandas.pydata.org/) | 0.24.2|
|[pyasdf](http://seismicdata.github.io/pyasdf/) |0.4.0|
|[python](https://www.python.org/) |3.7.3|
|[mpi4py](https://mpi4py.readthedocs.io/en/stable/) | 3.0.1|


# Functionality
* download continous noise data based on obspy's [mass download](https://docs.obspy.org/packages/autogen/obspy.clients.fdsn.mass_downloader.html) module and save data in [ASDF](https://asdf-definition.readthedocs.io/en/latest/) format, which convinently assemble meta, wavefrom and auxililary data into one single file
* perform fast and easy cross-correlation on downloaded seismic data in ASDF format as well as those stored on local machine as SAC/miniSEED format
* options to do/save substacking of the cross-correlation functions 
* coded with MPI functionality to run in parallel
* a series of functions for ambient noise monitoring application

# Short tutorial
**1. Downloading seismic noise data (`S0_download_MPI.py`)**
    
The current setting in the scripts (in `src`) allows the users to download all available broadband CI stations operated at 4/Jul/2016 through SCEC. 

In the scripts, short summary is provided for most of the variables so that it should be straightforward to understand. Specifically, to have a 24-h continous data, we set `inc_hours=24`. `down_list` is set to be `False` since no prior station info is provided, and we rely on the geographic information to find the stations as shown at L64. `flag` should be `True` if intermediate outputs/operational time is needed during downloading process. To run the code on a single core, go to your terminal with a python environment of required libraries and run following command. 

```python
python S0_download_ASDF_MPI.py
```  

If you want to use multiple cores (e.g, 4), run the script with the following command instead (of course with [mpi4py](https://mpi4py.readthedocs.io/en/stable/) installed). 
```python
mpirun -n 4 python S0_download_ASDF_MPI.py
```

<img src="/docs/src/downloaded.png" width="800" height="30">

The snapshot above shows the output files from S0 with 24 hour long continous recordings (more details on reading the ASDF files with downloaded data can be found in docs/src/ASDF.md). You can show the continous waveforms using the plotting functions in the `plot_modules` like this.\

```python
import plot_modules
sfile = '/Users/chengxin/Documents/SCAL/RAW_DATA/2016_07_04_00_00_00T2016_07_05_00_00_00.h5'
plot_modules.plot_waveform(sfile,'CI','USC',0.01,0.4)                                                          
```
<img src="/docs/src/waveform.png" width="800" height="250">

Note that the script also offers the flexibility to download data from a station list (the format is shown in xx). In this case, the variable of `down_list` needs to be `True`.   

**2. Perform cross correlations (`S1_fft_cc_MPI.py`)**\
This is the core script of NoisePy, which performs fft to all noise data first before they are further cross-correlated. With this, it means we are performing [cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation) in frequency domain. In this script, several options are provided to do the cross correlation in different ways, including `raw`, `coherency` and `deconv` (see xx for details). We choose 'decon' as an example here. As you will find after running the script, it creates a new folder called `CCF` along with a new ASDF file of the same name as the downloaded file. To examine the cross-correlation functions of each small segments defined in the script, and show its temporal variations. 

```python
import plot_modules
sfile = '/Users/chengxin/Documents/SCAL/CCF/2016_07_04_00_00_00T2016_07_05_00_00_00.h5'
plot_modules.plot_substack_cc(sfile,0.1,0.2,200,True,'/Users/chengxin/Documents/SCAL/CCF/figures')     
```
<img src="/docs/src/substack_cc.png" width="800" height="250">

**3. Do stacking (`S2_stacking.py`)**\
This script assembles all computed cross-correlation functions from S1, and performs final stacking (including substacking) of them. We provide two options of linear and pws stacking methods for the stacking process. After running the script, we can make the move-out plot using the final stacked cross-correlation.

```python
import plot_modules,glob
sfiles = glob.glob('/Users/chengxin/Documents/SCAL/STACK/*/linear*.h5')
plot_modules.plot_all_moveout1(sfiles,0.1,0.2,'ZZ',1,200,True,'/Users/chengxin/Documents/SCAL/STACK')
```
<img src="/docs/src/linear_stack.png" width="400" height="300"><img src="/docs/src/pws_stack.png" width="400" height="300">

Note that, although we only show the process of one component data, the scripts are able to handle 3-component data. We encourage you to download the NoisePy package and play it on your own! We would really like to know your thoughts, comments and suggestions on it.


