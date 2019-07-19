# About NoisePy
NoisePy is a Python package designed for fast and easy ambient noise cross-correlations. In particular, this package provides additional modules to use the computed cross-correlation functions for monitoring purpose. 

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://github.come/mdenolle/NoisPy/latest) [![Build Status](https://travis-ci.org/mdenolle/Noise.jl.svg?branch=master)](https://travis-ci.org/mdenolle/NoisePy) [![Coverage Status](https://coveralls.io/repos/github/mdenolle/Noise.jl/badge.svg?branch=master)](https://coveralls.io/github/mdenolle/NoisePy?branch=master)

 
# Installation
To install this package, simply go to src directory and run install.py, which will check all dependencies required by this package. You can also find all dependencies in the file of dependency.lst 

# Functionality
This package contains 3 main scripts:
1. S0_dowload_ASDF_MPI.py
2. S1_fft_cc_ASDF_MPI.py
3. S2_stacking.py
with 1 module named as noise_module.py. 

As indicated by the script names, this package includes functionality to:
* download continous noise data of user-defined length from available data center and save them in ASDF format
* perform fast and easy cross-correlation (multiple components) for both data downloaded using this package and those stored on local machine as SAC or miniSEED format
* do stacking (sub-stacking) of the cross-correlation functions for noise application
Each script is coded with MPI function so computation efficiency can be improved if needed

# Examples
Some example data/scripts are located in directory of /example_data.
