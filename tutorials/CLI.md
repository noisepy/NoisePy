# NoisePy CLI Tutorial

NoisePy provides a CLI (command line interface) for running processing jobs. This tutorial works through the same tasks as the **Noisepy Colab Tutorial** from the [get_started.ipynb](get_started.ipynb) Jupyter Notebook.

NOTE: The `tutorials/cli/config.yml` file contains the parameters being used throughout the tutorial. All commands are meant to be run from the `tutorials/cli` directory.

## Step 0: download data


This step will download data using obspy and save them into ASDF files locally. The data will be stored for each time chunk defined in hours by inc_hours.

The download will clean up the raw data by detrending, removing the mean, bandpassing (broadly), removing the instrumental response, merging gaps, ignoring too-gappy data.


```sh
mkdir tmpdata
noisepy download --config config.yml --raw_data_path ./tmpdata/RAW_DATA
```

Verify the data is there:
```sh
ls -la ./tmpdata/RAW_DATA
```

Notice that this directory contains a `config.yml` file. This file will contain any parameters used during the `download` step. It could be different from the original `config.yml` since all the paramters can be overriden through CLI arguments. See `noisepy download --help.`

## Step 1: Cross-correlation

This step will perform the cross correlation. For each time chunk, it will read the data, perform classic ambient noise pre-processing (time and frequency normalization), FFT, cross correlation, substacking, saving cross correlations in to a temp ASDF file (this is not fast and will be improved).

```sh
noisepy cross_correlate --raw_data_path ./tmpdata/RAW_DATA --ccf_path ./tmpdata/CCF
```
Optionally, this step can be run via MPI (e.g. with 2 processes). See [Installation](https://github.com/noisepy/NoisePy/#installation):

```sh
mpiexec -n 2 noisepy cross_correlate --mpi --raw_data_path ./tmpdata/RAW_DATA --ccf_path ./tmpdata/CCF
```
NOTE: We didn't pass a `--config` argument explicitly because `noisepy` will always look for one in the input data directory for the given step, `./tmpdata/RAW_DATA` in this case.

Once again, verify the data for this step:
```sh
ls -la ./tmpdata/CCF
```

## Step 2: Stack the cross correlation

This combines the time-chunked ASDF files to stack over each time chunk and at each station pair.

```sh
noisepy stack --ccf_path ./tmpdata/CCF --stack_path ./tmpdata/STACK
```
Optionally, this step can be run via MPI (e.g. with 3 processes). See [Installation](https://github.com/noisepy/NoisePy/#installation):
```sh
mpiexec -n 3 noisepy stack --mpi --ccf_path ./tmpdata/CCF --stack_path ./tmpdata/STACK
```


```sh
ls -R tmpdata/STACK
```
