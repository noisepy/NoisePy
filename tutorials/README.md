# NoisePy Documentation

**NoisePy** is a software to compute large-scale cross correlations for HPC and Cloud infrastructure. The difference in using Noisepy for either infrastructure is the back-end data format that are either file-system (H5) or object-storage (npz/mseed) optimzed.

**NoisePy** also offers tools for ambient noise monitoring (velocity and attenuation) and for Earth imaging (measuring phase and group velocities).

NoisePy leverages several efforts published, please consider

* Jiang, C., Denolle, M. 2020. NoisePy: a new high-performance python tool for ambient noise seismology. Seismological Research Letters. 91, 1853-1866. https://doi.10.1785/0220190364.
* Yuan C, Bryan J, Denolle M. Numerical comparison of time-, frequency-and wavelet-domain methods for coda wave interferometry. Geophysical Journal International. 2021 Aug;226(2):828-46.  https://doi.org/10.1093/gji/ggab140
* Yang X, Bryan J, Okubo K, Jiang C, Clements T, Denolle MA. Optimal stacking of noise cross-correlation functions. Geophysical Journal International. 2023 Mar;232(3):1600-18. https://doi.org/10.1093/gji/ggac410

We gratefully acknowledge support from the [Packard Foundation](https://www.packard.org)


## NoisePy Workflow

The data processing workflow in NoisePy consists of three steps:

1. **(Optional) Step 0 - Download**: The `download()` function or the `noisepy download` CLI command can be used to download data from an FDSN web service. Alternatively, data from an [S3 bucket](https://s3.console.aws.amazon.com/s3/buckets/scedc-pds) can be copied locally using the `aws` CLI, or streamed directly from S3. For users who want to work entirely locally, this step prepares and organize the data in a ``DataStore``.
2. **Step 1 - Cross Correlation**: Computes cross correlaton for pairs of stations/channels. This can done with either the `cross_correlate()` function or the `noisepy cross_correlate` CLI command.
3. **Step 2 - Stacking**: This steps takes the cross correlation computations across multiple timespans and stacks them for a given station/channel pair. This can done with either the `stack_cross_correlations()` function or the `noisepy stack` CLI command.

<img src="https://github.com/noisepy/NoisePy/blob/main/docs_old/figures/data_flow.png?raw=true">

### Data Storage

NoisePy accesses data through three "DataStore" abstract classes: `DataStore`, `CrossCorrelationDataStore` and `StackStore`. Concrete implementations are provided for ASDF (H5), miniSEED, Zarr, TileDB, npy formats. This part is implemented in a separate IO package [noisepy-io](https://github.com/noisepy/noisepy-io).

0. [optional] data download: for users who want to work entirely locally. This step prepares and organize the data in a ``RawDataStore``.
1. Cross correlations: data may be streamed from the DataStore, which can be hosted on the Cloud, pre-processing and cross correlations are done for each time chunk (e.g., one day for broadband data). Cross-correlations are saved for each time chunck in ``CrossCorrelationDataStore``.
2. Stacking: Data is aggregated and stacked over all time periods. Stacked data will be stored in ``StackStore``.

## Applications
### Monitoring
NoisePy includes various functions to measure dv/v. The software will read the ``CrossCorrelationDataStore`` to aggregate and measure dv/v. The outputs are tabular data in CSV.

### Imaging
NoisePy includes functions to measure phase and group velocity dispersion curve measurements. The software will read the ``StackStore`` and ouput curves as tabular data in CSV.
