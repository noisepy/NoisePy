# Running NoisePy with AWS

## EC2 and Jupyter Lab
Please refer to [SCOPED HPS Book](https://seisscoped.org/HPS-book/chapters/cloud/AWS_101.html) for full detailed instruction on launching an AWS EC2 instance and/or running the notebooks within a containerized environment.

## Submit Batch Job
For large job load, please refer to the [notebook tutorial](./noisepy_aws_batch.ipynb) for more instruction.

## Command Line Interface
You may create or edit the [config.yml](../config.yml) file with appropriate parameters. The cross-correlation function is written to the `ccf_path`.

```bash
noisepy cross_correlate --format numpy --raw_data_path s3://scedc-pds/continuous_waveforms/ \
--xml_path s3://scedc-pds/FDSNstationXML/CI/ \
--ccf_path s3://<S3_BUCKET>/<CC_PATH> \
--stations=SBC,RIO,DEV \
--start=2022-02-02 \
--end=2022-02-03
```

This toy problem gathers the all the cross-correlations calculated and stack them into the NumPy format on the S3 bucket, specificed by the `stack_path`.

```bash
noisepy stack \
--format numpy \
--ccf_path s3://<S3_BUCKET>/<CC_PATH> \
--stack_path s3://<S3_BUCKET>/<STACK_PATH> \
```

