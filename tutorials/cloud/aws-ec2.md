# Running NoisePy with AWS EC2 Service

## Pre-requisites
See our [checklist](./checklist.md)

## Setup the Virtual Machine
### Create an EC2 Instance
- Log into your AWS account and go into the EC2 Dashboard
- Click on Launch Instance
- Application and OS images:
    - Select the AWS Linux
- Instance type:
    - t2.micro (free) or bigger machines (RAM recommended TBD)
- Key pair for SSH:
    - Create a new Key pair (RSA)
- Network settings:
    - You can use most defaults but we recomment `Allow SSH traffic from` to `My IP`
    - In order to access Jupyter notebook on the instance, click `Allow HTTPS traffic from the internet`.
- Advanced details:
    - If applicable, select `IAM instance profile` to the appropriate role for EC2 service. See [IAM Role](./checklist.md#iam-role-and-permission) for reference.

More information about getting on the cloud in the [SCOPED HPS Book](https://seisscoped.org/HPS/softhardware/AWS_101.html).

### SSH Into Your Instance
Make your private key file only readable by you (assuming it's named/downloaded to `~/Downloads/ec2.pem`). Go to your instance's summary page and copy the `Public IPv4 DNS` in the format of `XXXXX.us-west-2.compute.amazonaws.com`.
```
cd ~/Downloads
chmod 400 ec2.pem
ssh -i ec2.pem ec2-user@<Public IPv4 DNS>
```

### Install NoisePy
This tutorial focuses on small, toy problems to be ran on notebooks or simple CLI. We include jupyter notebook instructions to explore the data and results. Options are available to install NoisePy for different purposes.

You may save your environment using AWS AMI. Then subsequent launcing of instances can re-use your environment.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
./miniconda3/bin/conda init bash
bash
```

#### Through Pip
```bash
conda create -y -n noisepy python==3.10
conda activate noisepy
pip install ipykernel jupyter noisepy-seis
```

#### Through Git
Download the entire development version of NoisePy repository from GitHub. The directory includes source codes and all tutorials.
```bash
sudo yum install -y git
git clone https://github.com/noisepy/NoisePy
cd NoisePy
pip install .
```

#### Through Docker
```bash
sudo yum install -y git docker
sudo systemctl start docker
sudo docker pull ghcr.io/noisepy/noisepy:latest
```

```bash
sudo docker run -v ~/tmp:/tmp cross_correlate --path /tmp
```

## Running Cross-correlation as a Toy Problem
Below we use stations from SCEDC public data archive on the AWS S3 to run a tiny cross-correlation workflow using NoisePy. The continuous waveform is publicized at `s3://scedc-pds/continuous_waveforms/` with associated StationXML file at `s3://scedc-pds/FDSNstationXML/`.

### Exploration Using Jupyter Notebooks
We recommend starting off with a notebook to explore simple jobs and the desirable configuration (e.g., noise pre-processing). Refer to the [SCOPED HPS Book](https://seisscoped.org/HPS/softhardware/AWS_101.html) to open a Jupyter notebook.

### Exploration Using CLI
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

## Plotting Results
See chapter TBD to read and plot results.
