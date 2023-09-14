# Running NoisePy on AWS


## AWS accounts and roles.
Make sure you have an account on AWS. AWS requires particular credentials to connect.

Noisepy uses S3/Cloudstore to store the cross correlations and stacked data. For this step, it is important that your **user/role** and the **bucket** have the appropriate permissions for users to read/write into the bucket.

In the browser, please add the following policy to the bucket:
```
{
    "Version": "2012-10-17",
    "Id": "Policy1674832359797",
    "Statement": [
        {
            "Sid": "Stmt1674832357905",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::<YOUR-ACCOUNT-ID>:role/<YOURROLE>"
            },
            "Action": "s3:*",
            "Resource": "arn:aws:s3:::<BUCKET-NAME>/*"
        }
    ]
}
```
In order to check whether the user can read/write in the bucket, we recommend testing from local:
```
aws s3 ls s3://<BUCKET-NAME>
```
Add a temporary file to make sure you have the credentials to add to the bucket
```
aws s3 cp temp s3://<BUCKET-NAME>
```

If this step works, and if your role and user account are attached to the bucket policy, the rest of the AWS noisepy tutorial should work.

## Create an EC2 instance

- Log into your AWS account and go into the EC2 Dashboard
- Click on Launch Instance
- Select the AWS Linux OS Image
- Create a new Key pair (RSA)
- You can use most defaults but we recomment changing `Network Settings --> Allow SSH traffic from` to `My IP`

More information about getting on the cloud in the [SCOPED HPS Book](https://seisscoped.org/HPS/softhardware/AWS_101.html).

## SSH into your instance

Make your private key file only readable by you (assuming it's named/downloaded to `~/Downloads/ec2.pem`). Go to your instance's summary page and copy the `Public IPv4 DNS`

```
cd ~/Downloads
chmod 400 ec2.pem
ssh -i ec2.pem ec2-user@<public dns name>
```

## Setup NoisePy

This tutorial focuses on small, toy problems to be ran on notebooks or simple CLI. We include jupyter notebook instructions to explore the data and results.



Inside your EC2 instance SSH session, run:

```
sudo yum install git
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
./miniconda/bin/conda init bash
bash
source .bashrc
conda create -n noisepy python==3.10
conda activate noisepy
git clone https://github.com/mdenolle/NoisePy
cd NoisePy
pip install ipykernel jupyter
pip install noisepy-seis
```

You may save your environment using AWS AMI. Then subsequent launcing of instances can re-use your environment.

Using docker
```
sudo yum install -y git docker
sudo systemctl start docker
sudo docker pull ghcr.io/mdenolle/noisepy:latest
```

```
sudo docker run -v ~/tmp:/tmp cross_correlate --path /tmp
```

## Cross-correlation - Toy Problem

### Exploration using notebooks
We recommend starting off with a notebook to explore simple jobs and the desirable configuration (e.g., noise pre-processing).  Refer to the [SCOPED HPS Book](https://seisscoped.org/HPS/softhardware/AWS_101.html) to open a Jupyter notebook.


### Exploration using CLI
You may edit the ``config.yml`` file with appropriate parameters. Refer to the NoiseConfig page to set up.


```
noisepy cross_correlate --format zarr --raw_data_path s3://scedc-pds/continuous_waveforms/ \
--xml_path s3://scedc-pds/FDSNstationXML/CI/ \
--ccf_path s3://<YOUR_S3_BUCKET>/<CC_PATH> \
--stations=SBC,RIO,DEV \
--start=2022-02-02 \
--end=2022-02-03
```

## Cross-correlation - Batch Deployment

This is a toy problem that uses a small set of stations, reads from a cloud store, and output data in zarr.

In terminal, type the following command.



## Stacking

This toy problem gathers the all the ross correlations calculated and stack them into a Zarr format on an S3 bucket.

```
noisepy stack \
--format zarr \
--ccf_path s3://<YOUR_S3_BUCKET>/<CC_PATH> \
--stack_path s3://<YOUR_S3_BUCKET>/<STACK_PATH> \
```


## Data and QC exploration

We recommend using a notebook to plot and explore the data.
TBD.
