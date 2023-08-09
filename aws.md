
Viewed
@@ -0,0 +1,57 @@
# Running NoisePy on EC2

## Create an EC2 instance

- Log into your AWS account and go into the EC2 Dashboard
- Click on Launch Instance
- Select the AWS Linux OS Image
- Create a new Key pair (RSA)
- You can use most defaults but we recomment changing `Network Settings --> Allow SSH traffic from` to `My IP`

## SSH into your instance

Make your private key file only readable by you (assuming it's named/downloaded to `~/Downloads/ec2.pem`). Go to your instance's summary page and copy the `Public IPv4 DNS`

```
cd ~/Downloads
chmod 400 ec2.pem
ssh -i ec2.pem ec2-user@<public dns name>
```


## Setup NoisePy

Inside your EC2 instance SSH session, run:

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source .bashrc
conda create -n noisepy python==3.10
conda activate noisepy
pip install noisepy-seis
```

## Cross-correlation

```
noisepy cross_correlate --format zarr --raw_data_path s3://scedc-pds/continuous_waveforms/ \
--xml_path s3://scedc-pds/FDSNstationXML/CI/ \
--ccf_path s3://<YOUR_S3_BUCKET>/<CC_PATH> \
--stations=SBC,RIO,DEV \
--start=2022-02-02 \
--end=2022-02-03
```

## Stacking

```
noisepy stack \
--format zarr \
--ccf_path s3://<YOUR_S3_BUCKET>/<CC_PATH> \
--stack_path s3://<YOUR_S3_BUCKET>/<STACK_PATH> \
```
