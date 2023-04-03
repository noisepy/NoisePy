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


## Setup Docker

Install docker
```
sudo yum update
sudo yum install docker
```

Setup group membership
```
sudo usermod -a -G docker ec2-user
id ec2-user
newgrp docker
```

Start docker service
```
sudo systemctl enable docker.service
sudo systemctl start docker.service
```

# Run NoisePy

Download data, e.g.:
```
 docker run -v ~/tmp:/tmp ghcr.io/mdenolle/noisepy download --start 2019_02_01_00_00_00 --end 2019_02_01_01_00_00 --stations ARV,BAK --inc_hours 1 --path /tmp
```
Run cross correlation:
```
 docker run -v ~/tmp:/tmp ghcr.io/mdenolle/noisepy cross_correlate --path /tmp
```

Stack:
```
 docker run -v ~/tmp:/tmp ghcr.io/mdenolle/noisepy stack --method linear --path /tmp
```
