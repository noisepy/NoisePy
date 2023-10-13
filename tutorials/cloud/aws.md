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
git clone https://github.com/noisepy/NoisePy
cd NoisePy
pip install ipykernel jupyter
pip install noisepy-seis
```

You may save your environment using AWS AMI. Then subsequent launcing of instances can re-use your environment.

Using docker
```
sudo yum install -y git docker
sudo systemctl start docker
sudo docker pull ghcr.io/noisepy/noisepy:latest
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

# Running on AWS Batch

The below steps require [setting up the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html) as well as the [jq tool](https://jqlang.github.io/jq/download/) (optional).


## Create role

AWS batch requires an IAM role to be created for running the jobs. This can be done from the IAM console.

Create a role using the following options:

- Trusted Entity Type: AWS Service
- Use Case: Elastic Container Service
    - Elastic Container Service Task

On the next page, search for and add:
- AmazonECSTaskExecutionRolePolicy
- AmazonS3FullAccess

Once the role is created, one more permission is needed:
- Go to: Permissions tab --> Add Permissions --> Create inline policy
- Search for "batch"
- Click on **Batch**
- Select Read / Describe Jobs
- Click Next
- Add a poolicy name, e.g. "Describe_Batch_Jobs"
- Click Create Policy

Finally, go to the S3 bucket where you'll be writing the results of the jobs.
Open the Permissions tab and add a statement to the bucket policy granting full access to the role you
just created:

```
		{
			"Sid": "Statement3",
			"Principal": {
			    "AWS": "arn:...your job role ARN."
			},
			"Effect": "Allow",
			"Action": "s3:*",
			"Resource": "arn:...your bucket ARN."
		}
```

Note that the job role ARN will be in the format of `arn:aws:iam::<YOUR_ACCOUNT_ID>:role/<JOB_ROLE_NAME>`. The bucket ARN will be in the format of `arn:aws:s3:::<YOUR_S3_BUCKET>`.


## Create a Compute Environment

You'll need two pieces of information to create the compute environment. The list of subnets in your VPC and the default security group ID. You can use the following commands to retrieve them:

```
aws ec2 describe-subnets  | jq ".Subnets[] | .SubnetId"
```
```
aws ec2 describe-security-groups --filters "Name=group-name,Values=default" | jq ".SecurityGroups[0].GroupId"
```

Use this values to update the missing fields in `compute_environment.yaml` and the run:

```
aws batch create-compute-environment --no-cli-pager --cli-input-yaml file://compute_environment.yaml
```

Make a note of the compute environment ARN to use in the next step.

## Create a Job queue

Add the compute environment and a name to `job_queue.yaml` and then run:

```
aws batch create-job-queue --no-cli-pager --cli-input-yaml file://job_queue.yaml
```

## Create a Job Definition

Update the `jobRoleArn` and `executionRoleArn` fields in the `job_definition.yaml` file with the ARN of the role created in the first step. Add a name for the `jobDefinition`. Finally, run:

```
aws batch register-job-definition --no-cli-pager --cli-input-yaml file://job_definition.yaml
```

## Submit a Cross-Correlation job

Update `job_cc.yaml` with the names of your `jobQueue` and `jobDefinition` created in the last steps. Then update the S3 bucket paths
to the locations you want to use for the output and your `config.yaml` file.

```
aws batch submit-job --no-cli-pager --cli-input-yaml file://job_cc.yaml --job-name "<your job name>"
```

## Submit a Stacking job

Update `job_stack.yaml` with the names of your `jobQueue` and `jobDefinition` created in the last steps. Then update the S3 bucket paths
to the locations you want to use for your input CCFs (e.g. the output of the previous CC run), and the stack output. By default, NoisePy will look for a config
file in the `--ccf_path` location to use the same configuration for stacking that was used for cross-correlation.

```
aws batch submit-job --no-cli-pager --cli-input-yaml file://job_stack.yaml --job-name "<your job name>"
```

## Multi-node (array) jobs

See comment above `arrayProperties` in `job_cc.yaml` and `job_stack.yaml` for instructions on how to process in parallel across multiple nodes.
