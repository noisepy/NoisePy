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

Update the `jobRoleArn` and `executionRoleArn` fields in the `job_definition.yaml` file with the ARN of the role created in the first step. Add a name for the `jobDefinition` and update the `command` argument as needed (e.g., update the `ccf_path` argument). This command will become the default command but can still be overriden in individual jobs. You can adjust the timeout as appropriate too. Finally, run:

```
aws batch register-job-definition --no-cli-pager --cli-input-yaml file://job_definition.yaml
```

## Submit a job

Update `job.yaml` with a name and the names of your job queue and job definitions created in the last steps. You can then submit a job with:

```
aws batch submit-job --no-cli-pager --cli-input-yaml file://job.yaml --job-name "job_name_override"
```

## Multi-node (array) jobs

See comment above `arrayProperties` in `job.yaml` for instructions on how to process in parallel across multiple nodes.
