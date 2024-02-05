# Running NoisePy with AWS Batch Service (Advanced)

## Pre-requisites
* You are not required to run this on a AWS EC2 instance, but you would need [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html) as well as the [jq tool](https://jqlang.github.io/jq/download/) installed.

* AWS Batch requires a special IAM role to be created for running the jobs. This can be done from the IAM console. See [instructions](./checklist.md#iam-role-and-permission) to create the role. 

* Be sure to go to the S3 bucket where you'll be writing the results of the jobs and [modify the permissions](./checklist.md#s3-object-storage-and-policy) accordingly.

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

### Create a Job queue
Add the compute environment and a name to `job_queue.yaml` and then run:

```
aws batch create-job-queue --no-cli-pager --cli-input-yaml file://job_queue.yaml
```

### Create a Job Definition
Update the `jobRoleArn` and `executionRoleArn` fields in the `job_definition.yaml` file with the ARN of the role created in the first step. Add a name for the `jobDefinition`. Finally, run:

```
aws batch register-job-definition --no-cli-pager --cli-input-yaml file://job_definition.yaml
```

### Submit a Cross-Correlation job
Update `job_cc.yaml` with the names of your `jobQueue` and `jobDefinition` created in the last steps. Then update the S3 bucket paths
to the locations you want to use for the output and your `config.yaml` file.

```
aws batch submit-job --no-cli-pager --cli-input-yaml file://job_cc.yaml --job-name "<your job name>"
```

### Submit a Stacking job
Update `job_stack.yaml` with the names of your `jobQueue` and `jobDefinition` created in the last steps. Then update the S3 bucket paths
to the locations you want to use for your input CCFs (e.g. the output of the previous CC run), and the stack output. By default, NoisePy will look for a config
file in the `--ccf_path` location to use the same configuration for stacking that was used for cross-correlation.

```
aws batch submit-job --no-cli-pager --cli-input-yaml file://job_stack.yaml --job-name "<your job name>"
```

### Multi-node (array) jobs
See comment above `arrayProperties` in `job_cc.yaml` and `job_stack.yaml` for instructions on how to process in parallel across multiple nodes.

## Plotting Results
See chapter TBD to read and plot results.