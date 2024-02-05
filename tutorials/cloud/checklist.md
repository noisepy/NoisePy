# General Checklist Running NoisePy on the AWS
This will be a frequently referred chapter for running any (AWS) Cloud-native codes. Check through each of the items below and make sure you have them configured right.

## AWS Account
* `<ACCOUNT-ID>`: A 12-digit number uniquely identify your account.

Make sure you have an account on AWS idenfitied by a 12-digit number. AWS requires particular credentials to connect.

## IAM Role and Permission
* `<ROLE>`: A virtual identity that has specific permissions. The role ARN is in the format of `arn:aws:iam::<ACCOUNT-ID>:role/<ROLE>`.

AWS batch requires an IAM role to be created for running the jobs. This can be done from the IAM console on the AWS web console. Depending on the type of service to use, separate roles may be created. 

* **EC2 service** generally uses the following configuration:
    - Trusted Entity Type: AWS Service
    - Use Case: EC2
    - Permission Policies, search and add:
        - AmazonEC2FullAccess
        - AmazonS3FullAccess

* **Batch service** generally uses the following configuration:
    - Trusted Entity Type: AWS Service
    - Use Case: Elastic Container Service
        - Elastic Container Service Task
    - Permission Policies, search and add:
        - AmazonECSTaskExecutionRolePolicy
        - AmazonS3FullAccess

    Once the role is created, one more permission is needed:
    - Go to: Permissions tab --> Add Permissions --> Create inline policy
    - Search for "batch"
    - Click on **Batch**
    - Select Read / Describe Jobs
    - Click Next
    - Add a policy name, e.g. "Describe_Batch_Jobs"
    - Click Create Policy

## S3 Object Storage and Policy
* `<S3_BUCKET>`: A dedicated container on S3 with specific permissions.

NoisePy uses S3 Cloudstore to store the cross correlations and stacked data. For this step, it is important that your **user/role** and the **bucket** have the appropriate permissions for users to read/write into the bucket.

The following statement in the JSON format is called **policy**. It explicitly defined which operation is allowed/denied by which user/role. In the case below, all operation are allowed (specified by the `s3:*` argument in the `Action` field) by services under your account with attached role (speicified by the `"arn:aws:iam::<ACCOUNT-ID>:role/<ROLE>"` argument) on any file/resources in the bucket (speified by `"arn:aws:s3:::<S3_BUCKET>/*"`).
```json
{
    "Version": "2012-10-17",
    "Id": "Policy1674832359797",
    "Statement": [
        {
            "Sid": "Stmt1674832357905",
            "Effect": "Allow",
            "Principal": {
                "AWS": "arn:aws:iam::<ACCOUNT-ID>:role/<ROLE>"
            },
            "Action": "s3:*",
            "Resource": "arn:aws:s3:::<S3_BUCKET>/*"
        }
    ]
}
```
## AWS Commmand Line Interface (CLI)
In order to check whether the user can read/write in the bucket, we recommend testing from local. The AWS CLI is required (install [here](https://aws.amazon.com/cli/)). This tool is already installed if you are on a EC2 instance running Amazon Linux.

```bash
# list the bucket
aws s3 ls s3://<BUCKET-NAME>

# add a temporary file
aws s3 cp temp s3://<BUCKET-NAME>

# remove a temporary file
aws s3 rm s3://<BUCKET-NAME>/temp
```

If this step works, and if your role and user account are attached to the bucket policy, the rest of the AWS NoisePy tutorial should work.