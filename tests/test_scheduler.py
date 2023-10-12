import os
from typing import List
from unittest.mock import MagicMock, Mock, patch  # noqa: F401

import pytest

from noisepy.seis.constants import AWS_BATCH_JOB_ARRAY_INDEX, AWS_BATCH_JOB_ID
from noisepy.seis.scheduler import AWSBatchArrayScheduler, SingleNodeScheduler

VALID_JOB_ID = "job_id:123"
# Single Node Scheduler


# Define a sample initializer function for testing
def sample_initializer() -> List:
    return [1, 2, 3]


def test_initialize():
    scheduler = SingleNodeScheduler()
    result = scheduler.initialize(sample_initializer, 5)

    assert isinstance(result, list)
    assert result == sample_initializer()


def test_single_node_scheduler_get_indices():
    single_node_scheduler = SingleNodeScheduler()
    items = ["a", "b", "c", "d"]
    expected_indices = [0, 1, 2, 3]
    indices = single_node_scheduler.get_indices(items)

    assert indices == expected_indices


def test_synchronize():
    scheduler = SingleNodeScheduler()
    result = scheduler.synchronize()

    assert result is None


# AWS Batch Array Scheduler


# get_array_size test
@patch("boto3.client")
def test_get_array_size(mock_boto3_client):
    boto3_response = {"jobs": [{"arrayProperties": {"size": 10}}]}

    # Test 1 - valid job id format
    # Mock the AWS_BATCH_JOB_ID environment variable with a valid format
    os.environ[AWS_BATCH_JOB_ID] = VALID_JOB_ID
    mock_boto3_client.return_value.describe_jobs.return_value = boto3_response
    array_size = AWSBatchArrayScheduler._get_array_size()
    assert array_size == 10

    # Test 2 : Invalid job_id format
    os.environ[AWS_BATCH_JOB_ID] = "invalid_format"
    with pytest.raises(ValueError) as e:
        AWSBatchArrayScheduler._get_array_size()
    assert "invalid_format" in str(e.value)

    # Test case 3: No "jobs" key in the response
    os.environ[AWS_BATCH_JOB_ID] = VALID_JOB_ID
    mock_boto3_client.return_value.describe_jobs.return_value = {}
    with pytest.raises(ValueError) as e:
        AWSBatchArrayScheduler._get_array_size()
    assert str(e.value) == "Could not find parent job with ID job_id"

    # Test case 4: Empty "jobs" list in the response
    os.environ[AWS_BATCH_JOB_ID] = VALID_JOB_ID
    mock_boto3_client.return_value.describe_jobs.return_value = {"jobs": []}
    with pytest.raises(ValueError) as e:
        AWSBatchArrayScheduler._get_array_size()
    assert str(e.value) == "Could not find parent job with ID job_id"


# get_indices method test
def test_get_indices_valid_index():
    os.environ[AWS_BATCH_JOB_ID] = VALID_JOB_ID
    os.environ[AWS_BATCH_JOB_ARRAY_INDEX] = "2"
    scheduler = AWSBatchArrayScheduler()
    # Mock the _get_array_size method to return a known value
    get_array_size_mock = MagicMock(return_value=5)

    # Assign the MagicMock to the _get_array_size method of the class
    AWSBatchArrayScheduler._get_array_size = get_array_size_mock

    item = [1, 2, 3, 4, 5]
    indices = scheduler.get_indices(item)

    assert indices == [2]


def test_get_indices_missing_index():
    # os.environ["AWS_BATCH_JOB_ARRAY_INDEX"] = ""
    os.environ[AWS_BATCH_JOB_ID] = VALID_JOB_ID

    items = [1, 2, 3, 4, 5]
    scheduler = AWSBatchArrayScheduler()
    with pytest.raises(ValueError) as e:
        scheduler.get_indices(items)
    assert str(e.value) == "AWS_BATCH_JOB_ARRAY_INDEX environment variable not set"


# is_array_job test
def test_is_array_job():
    # no index
    assert AWSBatchArrayScheduler.is_array_job()
    # with index
    os.environ[AWS_BATCH_JOB_ARRAY_INDEX] = "2"
    assert AWSBatchArrayScheduler.is_array_job()
