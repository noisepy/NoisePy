import os
from typing import List
from unittest.mock import MagicMock, Mock, patch  # noqa: F401

import pytest

from noisepy.seis.constants import AWS_BATCH_JOB_ARRAY_INDEX, AWS_BATCH_JOB_ID
from noisepy.seis.scheduler import (
    AWSBatchArrayScheduler,
    MPIScheduler,
    SingleNodeScheduler,
)

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
VALID_JOB_ID = "job_id:123"


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

    os.environ.pop(AWS_BATCH_JOB_ARRAY_INDEX, None)


def test_get_indices_missing_index():
    os.environ.pop(AWS_BATCH_JOB_ARRAY_INDEX, None)
    os.environ[AWS_BATCH_JOB_ID] = VALID_JOB_ID

    items = [1, 2, 3, 4, 5]
    scheduler = AWSBatchArrayScheduler()
    with pytest.raises(ValueError) as e:
        scheduler.get_indices(items)
    assert str(e.value) == "AWS_BATCH_JOB_ARRAY_INDEX environment variable not set"


# is_array_job test
def test_is_array_job():
    # no index
    assert not AWSBatchArrayScheduler.is_array_job()
    # with index
    os.environ[AWS_BATCH_JOB_ARRAY_INDEX] = "2"
    assert AWSBatchArrayScheduler.is_array_job()


# MPIScheduler Test


# Create a fixture to provide an instance of MPIScheduler for testing
@pytest.fixture
def mpi_scheduler():
    scheduler = MPIScheduler(root=0)
    scheduler.comm = MagicMock()

    return scheduler


# Test the initialize method
def test_initializer(mpi_scheduler):
    expected_values = [1, 1, 1]

    # Define a simple initializer function for testing
    def initializer():
        return expected_values

    # Mock the bcast method to return the expected values

    mpi_scheduler.comm.bcast.return_value = 1

    # Test for the root process (rank 0)
    mpi_scheduler.comm.Get_rank.return_value = 0
    result = mpi_scheduler.initialize(initializer, shared_vars=3)
    assert result == expected_values

    mpi_scheduler.comm.Get_rank.return_value = 1
    result = mpi_scheduler.initialize(initializer, shared_vars=3)
    assert result == expected_values


# Test the get_indices method
def test_get_indices(mpi_scheduler):
    # Define a list of items for testing
    items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Set the rank and size for testing
    mpi_scheduler.comm.Get_rank.return_value = 2
    mpi_scheduler.comm.Get_size.return_value = 4

    expected_indices = [2, 6]

    result = mpi_scheduler.get_indices(items)
    assert result == expected_indices


# Test the synchronize method
def test_synchronizeMPI(mpi_scheduler):
    # Call the synchronize method
    mpi_scheduler.synchronize()

    # Verify that the barrier operation was called
    mpi_scheduler.comm.barrier.assert_called_once()
