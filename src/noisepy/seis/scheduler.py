import logging
import os
from abc import ABC, abstractmethod
from typing import Callable, List

from noisepy.seis.constants import AWS_BATCH_JOB_ARRAY_INDEX, AWS_BATCH_JOB_ID

logger = logging.getLogger(__name__)


class Scheduler(ABC):
    """A minimal abstraction to select work to do in a process"""

    @abstractmethod
    def initialize(self, initializer: Callable[[], List], shared_vars: int) -> List:
        """
        Initialize the scheduler
        Args:
            initializer: Function to call to initialize shared state variables
            shared_vars: Number of variables that the initializer will return
        Returns:
            List of variables return by the initializer
        """
        pass

    @abstractmethod
    def get_indices(items: list) -> List[int]:
        """Get the indices of the items this process should work on"""
        pass

    @abstractmethod
    def synchronize(self):
        """Synchronizes all processes at the point this method is called"""
        pass


class SingleNodeScheduler(Scheduler):
    """A basic Scheduler implementation for a single process"""

    def initialize(self, initializer: Callable[[], List], shared_vars: int) -> List:
        return initializer()

    def get_indices(self, items: list) -> List[int]:
        indices = list(range(len(items)))
        logger.debug(f"RANK -, INDICES: {len(indices)}")
        return indices

    def synchronize(self):
        pass


class AWSBatchArrayScheduler(SingleNodeScheduler):
    """A scheduler implementation for AWS Batch using array jobs"""

    def is_array_job() -> bool:
        aws_array_index = os.environ.get(AWS_BATCH_JOB_ARRAY_INDEX)
        return aws_array_index is not None

    def get_indices(self, items: list) -> List[int]:
        # read environment variable set by AWS Batch
        aws_array_index = os.environ.get(AWS_BATCH_JOB_ARRAY_INDEX)
        if aws_array_index is None:
            raise ValueError("AWS_BATCH_JOB_ARRAY_INDEX environment variable not set")
        logger.info(f"Running in AWS Batch array job, index: {aws_array_index}")
        array_size = AWSBatchArrayScheduler._get_array_size()
        indices = list(range(int(aws_array_index), len(items), array_size))
        logger.debug(f"RANK {aws_array_index}/{array_size}, INDICES: {len(indices)}")
        return indices

    def _get_array_size():
        import boto3

        worker_job_id = os.environ.get(AWS_BATCH_JOB_ID, "")
        if ":" not in worker_job_id:
            raise ValueError(f"{AWS_BATCH_JOB_ID} is not in the expected format for an array job: '{worker_job_id}'")

        parent_job_id = worker_job_id.split(":")[0]
        logger.info(f"AWS BATCH Parent job ID: {parent_job_id}")
        response = boto3.client("batch").describe_jobs(jobs=[parent_job_id])
        if "jobs" not in response or len(response["jobs"]) == 0:
            raise ValueError(f"Could not find parent job with ID {parent_job_id}")
        parent_job = response["jobs"][0]
        array_size = parent_job.get("arrayProperties", {}).get("size")
        logger.info(f"AWS BATCH Array size: {array_size}")
        return int(array_size)


class MPIScheduler(Scheduler):
    """A Scheduler implementation that uses MPI to distribute the work across a set of processes"""

    def __init__(self, root: int = 0) -> None:
        from mpi4py import MPI

        super().__init__()
        self.comm = MPI.COMM_WORLD
        self.root = root

    def initialize(self, initializer: Callable[[], List], shared_vars: int) -> List:
        rank = self.comm.Get_rank()
        if rank == self.root:
            variables = initializer()
            if len(variables) != shared_vars:
                raise ValueError(
                    f"The shared_vars argument ({shared_vars}) must match the number of values returned "
                    f"by the initializer ({len(variables)})"
                )
            for v in variables:
                # send shared variables
                self.comm.bcast(v, root=self.root)
            logger.debug(f"RANK {rank}, variables = {variables}")
            return variables
        else:
            vars = []
            for i in range(shared_vars):
                # receive shared variables
                vars.append(self.comm.bcast(None, root=self.root))
            logger.debug(f"RANK {rank}, vars = {vars}")
            return vars

    def get_indices(self, items: list) -> List[int]:
        rank = self.comm.Get_rank()
        size = self.comm.Get_size()

        rng = range(rank, len(items), size)
        indices = list(rng)
        logger.debug(f"RANK {rank}, INDICES: {len(indices)}")
        return indices

    def synchronize(self):
        self.comm.barrier()
