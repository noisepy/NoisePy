import logging
from abc import ABC, abstractmethod
from typing import Callable, List

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
        logger.debug(f"RANK -, INDICES: {indices}")
        return indices

    def synchronize(self):
        pass


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
        logger.debug(f"RANK {rank}, INDICES: {indices}")
        return indices

    def synchronize(self):
        self.comm.barrier()
