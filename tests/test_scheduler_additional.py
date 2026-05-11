import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def load_scheduler_symbols():
    """Load scheduler symbols, with a local source-file fallback for dev environments."""
    try:
        from noisepy.seis.constants import AWS_BATCH_JOB_ARRAY_INDEX, AWS_BATCH_JOB_ID
        from noisepy.seis.scheduler import AWSBatchArrayScheduler, MPIScheduler

        return AWS_BATCH_JOB_ARRAY_INDEX, AWS_BATCH_JOB_ID, AWSBatchArrayScheduler, MPIScheduler
    except ModuleNotFoundError as exc:
        if exc.name is None or not exc.name.startswith("noisepy.seis"):
            raise
        base = Path(__file__).resolve().parents[1] / "src" / "noisepy" / "seis"

        if "noisepy" not in sys.modules:
            pkg = types.ModuleType("noisepy")
            pkg.__path__ = [str(base.parent)]
            sys.modules["noisepy"] = pkg
        if "noisepy.seis" not in sys.modules:
            pkg = types.ModuleType("noisepy.seis")
            pkg.__path__ = [str(base)]
            sys.modules["noisepy.seis"] = pkg

        const_spec = importlib.util.spec_from_file_location("noisepy.seis.constants", base / "constants.py")
        const_mod = importlib.util.module_from_spec(const_spec)
        assert const_spec.loader is not None, "Failed to load constants module"
        const_spec.loader.exec_module(const_mod)
        sys.modules["noisepy.seis.constants"] = const_mod

        scheduler_spec = importlib.util.spec_from_file_location("noisepy.seis.scheduler", base / "scheduler.py")
        scheduler_mod = importlib.util.module_from_spec(scheduler_spec)
        assert scheduler_spec.loader is not None, "Failed to load scheduler module"
        scheduler_spec.loader.exec_module(scheduler_mod)
        sys.modules["noisepy.seis.scheduler"] = scheduler_mod

        return (
            const_mod.AWS_BATCH_JOB_ARRAY_INDEX,
            const_mod.AWS_BATCH_JOB_ID,
            scheduler_mod.AWSBatchArrayScheduler,
            scheduler_mod.MPIScheduler,
        )


AWS_BATCH_JOB_ARRAY_INDEX, AWS_BATCH_JOB_ID, AWSBatchArrayScheduler, MPIScheduler = load_scheduler_symbols()


@patch("boto3.client")
def test_get_array_size_raises_when_size_missing(mock_boto3_client, monkeypatch):
    monkeypatch.setenv(AWS_BATCH_JOB_ID, "parent:1")
    mock_boto3_client.return_value.describe_jobs.return_value = {"jobs": [{"arrayProperties": {}}]}

    with pytest.raises(TypeError, match="int\\(\\) argument must be"):
        AWSBatchArrayScheduler._get_array_size()


def test_get_indices_non_integer_array_index_raises(monkeypatch):
    monkeypatch.setenv(AWS_BATCH_JOB_ARRAY_INDEX, "not-an-int")
    monkeypatch.setattr(AWSBatchArrayScheduler, "_get_array_size", staticmethod(lambda: 3))

    with pytest.raises(ValueError, match="invalid literal"):
        AWSBatchArrayScheduler().get_indices([1, 2, 3, 4])


def test_mpi_initialize_raises_when_shared_vars_mismatch():
    """Root rank must fail when initializer return count differs from shared_vars."""
    scheduler = MPIScheduler.__new__(MPIScheduler)
    scheduler.root = 0
    scheduler.comm = MagicMock()
    scheduler.comm.Get_rank.return_value = 0

    with pytest.raises(ValueError, match="shared_vars"):
        scheduler.initialize(lambda: ["only-one"], shared_vars=2)


def test_mpi_get_indices_empty_when_rank_has_no_work():
    """Ranks beyond the item range should receive no indices to process."""
    scheduler = MPIScheduler.__new__(MPIScheduler)
    scheduler.root = 0
    scheduler.comm = MagicMock()
    scheduler.comm.Get_rank.return_value = 5
    scheduler.comm.Get_size.return_value = 10

    assert scheduler.get_indices([0, 1, 2]) == []
