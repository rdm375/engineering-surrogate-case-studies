"""Execution backends for case-study jobs."""

from .local import LocalExecutor
from .slurm import SlurmExecutor, SlurmSubmission

__all__ = ["LocalExecutor", "SlurmExecutor", "SlurmSubmission"]
