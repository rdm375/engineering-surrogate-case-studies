"""Thin Slurm submission backend.

Slurm is an execution backend for a case study, not a numerical dependency.
The backend submits existing .sbatch files and records scheduler identifiers;
it intentionally does not encode site-specific partitions or accounts.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

_JOB_RE = re.compile(r"^(?:Submitted batch job\s+)?(?P<job_id>\d+)(?:;.*)?$")


@dataclass(frozen=True)
class SlurmSubmission:
    job_id: str
    script: Path
    command: tuple[str, ...]
    stdout: str


class SlurmExecutor:
    """Submit jobs through ``sbatch`` using parsable scheduler output."""

    def __init__(self, sbatch: str = "sbatch") -> None:
        self.sbatch = sbatch

    def available(self) -> bool:
        return shutil.which(self.sbatch) is not None

    def submit(
        self,
        script: Path,
        *,
        script_args: Sequence[str] = (),
        exports: Mapping[str, str] | None = None,
        extra_sbatch_args: Sequence[str] = (),
        cwd: Path | None = None,
    ) -> SlurmSubmission:
        command = [self.sbatch, "--parsable", *extra_sbatch_args]
        if exports:
            encoded = ",".join(f"{key}={value}" for key, value in exports.items())
            command.append(f"--export=ALL,{encoded}")
        command.extend([str(script), *script_args])

        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
        )
        stdout = completed.stdout.strip()
        match = _JOB_RE.match(stdout)
        if match is None:
            raise RuntimeError(f"could not parse sbatch job id from: {stdout!r}")

        return SlurmSubmission(
            job_id=match.group("job_id"),
            script=script,
            command=tuple(command),
            stdout=stdout,
        )
