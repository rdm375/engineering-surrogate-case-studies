"""Local subprocess execution backend."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class LocalResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class LocalExecutor:
    """Execute a study job synchronously on the current machine."""

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        check: bool = True,
    ) -> LocalResult:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=merged_env,
            text=True,
            capture_output=True,
            check=check,
        )
        return LocalResult(
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
