"""Reproducibility and provenance helpers."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path


def _command(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            args, cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state(path: Path) -> dict[str, object]:
    status = _command(["git", "status", "--porcelain"], path)
    return {
        "path": str(path.resolve()),
        "commit": _command(["git", "rev-parse", "HEAD"], path),
        "describe": _command(["git", "describe", "--tags", "--always"], path),
        "branch": _command(["git", "branch", "--show-current"], path),
        "dirty": bool(status),
        "remote": _command(["git", "remote", "get-url", "origin"], path),
    }


def collect_environment(repositories: dict[str, Path]) -> dict[str, object]:
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
        "repositories": {
            name: git_state(path) for name, path in repositories.items()
        },
        "nvidia_smi": _command([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]),
    }


def save_environment(output: Path, repositories: dict[str, Path]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(collect_environment(repositories), indent=2) + "\n",
        encoding="utf-8",
    )
