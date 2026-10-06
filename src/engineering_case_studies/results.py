"""Canonical result records used by engineering surrogate case studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PerformanceResult:
    study: str
    method: str
    backend: str
    batch_size: int
    ns_per_eval: float
    evaluations_per_second: float
    total_seconds: float
    source: str
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AccuracyResult:
    study: str
    method: str
    budget: int
    rmse: float
    p99_abs: float
    linf: float
    source: str
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConstructionResult:
    study: str
    method: str
    budget: int
    oracle_evaluations: int
    oracle_seconds: float
    adaptive_seconds: float
    representation_seconds: float
    compiler_seconds: float
    total_seconds: float
    source: str
    metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
