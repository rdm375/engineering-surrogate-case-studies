"""Pareto-front utilities."""

from __future__ import annotations

from collections.abc import Iterable


def nondominated(points: Iterable[tuple[float, float]]) -> list[bool]:
    """Return a mask for a two-objective minimization problem.

    Each point is ``(error, cost)``. A point is dominated when another point is
    no worse in either objective and strictly better in at least one.
    """
    pts = list(points)
    keep: list[bool] = []
    for i, (error_i, cost_i) in enumerate(pts):
        dominated = False
        for j, (error_j, cost_j) in enumerate(pts):
            if i == j:
                continue
            if (
                error_j <= error_i
                and cost_j <= cost_i
                and (error_j < error_i or cost_j < cost_i)
            ):
                dominated = True
                break
        keep.append(not dominated)
    return keep
