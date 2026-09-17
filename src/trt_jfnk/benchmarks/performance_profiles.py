"""Small helpers for comparing nonlinear solver cases."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def read_metrics(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarize(path: str | Path) -> dict[str, float | int]:
    rows = read_metrics(path)
    if not rows:
        raise ValueError("metrics file is empty")
    times = np.asarray([float(row["solve_seconds"]) for row in rows])
    newton = np.asarray([int(row["newton_iterations"]) for row in rows])
    krylov_values = np.asarray([int(row["krylov_iterations"]) for row in rows])
    known_krylov = krylov_values[krylov_values >= 0]
    converged = np.asarray([row["converged"].lower() == "true" for row in rows])
    return {
        "steps": len(rows),
        "converged_steps": int(np.sum(converged)),
        "total_newton": int(np.sum(newton)),
        "total_krylov": int(np.sum(known_krylov)) if known_krylov.size else -1,
        "median_krylov": float(np.median(known_krylov)) if known_krylov.size else -1.0,
        "p95_krylov": float(np.percentile(known_krylov, 95)) if known_krylov.size else -1.0,
        "total_seconds": float(np.sum(times)),
        "median_step_seconds": float(np.median(times)),
    }
