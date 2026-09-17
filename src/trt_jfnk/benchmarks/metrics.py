"""Benchmark records and portable CSV/NPZ output."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import numpy as np


@dataclass(frozen=True)
class StepMetrics:
    step: int
    time: float
    converged: bool
    newton_iterations: int
    krylov_iterations: int
    scaled_residual_norm: float
    physical_residual_norm: float
    solve_seconds: float
    state_scale_0: float
    state_scale_1: float
    residual_scale_0: float
    residual_scale_1: float
    reason: str


def write_step_metrics(path: str | Path, records: list[StepMetrics]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("cannot write an empty metrics table")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def write_configuration(path: str | Path, configuration: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_state_archive(
    path: str | Path,
    state,
    grid,
    model,
    final_time: float,
    **extra,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    first, second = model.unpack(state, grid)
    payload = {
        "x": np.asarray(jax.device_get(grid.x)),
        "y": np.asarray(jax.device_get(grid.y)),
        "field_0": np.asarray(jax.device_get(first)),
        "field_1": np.asarray(jax.device_get(second)),
        "final_time": np.asarray(final_time),
    }
    payload.update(extra)
    np.savez_compressed(path, **payload)
