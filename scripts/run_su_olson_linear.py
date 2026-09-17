#!/usr/bin/env python3
"""Run the linear Su--Olson regression through the same solver stack."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Permit direct execution from an unpacked source tree.  An editable install
# (``python -m pip install -e .``) remains the preferred development setup.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import jax
import jax.numpy as jnp

from trt_jfnk.benchmarks.configurations import SuOlsonBenchmarkConfig
from trt_jfnk.benchmarks.metrics import StepMetrics, write_state_archive, write_step_metrics
from trt_jfnk.solvers.jvp import JVPOptions
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.newton import NewtonOptions, newton_krylov
from trt_jfnk.solvers.scaling import ScalePolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/su_olson_linear"))
    parser.add_argument("--jvp", choices=("ad", "fd"), default="ad")
    parser.add_argument("--scaling", choices=("none", "fixed", "state"), default="none")
    parser.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--krylov-backend", choices=("scipy", "cupy", "jax"), default="scipy")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--nx", type=int, default=65)
    parser.add_argument("--ny", type=int, default=8)
    args = parser.parse_args()
    if args.platform != "auto":
        jax.config.update("jax_platform_name", args.platform)
    jax.config.update("jax_enable_x64", True)

    config = SuOlsonBenchmarkConfig(nx=args.nx, ny=args.ny, steps=args.steps)
    grid, boundary, model = config.grid(), config.boundary(), config.model()
    state = config.initial_state(dtype=jnp.float64)
    policy = ScalePolicy(args.scaling, (1.0, 1.0))
    options = NewtonOptions(
        jvp=JVPOptions(args.jvp),
        krylov=KrylovOptions(backend=args.krylov_backend, max_iterations=300),
    )
    records = []
    for step in range(1, config.steps + 1):
        old_state = state
        time_old, time_new = (step - 1) * config.dt, step * config.dt
        source_old = config.source(time_old, dtype=jnp.float64)
        source_new = config.source(time_new, dtype=jnp.float64)
        residual = lambda candidate: model.residual(
            candidate, old_state, config.dt, source_new, source_old, grid, boundary
        )
        start = time.perf_counter()
        result = newton_krylov(residual, old_state, grid.size, policy, options)
        jax.block_until_ready(result.state)
        state = result.state
        elapsed = time.perf_counter() - start
        context = result.scale_context
        records.append(
            StepMetrics(
                step,
                time_new,
                result.converged,
                result.iterations,
                result.total_krylov_iterations,
                result.scaled_residual_norm,
                result.physical_residual_norm,
                elapsed,
                *context.state_block_scales,
                *context.residual_block_scales,
                result.reason,
            )
        )
        if not result.converged:
            break
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_step_metrics(args.output_dir / "metrics.csv", records)
    write_state_archive(args.output_dir / "state.npz", state, grid, model, records[-1].time)


if __name__ == "__main__":
    main()
