"""Direct benchmark for Hypothesis H1: Tangent accuracy of AD vs FD JVPs.

Hypothesis H1 states:
For a fixed discrete residual, AD JVPs remain accurate to floating-point
roundoff independently of a finite-difference perturbation parameter. FD JVPs
develop scale-dependent truncation and cancellation errors, especially in FP32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from trt_jfnk.benchmarks.configurations import SuOlsonBenchmarkConfig
from trt_jfnk.solvers.jvp import _fd_step


@dataclass(frozen=True)
class TangentErrorRecord:
    precision: str
    scheme: str  # "ad", "fd_forward", "fd_central"
    epsilon: float
    rel_error: float
    abs_error: float
    tangent_norm: float


def run_tangent_accuracy_benchmark(
    config: SuOlsonBenchmarkConfig | None = None,
    epsilons: Sequence[float] | None = None,
    step_time: float = 0.05,
) -> list[TangentErrorRecord]:
    """Execute direct tangent accuracy comparison for Su-Olson."""
    if config is None:
        config = SuOlsonBenchmarkConfig(nx=65, ny=8)
    if epsilons is None:
        epsilons = [10.0**p for p in range(-12, 1)]

    grid = config.grid()
    boundary = config.boundary()
    model = config.model()

    # 1. Build reference state and direction in FP64
    jax.config.update("jax_enable_x64", True)
    state_64 = config.initial_state(dtype=jnp.float64)
    dt = config.dt
    source_old = config.source(0.0, dtype=jnp.float64)
    source_new = config.source(step_time, dtype=jnp.float64)

    # Establish an active state with some spatial variation
    # Take a candidate perturbation from source driving
    candidate_64 = state_64 + 0.1 * model.pack(source_new, 0.1 * source_new)

    def residual_64(x):
        return model.residual(x, state_64, dt, source_new, source_old, grid, boundary)

    # Generate a deterministic normalized direction
    key = jax.random.PRNGKey(42)
    direction_raw = jax.random.normal(key, shape=candidate_64.shape, dtype=jnp.float64)
    direction_64 = direction_raw / jnp.linalg.norm(direction_raw)

    # Compute reference tangent in FP64 AD
    ref_tangent_64 = jax.jvp(residual_64, (candidate_64,), (direction_64,))[1]
    ref_tangent_np = np.asarray(jax.device_get(ref_tangent_64), dtype=np.float64)
    ref_norm = float(np.linalg.norm(ref_tangent_np))
    if ref_norm == 0.0:
        ref_norm = 1.0

    records: list[TangentErrorRecord] = []

    for prec in ("float64", "float32"):
        enable_x64 = prec == "float64"
        jax.config.update("jax_enable_x64", enable_x64)
        dtype = jnp.float64 if enable_x64 else jnp.float32

        candidate = jnp.asarray(candidate_64, dtype=dtype)
        state_old = jnp.asarray(state_64, dtype=dtype)
        src_new = jnp.asarray(source_new, dtype=dtype)
        src_old = jnp.asarray(source_old, dtype=dtype)
        direction = jnp.asarray(direction_64, dtype=dtype)

        def residual(x):
            return model.residual(x, state_old, dt, src_new, src_old, grid, boundary)

        f_x = residual(candidate)

        # AD Tangent
        tangent_ad = jax.jvp(residual, (candidate,), (direction,))[1]
        tangent_ad_np = np.asarray(jax.device_get(tangent_ad), dtype=np.float64)
        diff_ad = float(np.linalg.norm(tangent_ad_np - ref_tangent_np))
        rel_ad = diff_ad / ref_norm
        records.append(
            TangentErrorRecord(
                precision=prec,
                scheme="ad",
                epsilon=0.0,
                rel_error=rel_ad,
                abs_error=diff_ad,
                tangent_norm=float(np.linalg.norm(tangent_ad_np)),
            )
        )

        # Sweep FD perturbations
        for eps in epsilons:
            # Forward difference
            f_perturbed = residual(candidate + eps * direction)
            tangent_fwd = (f_perturbed - f_x) / eps
            tangent_fwd_np = np.asarray(jax.device_get(tangent_fwd), dtype=np.float64)
            diff_fwd = float(np.linalg.norm(tangent_fwd_np - ref_tangent_np))
            rel_fwd = diff_fwd / ref_norm
            records.append(
                TangentErrorRecord(
                    precision=prec,
                    scheme="fd_forward",
                    epsilon=eps,
                    rel_error=rel_fwd,
                    abs_error=diff_fwd,
                    tangent_norm=float(np.linalg.norm(tangent_fwd_np)),
                )
            )

            # Central difference
            f_back = residual(candidate - eps * direction)
            tangent_cnt = (f_perturbed - f_back) / (2.0 * eps)
            tangent_cnt_np = np.asarray(jax.device_get(tangent_cnt), dtype=np.float64)
            diff_cnt = float(np.linalg.norm(tangent_cnt_np - ref_tangent_np))
            rel_cnt = diff_cnt / ref_norm
            records.append(
                TangentErrorRecord(
                    precision=prec,
                    scheme="fd_central",
                    epsilon=eps,
                    rel_error=rel_cnt,
                    abs_error=diff_cnt,
                    tangent_norm=float(np.linalg.norm(tangent_cnt_np)),
                )
            )

    # Restore default x64 config
    jax.config.update("jax_enable_x64", True)
    return records
