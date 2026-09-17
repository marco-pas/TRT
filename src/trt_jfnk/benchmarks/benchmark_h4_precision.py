"""Benchmark for Hypothesis H4: Reduced precision with JFNK scaling and block preconditioning.

Hypothesis H4 states:
The combination of AD JVPs, dimensionless norms, and block physics
preconditioning extends the range of TRT parameters for which FP32 produces
a converged and physically accurate solution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from trt_jfnk.benchmarks.configurations import SuOlsonBenchmarkConfig
from trt_jfnk.solvers.block_preconditioners import su_olson_local_block_factory
from trt_jfnk.solvers.jvp import JVPOptions
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.newton import NewtonOptions, newton_krylov
from trt_jfnk.solvers.scaling import ScalePolicy


@dataclass(frozen=True)
class H4RunResult:
    name: str
    precision: str
    jvp: str
    scaling: str
    preconditioner: str
    coupling_epsilon: float
    converged: bool
    steps_completed: int
    total_newton: int
    total_krylov: int
    physical_residual: float
    scaled_residual: float
    rel_error_to_fp64_ref: float
    total_solve_time: float
    reason: str


def run_solver_configuration(
    config: SuOlsonBenchmarkConfig,
    precision: Literal["float64", "float32"],
    jvp_mode: Literal["ad", "fd"],
    scaling_mode: Literal["none", "fixed", "state"],
    preconditioner_mode: Literal["none", "local-block"],
    krylov_backend: Literal["scipy", "cupy", "jax"] = "scipy",
    newton_atol: float = 1.0e-8,
    newton_rtol: float = 1.0e-6,
    max_newton: int = 15,
    max_krylov: int = 200,
) -> tuple[jax.Array | None, dict[str, object]]:
    """Run one complete multi-step solve of the Su-Olson problem."""
    enable_x64 = precision == "float64"
    jax.config.update("jax_enable_x64", enable_x64)
    dtype = jnp.float64 if enable_x64 else jnp.float32

    # In FP32, adjust atol threshold to avoid requesting impossible machine precision
    effective_atol = newton_atol if enable_x64 else max(newton_atol, 1.0e-5)
    effective_rtol = newton_rtol if enable_x64 else max(newton_rtol, 1.0e-4)

    grid = config.grid()
    boundary = config.boundary()
    model = config.model()
    state = config.initial_state(dtype=dtype)

    policy = ScalePolicy(mode=scaling_mode, state_references=(1.0, 1.0))
    options = NewtonOptions(
        atol=effective_atol,
        rtol=effective_rtol,
        max_iterations=max_newton,
        jvp=JVPOptions(mode=jvp_mode),
        krylov=KrylovOptions(
            backend=krylov_backend,
            method="gmres",
            rtol=1.0e-5 if not enable_x64 else 1.0e-8,
            max_iterations=max_krylov,
        ),
    )

    prec_factory = None
    if preconditioner_mode == "local-block":
        prec_factory = su_olson_local_block_factory(model, config.dt, grid, boundary)

    total_newton = 0
    total_krylov = 0
    krylov_tracked = True
    start_time = time.perf_counter()
    converged_all = True
    last_reason = "converged"
    last_scaled_norm = 0.0
    last_phys_norm = 0.0

    steps_completed = 0
    for step in range(1, config.steps + 1):
        old_state = state
        time_old = (step - 1) * config.dt
        time_new = step * config.dt
        source_old = config.source(time_old, dtype=dtype)
        source_new = config.source(time_new, dtype=dtype)

        def residual(cand):
            return model.residual(cand, old_state, config.dt, source_new, source_old, grid, boundary)

        result = newton_krylov(
            residual,
            old_state,
            grid.size,
            policy,
            options,
            preconditioner_factory=prec_factory,
        )
        jax.block_until_ready(result.state)

        total_newton += result.iterations
        if result.total_krylov_iterations >= 0:
            total_krylov += result.total_krylov_iterations
        else:
            krylov_tracked = False
        last_scaled_norm = result.scaled_residual_norm
        last_phys_norm = result.physical_residual_norm
        last_reason = result.reason

        if not result.converged:
            converged_all = False
            break

        state = result.state
        steps_completed += 1

    total_time = time.perf_counter() - start_time
    stats = {
        "converged": converged_all,
        "steps_completed": steps_completed,
        "total_newton": total_newton,
        "total_krylov": total_krylov if krylov_tracked else -1,
        "scaled_residual": last_scaled_norm,
        "physical_residual": last_phys_norm,
        "total_time": total_time,
        "reason": last_reason,
    }
    return state if converged_all else None, stats


def run_reduced_precision_benchmark(
    coupling_epsilons: Sequence[float] = (1.0e-2, 1.0e-4, 1.0e-6),
    steps: int = 20,
    nx: int = 65,
    ny: int = 8,
    krylov_backend: Literal["scipy", "cupy", "jax"] = "scipy",
) -> list[H4RunResult]:
    """Compare solver configurations across precisions and coupling parameters."""
    results: list[H4RunResult] = []

    configs_to_test = [
        # (name, precision, jvp, scaling, preconditioner)
        ("FP64 Reference", "float64", "ad", "state", "local-block"),
        ("FP32 + FD +  None + None", "float32", "fd", "none", "none"),
        ("FP32 + AD +  None + None", "float32", "ad", "none", "none"),
        ("FP32 + AD + Scale + None", "float32", "ad", "state", "none"),
        ("FP32 + AD + Scale + Prec", "float32", "ad", "state", "local-block"),
        ("FP32 + FD + Scale + Prec", "float32", "fd", "state", "local-block"),
    ]

    for eps in coupling_epsilons:
        base_config = SuOlsonBenchmarkConfig(nx=nx, ny=ny, steps=steps, epsilon=eps)

        # 1. Run FP64 Reference first to get benchmark state
        ref_state_64, ref_stats = run_solver_configuration(
            base_config,
            precision="float64",
            jvp_mode="ad",
            scaling_mode="state",
            preconditioner_mode="local-block",
            krylov_backend=krylov_backend,
        )

        if ref_state_64 is not None:
            ref_state_np = np.asarray(jax.device_get(ref_state_64), dtype=np.float64)
            ref_norm_64 = float(np.linalg.norm(ref_state_np))
        else:
            ref_state_np = None
            ref_norm_64 = 1.0

        for name, prec, jvp, scale, prec_type in configs_to_test:
            if name == "FP64 Reference":
                state, stats = ref_state_64, ref_stats
                rel_err = 0.0
            else:
                state, stats = run_solver_configuration(
                    base_config,
                    precision=prec,
                    jvp_mode=jvp,
                    scaling_mode=scale,
                    preconditioner_mode=prec_type,
                    krylov_backend=krylov_backend,
                )
                if state is not None and ref_state_np is not None:
                    state_np = np.asarray(jax.device_get(state), dtype=np.float64)
                    diff = float(np.linalg.norm(state_np - ref_state_np))
                    rel_err = diff / ref_norm_64
                else:
                    rel_err = float("nan")

            results.append(
                H4RunResult(
                    name=name,
                    precision=prec,
                    jvp=jvp,
                    scaling=scale,
                    preconditioner=prec_type,
                    coupling_epsilon=eps,
                    converged=bool(stats["converged"]),
                    steps_completed=int(stats["steps_completed"]),
                    total_newton=int(stats["total_newton"]),
                    total_krylov=int(stats["total_krylov"]),
                    physical_residual=float(stats["physical_residual"]),
                    scaled_residual=float(stats["scaled_residual"]),
                    rel_error_to_fp64_ref=rel_err,
                    total_solve_time=float(stats["total_time"]),
                    reason=str(stats["reason"]),
                )
            )

    return results
