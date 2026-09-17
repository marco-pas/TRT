#!/usr/bin/env python3
"""Run the nonlinear gray-TRT benchmark with the modular JFNK stack."""

from __future__ import annotations

import argparse
import json
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

from trt_jfnk.benchmarks.configurations import GrayBenchmarkConfig
from trt_jfnk.benchmarks.metrics import (
    StepMetrics,
    write_configuration,
    write_state_archive,
    write_step_metrics,
)
from trt_jfnk.solvers.block_preconditioners import gray_local_block_factory
from trt_jfnk.solvers.jvp import JVPOptions
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.newton import NewtonOptions, newton_krylov
from trt_jfnk.solvers.scaling import ScalePolicy


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", type=Path, default=Path("results/gray_ad_scaled"))
    result.add_argument("--jvp", choices=("ad", "fd"), default="ad")
    result.add_argument("--fd-scheme", choices=("forward", "central"), default="forward")
    result.add_argument("--scaling", choices=("none", "fixed", "state"), default="state")
    result.add_argument("--radiation-scale", type=float, default=1.0e-2)
    result.add_argument("--temperature-scale", type=float, default=2.0e-1)
    result.add_argument("--residual-radiation-scale", type=float)
    result.add_argument("--residual-temperature-scale", type=float)
    result.add_argument("--preconditioner", choices=("none", "local-block"), default="local-block")
    result.add_argument("--krylov-backend", choices=("scipy", "cupy", "jax"), default="scipy")
    result.add_argument("--krylov-method", choices=("gmres", "bicgstab"), default="gmres")
    result.add_argument("--jax-solve-method", choices=("batched", "incremental"), default="batched")
    result.add_argument("--platform", choices=("auto", "cpu", "gpu"), default="auto")
    result.add_argument("--precision", choices=("float64", "float32"), default="float64")
    result.add_argument("--nx", type=int, default=65)
    result.add_argument("--ny", type=int, default=8)
    result.add_argument("--steps", type=int, default=40)
    result.add_argument("--dt", type=float, default=5.0e-3)
    result.add_argument("--theta", type=float, default=1.0)
    result.add_argument("--sigma-a0", type=float, default=1.0)
    result.add_argument("--opacity-exponent", type=float, default=3.0)
    result.add_argument("--initial-temperature", type=float, default=0.2)
    result.add_argument("--source-amplitude", type=float, default=20.0)
    result.add_argument("--source-width", type=float, default=0.30)
    result.add_argument("--source-end-time", type=float, default=0.10)
    result.add_argument("--opacity-temperature-floor", type=float, default=0.05)
    result.add_argument("--scattering-ratio", type=float, default=0.0)
    result.add_argument("--cv0", type=float, default=0.1)
    result.add_argument("--cv1", type=float, default=1.0)
    result.add_argument("--material-exponent", type=int, default=3)
    result.add_argument("--max-newton", type=int, default=20)
    result.add_argument("--max-krylov", type=int, default=300)
    result.add_argument("--restart", type=int, default=50)
    result.add_argument("--newton-atol", type=float, default=1.0e-10)
    result.add_argument("--newton-rtol", type=float, default=1.0e-8)
    result.add_argument("--krylov-rtol", type=float, default=1.0e-8)
    result.add_argument("--continue-on-failure", action="store_true")
    return result


def synchronize(value) -> None:
    jax.block_until_ready(value)


def main() -> None:
    args = parser().parse_args()
    if args.platform != "auto":
        jax.config.update("jax_platform_name", args.platform)
    jax.config.update("jax_enable_x64", args.precision == "float64")
    dtype = jnp.float64 if args.precision == "float64" else jnp.float32

    config = GrayBenchmarkConfig(
        nx=args.nx,
        ny=args.ny,
        dt=args.dt,
        steps=args.steps,
        theta=args.theta,
        initial_temperature=args.initial_temperature,
        source_amplitude=args.source_amplitude,
        source_width=args.source_width,
        source_end_time=args.source_end_time,
        sigma_a0=args.sigma_a0,
        opacity_exponent=args.opacity_exponent,
        opacity_temperature_floor=args.opacity_temperature_floor,
        scattering_ratio=args.scattering_ratio,
        cv0=args.cv0,
        cv1=args.cv1,
        material_exponent=args.material_exponent,
    )
    grid = config.grid()
    boundary = config.boundary()
    model = config.model()
    state = config.initial_state(dtype=dtype)

    residual_references = None
    if args.residual_radiation_scale is not None or args.residual_temperature_scale is not None:
        if args.residual_radiation_scale is None or args.residual_temperature_scale is None:
            raise ValueError("set both residual scale arguments or neither")
        residual_references = (
            args.residual_radiation_scale,
            args.residual_temperature_scale,
        )
    scale_policy = ScalePolicy(
        mode=args.scaling,
        state_references=(args.radiation_scale, args.temperature_scale),
        residual_references=residual_references,
    )
    options = NewtonOptions(
        atol=args.newton_atol,
        rtol=args.newton_rtol,
        max_iterations=args.max_newton,
        jvp=JVPOptions(mode=args.jvp, fd_scheme=args.fd_scheme),
        krylov=KrylovOptions(
            backend=args.krylov_backend,
            method=args.krylov_method,
            rtol=args.krylov_rtol,
            max_iterations=args.max_krylov,
            restart=args.restart,
            jax_solve_method=args.jax_solve_method,
        ),
    )
    preconditioner_factory = None
    if args.preconditioner == "local-block":
        preconditioner_factory = gray_local_block_factory(model, config.dt, grid, boundary)

    records: list[StepMetrics] = []
    for step in range(1, config.steps + 1):
        old_state = state
        time_old = (step - 1) * config.dt
        time_new = step * config.dt
        source_old = config.source(time_old, dtype=dtype)
        source_new = config.source(time_new, dtype=dtype)

        def residual(candidate):
            return model.residual(
                candidate,
                old_state,
                config.dt,
                source_new,
                source_old,
                grid,
                boundary,
            )

        admissible = lambda candidate: model.is_admissible(
            candidate, grid, radiation_floor=-1.0e-12, temperature_floor=1.0e-8
        )
        start = time.perf_counter()
        result = newton_krylov(
            residual,
            old_state,
            grid.size,
            scale_policy,
            options,
            admissible_fn=admissible,
            preconditioner_factory=preconditioner_factory,
        )
        synchronize(result.state)
        elapsed = time.perf_counter() - start
        state = result.state
        context = result.scale_context
        records.append(
            StepMetrics(
                step=step,
                time=time_new,
                converged=result.converged,
                newton_iterations=result.iterations,
                krylov_iterations=result.total_krylov_iterations,
                scaled_residual_norm=result.scaled_residual_norm,
                physical_residual_norm=result.physical_residual_norm,
                solve_seconds=elapsed,
                state_scale_0=context.state_block_scales[0],
                state_scale_1=context.state_block_scales[1],
                residual_scale_0=context.residual_block_scales[0],
                residual_scale_1=context.residual_block_scales[1],
                reason=result.reason,
            )
        )
        print(
            f"step={step:4d} t={time_new:.4e} ok={result.converged} "
            f"N={result.iterations:2d} K={result.total_krylov_iterations:4d} "
            f"||Fhat||={result.scaled_residual_norm:.3e} {elapsed:.3f}s"
        )
        if not result.converged and not args.continue_on_failure:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_step_metrics(args.output_dir / "metrics.csv", records)
    write_state_archive(
        args.output_dir / "state.npz",
        state,
        grid,
        model,
        records[-1].time,
        field_0_name="radiation_E",
        field_1_name="temperature_T",
    )
    metadata = config.as_dict()
    metadata.update(
        {
            "jvp": args.jvp,
            "fd_scheme": args.fd_scheme,
            "scaling": args.scaling,
            "state_references": list(scale_policy.state_references),
            "residual_references": residual_references,
            "preconditioner": args.preconditioner,
            "krylov_backend": args.krylov_backend,
            "krylov_method": args.krylov_method,
            "precision": args.precision,
            "platform": str(jax.devices()[0].platform),
        }
    )
    write_configuration(args.output_dir / "config.json", metadata)
    print(json.dumps({"output_dir": str(args.output_dir), "steps_completed": len(records)}, indent=2))


if __name__ == "__main__":
    main()
