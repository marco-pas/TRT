#!/usr/bin/env python3
"""Run direct benchmark investigations for Hypotheses H1 and H4 on Su-Olson.

Usage:
    python scripts/run_benchmarks_su_olson.py --target all
    python scripts/run_benchmarks_su_olson.py --target h1
    python scripts/run_benchmarks_su_olson.py --target h4 --platform gpu --krylov-backend cupy
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import jax

from trt_jfnk.benchmarks.benchmark_h1_tangent import run_tangent_accuracy_benchmark
from trt_jfnk.benchmarks.benchmark_h4_precision import run_reduced_precision_benchmark
from trt_jfnk.benchmarks.configurations import SuOlsonBenchmarkConfig


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("all", "h1", "h4"),
        default="all",
        help="Which hypothesis benchmark to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "benchmarks_su_olson",
        help="Directory where benchmark CSV metrics will be saved",
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Hardware platform",
    )
    parser.add_argument(
        "--krylov-backend",
        choices=("scipy", "cupy", "jax"),
        default="scipy",
        help="Backend for linear solves (SciPy on CPU or CuPy on GPU)",
    )
    parser.add_argument("--steps", type=int, default=20, help="Number of steps for H4")
    parser.add_argument("--nx", type=int, default=65, help="Grid points in x")
    parser.add_argument("--ny", type=int, default=8, help="Grid points in y")
    return parser


def run_h1(output_dir: Path, nx: int, ny: int) -> None:
    print("\n")
    print("RUNNING BENCHMARK H1: TANGENT ACCURACY (AD vs FD)")
    print("\n")
    print("Evaluating directional derivative accuracy ||J_test v - J_ref v|| / ||J_ref v||")
    print("Reference: FP64 Automatic Differentiation (AD JVP)\n")

    config = SuOlsonBenchmarkConfig(nx=nx, ny=ny)
    epsilons = [10.0**p for p in range(-12, 1)]
    records = run_tangent_accuracy_benchmark(config=config, epsilons=epsilons)

    # Save to CSV
    csv_path = output_dir / "h1_tangent_accuracy.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["precision", "scheme", "epsilon", "rel_error", "abs_error", "tangent_norm"]
        )
        for r in records:
            writer.writerow(
                [r.precision, r.scheme, r.epsilon, r.rel_error, r.abs_error, r.tangent_norm]
            )
    print(f"-> Full H1 results written to: {csv_path}")

    # Display clean comparison table for representative epsilons
    print("\n-- H1 Summary Table: Relative Tangent Error vs Reference --\n")
    print(
        f"{'Scheme':<17} | {'Epsilon':<13} | {'FP64 Rel Error':<18} | {'FP32 Rel Error':<18} | {'Error Regime'}"
    )
    print("-" * 85)

    # Extract AD
    ad_64 = next(r for r in records if r.precision == "float64" and r.scheme == "ad")
    ad_32 = next(r for r in records if r.precision == "float32" and r.scheme == "ad")
    print(
        f"{'AD (jax.jvp)':<17} | {'N/A (exact)':<13} | {ad_64.rel_error:<18.2e} | {ad_32.rel_error:<18.2e} | Exact to machine eps"
    )

    sample_eps = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
    for eps in sample_eps:
        fwd_64 = next(
            r
            for r in records
            if r.precision == "float64" and r.scheme == "fd_forward" and math.isclose(r.epsilon, eps)
        )
        fwd_32 = next(
            r
            for r in records
            if r.precision == "float32" and r.scheme == "fd_forward" and math.isclose(r.epsilon, eps)
        )
        if eps <= 1e-8:
            regime = "Cancellation"
        elif eps == 1e-4:
            regime = "Near optimal for FP32"
        elif eps == 1e-8:
            regime = "Near optimal for FP64"
        else:
            regime = "Truncation"
        print(
            f"{'FD Forward':<17} | {eps:<13.0e} | {fwd_64.rel_error:<18.2e} | {fwd_32.rel_error:<18.2e} | {regime}"
        )


def run_h4(
    output_dir: Path, nx: int, ny: int, steps: int, krylov_backend: str
) -> None:
    print("\n")
    print("RUNNING BENCHMARK H4: REDUCED PRECISION (FP32 vs FP64) WITH PRECONDITIONING")
    print("\n")
    print(
        f"Evaluating multi-step nonlinear solver across coupling stiffness eps and solver configurations"
    )
    print(f"Backend: {krylov_backend}, Grid: {nx}x{ny}, Steps: {steps}\n")

    coupling_epsilons = [1.0, 1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8]
    records = run_reduced_precision_benchmark(
        coupling_epsilons=coupling_epsilons,
        steps=steps,
        nx=nx,
        ny=ny,
        krylov_backend=krylov_backend,
    )

    # Save to CSV
    csv_path = output_dir / "h4_reduced_precision.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name",
                "precision",
                "jvp",
                "scaling",
                "preconditioner",
                "coupling_epsilon",
                "converged",
                "steps_completed",
                "total_newton",
                "total_krylov",
                "physical_residual",
                "scaled_residual",
                "rel_error_to_fp64_ref",
                "total_solve_time",
                "reason",
            ]
        )
        for r in records:
            writer.writerow(
                [
                    r.name,
                    r.precision,
                    r.jvp,
                    r.scaling,
                    r.preconditioner,
                    r.coupling_epsilon,
                    r.converged,
                    r.steps_completed,
                    r.total_newton,
                    r.total_krylov,
                    r.physical_residual,
                    r.scaled_residual,
                    r.rel_error_to_fp64_ref,
                    r.total_solve_time,
                    r.reason,
                ]
            )
    print(f"-> Full H4 results written to: {csv_path}")

    # Display clean comparison table
    print("\n-- H4 Summary Table: Solver Robustness & Accuracy --\n")
    header = f"{'Configuration':<42} | {'Coupling ε':<13} | {'Conv?':<5} | {'Steps':<5} | {'Newton':<6} | {'Krylov':<6} | {'Rel Error to FP64':<18} | {'Time (s)':<8}"
    print(header)
    print("-" * len(header))
    current_eps = None
    for r in records:
        if current_eps is not None and r.coupling_epsilon != current_eps:
            print("-" * len(header))
        current_eps = r.coupling_epsilon

        rel_str = f"{r.rel_error_to_fp64_ref:.2e}" if not math.isnan(r.rel_error_to_fp64_ref) else "N/A"
        conv_str = "YES" if r.converged else "FAIL"
        krylov_str = f"{r.total_krylov:<6d}" if r.total_krylov >= 0 else "N/A"
        print(
            f"{r.name:<42} | {r.coupling_epsilon:<13.0e} | {conv_str:<5} | {r.steps_completed:<5d} | {r.total_newton:<6d} | {krylov_str:<6} | {rel_str:<18} | {r.total_solve_time:<8.2f}"
        )


def main() -> None:
    args = parse_arguments().parse_args()
    if args.platform != "auto":
        jax.config.update("jax_platform_name", args.platform)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    import math

    # Expose math in namespace
    globals()["math"] = math

    if args.target in ("all", "h1"):
        run_h1(args.output_dir, args.nx, args.ny)

    if args.target in ("all", "h4"):
        run_h4(args.output_dir, args.nx, args.ny, args.steps, args.krylov_backend)

    print("\n")
    print(f"BENCHMARKS COMPLETE! All output data saved in: {args.output_dir}")
    print("\n")


if __name__ == "__main__":
    main()

