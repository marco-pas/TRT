#!/usr/bin/env python3
"""Simulation and verification driver for the Ghosh (2014) planar slab TRT benchmark.

Compares cell-centered Finite Volume JFNK simulation against the exact
closed-form analytical reference solution from:
    Karabi Ghosh (2014), "Analytical benchmark for non-equilibrium radiation
    diffusion in finite size systems", Annals of Nuclear Energy 63, 59-68.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from trt_jfnk.benchmarks.ghosh_slab_analytic import GhoshSlabAnalyticBenchmark
from trt_jfnk.benchmarks.ghosh_slab_fv import GhoshSlabFVProblem
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.newton import NewtonOptions, newton_krylov
from trt_jfnk.solvers.scaling import ScalePolicy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run cell-centered FV JFNK simulation for the Ghosh planar slab TRT benchmark."
    )
    parser.add_argument("--b", type=float, default=1.0, help="Slab thickness (default: 1.0)")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Coupling parameter epsilon (default: 0.1)")
    parser.add_argument("--nx", type=int, default=50, help="Number of cells along x (default: 50)")
    parser.add_argument("--ny", type=int, default=1, help="Number of cells along y (default: 1)")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step dt (default: 0.001)")
    parser.add_argument("--t-final", type=float, default=1.0, help="Final simulation time (default: 1.0)")
    parser.add_argument("--theta", type=float, default=1.0, help="Theta parameter: 1.0=Backward Euler, 0.5=Crank-Nicolson (default: 1.0)")
    parser.add_argument(
        "--preconditioner",
        choices=["schur", "local-block", "none"],
        default="schur",
        help="Preconditioner choice (default: schur)",
    )
    parser.add_argument(
        "--krylov-backend",
        choices=["scipy", "cupy", "jax"],
        default="scipy",
        help="Krylov solver backend (default: scipy)",
    )
    parser.add_argument(
        "--platform",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="JAX execution platform (default: auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ghosh_slab"),
        help="Directory to save output files",
    )
    parser.add_argument(
        "--assert-error-threshold",
        type=float,
        default=None,
        help="Fail with exit code 1 if max L_inf or L_2 error exceeds this threshold",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.platform != "auto":
        jax.config.update("jax_platform_name", args.platform)
    jax.config.update("jax_enable_x64", True)

    problem = GhoshSlabFVProblem(
        b=args.b,
        epsilon=args.epsilon,
        nx=args.nx,
        ny=args.ny,
        theta=args.theta,
    )
    benchmark = GhoshSlabAnalyticBenchmark(
        b=args.b,
        epsilon=args.epsilon,
        n_roots=40,
    )

    prec_factory = problem.preconditioner(args.dt, kind=args.preconditioner)
    options = NewtonOptions(
        atol=1e-10,
        rtol=1e-8,
        krylov=KrylovOptions(rtol=1e-8, backend=args.krylov_backend),
    )

    n_steps = int(round(args.t_final / args.dt))
    sample_targets = [0.01, 0.1, 1.0, 5.0, 10.0]
    sample_targets = [t for t in sample_targets if t <= args.t_final + 1e-9]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print(" Ghosh (2014) Finite Planar Slab TRT Benchmark Simulation (Cell-Centered FV)")
    print("================================================================================")
    print(f" Parameters: b = {args.b}, epsilon = {args.epsilon}, Grid: {args.nx} x {args.ny}")
    print(f" Time step: dt = {args.dt}, Steps: {n_steps}, Final time: {args.t_final}")
    print(f" Solver: Newton-Krylov (backend={args.krylov_backend}, prec={args.preconditioner})")
    print("--------------------------------------------------------------------------------")

    u = problem.initial()
    current_time = 0.0
    total_newton = 0
    total_krylov = 0
    start_sim = time.perf_counter()

    sample_results = []
    step_records = []

    for step in range(1, n_steps + 1):
        old = u
        current_time = step * args.dt

        if step % 50 == 0:
            print(f"     Step: {step}/{n_steps}")

        residual = lambda v: problem.residual(v, old, dt=args.dt, time=current_time)
        res = newton_krylov(
            residual,
            old,
            problem.size,
            ScalePolicy("none"),
            options,
            problem.admissible,
            prec_factory,
        )

        if not res.converged:
            print(f"ERROR: Step {step} failed to converge at t = {current_time:.4f}")
            sys.exit(1)

        u = res.state
        total_newton += res.iterations
        total_krylov += res.total_krylov_iterations

        energy_defect = float(
            problem.energy(u) - problem.energy(old) - args.dt * problem.input_power(u)
        )
        u_field, _ = problem.unpack(u)
        step_records.append({
            "step": step,
            "time": current_time,
            "converged": res.converged,
            "newton": res.iterations,
            "krylov": res.total_krylov_iterations,
            "physical_residual": float(res.physical_residual_norm),
            "scaled_residual": float(res.scaled_residual_norm),
            "energy_defect": energy_defect,
            "min_u": float(u_field.min()),
            "max_u": float(u_field.max()),
        })

        # Check if current_time matches any sample target
        for target in sample_targets:
            if abs(current_time - target) < 0.5 * args.dt:
                u_num, v_num = problem.unpack(u)
                u_num_1d = np.asarray(u_num[:, 0])
                v_num_1d = np.asarray(v_num[:, 0])

                u_ref, v_ref = benchmark.evaluate(problem.x, tau=current_time)

                l2_u = float(np.linalg.norm(u_num_1d - u_ref) / np.sqrt(problem.nx))
                l2_v = float(np.linalg.norm(v_num_1d - v_ref) / np.sqrt(problem.nx))
                linf_u = float(np.max(np.abs(u_num_1d - u_ref)))
                linf_v = float(np.max(np.abs(v_num_1d - v_ref)))

                sample_results.append({
                    "tau": current_time,
                    "step": step,
                    "l2_u": l2_u,
                    "l2_v": l2_v,
                    "linf_u": linf_u,
                    "linf_v": linf_v,
                    "u_left": float(u_num_1d[0]),
                    "u_right": float(u_num_1d[-1]),
                    "v_left": float(v_num_1d[0]),
                    "v_right": float(v_num_1d[-1]),
                })

    sim_duration = time.perf_counter() - start_sim

    print("\n Verification Against Exact Analytical Reference Solution:")
    print("--------------------------------------------------------------------------------")
    print(f" {'tau':>7} | {'L2(u)':>11} | {'Linf(u)':>11} | {'L2(v)':>11} | {'Linf(v)':>11} | {'u(0) / v(0)':>17}")
    print("--------------------------------------------------------------------------------")
    for r in sample_results:
        print(
            f" {r['tau']:7.3f} | {r['l2_u']:11.4e} | {r['linf_u']:11.4e} | "
            f"{r['l2_v']:11.4e} | {r['linf_v']:11.4e} | "
            f"{r['u_left']:7.4f} / {r['v_left']:7.4f}"
        )
    print("--------------------------------------------------------------------------------")
    print(f" Total Simulation Time: {sim_duration:.3f} s")
    print(f" Avg Newton iters/step: {total_newton / n_steps:.2f}")
    print(f" Avg Krylov iters/step: {total_krylov / n_steps:.2f}")

    # Steady state check
    u_num, v_num = problem.unpack(u)
    u_inf, _ = problem.steady_state()
    l2_ss = float(np.linalg.norm(np.asarray(u_num) - np.asarray(u_inf)) / np.sqrt(problem.size))
    print(f" Difference from Steady-State at t = {args.t_final:.2f}: L2 = {l2_ss:.4e}")
    print("================================================================================")

    if args.assert_error_threshold is not None:
        u_final, v_final = problem.unpack(u)
        u_exact_final, v_exact_final = benchmark.evaluate(problem.x, tau=args.t_final)
        u_1d = np.asarray(u_final[:, 0])
        v_1d = np.asarray(v_final[:, 0])

        linf_u = float(np.max(np.abs(u_1d - u_exact_final)))
        l2_u = float(np.linalg.norm(u_1d - u_exact_final) / np.sqrt(problem.nx))
        linf_v = float(np.max(np.abs(v_1d - v_exact_final)))
        l2_v = float(np.linalg.norm(v_1d - v_exact_final) / np.sqrt(problem.nx))
        max_error = max(linf_u, l2_u, linf_v, l2_v)

        print(f" Error Verification against threshold {args.assert_error_threshold:.1e}:")
        print(f"   L_inf(u) = {linf_u:.4e}")
        print(f"   L_2(u)   = {l2_u:.4e}")
        print(f"   L_inf(v) = {linf_v:.4e}")
        print(f"   L_2(v)   = {l2_v:.4e}")

        if max_error >= args.assert_error_threshold:
            print(f" CI ERROR: Maximum error {max_error:.4e} exceeds threshold {args.assert_error_threshold:.1e}")
            sys.exit(1)
        print(f" CI PASS: All L_inf and L_2 errors are strictly < {args.assert_error_threshold:.1e}")
        print("================================================================================")

    # Save results
    config = {
        "b": args.b,
        "epsilon": args.epsilon,
        "nx": args.nx,
        "ny": args.ny,
        "dt": args.dt,
        "t_final": args.t_final,
        "theta": args.theta,
        "preconditioner": args.preconditioner,
        "krylov_backend": args.krylov_backend,
        "sim_duration_seconds": sim_duration,
        "total_newton_iterations": total_newton,
        "total_krylov_iterations": total_krylov,
        "sample_results": sample_results,
    }
    (args.output_dir / "ghosh_slab_results.json").write_text(json.dumps(config, indent=2))
    if step_records:
        with (args.output_dir / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(step_records[0].keys()))
            writer.writeheader()
            writer.writerows(step_records)
    u_final, v_final = problem.unpack(u)
    np.savez(
        args.output_dir / "ghosh_slab_fields.npz",
        x=np.asarray(problem.x),
        u=np.asarray(u_final),
        v=np.asarray(v_final),
        time=current_time,
    )
    print(f" Results successfully saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
