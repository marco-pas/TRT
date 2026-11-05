#!/usr/bin/env python3
"""Driver script to run Non-Linear MMS TRT benchmark and save numerical vs exact solutions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from trt_jfnk.benchmarks.mms import MMSConfig, MMSProblem1D
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.newton import NewtonOptions, newton_krylov
from trt_jfnk.solvers.scaling import ScalePolicy


def solve_mms(
    problem: MMSProblem1D,
    dt: float = 0.001,
    t_final: float = 0.05,
    preconditioner: str = "schur",
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve the 1D non-linear MMS problem from t=0 to t=t_final with JFNK."""
    n_steps = int(round(t_final / dt))
    state = problem.initial(0.0)
    prec_factory = problem.preconditioner(dt, kind=preconditioner)

    options = NewtonOptions(
        atol=atol,
        rtol=rtol,
        krylov=KrylovOptions(rtol=rtol, atol=atol),
    )

    total_krylov = 0
    for s in range(n_steps):
        time = (s + 1) * dt
        old_state = state

        def residual_fn(u):
            return problem.residual(u, old_state, dt, time)

        res = newton_krylov(
            residual_fn,
            old_state,
            problem.size,
            ScalePolicy("none"),
            options,
            problem.admissible,
            prec_factory,
        )
        if not res.converged:
            raise RuntimeError(f"Newton solve failed at step {s + 1}, time {time:.4f}")

        total_krylov += res.total_krylov_iterations
        state = res.state

    E_final, T_final = map(np.asarray, problem.unpack(state))
    return E_final, T_final, total_krylov


def plot_comparison(
    x: np.ndarray,
    E_num: np.ndarray,
    T_num: np.ndarray,
    E_exact: np.ndarray,
    T_exact: np.ndarray,
    config: MMSConfig,
    t_final: float,
    output_png: Path,
):
    """Generate publication-quality 4-panel comparison plot."""
    err_E = np.abs(E_num - E_exact)
    err_T = np.abs(T_num - T_exact)
    nx = len(x)

    l2_E = np.linalg.norm(err_E) / np.sqrt(nx)
    linf_E = np.max(err_E)
    l2_T = np.linalg.norm(err_T) / np.sqrt(nx)
    linf_T = np.max(err_T)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: Radiation E
    ax1 = axes[0, 0]
    ax1.plot(x, E_exact, "k-", linewidth=2.0, label="Exact Manufactured $E$")
    ax1.plot(x, E_num, "ro", markersize=5, fillstyle="none", markeredgewidth=1.5, label="Numerical FV JFNK $E$")
    ax1.set_xlabel("Position $x$", fontsize=12)
    ax1.set_ylabel("Radiation Energy Density $E$", fontsize=12)
    ax1.set_title(f"Radiation Field $E(x)$ at $t = {t_final:.4f}$", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=11)

    # Panel 2: Temperature T
    ax2 = axes[0, 1]
    ax2.plot(x, T_exact, "k-", linewidth=2.0, label="Exact Manufactured $T$")
    ax2.plot(x, T_num, "bs", markersize=5, fillstyle="none", markeredgewidth=1.5, label="Numerical FV JFNK $T$")
    ax2.set_xlabel("Position $x$", fontsize=12)
    ax2.set_ylabel("Material Temperature $T$", fontsize=12)
    ax2.set_title(f"Temperature Field $T(x)$ at $t = {t_final:.4f}$", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=11)

    # Panel 3: Absolute Error in E
    ax3 = axes[1, 0]
    ax3.semilogy(x, err_E, "r.-", linewidth=1.5, markersize=6)
    ax3.set_xlabel("Position $x$", fontsize=12)
    ax3.set_ylabel("Pointwise Error $|E_{\\mathrm{num}} - E_{\\mathrm{exact}}|$", fontsize=12)
    ax3.set_title(f"Radiation Error ($L_2 = {l2_E:.2e}$, $L_\\infty = {linf_E:.2e}$)", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.6)

    # Panel 4: Absolute Error in T
    ax4 = axes[1, 1]
    ax4.semilogy(x, err_T, "b.-", linewidth=1.5, markersize=6)
    ax4.set_xlabel("Position $x$", fontsize=12)
    ax4.set_ylabel("Pointwise Error $|T_{\\mathrm{num}} - T_{\\mathrm{exact}}|$", fontsize=12)
    ax4.set_title(f"Temperature Error ($L_2 = {l2_T:.2e}$, $L_\\infty = {linf_T:.2e}$)", fontsize=12, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle(
        f"Non-Linear TRT MMS Benchmark: $p={config.p}$ (opacity), $m={config.m}$ (capacity), $N_x={nx}$",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved comparison plot to {output_png}")


def run_refinement_study(
    p: int,
    m: int,
    resolutions: list[int] = [16, 32, 64, 128],
    dt: float = 0.0002,
    t_final: float = 0.02,
    preconditioner: str = "schur",
):
    """Run spatial grid refinement study and display convergence orders."""
    config = MMSConfig(p=p, m=m)
    print(f"\n================================================================================")
    print(f"SPATIAL REFINEMENT CONVERGENCE STUDY: p={p}, m={m}, dt={dt:.1e}, t_final={t_final}")
    print(f"================================================================================")
    print(f"{'Nx':>5} | {'dx':>8} | {'L2 Err E':>12} | {'Order E':>8} | {'L2 Err T':>12} | {'Order T':>8}")
    print(f"--------------------------------------------------------------------------------")

    prev_dx, prev_eE, prev_eT = None, None, None
    results = []

    for nx in resolutions:
        problem = MMSProblem1D(nx=nx, config=config)
        E_num, T_num, _ = solve_mms(problem, dt=dt, t_final=t_final, preconditioner=preconditioner)
        l2_E, linf_E, l2_T, linf_T = problem.compute_errors(problem.pack(E_num, T_num), t_final)

        dx = config.length / nx
        order_E = (
            np.log(prev_eE / l2_E) / np.log(prev_dx / dx) if prev_eE is not None else float("nan")
        )
        order_T = (
            np.log(prev_eT / l2_T) / np.log(prev_dx / dx) if prev_eT is not None else float("nan")
        )

        order_E_str = f"{order_E:8.3f}" if not np.isnan(order_E) else "     ---"
        order_T_str = f"{order_T:8.3f}" if not np.isnan(order_T) else "     ---"

        print(f"{nx:5d} | {dx:8.5f} | {l2_E:12.5e} | {order_E_str} | {l2_T:12.5e} | {order_T_str}")

        results.append((nx, dx, l2_E, order_E, l2_T, order_T))
        prev_dx, prev_eE, prev_eT = dx, l2_E, l2_T

    print(f"================================================================================\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Non-Linear MMS TRT benchmark.")
    parser.add_argument("--p", type=int, default=3, help="Opacity exponent (default: 3)")
    parser.add_argument("--m", type=int, default=3, help="Capacity exponent (default: 3)")
    parser.add_argument("--nx", type=int, default=32, help="Grid cells along x (default: 32)")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step (default: 0.001)")
    parser.add_argument("--t-final", type=float, default=0.05, help="Final simulation time (default: 0.05)")
    parser.add_argument(
        "--preconditioner",
        choices=["schur", "local-block", "none"],
        default="schur",
        help="Preconditioner kind",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/mms"),
        help="Directory to save solutions and plots",
    )
    parser.add_argument(
        "--refinement",
        action="store_true",
        help="Run spatial grid refinement study across multiple resolutions",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.refinement:
        run_refinement_study(p=args.p, m=args.m, dt=args.dt, t_final=args.t_final, preconditioner=args.preconditioner)

    # Standard single-run solve and export
    config = MMSConfig(p=args.p, m=args.m)
    problem = MMSProblem1D(nx=args.nx, config=config)
    print(f"Running non-linear MMS TRT solve: p={args.p}, m={args.m}, nx={args.nx}, dt={args.dt}, t_final={args.t_final}")

    E_num, T_num, krylov_total = solve_mms(
        problem,
        dt=args.dt,
        t_final=args.t_final,
        preconditioner=args.preconditioner,
    )

    xc = np.asarray(problem.x)
    E_exact, T_exact = map(np.asarray, problem.exact_solution(args.t_final))
    l2_E, linf_E, l2_T, linf_T = problem.compute_errors(problem.pack(E_num, T_num), args.t_final)

    print(f"Final Errors at t={args.t_final:.4f}:")
    print(f"  Radiation E:   L2 = {l2_E:.4e}, Linf = {linf_E:.4e}")
    print(f"  Temperature T: L2 = {l2_T:.4e}, Linf = {linf_T:.4e}")
    print(f"  Total Krylov Iterations: {krylov_total}")

    # Save NPZ fields
    npz_path = args.output_dir / f"mms_solution_p{args.p}_m{args.m}.npz"
    np.savez_compressed(
        npz_path,
        x=xc,
        E_num=E_num,
        T_num=T_num,
        E_exact=E_exact,
        T_exact=T_exact,
        err_E=np.abs(E_num - E_exact),
        err_T=np.abs(T_num - T_exact),
        time=args.t_final,
        p=args.p,
        m=args.m,
        nx=args.nx,
        dt=args.dt,
        l2_E=l2_E,
        linf_E=linf_E,
        l2_T=l2_T,
        linf_T=linf_T,
    )
    print(f"Saved numerical vs manufactured fields to {npz_path}")

    # Generate comparison plot
    plot_path = args.output_dir / f"mms_comparison_p{args.p}_m{args.m}.png"
    plot_comparison(xc, E_num, T_num, E_exact, T_exact, config, args.t_final, plot_path)


if __name__ == "__main__":
    main()
