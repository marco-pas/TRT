#!/usr/bin/env python3
"""Driver script to run Stiff Multiscale Non-Linear MMS TRT benchmark.

Features extreme temperature contrast (T_cold=10, T_hot=10000), saves numerical vs exact
solutions, plots dual linear/log comparison figures, and conducts spatial convergence studies.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from trt_jfnk.benchmarks.mms_stiff import StiffMMSConfig, StiffMMSProblem1D
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.newton import NewtonOptions, newton_krylov
from trt_jfnk.solvers.scaling import ScalePolicy


def solve_stiff_mms(
    problem: StiffMMSProblem1D,
    dt: float = 1e-4,
    t_final: float = 1e-3,
    preconditioner: str = "schur",
    scaling: str = "state",
    atol: float = 1e-10,
    rtol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve the 1D stiff MMS problem from t=0 to t=t_final with JFNK."""
    n_steps = int(round(t_final / dt))
    state = problem.initial(0.0)
    prec_factory = problem.preconditioner(dt, kind=preconditioner)

    options = NewtonOptions(
        atol=atol,
        rtol=rtol,
        krylov=KrylovOptions(rtol=rtol, atol=atol, max_iterations=60),
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
            ScalePolicy(scaling),
            options,
            problem.admissible,
            prec_factory,
        )
        if not res.converged:
            raise RuntimeError(
                f"Newton solve failed at step {s + 1}, time {time:.4e}: {res.reason}"
            )

        total_krylov += res.total_krylov_iterations
        state = res.state

    E_final, T_final = map(np.asarray, problem.unpack(state))
    return E_final, T_final, total_krylov


def plot_stiff_comparison(
    x: np.ndarray,
    E_num: np.ndarray,
    T_num: np.ndarray,
    E_exact: np.ndarray,
    T_exact: np.ndarray,
    config: StiffMMSConfig,
    t_final: float,
    output_png: Path,
):
    """Generate publication-quality 4-panel comparison plot for stiff MMS benchmark."""
    err_E = np.abs(E_num - E_exact)
    err_T = np.abs(T_num - T_exact)
    rel_E = err_E / np.maximum(E_exact, 1e-30)
    rel_T = err_T / np.maximum(T_exact, 1e-30)
    nx = len(x)

    l2_rel_E = np.linalg.norm(rel_E) / np.sqrt(nx)
    linf_rel_E = np.max(rel_E)
    l2_rel_T = np.linalg.norm(rel_T) / np.sqrt(nx)
    linf_rel_T = np.max(rel_T)

    sigma_val = config.sigma_a0 * (config.get_T_ref() / T_exact) ** config.p
    D_val = config.c / (3.0 * sigma_val)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: Radiation E (Linear + Log scale inset)
    ax1 = axes[0, 0]
    ax1.plot(x, E_exact, "k-", linewidth=2.0, label="Exact Sinc-Exp $E$")
    ax1.plot(
        x,
        E_num,
        "ro",
        markersize=4.5,
        fillstyle="none",
        markeredgewidth=1.4,
        label="Numerical FV JFNK $E$",
    )
    ax1.set_xlabel("Position $x$", fontsize=12)
    ax1.set_ylabel("Radiation Energy Density $E$", fontsize=12)
    ax1.set_title(
        f"Radiation Field $E(x)$ at $t = {t_final:.4e}$", fontsize=13, fontweight="bold"
    )
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    # Inset or log scale for E
    ax1_ins = ax1.inset_axes([0.15, 0.45, 0.35, 0.45])
    ax1_ins.semilogy(x, E_exact, "k-", linewidth=1.5)
    ax1_ins.semilogy(x, E_num, "ro", markersize=3, fillstyle="none")
    ax1_ins.set_title("Log Scale $\\log_{10}(E)$", fontsize=9)
    ax1_ins.grid(True, linestyle=":", alpha=0.4)

    # Panel 2: Material Temperature T
    ax2 = axes[0, 1]
    ax2.plot(x, T_exact, "b-", linewidth=2.0, label="Exact Sinc-Exp $T$")
    ax2.plot(
        x,
        T_num,
        "s",
        color="darkorange",
        markersize=4.5,
        fillstyle="none",
        markeredgewidth=1.4,
        label="Numerical FV JFNK $T$",
    )
    ax2.set_xlabel("Position $x$", fontsize=12)
    ax2.set_ylabel("Material Temperature $T$", fontsize=12)
    ax2.set_title(
        f"Temperature Field $T(x)$ at $t = {t_final:.4e}$", fontsize=13, fontweight="bold"
    )
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    # Inset or log scale for T
    ax2_ins = ax2.inset_axes([0.15, 0.45, 0.35, 0.45])
    ax2_ins.semilogy(x, T_exact, "b-", linewidth=1.5)
    ax2_ins.semilogy(x, T_num, "s", color="darkorange", markersize=3, fillstyle="none")
    ax2_ins.set_title("Log Scale $\\log_{10}(T)$", fontsize=9)
    ax2_ins.grid(True, linestyle=":", alpha=0.4)

    # Panel 3: Relative Spatial Errors
    ax3 = axes[1, 0]
    ax3.semilogy(
        x, rel_E, "r-o", markersize=4, label=f"Rel Error $E$ (Rel $L_2={l2_rel_E:.2e}$)"
    )
    ax3.semilogy(
        x, rel_T, "b-s", markersize=4, label=f"Rel Error $T$ (Rel $L_2={l2_rel_T:.2e}$)"
    )
    ax3.set_xlabel("Position $x$", fontsize=12)
    ax3.set_ylabel("Relative Error $|u - u_{exact}| / u_{exact}$", fontsize=12)
    ax3.set_title("Relative Spatial Discretization Error", fontsize=13, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", frameon=True)

    # Panel 4: Physical Opacity & Diffusion Coefficient across 9 orders of magnitude
    ax4 = axes[1, 1]
    color = "tab:purple"
    ax4.set_xlabel("Position $x$", fontsize=12)
    ax4.set_ylabel("Opacity $\\sigma_a(x)$ [$\\mathrm{cm}^{-1}$]", color=color, fontsize=12)
    l1 = ax4.semilogy(x, sigma_val, color=color, linewidth=2.0, label="Opacity $\\sigma_a(T)$")
    ax4.tick_params(axis="y", labelcolor=color)
    ax4.grid(True, linestyle=":", alpha=0.6)

    ax4_twin = ax4.twinx()
    color2 = "tab:green"
    ax4_twin.set_ylabel("Diffusion Coeff $D(x)$", color=color2, fontsize=12)
    l2 = ax4_twin.semilogy(
        x, D_val, color=color2, linewidth=2.0, linestyle="--", label="Diffusion $D(T)$"
    )
    ax4_twin.tick_params(axis="y", labelcolor=color2)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc="upper right", frameon=True)
    ax4.set_title("Multiscale Optical Depth & Diffusion Span", fontsize=13, fontweight="bold")

    fig.suptitle(
        f"Stiff Multiscale Non-Linear TRT MMS Benchmark ($T_{{cold}}={config.T_cold:.0f}, T_{{hot}}={config.T_hot:.0f}, N_x={nx}$)",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    print(f"Comparison plot saved to: {output_png}")


def run_refinement_study(
    config: StiffMMSConfig,
    dt: float = 1e-5,
    t_final: float = 1e-4,
    nx_list: list[int] | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run spatial mesh refinement study to verify O(Delta x^2) convergence under stiffness."""
    if nx_list is None:
        nx_list = [16, 32, 64, 128, 256]
    if output_dir is None:
        output_dir = Path("results/mms_stiff")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "nx": nx_list,
        "dx": [1.0 / n for n in nx_list],
        "l2_E": [],
        "linf_E": [],
        "l2_T": [],
        "linf_T": [],
        "rel_l2_E": [],
        "rel_l2_T": [],
        "krylov_iters": [],
    }

    print("\n" + "=" * 92)
    print(
        f"RUNNING STIFF MMS SPATIAL REFINEMENT STUDY (T_cold={config.T_cold}, T_hot={config.T_hot})"
    )
    print("=" * 92)
    print(
        f"{'Nx':>6} | {'dx':>10} | {'Rel L2(E)':>12} | {'Rate(E)':>8} | {'Rel L2(T)':>12} | {'Rate(T)':>8} | {'Krylov':>8}"
    )
    print("-" * 92)

    for i, nx in enumerate(nx_list):
        problem = StiffMMSProblem1D(nx=nx, config=config)
        E_num, T_num, total_krylov = solve_stiff_mms(
            problem,
            dt=dt,
            t_final=t_final,
            preconditioner="schur",
            scaling="state",
            atol=1e-10,
            rtol=1e-8,
        )

        state = problem.pack(jnp.asarray(E_num), jnp.asarray(T_num))
        l2_E, linf_E, l2_T, linf_T, rel_E, rel_T = problem.compute_errors(state, t_final)

        results["l2_E"].append(l2_E)
        results["linf_E"].append(linf_E)
        results["l2_T"].append(l2_T)
        results["linf_T"].append(linf_T)
        results["rel_l2_E"].append(rel_E)
        results["rel_l2_T"].append(rel_T)
        results["krylov_iters"].append(total_krylov)

        rate_E_str = "-"
        rate_T_str = "-"
        if i > 0:
            rate_E = np.log2(results["rel_l2_E"][i - 1] / results["rel_l2_E"][i])
            rate_T = np.log2(results["rel_l2_T"][i - 1] / results["rel_l2_T"][i])
            rate_E_str = f"{rate_E:.2f}"
            rate_T_str = f"{rate_T:.2f}"

        print(
            f"{nx:6d} | {1.0 / nx:10.5f} | {rel_E:12.4e} | {rate_E_str:>8} | {rel_T:12.4e} | {rate_T_str:>8} | {total_krylov:8d}"
        )

    print("=" * 92)

    # Save convergence data to NPZ
    npz_path = output_dir / "mms_stiff_convergence.npz"
    np.savez(
        npz_path,
        nx=np.array(results["nx"]),
        dx=np.array(results["dx"]),
        l2_E=np.array(results["l2_E"]),
        linf_E=np.array(results["linf_E"]),
        l2_T=np.array(results["l2_T"]),
        linf_T=np.array(results["linf_T"]),
        rel_l2_E=np.array(results["rel_l2_E"]),
        rel_l2_T=np.array(results["rel_l2_T"]),
        krylov_iters=np.array(results["krylov_iters"]),
        T_cold=config.T_cold,
        T_hot=config.T_hot,
        p=config.p,
        m=config.m,
        dt=dt,
        t_final=t_final,
    )
    print(f"Convergence data saved to: {npz_path}")

    # Plot convergence rates
    plot_convergence(results, output_dir / "mms_stiff_convergence.png")
    return results


def plot_convergence(results: dict, output_png: Path):
    """Plot spatial convergence rates for stiff MMS benchmark."""
    dx = np.array(results["dx"])
    rel_E = np.array(results["rel_l2_E"])
    rel_T = np.array(results["rel_l2_T"])

    fig, ax = plt.subplots(figsize=(8, 6.5))

    ax.loglog(
        dx,
        rel_E,
        "r-o",
        linewidth=2.0,
        markersize=6,
        label=r"Relative $L_2(E)$ Error",
    )
    ax.loglog(
        dx,
        rel_T,
        "b-s",
        linewidth=2.0,
        markersize=6,
        label=r"Relative $L_2(T)$ Error",
    )

    # Reference O(dx^2) line anchored to finest grid point
    ref_dx = np.array([dx[0], dx[-1]])
    ref_rate_2 = rel_T[-1] * (ref_dx / dx[-1]) ** 2
    ax.loglog(
        ref_dx,
        ref_rate_2,
        "k--",
        linewidth=1.8,
        label=r"Theoretical $\mathcal{O}(\Delta x^2)$ Slope",
    )

    ax.set_xlabel(r"Grid Spacing $\Delta x = 1/N_x$", fontsize=12)
    ax.set_ylabel(r"Relative $L_2$ Error Norm", fontsize=12)
    ax.set_title(
        r"Spatial Convergence Verification for Stiff TRT MMS",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=11, frameon=True)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    print(f"Convergence plot saved to: {output_png}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Stiff Multiscale Non-Linear MMS TRT Benchmark"
    )
    parser.add_argument(
        "--T-cold",
        type=float,
        default=10.0,
        help="Cold baseline temperature (default: 10.0)",
    )
    parser.add_argument(
        "--T-hot",
        type=float,
        default=10000.0,
        help="Hot peak temperature (default: 10000.0)",
    )
    parser.add_argument(
        "--nx", type=int, default=64, help="Grid resolution Nx (default: 64)"
    )
    parser.add_argument(
        "--dt", type=float, default=1e-4, help="Time step dt (default: 1e-4)"
    )
    parser.add_argument(
        "--t-final",
        type=float,
        default=1e-3,
        help="Simulation final time (default: 1e-3)",
    )
    parser.add_argument(
        "--preconditioner",
        type=str,
        default="schur",
        choices=["schur", "local-block", "none"],
    )
    parser.add_argument(
        "--scaling",
        type=str,
        default="state",
        choices=["state", "none"],
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mms_stiff",
        help="Output directory for results",
    )
    parser.add_argument(
        "--refinement",
        action="store_true",
        help="Run spatial refinement convergence sweep",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = StiffMMSConfig(T_cold=args.T_cold, T_hot=args.T_hot)

    # 1. Run standard single-resolution benchmark
    print("=" * 80)
    print("SOLVING STIFF MULTISCALE MMS TRT BENCHMARK")
    print(f"Config: T_cold={config.T_cold}, T_hot={config.T_hot}, p={config.p}, m={config.m}")
    print(
        f"Grid: Nx={args.nx}, dx={1.0 / args.nx:.5f}, dt={args.dt:.2e}, t_final={args.t_final:.2e}"
    )
    print(f"Preconditioner: {args.preconditioner}, Scaling: {args.scaling}")
    print("=" * 80)

    problem = StiffMMSProblem1D(nx=args.nx, config=config)
    E_num, T_num, total_krylov = solve_stiff_mms(
        problem,
        dt=args.dt,
        t_final=args.t_final,
        preconditioner=args.preconditioner,
        scaling=args.scaling,
    )

    xc = np.asarray(problem.x)
    E_exact, T_exact = map(np.asarray, problem.exact_solution(args.t_final))

    state = problem.pack(jnp.asarray(E_num), jnp.asarray(T_num))
    l2_E, linf_E, l2_T, linf_T, rel_E, rel_T = problem.compute_errors(state, args.t_final)

    print(f"\nFinal Solution Errors at t = {args.t_final:.4e}:")
    print(f"  Radiation E: Rel L2 = {rel_E:.4e}, Linf = {linf_E:.4e}")
    print(f"  Temperature T: Rel L2 = {rel_T:.4e}, Linf = {linf_T:.4e}")
    print(f"  Total Krylov Iterations: {total_krylov}")

    # Save NPZ archive of fields
    npz_fields = out_dir / "mms_stiff_fields.npz"
    np.savez(
        npz_fields,
        x=xc,
        E_num=E_num,
        T_num=T_num,
        E_exact=E_exact,
        T_exact=T_exact,
        T_cold=config.T_cold,
        T_hot=config.T_hot,
        dt=args.dt,
        t_final=args.t_final,
        nx=args.nx,
        rel_l2_E=rel_E,
        rel_l2_T=rel_T,
        total_krylov=total_krylov,
    )
    print(f"Field solutions saved to: {npz_fields}")

    # Plot comparison figure
    cmp_png = out_dir / "mms_stiff_comparison.png"
    plot_stiff_comparison(
        xc, E_num, T_num, E_exact, T_exact, config, args.t_final, cmp_png
    )

    # Copy plot to brain artifacts directory
    artifact_dir = Path(
        "/afs/pdc.kth.se/home/m/marcopas/.gemini/antigravity/brain/e17bf408-40ca-4f93-97d1-d18ac040e3cc"
    )
    if artifact_dir.exists():
        shutil.copy(cmp_png, artifact_dir / cmp_png.name)
        print(f"Copied {cmp_png.name} to artifacts directory.")

    # 2. Run spatial refinement sweep if requested
    if args.refinement:
        ref_results = run_refinement_study(
            config=config,
            dt=1e-5,
            t_final=1e-4,
            nx_list=[16, 32, 64, 128, 256, 512, 1024],
            output_dir=out_dir,
        )
        conv_png = out_dir / "mms_stiff_convergence.png"
        if artifact_dir.exists() and conv_png.exists():
            shutil.copy(conv_png, artifact_dir / conv_png.name)
            print(f"Copied {conv_png.name} to artifacts directory.")


if __name__ == "__main__":
    main()

