#!/usr/bin/env python3
"""Benchmark comparison of solver variants across JVP, Scaling, and Preconditioning for Stiff TRT MMS.

Evaluates 7 solver configurations on the multiscale stiff MMS problem (128 grid points, 100 iterations/steps):
1. FD:         JVP=finite-difference, Scale=none,  Prec=none
2. AD:         JVP=automatic-diff,    Scale=none,  Prec=none
3. S-FD:       JVP=finite-difference, Scale=state, Prec=none
4. S-AD:       JVP=automatic-diff,    Scale=state, Prec=none
5. AD-BD:      JVP=automatic-diff,    Scale=none,  Prec=block-diagonal (local-block)
6. S-AD-BD:    JVP=automatic-diff,    Scale=state, Prec=block-diagonal (local-block)
7. S-AD-Schur: JVP=automatic-diff,    Scale=state, Prec=Schur-complement (Schur-AMG)

Measures wall-clock time, iteration counts, relative errors, and generates 4-panel comparison plots
named `mms_stiff_comparison_<LABEL>.png` for each configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import time

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from trt_jfnk.benchmarks.mms_stiff import StiffMMSConfig, StiffMMSProblem1D
from trt_jfnk.solvers.jvp import JVPOptions
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.newton import NewtonOptions, newton_krylov
from trt_jfnk.solvers.scaling import ScalePolicy

METHOD_SPECS = [
    {
        "label": "FD",
        "jvp_mode": "fd",
        "scale_mode": "none",
        "prec_kind": "none",
        "table_jvp": "finite difference",
        "table_scale": "none",
        "table_prec": "none",
        "desc": "Unscaled FD JVP, unpreconditioned",
    },
    {
        "label": "AD",
        "jvp_mode": "ad",
        "scale_mode": "none",
        "prec_kind": "none",
        "table_jvp": "automatic differentiation",
        "table_scale": "none",
        "table_prec": "none",
        "desc": "Unscaled AD JVP, unpreconditioned",
    },
    {
        "label": "S-FD",
        "jvp_mode": "fd",
        "scale_mode": "state",
        "prec_kind": "none",
        "table_jvp": "finite difference",
        "table_scale": "global",
        "table_prec": "none",
        "desc": "Scaled FD JVP, unpreconditioned",
    },
    {
        "label": "S-AD",
        "jvp_mode": "ad",
        "scale_mode": "state",
        "prec_kind": "none",
        "table_jvp": "automatic differentiation",
        "table_scale": "global/local",
        "table_prec": "none",
        "desc": "Scaled AD JVP, unpreconditioned",
    },
    {
        "label": "AD-BD",
        "jvp_mode": "ad",
        "scale_mode": "none",
        "prec_kind": "local-block",
        "table_jvp": "automatic differentiation",
        "table_scale": "none",
        "table_prec": "block diagonal",
        "desc": "Unscaled AD JVP, block-diagonal preconditioner",
    },
    {
        "label": "S-AD-BD",
        "jvp_mode": "ad",
        "scale_mode": "state",
        "prec_kind": "local-block",
        "table_jvp": "automatic differentiation",
        "table_scale": "global/local",
        "table_prec": "block diagonal",
        "desc": "Scaled AD JVP, block-diagonal preconditioner",
    },
    {
        "label": "S-AD-Schur",
        "jvp_mode": "ad",
        "scale_mode": "state",
        "prec_kind": "schur",
        "table_jvp": "automatic differentiation",
        "table_scale": "global/local",
        "table_prec": "Schur-AMG",
        "desc": "Scaled AD JVP, Schur-complement preconditioner",
    },
]


def plot_method_comparison(
    x: np.ndarray,
    E_num: np.ndarray,
    T_num: np.ndarray,
    E_exact: np.ndarray,
    T_exact: np.ndarray,
    config: StiffMMSConfig,
    t_eval: float,
    label: str,
    status: str,
    rel_l2_E: float,
    rel_l2_T: float,
    output_png: Path,
):
    """Generate 4-panel comparison plot for a specific solver method."""
    err_E = np.abs(E_num - E_exact)
    err_T = np.abs(T_num - T_exact)
    rel_E = err_E / np.maximum(E_exact, 1e-30)
    rel_T = err_T / np.maximum(T_exact, 1e-30)

    sigma_val = config.sigma_a0 * (config.get_T_ref() / np.maximum(T_exact, 1e-30)) ** config.p
    D_val = config.c / (3.0 * np.maximum(sigma_val, 1e-30))

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: Radiation E
    ax1 = axes[0, 0]
    ax1.plot(x, E_exact, "k-", linewidth=2.0, label="Exact Sinc-Exp $E$")
    ax1.plot(
        x,
        E_num,
        "ro",
        markersize=3.5,
        fillstyle="none",
        markeredgewidth=1.3,
        label=f"Numerical $E$ ({label})",
    )
    ax1.set_xlabel("Position $x$", fontsize=12)
    ax1.set_ylabel("Radiation Energy Density $E$", fontsize=12)
    ax1.set_title(f"Radiation Field $E(x)$ at $t = {t_eval:.4e}$", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True)

    ax1_ins = ax1.inset_axes([0.15, 0.45, 0.35, 0.45])
    ax1_ins.semilogy(x, E_exact, "k-", linewidth=1.5)
    ax1_ins.semilogy(x, np.maximum(E_num, 1e-30), "ro", markersize=2.5, fillstyle="none")
    ax1_ins.set_title("Log Scale $\\log_{10}(E)$", fontsize=9)
    ax1_ins.grid(True, linestyle=":", alpha=0.4)

    # Panel 2: Temperature T
    ax2 = axes[0, 1]
    ax2.plot(x, T_exact, "b-", linewidth=2.0, label="Exact Sinc-Exp $T$")
    ax2.plot(
        x,
        T_num,
        "s",
        color="darkorange",
        markersize=3.5,
        fillstyle="none",
        markeredgewidth=1.3,
        label=f"Numerical $T$ ({label})",
    )
    ax2.set_xlabel("Position $x$", fontsize=12)
    ax2.set_ylabel("Material Temperature $T$", fontsize=12)
    ax2.set_title(f"Temperature Field $T(x)$ at $t = {t_eval:.4e}$", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True)

    ax2_ins = ax2.inset_axes([0.15, 0.45, 0.35, 0.45])
    ax2_ins.semilogy(x, T_exact, "b-", linewidth=1.5)
    ax2_ins.semilogy(x, np.maximum(T_num, 1e-30), "s", color="darkorange", markersize=2.5, fillstyle="none")
    ax2_ins.set_title("Log Scale $\\log_{10}(T)$", fontsize=9)
    ax2_ins.grid(True, linestyle=":", alpha=0.4)

    # Panel 3: Relative Spatial Errors
    ax3 = axes[1, 0]
    ax3.semilogy(x, np.maximum(rel_E, 1e-16), "r-o", markersize=3, label=f"Rel Error $E$ ($L_2={rel_l2_E:.2e}$)")
    ax3.semilogy(x, np.maximum(rel_T, 1e-16), "b-s", markersize=3, label=f"Rel Error $T$ ($L_2={rel_l2_T:.2e}$)")
    ax3.set_xlabel("Position $x$", fontsize=12)
    ax3.set_ylabel("Relative Error $|u - u_{exact}| / u_{exact}$", fontsize=12)
    ax3.set_title(f"Relative Error Distribution [{status}]", fontsize=13, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", frameon=True)

    # Panel 4: Physical Opacity & Diffusion Coefficient
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
    l2 = ax4_twin.semilogy(x, D_val, color=color2, linewidth=2.0, linestyle="--", label="Diffusion $D(T)$")
    ax4_twin.tick_params(axis="y", labelcolor=color2)

    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc="upper right", frameon=True)
    ax4.set_title("Multiscale Optical Depth & Diffusion Span (9 Orders)", fontsize=13, fontweight="bold")

    status_tag = "CONVERGED" if status == "Converged" else f"STALLED / FAILED ({status})"
    fig.suptitle(
        f"Stiff Non-Linear TRT MMS Benchmark: [{label}] ({status_tag})\n"
        f"$N_x = {len(x)}, \\quad \\Delta t = 10^{{-5}}, \\quad T_{{\\text{{cold}}}} = {config.T_cold:.0f}, \\quad T_{{\\text{{hot}}}} = {config.T_hot:.0f}$",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {output_png}")


def run_single_method(
    spec: dict,
    problem: StiffMMSProblem1D,
    n_steps: int = 100,
    dt: float = 1e-5,
) -> dict:
    """Run up to n_steps iterations for a given solver variant and measure time and accuracy."""
    label = spec["label"]
    jvp_mode = spec["jvp_mode"]
    scale_mode = spec["scale_mode"]
    prec_kind = spec["prec_kind"]

    print(f"\n---> Running [{label}]: JVP={jvp_mode}, Scaling={scale_mode}, Prec={prec_kind}")

    prec_factory = problem.preconditioner(dt, kind=prec_kind)
    options = NewtonOptions(
        atol=1e-10,
        rtol=1e-8,
        jvp=JVPOptions(mode=jvp_mode),
        krylov=KrylovOptions(rtol=1e-8, atol=1e-10, max_iterations=60),
    )

    state = problem.initial(0.0)
    total_krylov = 0
    total_newton = 0
    steps_completed = 0
    status = "Converged"
    total_solver_time = 0.0
    total_solver_cpu = 0.0

    for s in range(n_steps):
        t_curr = (s + 1) * dt
        old_state = state

        def residual_fn(u):
            return problem.residual(u, old_state, dt, t_curr)

        t_solve_start = time.perf_counter()
        t_cpu_start = time.process_time()
        try:
            res = newton_krylov(
                residual_fn,
                old_state,
                problem.size,
                ScalePolicy(scale_mode),
                options,
                problem.admissible,
                prec_factory,
            )
            res.state.block_until_ready()
            solve_elapsed = time.perf_counter() - t_solve_start
            cpu_elapsed = time.process_time() - t_cpu_start
            total_solver_time += solve_elapsed
            total_solver_cpu += cpu_elapsed

            if not res.converged:
                status = f"Stalled at step {s + 1} ({res.reason})"
                total_krylov += res.total_krylov_iterations
                total_newton += res.iterations
                state = res.state
                steps_completed = s + 1
                print(f"      [{label}] {status}")
                break

            total_krylov += res.total_krylov_iterations
            total_newton += res.iterations
            state = res.state
            steps_completed = s + 1

        except Exception as exc:
            status = f"Failed at step {s + 1}: {exc}"
            steps_completed = s
            print(f"      [{label}] {status}")
            break

    final_time = steps_completed * dt if steps_completed > 0 else dt

    # Evaluate accuracy
    l2_E, linf_E, l2_T, linf_T, rel_E, rel_T = problem.compute_errors(state, final_time)
    avg_krylov = total_krylov / max(1, steps_completed)
    avg_newton = total_newton / max(1, steps_completed)

    print(f"      Status: {status} | Steps: {steps_completed}/{n_steps} | Solver Time: {total_solver_time:.2f}s (CPU: {total_solver_cpu:.2f}s) | Krylov/step: {avg_krylov:.1f} | Rel L2(E): {rel_E:.3e} | Rel L2(T): {rel_T:.3e}")

    return {
        "spec": spec,
        "label": label,
        "status": status,
        "steps_completed": steps_completed,
        "total_steps": n_steps,
        "solver_time": total_solver_time,
        "solver_cpu_time": total_solver_cpu,
        "total_krylov": total_krylov,
        "avg_krylov": avg_krylov,
        "avg_newton": avg_newton,
        "rel_l2_E": rel_E,
        "linf_E": linf_E,
        "rel_l2_T": rel_T,
        "linf_T": linf_T,
        "final_state": state,
        "final_time": final_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark solver variants for Stiff TRT MMS")
    parser.add_argument("--nx", type=int, default=128, help="Number of spatial points (default: 128)")
    parser.add_argument("--steps", type=int, default=100, help="Number of iterations/steps (default: 100)")
    parser.add_argument("--dt", type=float, default=1e-5, help="Time step (default: 1e-5)")
    parser.add_argument("--output-dir", type=str, default="results/mms_stiff", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/afs/pdc.kth.se/home/m/marcopas/.gemini/antigravity/brain/e17bf408-40ca-4f93-97d1-d18ac040e3cc")

    config = StiffMMSConfig(T_cold=10.0, T_hot=10000.0)
    problem = StiffMMSProblem1D(nx=args.nx, config=config)

    print("=" * 100)
    print("STIFF TRT MMS SOLVER COMPARISON BENCHMARK (128 POINTS, 100 ITERATIONS)")
    print(f"Grid: Nx={args.nx}, dt={args.dt:.2e}, Total steps={args.steps}")
    print(f"Physics: T_cold={config.T_cold}, T_hot={config.T_hot}, p={config.p}, m={config.m}")
    print("=" * 100)

    results = []
    xc = np.asarray(problem.x)

    for spec in METHOD_SPECS:
        res = run_single_method(spec, problem, n_steps=args.steps, dt=args.dt)
        results.append(res)

        # Plot for each method named mms_stiff_comparison_<LABEL>.png
        E_num, T_num = map(np.asarray, problem.unpack(res["final_state"]))
        E_exact, T_exact = map(np.asarray, problem.exact_solution(res["final_time"]))

        png_name = f"mms_stiff_comparison_{spec['label']}.png"
        plot_path = out_dir / png_name
        plot_method_comparison(
            x=xc,
            E_num=E_num,
            T_num=T_num,
            E_exact=E_exact,
            T_exact=T_exact,
            config=config,
            t_eval=res["final_time"],
            label=spec["label"],
            status=res["status"],
            rel_l2_E=res["rel_l2_E"],
            rel_l2_T=res["rel_l2_T"],
            output_png=plot_path,
        )

        if artifact_dir.exists():
            shutil.copy(plot_path, artifact_dir / png_name)
            print(f"Copied {png_name} to artifacts directory.")

    # Print LaTeX Table
    print("\n" + "=" * 100)
    print("LATEX TABLE OF RESULTS:")
    print("=" * 100)
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\begin{tabular}{llllcccc}")
    print(r"\toprule")
    print(r"Label & JVP & Scale policy & Physics preconditioner & Steps & Solver Time (s) & Avg GMRES & Rel $L_2(T)$ \\")
    print(r"\midrule")
    for r in results:
        sp = r["spec"]
        time_str = f"{r['solver_time']:.2f}"
        gmres_str = f"{r['avg_krylov']:.1f}"
        err_str = f"{r['rel_l2_T']:.2e}"
        steps_str = f"{r['steps_completed']}/{r['total_steps']}"
        print(f"{sp['label']} & {sp['table_jvp']} & {sp['table_scale']} & {sp['table_prec']} & {steps_str} & {time_str} & {gmres_str} & {err_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Performance and accuracy comparison of JFNK solver variants on Stiff TRT MMS (128 points, 100 steps, $T \in [10, 10000]$). Solver Time measures cumulative time spent in nonlinear Newton-Krylov solver.}")
    print(r"\label{tab:stiff_mms_comparison}")
    print(r"\end{table}")

    # Print Markdown Table
    print("\n" + "=" * 100)
    print("MARKDOWN TABLE OF RESULTS:")
    print("=" * 100)
    print("| Label | JVP | Scale policy | Physics preconditioner | Status | Steps | Solver Time (s) | CPU Time (s) | Avg GMRES/Step | Rel $L_2(E)$ | Rel $L_2(T)$ |")
    print("| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        sp = r["spec"]
        status_disp = "Converged" if r["status"] == "Converged" else "Stalled"
        print(f"| **{sp['label']}** | {sp['table_jvp']} | {sp['table_scale']} | {sp['table_prec']} | {status_disp} | {r['steps_completed']}/{r['total_steps']} | {r['solver_time']:.2f} | {r['solver_cpu_time']:.2f} | {r['avg_krylov']:.1f} | {r['rel_l2_E']:.3e} | {r['rel_l2_T']:.3e} |")
    print("=" * 100)

    # Save summary NPZ
    npz_summary = out_dir / "method_comparison_results.npz"
    np.savez(
        npz_summary,
        labels=np.array([r["label"] for r in results]),
        solver_times=np.array([r["solver_time"] for r in results]),
        solver_cpu_times=np.array([r["solver_cpu_time"] for r in results]),
        avg_krylov=np.array([r["avg_krylov"] for r in results]),
        rel_l2_E=np.array([r["rel_l2_E"] for r in results]),
        rel_l2_T=np.array([r["rel_l2_T"] for r in results]),
        steps_completed=np.array([r["steps_completed"] for r in results]),
    )
    print(f"\nAll comparison results saved to: {npz_summary}")


if __name__ == "__main__":
    main()

