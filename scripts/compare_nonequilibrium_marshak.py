#!/usr/bin/env python3
"""Benchmark 7 JFNK Solver Variants on 1D Two-Temperature Non-Equilibrium Marshak Wave Problem.

Grid: 128 points, 100 time steps.
Compares:
1. FD: finite difference, scaling=none, prec=none
2. AD: automatic differentiation, scaling=none, prec=none
3. S-FD: finite difference, scaling=global/state, prec=none
4. S-AD: automatic differentiation, scaling=global/state, prec=none
5. AD-BD: automatic differentiation, scaling=none, prec=block diagonal
6. S-AD-BD: automatic differentiation, scaling=global/state, prec=block diagonal
7. S-AD-Schur: automatic differentiation, scaling=global/state, prec=Schur-AMG

Measures pure Solver Time inside newton_krylov with asynchronous device synchronization.
Saves 4-panel figures for each method to results/marshak/marshak_comparison_<LABEL>.png
and outputs formatted LaTeX and Markdown comparison tables.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from trt_jfnk.benchmarks.nonequilibrium_marshak import (
    NonequilibriumMarshakConfig,
    NonequilibriumMarshakProblem1D,
)
from trt_jfnk.solvers.jvp import JVPOptions
from trt_jfnk.solvers.krylov import KrylovOptions
from trt_jfnk.solvers.newton import NewtonOptions, newton_krylov
from trt_jfnk.solvers.scaling import ScalePolicy

METHOD_CONFIGS = [
    {
        "label": "FD",
        "jvp": "fd",
        "scale": "none",
        "prec": "none",
        "jvp_desc": "finite difference",
        "scale_desc": "none",
        "prec_desc": "none",
    },
    {
        "label": "AD",
        "jvp": "ad",
        "scale": "none",
        "prec": "none",
        "jvp_desc": "automatic differentiation",
        "scale_desc": "none",
        "prec_desc": "none",
    },
    {
        "label": "S-FD",
        "jvp": "fd",
        "scale": "state",
        "prec": "none",
        "jvp_desc": "finite difference",
        "scale_desc": "global",
        "prec_desc": "none",
    },
    {
        "label": "S-AD",
        "jvp": "ad",
        "scale": "state",
        "prec": "none",
        "jvp_desc": "automatic differentiation",
        "scale_desc": "global/local",
        "prec_desc": "none",
    },
    {
        "label": "AD-BD",
        "jvp": "ad",
        "scale": "none",
        "prec": "local-block",
        "jvp_desc": "automatic differentiation",
        "scale_desc": "none",
        "prec_desc": "block diagonal",
    },
    {
        "label": "S-AD-BD",
        "jvp": "ad",
        "scale": "state",
        "prec": "local-block",
        "jvp_desc": "automatic differentiation",
        "scale_desc": "global/local",
        "prec_desc": "block diagonal",
    },
    {
        "label": "S-AD-Schur",
        "jvp": "ad",
        "scale": "state",
        "prec": "schur",
        "jvp_desc": "automatic differentiation",
        "scale_desc": "global/local",
        "prec_desc": "Schur-AMG",
    },
]


def run_single_method(
    problem: NonequilibriumMarshakProblem1D,
    config: dict,
    steps: int,
    dt: float,
    newton_rtol: float = 1e-8,
    newton_atol: float = 1e-10,
    krylov_max_iter: int = 60,
) -> dict:
    """Run simulation for a single solver configuration and record pure Solver Time."""
    options = NewtonOptions(
        atol=newton_atol,
        rtol=newton_rtol,
        max_iterations=30,
        jvp=JVPOptions(mode=config["jvp"]),
        krylov=KrylovOptions(
            rtol=newton_rtol,
            atol=newton_atol,
            max_iterations=krylov_max_iter,
        ),
    )
    policy = ScalePolicy(
        config["scale"],
        (problem.config.cold**4, problem.config.cold),
    )
    prec_factory = problem.preconditioner(dt, kind=config["prec"])

    state = problem.initial()
    total_krylov_iters = 0
    total_newton_iters = 0
    successful_steps = 0
    solver_time = 0.0
    solver_cpu_time = 0.0
    status_msg = "Converged"

    for step in range(steps):
        time_val = (step + 1) * dt
        old_state = state

        # Strictly time newton_krylov solver execution
        t0 = time.perf_counter()
        c0 = time.process_time()

        res = newton_krylov(
            lambda u: problem.residual(u, old_state, dt, time_val),
            old_state,
            problem.size,
            policy,
            options,
            problem.admissible,
            prec_factory,
        )
        res.state.block_until_ready()

        t1 = time.perf_counter()
        c1 = time.process_time()

        solver_time += t1 - t0
        solver_cpu_time += c1 - c0

        total_krylov_iters += res.total_krylov_iterations
        total_newton_iters += res.iterations

        if not res.converged:
            status_msg = f"Stalled at step {step + 1} ({res.reason})"
            print(f"      [{config['label']}] {status_msg}")
            state = res.state
            break

        state = res.state
        successful_steps += 1

    avg_krylov = (
        total_krylov_iters / max(1, successful_steps)
        if successful_steps > 0
        else float("nan")
    )
    avg_newton = (
        total_newton_iters / max(1, successful_steps)
        if successful_steps > 0
        else float("nan")
    )

    E_final, T_final = problem.unpack(state)

    return {
        "label": config["label"],
        "jvp_desc": config["jvp_desc"],
        "scale_desc": config["scale_desc"],
        "prec_desc": config["prec_desc"],
        "status": status_msg,
        "converged": (successful_steps == steps),
        "steps": f"{successful_steps}/{steps}",
        "successful_steps": successful_steps,
        "solver_time": solver_time,
        "solver_cpu_time": solver_cpu_time,
        "avg_krylov": avg_krylov,
        "avg_newton": avg_newton,
        "E_final": np.asarray(E_final),
        "T_final": np.asarray(T_final),
    }


def plot_method_results(
    problem: NonequilibriumMarshakProblem1D,
    res: dict,
    ref_res: dict,
    out_path: Path,
    time_final: float,
) -> None:
    """Save 4-panel comparison figure showing E, T, and Non-Equilibrium departure."""
    x = np.asarray(problem.x)
    E_num = res["E_final"]
    T_num = res["T_final"]
    E_ref = ref_res["E_final"]
    T_ref = ref_res["T_final"]
    a = problem.config.a

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: Radiation Energy Density E(x)
    ax1 = axes[0, 0]
    ax1.plot(x, E_ref, "k--", lw=2.5, label="Reference (S-AD-Schur)")
    ax1.plot(x, E_num, "C0-", lw=1.8, label=f"Method: {res['label']}")
    ax1.axhline(a * (problem.config.bath**4), color="red", ls=":", alpha=0.7, label=r"$a T_{\rm bath}^4 = 1.0$")
    ax1.axhline(a * (problem.config.cold**4), color="blue", ls=":", alpha=0.7, label=r"$a T_{\rm cold}^4 = 0.0016$")
    ax1.set_title(r"Radiation Energy Density $E(x, t_{\rm final})$", fontsize=12, fontweight="bold")
    ax1.set_xlabel(r"$x$", fontsize=11)
    ax1.set_ylabel(r"$E(x)$", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=10)

    # Panel 2: Material Temperature T(x)
    ax2 = axes[0, 1]
    ax2.plot(x, T_ref, "k--", lw=2.5, label="Reference (S-AD-Schur)")
    ax2.plot(x, T_num, "C1-", lw=1.8, label=f"Method: {res['label']}")
    ax2.axhline(problem.config.bath, color="red", ls=":", alpha=0.7, label=r"$T_{\rm bath} = 1.0$")
    ax2.axhline(problem.config.cold, color="blue", ls=":", alpha=0.7, label=r"$T_{\rm cold} = 0.2$")
    ax2.set_title(r"Material Temperature $T(x, t_{\rm final})$", fontsize=12, fontweight="bold")
    ax2.set_xlabel(r"$x$", fontsize=11)
    ax2.set_ylabel(r"$T(x)$", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=10)

    # Panel 3: Non-Equilibrium Wavefront: E(x) vs a T^4(x)
    ax3 = axes[1, 0]
    ax3.plot(x, E_num, "C0-", lw=2.0, label=r"Radiation $E(x)$")
    ax3.plot(x, a * (T_num**4), "C3--", lw=2.0, label=r"Equilibrium Emission $a T^4(x)$")
    ax3.fill_between(x, a * (T_num**4), E_num, where=(E_num >= a * (T_num**4)), color="orange", alpha=0.25, label=r"Non-Equilibrium Gap ($E > a T^4$)")
    ax3.set_title(r"Non-Equilibrium Wavefront: $E(x)$ vs $a T^4(x)$", fontsize=12, fontweight="bold")
    ax3.set_xlabel(r"$x$", fontsize=11)
    ax3.set_ylabel(r"Energy / Emission Density", fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", fontsize=10)

    # Panel 4: Absolute Non-Equilibrium Departure |E(x) - a T^4(x)|
    ax4 = axes[1, 1]
    departure = np.abs(E_num - a * (T_num**4))
    ax4.semilogy(x, np.maximum(departure, 1e-12), "m-", lw=2.0, label=r"$|E(x) - a T^4(x)|$")
    ax4.set_title(r"Non-Equilibrium Departure $|E(x) - a T^4(x)|$", fontsize=12, fontweight="bold")
    ax4.set_xlabel(r"$x$", fontsize=11)
    ax4.set_ylabel(r"Log Departure", fontsize=11)
    ax4.grid(True, which="both", alpha=0.3)
    ax4.legend(loc="upper right", fontsize=10)

    fig.suptitle(
        f"Non-Equilibrium Marshak Wave: Method [{res['label']}]\n"
        f"Grid $N_x={problem.nx}$, $t={time_final:.4f}$ (100 steps, $\\Delta t={time_final/100:.4f}$), "
        f"Status: {res['status']}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=128, help="Number of spatial cells (default: 128)")
    parser.add_argument("--steps", type=int, default=100, help="Number of time steps (default: 100)")
    parser.add_argument("--dt", type=float, default=0.001, help="Time step size (default: 0.001)")
    parser.add_argument("--cold", type=float, default=0.2, help="Initial cold medium temperature (default: 0.2)")
    parser.add_argument("--bath", type=float, default=1.0, help="Incoming boundary bath temperature (default: 1.0)")
    parser.add_argument("--sigma0", type=float, default=1.0, help="Base opacity sigma0 (default: 1.0)")
    parser.add_argument("--exponent", type=float, default=3.0, help="Opacity exponent (default: 3.0)")
    parser.add_argument("--output-dir", type=str, default="results/marshak", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/afs/pdc.kth.se/home/m/marcopas/.gemini/antigravity/brain/e17bf408-40ca-4f93-97d1-d18ac040e3cc")

    cfg = NonequilibriumMarshakConfig(
        cold=args.cold,
        bath=args.bath,
        sigma0=args.sigma0,
        exponent=args.exponent,
    )
    problem = NonequilibriumMarshakProblem1D(nx=args.nx, config=cfg)
    t_final = args.steps * args.dt

    print("=" * 100)
    print("1D TWO-TEMPERATURE NON-EQUILIBRIUM MARSHAK BENCHMARK (128 POINTS, 100 ITERATIONS)")
    print(f"Grid: Nx={args.nx}, dt={args.dt:.4e}, Total steps={args.steps}, t_final={t_final:.4f}")
    print(f"Parameters: T_cold={cfg.cold}, T_bath={cfg.bath}, sigma0={cfg.sigma0}, opacity_exponent={cfg.exponent}")
    print("=" * 100)

    # First, run S-AD-Schur to act as the baseline reference solution
    print("\n---> Running [S-AD-Schur] (Reference & Variant): JVP=ad, Scaling=state, Prec=schur")
    schur_cfg = [c for c in METHOD_CONFIGS if c["label"] == "S-AD-Schur"][0]
    ref_res = run_single_method(problem, schur_cfg, args.steps, args.dt)
    print(
        f"      Status: {ref_res['status']} | Steps: {ref_res['steps']} | "
        f"Solver Time: {ref_res['solver_time']:.2f}s (CPU: {ref_res['solver_cpu_time']:.2f}s) | "
        f"Krylov/step: {ref_res['avg_krylov']:.1f}"
    )

    all_results = []

    for m_cfg in METHOD_CONFIGS:
        lbl = m_cfg["label"]
        if lbl == "S-AD-Schur":
            res = ref_res.copy()
            res["err_E"] = 0.0
            res["err_T"] = 0.0
        else:
            print(f"\n---> Running [{lbl}]: JVP={m_cfg['jvp']}, Scaling={m_cfg['scale']}, Prec={m_cfg['prec']}")
            res = run_single_method(problem, m_cfg, args.steps, args.dt)
            err_E = np.linalg.norm(res["E_final"] - ref_res["E_final"]) / (np.linalg.norm(ref_res["E_final"]) + 1e-14)
            err_T = np.linalg.norm(res["T_final"] - ref_res["T_final"]) / (np.linalg.norm(ref_res["T_final"]) + 1e-14)
            res["err_E"] = err_E
            res["err_T"] = err_T
            print(
                f"      Status: {res['status']} | Steps: {res['steps']} | "
                f"Solver Time: {res['solver_time']:.2f}s (CPU: {res['solver_cpu_time']:.2f}s) | "
                f"Krylov/step: {res['avg_krylov']:.1f}"
            )

        all_results.append(res)

        plot_path = out_dir / f"marshak_comparison_{lbl}.png"
        plot_method_results(problem, res, ref_res, plot_path, t_final)
        print(f"Saved plot: {plot_path}")

        # Copy to artifact directory
        art_dest = artifact_dir / f"marshak_comparison_{lbl}.png"
        try:
            shutil.copy(plot_path, art_dest)
            print(f"Copied {plot_path.name} to artifacts directory.")
        except Exception as e:
            print(f"Could not copy to artifacts: {e}")

    # Print LaTeX Table
    print("\n" + "=" * 100)
    print("LATEX TABLE OF RESULTS (NON-EQUILIBRIUM MARSHAK WAVE, 128 POINTS, 100 STEPS):")
    print("=" * 100)
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\begin{tabular}{llllcccc}",
        r"\toprule",
        r"Label & JVP & Scale policy & Physics preconditioner & Steps & Solver Time (s) & Avg GMRES & Rel $L_2(T)$ \\",
        r"\midrule",
    ]
    for r in all_results:
        latex_lines.append(
            f"{r['label']} & {r['jvp_desc']} & {r['scale_desc']} & {r['prec_desc']} & "
            f"{r['steps']} & {r['solver_time']:.2f} & {r['avg_krylov']:.1f} & {r['err_T']:.2e} \\\\"
        )
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Performance and accuracy comparison of JFNK solver variants on 1D Non-Equilibrium Marshak Wave TRT ($N_x=128$, 100 steps). Solver Time measures cumulative time spent in nonlinear Newton-Krylov solver.}",
        r"\label{tab:marshak_comparison}",
        r"\end{table}",
    ])
    print("\n".join(latex_lines))

    # Print Markdown Table
    print("\n" + "=" * 100)
    print("MARKDOWN TABLE OF RESULTS (NON-EQUILIBRIUM MARSHAK WAVE, 128 POINTS, 100 STEPS):")
    print("=" * 100)
    md_lines = [
        "| Label | JVP | Scale policy | Physics preconditioner | Status | Steps | Solver Time (s) | CPU Time (s) | Avg GMRES/Step | Rel $L_2(E)$ | Rel $L_2(T)$ |",
        "| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in all_results:
        stat = "Converged" if r["converged"] else "Stalled"
        md_lines.append(
            f"| **{r['label']}** | {r['jvp_desc']} | {r['scale_desc']} | {r['prec_desc']} | "
            f"{stat} | {r['steps']} | {r['solver_time']:.2f} | {r['solver_cpu_time']:.2f} | "
            f"{r['avg_krylov']:.1f} | {r['err_E']:.3e} | {r['err_T']:.3e} |"
        )
    print("\n".join(md_lines))
    print("=" * 100)

    # Save summary data
    np.savez(
        out_dir / "method_comparison_results.npz",
        labels=[r["label"] for r in all_results],
        solver_times=[r["solver_time"] for r in all_results],
        avg_krylov=[r["avg_krylov"] for r in all_results],
        err_T=[r["err_T"] for r in all_results],
        err_E=[r["err_E"] for r in all_results],
    )
    print(f"\nAll comparison results saved to: {out_dir / 'method_comparison_results.npz'}")


if __name__ == "__main__":
    main()
