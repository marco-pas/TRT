#!/usr/bin/env python3
"""Benchmark comparison of solver variants across JVP, Scaling, and Preconditioning for Perturbed Su-Olson TRT.

Evaluates 7 solver configurations on the non-linear Perturbed Su-Olson problem (m = 3 + delta, delta = 0.01):
1. FD:         JVP=finite-difference, Scale=none,  Prec=none
2. AD:         JVP=automatic-diff,    Scale=none,  Prec=none
3. S-FD:       JVP=finite-difference, Scale=state, Prec=none
4. S-AD:       JVP=automatic-diff,    Scale=state, Prec=none
5. AD-BD:      JVP=automatic-diff,    Scale=none,  Prec=block-diagonal (local-block)
6. S-AD-BD:    JVP=automatic-diff,    Scale=state, Prec=block-diagonal (local-block)
7. S-AD-Schur: JVP=automatic-diff,    Scale=state, Prec=Schur-complement (Schur-AMG)

Compares the non-linear perturbed fields (m=3.01) against the linear unperturbed Su-Olson base state (m=3.00),
measures pure solver time (time inside newton_krylov), CPU time, Krylov iterations, and generates 4-panel
comparison plots named `su_olson_perturbed_comparison_<LABEL>.png`.
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

from trt_jfnk.benchmarks.perturbed_su_olson import (
    PerturbedSuOlsonConfig,
    PerturbedSuOlsonProblem1D,
)
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


def plot_perturbed_su_olson(
    x: np.ndarray,
    E_pert: np.ndarray,
    T_pert: np.ndarray,
    E_base: np.ndarray,
    T_base: np.ndarray,
    config: PerturbedSuOlsonConfig,
    t_eval: float,
    label: str,
    status: str,
    output_png: Path,
):
    """Generate 4-panel publication-quality comparison plot for Perturbed Su-Olson benchmark."""
    diff_E = E_pert - E_base
    diff_T = T_pert - T_base
    sensitivity_T = diff_T / config.delta
    sensitivity_E = diff_E / config.delta

    cv_pert = config.heat_capacity(T_pert, config.m)
    cv_base = config.heat_capacity(T_base, config.m_base)
    diff_cv = cv_pert - cv_base

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Panel 1: Radiation E (Perturbed vs Base)
    ax1 = axes[0, 0]
    ax1.plot(x, E_base, "k--", linewidth=2.0, label="Base Linear ($m=3.0$)")
    ax1.plot(x, E_pert, "r-", linewidth=2.0, label=f"Perturbed ($m={config.m:.2f}$, {label})")
    ax1.set_xlabel("Position $x$", fontsize=12)
    ax1.set_ylabel("Radiation Energy Density $E$", fontsize=12)
    ax1.set_title(f"Radiation Field $E(x)$ at $t = {t_eval:.4f}$", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, fontsize=11)

    # Panel 2: Material Temperature T (Perturbed vs Base)
    ax2 = axes[0, 1]
    ax2.plot(x, T_base, "k--", linewidth=2.0, label="Base Linear ($m=3.0$)")
    ax2.plot(x, T_pert, "b-", linewidth=2.0, label=f"Perturbed ($m={config.m:.2f}$, {label})")
    ax2.set_xlabel("Position $x$", fontsize=12)
    ax2.set_ylabel("Material Temperature $T$", fontsize=12)
    ax2.set_title(f"Temperature Field $T(x)$ at $t = {t_eval:.4f}$", fontsize=13, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, fontsize=11)

    # Panel 3: Perturbation Sensitivities Delta T / delta and Delta E / delta
    ax3 = axes[1, 0]
    l1 = ax3.plot(x, sensitivity_T, "b-", linewidth=2.0, label=r"Temperature Sensitivity $\frac{\partial T}{\partial \delta} \approx \frac{\Delta T}{\delta}$")
    ax3.set_xlabel("Position $x$", fontsize=12)
    ax3.set_ylabel(r"Temperature Sensitivity $\Delta T / \delta$", color="b", fontsize=12)
    ax3.tick_params(axis="y", labelcolor="b")
    ax3.grid(True, linestyle="--", alpha=0.6)

    ax3_twin = ax3.twinx()
    l2 = ax3_twin.plot(x, sensitivity_E, "r--", linewidth=1.8, label=r"Radiation Sensitivity $\frac{\partial E}{\partial \delta} \approx \frac{\Delta E}{\delta}$")
    ax3_twin.set_ylabel(r"Radiation Sensitivity $\Delta E / \delta$", color="r", fontsize=12)
    ax3_twin.tick_params(axis="y", labelcolor="r")

    lines = l1 + l2
    labels = [line.get_label() for line in lines]
    ax3.legend(lines, labels, loc="upper right", frameon=True, fontsize=10)
    ax3.set_title(r"Perturbation Sensitivities ($\delta = 0.01$)", fontsize=13, fontweight="bold")

    # Panel 4: Material Heat Capacity and Delta cv
    ax4 = axes[1, 1]
    ax4.plot(x, cv_base, "k--", linewidth=1.8, label=r"Base $c_v(T) = c_{v0} + 4 T^3$")
    ax4.plot(x, cv_pert, "g-", linewidth=2.0, label=rf"Perturbed $c_v(T) = c_{{v0}} + 4 T^{{{config.m:.2f}}}$")
    ax4.set_xlabel("Position $x$", fontsize=12)
    ax4.set_ylabel(r"Heat Capacity $c_v(T)$", fontsize=12)
    ax4.grid(True, linestyle="--", alpha=0.6)
    ax4.legend(loc="upper right", frameon=True, fontsize=10)
    ax4.set_title(r"Material Heat Capacity Modification", fontsize=13, fontweight="bold")

    status_tag = "CONVERGED" if status == "Converged" else f"STALLED / FAILED ({status})"
    plt.suptitle(
        f"Perturbed Su-Olson TRT Benchmark: [{label}] ({status_tag})\n"
        f"$m = 3 + \\delta \\quad (\\delta = {config.delta}), \\quad N_x = {len(x)}, \\quad \\Delta t = 5\\times 10^{{-4}}, \\quad t = {t_eval:.4f}$",
        fontsize=14,
        fontweight="bold",
        y=0.99,
    )
    plt.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Saved plot: {output_png}")


def run_single_method(
    spec: dict,
    problem: PerturbedSuOlsonProblem1D,
    n_steps: int = 100,
    dt: float = 0.0005,
) -> dict:
    """Run up to n_steps iterations for a given solver variant and measure pure solver time."""
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

    state = problem.initial()
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
    avg_krylov = total_krylov / max(1, steps_completed)
    avg_newton = total_newton / max(1, steps_completed)

    print(
        f"      Status: {status} | Steps: {steps_completed}/{n_steps} | "
        f"Solver Time: {total_solver_time:.2f}s (CPU: {total_solver_cpu:.2f}s) | "
        f"Krylov/step: {avg_krylov:.1f}"
    )

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
        "final_state": state,
        "final_time": final_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark solver variants for Perturbed Su-Olson TRT")
    parser.add_argument("--nx", type=int, default=128, help="Number of spatial points (default: 128)")
    parser.add_argument("--steps", type=int, default=100, help="Number of iterations/steps (default: 100)")
    parser.add_argument("--dt", type=float, default=0.0005, help="Time step dt (default: 0.0005)")
    parser.add_argument("--delta", type=float, default=0.01, help="Exponent perturbation delta (default: 0.01)")
    parser.add_argument("--output-dir", type=str, default="results/perturbed_su_olson", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path("/afs/pdc.kth.se/home/m/marcopas/.gemini/antigravity/brain/e17bf408-40ca-4f93-97d1-d18ac040e3cc")

    config = PerturbedSuOlsonConfig(delta=args.delta)
    problem_pert = PerturbedSuOlsonProblem1D(nx=args.nx, config=config, m_exp=config.m)
    problem_base = PerturbedSuOlsonProblem1D(nx=args.nx, config=config, m_exp=config.m_base)

    print("=" * 100)
    print("PERTURBED SU-OLSON TRT BENCHMARK (128 POINTS, 100 ITERATIONS)")
    print(f"Grid: Nx={args.nx}, dt={args.dt:.4e}, Total steps={args.steps}, t_final={args.steps * args.dt:.4f}")
    print(f"Perturbation: m = 3 + delta with delta = {config.delta:.4f} (m = {config.m:.4f})")
    print("=" * 100)

    # 1. Compute Base Linear Solution (delta = 0, m = 3.0)
    print("\nComputing base unperturbed solution (delta = 0, m = 3.0)...")
    base_res = run_single_method(
        {
            "label": "Base-Schur",
            "jvp_mode": "ad",
            "scale_mode": "state",
            "prec_kind": "schur",
            "table_jvp": "automatic differentiation",
            "table_scale": "global/local",
            "table_prec": "Schur-AMG",
            "desc": "Base linear state",
        },
        problem_base,
        n_steps=args.steps,
        dt=args.dt,
    )
    E_base, T_base = map(np.asarray, problem_base.unpack(base_res["final_state"]))

    # 2. Run the 7 Solver Configurations for the Perturbed Problem (delta = 0.01, m = 3.01)
    results = []
    xc = np.asarray(problem_pert.x)

    for spec in METHOD_SPECS:
        res = run_single_method(spec, problem_pert, n_steps=args.steps, dt=args.dt)

        # Compute perturbation differences relative to base linear solution
        l2_E, linf_E, l2_T, linf_T, diff_E, diff_T = problem_pert.compute_perturbation_effect(
            res["final_state"], base_res["final_state"]
        )
        res["l2_diff_E"] = l2_E
        res["linf_diff_E"] = linf_E
        res["l2_diff_T"] = l2_T
        res["linf_diff_T"] = linf_T

        results.append(res)

        E_pert, T_pert = map(np.asarray, problem_pert.unpack(res["final_state"]))

        png_name = f"su_olson_perturbed_comparison_{spec['label']}.png"
        plot_path = out_dir / png_name
        plot_perturbed_su_olson(
            x=xc,
            E_pert=E_pert,
            T_pert=T_pert,
            E_base=E_base,
            T_base=T_base,
            config=config,
            t_eval=res["final_time"],
            label=spec["label"],
            status=res["status"],
            output_png=plot_path,
        )

        if artifact_dir.exists():
            shutil.copy(plot_path, artifact_dir / png_name)
            print(f"Copied {png_name} to artifacts directory.")

    # Print LaTeX Table
    print("\n" + "=" * 100)
    print("LATEX TABLE OF RESULTS (PERTURBED SU-OLSON, delta = 0.01):")
    print("=" * 100)
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\begin{tabular}{llllcccc}")
    print(r"\toprule")
    print(r"Label & JVP & Scale policy & Physics preconditioner & Steps & Solver Time (s) & Avg GMRES & $\Delta T$ ($L_2$) \\")
    print(r"\midrule")
    for r in results:
        sp = r["spec"]
        time_str = f"{r['solver_time']:.2f}"
        gmres_str = f"{r['avg_krylov']:.1f}"
        err_str = f"{r['l2_diff_T']:.2e}"
        steps_str = f"{r['steps_completed']}/{r['total_steps']}"
        print(f"{sp['label']} & {sp['table_jvp']} & {sp['table_scale']} & {sp['table_prec']} & {steps_str} & {time_str} & {gmres_str} & {err_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\caption{Performance and accuracy comparison of JFNK solver variants on Perturbed Su-Olson TRT ($m = 3 + \delta, \delta = 0.01, N_x=128$, 100 steps). Solver Time measures cumulative time spent in nonlinear Newton-Krylov solver.}")
    print(r"\label{tab:perturbed_su_olson_comparison}")
    print(r"\end{table}")

    # Print Markdown Table
    print("\n" + "=" * 100)
    print("MARKDOWN TABLE OF RESULTS (PERTURBED SU-OLSON, delta = 0.01):")
    print("=" * 100)
    print("| Label | JVP | Scale policy | Physics preconditioner | Status | Steps | Solver Time (s) | CPU Time (s) | Avg GMRES/Step | $\\Delta E$ ($L_2$) | $\\Delta T$ ($L_2$) |")
    print("| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        sp = r["spec"]
        status_disp = "Converged" if r["status"] == "Converged" else "Stalled"
        print(f"| **{sp['label']}** | {sp['table_jvp']} | {sp['table_scale']} | {sp['table_prec']} | {status_disp} | {r['steps_completed']}/{r['total_steps']} | {r['solver_time']:.2f} | {r['solver_cpu_time']:.2f} | {r['avg_krylov']:.1f} | {r['l2_diff_E']:.3e} | {r['l2_diff_T']:.3e} |")
    print("=" * 100)

    # Save summary NPZ
    npz_summary = out_dir / "method_comparison_results.npz"
    np.savez(
        npz_summary,
        labels=np.array([r["label"] for r in results]),
        solver_times=np.array([r["solver_time"] for r in results]),
        solver_cpu_times=np.array([r["solver_cpu_time"] for r in results]),
        avg_krylov=np.array([r["avg_krylov"] for r in results]),
        l2_diff_E=np.array([r["l2_diff_E"] for r in results]),
        l2_diff_T=np.array([r["l2_diff_T"] for r in results]),
        steps_completed=np.array([r["steps_completed"] for r in results]),
        delta=args.delta,
        dt=args.dt,
        nx=args.nx,
    )
    print(f"\nAll comparison results saved to: {npz_summary}")


if __name__ == "__main__":
    main()

