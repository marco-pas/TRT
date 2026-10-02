#!/usr/bin/env python3
"""Generate publication-quality comparison plots: Theory vs Numerical FV for Ghosh (2014) slab.

Plots the final solution and time-evolution profiles comparing the cell-centered
Finite Volume (FV) JFNK numerical solution against the exact analytical solution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np

from trt_jfnk.benchmarks.ghosh_slab_analytic import GhoshSlabAnalyticBenchmark


def plot_final_solution(
    npz_path: Path = Path("results/ghosh_slab/ghosh_slab_fields.npz"),
    output_path: Path = Path("results/ghosh_slab/ghosh_slab_final_comparison.png"),
    epsilon: float = 0.1,
    n_roots: int = 40,
):
    if not npz_path.exists():
        raise FileNotFoundError(f"Fields file not found at {npz_path}. Run scripts/run_ghosh_slab.py first.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path)
    x_cells = data["x"]
    u_num = data["u"][:, 0] if data["u"].ndim > 1 else data["u"]
    v_num = data["v"][:, 0] if data["v"].ndim > 1 else data["v"]
    tau_final = float(data["time"])

    # Determine slab thickness b from cell centers: b = x_last + 0.5 * dx
    dx = float(x_cells[1] - x_cells[0])
    b = float(x_cells[-1] + 0.5 * dx)

    # Instantiate exact analytical benchmark
    benchmark = GhoshSlabAnalyticBenchmark(b=b, epsilon=epsilon, n_roots=n_roots)

    # Evaluate exact theory on dense spatial grid and cell centers
    x_dense = np.linspace(0.0, b, 300)
    u_exact_dense, v_exact_dense = benchmark.evaluate(x_dense, tau=tau_final)
    u_exact_pts, v_exact_pts = benchmark.evaluate(x_cells, tau=tau_final)
    u_inf_dense = benchmark.steady_state(x_dense)

    # Pointwise absolute errors
    err_u = np.abs(u_num - u_exact_pts)
    err_v = np.abs(v_num - v_exact_pts)

    max_err_u = float(np.max(err_u))
    max_err_v = float(np.max(err_v))
    l2_err_u = float(np.linalg.norm(err_u) / np.sqrt(len(x_cells)))
    l2_err_v = float(np.linalg.norm(err_v) / np.sqrt(len(x_cells)))

    # Set up matplotlib style
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8))

    # --- Panel 1: Final Solution (Theory vs Numerical) ---
    ax1.set_title(
        rf"(a) Solutions at $\tau = {tau_final:.2f}$ ($b = {b:.1f}$, $\epsilon = {epsilon}$)"
    )
    ax1.set_xlabel(r"Depth, $x = \sqrt{3}\sigma z$")
    ax1.set_ylabel(r"Energy Density")

    # Exact theory curves
    ax1.plot(x_dense, u_exact_dense, "b-", label=r"Analytical $u(x, \tau)$")
    ax1.plot(x_dense, v_exact_dense, "r-", label=r"Analytical $v(x, \tau)$")
    ax1.plot(x_dense, u_inf_dense, "k--", alpha=0.6, label=r"Steady State $u_\infty(x)$")

    # Numerical FV points (subsample if many cells)
    stride = max(1, len(x_cells) // 30)
    ax1.plot(
        x_cells[::stride],
        u_num[::stride],
        "bo",
        fillstyle="none",
        markeredgewidth=1.6,
        label=r"Numerical $u_{\mathrm{num}}$",
    )
    ax1.plot(
        x_cells[::stride],
        v_num[::stride],
        "r^",
        fillstyle="none",
        markeredgewidth=1.6,
        label=r"Numerical $v_{\mathrm{num}}$",
    )

    ax1.set_xlim(0, b)
    ax1.set_ylim(bottom=0.0)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="upper right", framealpha=0.92)

    # --- Panel 2: Pointwise Discretization Error ---
    ax2.set_title(rf"(b) Error $|\mathrm{{FV}} - \mathrm{{Theory}}|$")
    ax2.set_xlabel(r"Depth, $x = \sqrt{3}\sigma z$")
    ax2.set_ylabel(r"Abs Error")

    ax2.semilogy(
        x_cells,
        err_u,
        "b-o",
        markevery=stride,
        fillstyle="none",
        markeredgewidth=1.5,
        label=rf"$|u_{{\mathrm{{num}}}} - u_{{\mathrm{{exact}}}}|$ ($L_\infty = {max_err_u:.2e}, L_2 = {l2_err_u:.2e}$)",
    )
    ax2.semilogy(
        x_cells,
        err_v,
        "r-s",
        markevery=stride,
        fillstyle="none",
        markeredgewidth=1.5,
        label=rf"$|v_{{\mathrm{{num}}}} - v_{{\mathrm{{exact}}}}|$ ($L_\infty = {max_err_v:.2e}, L_2 = {l2_err_v:.2e}$)",
    )

    ax2.set_xlim(0, b)
    ax2.grid(True, which="both", linestyle=":", alpha=0.7)
    ax2.legend(loc="upper right", framealpha=0.92)

    fig.suptitle(
        r" Benchmark from 'Ghosh, 2014' ",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()

    print(f"Plot successfully saved to: {output_path}")
    print(f"Summary metrics at tau = {tau_final}:")
    print(f"  Radiation error: L_inf = {max_err_u:.4e}, L_2 = {l2_err_u:.4e}")
    print(f"  Material error:  L_inf = {max_err_v:.4e}, L_2 = {l2_err_v:.4e}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot Theory vs Numerical FV comparison for Ghosh planar slab."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/ghosh_slab/ghosh_slab_fields.npz"),
        help="Path to numerical field npz file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ghosh_slab/ghosh_slab_final_comparison.png"),
        help="Path to save output plot",
    )
    parser.add_argument("--epsilon", type=float, default=0.1, help="Coupling parameter epsilon")
    args = parser.parse_args()

    plot_final_solution(
        npz_path=args.input,
        output_path=args.output,
        epsilon=args.epsilon,
    )


if __name__ == "__main__":
    main()
