"""Exact analytical solution for the finite planar slab non-equilibrium TRT benchmark.

Reference:
    Karabi Ghosh (2014), "Analytical benchmark for non-equilibrium radiation
    diffusion in finite size systems", Annals of Nuclear Energy 63, 59-68.
    DOI: 10.1016/j.anucene.2013.07.028

Solves the dimensionless non-equilibrium Su-Olson linear TRT equations on x in [0, b]:
    eps * du/dtau = d^2 u / dx^2 + v - u
    dv/dtau       = u - v

Subject to:
    u(x, 0) = 0,  v(x, 0) = 0
    u(0, tau) - (2 / sqrt(3)) * du/dx(0, tau) = 1  (incident Marshak flux)
    u(b, tau) + (2 / sqrt(3)) * du/dx(b, tau) = 0  (vacuum leakage)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import brentq


SQRT3 = np.sqrt(3.0)


def compute_transcendental_roots(b: float = 1.0, n_roots: int = 30) -> np.ndarray:
    """Compute the first ``n_roots`` positive roots of the transcendental equation:

        (4 beta^2 - 3) sin(beta * b) - 4 sqrt(3) beta cos(beta * b) = 0

    which is equivalent to:
        tan(beta * b) = (4 sqrt(3) beta) / (4 beta^2 - 3)
    """
    def h(beta: float) -> float:
        return (4.0 * beta**2 - 3.0) * np.sin(beta * b) - 4.0 * SQRT3 * beta * np.cos(beta * b)

    roots: list[float] = []

    # First root beta_1 lies in (sqrt(3)/2, pi/b)
    beta_singularity = SQRT3 / 2.0 / b
    r1 = brentq(h, beta_singularity + 1e-5, np.pi / b)
    roots.append(r1)

    # Subsequent roots occur once per tangent branch
    m = 1
    while len(roots) < n_roots:
        left = m * np.pi / b
        right = (m + 1) * np.pi / b
        sub = np.linspace(left, right, 100)
        vals = h(sub)
        sc = np.where(np.diff(np.sign(vals)))[0]
        for idx in sc:
            r = brentq(h, sub[idx], sub[idx + 1])
            if r > roots[-1] + 1e-4:
                roots.append(r)
                break
        m += 1

    return np.array(roots[:n_roots], dtype=np.float64)


def ghosh_slab_steady_state(x: np.ndarray | float, b: float = 1.0) -> np.ndarray:
    """Asymptotic linear steady-state solution u_inf(x) = v_inf(x) as tau -> infinity."""
    x_arr = np.asarray(x, dtype=np.float64)
    return (3.0 * b + 2.0 * SQRT3 - 3.0 * x_arr) / (3.0 * b + 4.0 * SQRT3)


@dataclass(frozen=True)
class GhoshSlabAnalyticBenchmark:
    """Precomputed analytical benchmark generator for the finite planar slab."""

    b: float = 1.0
    epsilon: float = 0.1
    n_roots: int = 30

    def __post_init__(self) -> None:
        if self.b <= 0.0 or self.epsilon <= 0.0:
            raise ValueError("b and epsilon must be strictly positive")
        if self.n_roots < 1:
            raise ValueError("n_roots must be at least 1")

        roots = compute_transcendental_roots(self.b, self.n_roots)
        # Store roots and associated pole quantities in object
        object.__setattr__(self, "roots", roots)

        # Precompute poles and residue coefficients
        poles_s1 = []
        poles_s2 = []
        brackets = []
        dbeta_ds1 = []
        dbeta_ds2 = []

        for beta_k in roots:
            disc = (self.epsilon + beta_k**2 + 1.0)**2 - 4.0 * self.epsilon * beta_k**2
            sqrt_disc = np.sqrt(disc)
            s1 = (-(self.epsilon + beta_k**2 + 1.0) + sqrt_disc) / (2.0 * self.epsilon)
            s2 = (-(self.epsilon + beta_k**2 + 1.0) - sqrt_disc) / (2.0 * self.epsilon)

            poles_s1.append(s1)
            poles_s2.append(s2)

            Q = 3.0 * self.b + 4.0 * SQRT3 - 4.0 * beta_k**2 * self.b
            R = 4.0 * SQRT3 * beta_k * self.b + 8.0 * beta_k
            brk = Q * np.cos(beta_k * self.b) - R * np.sin(beta_k * self.b)
            brackets.append(brk)

            dbeta_ds1.append(-0.5 / beta_k * (self.epsilon + 1.0 / (s1 + 1.0)**2))
            dbeta_ds2.append(-0.5 / beta_k * (self.epsilon + 1.0 / (s2 + 1.0)**2))

        object.__setattr__(self, "_s1", np.array(poles_s1))
        object.__setattr__(self, "_s2", np.array(poles_s2))
        object.__setattr__(self, "_bracket", np.array(brackets))
        object.__setattr__(self, "_dbeta_ds1", np.array(dbeta_ds1))
        object.__setattr__(self, "_dbeta_ds2", np.array(dbeta_ds2))

    def steady_state(self, x: np.ndarray | float) -> np.ndarray:
        return ghosh_slab_steady_state(x, self.b)

    def evaluate(
        self, x: np.ndarray | Sequence[float] | float, tau: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate radiation u(x, tau) and material energy v(x, tau) at a single time tau."""
        x_arr = np.asarray(x, dtype=np.float64)
        scalar_input = x_arr.ndim == 0
        if scalar_input:
            x_arr = x_arr.reshape(1)

        u = self.steady_state(x_arr).copy()
        v = self.steady_state(x_arr).copy()

        if tau < 0.0:
            raise ValueError("tau must be non-negative")

        roots = self.roots
        s1 = self._s1
        s2 = self._s2
        bracket = self._bracket
        dbeta_ds1 = self._dbeta_ds1
        dbeta_ds2 = self._dbeta_ds2

        # Sum residue terms over each root and its two poles
        for k in range(len(roots)):
            beta_k = roots[k]
            # Spatial eigenmode: 3 sin(beta * (b - x)) + 2 sqrt(3) beta cos(beta * (b - x))
            arg = beta_k * (self.b - x_arr)
            spatial = 3.0 * np.sin(arg) + 2.0 * SQRT3 * beta_k * np.cos(arg)

            # Pole s1
            denom_u1 = s1[k] * bracket[k] * dbeta_ds1[k]
            denom_v1 = denom_u1 * (s1[k] + 1.0)
            exp_s1 = np.exp(s1[k] * tau)
            u += exp_s1 * spatial / denom_u1
            v += exp_s1 * spatial / denom_v1

            # Pole s2
            denom_u2 = s2[k] * bracket[k] * dbeta_ds2[k]
            denom_v2 = denom_u2 * (s2[k] + 1.0)
            exp_s2 = np.exp(s2[k] * tau)
            u += exp_s2 * spatial / denom_u2
            v += exp_s2 * spatial / denom_v2

        if scalar_input:
            return float(u[0]), float(v[0])
        return u, v

    def evaluate_timeseries(
        self, x: np.ndarray | Sequence[float], times: np.ndarray | Sequence[float]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate u and v over multiple times.

        Returns:
            (u_grid, v_grid) each of shape (len(times), len(x)).
        """
        x_arr = np.asarray(x, dtype=np.float64)
        t_arr = np.asarray(times, dtype=np.float64)

        u_out = np.zeros((len(t_arr), len(x_arr)), dtype=np.float64)
        v_out = np.zeros((len(t_arr), len(x_arr)), dtype=np.float64)

        for i, tau in enumerate(t_arr):
            u_out[i], v_out[i] = self.evaluate(x_arr, tau)

        return u_out, v_out
