"""Perturbed Su-Olson non-linear benchmark problem with perturbed heat capacity exponent.

Implements the non-linear TRT system where the material heat capacity exponent is perturbed:
    m = 3 + delta  (default delta = 0.01)
    c_v(T) = cv0 + alpha * T^(3 + delta)
    e(T) = cv0 * T + (alpha / (4 + delta)) * T^(4 + delta)

Provides the exact 1D Finite Volume (FV) problem with cell-centered discretization, periodic
boundary conditions, analytical local-block and Schur complement preconditioners, and reference
comparison against the linear unperturbed Su-Olson base state (delta = 0).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Tuple

import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class PerturbedSuOlsonConfig:
    """Parameters for the perturbed Su-Olson benchmark."""

    delta: float = 0.01  # Exponent perturbation: m = m_base + delta
    m_base: float = 3.0  # Base linear Su-Olson exponent
    alpha: float = 4.0  # Heat capacity coefficient (alpha=4 => e(T)=T^4 at delta=0)
    cv0: float = 0.1  # Heat capacity floor regularizer

    # Physical constants
    c: float = 1.0  # Speed of light
    a: float = 1.0  # Radiation constant
    sigma0: float = 1.0  # Constant absorption opacity

    # Background initial condition
    T_cold: float = 0.5

    # Volumetric source pulse
    Q0: float = 50.0  # Source amplitude
    x_src: float = 0.5  # Source center
    w_src: float = 0.08  # Gaussian width
    t_pulse: float = 0.05  # Pulse cutoff time

    # Domain
    length: float = 1.0

    @property
    def m(self) -> float:
        return self.m_base + self.delta

    @property
    def diffusion_coefficient(self) -> float:
        return self.c / (3.0 * self.sigma0)

    def heat_capacity(self, T, m_exp: float | None = None):
        exponent = self.m if m_exp is None else m_exp
        return self.cv0 + self.alpha * T**exponent

    def internal_energy(self, T, m_exp: float | None = None):
        exponent = self.m if m_exp is None else m_exp
        power = exponent + 1.0
        return self.cv0 * T + (self.alpha / power) * T**power

    def source(self, x: np.ndarray | jnp.ndarray, t: float):
        pulse = 1.0 if t <= self.t_pulse else 0.0
        return self.Q0 * jnp.exp(-((x - self.x_src) ** 2) / (2.0 * self.w_src**2)) * pulse


@dataclass
class PerturbedSuOlsonProblem1D:
    """1D Cell-Centered Finite Volume Problem for Perturbed Su-Olson TRT."""

    nx: int = 128
    config: PerturbedSuOlsonConfig = field(default_factory=PerturbedSuOlsonConfig)
    m_exp: float | None = None  # If None, defaults to config.m

    def __post_init__(self):
        if self.nx < 4:
            raise ValueError("nx must be at least 4")
        if self.m_exp is None:
            self.m_exp = self.config.m
        self.dx = self.config.length / self.nx
        self.x = (jnp.arange(self.nx) + 0.5) * self.dx
        self.size = self.nx
        self.total_size = 2 * self.nx
        self.D = self.config.diffusion_coefficient

    def pack(self, E, T):
        return jnp.concatenate((E.ravel(), T.ravel()))

    def unpack(self, u):
        return u[: self.nx], u[self.nx :]

    def initial(self):
        T_init = jnp.full(self.nx, self.config.T_cold)
        E_init = self.config.a * T_init**4
        return self.pack(E_init, T_init)

    def diffusion(self, E):
        """Discrete periodic cell-centered diffusion: D * d^2 E / dx^2."""
        return self.D * (jnp.roll(E, -1) - 2.0 * E + jnp.roll(E, 1)) / (self.dx**2)

    def residual(self, new_state, old_state, dt: float, time: float):
        """Discrete implicit residual: F(u^{n+1}) = 0."""
        E, T = self.unpack(new_state)
        E0, T0 = self.unpack(old_state)

        diff = self.diffusion(E)
        g = self.config.sigma0 * (E - self.config.a * T**4)
        q = self.config.source(self.x, time)

        res_E = E - E0 - dt * (diff - g + q)
        res_T = (
            self.config.internal_energy(T, self.m_exp)
            - self.config.internal_energy(T0, self.m_exp)
            - dt * g
        )
        return self.pack(res_E, res_T)

    def admissible(self, u):
        E, T = self.unpack(u)
        return jnp.all(jnp.isfinite(u)) & jnp.all(E > 0.0) & jnp.all(T > 0.0)

    def local_blocks(self, u, dt: float):
        """Analytic 2x2 local physical Jacobian blocks per cell."""
        _, T = self.unpack(u)
        cfg = self.config
        cv = cfg.heat_capacity(T, self.m_exp)
        g_T = -4.0 * cfg.a * cfg.sigma0 * T**3

        a = 1.0 + dt * cfg.sigma0
        b = dt * g_T
        c = -dt * cfg.sigma0
        d = cv - dt * g_T
        return a, b, c, d

    def negative_diffusion_matrix(self):
        """Sparse periodic negative diffusion matrix -L with constant D."""
        w_diff = self.D / (self.dx**2)
        rows, cols, vals = [], [], []
        for i in range(self.nx):
            ip = (i + 1) % self.nx
            im = (i - 1 + self.nx) % self.nx
            rows.extend([i, i, i])
            cols.extend([i, ip, im])
            vals.extend([2.0 * w_diff, -w_diff, -w_diff])
        return coo_matrix((vals, (rows, cols)), shape=(self.nx, self.nx)).tocsc()

    def preconditioner(self, dt: float, kind: str = "schur") -> Callable | None:
        if kind == "none":
            return None

        neg_L = self.negative_diffusion_matrix()

        def factory(u, context):
            a, b, c, d = self.local_blocks(u, dt)
            if kind == "local-block":
                determinant = a * d - b * c
                if bool(jnp.any(jnp.abs(determinant) < 1e-20)):
                    raise RuntimeError("singular local preconditioner")

                def physical(v):
                    x, y = v[: self.nx], v[self.nx :]
                    sol_x = (d * x - b * y) / determinant
                    sol_y = (-c * x + a * y) / determinant
                    return jnp.concatenate((sol_x, sol_y))

            elif kind == "schur":
                a_np, b_np, c_np, d_np = map(np.asarray, (a, b, c, d))
                if np.any(np.abs(d_np) < 1e-20):
                    raise RuntimeError("singular material block in Schur complement")

                S = diags(a_np - b_np * c_np / d_np) + dt * neg_L
                factor = splu(S.tocsc())

                def physical(v):
                    v_np = np.asarray(v)
                    x, y = v_np[: self.nx], v_np[self.nx :]
                    rhs_e = x - (b_np / d_np) * y
                    e = factor.solve(rhs_e)
                    t = (y - c_np * e) / d_np
                    return jnp.asarray(np.concatenate((e, t)))

            else:
                raise ValueError(f"unknown preconditioner kind: {kind}")

            return lambda v: physical(context.residual_scale * v) / context.state_scale

        return factory

    def compute_perturbation_effect(
        self, state_pert, state_base
    ) -> Tuple[float, float, float, float, np.ndarray, np.ndarray]:
        """Compute differences and sensitivities between perturbed (m=3+delta) and base (m=3) states."""
        E_pert, T_pert = map(np.asarray, self.unpack(state_pert))
        E_base, T_base = map(np.asarray, self.unpack(state_base))

        diff_E = E_pert - E_base
        diff_T = T_pert - T_base

        l2_diff_E = float(np.linalg.norm(diff_E) / np.sqrt(self.nx))
        linf_diff_E = float(np.max(np.abs(diff_E)))
        l2_diff_T = float(np.linalg.norm(diff_T) / np.sqrt(self.nx))
        linf_diff_T = float(np.max(np.abs(diff_T)))

        return l2_diff_E, linf_diff_E, l2_diff_T, linf_diff_T, diff_E, diff_T
