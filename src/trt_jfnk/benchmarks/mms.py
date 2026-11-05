"""Method of Manufactured Solutions (MMS) for non-linear TRT verification.

Provides exact symbolic derivations of source terms $Q_E(x, t)$ and $Q_T(x, t)$ using SymPy,
and a cell-centered 1D periodic Finite Volume (FV) problem class for asymptotic convergence
and code verification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Tuple

import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import splu
import sympy as sp


@dataclass(frozen=True)
class MMSConfig:
    """Parameters defining the non-linear TRT system and manufactured solution."""

    # Non-linear exponents
    p: int = 3  # Opacity exponent: sigma_a(T) propto T^(-p)
    m: int = 3  # Material capacity exponent: cv(T) = cv0 + cv1 * T^m

    # Physical constants
    c: float = 1.0  # Speed of light
    a: float = 1.0  # Radiation constant
    sigma_a0: float = 1.0  # Base absorption opacity
    T_ref: float = 1.0  # Reference temperature
    T_floor: float = 0.01  # Effective temperature regularizer floor

    # Material capacity constants
    cv0: float = 0.1
    cv1: float = 1.0

    # Manufactured solution amplitudes and offsets
    E0: float = 1.0
    E1: float = 0.3
    T0: float = 1.0
    T1: float = 0.2

    # Spatial domain and temporal decay rates
    length: float = 1.0
    lambda_E: float = 0.2
    lambda_T: float = 0.5


class SymbolicTRTMMS:
    """Derives exact continuous manufactured solution and analytical source terms via SymPy."""

    def __init__(self, config: MMSConfig | None = None):
        self.config = config or MMSConfig()
        self._derive()

    def _derive(self) -> None:
        cfg = self.config
        x, t = sp.symbols("x t", real=True)
        k = 2 * sp.pi / cfg.length

        # Exact smooth fields (strictly positive for E0 > E1 and T0 > T1)
        T_sym = cfg.T0 + cfg.T1 * sp.sin(k * x) * sp.exp(-cfg.lambda_T * t)
        E_sym = cfg.E0 + cfg.E1 * sp.cos(k * x) * sp.exp(-cfg.lambda_E * t)

        # Constitutive models
        T_eff = sp.sqrt(T_sym**2 + cfg.T_floor**2)
        sigma_a = cfg.sigma_a0 * (cfg.T_ref / T_eff) ** cfg.p
        D = cfg.c / (3 * sigma_a)
        cv = cfg.cv0 + cfg.cv1 * T_sym**cfg.m
        internal_energy = cfg.cv0 * T_sym + cfg.cv1 * T_sym ** (cfg.m + 1) / (cfg.m + 1)

        # Time derivatives
        dE_dt = sp.diff(E_sym, t)
        de_dt = cv * sp.diff(T_sym, t)

        # Spatial diffusion: div(D grad E) = d/dx(D * dE/dx)
        dE_dx = sp.diff(E_sym, x)
        flux = D * dE_dx
        div_flux = sp.diff(flux, x)

        # Radiation-matter exchange
        coupling = sigma_a * (E_sym - cfg.a * T_sym**4)

        # Exact forcing sources
        QE_sym = dE_dt - div_flux + coupling
        QT_sym = de_dt - coupling

        # Lambdify for fast numerical evaluation (vectorized over NumPy arrays)
        self.E_exact = sp.lambdify((x, t), E_sym, modules="numpy")
        self.T_exact = sp.lambdify((x, t), T_sym, modules="numpy")
        self.QE = sp.lambdify((x, t), QE_sym, modules="numpy")
        self.QT = sp.lambdify((x, t), QT_sym, modules="numpy")
        # Constitutive functions of temperature variable
        T_var = sp.Symbol("T_var", positive=True)
        T_eff_var = sp.sqrt(T_var**2 + cfg.T_floor**2)
        sigma_var = cfg.sigma_a0 * (cfg.T_ref / T_eff_var) ** cfg.p
        D_var = cfg.c / (3 * sigma_var)
        cv_var = cfg.cv0 + cfg.cv1 * T_var**cfg.m
        e_var = cfg.cv0 * T_var + cfg.cv1 * T_var ** (cfg.m + 1) / (cfg.m + 1)

        self.e_exact = sp.lambdify(T_var, e_var, modules="numpy")
        self.cv_exact = sp.lambdify(T_var, cv_var, modules="numpy")
        self.sigma_exact = sp.lambdify(T_var, sigma_var, modules="numpy")
        self.D_exact = sp.lambdify(T_var, D_var, modules="numpy")



@dataclass
class MMSProblem1D:
    """1D Cell-Centered Finite Volume non-linear TRT problem with periodic boundary conditions.

    Evaluates discrete residuals with manufactured sources and supports exact SuperLU
    Schur or cell-local block preconditioning.
    """

    nx: int = 32
    config: MMSConfig = field(default_factory=MMSConfig)
    theta: float = 1.0  # 1.0 = Backward Euler, 0.5 = Crank-Nicolson

    def __post_init__(self):
        if self.nx < 4:
            raise ValueError("nx must be at least 4")
        self.dx = self.config.length / self.nx
        self.x = (jnp.arange(self.nx) + 0.5) * self.dx  # Cell centers
        self.size = self.nx
        self.total_size = 2 * self.nx
        self.symbolic = SymbolicTRTMMS(self.config)

    def pack(self, E, T):
        return jnp.concatenate((E.ravel(), T.ravel()))

    def unpack(self, u):
        return u[: self.nx], u[self.nx :]

    def initial(self, time: float = 0.0):
        xc = np.asarray(self.x)
        E0 = jnp.asarray(self.symbolic.E_exact(xc, time))
        T0 = jnp.asarray(self.symbolic.T_exact(xc, time))
        return self.pack(E0, T0)

    def exact_solution(self, time: float):
        xc = np.asarray(self.x)
        E = jnp.asarray(self.symbolic.E_exact(xc, time))
        T = jnp.asarray(self.symbolic.T_exact(xc, time))
        return E, T

    def internal_energy(self, T):
        cfg = self.config
        return cfg.cv0 * T + cfg.cv1 * T ** (cfg.m + 1) / (cfg.m + 1)

    def heat_capacity(self, T):
        cfg = self.config
        return cfg.cv0 + cfg.cv1 * T**cfg.m

    def coefficients(self, T):
        cfg = self.config
        T_eff = jnp.sqrt(T**2 + cfg.T_floor**2)
        sigma = cfg.sigma_a0 * (cfg.T_ref / T_eff) ** cfg.p
        D = cfg.c / (3.0 * sigma)
        return sigma, D

    def diffusion(self, E, T):
        """Discrete periodic cell-centered diffusion: div(D grad E)."""
        _, D = self.coefficients(T)
        D_next = jnp.roll(D, -1)
        E_next = jnp.roll(E, -1)
        # Harmonic mean at periodic faces
        D_face = 2.0 * D * D_next / (D + D_next)
        flux_p = -D_face * (E_next - E) / self.dx
        flux_m = jnp.roll(flux_p, 1)
        return -(flux_p - flux_m) / self.dx

    def sources(self, time: float):
        xc = np.asarray(self.x)
        qE = jnp.asarray(self.symbolic.QE(xc, time))
        qT = jnp.asarray(self.symbolic.QT(xc, time))
        return qE, qT

    def residual(self, new_state, old_state, dt: float, time: float):
        """Discrete residual: F(u^{n+1}) = 0."""
        E, T = self.unpack(new_state)
        E0, T0 = self.unpack(old_state)

        sigma, _ = self.coefficients(T)
        g = sigma * (E - self.config.a * T**4)
        diff = self.diffusion(E, T)
        qE, qT = self.sources(time)

        res_E = E - E0 - dt * (diff - g + qE)
        res_T = self.internal_energy(T) - self.internal_energy(T0) - dt * (g + qT)
        return self.pack(res_E, res_T)

    def admissible(self, u):
        E, T = self.unpack(u)
        return jnp.all(jnp.isfinite(u)) & jnp.all(E > 0.0) & jnp.all(T > 0.0)

    def local_blocks(self, u, dt: float):
        """Analytic 2x2 local physical Jacobian blocks per cell."""
        E, T = self.unpack(u)
        cfg = self.config
        sigma, _ = self.coefficients(T)

        T_eff = jnp.sqrt(T**2 + cfg.T_floor**2)
        # d(sigma)/dT = -p * sigma * T / T_eff^2
        sigma_prime = -cfg.p * sigma * T / (T_eff**2)
        g_T = sigma_prime * (E - cfg.a * T**4) - 4.0 * cfg.a * sigma * T**3
        cv = self.heat_capacity(T)

        a = 1.0 + dt * sigma
        b = dt * g_T
        c = -dt * sigma
        d = cv - dt * g_T
        return a, b, c, d

    def negative_diffusion_matrix(self, u):
        """Sparse periodic negative diffusion matrix -L with frozen D(T)."""
        _, T = self.unpack(u)
        _, D = self.coefficients(T)
        D_np = np.asarray(D)
        idx = np.arange(self.nx)
        D_next = np.roll(D_np, -1)
        w = 2.0 * D_np * D_next / (D_np + D_next) / (self.dx**2)

        rows = []
        cols = []
        vals = []

        # Periodic tridiagonal stencil with corner wraps
        for i in range(self.nx):
            ip = (i + 1) % self.nx
            im = (i - 1 + self.nx) % self.nx
            # Cell i sees flux at i+1/2 and i-1/2
            w_p = w[i]
            w_m = w[im]
            rows.extend([i, i, i])
            cols.extend([i, ip, im])
            vals.extend([w_p + w_m, -w_p, -w_m])

        return coo_matrix((vals, (rows, cols)), shape=(self.nx, self.nx)).tocsc()

    def preconditioner(self, dt: float, kind: str = "schur") -> Callable | None:
        if kind == "none":
            return None

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

                # Schur complement matrix for radiation: S = diag(a - b*c/d) + dt * (-L)
                S = diags(a_np - b_np * c_np / d_np) + dt * self.negative_diffusion_matrix(u)
                factor = splu(S.tocsc())

                def physical(v):
                    v_np = np.asarray(v)
                    x, y = v_np[: self.nx], v_np[self.nx :]
                    # Eliminate y: S e = x - (b/d) y
                    rhs_e = x - (b_np / d_np) * y
                    e = factor.solve(rhs_e)
                    # Backsubstitute: t = (y - c*e) / d
                    t = (y - c_np * e) / d_np
                    return jnp.asarray(np.concatenate((e, t)))

            else:
                raise ValueError(f"unknown preconditioner kind: {kind}")

            return lambda v: physical(context.residual_scale * v) / context.state_scale

        return factory

    def compute_errors(self, state, time: float) -> Tuple[float, float, float, float]:
        """Compute L2 and Linf errors against exact manufactured solution."""
        E_num, T_num = map(np.asarray, self.unpack(state))
        E_exact, T_exact = map(np.asarray, self.exact_solution(time))

        err_E = np.abs(E_num - E_exact)
        err_T = np.abs(T_num - T_exact)

        l2_E = float(np.linalg.norm(err_E) / np.sqrt(self.nx))
        linf_E = float(np.max(err_E))
        l2_T = float(np.linalg.norm(err_T) / np.sqrt(self.nx))
        linf_T = float(np.max(err_T))

        return l2_E, linf_E, l2_T, linf_T
