"""Stiff multiscale Method of Manufactured Solutions (MMS) for non-linear TRT verification.

Features a sharp central sinc peak modulated by an exponential envelope, spanning extreme
temperature contrasts (e.g. T_cold = 10 to T_hot = 10,000, dynamic range 10^3; E spans 10^12;
opacity spans 10^9). Provides exact symbolic derivations of source terms QE(x, t) and QT(x, t)
via SymPy, and a cell-centered 1D periodic Finite Volume (FV) problem with Schur complement
and scale-equilibrated preconditioning.
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
class StiffMMSConfig:
    """Parameters defining the stiff multiscale non-linear TRT system and manufactured solution."""

    # Non-linear exponents
    p: int = 3  # Opacity exponent: sigma_a(T) = sigma_a0 * (T_ref / T)^p
    m: int = 3  # Heat capacity exponent: cv(T) = cv0 + cv1 * (T / T_ref)^m

    # Physical constants
    c: float = 1.0  # Speed of light
    a: float = 1.0  # Radiation constant
    sigma_a0: float = 1.0  # Base absorption opacity at T_ref
    T_floor: float = 1e-4  # Effective temperature regularizer floor

    # Material capacity constants
    cv0: float = 0.1
    cv1: float = 1.0

    # Temperature extremes (e.g., T_cold = 10.0, T_hot = 10000.0)
    T_cold: float = 10.0
    T_hot: float = 10000.0

    # Radiation energy density extremes (default: a * T^4)
    E_cold: float | None = None
    E_hot: float | None = None

    # Reference temperature (defaults to T_hot)
    T_ref: float | None = None

    # Profile geometry: peak center, sinc scale s, and Gaussian envelope width w
    x0: float = 0.5
    s_T: float = 0.06
    w_T: float = 0.10
    s_E: float = 0.06
    w_E: float = 0.10

    # Domain length and temporal decay rates
    length: float = 1.0
    lambda_E: float = 0.02
    lambda_T: float = 0.05

    def get_T_ref(self) -> float:
        return self.T_ref if self.T_ref is not None else self.T_hot

    def get_E_cold(self) -> float:
        return self.E_cold if self.E_cold is not None else self.a * (self.T_cold**4)

    def get_E_hot(self) -> float:
        return self.E_hot if self.E_hot is not None else self.a * (self.T_hot**4)


class SymbolicStiffTRTMMS:
    """Derives exact continuous stiff manufactured solution and analytical source terms via SymPy."""

    def __init__(self, config: StiffMMSConfig | None = None):
        self.config = config or StiffMMSConfig()
        self._derive()

    def _derive(self) -> None:
        cfg = self.config
        x, t = sp.symbols("x t", real=True)

        T_ref_val = cfg.get_T_ref()
        E_cold_val = cfg.get_E_cold()
        E_hot_val = cfg.get_E_hot()

        # Modulation envelopes: sinc(z) * exp(-(x - x0)^2 / (2 w^2)) squared
        z_T = (x - cfg.x0) / cfg.s_T
        phi_T = (sp.sin(z_T) / z_T * sp.exp(-((x - cfg.x0) ** 2) / (2.0 * cfg.w_T**2))) ** 2

        z_E = (x - cfg.x0) / cfg.s_E
        phi_E = (sp.sin(z_E) / z_E * sp.exp(-((x - cfg.x0) ** 2) / (2.0 * cfg.w_E**2))) ** 2

        # Manufactured fields
        T_sym = cfg.T_cold + (cfg.T_hot - cfg.T_cold) * phi_T * sp.exp(-cfg.lambda_T * t)
        E_sym = E_cold_val + (E_hot_val - E_cold_val) * phi_E * sp.exp(-cfg.lambda_E * t)

        # Constitutive models
        T_eff = sp.sqrt(T_sym**2 + cfg.T_floor**2)
        sigma_a = cfg.sigma_a0 * (T_ref_val / T_eff) ** cfg.p
        D = cfg.c / (3.0 * sigma_a)
        cv = cfg.cv0 + cfg.cv1 * (T_sym / T_ref_val) ** cfg.m
        internal_energy = cfg.cv0 * T_sym + (cfg.cv1 * T_ref_val / (cfg.m + 1.0)) * (
            T_sym / T_ref_val
        ) ** (cfg.m + 1.0)

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

        # Raw lambdified functions
        self._raw_E = sp.lambdify((x, t), E_sym, modules="numpy")
        self._raw_T = sp.lambdify((x, t), T_sym, modules="numpy")
        self._raw_QE = sp.lambdify((x, t), QE_sym, modules="numpy")
        self._raw_QT = sp.lambdify((x, t), QT_sym, modules="numpy")

        # Constitutive functions of temperature variable
        T_var = sp.Symbol("T_var", positive=True)
        T_eff_var = sp.sqrt(T_var**2 + cfg.T_floor**2)
        sigma_var = cfg.sigma_a0 * (T_ref_val / T_eff_var) ** cfg.p
        D_var = cfg.c / (3.0 * sigma_var)
        cv_var = cfg.cv0 + cfg.cv1 * (T_var / T_ref_val) ** cfg.m
        e_var = cfg.cv0 * T_var + (cfg.cv1 * T_ref_val / (cfg.m + 1.0)) * (
            T_var / T_ref_val
        ) ** (cfg.m + 1.0)

        self.e_exact = sp.lambdify(T_var, e_var, modules="numpy")
        self.cv_exact = sp.lambdify(T_var, cv_var, modules="numpy")
        self.sigma_exact = sp.lambdify(T_var, sigma_var, modules="numpy")
        self.D_exact = sp.lambdify(T_var, D_var, modules="numpy")

    def _safe_eval(self, fn: Callable, x: np.ndarray | float, t: float) -> np.ndarray:
        """Safely evaluate lambdified functions near singular point x0 = 0.5 by symmetry."""
        x_arr = np.asarray(x, dtype=np.float64)
        is_center = np.isclose(x_arr, self.config.x0, atol=1e-12)
        if np.any(is_center):
            # Symmetry of phi(x) around x0 guarantees zero odd derivatives;
            # 1e-8 offset avoids 0/0 while keeping error under machine precision
            x_safe = np.where(is_center, x_arr + 1e-8, x_arr)
            return fn(x_safe, t)
        return fn(x_arr, t)

    def E_exact(self, x: np.ndarray | float, t: float) -> np.ndarray:
        return self._safe_eval(self._raw_E, x, t)

    def T_exact(self, x: np.ndarray | float, t: float) -> np.ndarray:
        return self._safe_eval(self._raw_T, x, t)

    def QE(self, x: np.ndarray | float, t: float) -> np.ndarray:
        return self._safe_eval(self._raw_QE, x, t)

    def QT(self, x: np.ndarray | float, t: float) -> np.ndarray:
        return self._safe_eval(self._raw_QT, x, t)


@dataclass
class StiffMMSProblem1D:
    """1D Cell-Centered Finite Volume non-linear TRT problem with periodic boundary conditions.

    Solves the stiff multiscale TRT system with Schur complement or local-block
    preconditioning and scale-equilibrated Newton iterations.
    """

    nx: int = 64
    config: StiffMMSConfig = field(default_factory=StiffMMSConfig)
    theta: float = 1.0  # 1.0 = Backward Euler

    def __post_init__(self):
        if self.nx < 4:
            raise ValueError("nx must be at least 4")
        self.dx = self.config.length / self.nx
        self.x = (jnp.arange(self.nx) + 0.5) * self.dx  # Cell centers
        self.size = self.nx
        self.total_size = 2 * self.nx
        self.symbolic = SymbolicStiffTRTMMS(self.config)

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
        T_ref_val = cfg.get_T_ref()
        return cfg.cv0 * T + (cfg.cv1 * T_ref_val / (cfg.m + 1.0)) * (T / T_ref_val) ** (
            cfg.m + 1.0
        )

    def heat_capacity(self, T):
        cfg = self.config
        T_ref_val = cfg.get_T_ref()
        return cfg.cv0 + cfg.cv1 * (T / T_ref_val) ** cfg.m

    def coefficients(self, T):
        cfg = self.config
        T_ref_val = cfg.get_T_ref()
        T_eff = jnp.sqrt(T**2 + cfg.T_floor**2)
        sigma = cfg.sigma_a0 * (T_ref_val / T_eff) ** cfg.p
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
        T_ref_val = cfg.get_T_ref()
        sigma, _ = self.coefficients(T)

        T_eff = jnp.sqrt(T**2 + cfg.T_floor**2)
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
        D_next = np.roll(D_np, -1)
        w = 2.0 * D_np * D_next / (D_np + D_next) / (self.dx**2)

        rows = []
        cols = []
        vals = []

        for i in range(self.nx):
            ip = (i + 1) % self.nx
            im = (i - 1 + self.nx) % self.nx
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

                S = diags(a_np - b_np * c_np / d_np) + dt * self.negative_diffusion_matrix(u)
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

    def compute_errors(
        self, state, time: float
    ) -> Tuple[float, float, float, float, float, float]:
        """Compute absolute and relative L2 and Linf errors against exact manufactured solution."""
        E_num, T_num = map(np.asarray, self.unpack(state))
        E_exact, T_exact = map(np.asarray, self.exact_solution(time))

        err_E = np.abs(E_num - E_exact)
        err_T = np.abs(T_num - T_exact)

        l2_E = float(np.linalg.norm(err_E) / np.sqrt(self.nx))
        linf_E = float(np.max(err_E))
        l2_T = float(np.linalg.norm(err_T) / np.sqrt(self.nx))
        linf_T = float(np.max(err_T))

        norm_E = float(np.linalg.norm(E_exact) / np.sqrt(self.nx))
        norm_T = float(np.linalg.norm(T_exact) / np.sqrt(self.nx))
        rel_l2_E = l2_E / (norm_E + 1e-30)
        rel_l2_T = l2_T / (norm_T + 1e-30)

        return l2_E, linf_E, l2_T, linf_T, rel_l2_E, rel_l2_T

