"""1D Two-Temperature Non-Equilibrium Marshak Wave Benchmark Problem.

In thermal radiative transfer (TRT), the Marshak wave describes radiation penetrating
an initially cold, absorbing slab driven by an incident thermal radiation bath at x = 0.

When matter and radiation have not reached local thermodynamic equilibrium (LTE),
E != a T^4, and the system is governed by two coupled, out-of-equilibrium nonlinear equations:

    dE/dt - d/dx ( D(T) dE/dx ) = -c sigma_a(T) ( E - a T^4 )
    c_v dT/dt                   =  c sigma_a(T) ( E - a T^4 )

with:
    - Incident Marshak Robin boundary condition at x = 0:
        E(0, t) - 2 D(T(0, t)) dE/dx(0, t) = a T_bath^4
    - Reflecting boundary condition at x = L:
        dE/dx(L, t) = 0
    - Initial cold equilibrium:
        T(x, 0) = T_cold,  E(x, 0) = a T_cold^4  with  T_cold << T_bath.

This problem is intensely nonlinear due to:
1. Quartic emission a T^4 out of equilibrium (E != a T^4).
2. Temperature-dependent opacity sigma_a(T) = sigma_0 * (T_ref / T_eff)^p and D(T) = c / (3 sigma_a(T)).
3. Non-equilibrium timescale stiffness varying by orders of magnitude across the thermal wave front.
4. Nonlinear Marshak boundary coupling at x = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from trt_jfnk.discretization.diffusion import harmonic_mean
from trt_jfnk.models.material import PolynomialMaterial
from trt_jfnk.models.opacity import PowerLawOpacity


@dataclass(frozen=True)
class NonequilibriumMarshakConfig:
    """Physical configuration for two-temperature non-equilibrium Marshak wave."""

    L: float = 1.0
    cold: float = 0.2
    bath: float = 1.0
    sigma0: float = 1.0
    exponent: float = 3.0
    floor: float = 0.01
    cv0: float = 1.0
    cv1: float = 0.0
    c: float = 1.0
    a: float = 1.0


class NonequilibriumMarshakProblem1D:
    """1D Cell-Centered Finite Volume Two-Temperature Non-Equilibrium Marshak Wave."""

    def __init__(
        self,
        nx: int = 128,
        config: NonequilibriumMarshakConfig | None = None,
    ) -> None:
        self.nx = int(nx)
        self.config = config if config is not None else NonequilibriumMarshakConfig()
        self.dx = self.config.L / self.nx
        self.size = self.nx
        self.total_size = 2 * self.nx
        self.x = (jnp.arange(self.nx) + 0.5) * self.dx

        self.opacity = PowerLawOpacity(
            sigma_a0=self.config.sigma0,
            exponent=self.config.exponent,
            reference_temperature=1.0,
            temperature_floor=self.config.floor,
        )
        self.material = PolynomialMaterial(
            cv0=self.config.cv0,
            cv1=self.config.cv1,
        )

    def pack(self, E: jnp.ndarray, T: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate((E, T))

    def unpack(self, u: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return u[: self.nx], u[self.nx :]

    def initial(self) -> jnp.ndarray:
        """Initial cold equilibrium state E_0 = a T_cold^4, T_0 = T_cold."""
        T = jnp.full(self.nx, self.config.cold)
        E = self.config.a * (T**4)
        return self.pack(E, T)

    def coefficients(self, T: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute absorption opacity sigma_a(T) and diffusion coefficient D(T)."""
        sigma = self.opacity.absorption(T)
        D = self.config.c / (3.0 * sigma)
        return sigma, D

    def face_fluxes(self, E: jnp.ndarray, T: jnp.ndarray) -> jnp.ndarray:
        """Face fluxes F = -D dE/dx with Marshak incoming face at x=0 and reflecting at x=L."""
        _, D = self.coefficients(T)
        D_faces = harmonic_mean(D[:-1], D[1:])
        fx_interior = -D_faces * (E[1:] - E[:-1]) / self.dx

        # Marshak incoming left face: (a * bath^4 - E[0]) / (2 + 0.5 * dx / D[0])
        # In conservation: -dF/dx cell 0 = (F_left - F_{1/2}) / dx, so F_left = incoming
        incoming = (self.config.a * (self.config.bath**4) - E[0]) / (
            2.0 + 0.5 * self.dx / D[0]
        )
        f_left = incoming
        f_right = 0.0  # reflecting outer face

        return jnp.concatenate([jnp.array([f_left]), fx_interior, jnp.array([f_right])])

    def diffusion(self, E: jnp.ndarray, T: jnp.ndarray) -> jnp.ndarray:
        """Discrete cell-centered divergence: -div F."""
        fx = self.face_fluxes(E, T)
        return -(fx[1:] - fx[:-1]) / self.dx

    def residual(
        self,
        new_state: jnp.ndarray,
        old_state: jnp.ndarray,
        dt: float,
        time: float,
    ) -> jnp.ndarray:
        """Discrete implicit residual: F(u^{n+1}) = 0."""
        E, T = self.unpack(new_state)
        E0, T0 = self.unpack(old_state)

        sigma, _ = self.coefficients(T)
        g = self.config.c * sigma * (E - self.config.a * (T**4))
        diff = self.diffusion(E, T)

        res_E = E - E0 - dt * (diff - g)
        res_T = (
            self.material.internal_energy(T)
            - self.material.internal_energy(T0)
            - dt * g
        )
        return self.pack(res_E, res_T)

    def admissible(self, u: jnp.ndarray) -> jnp.ndarray:
        E, T = self.unpack(u)
        return jnp.all(jnp.isfinite(u)) & jnp.all(E > 0.0) & jnp.all(T > 1e-8)

    def local_blocks(self, u: jnp.ndarray, dt: float):
        """Analytic 2x2 local physical Jacobian blocks per cell."""
        E, T = self.unpack(u)
        sigma, _ = self.coefficients(T)
        sp = self.opacity.absorption_derivative(T)
        gt = self.config.c * (
            sp * (E - self.config.a * (T**4)) - 4.0 * self.config.a * sigma * (T**3)
        )

        a = 1.0 + dt * self.config.c * sigma
        b = dt * gt
        c = -dt * self.config.c * sigma
        d = self.material.heat_capacity(T) - dt * gt
        return a, b, c, d

    def negative_diffusion_matrix(self, u: jnp.ndarray):
        """Sparse tridiagonal negative diffusion matrix -L_E with Robin Marshak drive."""
        _, T = self.unpack(u)
        _, D = self.coefficients(T)
        D_np = np.asarray(D)
        w = 2.0 * D_np[:-1] * D_np[1:] / (D_np[:-1] + D_np[1:]) / (self.dx**2)

        diag = np.zeros(self.nx)
        diag[0] = w[0] + 1.0 / (2.0 + 0.5 * self.dx / D_np[0]) / self.dx
        diag[1:-1] = w[:-1] + w[1:]
        diag[-1] = w[-1]

        sub = -w
        sup = -w
        return diags([sub, diag, sup], [-1, 0, 1], shape=(self.nx, self.nx)).tocsc()

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

                neg_L = self.negative_diffusion_matrix(u)
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
                raise ValueError(f"Unknown preconditioner kind: {kind}")

            return lambda v: physical(context.residual_scale * v) / context.state_scale

        return factory

