"""Cell-centred finite volume formulation of the Karabi Ghosh (2014) planar slab benchmark.

Solves the dimensionless non-equilibrium thermal radiative transfer (TRT) equations
on a finite slab x in [0, b]:
    eps * du/dtau - d^2 u / dx^2 = -(u - v)
    dv/dtau = u - v

Subject to:
    u(x, 0) = 0,  v(x, 0) = 0
    u(0, tau) - (2 / sqrt(3)) * du/dx(0, tau) = 1  (Marshak incident radiation flux)
    u(b, tau) + (2 / sqrt(3)) * du/dx(b, tau) = 0  (vacuum boundary leakage)
    Top/bottom boundaries: reflecting (zero flux)

Reference:
    Karabi Ghosh (2014), "Analytical benchmark for non-equilibrium radiation
    diffusion in finite size systems", Annals of Nuclear Energy 63, 59-68.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from trt_jfnk.benchmarks.ghosh_slab_analytic import ghosh_slab_steady_state

SQRT3 = float(np.sqrt(3.0))


@dataclass
class GhoshSlabFVProblem:
    b: float = 1.0
    epsilon: float = 0.1
    nx: int = 50
    ny: int = 1
    theta: float = 1.0

    def __post_init__(self) -> None:
        if self.b <= 0.0:
            raise ValueError("slab thickness b must be strictly positive")
        if self.epsilon <= 0.0:
            raise ValueError("coupling parameter epsilon must be strictly positive")
        if self.nx < 2 or self.ny < 1:
            raise ValueError("nx must be >= 2 and ny must be >= 1")
        if not (0.5 <= self.theta <= 1.0):
            raise ValueError("theta must lie in [0.5, 1.0]")

        self.dx = self.b / float(self.nx)
        self.dy = 1.0 / float(self.ny)
        self.size = self.nx * self.ny

        # Cell centers
        self.x = (jnp.arange(self.nx, dtype=jnp.float64) + 0.5) * self.dx
        self.y = (jnp.arange(self.ny, dtype=jnp.float64) + 0.5) * self.dy
        self.X, self.Y = jnp.meshgrid(self.x, self.y, indexing="ij")

        # Boundary Robin weighting coefficients across half cell:
        # F_0 = (1 - u_0) / (2/sqrt(3) + dx/2)
        # F_b = (u_{nx-1} - 0) / (2/sqrt(3) + dx/2)
        self.w0 = 1.0 / (2.0 / SQRT3 + 0.5 * self.dx)
        self.wb = 1.0 / (2.0 / SQRT3 + 0.5 * self.dx)

    def pack(self, u: jax.Array, v: jax.Array) -> jax.Array:
        """Pack radiation and material fields into a single state vector."""
        return jnp.concatenate((u.ravel(), v.ravel()))

    def unpack(self, state: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Unpack state vector into 2D radiation u and material v arrays."""
        u = state[: self.size].reshape(self.nx, self.ny)
        v = state[self.size :].reshape(self.nx, self.ny)
        return u, v

    def initial(self) -> jax.Array:
        """Return zero initial conditions u(x, 0) = 0, v(x, 0) = 0."""
        return jnp.zeros(2 * self.size, dtype=jnp.float64)

    def steady_state(self) -> tuple[jax.Array, jax.Array]:
        """Evaluate exact analytical steady-state on the grid."""
        u_inf_1d = ghosh_slab_steady_state(self.x, self.b)
        u_inf = jnp.broadcast_to(u_inf_1d[:, None], (self.nx, self.ny))
        return u_inf, u_inf

    def face_fluxes(self, u: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute oriented face fluxes F = -grad(u).

        Returns:
            fx: (nx + 1, ny) fluxes at x faces (fx[0] = Marshak, fx[-1] = vacuum leakage)
            fy: (nx, ny + 1) fluxes at y faces (all zero for reflecting boundaries)
        """
        fx = jnp.zeros((self.nx + 1, self.ny), dtype=u.dtype)
        fy = jnp.zeros((self.nx, self.ny + 1), dtype=u.dtype)

        # Interior x faces: standard central difference
        fx = fx.at[1:-1].set(-(u[1:] - u[:-1]) / self.dx)

        # Left boundary face (x = 0): incoming Marshak radiation
        fx = fx.at[0].set(self.w0 * (1.0 - u[0]))

        # Right boundary face (x = b): vacuum leakage
        fx = fx.at[-1].set(self.wb * u[-1])

        # Interior y faces (if ny > 1)
        if self.ny > 1:
            fy = fy.at[:, 1:-1].set(-(u[:, 1:] - u[:, :-1]) / self.dy)

        return fx, fy

    def diffusion(self, u: jax.Array) -> jax.Array:
        """Compute spatial diffusion divergence L(u) = -div(F) = d^2 u / dx^2."""
        fx, fy = self.face_fluxes(u)
        div = -(fx[1:] - fx[:-1]) / self.dx
        if self.ny > 1:
            div = div - (fy[:, 1:] - fy[:, :-1]) / self.dy
        return div

    def residual(
        self,
        new_state: jax.Array,
        old_state: jax.Array,
        dt: float,
        time: float = 0.0,
    ) -> jax.Array:
        """Compute discrete time-stepping residual R(new_state) = 0."""
        u, v = self.unpack(new_state)
        u0, v0 = self.unpack(old_state)

        diff_new = self.diffusion(u)
        diff_old = self.diffusion(u0)

        g_new = u - v
        g_old = u0 - v0

        # Theta-method blending: theta=1.0 is Backward Euler; theta=0.5 is Crank-Nicolson
        blend_diff = self.theta * diff_new + (1.0 - self.theta) * diff_old
        blend_g = self.theta * g_new + (1.0 - self.theta) * g_old

        res_u = self.epsilon * (u - u0) - dt * (blend_diff - blend_g)
        res_v = (v - v0) - dt * blend_g

        return self.pack(res_u, res_v)

    def is_admissible(self, state: jax.Array) -> bool:
        """Check physical admissibility: non-negative and finite."""
        u, v = self.unpack(state)
        return bool(
            jnp.all(jnp.isfinite(state))
            and jnp.all(u >= -1.0e-7)
            and jnp.all(v >= -1.0e-7)
        )

    def admissible(self, state: jax.Array) -> bool:
        """Alias for is_admissible."""
        return self.is_admissible(state)

    def energy(self, state: jax.Array) -> float:
        """Compute total stored dimensionless energy integral: int (eps * u + v) dx dy."""
        u, v = self.unpack(state)
        return float(jnp.sum(self.epsilon * u + v) * self.dx * self.dy)

    def input_power(self, state: jax.Array, time: float = 0.0) -> float:
        """Compute net boundary power inflow: int (F(0) - F(b)) dy."""
        u, _ = self.unpack(state)
        fx, fy = self.face_fluxes(u)
        return float((jnp.sum(fx[0]) - jnp.sum(fx[-1])) * self.dy)

    def local_blocks(self, state: jax.Array, dt: float) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Compute the point-local 2x2 Jacobian blocks (a, b, c, d) omitting diffusion."""
        w = dt * self.theta
        a = self.epsilon + w
        b = -w
        c = -w
        d = 1.0 + w
        shape = (self.size,)
        dtype = state.dtype
        return (
            jnp.full(shape, a, dtype=dtype),
            jnp.full(shape, b, dtype=dtype),
            jnp.full(shape, c, dtype=dtype),
            jnp.full(shape, d, dtype=dtype),
        )

    def negative_diffusion_matrix(self, state: jax.Array | None = None) -> coo_matrix:
        """Construct sparse CSC matrix A representing -L (frozen diffusion operator)."""
        idx = np.arange(self.size).reshape(self.nx, self.ny)
        rows: list[np.ndarray | int] = []
        cols: list[np.ndarray | int] = []
        vals: list[np.ndarray | float] = []

        # X interior faces
        a_x, b_x = idx[:-1, :].ravel(), idx[1:, :].ravel()
        w_x = np.full(a_x.shape, 1.0 / self.dx**2)
        for rr, cc, vv in ((a_x, a_x, w_x), (b_x, b_x, w_x), (a_x, b_x, -w_x), (b_x, a_x, -w_x)):
            rows.extend(rr)
            cols.extend(cc)
            vals.extend(vv)

        # Left boundary Marshak Robin face
        w0_cell = self.w0 / self.dx
        rows.extend(idx[0, :])
        cols.extend(idx[0, :])
        vals.extend(np.full(self.ny, w0_cell))

        # Right boundary vacuum Robin face
        wb_cell = self.wb / self.dx
        rows.extend(idx[-1, :])
        cols.extend(idx[-1, :])
        vals.extend(np.full(self.ny, wb_cell))

        # Y interior faces (if ny > 1)
        if self.ny > 1:
            a_y, b_y = idx[:, :-1].ravel(), idx[:, 1:].ravel()
            w_y = np.full(a_y.shape, 1.0 / self.dy**2)
            for rr, cc, vv in ((a_y, a_y, w_y), (b_y, b_y, w_y), (a_y, b_y, -w_y), (b_y, a_y, -w_y)):
                rows.extend(rr)
                cols.extend(cc)
                vals.extend(vv)

        return coo_matrix((vals, (rows, cols)), shape=(self.size, self.size)).tocsc()

    def preconditioner(
        self,
        dt: float,
        kind: Literal["none", "local-block", "schur"] = "local-block",
    ) -> Callable | None:
        """Build preconditioner factory for Newton-Krylov solver."""
        if kind == "none":
            return None

        w = dt * self.theta
        a = self.epsilon + w
        b = -w
        c = -w
        d = 1.0 + w
        det = a * d - b * c

        if kind == "local-block":
            def local_factory(u, context):
                def apply_local(rhs):
                    scaled_rhs = context.residual_scale * rhs
                    rx = scaled_rhs[: self.size]
                    ry = scaled_rhs[self.size :]
                    sx = (d * rx - b * ry) / det
                    sy = (-c * rx + a * ry) / det
                    solution = jnp.concatenate((sx, sy))
                    return solution / context.state_scale
                return apply_local
            return local_factory

        if kind == "schur":
            # Exact Schur complement operator: S = (a - b*c/d)*I + dt*theta*A
            s_diag = (a - b * c / d)
            A = self.negative_diffusion_matrix()
            S_mat = coo_matrix(s_diag * np.eye(self.size) + w * A.toarray()).tocsc()
            lu = splu(S_mat)

            def schur_factory(u, context):
                def apply_schur(rhs):
                    scaled_rhs = np.asarray(context.residual_scale * rhs)
                    rx = scaled_rhs[: self.size]
                    ry = scaled_rhs[self.size :]
                    # Eliminate material block ry:
                    rhs_schur = rx - (b / d) * ry
                    u_sol = lu.solve(rhs_schur)
                    v_sol = (ry - c * u_sol) / d
                    solution = jnp.asarray(np.concatenate((u_sol, v_sol)))
                    return solution / context.state_scale
                return apply_schur
            return schur_factory

        raise ValueError(f"unknown preconditioner kind: {kind}")
