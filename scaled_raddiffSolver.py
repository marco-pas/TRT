"""Scale-preconditioned JFNK driver for the Su--Olson TRT model.

This module is intentionally additive: place it next to the upstream
``raddiffSolver.py`` file.  It reuses the upstream spatial discretization,
Crank--Nicolson residual, source functions, and boundary-condition routines.

The scaled nonlinear system is

    x = S y,             F_hat(y) = R^{-1} F(S y),
    J_hat(y) p = R^{-1} J(S y) S p.

``S`` and ``R`` are positive diagonal block scales that are frozen for one
time step.  AD differentiates the complete scaled residual, while FD perturbs
the dimensionless variable ``y``.  Thus AD and FD implement the same operator
and the transformed system has exactly the same physical root as the
unscaled system.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import bmat, csr_matrix, diags, identity, lil_matrix
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres

# The add-on supports both layouts:
#
#   JFNK-ADvsFD/raddiffSolver.py
#   JFNK-ADvsFD/scaled_raddiffSolver.py
#
# and the user's convenient subdirectory layout:
#
#   JFNK-ADvsFD/raddiffSolver.py
#   JFNK-ADvsFD/scaled_prec/scaled_raddiffSolver.py
try:
    import raddiffSolver as base
except ModuleNotFoundError as error:
    # Do not mask a missing dependency imported *inside* raddiffSolver.py.
    if error.name != "raddiffSolver":
        raise

    module_path = Path(__file__).resolve()
    repository_root = next(
        (parent for parent in module_path.parents if (parent / "raddiffSolver.py").is_file()),
        None,
    )
    if repository_root is None:
        raise ModuleNotFoundError(
            "Could not locate the upstream raddiffSolver.py. Place scaled_raddiffSolver.py "
            "either in the JFNK-ADvsFD repository root or in a subdirectory beneath it."
        ) from error

    sys.path.insert(0, str(repository_root))
    import raddiffSolver as base


Array = jax.Array
ScaleMode = Literal["none", "fixed", "state"]
JVPMode = Literal["ad", "fd"]
KrylovMethod = Literal["gmres", "bicgstab"]

DIRICHLET = base.DIRICHLET
PERIODIC = base.PERIODIC


@dataclass(frozen=True)
class ScaleConfig:
    """Policy used to construct the frozen block scales for one time step.

    ``fixed`` uses ``u_ref`` and ``v_ref`` directly.  ``state`` uses
    ``max(reference, ||field||_infinity, floor)``.  The latter is useful for
    nonlinear extensions, but scales must still remain frozen inside a Newton
    solve.  If residual references are omitted, R=S (a diagonal similarity
    scaling for the linear Su--Olson Jacobian).
    """

    mode: ScaleMode = "none"
    u_ref: float = 1.0
    v_ref: float = 1.0
    residual_u_ref: float | None = None
    residual_v_ref: float | None = None
    floor: float = 1.0e-12

    def validate(self) -> None:
        if self.mode not in {"none", "fixed", "state"}:
            raise ValueError(f"Unknown scale mode: {self.mode}")
        if not math.isfinite(self.floor) or self.floor <= 0.0:
            raise ValueError("scale floor must be finite and positive")
        if self.mode == "fixed" and (self.u_ref <= 0.0 or self.v_ref <= 0.0):
            raise ValueError("fixed state references must be positive")
        for name, value in (
            ("u_ref", self.u_ref),
            ("v_ref", self.v_ref),
            ("residual_u_ref", self.residual_u_ref),
            ("residual_v_ref", self.residual_v_ref),
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name, value in (
            ("residual_u_ref", self.residual_u_ref),
            ("residual_v_ref", self.residual_v_ref),
        ):
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class ScaleContext:
    """Concrete state and residual scales, frozen during a Newton solve."""

    state_scale: Array
    residual_scale: Array
    u_scale: float
    v_scale: float
    residual_u_scale: float
    residual_v_scale: float
    shape: tuple[int, int]

    @classmethod
    def from_fields(
        cls,
        U: Array,
        V: Array,
        config: ScaleConfig,
    ) -> "ScaleContext":
        config.validate()
        if U.shape != V.shape or U.ndim != 2:
            raise ValueError("U and V must be two-dimensional arrays of equal shape")

        if config.mode == "none":
            su = sv = ru = rv = 1.0
        else:
            if config.mode == "fixed":
                su = max(config.u_ref, config.floor)
                sv = max(config.v_ref, config.floor)
            else:
                u_inf = float(jnp.max(jnp.abs(U))) if U.size else 0.0
                v_inf = float(jnp.max(jnp.abs(V))) if V.size else 0.0
                su = max(config.u_ref, u_inf, config.floor)
                sv = max(config.v_ref, v_inf, config.floor)

            ru = su if config.residual_u_ref is None else config.residual_u_ref
            rv = sv if config.residual_v_ref is None else config.residual_v_ref
            ru = max(ru, config.floor)
            rv = max(rv, config.floor)

        n = U.size
        dtype = U.dtype
        state_scale = jnp.concatenate(
            [jnp.full((n,), su, dtype=dtype), jnp.full((n,), sv, dtype=dtype)]
        )
        residual_scale = jnp.concatenate(
            [jnp.full((n,), ru, dtype=dtype), jnp.full((n,), rv, dtype=dtype)]
        )
        return cls(
            state_scale=state_scale,
            residual_scale=residual_scale,
            u_scale=float(su),
            v_scale=float(sv),
            residual_u_scale=float(ru),
            residual_v_scale=float(rv),
            shape=tuple(U.shape),
        )

    def to_scaled_state(self, physical_state: Array) -> Array:
        return physical_state / self.state_scale

    def to_physical_state(self, scaled_state: Array) -> Array:
        return self.state_scale * scaled_state

    def to_scaled_residual(self, physical_residual: Array) -> Array:
        return physical_residual / self.residual_scale

    def to_physical_correction(self, scaled_correction: Array) -> Array:
        return self.state_scale * scaled_correction


def scaled_residual_flat(
    scaled_state: Array,
    state_scale: Array,
    residual_scale: Array,
    U_old: Array,
    V_old: Array,
    lap_U_old: Array,
    dt: float,
    epsilon: float,
    Q_k: Array,
    Q_old: Array,
    dx: float,
    dy: float,
    bc_x: str,
    bc_y: str,
    Nx: int,
    Ny: int,
) -> Array:
    """Evaluate R^{-1} F(S y) using the upstream residual graph."""

    physical_state = state_scale * scaled_state
    physical_residual = base.residual_flat(
        physical_state,
        U_old,
        V_old,
        lap_U_old,
        dt,
        epsilon,
        Q_k,
        Q_old,
        dx,
        dy,
        bc_x,
        bc_y,
        Nx,
        Ny,
    )
    return physical_residual / residual_scale


@partial(jax.jit, static_argnums=(13, 14, 15, 16))
def JacobianActionADScaled(
    scaled_state: Array,
    direction: Array,
    state_scale: Array,
    residual_scale: Array,
    U_old: Array,
    V_old: Array,
    lap_U_old: Array,
    dt: float,
    epsilon: float,
    Q_k: Array,
    Q_old: Array,
    dx: float,
    dy: float,
    bc_x: str,
    bc_y: str,
    Nx: int,
    Ny: int,
) -> Array:
    """Exact AD action R^{-1} J(S y) S p."""

    def transformed_residual(y: Array) -> Array:
        return scaled_residual_flat(
            y,
            state_scale,
            residual_scale,
            U_old,
            V_old,
            lap_U_old,
            dt,
            epsilon,
            Q_k,
            Q_old,
            dx,
            dy,
            bc_x,
            bc_y,
            Nx,
            Ny,
        )

    _, action = jax.jvp(transformed_residual, (scaled_state,), (direction,))
    return action


@partial(jax.jit, static_argnums=(13, 14, 15, 16))
def JacobianActionFDScaled(
    scaled_state: Array,
    direction: Array,
    state_scale: Array,
    residual_scale: Array,
    U_old: Array,
    V_old: Array,
    lap_U_old: Array,
    dt: float,
    epsilon: float,
    Q_k: Array,
    Q_old: Array,
    dx: float,
    dy: float,
    bc_x: str,
    bc_y: str,
    Nx: int,
    Ny: int,
) -> Array:
    """Forward-difference action on the same dimensionless residual graph.

    The perturbation size is selected in y-space.  This is the key comparison
    requirement: an FD action in physical coordinates would be a different
    numerical experiment when U and V have disparate scales.
    """

    machine_eps = jnp.finfo(scaled_state.dtype).eps
    step_base = jnp.sqrt(machine_eps)
    state_norm = jnp.linalg.norm(scaled_state)
    direction_norm = jnp.linalg.norm(direction)
    safe_direction_norm = jnp.where(direction_norm > 0.0, direction_norm, 1.0)
    h = step_base * jnp.maximum(1.0, state_norm) / safe_direction_norm
    h = jnp.clip(h, step_base, jnp.sqrt(step_base))
    h = jnp.where(jnp.isfinite(h), h, step_base)

    residual = scaled_residual_flat(
        scaled_state,
        state_scale,
        residual_scale,
        U_old,
        V_old,
        lap_U_old,
        dt,
        epsilon,
        Q_k,
        Q_old,
        dx,
        dy,
        bc_x,
        bc_y,
        Nx,
        Ny,
    )
    residual_perturbed = scaled_residual_flat(
        scaled_state + h * direction,
        state_scale,
        residual_scale,
        U_old,
        V_old,
        lap_U_old,
        dt,
        epsilon,
        Q_k,
        Q_old,
        dx,
        dy,
        bc_x,
        bc_y,
        Nx,
        Ny,
    )
    return (residual_perturbed - residual) / h


def scaled_residual_norm(physical_residual: Array, residual_scale: Array) -> float:
    """Dimensionless residual norm used for Newton stopping and globalization."""

    return float(jnp.linalg.norm(physical_residual / residual_scale))


def scaled_merit(scaled_residual: Array) -> float:
    """Return phi(y)=1/2 ||F_hat(y)||_2^2."""

    value = jnp.vdot(scaled_residual, scaled_residual).real
    return 0.5 * float(value)


def scaled_newton_stopping_test(
    scaled_residual: Array,
    initial_scaled_norm: float,
    atol: float,
    rtol: float,
) -> tuple[bool, float, float]:
    """Check ||F_hat_k|| <= atol + rtol ||F_hat_0||."""

    norm_now = float(jnp.linalg.norm(scaled_residual))
    threshold = float(atol + rtol * initial_scaled_norm)
    return norm_now <= threshold, norm_now, threshold


@dataclass(frozen=True)
class LineSearchResult:
    scaled_state: Array
    scaled_residual: Array
    alpha: float
    backtracks: int
    accepted: bool
    armijo_satisfied: bool
    merit: float


def scaled_backtracking_line_search(
    scaled_state: Array,
    scaled_step: Array,
    residual_fn: Callable[[Array], Array],
    current_residual: Array | None = None,
    project_fn: Callable[[Array], Array] | None = None,
    armijo: float = 1.0e-4,
    contraction: float = 0.5,
    max_backtracks: int = 12,
) -> LineSearchResult:
    """Backtrack on the dimensionless merit function.

    For an exact Newton direction, d(phi)/d(alpha) at zero is
    ``-||F_hat||^2``.  The condition below is the corresponding Armijo test.
    If roundoff prevents the strict test, the best strictly decreasing trial
    is accepted; a non-decreasing step is rejected.
    """

    if not 0.0 < armijo < 1.0:
        raise ValueError("armijo must lie in (0, 1)")
    if not 0.0 < contraction < 1.0:
        raise ValueError("contraction must lie in (0, 1)")
    if max_backtracks < 0:
        raise ValueError("max_backtracks must be nonnegative")

    residual_0 = residual_fn(scaled_state) if current_residual is None else current_residual
    merit_0 = scaled_merit(residual_0)
    best_state = scaled_state
    best_residual = residual_0
    best_merit = merit_0
    best_alpha = 0.0
    best_backtracks = 0

    for backtracks in range(max_backtracks + 1):
        alpha = contraction**backtracks
        trial_state = scaled_state + alpha * scaled_step
        if project_fn is not None:
            trial_state = project_fn(trial_state)
        trial_residual = residual_fn(trial_state)
        trial_merit = scaled_merit(trial_residual)

        if math.isfinite(trial_merit) and trial_merit < best_merit:
            best_state = trial_state
            best_residual = trial_residual
            best_merit = trial_merit
            best_alpha = alpha
            best_backtracks = backtracks

        if math.isfinite(trial_merit) and trial_merit <= (1.0 - 2.0 * armijo * alpha) * merit_0:
            return LineSearchResult(
                scaled_state=trial_state,
                scaled_residual=trial_residual,
                alpha=float(alpha),
                backtracks=backtracks,
                accepted=True,
                armijo_satisfied=True,
                merit=trial_merit,
            )

    accepted = best_merit < merit_0
    return LineSearchResult(
        scaled_state=best_state,
        scaled_residual=best_residual,
        alpha=float(best_alpha),
        backtracks=best_backtracks,
        accepted=accepted,
        armijo_satisfied=False,
        merit=best_merit,
    )


def assemble_laplacian_matrix(
    Nx: int,
    Ny: int,
    dx: float,
    dy: float,
    bc_x: str = DIRICHLET,
    bc_y: str = PERIODIC,
    dtype: np.dtype | type = np.float64,
) -> csr_matrix:
    """Assemble the matrix exactly matching ``raddiffSolver.laplacian``.

    Arrays are flattened in C order, i.e. index ``i*Ny+j``.  The upstream
    operator zeroes an entire Laplacian row on every Dirichlet boundary; this
    detail is preserved here for regression testing.
    """

    if Nx < 2 or Ny < 2:
        raise ValueError("Nx and Ny must both be at least two")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx and dy must be positive")
    if bc_x not in {DIRICHLET, PERIODIC} or bc_y not in {DIRICHLET, PERIODIC}:
        raise ValueError("boundary conditions must be 'dirichlet' or 'periodic'")

    n = Nx * Ny
    matrix = lil_matrix((n, n), dtype=dtype)

    def index(i: int, j: int) -> int:
        return i * Ny + j

    for i in range(Nx):
        for j in range(Ny):
            row = index(i, j)
            on_zero_row = (
                (bc_x == DIRICHLET and i in {0, Nx - 1})
                or (bc_y == DIRICHLET and j in {0, Ny - 1})
            )
            if on_zero_row:
                continue

            inv_dx2 = 1.0 / dx**2
            if bc_x == PERIODIC:
                matrix[row, index((i - 1) % Nx, j)] += inv_dx2
                matrix[row, index((i + 1) % Nx, j)] += inv_dx2
                matrix[row, row] -= 2.0 * inv_dx2
            else:
                matrix[row, index(i - 1, j)] += inv_dx2
                matrix[row, index(i + 1, j)] += inv_dx2
                matrix[row, row] -= 2.0 * inv_dx2

            inv_dy2 = 1.0 / dy**2
            if bc_y == PERIODIC:
                matrix[row, index(i, (j - 1) % Ny)] += inv_dy2
                matrix[row, index(i, (j + 1) % Ny)] += inv_dy2
                matrix[row, row] -= 2.0 * inv_dy2
            else:
                matrix[row, index(i, j - 1)] += inv_dy2
                matrix[row, index(i, j + 1)] += inv_dy2
                matrix[row, row] -= 2.0 * inv_dy2

    return matrix.tocsr()


def assemble_su_olson_jacobian(
    Nx: int,
    Ny: int,
    dt: float,
    epsilon: float,
    dx: float,
    dy: float,
    bc_x: str = DIRICHLET,
    bc_y: str = PERIODIC,
    dtype: np.dtype | type = np.float64,
) -> csr_matrix:
    """Exact Jacobian of the linear Crank--Nicolson Su--Olson residual.

    J = [[I - dt/6 L + dt/2 I,        -dt/2 I],
         [             -epsilon dt/2 I, I + epsilon dt/2 I]].
    """

    n = Nx * Ny
    lap = assemble_laplacian_matrix(Nx, Ny, dx, dy, bc_x, bc_y, dtype=dtype)
    eye = identity(n, format="csr", dtype=dtype)
    juu = (1.0 + 0.5 * dt) * eye - (dt / 6.0) * lap
    juv = (-0.5 * dt) * eye
    jvu = (-0.5 * epsilon * dt) * eye
    jvv = (1.0 + 0.5 * epsilon * dt) * eye
    return bmat([[juu, juv], [jvu, jvv]], format="csr")


def assemble_scaled_su_olson_jacobian(
    jacobian: csr_matrix,
    context: ScaleContext,
) -> csr_matrix:
    """Explicitly form R^{-1} J S for small-grid verification only."""

    state_scale = np.asarray(jax.device_get(context.state_scale))
    residual_scale = np.asarray(jax.device_get(context.residual_scale))
    if jacobian.shape != (state_scale.size, state_scale.size):
        raise ValueError("Jacobian and ScaleContext sizes do not match")
    return (diags(1.0 / residual_scale) @ jacobian @ diags(state_scale)).tocsr()


@dataclass(frozen=True)
class NewtonKrylovOptions:
    jvp: JVPMode = "ad"
    krylov: KrylovMethod = "gmres"
    newton_atol: float = 1.0e-10
    newton_rtol: float = 1.0e-8
    max_newton: int = 20
    krylov_rtol: float = 1.0e-8
    krylov_atol: float = 0.0
    max_krylov: int = 200
    restart: int = 50
    armijo: float = 1.0e-4
    backtrack_factor: float = 0.5
    max_backtracks: int = 12

    def validate(self) -> None:
        if self.jvp not in {"ad", "fd"}:
            raise ValueError(f"Unknown JVP mode: {self.jvp}")
        if self.krylov not in {"gmres", "bicgstab"}:
            raise ValueError(f"Unknown Krylov method: {self.krylov}")
        if self.newton_atol < 0.0 or self.newton_rtol < 0.0:
            raise ValueError("Newton tolerances must be nonnegative")
        if self.krylov_atol < 0.0 or self.krylov_rtol <= 0.0:
            raise ValueError("Krylov tolerances are invalid")
        if self.max_newton < 0 or self.max_krylov <= 0 or self.restart <= 0:
            raise ValueError("iteration limits must be positive")


@dataclass(frozen=True)
class NewtonIterationRecord:
    iteration: int
    scaled_residual_before: float
    physical_residual_before: float
    krylov_iterations: int
    krylov_info: int
    alpha: float
    backtracks: int
    armijo_satisfied: bool


@dataclass(frozen=True)
class StepStatistics:
    converged: bool
    reason: str
    newton_corrections: int
    total_krylov_iterations: int
    initial_scaled_residual: float
    final_scaled_residual: float
    final_physical_residual: float
    threshold: float
    elapsed_seconds: float
    history: tuple[NewtonIterationRecord, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StepResult:
    U: Array
    V: Array
    scale_context: ScaleContext
    statistics: StepStatistics


class _KrylovCounter:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, *_args: object) -> None:
        self.count += 1


def _host(array: Array) -> np.ndarray:
    """Transfer a JAX array to an owned, writable, C-contiguous host array.

    ``np.asarray(jax.device_get(array))`` may expose JAX's underlying CPU
    buffer as a read-only NumPy view.  SciPy's GMRES modifies each matrix-vector
    product in place during modified Gram--Schmidt, so returning that view
    causes ``ValueError: output array is read-only``.  An explicit copy gives
    SciPy ownership of the work vector.
    """

    return np.array(jax.device_get(array), copy=True, order="C")


def _project_scaled_state(
    scaled_state: Array,
    context: ScaleContext,
    bc_x: str,
    bc_y: str,
    Nx: int,
    Ny: int,
) -> Array:
    physical = context.to_physical_state(scaled_state)
    U, V = base.flattenJnp(physical, Nx, Ny)
    U = base.apply_BC(U, bc_x, bc_y)
    V = base.apply_BC(V, bc_x, bc_y)
    return context.to_scaled_state(base.concatenateJnp(U, V))


def solve_su_olson_step(
    U_old: Array,
    V_old: Array,
    Q_old: Array,
    Q_new: Array,
    dt: float,
    epsilon: float,
    dx: float,
    dy: float,
    bc_x: str = DIRICHLET,
    bc_y: str = PERIODIC,
    scale_config: ScaleConfig | None = None,
    options: NewtonKrylovOptions | None = None,
    initial_guess: tuple[Array, Array] | None = None,
) -> StepResult:
    """Solve one Crank--Nicolson step in scaled coordinates."""

    if U_old.shape != V_old.shape or U_old.shape != Q_old.shape or U_old.shape != Q_new.shape:
        raise ValueError("U, V, Q_old, and Q_new must have identical shapes")
    if U_old.ndim != 2:
        raise ValueError("fields must be two-dimensional")
    if dt <= 0.0 or epsilon <= 0.0:
        raise ValueError("dt and epsilon must be positive")

    scale_config = ScaleConfig() if scale_config is None else scale_config
    options = NewtonKrylovOptions() if options is None else options
    scale_config.validate()
    options.validate()

    Nx, Ny = U_old.shape
    if initial_guess is None:
        U_guess, V_guess = U_old, V_old
    else:
        U_guess, V_guess = initial_guess
        if U_guess.shape != U_old.shape or V_guess.shape != V_old.shape:
            raise ValueError("initial guess shape does not match old state")

    # This is deliberately outside the Newton loop: S and R are frozen.
    context = ScaleContext.from_fields(U_guess, V_guess, scale_config)
    lap_U_old = base.laplacian(U_old, dx, dy, bc_x=bc_x, bc_y=bc_y)
    physical_guess = base.concatenateJnp(U_guess, V_guess)
    scaled_state = context.to_scaled_state(physical_guess)
    scaled_state = _project_scaled_state(scaled_state, context, bc_x, bc_y, Nx, Ny)

    def residual_fn(y: Array) -> Array:
        return scaled_residual_flat(
            y,
            context.state_scale,
            context.residual_scale,
            U_old,
            V_old,
            lap_U_old,
            dt,
            epsilon,
            Q_new,
            Q_old,
            dx,
            dy,
            bc_x,
            bc_y,
            Nx,
            Ny,
        )

    def project_fn(y: Array) -> Array:
        return _project_scaled_state(y, context, bc_x, bc_y, Nx, Ny)

    start = time.perf_counter()
    scaled_residual = residual_fn(scaled_state)
    initial_norm = float(jnp.linalg.norm(scaled_residual))
    threshold = options.newton_atol + options.newton_rtol * initial_norm
    records: list[NewtonIterationRecord] = []
    total_krylov = 0
    converged = False
    reason = "maximum Newton iterations reached"

    for newton_iteration in range(options.max_newton + 1):
        converged, scaled_norm, threshold = scaled_newton_stopping_test(
            scaled_residual,
            initial_norm,
            options.newton_atol,
            options.newton_rtol,
        )
        if converged:
            reason = "scaled residual tolerance satisfied"
            break
        if newton_iteration == options.max_newton:
            break

        physical_norm = float(
            jnp.linalg.norm(scaled_residual * context.residual_scale)
        )

        if options.jvp == "ad":
            jvp_function = JacobianActionADScaled
        else:
            jvp_function = JacobianActionFDScaled

        def matvec(host_direction: np.ndarray) -> np.ndarray:
            direction = jnp.asarray(host_direction, dtype=scaled_state.dtype)
            action = jvp_function(
                scaled_state,
                direction,
                context.state_scale,
                context.residual_scale,
                U_old,
                V_old,
                lap_U_old,
                dt,
                epsilon,
                Q_new,
                Q_old,
                dx,
                dy,
                bc_x,
                bc_y,
                Nx,
                Ny,
            )
            return _host(action)

        dimension = scaled_state.size
        operator = LinearOperator(
            (dimension, dimension),
            matvec=matvec,
            dtype=_host(scaled_state).dtype,
        )
        rhs = -_host(scaled_residual)
        counter = _KrylovCounter()
        if options.krylov == "gmres":
            scaled_step_host, info = gmres(
                operator,
                rhs,
                rtol=options.krylov_rtol,
                atol=options.krylov_atol,
                restart=min(options.restart, dimension),
                maxiter=options.max_krylov,
                callback=counter,
                callback_type="pr_norm",
            )
        else:
            scaled_step_host, info = bicgstab(
                operator,
                rhs,
                rtol=options.krylov_rtol,
                atol=options.krylov_atol,
                maxiter=options.max_krylov,
                callback=counter,
            )
        total_krylov += counter.count
        if info < 0 or not np.all(np.isfinite(scaled_step_host)):
            reason = f"{options.krylov} breakdown (info={info})"
            break

        scaled_step = jnp.asarray(scaled_step_host, dtype=scaled_state.dtype)
        line_search = scaled_backtracking_line_search(
            scaled_state,
            scaled_step,
            residual_fn,
            current_residual=scaled_residual,
            project_fn=project_fn,
            armijo=options.armijo,
            contraction=options.backtrack_factor,
            max_backtracks=options.max_backtracks,
        )
        records.append(
            NewtonIterationRecord(
                iteration=newton_iteration,
                scaled_residual_before=scaled_norm,
                physical_residual_before=physical_norm,
                krylov_iterations=counter.count,
                krylov_info=int(info),
                alpha=line_search.alpha,
                backtracks=line_search.backtracks,
                armijo_satisfied=line_search.armijo_satisfied,
            )
        )
        if not line_search.accepted:
            reason = "scaled merit line search failed to decrease"
            break

        scaled_state = line_search.scaled_state
        scaled_residual = line_search.scaled_residual

    physical_state = context.to_physical_state(scaled_state)
    U_new, V_new = base.flattenJnp(physical_state, Nx, Ny)
    U_new = base.apply_BC(U_new, bc_x, bc_y)
    V_new = base.apply_BC(V_new, bc_x, bc_y)
    final_scaled_norm = float(jnp.linalg.norm(scaled_residual))
    final_physical_norm = float(
        jnp.linalg.norm(scaled_residual * context.residual_scale)
    )
    elapsed = time.perf_counter() - start
    statistics = StepStatistics(
        converged=converged,
        reason=reason,
        newton_corrections=len(records),
        total_krylov_iterations=total_krylov,
        initial_scaled_residual=initial_norm,
        final_scaled_residual=final_scaled_norm,
        final_physical_residual=final_physical_norm,
        threshold=float(threshold),
        elapsed_seconds=elapsed,
        history=tuple(records),
    )
    return StepResult(U=U_new, V=V_new, scale_context=context, statistics=statistics)


def _resolve_problem_configuration(args: argparse.Namespace) -> tuple[str, str, str, str]:
    if args.preset == "classic-su-olson":
        initial_condition = "SO"
        source = "central"
        bc_x = DIRICHLET
        bc_y = PERIODIC
    else:
        initial_condition = "SO"
        source = "pulsar"
        bc_x = DIRICHLET
        bc_y = DIRICHLET

    return (
        args.initial_condition or initial_condition,
        args.source or source,
        args.bc_x or bc_x,
        args.bc_y or bc_y,
    )


def run_scaled_simulation(args: argparse.Namespace) -> dict[str, object]:
    """Run a command-line configured TRT experiment and return its data."""

    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")
    elif not getattr(base, "HAS_GPU_LIBS", False):
        raise ImportError("--device gpu requires the optional GPU libraries used upstream")

    dtype = base.configure_precision(args.precision)
    initial_condition, source, bc_x, bc_y = _resolve_problem_configuration(args)

    x = jnp.linspace(
        args.x_min,
        args.x_max,
        args.nx,
        endpoint=(bc_x == DIRICHLET),
        dtype=dtype,
    )
    y = jnp.linspace(
        args.y_min,
        args.y_max,
        args.ny,
        endpoint=(bc_y == DIRICHLET),
        dtype=dtype,
    )
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    U, V = base.get_initial_conditions(X, Y, dtype, initial_condition)
    U = base.apply_BC(U, bc_x, bc_y)
    V = base.apply_BC(V, bc_x, bc_y)

    scale_config = ScaleConfig(
        mode=args.scale_mode,
        u_ref=args.scale_u,
        v_ref=args.scale_v,
        residual_u_ref=args.residual_scale_u,
        residual_v_ref=args.residual_scale_v,
        floor=args.scale_floor,
    )
    options = NewtonKrylovOptions(
        jvp=args.jvp,
        krylov=args.krylov,
        newton_atol=args.newton_atol,
        newton_rtol=args.newton_rtol,
        max_newton=args.max_newton,
        krylov_rtol=args.krylov_rtol,
        krylov_atol=args.krylov_atol,
        max_krylov=args.max_krylov,
        restart=args.restart,
        armijo=args.armijo,
        backtrack_factor=args.backtrack_factor,
        max_backtracks=args.max_backtracks,
    )

    tau = 0.0
    metrics: list[dict[str, object]] = []
    wall_start = time.perf_counter()
    for step in range(args.steps):
        dt = (
            args.dt
            if args.dt is not None
            else base.calc_dt(dx, dy, args.epsilon, args.courant)
        )
        Q_old = base.get_source_term(
            source,
            X,
            Y,
            tau,
            args.q0,
            args.x_source,
            args.tau_source,
            args.source_center,
            dtype,
        )
        Q_new = base.get_source_term(
            source,
            X,
            Y,
            tau + dt,
            args.q0,
            args.x_source,
            args.tau_source,
            args.source_center,
            dtype,
        )
        result = solve_su_olson_step(
            U,
            V,
            Q_old,
            Q_new,
            dt,
            args.epsilon,
            dx,
            dy,
            bc_x=bc_x,
            bc_y=bc_y,
            scale_config=scale_config,
            options=options,
        )
        U, V = result.U, result.V
        tau += dt
        stats = result.statistics
        row = {
            "step": step + 1,
            "tau": tau,
            "dt": dt,
            "converged": stats.converged,
            "newton_corrections": stats.newton_corrections,
            "krylov_iterations": stats.total_krylov_iterations,
            "initial_scaled_residual": stats.initial_scaled_residual,
            "final_scaled_residual": stats.final_scaled_residual,
            "final_physical_residual": stats.final_physical_residual,
            "elapsed_seconds": stats.elapsed_seconds,
            "u_scale": result.scale_context.u_scale,
            "v_scale": result.scale_context.v_scale,
            "residual_u_scale": result.scale_context.residual_u_scale,
            "residual_v_scale": result.scale_context.residual_v_scale,
            "reason": stats.reason,
        }
        metrics.append(row)
        if not args.quiet:
            print(
                f"step={step + 1:5d} tau={tau:.6e} "
                f"Newton={stats.newton_corrections:2d} "
                f"Krylov={stats.total_krylov_iterations:4d} "
                f"||Fhat||={stats.final_scaled_residual:.3e} "
                f"S=({result.scale_context.u_scale:.3e},"
                f"{result.scale_context.v_scale:.3e}) "
                f"converged={stats.converged}"
            )
        if not stats.converged and args.fail_on_nonconvergence:
            raise RuntimeError(
                f"time step {step + 1} failed: {stats.reason}; "
                f"||Fhat||={stats.final_scaled_residual:.3e}"
            )

    wall_seconds = time.perf_counter() - wall_start
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path,
            U=_host(U),
            V=_host(V),
            x=_host(x),
            y=_host(y),
            tau=np.asarray(tau),
        )
    if args.metrics_csv is not None:
        metrics_path = Path(args.metrics_csv)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()) if metrics else [])
            if metrics:
                writer.writeheader()
                writer.writerows(metrics)

    if not args.quiet:
        total_krylov = sum(int(row["krylov_iterations"]) for row in metrics)
        print(
            f"completed {args.steps} steps at tau={tau:.6e} in {wall_seconds:.3f} s; "
            f"total Krylov iterations={total_krylov}"
        )
    return {
        "U": U,
        "V": V,
        "x": x,
        "y": y,
        "tau": tau,
        "metrics": metrics,
        "wall_seconds": wall_seconds,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scale-preconditioned AD/FD JFNK solver for Su--Olson TRT"
    )
    parser.add_argument("--preset", choices=["classic-su-olson", "dynamic"], default="classic-su-olson")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--precision", choices=["float32", "float64"], default="float64")
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--x-min", type=float, default=-5.0)
    parser.add_argument("--x-max", type=float, default=5.0)
    parser.add_argument("--y-min", type=float, default=-5.0)
    parser.add_argument("--y-max", type=float, default=5.0)
    parser.add_argument("--bc-x", choices=[DIRICHLET, PERIODIC], default=None)
    parser.add_argument("--bc-y", choices=[DIRICHLET, PERIODIC], default=None)
    parser.add_argument("--initial-condition", choices=["SO", "checkerboard", "rings"], default=None)
    parser.add_argument("--source", choices=["central", "pulsar"], default=None)

    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--q0", type=float, default=1.0)
    parser.add_argument("--x-source", type=float, default=0.5)
    parser.add_argument("--tau-source", type=float, default=float("inf"))
    parser.add_argument("--source-center", type=float, default=0.0)
    parser.add_argument("--courant", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=None, help="fixed dt; otherwise use upstream calc_dt")

    parser.add_argument("--jvp", choices=["ad", "fd"], default="ad")
    parser.add_argument("--krylov", choices=["gmres", "bicgstab"], default="gmres")
    parser.add_argument("--newton-atol", type=float, default=1.0e-10)
    parser.add_argument("--newton-rtol", type=float, default=1.0e-8)
    parser.add_argument("--max-newton", type=int, default=20)
    parser.add_argument("--krylov-rtol", type=float, default=1.0e-8)
    parser.add_argument("--krylov-atol", type=float, default=0.0)
    parser.add_argument("--max-krylov", type=int, default=200)
    parser.add_argument("--restart", type=int, default=50)
    parser.add_argument("--armijo", type=float, default=1.0e-4)
    parser.add_argument("--backtrack-factor", type=float, default=0.5)
    parser.add_argument("--max-backtracks", type=int, default=12)

    parser.add_argument("--scale-mode", choices=["none", "fixed", "state"], default="none")
    parser.add_argument("--scale-u", type=float, default=1.0)
    parser.add_argument("--scale-v", type=float, default=1.0)
    parser.add_argument("--residual-scale-u", type=float, default=None)
    parser.add_argument("--residual-scale-v", type=float, default=None)
    parser.add_argument("--scale-floor", type=float, default=1.0e-12)

    parser.add_argument("--output", type=str, default=None, help="optional final-state .npz path")
    parser.add_argument("--metrics-csv", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--fail-on-nonconvergence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.nx < 2 or args.ny < 2 or args.steps < 1:
        raise ValueError("nx, ny, and steps must satisfy nx>=2, ny>=2, steps>=1")
    run_scaled_simulation(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
