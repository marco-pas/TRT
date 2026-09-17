"""SciPy CPU and CuPy GPU Krylov backends."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse.linalg import LinearOperator, bicgstab, gmres


@dataclass(frozen=True)
class KrylovOptions:
    backend: Literal["scipy", "cupy", "jax"] = "scipy"
    method: Literal["gmres", "bicgstab"] = "gmres"
    rtol: float = 1.0e-8
    atol: float = 0.0
    max_iterations: int = 300
    restart: int = 50
    jax_solve_method: Literal["batched", "incremental"] = "batched"

    def __post_init__(self) -> None:
        if self.backend not in {"scipy", "cupy", "jax"}:
            raise ValueError("Krylov backend must be scipy, cupy, or jax")
        if self.method not in {"gmres", "bicgstab"}:
            raise ValueError("Krylov method must be gmres or bicgstab")
        if self.rtol < 0.0 or self.atol < 0.0:
            raise ValueError("Krylov tolerances must be nonnegative")
        if self.max_iterations < 1 or self.restart < 1:
            raise ValueError("Krylov iteration limits must be positive")


@dataclass(frozen=True)
class KrylovResult:
    solution: jax.Array
    converged: bool
    iterations: int
    residual_norm: float
    info: int | None


def _as_writable_numpy(value) -> np.ndarray:
    """Host copy that SciPy is free to mutate during orthogonalization."""

    return np.array(jax.device_get(value), copy=True, order="C")


def _solve_scipy(operator, rhs, preconditioner, options: KrylovOptions) -> KrylovResult:
    rhs_numpy = _as_writable_numpy(rhs)
    size = rhs_numpy.size

    def matvec(vector):
        return _as_writable_numpy(operator(jnp.asarray(vector, dtype=rhs.dtype)))

    linear_operator = LinearOperator((size, size), matvec=matvec, dtype=rhs_numpy.dtype)
    preconditioner_operator = None
    if preconditioner is not None:
        def psolve(vector):
            return _as_writable_numpy(preconditioner(jnp.asarray(vector, dtype=rhs.dtype)))

        preconditioner_operator = LinearOperator(
            (size, size), matvec=psolve, dtype=rhs_numpy.dtype
        )

    iteration_count = 0

    def count_iteration(_value):
        nonlocal iteration_count
        iteration_count += 1

    if options.method == "gmres":
        solution, info = gmres(
            linear_operator,
            rhs_numpy,
            M=preconditioner_operator,
            rtol=options.rtol,
            atol=options.atol,
            restart=options.restart,
            maxiter=options.max_iterations,
            callback=count_iteration,
            callback_type="legacy",
        )
    else:
        solution, info = bicgstab(
            linear_operator,
            rhs_numpy,
            M=preconditioner_operator,
            rtol=options.rtol,
            atol=options.atol,
            maxiter=options.max_iterations,
            callback=count_iteration,
        )

    solution_device = jnp.asarray(solution, dtype=rhs.dtype)
    residual_norm = float(jnp.linalg.norm(operator(solution_device) - rhs))
    threshold = max(options.atol, options.rtol * float(jnp.linalg.norm(rhs)))
    return KrylovResult(
        solution=solution_device,
        converged=bool(info == 0 and residual_norm <= max(threshold, 10.0 * np.finfo(rhs_numpy.dtype).eps)),
        iterations=iteration_count,
        residual_norm=residual_norm,
        info=int(info),
    )


def _solve_cupy(operator, rhs, preconditioner, options: KrylovOptions) -> KrylovResult:
    try:
        import cupy as cp
        import cupyx.scipy.sparse.linalg as cupy_spla
        from jax import dlpack as jax_dlpack
    except ImportError as err:
        raise ImportError(
            "The 'cupy' Krylov backend requires CuPy and CUDA. "
            "Please ensure CuPy is installed (e.g., pip install cupy-cuda12x)."
        ) from err

    try:
        rhs_cp = cp.from_dlpack(rhs)
    except Exception:
        rhs_cp = cp.from_dlpack(jax_dlpack.to_dlpack(rhs))

    size = rhs_cp.size
    matvec_count = 0
    iteration_count = 0

    def count_iteration(_value):
        nonlocal iteration_count
        iteration_count += 1

    def matvec(v_cp):
        nonlocal matvec_count
        matvec_count += 1
        v_jax = jax_dlpack.from_dlpack(v_cp)
        out_jax = operator(v_jax)
        return cp.from_dlpack(out_jax)

    linear_operator = cupy_spla.LinearOperator(
        (size, size), matvec=matvec, dtype=rhs_cp.dtype
    )

    preconditioner_operator = None
    if preconditioner is not None:
        def psolve(v_cp):
            v_jax = jax_dlpack.from_dlpack(v_cp)
            out_jax = preconditioner(v_jax)
            return cp.from_dlpack(out_jax)

        preconditioner_operator = cupy_spla.LinearOperator(
            (size, size), matvec=psolve, dtype=rhs_cp.dtype
        )

    if options.method == "gmres":
        solution, info = cupy_spla.gmres(
            linear_operator,
            rhs_cp,
            M=preconditioner_operator,
            rtol=options.rtol,
            atol=options.atol,
            restart=options.restart,
            maxiter=options.max_iterations,
            callback=count_iteration,
        )
        # For GMRES in CuPy, the callback fires only on restarts.
        # Because each Arnoldi step evaluates matvec once, matvec_count gives the exact iteration count.
        total_iters = matvec_count
    else:
        solution, info = cupy_spla.bicgstab(
            linear_operator,
            rhs_cp,
            M=preconditioner_operator,
            rtol=options.rtol,
            atol=options.atol,
            maxiter=options.max_iterations,
            callback=count_iteration,
        )
        # For BiCGSTAB in CuPy, callback is executed every iteration.
        total_iters = iteration_count if iteration_count > 0 else (matvec_count // 2)

    solution_jax = jax_dlpack.from_dlpack(solution)
    if solution_jax.dtype != rhs.dtype:
        solution_jax = solution_jax.astype(rhs.dtype)

    residual_norm = float(jnp.linalg.norm(operator(solution_jax) - rhs))
    threshold = max(options.atol, options.rtol * float(jnp.linalg.norm(rhs)))
    info_int = int(info) if info is not None else 0
    converged = bool(
        info_int == 0
        and residual_norm <= max(threshold, 10.0 * np.finfo(np.dtype(rhs.dtype)).eps)
    )
    return KrylovResult(
        solution=solution_jax,
        converged=converged,
        iterations=total_iters,
        residual_norm=residual_norm,
        info=info_int,
    )


# --- JAX Krylov Solver (Commented out for now; CuPy is used on GPU) ---
# def _solve_jax(operator, rhs, preconditioner, options: KrylovOptions) -> KrylovResult:
#     import jax.scipy.sparse.linalg as jsparse
#
#     preconditioner = (lambda vector: vector) if preconditioner is None else preconditioner
#     if options.method == "gmres":
#         restart_cycles = max(1, math.ceil(options.max_iterations / options.restart))
#         try:
#             solution, info = jsparse.gmres(
#                 operator,
#                 rhs,
#                 tol=options.rtol,
#                 atol=options.atol,
#                 restart=options.restart,
#                 maxiter=restart_cycles,
#                 M=preconditioner,
#                 solve_method=options.jax_solve_method,
#             )
#         except AssertionError:
#             from jax._src.scipy.sparse.linalg import (
#                 _gmres_batched,
#                 _gmres_incremental,
#                 _gmres_solve,
#                 _norm,
#             )
#
#             b_norm = _norm(rhs)
#             atol = jnp.maximum(options.rtol * b_norm, options.atol)
#             Mb = preconditioner(rhs)
#             Mb_norm = _norm(Mb)
#             ptol = Mb_norm * jnp.minimum(1.0, atol / b_norm)
#             gmres_func = (
#                 _gmres_incremental
#                 if options.jax_solve_method == "incremental"
#                 else _gmres_batched
#             )
#             solution = _gmres_solve(
#                 operator,
#                 rhs,
#                 jnp.zeros_like(rhs),
#                 atol,
#                 ptol,
#                 options.restart,
#                 restart_cycles,
#                 preconditioner,
#                 gmres_func,
#             )
#             info = None
#     else:
#         try:
#             solution, info = jsparse.bicgstab(
#                 operator,
#                 rhs,
#                 tol=options.rtol,
#                 atol=options.atol,
#                 maxiter=options.max_iterations,
#                 M=preconditioner,
#             )
#         except AssertionError:
#             from jax._src.scipy.sparse.linalg import _bicgstab_solve, _norm
#
#             b_norm = _norm(rhs)
#             atol = jnp.maximum(options.rtol * b_norm, options.atol)
#             solution = _bicgstab_solve(
#                 operator,
#                 rhs,
#                 jnp.zeros_like(rhs),
#                 options.rtol,
#                 atol,
#                 options.max_iterations,
#                 preconditioner,
#             )
#             info = None
#
#     residual_norm = float(jnp.linalg.norm(operator(solution) - rhs))
#     threshold = max(options.atol, options.rtol * float(jnp.linalg.norm(rhs)))
#     info_value = None if info is None else int(jax.device_get(info))
#     converged = residual_norm <= max(threshold, 10.0 * np.finfo(np.dtype(rhs.dtype)).eps)
#     return KrylovResult(solution, converged, -1, residual_norm, info_value)


def solve_krylov(
    operator: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    options: KrylovOptions,
    preconditioner: Callable[[jax.Array], jax.Array] | None = None,
) -> KrylovResult:
    """Solve a matrix-free linear system on the selected backend."""

    if options.backend == "scipy":
        return _solve_scipy(operator, rhs, preconditioner, options)
    if options.backend == "cupy":
        return _solve_cupy(operator, rhs, preconditioner, options)
    if options.backend == "jax":
        raise NotImplementedError(
            "The pure-JAX Krylov solver is currently commented out in favor of "
            "'cupy' (GPU) and 'scipy' (CPU) backends. Please set backend='cupy' or backend='scipy'."
        )
    raise ValueError(f"Unknown Krylov backend: {options.backend}")
