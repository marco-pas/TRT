"""One-step discrete adjoint for an implicit time step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax

from trt_jfnk.solvers.krylov import KrylovOptions, KrylovResult, solve_krylov

from .implicit_step import build_transpose_action


@dataclass(frozen=True)
class AdjointStepResult:
    old_state_cotangent: jax.Array
    multiplier: jax.Array
    linear_result: KrylovResult


def one_step_adjoint(
    step_residual: Callable[[jax.Array, jax.Array], jax.Array],
    new_state: jax.Array,
    old_state: jax.Array,
    objective: Callable[[jax.Array], object],
    krylov_options: KrylovOptions,
) -> AdjointStepResult:
    r"""Differentiate one converged implicit step.

    For ``F(x_{n+1},x_n)=0`` and ``g(x_{n+1})``, solve

    ``F_new.T lambda = grad(g)`` and return ``-F_old.T lambda``.
    """

    objective_gradient = jax.grad(objective)(new_state)
    residual_new = lambda candidate: step_residual(candidate, old_state)
    transpose_new = build_transpose_action(residual_new, new_state)
    linear_result = solve_krylov(transpose_new, objective_gradient, krylov_options)
    multiplier = linear_result.solution
    _value, old_pullback = jax.vjp(lambda previous: step_residual(new_state, previous), old_state)
    old_cotangent = -old_pullback(multiplier)[0]
    return AdjointStepResult(old_cotangent, multiplier, linear_result)
