"""Scaled Jacobian-free Newton--Krylov nonlinear solve."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp

from .block_preconditioners import PreconditionerFactory
from .jvp import JVPOptions, build_jacobian_action
from .krylov import KrylovOptions, solve_krylov
from .scaling import ScaleContext, ScalePolicy, build_scale_context


@dataclass(frozen=True)
class NewtonOptions:
    atol: float = 1.0e-10
    rtol: float = 1.0e-8
    max_iterations: int = 20
    armijo: float = 1.0e-4
    backtrack_factor: float = 0.5
    max_backtracks: int = 14
    require_krylov_convergence: bool = True
    jvp: JVPOptions = field(default_factory=JVPOptions)
    krylov: KrylovOptions = field(default_factory=KrylovOptions)

    def __post_init__(self) -> None:
        if self.atol < 0.0 or self.rtol < 0.0 or self.max_iterations < 1:
            raise ValueError("invalid Newton tolerance or iteration limit")
        if not 0.0 < self.armijo < 1.0:
            raise ValueError("armijo must lie in (0,1)")
        if not 0.0 < self.backtrack_factor < 1.0:
            raise ValueError("backtrack_factor must lie in (0,1)")


@dataclass(frozen=True)
class NewtonRecord:
    iteration: int
    scaled_residual_norm: float
    physical_residual_norm: float
    krylov_iterations: int
    krylov_residual_norm: float
    step_length: float
    backtracks: int


@dataclass(frozen=True)
class NewtonResult:
    state: jax.Array
    converged: bool
    reason: str
    iterations: int
    total_krylov_iterations: int
    scaled_residual_norm: float
    physical_residual_norm: float
    scale_context: ScaleContext
    history: tuple[NewtonRecord, ...]


def _host_bool(value) -> bool:
    return bool(jax.device_get(value))


def newton_krylov(
    residual_fn: Callable[[jax.Array], jax.Array],
    initial_state: jax.Array,
    block_size: int,
    scale_policy: ScalePolicy,
    options: NewtonOptions,
    admissible_fn: Callable[[jax.Array], object] | None = None,
    preconditioner_factory: PreconditionerFactory | None = None,
) -> NewtonResult:
    r"""Solve ``F(x)=0`` through ``x=S y`` and ``Fhat=R^{-1}F(Sy)``."""

    context = build_scale_context(initial_state, block_size, scale_policy)
    scaled_state = context.to_scaled_state(initial_state)

    def scaled_residual(y):
        return context.to_scaled_residual(residual_fn(context.to_physical_state(y)))

    # Residual evaluation can be compiled independently from the JVP closure.
    scaled_residual_eval = jax.jit(scaled_residual) if options.jvp.jit else scaled_residual
    residual = scaled_residual_eval(scaled_state)
    initial_norm = float(jnp.linalg.norm(residual))
    threshold = options.atol + options.rtol * initial_norm
    history: list[NewtonRecord] = []
    total_krylov = 0
    krylov_iterations_known = True

    for iteration in range(options.max_iterations + 1):
        scaled_norm = float(jnp.linalg.norm(residual))
        physical_norm = float(jnp.linalg.norm(residual_fn(context.to_physical_state(scaled_state))))
        if not math.isfinite(scaled_norm):
            reason = "non-finite residual"
            break
        if scaled_norm <= threshold:
            return NewtonResult(
                state=context.to_physical_state(scaled_state),
                converged=True,
                reason="converged",
                iterations=iteration,
                total_krylov_iterations=(total_krylov if krylov_iterations_known else -1),
                scaled_residual_norm=scaled_norm,
                physical_residual_norm=physical_norm,
                scale_context=context,
                history=tuple(history),
            )
        if iteration == options.max_iterations:
            reason = "maximum Newton iterations reached"
            break

        jacobian_action = build_jacobian_action(scaled_residual, scaled_state, options.jvp)
        physical_state = context.to_physical_state(scaled_state)
        preconditioner = (
            None
            if preconditioner_factory is None
            else preconditioner_factory(physical_state, context)
        )
        linear = solve_krylov(jacobian_action, -residual, options.krylov, preconditioner)
        if linear.iterations >= 0:
            total_krylov += linear.iterations
        else:
            krylov_iterations_known = False
        if options.require_krylov_convergence and not linear.converged:
            reason = f"Krylov solve failed (info={linear.info})"
            history.append(
                NewtonRecord(
                    iteration=iteration,
                    scaled_residual_norm=scaled_norm,
                    physical_residual_norm=physical_norm,
                    krylov_iterations=linear.iterations,
                    krylov_residual_norm=linear.residual_norm,
                    step_length=0.0,
                    backtracks=0,
                )
            )
            break

        merit = 0.5 * scaled_norm**2
        accepted = False
        chosen_alpha = 0.0
        chosen_backtracks = 0
        trial_state = scaled_state
        trial_residual = residual
        for backtracks in range(options.max_backtracks + 1):
            alpha = options.backtrack_factor**backtracks
            candidate = scaled_state + alpha * linear.solution
            candidate_physical = context.to_physical_state(candidate)
            if admissible_fn is not None and not _host_bool(admissible_fn(candidate_physical)):
                continue
            candidate_residual = scaled_residual_eval(candidate)
            candidate_norm = float(jnp.linalg.norm(candidate_residual))
            candidate_merit = 0.5 * candidate_norm**2
            if math.isfinite(candidate_merit) and candidate_merit <= (
                1.0 - 2.0 * options.armijo * alpha
            ) * merit:
                accepted = True
                chosen_alpha = alpha
                chosen_backtracks = backtracks
                trial_state = candidate
                trial_residual = candidate_residual
                break

        history.append(
            NewtonRecord(
                iteration=iteration,
                scaled_residual_norm=scaled_norm,
                physical_residual_norm=physical_norm,
                krylov_iterations=linear.iterations,
                krylov_residual_norm=linear.residual_norm,
                step_length=chosen_alpha,
                backtracks=chosen_backtracks,
            )
        )
        if not accepted:
            reason = "line search failed"
            break
        scaled_state = trial_state
        residual = trial_residual
    else:  # pragma: no cover - loop exits through return/break
        reason = "maximum Newton iterations reached"

    final_scaled_norm = float(jnp.linalg.norm(residual))
    final_state = context.to_physical_state(scaled_state)
    final_physical_norm = float(jnp.linalg.norm(residual_fn(final_state)))
    return NewtonResult(
        state=final_state,
        converged=False,
        reason=reason,
        iterations=len(history),
        total_krylov_iterations=(total_krylov if krylov_iterations_known else -1),
        scaled_residual_norm=final_scaled_norm,
        physical_residual_norm=final_physical_norm,
        scale_context=context,
        history=tuple(history),
    )
