"""Automatic-differentiation and finite-difference Jacobian actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class JVPOptions:
    mode: Literal["ad", "fd"] = "ad"
    fd_scheme: Literal["forward", "central"] = "forward"
    fd_relative_step: float | None = None
    jit: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"ad", "fd"}:
            raise ValueError("JVP mode must be ad or fd")
        if self.fd_scheme not in {"forward", "central"}:
            raise ValueError("FD scheme must be forward or central")
        if self.fd_relative_step is not None and self.fd_relative_step <= 0.0:
            raise ValueError("FD relative step must be positive")


def _fd_step(state, direction, relative_step: float | None):
    epsilon = jnp.finfo(state.dtype).eps
    base = jnp.sqrt(epsilon) if relative_step is None else relative_step
    direction_norm = jnp.linalg.norm(direction)
    denominator = jnp.where(direction_norm > 0.0, direction_norm, 1.0)
    step = base * jnp.maximum(1.0, jnp.linalg.norm(state)) / denominator
    return jnp.where(jnp.isfinite(step), step, base)


def build_jacobian_action(
    residual_fn: Callable[[jax.Array], jax.Array],
    state: jax.Array,
    options: JVPOptions,
) -> Callable[[jax.Array], jax.Array]:
    """Build ``direction -> J(state) direction`` for one Newton iterate."""

    if options.mode == "ad":
        def action(direction):
            return jax.jvp(residual_fn, (state,), (direction,))[1]
    else:
        residual_at_state = residual_fn(state)

        if options.fd_scheme == "forward":
            def action(direction):
                step = _fd_step(state, direction, options.fd_relative_step)
                return (residual_fn(state + step * direction) - residual_at_state) / step
        else:
            def action(direction):
                step = _fd_step(state, direction, options.fd_relative_step)
                return (
                    residual_fn(state + step * direction)
                    - residual_fn(state - step * direction)
                ) / (2.0 * step)

    return jax.jit(action) if options.jit else action
