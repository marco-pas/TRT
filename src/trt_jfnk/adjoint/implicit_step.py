"""Jacobian-transpose actions for an implicit residual."""

from __future__ import annotations

from typing import Callable

import jax


def transpose_jacobian_action(
    residual_fn: Callable[[jax.Array], jax.Array],
    state: jax.Array,
    cotangent: jax.Array,
):
    """Return ``(d residual / d state)^T cotangent`` using reverse AD."""

    _value, pullback = jax.vjp(residual_fn, state)
    return pullback(cotangent)[0]


def build_transpose_action(residual_fn, state, jit: bool = True):
    _value, pullback = jax.vjp(residual_fn, state)

    def action(cotangent):
        return pullback(cotangent)[0]

    return jax.jit(action) if jit else action
