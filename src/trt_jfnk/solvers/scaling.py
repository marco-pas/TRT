"""Frozen block scaling for nonlinear systems."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

ScaleMode = Literal["none", "fixed", "state"]


@dataclass(frozen=True)
class ScalePolicy:
    """Policy for constructing two-block state and residual scales.

    ``state`` selects ``max(reference, infinity_norm(block), floor)``.
    Residual references default to the state references, i.e. ``R=S``.
    The resulting context is frozen for an entire Newton solve.
    """

    mode: ScaleMode = "none"
    state_references: tuple[float, float] = (1.0, 1.0)
    residual_references: tuple[float, float] | None = None
    floor: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.mode not in {"none", "fixed", "state"}:
            raise ValueError("scale mode must be none, fixed, or state")
        if not math.isfinite(self.floor) or self.floor <= 0.0:
            raise ValueError("scale floor must be finite and positive")
        for value in self.state_references:
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("state references must be finite and positive")
        if self.residual_references is not None:
            for value in self.residual_references:
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError("residual references must be finite and positive")


@dataclass(frozen=True)
class ScaleContext:
    state_scale: jax.Array
    residual_scale: jax.Array
    state_block_scales: tuple[float, float]
    residual_block_scales: tuple[float, float]
    block_size: int

    def to_scaled_state(self, state):
        return state / self.state_scale

    def to_physical_state(self, scaled_state):
        return self.state_scale * scaled_state

    def to_scaled_residual(self, residual):
        return residual / self.residual_scale

    def to_physical_correction(self, scaled_correction):
        return self.state_scale * scaled_correction


def build_scale_context(state, block_size: int, policy: ScalePolicy) -> ScaleContext:
    if state.ndim != 1 or state.size != 2 * block_size:
        raise ValueError("state must be a flat vector containing two equal blocks")

    if policy.mode == "none":
        state_scales = (1.0, 1.0)
        residual_scales = (1.0, 1.0)
    else:
        state_scales_list: list[float] = []
        for block, reference in enumerate(policy.state_references):
            if policy.mode == "fixed":
                value = reference
            else:
                values = state[block * block_size : (block + 1) * block_size]
                norm = float(np.max(np.abs(np.asarray(jax.device_get(values)))))
                value = max(reference, norm)
            state_scales_list.append(max(value, policy.floor))
        state_scales = (state_scales_list[0], state_scales_list[1])
        residual_scales = (
            state_scales if policy.residual_references is None else policy.residual_references
        )
        residual_scales = (
            max(residual_scales[0], policy.floor),
            max(residual_scales[1], policy.floor),
        )

    dtype = state.dtype
    state_scale = jnp.concatenate(
        (
            jnp.full((block_size,), state_scales[0], dtype=dtype),
            jnp.full((block_size,), state_scales[1], dtype=dtype),
        )
    )
    residual_scale = jnp.concatenate(
        (
            jnp.full((block_size,), residual_scales[0], dtype=dtype),
            jnp.full((block_size,), residual_scales[1], dtype=dtype),
        )
    )
    return ScaleContext(
        state_scale=state_scale,
        residual_scale=residual_scale,
        state_block_scales=state_scales,
        residual_block_scales=residual_scales,
        block_size=block_size,
    )
