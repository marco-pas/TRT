"""Physics-based block preconditioners."""

from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from trt_jfnk.discretization.boundary import BoundaryCondition2D
from trt_jfnk.discretization.grid import Grid2D
from trt_jfnk.models.gray_trt import GrayTRTModel
from trt_jfnk.models.su_olson_linear import SuOlsonLinearModel

from .scaling import ScaleContext


PreconditionerFactory = Callable[[jax.Array, ScaleContext], Callable[[jax.Array], jax.Array]]


@jax.jit
def _apply_local_inverse(vector, a_hat, b_hat, c_hat, d_hat, determinant):
    block_size = a_hat.size
    first = vector[:block_size]
    second = vector[block_size:]
    answer_first = (d_hat * first - b_hat * second) / determinant
    answer_second = (-c_hat * first + a_hat * second) / determinant
    return jnp.concatenate((answer_first, answer_second))


def identity_preconditioner(_state, _context):
    return lambda vector: vector


def gray_local_block_factory(
    model: GrayTRTModel,
    dt: float,
    grid: Grid2D,
    bc: BoundaryCondition2D,
    determinant_floor: float = 1.0e-14,
) -> PreconditionerFactory:
    r"""Build an inverse for the scaled local exchange/material block.

    If the physical approximation is ``M=[[a,b],[c,d]]``, then the solver
    sees ``R^{-1} M S``.  This routine inverts that transformed 2-by-2 block,
    so it is correct for arbitrary state and residual scales.
    """

    def factory(state, context: ScaleContext):
        a, b, c, d = model.local_jacobian_blocks(state, dt, grid, bc)
        state_e, state_t = context.state_block_scales
        residual_e, residual_t = context.residual_block_scales
        a_hat = a.reshape(-1) * state_e / residual_e
        b_hat = b.reshape(-1) * state_t / residual_e
        c_hat = c.reshape(-1) * state_e / residual_t
        d_hat = d.reshape(-1) * state_t / residual_t
        determinant = a_hat * d_hat - b_hat * c_hat
        signed_floor = jnp.where(determinant >= 0.0, determinant_floor, -determinant_floor)
        determinant = jnp.where(jnp.abs(determinant) < determinant_floor, signed_floor, determinant)

        return partial(
            _apply_local_inverse,
            a_hat=a_hat,
            b_hat=b_hat,
            c_hat=c_hat,
            d_hat=d_hat,
            determinant=determinant,
        )

    return factory


def su_olson_local_block_factory(
    model: SuOlsonLinearModel,
    dt: float,
    grid: Grid2D,
    bc: BoundaryCondition2D,
    determinant_floor: float = 1.0e-14,
) -> PreconditionerFactory:
    r"""Build an inverse for the scaled local Su--Olson exchange block."""

    def factory(state, context: ScaleContext):
        a, b, c, d = model.local_jacobian_blocks(state, dt, grid, bc)
        state_u, state_v = context.state_block_scales
        residual_u, residual_v = context.residual_block_scales
        a_hat = a.reshape(-1) * state_u / residual_u
        b_hat = b.reshape(-1) * state_v / residual_u
        c_hat = c.reshape(-1) * state_u / residual_v
        d_hat = d.reshape(-1) * state_v / residual_v
        determinant = a_hat * d_hat - b_hat * c_hat
        signed_floor = jnp.where(determinant >= 0.0, determinant_floor, -determinant_floor)
        determinant = jnp.where(jnp.abs(determinant) < determinant_floor, signed_floor, determinant)

        return partial(
            _apply_local_inverse,
            a_hat=a_hat,
            b_hat=b_hat,
            c_hat=c_hat,
            d_hat=d_hat,
            determinant=determinant,
        )

    return factory

