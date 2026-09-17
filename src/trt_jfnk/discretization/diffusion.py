"""Conservative diffusion operators."""

from __future__ import annotations

import jax.numpy as jnp

from .boundary import BoundaryCondition2D, DIRICHLET, PERIODIC, boundary_mask
from .grid import Grid2D


def harmonic_mean(left, right, floor: float = 1.0e-30):
    denominator = left + right
    denominator = jnp.where(jnp.abs(denominator) > floor, denominator, floor)
    return 2.0 * left * right / denominator


def variable_diffusion(field, coefficient, grid: Grid2D, bc: BoundaryCondition2D):
    r"""Return the finite-volume approximation to ``div(D grad(field))``.

    Harmonic face averaging is used for discontinuous/temperature-dependent
    diffusion coefficients.  Dirichlet diffusion rows are zeroed because the
    complete model later replaces those rows with the boundary equation.
    """

    if field.shape != grid.shape or coefficient.shape != grid.shape:
        raise ValueError("field and coefficient must have grid.shape")

    xp = jnp.roll(field, -1, axis=0)
    xm = jnp.roll(field, 1, axis=0)
    yp = jnp.roll(field, -1, axis=1)
    ym = jnp.roll(field, 1, axis=1)
    dp = harmonic_mean(coefficient, jnp.roll(coefficient, -1, axis=0))
    dm = harmonic_mean(coefficient, jnp.roll(coefficient, 1, axis=0))
    ep = harmonic_mean(coefficient, jnp.roll(coefficient, -1, axis=1))
    em = harmonic_mean(coefficient, jnp.roll(coefficient, 1, axis=1))

    div_x = (dp * (xp - field) - dm * (field - xm)) / grid.dx**2
    div_y = (ep * (yp - field) - em * (field - ym)) / grid.dy**2
    result = div_x + div_y

    # Rolled neighbor values are invalid only on rows later replaced by a
    # Dirichlet equation.  Interior rows correctly see the boundary values.
    if bc.x_kind == DIRICHLET or bc.y_kind == DIRICHLET:
        result = jnp.where(boundary_mask(grid, bc), 0.0, result)
    return result


def laplacian(field, grid: Grid2D, bc: BoundaryCondition2D):
    return variable_diffusion(field, jnp.ones_like(field), grid, bc)
