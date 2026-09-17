"""Boundary-condition definitions and residual-row replacement."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .grid import Grid2D

DIRICHLET = "dirichlet"
PERIODIC = "periodic"


@dataclass(frozen=True)
class BoundaryCondition2D:
    x_kind: str = DIRICHLET
    y_kind: str = PERIODIC
    radiation_value: float = 0.0

    def __post_init__(self) -> None:
        valid = {DIRICHLET, PERIODIC}
        if self.x_kind not in valid or self.y_kind not in valid:
            raise ValueError("boundary kinds must be 'dirichlet' or 'periodic'")


def boundary_mask(grid: Grid2D, bc: BoundaryCondition2D):
    """Boolean mask for radiation Dirichlet degrees of freedom."""

    i = jnp.arange(grid.nx)[:, None]
    j = jnp.arange(grid.ny)[None, :]
    mask = jnp.zeros(grid.shape, dtype=bool)
    if bc.x_kind == DIRICHLET:
        mask = jnp.logical_or(mask, jnp.logical_or(i == 0, i == grid.nx - 1))
    if bc.y_kind == DIRICHLET:
        mask = jnp.logical_or(mask, jnp.logical_or(j == 0, j == grid.ny - 1))
    return mask


def enforce_radiation_state(field, grid: Grid2D, bc: BoundaryCondition2D):
    mask = boundary_mask(grid, bc)
    return jnp.where(mask, jnp.asarray(bc.radiation_value, dtype=field.dtype), field)


def enforce_radiation_residual(residual, field, grid: Grid2D, bc: BoundaryCondition2D):
    """Replace Dirichlet PDE rows by ``field - boundary_value``."""

    mask = boundary_mask(grid, bc)
    boundary_equation = field - jnp.asarray(bc.radiation_value, dtype=field.dtype)
    return jnp.where(mask, boundary_equation, residual)
