"""Example scalar objectives for verification and inverse problems."""

from __future__ import annotations

import jax.numpy as jnp


def total_radiation(state, model, grid):
    radiation, _second = model.unpack(state, grid)
    return jnp.sum(radiation) * grid.dx * grid.dy


def total_material_energy(state, model, grid):
    _radiation, temperature = model.unpack(state, grid)
    if not hasattr(model, "material"):
        raise TypeError("model must expose nonlinear material thermodynamics")
    return jnp.sum(model.material.internal_energy(temperature)) * grid.dx * grid.dy


def detector_average(state, model, grid, center: float, width: float):
    radiation, _second = model.unpack(state, grid)
    weights = jnp.exp(-0.5 * ((grid.x - center) / width) ** 2)[:, None]
    return jnp.sum(weights * radiation) / (jnp.sum(weights) * grid.ny)
