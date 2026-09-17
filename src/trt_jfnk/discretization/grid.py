"""Uniform two-dimensional cell/node grid used by the TRT examples."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Grid2D:
    """Uniform tensor-product grid.

    Arrays use shape ``(nx, ny)`` and C-order flattening.  Endpoints are
    included, which is natural for the Dirichlet benchmark.  A periodic grid
    can set ``periodic_x``/``periodic_y`` so that the duplicate endpoint is
    omitted in that direction.
    """

    nx: int
    ny: int
    x_min: float = -5.0
    x_max: float = 5.0
    y_min: float = -0.5
    y_max: float = 0.5
    periodic_x: bool = False
    periodic_y: bool = True

    def __post_init__(self) -> None:
        if self.nx < 3 or self.ny < 3:
            raise ValueError("nx and ny must both be at least three")
        if not self.x_max > self.x_min or not self.y_max > self.y_min:
            raise ValueError("grid upper bounds must exceed lower bounds")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)

    @property
    def size(self) -> int:
        return self.nx * self.ny

    @property
    def dx(self) -> float:
        divisor = self.nx if self.periodic_x else self.nx - 1
        return (self.x_max - self.x_min) / divisor

    @property
    def dy(self) -> float:
        divisor = self.ny if self.periodic_y else self.ny - 1
        return (self.y_max - self.y_min) / divisor

    @property
    def x(self):
        return jnp.linspace(self.x_min, self.x_max, self.nx, endpoint=not self.periodic_x)

    @property
    def y(self):
        return jnp.linspace(self.y_min, self.y_max, self.ny, endpoint=not self.periodic_y)

    @property
    def mesh(self):
        return jnp.meshgrid(self.x, self.y, indexing="ij")

    def zeros(self, dtype=None):
        return jnp.zeros(self.shape, dtype=dtype)
