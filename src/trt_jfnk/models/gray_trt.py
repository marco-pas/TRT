"""Nonlinear gray thermal-radiation diffusion model."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp

from trt_jfnk.discretization.boundary import (
    BoundaryCondition2D,
    boundary_mask,
    enforce_radiation_residual,
)
from trt_jfnk.discretization.diffusion import variable_diffusion
from trt_jfnk.discretization.grid import Grid2D
from trt_jfnk.discretization.time_integrators import ThetaMethod

from .material import PolynomialMaterial
from .opacity import PowerLawOpacity


@dataclass(frozen=True)
class GrayTRTModel:
    r"""Two-temperature gray diffusion with nonlinear material coupling.

    The dimensionless equations are

    .. math::
       \partial_t E &= \nabla\!\cdot(D(T)\nabla E)
          -\sigma_a(T)(E-aT^4)+Q,\\
       \partial_t e(T) &= \sigma_a(T)(E-aT^4).

    A theta method is used in time.  Backward Euler is the default because it
    is the cleanest first nonlinear-robustness experiment.
    """

    opacity: PowerLawOpacity = field(default_factory=PowerLawOpacity)
    material: PolynomialMaterial = field(default_factory=PolynomialMaterial)
    integrator: ThetaMethod = field(default_factory=ThetaMethod)
    radiation_constant: float = 1.0
    light_speed: float = 1.0

    def pack(self, radiation, temperature):
        if radiation.shape != temperature.shape:
            raise ValueError("radiation and temperature shapes must match")
        return jnp.concatenate((radiation.reshape(-1), temperature.reshape(-1)))

    def unpack(self, state, grid: Grid2D):
        if state.size != 2 * grid.size:
            raise ValueError("state must contain two grid-sized blocks")
        return state[: grid.size].reshape(grid.shape), state[grid.size :].reshape(grid.shape)

    def emission(self, temperature):
        return self.radiation_constant * temperature**4

    def exchange(self, radiation, temperature):
        return self.opacity.absorption(temperature) * (radiation - self.emission(temperature))

    def rhs(self, radiation, temperature, source, grid: Grid2D, bc: BoundaryCondition2D):
        diffusion = self.opacity.diffusion_coefficient(temperature, self.light_speed)
        coupling = self.exchange(radiation, temperature)
        radiation_rhs = variable_diffusion(radiation, diffusion, grid, bc) - coupling + source
        material_energy_rhs = coupling
        return radiation_rhs, material_energy_rhs

    def residual(
        self,
        state,
        old_state,
        dt: float,
        source_new,
        source_old,
        grid: Grid2D,
        bc: BoundaryCondition2D,
    ):
        radiation, temperature = self.unpack(state, grid)
        radiation_old, temperature_old = self.unpack(old_state, grid)
        rhs_radiation, rhs_energy = self.rhs(radiation, temperature, source_new, grid, bc)
        rhs_radiation_old, rhs_energy_old = self.rhs(
            radiation_old, temperature_old, source_old, grid, bc
        )

        residual_radiation = radiation - radiation_old - dt * self.integrator.blend(
            rhs_radiation, rhs_radiation_old
        )
        residual_energy = (
            self.material.internal_energy(temperature)
            - self.material.internal_energy(temperature_old)
            - dt * self.integrator.blend(rhs_energy, rhs_energy_old)
        )
        residual_radiation = enforce_radiation_residual(
            residual_radiation, radiation, grid, bc
        )
        return self.pack(residual_radiation, residual_energy)

    def is_admissible(
        self,
        state,
        grid: Grid2D,
        radiation_floor: float = 0.0,
        temperature_floor: float = 1.0e-8,
    ):
        radiation, temperature = self.unpack(state, grid)
        return jnp.logical_and(
            jnp.all(radiation >= radiation_floor),
            jnp.all(temperature >= temperature_floor),
        )

    def local_jacobian_blocks(self, state, dt: float, grid: Grid2D, bc: BoundaryCondition2D):
        r"""Point-local Jacobian approximation used as a block preconditioner.

        Derivatives of the diffusion term are deliberately omitted.  The
        retained exchange and material derivatives are the stiff nonlinear
        part and give an inexpensive independent 2-by-2 solve in every cell.
        """

        radiation, temperature = self.unpack(state, grid)
        sigma = self.opacity.absorption(temperature)
        sigma_prime = self.opacity.absorption_derivative(temperature)
        mismatch = radiation - self.emission(temperature)
        exchange_temperature = (
            sigma_prime * mismatch
            - 4.0 * self.radiation_constant * sigma * temperature**3
        )
        weight = dt * self.integrator.theta

        a = 1.0 + weight * sigma
        b = weight * exchange_temperature
        c = -weight * sigma
        d = self.material.heat_capacity(temperature) - weight * exchange_temperature

        mask = boundary_mask(grid, bc)
        a = jnp.where(mask, 1.0, a)
        b = jnp.where(mask, 0.0, b)
        return a, b, c, d
