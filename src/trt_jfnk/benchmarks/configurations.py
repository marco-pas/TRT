"""Named, serializable configurations for linear and nonlinear experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import jax.numpy as jnp

from trt_jfnk.discretization.boundary import BoundaryCondition2D, DIRICHLET, PERIODIC
from trt_jfnk.discretization.grid import Grid2D
from trt_jfnk.discretization.time_integrators import ThetaMethod
from trt_jfnk.models.gray_trt import GrayTRTModel
from trt_jfnk.models.material import PolynomialMaterial
from trt_jfnk.models.opacity import PowerLawOpacity
from trt_jfnk.models.su_olson_linear import SuOlsonLinearModel


@dataclass(frozen=True)
class GrayBenchmarkConfig:
    """Moderately stiff nonlinear gray-TRT pulse benchmark."""

    nx: int = 65
    ny: int = 8
    x_min: float = -5.0
    x_max: float = 5.0
    y_min: float = -0.5
    y_max: float = 0.5
    dt: float = 5.0e-3
    steps: int = 40
    theta: float = 1.0
    initial_temperature: float = 0.2
    source_amplitude: float = 20.0
    source_width: float = 0.30
    source_end_time: float = 0.10
    sigma_a0: float = 1.0
    opacity_exponent: float = 3.0
    opacity_temperature_floor: float = 0.05
    scattering_ratio: float = 0.0
    cv0: float = 0.1
    cv1: float = 1.0
    material_exponent: int = 3
    radiation_constant: float = 1.0
    light_speed: float = 1.0

    def __post_init__(self) -> None:
        if self.dt <= 0.0 or self.steps < 1 or self.source_width <= 0.0:
            raise ValueError("dt, steps, and source width must be positive")
        if self.initial_temperature <= 0.0:
            raise ValueError("initial temperature must be positive")

    def grid(self) -> Grid2D:
        return Grid2D(
            self.nx,
            self.ny,
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            periodic_x=False,
            periodic_y=True,
        )

    def boundary(self) -> BoundaryCondition2D:
        equilibrium = self.radiation_constant * self.initial_temperature**4
        return BoundaryCondition2D(DIRICHLET, PERIODIC, equilibrium)

    def model(self) -> GrayTRTModel:
        return GrayTRTModel(
            opacity=PowerLawOpacity(
                sigma_a0=self.sigma_a0,
                exponent=self.opacity_exponent,
                reference_temperature=1.0,
                temperature_floor=self.opacity_temperature_floor,
                scattering_ratio=self.scattering_ratio,
            ),
            material=PolynomialMaterial(self.cv0, self.cv1, self.material_exponent),
            integrator=ThetaMethod(self.theta),
            radiation_constant=self.radiation_constant,
            light_speed=self.light_speed,
        )

    def initial_state(self, dtype=None):
        grid = self.grid()
        temperature = jnp.full(grid.shape, self.initial_temperature, dtype=dtype)
        radiation = self.radiation_constant * temperature**4
        return self.model().pack(radiation, temperature)

    def source(self, time: float, dtype=None):
        grid = self.grid()
        x, _y = grid.mesh
        active = jnp.asarray(time <= self.source_end_time, dtype=dtype)
        gaussian = jnp.exp(-0.5 * (x / self.source_width) ** 2)
        return active * self.source_amplitude * gaussian

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SuOlsonBenchmarkConfig:
    nx: int = 65
    ny: int = 8
    dt: float = 2.0e-2
    steps: int = 50
    epsilon: float = 1.0e-2
    source_amplitude: float = 1.0
    source_width: float = 0.5
    source_end_time: float = 0.5

    def grid(self) -> Grid2D:
        return Grid2D(self.nx, self.ny, periodic_x=False, periodic_y=True)

    def boundary(self) -> BoundaryCondition2D:
        return BoundaryCondition2D(DIRICHLET, PERIODIC, 0.0)

    def model(self) -> SuOlsonLinearModel:
        return SuOlsonLinearModel(epsilon=self.epsilon)

    def initial_state(self, dtype=None):
        grid = self.grid()
        zeros = jnp.zeros(grid.shape, dtype=dtype)
        return self.model().pack(zeros, zeros)

    def source(self, time: float, dtype=None):
        grid = self.grid()
        x, _y = grid.mesh
        active = jnp.asarray(time <= self.source_end_time, dtype=dtype)
        return active * self.source_amplitude * jnp.exp(-0.5 * (x / self.source_width) ** 2)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)
