"""Temperature-dependent gray opacity laws."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class PowerLawOpacity:
    r"""Regularized power-law opacity.

    .. math::
       \sigma_a(T)=\sigma_{a0}\left(T_{\rm ref}/T_{\rm eff}\right)^p,
       \qquad T_{\rm eff}=\sqrt{T^2+T_{\rm floor}^2}.

    ``scattering_ratio`` adds a transport-only contribution
    ``sigma_s = scattering_ratio * sigma_a``.
    """

    sigma_a0: float = 10.0
    exponent: float = 3.0
    reference_temperature: float = 1.0
    temperature_floor: float = 5.0e-2
    scattering_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_a0 <= 0.0 or self.reference_temperature <= 0.0:
            raise ValueError("opacity and reference temperature must be positive")
        if self.temperature_floor <= 0.0 or self.scattering_ratio < 0.0:
            raise ValueError("temperature floor must be positive and scattering nonnegative")

    def effective_temperature(self, temperature):
        floor = jnp.asarray(self.temperature_floor, dtype=temperature.dtype)
        return jnp.sqrt(temperature * temperature + floor * floor)

    def absorption(self, temperature):
        effective = self.effective_temperature(temperature)
        return self.sigma_a0 * (self.reference_temperature / effective) ** self.exponent

    def absorption_derivative(self, temperature):
        sigma = self.absorption(temperature)
        denominator = temperature * temperature + self.temperature_floor**2
        return -self.exponent * sigma * temperature / denominator

    def transport(self, temperature):
        return (1.0 + self.scattering_ratio) * self.absorption(temperature)

    def diffusion_coefficient(self, temperature, light_speed: float = 1.0):
        return light_speed / (3.0 * self.transport(temperature))
