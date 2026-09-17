"""Nonlinear material thermodynamics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolynomialMaterial:
    r"""Heat capacity and its consistent primitive.

    .. math::
       c_v(T)=c_{v0}+c_{v1}T^m,\qquad
       e(T)=c_{v0}T+\frac{c_{v1}}{m+1}T^{m+1}.
    """

    cv0: float = 0.1
    cv1: float = 1.0
    exponent: int = 3

    def __post_init__(self) -> None:
        if self.cv0 <= 0.0 or self.cv1 < 0.0:
            raise ValueError("cv0 must be positive and cv1 nonnegative")
        if self.exponent < 0:
            raise ValueError("material exponent must be nonnegative")

    def heat_capacity(self, temperature):
        return self.cv0 + self.cv1 * temperature**self.exponent

    def internal_energy(self, temperature):
        power = self.exponent + 1
        return self.cv0 * temperature + self.cv1 * temperature**power / power
