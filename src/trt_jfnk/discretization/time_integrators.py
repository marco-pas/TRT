"""One-step theta time integration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThetaMethod:
    theta: float = 1.0

    def __post_init__(self) -> None:
        if not 0.5 <= self.theta <= 1.0:
            raise ValueError("theta must lie in [0.5, 1]")

    def blend(self, new, old):
        return self.theta * new + (1.0 - self.theta) * old


BACKWARD_EULER = ThetaMethod(1.0)
CRANK_NICOLSON = ThetaMethod(0.5)
