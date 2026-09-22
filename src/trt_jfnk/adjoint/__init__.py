"""Implicit-step discrete-adjoint utilities."""

from .discrete_adjoint import one_step_adjoint
from .implicit_step import transpose_jacobian_action

__all__ = ["one_step_adjoint", "transpose_jacobian_action"]
