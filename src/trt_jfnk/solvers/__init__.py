"""Scaled Newton--Krylov solvers and preconditioners."""

from .jvp import JVPOptions
from .krylov import KrylovOptions
from .newton import NewtonOptions, NewtonResult, newton_krylov
from .scaling import ScaleContext, ScalePolicy

__all__ = [
    "JVPOptions",
    "KrylovOptions",
    "NewtonOptions",
    "NewtonResult",
    "ScaleContext",
    "ScalePolicy",
    "newton_krylov",
]
