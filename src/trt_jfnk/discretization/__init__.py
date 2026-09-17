"""Structured-grid spatial and temporal discretizations."""

from .boundary import BoundaryCondition2D, DIRICHLET, PERIODIC
from .grid import Grid2D
from .time_integrators import ThetaMethod

__all__ = ["BoundaryCondition2D", "DIRICHLET", "Grid2D", "PERIODIC", "ThetaMethod"]
