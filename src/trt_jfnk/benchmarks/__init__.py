"""Reproducible benchmark configurations and metrics."""

from .ghosh_slab_analytic import (
    GhoshSlabAnalyticBenchmark,
    compute_transcendental_roots,
    ghosh_slab_steady_state,
)
from .ghosh_slab_fv import GhoshSlabFVProblem
from .mms import MMSConfig, MMSProblem1D, SymbolicTRTMMS
from .mms_stiff import StiffMMSConfig, StiffMMSProblem1D, SymbolicStiffTRTMMS
from .perturbed_su_olson import PerturbedSuOlsonConfig, PerturbedSuOlsonProblem1D
from .nonequilibrium_marshak import (
    NonequilibriumMarshakConfig,
    NonequilibriumMarshakProblem1D,
)

__all__ = [
    "GhoshSlabAnalyticBenchmark",
    "compute_transcendental_roots",
    "ghosh_slab_steady_state",
    "GhoshSlabFVProblem",
    "MMSConfig",
    "MMSProblem1D",
    "SymbolicTRTMMS",
    "StiffMMSConfig",
    "StiffMMSProblem1D",
    "SymbolicStiffTRTMMS",
    "PerturbedSuOlsonConfig",
    "PerturbedSuOlsonProblem1D",
    "NonequilibriumMarshakConfig",
    "NonequilibriumMarshakProblem1D",
]

