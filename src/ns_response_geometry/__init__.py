"""Neutron-star EOS response geometry."""

from .eos import IncompressibleEOS, RelativisticPolytrope
from .geometry import induced_response, metric_projector, transverse_response
from .observables import StellarObservables, solve_observables
from .response import (
    local_response_geometry,
    log_observable_vector,
    response_jacobian,
    sequence_tangent,
)
from .response_eos import (
    TabulatedSoundSpeedEOS,
    build_sound_speed_eos,
    gaussian_basis,
    squared_exponential_covariance,
)
from .tov import Star, solve_star

__all__ = [
    "IncompressibleEOS",
    "RelativisticPolytrope",
    "TabulatedSoundSpeedEOS",
    "Star",
    "StellarObservables",
    "solve_star",
    "solve_observables",
    "build_sound_speed_eos",
    "gaussian_basis",
    "squared_exponential_covariance",
    "log_observable_vector",
    "response_jacobian",
    "sequence_tangent",
    "local_response_geometry",
    "induced_response",
    "metric_projector",
    "transverse_response",
]
