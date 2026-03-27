from .blanket_base import OpenMCBlanketSimulation
from dataclasses import dataclass

from jax_sbgeom.flux_surfaces import ParametrisedSurface

@dataclass
class Simulation2DOptimization(OpenMCBlanketSimulation):
    fsd              : ParametrisedSurface
    discrete_blanket : LayeredDiscreteBlanket
    source_kwargs    : dict