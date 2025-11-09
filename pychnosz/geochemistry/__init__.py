"""
Geochemistry package for CHNOSZ.

This package provides specialized geochemical calculations including
mineral equilibria, redox reactions, and environmental applications.
"""

from .minerals import (
    mineral_solubility, stability_field, phase_boundary,
    MineralEquilibria
)
from .redox import (
    eh_ph, pe, eh, logfO2,
    RedoxCalculator
)

__all__ = [
    'mineral_solubility', 'stability_field', 'phase_boundary',
    'MineralEquilibria', 'eh_ph', 'pe', 'eh', 'logfO2',
    'RedoxCalculator'
]