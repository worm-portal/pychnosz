"""Utility functions for CHNOSZ calculations."""

from .reset import reset
from .formula import makeup, get_formula, as_chemical_formula, mass, entropy, species_basis, calculate_ghs, ZC, i2A, FormulaError

# Optional imports for modules that may not exist yet
try:
    from .units import convert_units
except ImportError:
    convert_units = None

try:
    from .data import load_data
except ImportError:
    load_data = None

__all__ = [
    'reset',
    'makeup', 'get_formula', 'as_chemical_formula', 'mass', 'entropy',
    'species_basis', 'calculate_ghs', 'ZC', 'i2A', 'FormulaError'
]

# Add optional functions if they exist
if convert_units is not None:
    __all__.append('convert_units')
if load_data is not None:
    __all__.append('load_data')