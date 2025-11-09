"""Core thermodynamic calculation functions for CHNOSZ."""

from .thermo import ThermoSystem, thermo
from .info import info, find_species, get_species_data, list_species
from .basis import basis, get_basis, is_basis_defined, preset_basis, BasisError
from .species import species, get_species, is_species_defined, n_species, SpeciesError
from .retrieve import retrieve

# Optional imports for modules that may not exist yet
try:
    from .subcrt import subcrt
except ImportError:
    subcrt = None

try:
    from .affinity import affinity
except ImportError:
    affinity = None

try:
    from .equilibrate import equilibrate
except ImportError:
    equilibrate = None

try:
    from .diagram import diagram
except ImportError:
    diagram = None

__all__ = [
    'ThermoSystem', 'thermo',
    'info', 'find_species', 'get_species_data', 'list_species',
    'basis', 'get_basis', 'is_basis_defined', 'preset_basis', 'BasisError',
    'species', 'get_species', 'is_species_defined', 'n_species', 'SpeciesError',
    'retrieve'
]

# Add optional functions if they exist
if subcrt is not None:
    __all__.append('subcrt')
if affinity is not None:
    __all__.append('affinity')
if equilibrate is not None:
    __all__.append('equilibrate')
if diagram is not None:
    __all__.append('diagram')