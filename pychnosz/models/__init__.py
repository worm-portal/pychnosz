"""Equation of state models and water property models for CHNOSZ."""

from .water import water, available_properties, get_water_models, compare_models, WaterModelError

# Import Fortran-backed SUPCRT92 (falls back to Python if Fortran unavailable)
from .supcrt92_fortran import water_SUPCRT92, SUPCRT92Water

# Import HKF equation of state functions
from .hkf import hkf, gfun, convert_cm3bar

# Import CGL equation of state functions
from .cgl import cgl, quartz_coesite

# Import HKF helper functions
from .hkf_helpers import calc_logK, calc_G_TP, G2logK, dissrxn2logK, OBIGT2eos

# Optional imports for modules that may not exist yet
try:
    from .hkf import HKF
except ImportError:
    HKF = None

try:
    from .cgl import CGL
except ImportError:
    CGL = None

try:
    from .berman import Berman
except ImportError:
    Berman = None

__all__ = [
    'water', 'available_properties', 'get_water_models', 'compare_models', 'WaterModelError',
    'water_SUPCRT92', 'SUPCRT92Water',
    'hkf', 'gfun', 'convert_cm3bar',
    'cgl', 'quartz_coesite',
    'calc_logK', 'calc_G_TP', 'G2logK', 'dissrxn2logK', 'OBIGT2eos'
]

# Add optional functions if they exist
if HKF is not None:
    __all__.append('HKF')
if CGL is not None:
    __all__.append('CGL')
if Berman is not None:
    __all__.append('Berman')