"""
CHNOSZ Fortran interface package.

This package provides Python interfaces to the original CHNOSZ Fortran
subroutines for high-performance thermodynamic calculations.
"""

from .h2o92_interface import H2O92Interface, get_h2o92_interface

__all__ = ['H2O92Interface', 'get_h2o92_interface']

# Tell pdoc to skip the compiled Fortran library files
# These are .so/.dll files that are loaded via ctypes, not Python imports
__pdoc__ = {
    'h2o92': False,
}
