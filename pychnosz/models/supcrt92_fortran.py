"""
SUPCRT92 water model with Fortran backend.

This implementation uses the original H2O92 Fortran subroutine for exact
compatibility with R CHNOSZ. Falls back to Python approximations if
the Fortran library is not available.
"""

import numpy as np
import warnings
from typing import Union, List, Optional, Dict, Any

# Try to import Fortran interface
try:
    from ..fortran import get_h2o92_interface
    HAS_FORTRAN = True
except (ImportError, FileNotFoundError, RuntimeError) as e:
    HAS_FORTRAN = False
    _fortran_error = str(e)


class SUPCRT92Water:
    """
    SUPCRT92 water model with Fortran backend.
    
    This class provides an interface to the original SUPCRT92 Fortran
    subroutines for calculating water properties. If the Fortran library
    is not available, it falls back to Python approximations.
    
    The Fortran implementation gives exact compatibility with R CHNOSZ
    and includes all 23+ thermodynamic properties.
    """
    
    def __init__(self):
        """
        Initialize SUPCRT92 water model.
        
        SUPCRT92 requires the compiled FORTRAN interface for accuracy and 
        compatibility with R CHNOSZ. No pure Python fallback is provided.
        """
        if not HAS_FORTRAN:
            raise ImportError(
                f"SUPCRT92 water model requires compiled FORTRAN interface. "
                f"Error: {_fortran_error}. "
                f"Please compile the FORTRAN subroutines using: "
                f"python compile_fortran.py. "
                f"See setup_fortran_instructions.md for details."
            )
        
        # Initialize Fortran interface (required)
        try:
            self._fortran_interface = get_h2o92_interface()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FORTRAN interface: {e}") from e
    
    def calculate(self, 
                  properties: Union[str, List[str]], 
                  T: Union[float, np.ndarray] = 298.15,
                  P: Union[float, np.ndarray, str] = 1.0,
                  **kwargs) -> Union[float, np.ndarray, Dict[str, Any]]:
        """
        Calculate water properties using SUPCRT92 model.
        
        Parameters
        ----------
        properties : str or list of str
            Property or list of properties to calculate
        T : float or array
            Temperature in Kelvin
        P : float, array, or 'Psat'
            Pressure in bar, or 'Psat' for saturation pressure
        **kwargs
            Additional options (e.g., Psat_floor)
        
        Returns
        -------
        float, array, or dict
            Calculated properties
        """
        
        # Handle input types
        if isinstance(properties, str):
            properties = [properties]
            single_prop = True
        else:
            single_prop = False
            
        # Convert inputs to arrays
        T = np.atleast_1d(np.asarray(T, dtype=float))
        
        # Always use FORTRAN backend (no fallback)
        return self._calculate_fortran(properties, T, P, single_prop, **kwargs)
    
    def _calculate_fortran(self, properties: List[str], T: np.ndarray,
                          P: Union[np.ndarray, str], single_prop: bool,
                          **kwargs) -> Union[float, np.ndarray, Dict[str, Any]]:
        """Calculate properties using Fortran backend."""

        # Handle pressure input
        if isinstance(P, str) and P == 'Psat':
            P_vals = 'Psat'
        else:
            P = np.atleast_1d(np.asarray(P, dtype=float))
            if len(P) < len(T):
                P = np.resize(P, len(T))
            elif len(T) < len(P):
                T = np.resize(T, len(P))
            P_vals = P

        # Use batched calculation for better performance
        try:
            results = self._fortran_interface.calculate_properties_batch(
                T, P_vals, properties
            )
        except Exception as e:
            warnings.warn(f"Batch Fortran calculation failed: {e}")
            # Fallback to individual calculations if batch fails
            results = {}
            for prop in properties:
                results[prop] = np.full_like(T, np.nan)

            for i in range(len(T)):
                T_i = T[i]

                # Skip invalid points
                if np.isnan(T_i):
                    continue

                if isinstance(P_vals, str):
                    P_i = P_vals
                else:
                    P_i = P_vals[i]
                    if np.isnan(P_i):
                        continue

                try:
                    # Call Fortran interface
                    props_i = self._fortran_interface.calculate_properties(
                        T_i, P_i, properties
                    )

                    # Store results
                    for prop in properties:
                        if prop in props_i:
                            results[prop][i] = props_i[prop]

                except Exception as e:
                    warnings.warn(f"Fortran calculation failed at T={T_i:.1f}K, P={P_i}: {e}")
                    continue

        # Handle Psat_floor for saturation pressure
        if 'Psat' in results and 'Psat_floor' in kwargs:
            Psat_floor = kwargs['Psat_floor']
            if Psat_floor is not None:
                results['Psat'] = np.maximum(results['Psat'], Psat_floor)
        
        # Return results
        if single_prop:
            result = results[properties[0]]
            return result[0] if len(result) == 1 else result
        else:
            # Convert single-element arrays to scalars if appropriate
            if len(T) == 1:
                for key in results:
                    if len(results[key]) == 1:
                        results[key] = results[key][0]
            return results
    
    def available_properties(self) -> List[str]:
        """
        Get list of available properties.
        
        Returns
        -------
        List[str]
            List of available property names
        """
        if self._use_fortran:
            return self._fortran_interface.available_properties()
        else:
            return self._python_backend.available_properties()
    
    @property
    def backend(self) -> str:
        """Get the active backend ('fortran' or 'python')."""
        return 'fortran' if self._use_fortran else 'python'
    
    @property
    def has_fortran(self) -> bool:
        """Check if Fortran backend is available."""
        return HAS_FORTRAN


# Global instance - FORTRAN interface required
supcrt92_water = SUPCRT92Water()


def water_SUPCRT92(property: Union[str, List[str]], 
                   T: Union[float, np.ndarray] = 298.15,
                   P: Union[float, np.ndarray, str] = 1.0,
                   **kwargs) -> Union[float, np.ndarray, Dict[str, Any]]:
    """
    Calculate water properties using SUPCRT92 model.
    
    This function provides the same interface as the original Python
    implementation but with Fortran backend support for high accuracy.
    
    Parameters
    ----------
    property : str or list of str
        Property or list of properties to calculate
    T : float or array
        Temperature in Kelvin
    P : float, array, or 'Psat'
        Pressure in bar, or 'Psat' for saturation pressure
    **kwargs
        Additional options (e.g., Psat_floor)
    
    Returns
    -------
    float, array, or dict
        Calculated properties
        
    Examples
    --------
    >>> # Basic usage
    >>> rho = water_SUPCRT92('rho', 298.15, 1.0)
    >>> print(f"Density: {rho:.3f} g/cm³")
    
    >>> # Multiple properties  
    >>> props = water_SUPCRT92(['rho', 'epsilon'], 298.15, 1.0)
    >>> print(f"ρ = {props['rho']:.3f}, ε = {props['epsilon']:.1f}")
    
    >>> # Saturation pressure
    >>> Psat = water_SUPCRT92('Psat', 373.15, 'Psat')
    >>> print(f"Psat at 100°C: {Psat:.2f} bar")
    """
    return supcrt92_water.calculate(property, T, P, **kwargs)