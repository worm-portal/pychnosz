"""
Python interface to the H2O92 Fortran subroutine.

This module provides a ctypes-based interface to the original SUPCRT92 
Fortran code for high-precision water property calculations.

The interface matches exactly what the R CHNOSZ package uses.
"""

import os
import sys
import ctypes
from ctypes import c_int, c_double, c_bool, POINTER
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Tuple
import warnings


class H2O92Interface:
    """
    Python interface to the H2O92 Fortran subroutine.
    
    This provides the exact same computational kernel as used by
    the R CHNOSZ package, ensuring identical results.
    """
    
    def __init__(self):
        """Initialize the Fortran library interface."""
        self._lib = None
        self._load_fortran_library()
        self._setup_function_signatures()

        # Property names (matches H2O92 Fortran order)
        self.property_names = [
            'A', 'G', 'S', 'U', 'H', 'Cv', 'Cp', 'Speed', 'alpha', 'beta',
            'epsilon', 'visc', 'tcond', 'surten', 'tdiff', 'Prndtl', 'visck',
            'albe', 'ZBorn', 'YBorn', 'QBorn', 'daldT', 'XBorn'
        ]

        # Cache for water property calculations (significant speedup for repeated T,P)
        self._cache = {}
        self._cache_enabled = True
    
    def _load_fortran_library(self):
        """Load the compiled Fortran shared library."""
        lib_dir = Path(__file__).parent
        
        # Determine library extension
        if sys.platform == "win32":
            lib_name = "h2o92.dll"
        elif sys.platform == "darwin":
            lib_name = "h2o92.dylib"
        else:
            lib_name = "h2o92.so"
        
        lib_path = lib_dir / lib_name
        
        if not lib_path.exists():
            raise FileNotFoundError(
                f"Fortran library not found: {lib_path}\n"
                f"Please run compile_fortran.py to build the library first."
            )
        
        # On Windows, add MinGW bin to PATH to find runtime dependencies
        original_path = None
        if sys.platform == "win32":
            mingw_bin = r'C:\msys64\mingw64\bin'
            if os.path.exists(mingw_bin):
                original_path = os.environ.get('PATH', '')
                os.environ['PATH'] = mingw_bin + os.pathsep + original_path
        
        try:
            self._lib = ctypes.CDLL(str(lib_path))
        except OSError as e:
            # Try loading with WinDLL on Windows
            if sys.platform == "win32":
                try:
                    self._lib = ctypes.WinDLL(str(lib_path))
                except OSError:
                    raise RuntimeError(
                        f"Failed to load Fortran library {lib_path}: {e}\n"
                        f"This may be due to missing MinGW runtime dependencies.\n"
                        f"Try: \n"
                        f"1. Ensure MinGW-w64 is properly installed\n"
                        f"2. Add C:\\msys64\\mingw64\\bin to your system PATH\n"
                        f"3. Install required MinGW runtime libraries"
                    )
            else:
                raise RuntimeError(f"Failed to load Fortran library {lib_path}: {e}")
        finally:
            # Restore original PATH
            if original_path is not None:
                os.environ['PATH'] = original_path
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures for Fortran subroutines."""
        
        # H2O92 subroutine signature:
        # SUBROUTINE H2O92(specs, states, props, error)
        # INTEGER specs(10)
        # DOUBLE PRECISION states(4), props(46)  
        # LOGICAL error
        
        self._h2o92 = self._lib.h2o92_
        self._h2o92.argtypes = [
            POINTER(c_int * 10),      # specs(10)
            POINTER(c_double * 4),    # states(4)
            POINTER(c_double * 46),   # props(46)
            POINTER(c_bool)           # error
        ]
        self._h2o92.restype = None
        
        # Try to find IDEAL2 subroutine as well
        if hasattr(self._lib, 'ideal2_'):
            self._ideal2 = self._lib.ideal2_
            self._ideal2.argtypes = [
                POINTER(c_double),    # T
                POINTER(c_double),    # dummy args (8 total)
                POINTER(c_double),
                POINTER(c_double),
                POINTER(c_double),
                POINTER(c_double),
                POINTER(c_double),
                POINTER(c_double)
            ]
            self._ideal2.restype = None
    
    def calculate_properties_batch(self, T: np.ndarray, P: Union[np.ndarray, str],
                                   properties: List[str] = None) -> Dict[str, np.ndarray]:
        """
        Calculate water properties for multiple T,P points (vectorized).

        This is much faster than calling calculate_properties() in a loop
        because it reduces ctypes overhead.

        Parameters
        ----------
        T : array
            Temperatures in Kelvin
        P : array or "Psat"
            Pressures in bar, or "Psat" for saturation pressure
        properties : list of str, optional
            Properties to calculate. If None, calculates all available.

        Returns
        -------
        dict
            Dictionary with calculated property arrays
        """
        T = np.atleast_1d(T)
        n = len(T)

        # Handle P input
        if isinstance(P, str) and P == "Psat":
            P_is_psat = True
            P_vals = np.full(n, np.nan)  # Placeholder
        else:
            P_is_psat = False
            P_vals = np.atleast_1d(P)
            if len(P_vals) < n:
                P_vals = np.resize(P_vals, n)

        # Initialize output arrays for all properties
        all_props = self.property_names + ['V', 'rho', 'Psat', 'E', 'kT', 'A_DH', 'B_DH']
        results = {prop: np.full(n, np.nan) for prop in all_props}

        # Optimized calculation loop - minimize Python overhead
        mw_h2o = 18.0152
        cal_to_j = 4.184

        # Pre-create specs arrays (reuse if possible)
        if P_is_psat:
            specs = (c_int * 10)(2, 2, 2, 5, 1, 1, 1, 1, 4, 0)
        else:
            specs = (c_int * 10)(2, 2, 2, 5, 1, 0, 2, 1, 4, 0)

        # Reusable arrays
        states = (c_double * 4)()
        props = (c_double * 46)()
        error = c_bool(False)

        # Property name to index mapping (for fast lookup)
        # R CHNOSZ water.R:159 includes 'tcond' in energy conversion list
        energy_props_idx = {self.property_names.index(p): True for p in ['A', 'G', 'H', 'U', 'S', 'Cv', 'Cp', 'tcond'] if p in self.property_names}

        for i in range(n):
            if np.isnan(T[i]) or (not P_is_psat and np.isnan(P_vals[i])):
                continue

            # Check cache first (round to avoid floating point precision issues)
            if self._cache_enabled:
                if P_is_psat:
                    cache_key = (round(T[i], 6), 'Psat')
                else:
                    cache_key = (round(T[i], 6), round(P_vals[i], 6))

                if cache_key in self._cache:
                    # Use cached values
                    cached_props = self._cache[cache_key]
                    for prop_name in all_props:
                        if prop_name in cached_props:
                            results[prop_name][i] = cached_props[prop_name]
                    continue

            # Setup states array
            states[0] = T[i] - 273.15  # K to C
            if P_is_psat:
                states[1] = 0.0
                states[2] = 1.0
            else:
                states[1] = P_vals[i]
                states[2] = 1.0
            states[3] = 0.0

            # Reset error flag
            error.value = False

            # Call Fortran
            try:
                self._h2o92(ctypes.byref(specs), ctypes.byref(states),
                           ctypes.byref(props), ctypes.byref(error))
            except:
                continue

            if error.value:
                continue

            # Extract results - optimized
            rho = states[2]
            rho2 = states[3]

            if P_is_psat:
                inc = 1 if rho2 > rho else 0
                rho_liquid = rho2 if inc == 1 else rho
                results['Psat'][i] = states[1]
            else:
                rho_liquid = rho
                inc = 0

            # Store for caching
            if self._cache_enabled:
                cached_result = {}

            # Extract 23 properties - optimized loop
            for j in range(len(self.property_names)):
                prop_index = 2 * j + inc
                if prop_index < 46:
                    val = props[prop_index]
                    # Apply unit conversions only to energy properties
                    if j in energy_props_idx:
                        val *= cal_to_j
                    results[self.property_names[j]][i] = val
                    if self._cache_enabled:
                        cached_result[self.property_names[j]] = val

            # Derived properties
            if rho_liquid > 0:
                V_i = mw_h2o / rho_liquid
                results['V'][i] = V_i
                results['rho'][i] = rho_liquid * 1000

                alpha_i = results['alpha'][i]
                if not np.isnan(alpha_i):
                    results['E'][i] = V_i * alpha_i

                beta_i = results['beta'][i]
                if not np.isnan(beta_i):
                    results['kT'][i] = V_i * beta_i

                eps = results['epsilon'][i]
                if eps > 0:
                    sqrt_rho = rho_liquid**0.5
                    eps_T = eps * T[i]
                    results['A_DH'][i] = 1.8246e6 * sqrt_rho / (eps_T**1.5)
                    results['B_DH'][i] = 50.29e8 * sqrt_rho / (eps_T**0.5)

                # Cache derived properties too
                if self._cache_enabled:
                    cached_result['V'] = V_i
                    cached_result['rho'] = rho_liquid * 1000
                    if not np.isnan(alpha_i):
                        cached_result['E'] = results['E'][i]
                    if not np.isnan(beta_i):
                        cached_result['kT'] = results['kT'][i]
                    if eps > 0:
                        cached_result['A_DH'] = results['A_DH'][i]
                        cached_result['B_DH'] = results['B_DH'][i]
                    if P_is_psat:
                        cached_result['Psat'] = results['Psat'][i]

            # Store in cache
            if self._cache_enabled and cache_key:
                self._cache[cache_key] = cached_result

        # Filter requested properties
        if properties is not None:
            filtered_results = {}
            for prop in properties:
                if prop in results:
                    filtered_results[prop] = results[prop]
                else:
                    raise ValueError(f"Property '{prop}' not available")
            return filtered_results

        return results

    def calculate_properties(self, T: float, P: Union[float, str],
                           properties: List[str] = None) -> Dict[str, float]:
        """
        Calculate water properties using the H2O92 Fortran subroutine.

        Parameters
        ----------
        T : float
            Temperature in Kelvin
        P : float or "Psat"
            Pressure in bar, or "Psat" for saturation pressure
        properties : list of str, optional
            Properties to calculate. If None, calculates all available.

        Returns
        -------
        dict
            Dictionary with calculated properties

        Examples
        --------
        >>> h2o = H2O92Interface()
        >>> props = h2o.calculate_properties(298.15, 1.0, ['rho', 'epsilon'])
        >>> print(f"Density: {props['rho']:.3f} g/cm³")
        >>> print(f"Dielectric: {props['epsilon']:.1f}")
        """
        
        # Setup specs array (H2O92 parameters) - matches R exactly
        # it, id, ip, ih, itripl, isat, iopt, useLVS, epseqn, icrit
        # From R: specs <- c(2, 2, 2, 5, 1, isat, iopt, 1, 4, 0)
        if isinstance(P, str) and P == "Psat":
            isat = 1
            iopt = 1  # T,D input for saturation 
        else:
            isat = 0
            iopt = 2  # T,P input for single phase
        
        specs = (c_int * 10)(2, 2, 2, 5, 1, isat, iopt, 1, 4, 0)
        
        # Setup states array
        states = (c_double * 4)()
        # Temperature must be in Celsius (like R does: Tc <- convert(T, "C"))
        states[0] = T - 273.15  # Convert K to C
        if isinstance(P, str) and P == "Psat":
            states[1] = 0.0  # Pressure not used for saturation
            states[2] = 1.0  # Initial density guess (g/cm³)
        else:
            states[1] = P  # Pressure in bar
            states[2] = 1.0  # Initial density guess (g/cm³)
        states[3] = 0.0  # Second density for two-phase
        
        # Setup output arrays
        props = (c_double * 46)()  # 46 properties (23 vapor + 23 liquid)
        error = c_bool(False)
        
        # Call Fortran subroutine
        try:
            self._h2o92(ctypes.byref(specs), ctypes.byref(states),
                       ctypes.byref(props), ctypes.byref(error))
        except Exception as e:
            raise RuntimeError(f"Fortran subroutine call failed: {e}")
        
        # Check for errors
        if error.value:
            warnings.warn(f"H2O92 calculation error at T={T:.1f}K, P={P}")
            return {prop: np.nan for prop in self.property_names}
        
        # Extract results following R's approach exactly
        results = {}
        
        # Determine which phase to use (liquid vs vapor) - from R water.R
        # R code: rho <- H2O[[2]][3]; rho2 <- H2O[[2]][4]
        rho = states[2]   # First phase density  
        rho2 = states[3]  # Second phase density
        
        if isinstance(P, str) and P == "Psat":
            # For saturation: use liquid phase (denser)
            if rho2 > rho:
                rho_liquid = rho2
                inc = 1  # Second state is liquid (R: inc <- 1)
            else:
                rho_liquid = rho  
                inc = 0  # First state is liquid (R: inc <- 0)
            results['Psat'] = states[1]  # Saturation pressure
        else:
            # Single phase calculation
            rho_liquid = states[2]
            inc = 0  # Use first state
        
        # Extract properties following R's method exactly:
        # R: w <- t(H2O[[3]][seq(1, 45, length.out = 23)+inc])
        # seq(1, 45, length.out = 23) gives: 1, 3, 5, 7, ..., 45 (in R 1-based indexing)
        # In Python 0-based: 0, 2, 4, 6, ..., 44, then add inc
        for i, prop_name in enumerate(self.property_names):
            prop_index = 2 * i + inc  # Every other element, offset by inc
            if prop_index < 46:
                results[prop_name] = props[prop_index]
        
        # Apply R CHNOSZ-compatible unit conversions and derived property calculations
        mw_h2o = 18.0152  # g/mol (matches R SUP92.f)

        # Energy unit conversion: Following R CHNOSZ exactly
        # R gets values from FORTRAN (in cal/mol with ih=5), then converts TO Joules
        # Line 159-160 in R water.R:
        # isenergy <- names(w.out) %in% c("A", "G", "S", "U", "H", "Cv", "Cp", "tcond")
        # if(any(isenergy)) w.out[, isenergy] <- convert(w.out[, isenergy], "J")
        cal_to_j = 4.184  # Conversion factor from cal to J (R uses this)

        # Convert thermodynamic properties from cal/mol to J/mol (like R does)
        energy_props = ['A', 'G', 'H', 'U']  # Extensive thermodynamic properties
        for prop in energy_props:
            if prop in results:
                results[prop] = results[prop] * cal_to_j

        # Convert heat capacities and entropy from cal/mol/K to J/mol/K (like R does)
        entropy_props = ['S', 'Cv', 'Cp']
        for prop in entropy_props:
            if prop in results:
                results[prop] = results[prop] * cal_to_j

        # Convert thermal conductivity from cal/(s·cm·K) to W/(m·K) (like R does)
        # R includes 'tcond' in the energy conversion list (water.R:159)
        if 'tcond' in results:
            results['tcond'] = results['tcond'] * cal_to_j
        
        # Molar volume: cm³/mol (matches R calculation)
        if rho_liquid > 0:
            results['V'] = mw_h2o / rho_liquid  # cm³/mol
        else:
            results['V'] = np.nan
        
        # Density conversion: g/cm³ → kg/m³ (matches R line 131: rho <- rho.out*1000)
        results['rho'] = rho_liquid * 1000  # Convert g/cm³ to kg/m³ like R
        
        # Derived properties (matches R lines 135-136)
        if 'V' in results and not np.isnan(results['V']):
            # E = V * alpha (thermal expansivity)
            results['E'] = results['V'] * results['alpha'] if 'alpha' in results else np.nan
            # kT = V * beta (isothermal compressibility)  
            results['kT'] = results['V'] * results['beta'] if 'beta' in results else np.nan
        
        # Debye-Hückel parameters (matches R lines 140-141)
        # A_DH <- 1.8246e6 * rho.out^0.5 / (epsilon * T)^1.5
        # B_DH <- 50.29e8 * rho.out^0.5 / (epsilon * T)^0.5
        # Note: R actually does use 50.29e8 - must match R exactly
        if rho_liquid > 0 and 'epsilon' in results and results['epsilon'] > 0:
            results['A_DH'] = 1.8246e6 * (rho_liquid**0.5) / ((results['epsilon'] * T)**1.5)
            results['B_DH'] = 50.29e8 * (rho_liquid**0.5) / ((results['epsilon'] * T)**0.5)  # Match R: 50.29e8
        else:
            results['A_DH'] = np.nan
            results['B_DH'] = np.nan
        
        # Filter requested properties
        if properties is not None:
            filtered_results = {}
            for prop in properties:
                if prop in results:
                    filtered_results[prop] = results[prop]
                else:
                    raise ValueError(f"Property '{prop}' not available")
            return filtered_results
        
        return results
    
    def calculate_ideal_gas(self, T: float, property: str) -> float:
        """
        Calculate ideal gas properties using IDEAL2 subroutine.
        
        Parameters
        ----------
        T : float
            Temperature in Kelvin
        property : str
            'S' for entropy or 'Cp' for heat capacity
            
        Returns
        -------
        float
            Property value
        """
        if not hasattr(self, '_ideal2'):
            raise NotImplementedError("IDEAL2 subroutine not available in library")
        
        # Setup parameters (8 dummy args)
        args = [c_double(T)] + [c_double(0.0) for _ in range(7)]
        
        # Call Fortran subroutine
        self._ideal2(*[ctypes.byref(arg) for arg in args])
        
        if property == 'S':
            return args[3].value  # 4th output
        elif property == 'Cp':  
            return args[7].value  # 8th output
        else:
            raise ValueError(f"Property '{property}' not supported by IDEAL2")
    
    def available_properties(self) -> List[str]:
        """Get list of available water properties."""
        return self.property_names + ['V', 'rho', 'Psat', 'E', 'kT', 'A_DH', 'B_DH']

    def clear_cache(self):
        """Clear the water properties cache."""
        self._cache.clear()

    def enable_cache(self, enabled: bool = True):
        """Enable or disable caching of water properties."""
        self._cache_enabled = enabled
        if not enabled:
            self.clear_cache()


# Global instance for easy access
_h2o92_interface = None

def get_h2o92_interface() -> H2O92Interface:
    """Get global H2O92Interface instance (singleton pattern)."""
    global _h2o92_interface
    if _h2o92_interface is None:
        _h2o92_interface = H2O92Interface()
    return _h2o92_interface