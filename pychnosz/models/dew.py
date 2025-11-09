"""
DEW (Deep Earth Water) model implementation.

This module implements the Deep Earth Water model for calculating thermodynamic
and electrostatic properties of H2O at high pressures and temperatures relevant
to deep crustal and mantle conditions.

References:
- Sverjensky, D. A., Harrison, B., & Azzolini, D. (2014). Water in the deep Earth: 
  The dielectric constant and the solubilities of quartz and corundum to 60 kb 
  and 1200°C. Geochimica et Cosmochimica Acta, 129, 125-145.
- Pan, D., Spanu, L., Harrison, B., Sverjensky, D. A., & Car, R. (2013). 
  Dielectric properties of water under extreme conditions and validation of the 
  corresponding computational approaches. PNAS, 110(17), 6646-6650.
"""

import numpy as np
from typing import Union, List, Optional, Dict, Any
import warnings


class DEWWater:
    """
    Deep Earth Water (DEW) model implementation.
    
    This class provides thermodynamic and electrostatic properties of water
    at high pressures and temperatures using the DEW model correlations.
    """
    
    def __init__(self):
        """Initialize DEW water model."""
        # Physical constants  
        self.R = 8.314462618  # J/(mol·K) - Universal gas constant
        self.MW_H2O = 18.0152  # g/mol - Molecular weight of water
        
        # Critical constants for water
        self.Tc = 647.067  # K - Critical temperature
        self.Pc = 220.48   # bar - Critical pressure
        self.rhoc = 0.32174  # g/cm³ - Critical density
        
        # DEW model parameters
        self.a_epsilon = np.array([
            -1.57637700752506e3, -6.97284414953487e4, -3.14058873029023e6,
            1.11926957750896e7, 5.49375634503012e7, -1.33934314022535e8,
            -2.56395839070779e8, 3.73875501063673e8, 4.35976880906701e8,
            -2.11156427436252e8
        ])
        
        self.b_epsilon = np.array([
            5.07722476345932e-1, 1.48046755524790e1, 2.42452179259584e2,
            -1.73986255629880e3, -7.18635413197094e3, 1.21415969235037e4,
            1.92102380413670e4, -1.31967093058141e4, -1.35915853762697e4,
            3.17251296019127e3
        ])
        
        # Pressure and temperature limits for DEW model
        self.T_min = 273.15  # K
        self.T_max = 1473.15  # K (1200°C)
        self.P_min = 1.0      # bar
        self.P_max = 60000.0  # bar (60 kbar)
    
    def available_properties(self) -> List[str]:
        """
        Get list of available water properties.

        Note: DEW model only calculates a limited set of properties.
        This matches the R CHNOSZ implementation which only provides:
        G, epsilon, QBorn, V, rho, beta, A_DH, B_DH

        Other properties requested will return NaN.
        """
        return [
            # Properties actually calculated by DEW
            'G',        # Gibbs energy (J/mol)
            'epsilon',  # Dielectric constant (DEW specialty)
            'QBorn',    # Born Q function (1/bar)
            'V',        # Molar volume (cm³/mol)
            'rho',      # Density (kg/m³)
            'beta',     # Isothermal compressibility (1/bar)
            'A_DH',     # Debye-Hückel A parameter
            'B_DH',     # Debye-Hückel B parameter
        ]
    
    def calculate(self,
                  properties: Union[str, List[str]],
                  T: Union[float, np.ndarray] = 298.15,
                  P: Union[float, np.ndarray, str] = 1.0,
                  **kwargs) -> Union[float, np.ndarray, Dict[str, Any]]:
        """
        Calculate water properties using DEW model.

        Parameters
        ----------
        properties : str or list of str
            Property or properties to calculate
        T : float or array
            Temperature in Kelvin
        P : float, array, or 'Psat'
            Pressure in bar, or 'Psat' for saturation pressure
        **kwargs
            Additional options

        Returns
        -------
        float, array, or dict
            Calculated properties
        """
        # DEBUG
        debug_dew = False
        if debug_dew:
            print(f"\nDEBUG DEW.calculate() called:")
            print(f"  properties: {properties}")
            print(f"  T (input): {T}, type: {type(T)}")
            print(f"  P (input): {P}, type: {type(P)}")

        # Handle input types
        if isinstance(properties, str):
            properties = [properties]
            single_prop = True
        else:
            single_prop = False

        # Convert inputs to arrays
        T = np.atleast_1d(np.asarray(T, dtype=float))
        
        if isinstance(P, str) and P == 'Psat':
            P_is_Psat = True
            P_vals = self._calculate_Psat(T)
        else:
            P_is_Psat = False
            P = np.atleast_1d(np.asarray(P, dtype=float))
            if len(P) < len(T):
                P = np.resize(P, len(T))
            elif len(T) < len(P):
                T = np.resize(T, len(P))
            P_vals = P
        
        # Check validity of conditions
        valid = self._check_validity(T, P_vals)

        # Check for low T or low P conditions (T < 100°C or P < 1000 bar)
        # These should use SUPCRT92 instead of DEW (as in R CHNOSZ water.R line 381)
        ilow = (T < 373.15) | (P_vals < 1000)

        # Initialize results
        results = {}

        # Get list of properties that DEW actually calculates
        supported_props = self.available_properties()

        # For low T or low P conditions, use SUPCRT92 for ALL properties
        if np.any(ilow):
            from .supcrt92_fortran import water_SUPCRT92

            # Get SUPCRT92 results for low conditions
            T_low = T[ilow]
            P_low = P_vals[ilow]

            supcrt_results = water_SUPCRT92(properties, T_low, P_low)

            # Initialize all properties with appropriate array size
            for prop in properties:
                results[prop] = np.full_like(T, np.nan, dtype=float)

            # Fill in SUPCRT92 results for low conditions
            if isinstance(supcrt_results, dict):
                for prop in properties:
                    if prop in supcrt_results:
                        results[prop][ilow] = supcrt_results[prop]
            else:
                # Single property case
                results[properties[0]][ilow] = supcrt_results

            # Special case for Pr,Tr: epsilon should be 78.47 (DEW spreadsheet value)
            iPrTr = (np.abs(T - 298.15) < 0.1) & (np.abs(P_vals - 1.0) < 0.1)
            if 'epsilon' in properties and np.any(iPrTr):
                results['epsilon'][iPrTr] = 78.47

            # If all conditions are low, return SUPCRT results
            if np.all(ilow):
                if single_prop:
                    result = results[properties[0]]
                    return result[0] if len(result) == 1 else result
                else:
                    return results

        # Calculate density first (needed for many DEW properties)
        if any(prop in properties for prop in ['rho', 'V', 'epsilon', 'QBorn', 'beta', 'A_DH', 'B_DH']):
            rho_gcm3 = self._calculate_density(T, P_vals, valid)  # g/cm³
            rho = rho_gcm3 * 1000.0  # Convert to kg/m³ like SUPCRT
            V = self.MW_H2O / rho_gcm3  # cm³/mol (use g/cm³ for volume calculation)
        else:
            rho = None
            V = None

        # Calculate each requested property for high T and high P conditions
        for prop in properties:
            # If property is not supported by DEW, return NaN (like R CHNOSZ)
            if prop not in supported_props:
                if prop not in results:
                    results[prop] = np.full_like(T, np.nan, dtype=float)
            elif prop == 'rho':
                if prop not in results:
                    results[prop] = np.full_like(T, np.nan, dtype=float)
                results[prop][~ilow] = rho[~ilow]
            elif prop == 'V':
                if prop not in results:
                    results[prop] = np.full_like(T, np.nan, dtype=float)
                results[prop][~ilow] = V[~ilow]
            elif prop == 'epsilon':
                # Use g/cm³ density for dielectric constant calculation
                if prop not in results:
                    results[prop] = np.full_like(T, np.nan, dtype=float)
                if 'rho_gcm3' not in locals():
                    rho_gcm3 = self._calculate_density(T, P_vals, valid)
                epsilon_vals = self._calculate_dielectric_constant_with_density(T, P_vals, rho_gcm3, valid)
                results[prop][~ilow] = epsilon_vals[~ilow]
            elif prop == 'G':
                # Calculate Gibbs energy using the exact DEW method
                if prop not in results:
                    results[prop] = np.full_like(T, np.nan, dtype=float)
                for i in np.where(~ilow)[0]:
                    T_celsius = T[i] - 273.15  # Convert to Celsius
                    G_cal_per_mol = self._calculate_gibbs_of_water(P_vals[i], T_celsius)  # cal/mol
                    if not np.isnan(G_cal_per_mol):
                        results[prop][i] = G_cal_per_mol * 4.184  # Convert cal/mol to J/mol
            elif prop == 'QBorn':
                # Calculate QBorn using the exact DEW calculateQ function
                if prop not in results:
                    results[prop] = np.full_like(T, np.nan, dtype=float)
                if rho is None or V is None:
                    rho_gcm3 = self._calculate_density(T, P_vals, valid)
                else:
                    rho_gcm3 = rho / 1000.0  # Convert kg/m³ to g/cm³

                for i in np.where(~ilow)[0]:
                    if valid[i]:
                        T_celsius = T[i] - 273.15  # Convert to Celsius
                        results[prop][i] = self._calculate_Q(rho_gcm3[i], T_celsius)
            elif prop == 'beta':
                # Calculate beta (isothermal compressibility)
                if prop not in results:
                    results[prop] = np.full_like(T, np.nan, dtype=float)
                if rho is None or V is None:
                    rho_gcm3 = self._calculate_density(T, P_vals, valid)
                else:
                    rho_gcm3 = rho / 1000.0  # Convert kg/m³ to g/cm³

                for i in np.where(~ilow)[0]:
                    if valid[i]:
                        T_celsius = T[i] - 273.15  # Convert to Celsius
                        # Divide drhodP by rho to get units of bar^-1 (like R code)
                        drhodP = self._calculate_drhodP(rho_gcm3[i], T_celsius)
                        results[prop][i] = drhodP / rho_gcm3[i]
            elif prop in ['A_DH', 'B_DH']:
                # Calculate Debye-Hückel parameters
                if prop not in results:
                    results[prop] = np.full_like(T, np.nan, dtype=float)
                if 'rho_gcm3' not in locals():
                    rho_gcm3 = self._calculate_density(T, P_vals, valid)
                epsilon_vals = self._calculate_dielectric_constant_with_density(T, P_vals, rho_gcm3, valid)

                if prop == 'A_DH':
                    # A_DH = 1.8246e6 * rho^0.5 / (epsilon * T)^1.5
                    results[prop][~ilow] = 1.8246e6 * rho_gcm3[~ilow]**0.5 / (epsilon_vals[~ilow] * T[~ilow])**1.5
                else:  # B_DH
                    # B_DH = 50.29e8 * rho^0.5 / (epsilon * T)^0.5
                    results[prop][~ilow] = 50.29e8 * rho_gcm3[~ilow]**0.5 / (epsilon_vals[~ilow] * T[~ilow])**0.5
        
        # Return results
        if single_prop:
            result = results[properties[0]]
            return result[0] if len(result) == 1 else result
        else:
            # Convert to consistent array lengths
            for key in results:
                if np.isscalar(results[key]):
                    results[key] = np.full_like(T, results[key])
                elif len(results[key]) == 1 and len(T) > 1:
                    results[key] = np.full_like(T, results[key][0])
            return results
    
    def _check_validity(self, T: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Check validity of T-P conditions for DEW model."""
        valid = np.ones_like(T, dtype=bool)
        
        # Temperature limits
        valid &= (T >= self.T_min)
        valid &= (T <= self.T_max) 
        
        # Pressure limits
        valid &= (P >= self.P_min)
        valid &= (P <= self.P_max)
        
        # Avoid near-critical conditions where DEW may be less accurate
        valid &= ~((T > 0.95 * self.Tc) & (P < 2 * self.Pc))
        
        return valid
    
    def _calculate_Psat(self, T: np.ndarray) -> np.ndarray:
        """
        Calculate saturation pressure using Antoine equation.
        
        Valid up to critical point.
        """
        Psat = np.full_like(T, np.nan)
        valid = (T >= 273.16) & (T <= self.Tc)
        
        if np.any(valid):
            T_valid = T[valid]
            
            # Antoine equation coefficients for water (bar, K)
            A = 8.07131
            B = 1730.63
            C = -39.724
            
            # Antoine equation: log10(Psat) = A - B/(T + C)
            log10_Psat = A - B / (T_valid + C)
            Psat[valid] = 10**log10_Psat
        
        return Psat
    
    def _calculate_density(self, T: np.ndarray, P: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
        Calculate water density using DEW model correlations.
        
        This uses the exact bisection method from the R CHNOSZ DEW implementation
        to find the density that produces the target pressure.
        """
        rho = np.full_like(T, np.nan)
        
        if np.any(valid):
            T_valid = T[valid]
            P_valid = P[valid]
            
            # Use bisection method for each T, P pair (as in R DEW.R)
            rho_results = np.full(len(T_valid), np.nan)
            for i, (T_val, P_val) in enumerate(zip(T_valid, P_valid)):
                T_celsius = T_val - 273.15  # Convert to Celsius for DEW equations
                rho_results[i] = self._calculate_density_bisection(P_val, T_celsius)
            
            rho[valid] = rho_results
        
        return rho
    
    def _calculate_density_bisection(self, pressure: float, temperature_celsius: float, error: float = 0.01) -> float:
        """
        Calculate density using bisection method (exact R DEW.R implementation).
        
        Parameters
        ----------
        pressure : float
            Target pressure in bar
        temperature_celsius : float  
            Temperature in Celsius
        error : float
            Pressure error tolerance in bar (default 0.01 as in R code)
            
        Returns
        -------
        float
            Density in g/cm³
        """
        min_guess = 1e-5
        guess = 1e-5
        equation = 1  # The maxGuess is dependent on the value of "equation"
        max_guess = 7.5 * equation - 5.0  # Should be 2.5 for equation=1
        
        # Loop through and find the density (up to 50 iterations as in R)
        for i in range(50):
            # Calculate the pressure using the specified equation
            calc_p = self._calculate_pressure(guess, temperature_celsius)
            
            # If the calculated pressure is not equal to input pressure, 
            # determine a new guess based on bisection method
            if abs(calc_p - pressure) > error:
                if calc_p > pressure:
                    max_guess = guess
                    guess = (guess + min_guess) / 2.0
                elif calc_p < pressure:
                    min_guess = guess  
                    guess = (guess + max_guess) / 2.0
            else:
                return guess
                
        # If we didn't converge, return the last guess
        return guess
    
    def _calculate_pressure(self, density: float, temperature_celsius: float) -> float:
        """
        Calculate pressure from density and temperature using Zhang & Duan (2005) EOS.
        
        This is the exact implementation from R DEW.R calculatePressure function.
        
        Parameters
        ----------
        density : float
            Density in g/cm³
        temperature_celsius : float
            Temperature in Celsius
            
        Returns
        -------
        float
            Pressure in bar
        """
        # Constants from R DEW.R
        m = 18.01528         # Molar mass of water molecule in g/mol
        ZD05_R = 83.144      # Gas Constant in cm³ bar/mol/K
        ZD05_Vc = 55.9480373 # Critical volume in cm³/mol
        ZD05_Tc = 647.25     # Critical temperature in Kelvin
        
        TK = temperature_celsius + 273.15  # Temperature must be converted to Kelvin
        Vr = m / density / ZD05_Vc
        Tr = TK / ZD05_Tc
        
        B = 0.349824207 - 2.91046273 / (Tr * Tr) + 2.00914688 / (Tr * Tr * Tr)
        C = 0.112819964 + 0.748997714 / (Tr * Tr) - 0.87320704 / (Tr * Tr * Tr)
        D = 0.0170609505 - 0.0146355822 / (Tr * Tr) + 0.0579768283 / (Tr * Tr * Tr)
        E = -0.000841246372 + 0.00495186474 / (Tr * Tr) - 0.00916248538 / (Tr * Tr * Tr)
        f = -0.100358152 / Tr
        g = -0.00182674744 * Tr
        
        delta = (1 + B / Vr + C / (Vr * Vr) + D / (Vr**4) + E / (Vr**5) + 
                 (f / (Vr * Vr) + g / (Vr**4)) * np.exp(-0.0105999998 / (Vr * Vr)))
        
        return ZD05_R * TK * density * delta / m
    
    def _calculate_gibbs_of_water(self, pressure: float, temperature_celsius: float) -> float:
        """
        Calculate Gibbs Free Energy of water using exact R DEW.R implementation.
        
        This is the exact translation of the calculateGibbsOfWater function from R DEW.R
        
        Parameters
        ----------
        pressure : float
            Pressure in bar
        temperature_celsius : float
            Temperature in Celsius
            
        Returns
        -------
        float
            Gibbs Free Energy in cal/mol
        """
        # Gibbs Free Energy of water at 1 kb. This equation is a polynomial fit to data as a function of temperature.
        # It is valid in the range of 100 to 1000 C.
        GAtOneKb = (2.6880734E-09 * temperature_celsius**4 + 6.3163061E-07 * temperature_celsius**3 - 
                   0.019372355 * temperature_celsius**2 - 16.945093 * temperature_celsius - 55769.287)
        
        if pressure < 1000:  # Simply return zero, this method only works at P >= 1000 bars
            integral = np.nan
        elif pressure == 1000:  # Return the value calculated above from the polynomial fit
            integral = 0.0
        elif pressure > 1000:  # Integrate from 1 kb to P over the volume
            integral = 0.0
            # Integral is sum of rectangles with this width. This function in effect limits the spacing
            # to 20 bars so that very small pressures do not have unreasonably small widths. Otherwise the width
            # is chosen such that there are always 500 steps in the numerical integration. This ensures that for very
            # high pressures, there are not a huge number of steps calculated which is very computationally taxing.
            spacing = max(20.0, (pressure - 1000.0) / 500.0)
            
            # Use numpy arange to exactly match R's seq(1000, pressure, by = spacing) behavior
            # R's seq includes the endpoint, so we need to include pressure in our sequence
            P_values = np.arange(1000.0, pressure + spacing/2, spacing)  # +spacing/2 ensures we include endpoint
            
            for P_current in P_values:
                # This integral determines the density only down to an error of 100 bars
                # rather than the standard of 0.01. This is done to save computational
                # time. Tests indicate this reduces the computation by about a half while
                # introducing little error from the standard of 0.01.
                rho = self._calculate_density_bisection(P_current, temperature_celsius, error=100.0)
                integral += (18.01528 / rho / 41.84) * spacing
        
        return GAtOneKb + integral
    
    def _calculate_depsdrho(self, density: float, temperature_celsius: float) -> float:
        """
        Calculate partial derivative of dielectric constant with respect to density (dε/dρ).
        
        This is the exact implementation from R DEW.R calculate_depsdrho function.
        
        Parameters
        ----------
        density : float
            Density in g/cm³
        temperature_celsius : float
            Temperature in Celsius
            
        Returns
        -------
        float
            dε/dρ in cm³/g
        """
        # Power Function parameters (same as for epsilon calculation)
        a1 = -0.00157637700752506
        a2 = 0.0681028783422197
        a3 = 0.754875480393944
        b1 = -8.01665106535394E-05
        b2 = -0.0687161761831994
        b3 = 4.74797272182151
        
        A = a1 * temperature_celsius + a2 * np.sqrt(temperature_celsius) + a3
        B = b1 * temperature_celsius + b2 * np.sqrt(temperature_celsius) + b3
        
        # dε/dρ = A * exp(B) * density^(A-1)
        return A * np.exp(B) * (density ** (A - 1))
    
    def _calculate_drhodP(self, density: float, temperature_celsius: float) -> float:
        """
        Calculate partial derivative of density with respect to pressure (dρ/dP).
        
        This is the exact implementation from R DEW.R calculate_drhodP function.
        
        Parameters
        ----------
        density : float
            Density in g/cm³
        temperature_celsius : float
            Temperature in Celsius
            
        Returns
        -------
        float
            dρ/dP in g/cm³/bar
        """
        # Constants from R DEW.R
        m = 18.01528          # Molar mass of water molecule in g/mol
        ZD05_R = 83.144       # Gas Constant in cm³ bar/mol/K
        ZD05_Vc = 55.9480373  # Critical volume in cm³/mol
        ZD05_Tc = 647.25      # Critical temperature in Kelvin
        
        TK = temperature_celsius + 273.15       # temperature must be converted to Kelvin
        Tr = TK / ZD05_Tc
        cc = ZD05_Vc / m                # This term appears frequently in the equation
        Vr = m / (density * ZD05_Vc)
        
        B = 0.349824207 - 2.91046273 / (Tr * Tr) + 2.00914688 / (Tr * Tr * Tr)
        C = 0.112819964 + 0.748997714 / (Tr * Tr) - 0.87320704 / (Tr * Tr * Tr)
        D = 0.0170609505 - 0.0146355822 / (Tr * Tr) + 0.0579768283 / (Tr * Tr * Tr)
        E = -0.000841246372 + 0.00495186474 / (Tr * Tr) - 0.00916248538 / (Tr * Tr * Tr)
        f = -0.100358152 / Tr
        g = 0.0105999998 * Tr
        
        delta = (1 + B / Vr + C / (Vr**2) + D / (Vr**4) + E / (Vr**5) + 
                 (f / (Vr**2) + g / (Vr**4)) * np.exp(-0.0105999998 / (Vr**2)))
        
        kappa = (B * cc + 2 * C * (cc**2) * density + 4 * D * cc**4 * density**3 + 5 * E * cc**5 * density**4 +
                (2 * f * (cc**2) * density + 4 * g * cc**4 * density**3 - 
                 (f / (Vr**2) + g / (Vr**4)) * (2 * 0.0105999998 * (cc**2) * density)) * 
                np.exp(-0.0105999998 / (Vr**2)))
        
        return m / (ZD05_R * TK * (delta + density * kappa))
    
    def _calculate_Q(self, density: float, temperature_celsius: float) -> float:
        """
        Calculate Born Q function using exact R DEW.R implementation.
        
        This is the exact implementation from R DEW.R calculateQ function.
        
        Parameters
        ----------
        density : float
            Density in g/cm³
        temperature_celsius : float
            Temperature in Celsius
            
        Returns
        -------
        float
            Q in bar⁻¹
        """
        epsilon = self._calculate_epsilon_single(density, temperature_celsius)
        depsdrho = self._calculate_depsdrho(density, temperature_celsius)
        drhodP = self._calculate_drhodP(density, temperature_celsius)
        
        return depsdrho * drhodP / (epsilon**2)
    
    def _calculate_epsilon_single(self, density: float, temperature_celsius: float) -> float:
        """
        Calculate epsilon for single density and temperature values.
        
        Parameters
        ----------
        density : float
            Density in g/cm³
        temperature_celsius : float
            Temperature in Celsius
            
        Returns
        -------
        float
            Dielectric constant
        """
        # DEW power function parameters (same as in array version)
        a1 = -0.00157637700752506
        a2 = 0.0681028783422197
        a3 = 0.754875480393944
        b1 = -8.01665106535394E-05
        b2 = -0.0687161761831994
        b3 = 4.74797272182151
        
        A = a1 * temperature_celsius + a2 * np.sqrt(temperature_celsius) + a3
        B = b1 * temperature_celsius + b2 * np.sqrt(temperature_celsius) + b3
        
        return np.exp(B) * (density ** A)
    
    def _calculate_dielectric_constant_with_density(self, T: np.ndarray, P: np.ndarray, 
                                                   rho_gcm3: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """
        Calculate dielectric constant using pre-computed density in g/cm³.
        """
        epsilon = np.full_like(T, np.nan)
        
        if np.any(valid):
            T_valid = T[valid]
            P_valid = P[valid]
            rho_valid = rho_gcm3[valid]  # Already in g/cm³
            
            # Convert temperature to Celsius for DEW correlation
            T_celsius = T_valid - 273.15
            
            # DEW power function parameters (from R code)
            a1 = -0.00157637700752506
            a2 = 0.0681028783422197
            a3 = 0.754875480393944
            b1 = -8.01665106535394E-5
            b2 = -0.0687161761831994
            b3 = 4.74797272182151
            
            # Calculate A and B
            A = a1 * T_celsius + a2 * np.sqrt(T_celsius) + a3
            B = b1 * T_celsius + b2 * np.sqrt(T_celsius) + b3
            
            # DEW dielectric constant: epsilon = exp(B) * density^A
            epsilon_calc = np.exp(B) * (rho_valid ** A)
            
            # For low T or P conditions, use SUPCRT92 (AW90) as in R version
            low_condition = (T_celsius < 100.0) | (P_valid < 1000.0)
            
            if np.any(low_condition):
                # Use Archer & Wang for low conditions
                from .archer_wang import water_AW90
                
                # Convert density to kg/m³ and pressure to MPa
                rho_kg_m3 = rho_valid[low_condition] * 1000.0  # g/cm³ to kg/m³
                P_MPa = P_valid[low_condition] / 10.0  # bar to MPa
                T_low = T_valid[low_condition]
                
                epsilon_aw90 = water_AW90(T_low, rho_kg_m3, P_MPa)
                epsilon_calc[low_condition] = epsilon_aw90
            
            # Special case: at Pr,Tr use 78.47 as in R code
            prtr_condition = (np.abs(T_celsius - 25.0) < 0.1) & (np.abs(P_valid - 1.0) < 0.1)
            if np.any(prtr_condition):
                epsilon_calc[prtr_condition] = 78.47
            
            # Apply bounds to ensure physical values
            epsilon_calc = np.clip(epsilon_calc, 1.0, 200.0)
            
            epsilon[valid] = epsilon_calc
        
        return epsilon
    
    def _calculate_reference_density(self, T: np.ndarray) -> np.ndarray:
        """Calculate reference density at 1 bar pressure."""
        # Simplified fit to water density at 1 bar
        rho0 = 1.0 - 2.5e-4 * (T - 298.15) - 5e-7 * (T - 298.15)**2
        rho0 = np.maximum(rho0, 0.1)  # Ensure positive
        return rho0
    
    def _calculate_dielectric_constant(self, T: np.ndarray, P: np.ndarray, 
                                     valid: np.ndarray) -> np.ndarray:
        """
        Calculate dielectric constant using DEW model.
        
        This is the key feature of the DEW model - accurate dielectric constants
        at high P-T conditions based on molecular dynamics simulations.
        """
        epsilon = np.full_like(T, np.nan)
        
        if np.any(valid):
            T_valid = T[valid]
            P_valid = P[valid]
            
            # DEW dielectric constant correlation
            # Based on the R CHNOSZ implementation which uses the DEW power function
            # for high P-T conditions and falls back to SUPCRT92 (AW90) for low P-T.
            
            from .archer_wang import water_AW90
            
            # Calculate density first
            rho_full = self._calculate_density(T, P, valid)
            rho_valid = rho_full[valid]  # g/cm³
            
            # Convert temperature to Celsius for DEW correlation
            T_celsius = T_valid - 273.15
            
            # DEW power function parameters (from R code)
            a1 = -0.00157637700752506
            a2 = 0.0681028783422197
            a3 = 0.754875480393944
            b1 = -8.01665106535394E-5
            b2 = -0.0687161761831994
            b3 = 4.74797272182151
            
            # Calculate A and B
            A = a1 * T_celsius + a2 * np.sqrt(T_celsius) + a3
            B = b1 * T_celsius + b2 * np.sqrt(T_celsius) + b3
            
            # DEW dielectric constant: epsilon = exp(B) * density^A
            epsilon_calc = np.exp(B) * (rho_valid ** A)
            
            # For low T or P conditions, use SUPCRT92 (AW90) as in R version
            low_condition = (T_celsius < 100.0) | (P_valid < 1000.0)
            
            if np.any(low_condition):
                # Use Archer & Wang for low conditions
                # Convert density to kg/m³ and pressure to MPa
                rho_kg_m3 = rho_valid[low_condition] * 1000.0  # g/cm³ to kg/m³
                P_MPa = P_valid[low_condition] / 10.0  # bar to MPa
                T_low = T_valid[low_condition]
                
                epsilon_aw90 = water_AW90(T_low, rho_kg_m3, P_MPa)
                epsilon_calc[low_condition] = epsilon_aw90
            
            # Special case: at Pr,Tr use 78.47 as in R code
            prtr_condition = (np.abs(T_celsius - 25.0) < 0.1) & (np.abs(P_valid - 1.0) < 0.1)
            if np.any(prtr_condition):
                epsilon_calc[prtr_condition] = 78.47
            
            # Apply bounds to ensure physical values
            epsilon_calc = np.clip(epsilon_calc, 1.0, 200.0)
            
            epsilon[valid] = epsilon_calc
        
        return epsilon
    
    def _calculate_thermodynamic_properties(self, T: np.ndarray, P: np.ndarray,
                                          rho: np.ndarray, valid: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate thermodynamic properties using DEW model."""
        props = {}
        
        for prop in ['G', 'H', 'S', 'Cp', 'Cv', 'U', 'A']:
            props[prop] = np.full_like(T, np.nan)
        
        if np.any(valid):
            T_valid = T[valid]
            P_valid = P[valid]
            
            # Calculate Gibbs energy using the exact DEW method
            G_results = np.full(len(T_valid), np.nan)
            for i, (T_val, P_val) in enumerate(zip(T_valid, P_valid)):
                T_celsius = T_val - 273.15  # Convert to Celsius
                G_cal_per_mol = self._calculate_gibbs_of_water(P_val, T_celsius)  # cal/mol
                G_results[i] = G_cal_per_mol * 4.184  # Convert cal/mol to J/mol
            
            props['G'][valid] = G_results
            
            # For other properties, use simplified approximations (these are not as critical for DEW)
            if any(prop in ['H', 'S', 'Cp', 'Cv', 'U', 'A'] for prop in props.keys()):
                rho_valid = rho[valid] if rho is not None else self._calculate_density(T, P, valid)[valid]
                
                # Reference state properties (liquid water at 25°C, 1 bar)
                H_ref = -285830.0  # J/mol
                S_ref = 69.95      # J/(mol·K)
                Cp_ref = 75.31     # J/(mol·K)
                
                # Temperature effects
                dT = T_valid - 298.15
                
                # Heat capacity (empirical fit for high T-P)
                Cp = Cp_ref + 0.15 * dT - 2e-4 * dT**2 + 1e-7 * dT**3
                
                # Pressure effects on heat capacity
                Cp += 1e-5 * P_valid  # Small pressure dependence
                
                # Entropy (integrate Cp/T)
                S = S_ref + Cp_ref * np.log(T_valid / 298.15) + 0.15 * dT - 1e-4 * dT**2 + (1e-7/2) * dT**3
                
                # Enthalpy (integrate Cp)
                H = H_ref + Cp_ref * dT + 0.075 * dT**2 - (2e-4/3) * dT**3 + (1e-7/4) * dT**4
                
                # Pressure effects on enthalpy (∫V dP)
                V_molar = self.MW_H2O / (rho_valid / 1000.0)  # cm³/mol (convert kg/m³ to g/cm³)
                H += V_molar * (P_valid - 1.0) * 0.01  # Convert bar·cm³/mol to J/mol
                
                # Other properties
                Cv = Cp - self.R  # Simplified relation
                U = H - P_valid * V_molar * 0.01  # Internal energy
                A = U - T_valid * S  # Helmholtz energy
                
                # Store results
                props['H'][valid] = H
                props['S'][valid] = S
                props['Cp'][valid] = Cp
                props['Cv'][valid] = Cv
                props['U'][valid] = U
                props['A'][valid] = A
        
        return props
    
    def _calculate_mechanical_properties(self, T: np.ndarray, P: np.ndarray,
                                       rho: np.ndarray, valid: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate mechanical properties for high P-T conditions."""
        props = {}
        
        for prop in ['alpha', 'beta', 'kT', 'E']:
            props[prop] = np.full_like(T, np.nan)
        
        if np.any(valid):
            T_valid = T[valid]
            P_valid = P[valid]
            rho_valid = rho[valid] if rho is not None else self._calculate_density(T, P, valid)[valid]
            V_valid = self.MW_H2O / rho_valid  # cm³/mol
            
            # Thermal expansion coefficient (modified for high P-T)
            alpha = (2.14e-4 + 1e-6 * (T_valid - 298.15) - 2e-8 * P_valid)
            alpha = np.maximum(alpha, 1e-6)  # Ensure positive
            
            # Isothermal compressibility (decreases with pressure)
            beta = 4.5e-5 * np.exp(-P_valid / 10000.0) * (298.15 / T_valid)**0.5
            beta = np.maximum(beta, 1e-7)  # Ensure positive
            
            # Derived properties
            kT = V_valid * beta  # bar·cm³/mol
            E = V_valid * alpha  # cm³/(mol·K)
            
            props['alpha'][valid] = alpha
            props['beta'][valid] = beta
            props['kT'][valid] = kT
            props['E'][valid] = E
        
        return props
    
    def _calculate_born_functions(self, T: np.ndarray, P: np.ndarray, rho: np.ndarray,
                                valid: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate Born functions using exact DEW model."""
        props = {}
        
        for prop in ['QBorn', 'YBorn', 'XBorn', 'ZBorn']:
            props[prop] = np.full_like(T, np.nan)
        
        if np.any(valid):
            T_valid = T[valid]
            P_valid = P[valid]
            
            # Calculate QBorn using the exact DEW calculateQ function
            # This requires density in g/cm³, not kg/m³
            if rho is not None:
                rho_gcm3_valid = rho[valid] / 1000.0  # Convert kg/m³ to g/cm³
            else:
                rho_gcm3_full = self._calculate_density(T, P, valid)
                rho_gcm3_valid = rho_gcm3_full[valid]
            
            QBorn_results = np.full(len(T_valid), np.nan)
            epsilon_results = np.full(len(T_valid), np.nan)
            
            for i, (T_val, P_val, rho_val) in enumerate(zip(T_valid, P_valid, rho_gcm3_valid)):
                T_celsius = T_val - 273.15  # Convert to Celsius
                # Use exact DEW Q calculation
                QBorn_results[i] = self._calculate_Q(rho_val, T_celsius)
                epsilon_results[i] = self._calculate_epsilon_single(rho_val, T_celsius)
            
            # For other Born functions, use simplified relations (as in R water.R)
            # Get mechanical properties for thermal expansion
            mech_props = self._calculate_mechanical_properties(T, P, rho, valid)
            alpha_valid = mech_props['alpha'][valid]
            
            # Born functions
            YBorn = alpha_valid / epsilon_results  # 1/K
            XBorn = QBorn_results / epsilon_results  # 1/(bar·K) - note: this uses QBorn, not beta
            ZBorn = -1.0 / epsilon_results
            
            props['QBorn'][valid] = QBorn_results
            props['YBorn'][valid] = YBorn
            props['XBorn'][valid] = XBorn
            props['ZBorn'][valid] = ZBorn
        
        return props
    
    def _calculate_debye_huckel(self, T: np.ndarray, P: np.ndarray, rho: np.ndarray,
                              valid: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate Debye-Hückel parameters using DEW properties."""
        props = {}
        
        for prop in ['A_DH', 'B_DH']:
            props[prop] = np.full_like(T, np.nan)
        
        if np.any(valid):
            T_valid = T[valid]
            rho_valid = rho[valid] if rho is not None else self._calculate_density(T, P, valid)[valid]
            epsilon = self._calculate_dielectric_constant(T, P, valid)[valid]
            
            # Debye-Hückel parameters using DEW dielectric constants
            A_DH = 1.8246e6 * rho_valid**0.5 / (epsilon * T_valid)**1.5
            B_DH = 50.29e8 * rho_valid**0.5 / (epsilon * T_valid)**0.5
            
            props['A_DH'][valid] = A_DH
            props['B_DH'][valid] = B_DH
        
        return props
    
    def _calculate_transport_property(self, prop: str, T: np.ndarray, P: np.ndarray,
                                    rho: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Calculate transport properties (simplified for high P-T)."""
        result = np.full_like(T, np.nan)
        
        if np.any(valid):
            T_valid = T[valid]
            P_valid = P[valid]
            
            if prop == 'Speed':
                # Speed of sound (increases with pressure)
                result[valid] = (1402.7 + 5.0 * (T_valid - 298.15) + 
                               0.5 * np.sqrt(P_valid))
                
            elif prop == 'visc':
                # Viscosity (empirical fit for high P-T)
                result[valid] = (1e-3 * np.exp(-3.0 + 1000.0 / T_valid) * 
                               (1 + P_valid / 5000.0)**0.1)
                
            elif prop == 'tcond':
                # Thermal conductivity (increases with pressure and temperature)
                result[valid] = (0.6 + 0.002 * (T_valid - 298.15) + 
                               0.00005 * P_valid)
        
        return result


# Create global instance
dew_water = DEWWater()


def water_DEW(properties: Union[str, List[str]], 
              T: Union[float, np.ndarray] = 298.15,
              P: Union[float, np.ndarray, str] = 1.0,
              **kwargs) -> Union[float, np.ndarray, Dict[str, Any]]:
    """
    Calculate water properties using DEW model.
    
    Parameters
    ----------
    properties : str or list of str
        Property or properties to calculate
    T : float or array
        Temperature in Kelvin  
    P : float, array, or 'Psat'
        Pressure in bar, or 'Psat' for saturation pressure
    **kwargs
        Additional options
    
    Returns
    -------
    float, array, or dict
        Calculated water properties
        
    Examples
    --------
    >>> # High pressure conditions
    >>> epsilon = water_DEW('epsilon', T=873.15, P=10000)  # 600°C, 10 kbar
    >>> 
    >>> # Multiple properties at extreme conditions
    >>> props = water_DEW(['rho', 'epsilon'], T=1073.15, P=30000)  # 800°C, 30 kbar
    >>> 
    >>> # Born functions for electrolyte calculations
    >>> born = water_DEW(['QBorn', 'YBorn'], T=773.15, P=5000)
    """
    return dew_water.calculate(properties, T, P, **kwargs)


if __name__ == "__main__":
    # Quick test
    print("DEW Water Model Test")
    print("=" * 25)
    
    # Test extreme conditions
    T_test = 873.15  # 600°C
    P_test = 10000.0  # 10 kbar
    
    rho = water_DEW('rho', T=T_test, P=P_test)
    epsilon = water_DEW('epsilon', T=T_test, P=P_test)
    
    print(f"Water at {T_test} K, {P_test} bar:")
    print(f"  Density: {rho:.3f} g/cm³")
    print(f"  Dielectric constant: {epsilon:.1f}")
    
    # Test Born functions
    born = water_DEW(['QBorn', 'YBorn'], T=T_test, P=P_test)
    print(f"Born functions: Q={born['QBorn']:.2e}, Y={born['YBorn']:.2e}")
    
    # Compare with standard conditions
    T_std = 298.15
    P_std = 1.0
    epsilon_std = water_DEW('epsilon', T=T_std, P=P_std)
    print(f"Standard conditions (25°C, 1 bar): ε = {epsilon_std:.1f}")