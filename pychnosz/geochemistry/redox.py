"""
Redox reactions and Eh-pH diagram calculations for CHNOSZ.

This module implements redox equilibria calculations, Eh-pH diagrams,
and electron activity (pe) calculations for environmental geochemistry.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple, Any
import warnings

from ..core.subcrt import subcrt
from ..core.equilibrium import EquilibriumSolver


class RedoxCalculator:
    """
    Redox equilibria calculator for geochemical systems.
    
    This class handles redox reactions, pe-pH diagrams, and
    electron activity calculations.
    """
    
    def __init__(self):
        """Initialize the redox calculator."""
        self.equilibrium_solver = EquilibriumSolver()
        
        # Standard electrode potentials (V) at 25°C
        self.standard_potentials = {
            'O2/H2O': 1.229,
            'H+/H2': 0.000,
            'Fe+3/Fe+2': 0.771,
            'NO3-/NO2-': 0.835,
            'SO4-2/HS-': -0.217,
            'CO2/CH4': -0.244,
            'N2/NH4+': -0.277,
            'Fe+2/Fe': -0.447,
            'S/HS-': -0.065,
        }
        
        # Common redox couples and their reactions
        self.redox_reactions = {
            # Oxygen reduction
            'O2_H2O': {'O2': 1, 'H+': 4, 'e-': 4, 'H2O': -2},
            'H2O_H2': {'H2O': 2, 'e-': 2, 'H2': -1, 'OH-': -2},
            
            # Iron redox
            'Fe3_Fe2': {'Fe+3': 1, 'e-': 1, 'Fe+2': -1},
            'Fe2_Fe': {'Fe+2': 1, 'e-': 2, 'Fe': -1},
            'Fe2O3_Fe2': {'Fe2O3': 1, 'H+': 6, 'e-': 2, 'Fe+2': -2, 'H2O': -3},
            
            # Nitrogen redox
            'NO3_NO2': {'NO3-': 1, 'H+': 2, 'e-': 2, 'NO2-': -1, 'H2O': -1},
            'NO2_NH4': {'NO2-': 1, 'H+': 8, 'e-': 6, 'NH4+': -1, 'H2O': -2},
            
            # Sulfur redox
            'SO4_HS': {'SO4-2': 1, 'H+': 9, 'e-': 8, 'HS-': -1, 'H2O': -4},
            'S_HS': {'S': 1, 'H+': 1, 'e-': 2, 'HS-': -1},
            
            # Carbon redox
            'CO2_CH4': {'CO2': 1, 'H+': 8, 'e-': 8, 'CH4': -1, 'H2O': -2},
            'HCO3_CH4': {'HCO3-': 1, 'H+': 9, 'e-': 8, 'CH4': -1, 'H2O': -3}
        }
    
    def eh_ph_diagram(self, element: str, 
                     pH_range: Tuple[float, float] = (0, 14),
                     pe_range: Tuple[float, float] = (-10, 15),
                     T: float = 298.15, P: float = 1.0,
                     total_concentration: float = 1e-6,
                     resolution: int = 100) -> Dict[str, Any]:
        """
        Create Eh-pH (pe-pH) diagram for an element.
        
        Parameters
        ----------
        element : str
            Element symbol (e.g., 'Fe', 'S', 'N', 'C')
        pH_range : tuple, default (0, 14)
            pH range for diagram
        pe_range : tuple, default (-10, 15)
            pe (electron activity) range
        T : float, default 298.15
            Temperature in Kelvin
        P : float, default 1.0
            Pressure in bar
        total_concentration : float, default 1e-6
            Total element concentration (molal)
        resolution : int, default 100
            Grid resolution
            
        Returns
        -------
        dict
            Eh-pH diagram data
        """
        
        pH_grid = np.linspace(pH_range[0], pH_range[1], resolution)
        pe_grid = np.linspace(pe_range[0], pe_range[1], resolution)
        pH_mesh, pe_mesh = np.meshgrid(pH_grid, pe_grid)
        
        # Get species for this element
        element_species = self._get_element_species(element)
        
        if not element_species:
            raise ValueError(f"No species found for element {element}")
        
        # Calculate predominance at each point
        predominant = np.zeros_like(pH_mesh, dtype=int)
        activities = {species: np.zeros_like(pH_mesh) for species in element_species}
        
        for i in range(len(pe_grid)):
            for j in range(len(pH_grid)):
                pH, pe = pH_mesh[i, j], pe_mesh[i, j]
                
                # Calculate speciation at this point
                spec_result = self._calculate_redox_speciation(
                    element_species, pH, pe, T, P, total_concentration)
                
                # Find predominant species
                max_activity = -np.inf
                max_idx = 0
                
                for k, species in enumerate(element_species):
                    activity = spec_result.get(species, 1e-20)
                    activities[species][i, j] = activity
                    
                    if np.log10(activity) > max_activity:
                        max_activity = np.log10(activity)
                        max_idx = k
                
                predominant[i, j] = max_idx
        
        # Add water stability limits
        water_limits = self._water_stability_lines(pH_grid, T, P)
        
        return {
            'pH': pH_grid,
            'pe': pe_grid,
            'pH_mesh': pH_mesh,
            'pe_mesh': pe_mesh,
            'predominant': predominant,
            'activities': activities,
            'species_names': element_species,
            'water_limits': water_limits,
            'element': element,
            'T': T,
            'P': P,
            'total_concentration': total_concentration
        }
    
    def pe_calculation(self, redox_couple: Union[str, Dict[str, float]],
                      concentrations: Dict[str, float],
                      pH: float = 7.0, T: float = 298.15, P: float = 1.0) -> float:
        """
        Calculate pe (electron activity) for a redox couple.
        
        Parameters
        ----------
        redox_couple : str or dict
            Redox couple name or reaction dictionary
        concentrations : dict
            Species concentrations {species: concentration}
        pH : float, default 7.0
            Solution pH
        T : float, default 298.15
            Temperature in Kelvin
        P : float, default 1.0
            Pressure in bar
            
        Returns
        -------
        float
            pe value
        """
        
        if isinstance(redox_couple, str):
            if redox_couple not in self.redox_reactions:
                raise ValueError(f"Unknown redox couple: {redox_couple}")
            reaction = self.redox_reactions[redox_couple]
        else:
            reaction = redox_couple
        
        # Calculate equilibrium constant
        try:
            species_names = [sp for sp in reaction.keys() if sp != 'e-']
            coefficients = [reaction[sp] for sp in species_names]
            
            result = subcrt(species_names, coefficients, T=T, P=P, show=False)
            if result.out is not None and 'logK' in result.out.columns:
                logK = result.out['logK'].iloc[0]
            else:
                logK = 0.0
        except Exception as e:
            warnings.warn(f"Could not calculate logK: {e}")
            logK = 0.0
        
        # Apply Nernst equation
        n_electrons = abs(reaction.get('e-', 1))  # Number of electrons
        
        # Calculate activity quotient
        log_Q = 0.0
        for species, coeff in reaction.items():
            if species == 'e-':
                continue
            elif species == 'H+':
                activity = 10**(-pH)
            elif species in concentrations:
                activity = concentrations[species]
                # Apply activity coefficients if needed
                gamma = self._get_activity_coefficient(species, concentrations, T)
                activity *= gamma
            else:
                activity = 1.0  # Default for species not specified
            
            if activity > 0:
                log_Q += coeff * np.log10(activity)
        
        # Nernst equation: pe = pe° + (1/n) * log(Q)
        # where pe° = logK/n for the half-reaction
        pe = logK / n_electrons + log_Q / n_electrons
        
        return pe
    
    def eh_from_pe(self, pe: float, T: float = 298.15) -> float:
        """
        Convert pe to Eh (redox potential).
        
        Parameters
        ----------
        pe : float
            Electron activity (pe)
        T : float, default 298.15
            Temperature in Kelvin
            
        Returns
        -------
        float
            Redox potential (Eh) in Volts
        """
        
        # Eh = (RT/F) * ln(10) * pe
        # where R = 8.314 J/(mol·K), F = 96485 C/mol
        RT_F = 8.314 * T / 96485
        Eh = RT_F * 2.302585 * pe  # ln(10) = 2.302585
        
        return Eh
    
    def pe_from_eh(self, eh: float, T: float = 298.15) -> float:
        """
        Convert Eh (redox potential) to pe.
        
        Parameters
        ----------
        eh : float
            Redox potential (Eh) in Volts
        T : float, default 298.15
            Temperature in Kelvin
            
        Returns
        -------
        float
            Electron activity (pe)
        """
        
        # pe = F * Eh / (RT * ln(10))
        RT_F = 8.314 * T / 96485
        pe = eh / (RT_F * 2.302585)
        
        return pe
    
    def oxygen_fugacity(self, pe: float, pH: float = 7.0, 
                       T: float = 298.15, P: float = 1.0) -> float:
        """
        Calculate oxygen fugacity from pe and pH.
        
        Parameters
        ----------
        pe : float
            Electron activity
        pH : float, default 7.0
            Solution pH
        T : float, default 298.15
            Temperature in Kelvin
        P : float, default 1.0
            Pressure in bar
            
        Returns
        -------
        float
            log fO2 (log oxygen fugacity)
        """
        
        # O2 + 4H+ + 4e- = 2H2O
        # At equilibrium: log fO2 = 4*pe + 4*pH - logK
        
        try:
            # Calculate logK for oxygen-water reaction
            result = subcrt(['O2', 'H+', 'H2O'], [1, 4, -2], T=T, P=P, show=False)
            if result.out is not None and 'logK' in result.out.columns:
                logK = result.out['logK'].iloc[0]
            else:
                logK = 83.1  # Approximate value at 25°C
        except:
            logK = 83.1
        
        log_fO2 = 4 * pe + 4 * pH - logK
        
        return log_fO2
    
    def _get_element_species(self, element: str) -> List[str]:
        """Get list of species containing the specified element."""
        
        # Simplified species lists for common elements
        element_species = {
            'Fe': ['Fe+2', 'Fe+3', 'Fe2O3', 'FeOH+', 'Fe(OH)2', 'Fe(OH)3'],
            'S': ['SO4-2', 'SO3-2', 'S2O3-2', 'HS-', 'S0', 'S-2'],
            'N': ['NO3-', 'NO2-', 'NH4+', 'NH3', 'N2O', 'N2'],
            'C': ['CO2', 'HCO3-', 'CO3-2', 'CH4', 'HCOOH', 'CH3COO-'],
            'Mn': ['Mn+2', 'MnO4-', 'MnO2', 'Mn+3', 'MnOH+'],
            'As': ['AsO4-3', 'AsO3-3', 'H3AsO4', 'H3AsO3', 'As0']
        }
        
        return element_species.get(element, [f'{element}+2', f'{element}+3'])
    
    def _calculate_redox_speciation(self, species: List[str], pH: float, pe: float,
                                  T: float, P: float, total_conc: float) -> Dict[str, float]:
        """Calculate speciation at given pH and pe."""
        
        # Simplified speciation calculation
        # Full implementation would solve equilibrium system
        
        activities = {}
        
        for sp in species:
            # Simplified pe-pH dependence
            if '+' in sp:  # Cations (more stable at low pH, high pe)
                charge = self._extract_charge(sp)
                log_activity = -6 + charge * (14 - pH) / 14 + pe / 10
            elif '-' in sp:  # Anions (more stable at high pH, high pe)  
                charge = abs(self._extract_charge(sp))
                log_activity = -6 + charge * pH / 14 + pe / 10
            else:  # Neutral (less pH/pe dependence)
                log_activity = -6 + (pe - pH) / 20
            
            activities[sp] = 10**log_activity
        
        # Normalize to total concentration
        total_calculated = sum(activities.values())
        if total_calculated > 0:
            factor = total_conc / total_calculated
            activities = {sp: act * factor for sp, act in activities.items()}
        
        return activities
    
    def _extract_charge(self, species: str) -> int:
        """Extract charge from species name."""
        
        if '+' in species:
            parts = species.split('+')
            if len(parts) > 1 and parts[-1].isdigit():
                return int(parts[-1])
            else:
                return 1
        elif '-' in species:
            parts = species.split('-')
            if len(parts) > 1 and parts[-1].isdigit():
                return -int(parts[-1])
            else:
                return -1
        else:
            return 0
    
    def _water_stability_lines(self, pH_values: np.ndarray, 
                              T: float, P: float) -> Dict[str, np.ndarray]:
        """Calculate water stability limits (H2O/H2 and O2/H2O)."""
        
        # Upper limit: O2/H2O
        # O2 + 4H+ + 4e- = 2H2O
        # pe = 20.75 - pH (at 25°C, 1 atm O2)
        pe_upper = 20.75 - pH_values
        
        # Lower limit: H2O/H2
        # 2H2O + 2e- = H2 + 2OH-
        # pe = -pH (at 25°C, 1 atm H2)
        pe_lower = -pH_values
        
        # Temperature corrections (simplified)
        if T != 298.15:
            dT = T - 298.15
            pe_upper += dT * 0.001  # Small temperature dependence
            pe_lower -= dT * 0.001
        
        return {
            'upper': pe_upper,  # O2/H2O line
            'lower': pe_lower   # H2O/H2 line
        }
    
    def _get_activity_coefficient(self, species: str, composition: Dict[str, float],
                                 T: float) -> float:
        """Get activity coefficient for species."""
        
        # Simplified - would use proper activity models
        charge = abs(self._extract_charge(species))
        
        if charge == 0:
            return 1.0
        else:
            # Simple ionic strength correction
            I = 0.001  # Assume low ionic strength
            return 10**(-0.509 * charge**2 * np.sqrt(I))


# Global redox calculator
_redox_calculator = RedoxCalculator()


def eh_ph(element: str, **kwargs) -> Dict[str, Any]:
    """
    Create Eh-pH diagram for an element.
    
    Parameters
    ----------
    element : str
        Element symbol
    **kwargs
        Additional parameters for eh_ph_diagram()
        
    Returns
    -------
    dict
        Eh-pH diagram data
    """
    
    return _redox_calculator.eh_ph_diagram(element, **kwargs)


def pe(redox_couple: Union[str, Dict[str, float]], 
      concentrations: Dict[str, float], **kwargs) -> float:
    """
    Calculate pe for a redox couple.
    
    Parameters
    ----------
    redox_couple : str or dict
        Redox couple or reaction
    concentrations : dict
        Species concentrations
    **kwargs
        Additional parameters
        
    Returns
    -------
    float
        pe value
    """
    
    return _redox_calculator.pe_calculation(redox_couple, concentrations, **kwargs)


def eh(pe_value: float, T: float = 298.15) -> float:
    """
    Convert pe to Eh.
    
    Parameters
    ----------
    pe_value : float
        Electron activity
    T : float, default 298.15
        Temperature in Kelvin
        
    Returns
    -------
    float
        Eh in Volts
    """
    
    return _redox_calculator.eh_from_pe(pe_value, T)


def logfO2(pe_value: float, pH: float = 7.0, **kwargs) -> float:
    """
    Calculate log oxygen fugacity.
    
    Parameters
    ----------
    pe_value : float
        Electron activity
    pH : float, default 7.0
        Solution pH
    **kwargs
        Additional parameters
        
    Returns
    -------
    float
        log fO2
    """
    
    return _redox_calculator.oxygen_fugacity(pe_value, pH, **kwargs)