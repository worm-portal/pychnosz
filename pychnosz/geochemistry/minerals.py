"""
Mineral equilibria and phase diagram calculations for CHNOSZ.

This module implements mineral stability calculations, phase diagrams,
and solid-solution equilibria for geological applications.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple, Any
import warnings

from ..core.subcrt import subcrt
from ..core.equilibrium import EquilibriumSolver
from ..core.speciation import SpeciationEngine


class MineralEquilibria:
    """
    Mineral equilibria calculator for geological systems.
    
    This class handles mineral stability calculations, including:
    - Mineral solubility equilibria
    - Phase stability diagrams
    - Mineral-water reactions
    - Solid solution equilibria
    """
    
    def __init__(self):
        """Initialize the mineral equilibria calculator."""
        self.equilibrium_solver = EquilibriumSolver()
        self.speciation_engine = SpeciationEngine()
        
        # Common mineral dissolution reactions (simplified)
        self.dissolution_reactions = {
            'quartz': {'SiO2': 1, 'H2O': 0, 'H4SiO4': -1},
            'calcite': {'CaCO3': 1, 'H+': 2, 'Ca+2': -1, 'HCO3-': -1, 'H2O': -1},
            'pyrite': {'FeS2': 1, 'H+': 8, 'SO4-2': -2, 'Fe+2': -1, 'H2O': -4},
            'hematite': {'Fe2O3': 1, 'H+': 6, 'Fe+3': -2, 'H2O': -3},
            'magnetite': {'Fe3O4': 1, 'H+': 8, 'Fe+2': -1, 'Fe+3': -2, 'H2O': -4},
            'albite': {'NaAlSi3O8': 1, 'H+': 1, 'H2O': 4, 'Na+': -1, 'Al+3': -1, 'H4SiO4': -3},
            'anorthite': {'CaAl2Si2O8': 1, 'H+': 8, 'H2O': 0, 'Ca+2': -1, 'Al+3': -2, 'H4SiO4': -2},
            'kaolinite': {'Al2Si2O5(OH)4': 1, 'H+': 6, 'Al+3': -2, 'H4SiO4': -2, 'H2O': -1}
        }
    
    def mineral_solubility(self, mineral: str, 
                          T: float = 298.15, P: float = 1.0,
                          pH_range: Optional[Tuple[float, float]] = None,
                          ionic_strength: float = 0.0) -> Dict[str, Any]:
        """
        Calculate mineral solubility as a function of pH.
        
        Parameters
        ----------
        mineral : str
            Mineral name
        T : float, default 298.15
            Temperature in Kelvin
        P : float, default 1.0
            Pressure in bar
        pH_range : tuple, optional
            pH range (min, max). Default: (0, 14)
        ionic_strength : float, default 0.0
            Solution ionic strength
            
        Returns
        -------
        dict
            Solubility data vs pH
        """
        
        if pH_range is None:
            pH_range = (0, 14)
        
        pH_values = np.linspace(pH_range[0], pH_range[1], 100)
        
        if mineral.lower() not in self.dissolution_reactions:
            warnings.warn(f"Dissolution reaction not defined for {mineral}")
            return {'pH': pH_values, 'solubility': np.zeros_like(pH_values)}
        
        reaction = self.dissolution_reactions[mineral.lower()]
        
        try:
            # Get equilibrium constant for dissolution
            species_names = list(reaction.keys())
            coefficients = list(reaction.values())
            
            # Calculate logK at T, P
            result = subcrt(species_names, coefficients, T=T, P=P, show=False)
            if result.out is not None and 'logK' in result.out.columns:
                logK = result.out['logK'].iloc[0]
            else:
                logK = 0.0
                warnings.warn(f"Could not calculate logK for {mineral} dissolution")
            
        except Exception as e:
            warnings.warn(f"Error calculating equilibrium constant: {e}")
            logK = 0.0
        
        # Calculate solubility at each pH
        solubility = np.zeros_like(pH_values)
        
        for i, pH in enumerate(pH_values):
            try:
                # Simplified solubility calculation
                # This would be more complex in a full implementation
                
                if 'H+' in reaction:
                    h_coeff = reaction['H+']
                    # Basic pH dependence
                    log_solubility = logK - h_coeff * pH
                else:
                    log_solubility = logK
                
                # Convert to molality
                solubility[i] = 10**log_solubility
                
                # Apply ionic strength corrections if needed
                if ionic_strength > 0:
                    # Simple activity coefficient correction
                    gamma = self._estimate_activity_coefficient(ionic_strength)
                    solubility[i] *= gamma
                
            except:
                solubility[i] = 1e-20  # Very low solubility
        
        return {
            'pH': pH_values,
            'solubility': solubility,
            'logK': logK,
            'mineral': mineral,
            'T': T,
            'P': P
        }
    
    def stability_diagram(self, minerals: List[str],
                         x_axis: str, x_range: Tuple[float, float],
                         y_axis: str, y_range: Tuple[float, float],
                         T: float = 298.15, P: float = 1.0,
                         resolution: int = 50) -> Dict[str, Any]:
        """
        Create mineral stability diagram.
        
        Parameters
        ----------
        minerals : list of str
            Mineral names to consider
        x_axis : str
            X-axis variable ('pH', 'log_activity_SiO2', etc.)
        x_range : tuple
            X-axis range (min, max)
        y_axis : str
            Y-axis variable
        y_range : tuple
            Y-axis range (min, max)
        T : float, default 298.15
            Temperature in Kelvin
        P : float, default 1.0
            Pressure in bar
        resolution : int, default 50
            Grid resolution
            
        Returns
        -------
        dict
            Stability diagram data
        """
        
        x_values = np.linspace(x_range[0], x_range[1], resolution)
        y_values = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_values, y_values)
        
        # Initialize stability field array
        stable_mineral = np.zeros_like(X, dtype=int)
        
        # Calculate stability at each grid point
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                x_val, y_val = X[i, j], Y[i, j]
                
                # Find most stable mineral at this point
                min_energy = np.inf
                stable_idx = 0
                
                for k, mineral in enumerate(minerals):
                    try:
                        # Calculate relative stability
                        # This would involve full equilibrium calculations
                        # For now, use simplified energy estimates
                        
                        energy = self._calculate_stability_energy(
                            mineral, x_axis, x_val, y_axis, y_val, T, P)
                        
                        if energy < min_energy:
                            min_energy = energy
                            stable_idx = k
                        
                    except Exception as e:
                        warnings.warn(f"Error calculating stability for {mineral}: {e}")
                        continue
                
                stable_mineral[i, j] = stable_idx
        
        return {
            x_axis: x_values,
            y_axis: y_values,
            'stable_mineral': stable_mineral,
            'mineral_names': minerals,
            'T': T,
            'P': P
        }
    
    def phase_equilibrium(self, reaction: Dict[str, float],
                         T_range: Optional[Tuple[float, float]] = None,
                         P_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        Calculate phase equilibrium curve.
        
        Parameters
        ----------
        reaction : dict
            Reaction stoichiometry {species: coefficient}
        T_range : tuple, optional
            Temperature range (min, max) in K
        P_range : tuple, optional
            Pressure range (min, max) in bar
            
        Returns
        -------
        dict
            Equilibrium curve data
        """
        
        if T_range is None and P_range is None:
            T_range = (273.15, 673.15)  # 0-400°C
        
        if T_range is not None:
            # T-P curve at equilibrium
            T_values = np.linspace(T_range[0], T_range[1], 50)
            P_equilibrium = np.zeros_like(T_values)
            
            for i, T in enumerate(T_values):
                try:
                    # Calculate equilibrium pressure
                    # This would involve iterative solution
                    # For now, use simplified correlation
                    P_equilibrium[i] = self._calculate_equilibrium_pressure(reaction, T)
                    
                except:
                    P_equilibrium[i] = 1.0  # Default
            
            return {
                'T': T_values,
                'P': P_equilibrium,
                'reaction': reaction
            }
        
        else:
            # P-T curve - similar logic
            P_values = np.linspace(P_range[0], P_range[1], 50)
            T_equilibrium = np.zeros_like(P_values)
            
            for i, P in enumerate(P_values):
                try:
                    T_equilibrium[i] = self._calculate_equilibrium_temperature(reaction, P)
                except:
                    T_equilibrium[i] = 298.15
            
            return {
                'P': P_values,
                'T': T_equilibrium,
                'reaction': reaction
            }
    
    def solid_solution(self, end_members: List[str], compositions: List[float],
                      T: float = 298.15, P: float = 1.0,
                      model: str = 'ideal') -> Dict[str, Any]:
        """
        Calculate solid solution thermodynamics.
        
        Parameters
        ----------
        end_members : list of str
            End-member mineral names
        compositions : list of float
            Mole fractions of end-members (must sum to 1)
        T : float, default 298.15
            Temperature in Kelvin
        P : float, default 1.0
            Pressure in bar
        model : str, default 'ideal'
            Mixing model ('ideal', 'regular', 'margules')
            
        Returns
        -------
        dict
            Solid solution properties
        """
        
        if len(end_members) != len(compositions):
            raise ValueError("Number of end-members must match number of compositions")
        
        if abs(sum(compositions) - 1.0) > 1e-6:
            raise ValueError("Compositions must sum to 1.0")
        
        # Calculate end-member properties
        end_member_props = []
        for mineral in end_members:
            try:
                # This would get mineral properties from database
                props = self._get_mineral_properties(mineral, T, P)
                end_member_props.append(props)
            except:
                # Default properties
                props = {'G': -1000000, 'H': -1200000, 'S': 100, 'V': 50, 'Cp': 100}
                end_member_props.append(props)
        
        # Calculate mixing properties
        if model == 'ideal':
            # Ideal mixing
            G_mix = 8.314 * T * sum(x * np.log(x) for x in compositions if x > 0)
            H_mix = 0.0
            S_mix = -8.314 * sum(x * np.log(x) for x in compositions if x > 0)
            
        elif model == 'regular':
            # Regular solution (symmetric)
            W = 10000  # Interaction parameter (J/mol) - would be mineral-specific
            G_mix = W * compositions[0] * compositions[1]  # Binary case
            H_mix = G_mix
            S_mix = 0.0
            
        else:
            # Default to ideal
            G_mix = 8.314 * T * sum(x * np.log(x) for x in compositions if x > 0)
            H_mix = 0.0
            S_mix = -8.314 * sum(x * np.log(x) for x in compositions if x > 0)
        
        # Total properties
        total_props = {}
        for prop in ['G', 'H', 'S', 'V', 'Cp']:
            total_props[prop] = sum(compositions[i] * end_member_props[i][prop] 
                                  for i in range(len(end_members)))
        
        # Add mixing contributions
        total_props['G'] += G_mix
        total_props['H'] += H_mix
        total_props['S'] += S_mix
        
        return {
            'properties': total_props,
            'mixing': {'G': G_mix, 'H': H_mix, 'S': S_mix},
            'end_members': end_members,
            'compositions': compositions,
            'model': model
        }
    
    def _estimate_activity_coefficient(self, ionic_strength: float) -> float:
        """Estimate activity coefficient from ionic strength."""
        if ionic_strength <= 0:
            return 1.0
        else:
            # Simple Debye-Hückel approximation
            A = 0.509
            return 10**(-A * np.sqrt(ionic_strength))
    
    def _calculate_stability_energy(self, mineral: str, x_axis: str, x_val: float,
                                  y_axis: str, y_val: float, T: float, P: float) -> float:
        """Calculate relative stability energy for mineral."""
        
        # Simplified stability calculation
        # Would involve full thermodynamic analysis
        
        base_energy = np.random.random() * 1000  # Placeholder
        
        # Add dependencies on axes
        if 'pH' in x_axis:
            base_energy += abs(x_val - 7) * 100  # pH dependence
        
        if 'log' in y_axis:
            base_energy += abs(y_val + 3) * 50  # Activity dependence
        
        return base_energy
    
    def _calculate_equilibrium_pressure(self, reaction: Dict[str, float], T: float) -> float:
        """Calculate equilibrium pressure for reaction at given T."""
        
        # Simplified using Clapeyron equation approximation
        # dP/dT = ΔS/ΔV
        
        # Estimate volume and entropy changes
        delta_V = 5.0e-6  # m³/mol (typical for mineral reactions)
        delta_S = 20.0    # J/(mol·K) (typical)
        
        # Integrate from reference conditions
        T0, P0 = 298.15, 1.0
        dP_dT = delta_S / delta_V * 1e-5  # Convert to bar/K
        
        P_eq = P0 + dP_dT * (T - T0)
        
        return max(P_eq, 0.1)  # Minimum 0.1 bar
    
    def _calculate_equilibrium_temperature(self, reaction: Dict[str, float], P: float) -> float:
        """Calculate equilibrium temperature for reaction at given P."""
        
        # Inverse of pressure calculation
        delta_V = 5.0e-6
        delta_S = 20.0
        
        T0, P0 = 298.15, 1.0
        dT_dP = delta_V / delta_S * 1e5  # K/bar
        
        T_eq = T0 + dT_dP * (P - P0)
        
        return max(T_eq, 273.15)  # Minimum 0°C
    
    def _get_mineral_properties(self, mineral: str, T: float, P: float) -> Dict[str, float]:
        """Get thermodynamic properties of a mineral."""
        
        # This would look up mineral in database
        # For now, return typical values
        
        typical_props = {
            'quartz': {'G': -856300, 'H': -910700, 'S': 41.5, 'V': 22.7, 'Cp': 44.6},
            'calcite': {'G': -1128800, 'H': -1207400, 'S': 91.7, 'V': 36.9, 'Cp': 83.5},
            'pyrite': {'G': -160200, 'H': -171500, 'S': 52.9, 'V': 23.9, 'Cp': 62.2},
            'hematite': {'G': -744400, 'H': -824200, 'S': 87.4, 'V': 30.3, 'Cp': 103.9}
        }
        
        return typical_props.get(mineral.lower(), {
            'G': -500000, 'H': -600000, 'S': 50, 'V': 30, 'Cp': 80
        })


# Global mineral equilibria calculator
_mineral_calculator = MineralEquilibria()


def mineral_solubility(mineral: str, pH_range: Tuple[float, float] = (0, 14),
                      T: float = 298.15, P: float = 1.0) -> Dict[str, Any]:
    """
    Calculate mineral solubility vs pH.
    
    Parameters
    ----------
    mineral : str
        Mineral name
    pH_range : tuple, default (0, 14)
        pH range for calculation
    T : float, default 298.15
        Temperature in Kelvin
    P : float, default 1.0
        Pressure in bar
        
    Returns
    -------
    dict
        Solubility data
    """
    
    return _mineral_calculator.mineral_solubility(mineral, T, P, pH_range)


def stability_field(minerals: List[str], 
                   variables: Tuple[str, str],
                   ranges: Tuple[Tuple[float, float], Tuple[float, float]],
                   T: float = 298.15, P: float = 1.0) -> Dict[str, Any]:
    """
    Calculate mineral stability fields.
    
    Parameters
    ----------
    minerals : list of str
        Minerals to consider
    variables : tuple of str
        (x_variable, y_variable) for diagram
    ranges : tuple of tuples
        ((x_min, x_max), (y_min, y_max))
    T : float, default 298.15
        Temperature in Kelvin
    P : float, default 1.0
        Pressure in bar
        
    Returns
    -------
    dict
        Stability diagram data
    """
    
    x_var, y_var = variables
    x_range, y_range = ranges
    
    return _mineral_calculator.stability_diagram(
        minerals, x_var, x_range, y_var, y_range, T, P)


def phase_boundary(reaction: Dict[str, float],
                  T_range: Tuple[float, float] = (273.15, 673.15)) -> Dict[str, Any]:
    """
    Calculate phase boundary for a reaction.
    
    Parameters
    ----------
    reaction : dict
        Reaction stoichiometry {species: coefficient}
    T_range : tuple, default (273.15, 673.15)
        Temperature range in Kelvin
        
    Returns
    -------
    dict
        Phase boundary data
    """
    
    return _mineral_calculator.phase_equilibrium(reaction, T_range=T_range)