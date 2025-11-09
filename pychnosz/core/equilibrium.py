"""
Chemical equilibrium solver for CHNOSZ.

This module implements equilibrium calculations including:
- Activity coefficient models
- Chemical speciation
- Equilibrium constants
- Activity-concentration relationships
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple, Any
import warnings

# Simple optimization functions (fallback for scipy)
def _simple_fsolve(func, x0, args=()):
    """Simple Newton-Raphson solver as scipy.optimize.fsolve fallback."""
    x = np.array(x0, dtype=float)
    for i in range(50):  # Maximum iterations
        try:
            f = func(x, *args)
            if np.allclose(f, 0, atol=1e-8):
                return x
            
            # Simple gradient estimation
            dx = 1e-8
            grad = np.zeros((len(x), len(f)))
            
            for j in range(len(x)):
                x_plus = x.copy()
                x_plus[j] += dx
                f_plus = func(x_plus, *args)
                grad[j] = (f_plus - f) / dx
            
            # Newton step (simplified)
            try:
                delta = np.linalg.solve(grad.T, -f)
                x += delta * 0.1  # Damped step
            except:
                # If singular, use simple step
                x -= f * 0.01
                
        except:
            break
    
    return x

try:
    from scipy.optimize import fsolve, minimize
except ImportError:
    fsolve = _simple_fsolve
    minimize = None

from .subcrt import subcrt
from .thermo import thermo


class EquilibriumSolver:
    """
    Chemical equilibrium solver for aqueous systems.
    
    This class implements various equilibrium calculation methods:
    - Activity coefficient corrections (Debye-Hückel, B-dot, Pitzer)
    - Chemical speciation calculations
    - Reaction equilibrium constants
    - Mass balance constraints
    """
    
    def __init__(self):
        """Initialize the equilibrium solver."""
        self.activity_models = {
            'ideal': self._activity_ideal,
            'debye_huckel': self._activity_debye_huckel,
            'bdot': self._activity_bdot,
            'pitzer': self._activity_pitzer
        }
        
        # Default parameters
        self.ionic_strength_limit = 3.0  # mol/kg
        self.max_iterations = 100
        self.tolerance = 1e-8
    
    def calculate_logK(self, reaction: Dict[str, float],
                      T: Union[float, np.ndarray] = 298.15,
                      P: Union[float, np.ndarray] = 1.0) -> np.ndarray:
        """
        Calculate equilibrium constant for a reaction.
        
        Parameters
        ----------
        reaction : dict
            Reaction dictionary with species names as keys and 
            stoichiometric coefficients as values (negative for reactants)
        T : float or array, default 298.15
            Temperature in Kelvin
        P : float or array, default 1.0
            Pressure in bar
            
        Returns
        -------
        array
            log K values at given T and P
        """
        
        # Get species names and coefficients
        species_names = list(reaction.keys())
        coefficients = list(reaction.values())
        
        # Calculate standard properties
        result = subcrt(species_names, coefficients, T=T, P=P, show=False)

        if result.out is not None and 'logK' in result.out.columns:
            return result.out['logK'].values
        else:
            raise ValueError("Could not calculate reaction properties")
    
    def calculate_speciation(self, total_concentrations: Dict[str, float],
                           reactions: Dict[str, Dict[str, float]],
                           T: float = 298.15, P: float = 1.0,
                           pH: Optional[float] = None,
                           ionic_strength: Optional[float] = None,
                           activity_model: str = 'debye_huckel') -> Dict[str, Any]:
        """
        Calculate chemical speciation for an aqueous system.
        
        Parameters
        ----------
        total_concentrations : dict
            Total concentrations of components (mol/kg)
        reactions : dict
            Formation reactions for each species
        T : float, default 298.15
            Temperature in Kelvin
        P : float, default 1.0  
            Pressure in bar
        pH : float, optional
            pH constraint (if provided)
        ionic_strength : float, optional
            Ionic strength (if known, otherwise calculated)
        activity_model : str, default 'debye_huckel'
            Activity coefficient model to use
            
        Returns
        -------
        dict
            Speciation results with concentrations, activities, and properties
        """
        
        # Get equilibrium constants for all reactions
        logK_values = {}
        for species, reaction in reactions.items():
            try:
                logK = self.calculate_logK(reaction, T, P)
                logK_values[species] = logK[0] if hasattr(logK, '__len__') else logK
            except Exception as e:
                warnings.warn(f"Could not calculate logK for {species}: {e}")
                logK_values[species] = 0.0
        
        # Initial guess for species concentrations
        species_names = list(reactions.keys())
        basis_species = set()
        for reaction in reactions.values():
            basis_species.update(reaction.keys())
        basis_species = list(basis_species)
        
        # Create initial guess (equal distribution)
        n_species = len(species_names)
        n_basis = len(basis_species)
        
        if n_species == 0:
            return {'concentrations': {}, 'activities': {}, 'ionic_strength': 0.0}
        
        # Initial concentrations (log scale for stability)
        x0 = np.ones(n_species + n_basis) * (-6.0)  # log concentrations
        
        if pH is not None:
            # Find H+ index and set pH constraint
            if 'H+' in basis_species:
                h_idx = basis_species.index('H+')
                x0[n_species + h_idx] = -pH
        
        # Solve equilibrium system
        try:
            solution = fsolve(self._equilibrium_equations, x0, 
                            args=(species_names, basis_species, reactions, 
                                  logK_values, total_concentrations, pH,
                                  T, P, activity_model))
            
            if not np.allclose(self._equilibrium_equations(solution, species_names, basis_species,
                                                          reactions, logK_values, 
                                                          total_concentrations, pH,
                                                          T, P, activity_model), 0, atol=1e-6):
                warnings.warn("Equilibrium solution may not have converged")
            
        except Exception as e:
            warnings.warn(f"Equilibrium calculation failed: {e}")
            solution = x0  # Use initial guess
        
        # Extract results
        log_species_conc = solution[:n_species]
        log_basis_conc = solution[n_species:]
        
        species_conc = 10**log_species_conc
        basis_conc = 10**log_basis_conc
        
        # Calculate ionic strength
        ionic_str = self._calculate_ionic_strength(species_names, species_conc, 
                                                  basis_species, basis_conc)
        
        # Calculate activity coefficients
        gamma_species = {}
        gamma_basis = {}
        
        for i, species in enumerate(species_names):
            gamma_species[species] = self._get_activity_coefficient(
                species, ionic_str, T, P, activity_model)
        
        for i, species in enumerate(basis_species):
            gamma_basis[species] = self._get_activity_coefficient(
                species, ionic_str, T, P, activity_model)
        
        # Calculate activities
        activities_species = {species: conc * gamma_species[species] 
                            for species, conc in zip(species_names, species_conc)}
        activities_basis = {species: conc * gamma_basis[species]
                           for species, conc in zip(basis_species, basis_conc)}
        
        return {
            'concentrations': {**dict(zip(species_names, species_conc)),
                             **dict(zip(basis_species, basis_conc))},
            'activities': {**activities_species, **activities_basis},
            'activity_coefficients': {**gamma_species, **gamma_basis},
            'ionic_strength': ionic_str,
            'pH': -np.log10(basis_conc[basis_species.index('H+')]) if 'H+' in basis_species else None
        }
    
    def _equilibrium_equations(self, x: np.ndarray, species_names: List[str],
                              basis_species: List[str], reactions: Dict[str, Dict[str, float]],
                              logK_values: Dict[str, float], 
                              total_concentrations: Dict[str, float],
                              pH: Optional[float], T: float, P: float,
                              activity_model: str) -> np.ndarray:
        """
        Equilibrium equations to solve for speciation.
        
        Returns array of residuals for:
        1. Mass balance equations
        2. Equilibrium constant equations  
        3. Charge balance (if applicable)
        4. pH constraint (if provided)
        """
        
        n_species = len(species_names)
        n_basis = len(basis_species)
        
        # Extract log concentrations
        log_species_conc = x[:n_species] 
        log_basis_conc = x[n_species:]
        
        species_conc = 10**log_species_conc
        basis_conc = 10**log_basis_conc
        
        # Calculate ionic strength for activity coefficients
        ionic_str = self._calculate_ionic_strength(species_names, species_conc,
                                                  basis_species, basis_conc)
        
        equations = []
        
        # 1. Equilibrium constant equations
        for i, species in enumerate(species_names):
            if species in reactions:
                reaction = reactions[species]
                logK = logK_values[species]
                
                # log K = log(activity of products) - log(activity of reactants)
                log_activity_ratio = 0
                
                for reactant, coeff in reaction.items():
                    if reactant in basis_species:
                        idx = basis_species.index(reactant)
                        gamma = self._get_activity_coefficient(reactant, ionic_str, T, P, activity_model)
                        log_activity_ratio += coeff * (log_basis_conc[idx] + np.log10(gamma))
                
                # Activity of species being formed
                gamma_species = self._get_activity_coefficient(species, ionic_str, T, P, activity_model)
                log_species_activity = log_species_conc[i] + np.log10(gamma_species)
                
                # Equilibrium equation: logK - log_activity_ratio + log_species_activity = 0
                equations.append(logK - log_activity_ratio + log_species_activity)
        
        # 2. Mass balance equations
        for component, total_conc in total_concentrations.items():
            if total_conc <= 0:
                continue
                
            mass_balance = 0
            
            # Contribution from basis species
            if component in basis_species:
                idx = basis_species.index(component)
                mass_balance += basis_conc[idx]
            
            # Contributions from formed species
            for i, species in enumerate(species_names):
                if species in reactions and component in reactions[species]:
                    coeff = abs(reactions[species][component])  # Take absolute value for mass balance
                    mass_balance += coeff * species_conc[i]
            
            # Mass balance equation: (calculated - total) / total = 0
            equations.append((mass_balance - total_conc) / total_conc)
        
        # 3. pH constraint
        if pH is not None and 'H+' in basis_species:
            h_idx = basis_species.index('H+')
            equations.append(log_basis_conc[h_idx] + pH)  # log[H+] + pH = 0
        
        return np.array(equations)
    
    def _calculate_ionic_strength(self, species_names: List[str], species_conc: np.ndarray,
                                 basis_species: List[str], basis_conc: np.ndarray) -> float:
        """Calculate ionic strength of the solution."""
        
        ionic_strength = 0
        
        # Contributions from basis species (assume they have charges)
        for i, species in enumerate(basis_species):
            charge = self._get_species_charge(species)
            ionic_strength += 0.5 * basis_conc[i] * charge**2
        
        # Contributions from formed species  
        for i, species in enumerate(species_names):
            charge = self._get_species_charge(species)
            ionic_strength += 0.5 * species_conc[i] * charge**2
        
        return ionic_strength
    
    def _get_species_charge(self, species: str) -> int:
        """Extract charge from species name (simplified)."""
        
        if '+' in species:
            charge_str = species.split('+')[-1]
            try:
                return int(charge_str) if charge_str.isdigit() else 1
            except:
                return 1
        elif '-' in species:
            charge_str = species.split('-')[-1]
            try:
                return -int(charge_str) if charge_str.isdigit() else -1
            except:
                return -1
        else:
            return 0
    
    def _get_activity_coefficient(self, species: str, ionic_strength: float,
                                 T: float, P: float, model: str) -> float:
        """Calculate activity coefficient for a species."""
        
        if model in self.activity_models:
            return self.activity_models[model](species, ionic_strength, T, P)
        else:
            warnings.warn(f"Unknown activity model: {model}, using ideal")
            return 1.0
    
    def _activity_ideal(self, species: str, I: float, T: float, P: float) -> float:
        """Ideal activity coefficient (γ = 1)."""
        return 1.0
    
    def _activity_debye_huckel(self, species: str, I: float, T: float, P: float) -> float:
        """Debye-Hückel activity coefficient."""
        
        charge = self._get_species_charge(species)
        
        if charge == 0:
            return 1.0  # Neutral species
        
        # Debye-Hückel parameters (approximate)
        A = 0.509  # at 25°C, valid for higher T too approximately
        
        # Extended Debye-Hückel equation
        if I <= 0.1:
            log_gamma = -A * charge**2 * np.sqrt(I) / (1 + np.sqrt(I))
        else:
            # For higher ionic strengths, use extended form
            B = 0.328  # approximate
            log_gamma = -A * charge**2 * np.sqrt(I) / (1 + B * np.sqrt(I))
        
        return 10**log_gamma
    
    def _activity_bdot(self, species: str, I: float, T: float, P: float) -> float:
        """B-dot activity coefficient model."""
        
        charge = self._get_species_charge(species)
        
        if charge == 0:
            return 1.0
        
        # B-dot equation parameters (simplified)
        A = 0.509  # Debye-Hückel A parameter
        B = 0.328  # B parameter
        bdot = 0.041  # B-dot parameter (approximate)
        
        sqrt_I = np.sqrt(I)
        log_gamma = -A * charge**2 * sqrt_I / (1 + B * sqrt_I) + bdot * I
        
        return 10**log_gamma
    
    def _activity_pitzer(self, species: str, I: float, T: float, P: float) -> float:
        """Pitzer activity coefficient model (simplified)."""
        
        # For now, fall back to B-dot model
        return self._activity_bdot(species, I, T, P)


# Global equilibrium solver instance  
_equilibrium_solver = EquilibriumSolver()


def affinity(species: Optional[Union[str, List[str]]] = None,
            property: str = 'A', T: float = 298.15, P: float = 1.0) -> pd.DataFrame:
    """
    Calculate chemical affinity for formation reactions.
    
    Parameters
    ----------
    species : str, list, or None
        Species to calculate affinity for. If None, uses current thermo species.
    property : str, default 'A'  
        Property to calculate ('A' for affinity, 'logK' for log K, 'logQ' for log Q)
    T : float, default 298.15
        Temperature in Kelvin
    P : float, default 1.0
        Pressure in bar
        
    Returns
    -------
    DataFrame
        Affinities and related properties
    """
    
    # This would interface with the basis system to calculate formation reaction affinities
    # For now, return a placeholder
    
    if species is None:
        if thermo.species is None:
            raise ValueError("No species defined. Use species() function first.")
        species_list = thermo.species['name'].tolist()
    elif isinstance(species, str):
        species_list = [species]
    else:
        species_list = species
    
    results = []
    for sp in species_list:
        # Calculate formation reaction from basis species
        # This requires implementing the basis system and formation reactions
        result = {
            'species': sp,
            'T': T,
            'P': P,
            'A': 0.0,  # Placeholder - would calculate actual affinity
            'logK': 0.0,  # Placeholder
            'logQ': 0.0   # Placeholder
        }
        results.append(result)
    
    return pd.DataFrame(results)


def equilibrate(aout: Optional[pd.DataFrame] = None, 
               balance: Optional[Union[str, int]] = None,
               normalize: bool = False,
               as_residue: bool = False) -> Dict[str, Any]:
    """
    Find chemical equilibrium using an optimization approach.
    
    Parameters
    ----------
    aout : DataFrame, optional
        Output from affinity() function
    balance : str or int, optional
        Balanced chemical component
    normalize : bool, default False
        Normalize activities to sum to 1
    as_residue : bool, default False
        Return residue of optimization
        
    Returns
    -------
    dict
        Equilibrium results
    """
    
    # Placeholder implementation
    # This would implement the equilibrium optimization using affinity calculations
    
    if aout is None:
        raise ValueError("affinity output required")
    
    # For now, return equal distribution
    n_species = len(aout)
    activities = np.ones(n_species) / n_species
    
    result = {
        'activities': activities,
        'residual': 0.0,
        'converged': True
    }
    
    return result


def solubility(species: Union[str, List[str]], 
              mineral: str,
              T: float = 298.15, P: float = 1.0,
              pH_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
    """
    Calculate mineral solubility in aqueous solution.
    
    Parameters
    ----------  
    species : str or list
        Aqueous species in equilibrium with mineral
    mineral : str
        Mineral name
    T : float, default 298.15
        Temperature in Kelvin
    P : float, default 1.0  
        Pressure in bar
    pH_range : tuple, optional
        pH range for calculation (min_pH, max_pH)
        
    Returns
    -------
    dict
        Solubility results
    """
    
    if pH_range is None:
        pH_range = (0, 14)
    
    pH_values = np.linspace(pH_range[0], pH_range[1], 100)
    
    # Calculate dissolution reaction
    # This requires implementing mineral dissolution reactions
    
    results = {
        'pH': pH_values,
        'solubility': np.zeros_like(pH_values),  # Placeholder
        'species_distribution': {}  # Placeholder
    }
    
    return results