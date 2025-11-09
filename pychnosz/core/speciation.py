"""
Chemical speciation calculation engine for CHNOSZ.

This module provides high-level functions for chemical speciation calculations,
including predominance diagrams, activity-pH diagrams, and Eh-pH diagrams.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Optional, Tuple, Any
import warnings

# Optional matplotlib import
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from .subcrt import subcrt
from .equilibrium import EquilibriumSolver
from .thermo import thermo


class SpeciationEngine:
    """
    Chemical speciation calculation engine.
    
    This class provides methods for creating predominance diagrams,
    activity-concentration diagrams, and other speciation plots.
    """
    
    def __init__(self):
        """Initialize the speciation engine."""
        self.equilibrium_solver = EquilibriumSolver()
        
    def predominance_diagram(self, element: str, 
                           basis_species: List[str],
                           T: float = 298.15, P: float = 1.0,
                           pH_range: Tuple[float, float] = (0, 14),
                           pe_range: Optional[Tuple[float, float]] = None,
                           ionic_strength: float = 0.1,
                           resolution: int = 100) -> Dict[str, Any]:
        """
        Calculate predominance diagram for an element.
        
        Parameters
        ----------
        element : str
            Element symbol (e.g., 'Fe', 'S', 'C')
        basis_species : list of str
            Basis species for the calculation
        T : float, default 298.15
            Temperature in Kelvin
        P : float, default 1.0
            Pressure in bar  
        pH_range : tuple, default (0, 14)
            pH range for diagram
        pe_range : tuple, optional
            pe (redox potential) range. If None, creates pH-only diagram
        ionic_strength : float, default 0.1
            Solution ionic strength (mol/kg)
        resolution : int, default 100
            Grid resolution for calculations
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'pH': pH grid
            - 'pe': pe grid (if pe_range provided)
            - 'predominant': Predominant species grid
            - 'activities': Activity grids for each species
        """
        
        # Get all species containing the element
        element_species = self._find_element_species(element)
        
        if not element_species:
            raise ValueError(f"No species found containing element {element}")
        
        # Create grid
        pH_grid = np.linspace(pH_range[0], pH_range[1], resolution)
        
        if pe_range is not None:
            pe_grid = np.linspace(pe_range[0], pe_range[1], resolution)
            pH_mesh, pe_mesh = np.meshgrid(pH_grid, pe_grid)
            predominant = np.zeros_like(pH_mesh, dtype=int)
            activities = {species: np.zeros_like(pH_mesh) for species in element_species}
        else:
            pe_mesh = None
            predominant = np.zeros_like(pH_grid, dtype=int)
            activities = {species: np.zeros_like(pH_grid) for species in element_species}
        
        # Calculate speciation at each grid point
        for i, pH in enumerate(pH_grid):
            if pe_range is not None:
                for j, pe in enumerate(pe_grid):
                    spec_result = self._calculate_point_speciation(
                        element_species, basis_species, pH, pe, T, P, ionic_strength)
                    
                    # Find predominant species
                    max_activity = -np.inf
                    max_idx = 0
                    for k, species in enumerate(element_species):
                        activity = spec_result['activities'].get(species, 1e-20)
                        activities[species][j, i] = activity
                        if activity > max_activity:
                            max_activity = activity
                            max_idx = k
                    
                    predominant[j, i] = max_idx
            else:
                spec_result = self._calculate_point_speciation(
                    element_species, basis_species, pH, 0, T, P, ionic_strength)
                
                # Find predominant species
                max_activity = -np.inf
                max_idx = 0
                for k, species in enumerate(element_species):
                    activity = spec_result['activities'].get(species, 1e-20)
                    activities[species][i] = activity
                    if activity > max_activity:
                        max_activity = activity
                        max_idx = k
                
                predominant[i] = max_idx
        
        result = {
            'pH': pH_grid,
            'predominant': predominant,
            'activities': activities,
            'species_names': element_species
        }
        
        if pe_range is not None:
            result['pe'] = pe_grid
        
        return result
    
    def activity_diagram(self, species: List[str], 
                        x_variable: str, x_range: Tuple[float, float],
                        y_variable: str, y_range: Tuple[float, float],
                        T: float = 298.15, P: float = 1.0,
                        total_concentrations: Optional[Dict[str, float]] = None,
                        resolution: int = 100) -> Dict[str, Any]:
        """
        Calculate activity diagram (e.g., activity vs pH, Eh-pH).
        
        Parameters
        ----------
        species : list of str
            Species to include in diagram
        x_variable : str
            X-axis variable ('pH', 'pe', 'log_activity', etc.)
        x_range : tuple
            Range for x variable
        y_variable : str  
            Y-axis variable
        y_range : tuple
            Range for y variable
        T : float, default 298.15
            Temperature in Kelvin
        P : float, default 1.0
            Pressure in bar
        total_concentrations : dict, optional
            Total concentrations for components
        resolution : int, default 100
            Grid resolution
            
        Returns
        -------
        dict
            Activity diagram results
        """
        
        # Create grids
        x_grid = np.linspace(x_range[0], x_range[1], resolution)
        y_grid = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        # Initialize result grids
        activities = {species: np.zeros_like(X) for species in species}
        
        # Calculate at each grid point
        for i in range(len(y_grid)):
            for j in range(len(x_grid)):
                x_val, y_val = X[i, j], Y[i, j]
                
                # Set up calculation conditions
                conditions = {
                    'T': T,
                    'P': P,
                    x_variable: x_val,
                    y_variable: y_val
                }
                
                # Calculate speciation (simplified)
                try:
                    for k, sp in enumerate(species):
                        # This would be a full speciation calculation
                        # For now, use placeholder values
                        activities[sp][i, j] = 1e-6  # Placeholder
                        
                except Exception as e:
                    warnings.warn(f"Calculation failed at {x_variable}={x_val}, {y_variable}={y_val}: {e}")
                    for sp in species:
                        activities[sp][i, j] = np.nan
        
        return {
            x_variable: x_grid,
            y_variable: y_grid,
            'activities': activities,
            'species': species
        }
    
    def mosaic_diagram(self, basis1: str, basis2: str,
                      range1: Tuple[float, float], range2: Tuple[float, float],
                      T: float = 298.15, P: float = 1.0,
                      resolution: int = 50) -> Dict[str, Any]:
        """
        Calculate mosaic diagram (stability fields of basis species).
        
        Parameters
        ----------
        basis1, basis2 : str
            Two basis species that define the diagram axes
        range1, range2 : tuple
            Activity ranges for the two basis species
        T : float, default 298.15
            Temperature in Kelvin
        P : float, default 1.0
            Pressure in bar
        resolution : int, default 50
            Grid resolution
            
        Returns
        -------
        dict
            Mosaic diagram results
        """
        
        # Create grid
        log_a1 = np.linspace(range1[0], range1[1], resolution)
        log_a2 = np.linspace(range2[0], range2[1], resolution)
        A1, A2 = np.meshgrid(log_a1, log_a2)
        
        # Find all species that can be formed from these basis species
        formed_species = self._find_formed_species([basis1, basis2])
        
        # Calculate stability fields
        stable_species = np.zeros_like(A1, dtype=int)
        
        for i in range(len(log_a2)):
            for j in range(len(log_a1)):
                activities = {basis1: 10**A1[i, j], basis2: 10**A2[i, j]}
                
                # Find most stable species at this point
                min_energy = np.inf
                stable_idx = 0
                
                for k, species in enumerate(formed_species):
                    try:
                        # Calculate formation energy (simplified)
                        energy = self._calculate_formation_energy(species, activities, T, P)
                        if energy < min_energy:
                            min_energy = energy
                            stable_idx = k
                    except:
                        continue
                
                stable_species[i, j] = stable_idx
        
        return {
            basis1: log_a1,
            basis2: log_a2,
            'stable_species': stable_species,
            'species_names': formed_species
        }
    
    def solubility_diagram(self, mineral: str, aqueous_species: List[str],
                          pH_range: Tuple[float, float] = (0, 14),
                          T: float = 298.15, P: float = 1.0,
                          resolution: int = 100) -> Dict[str, Any]:
        """
        Calculate mineral solubility diagram.
        
        Parameters
        ----------
        mineral : str
            Mineral name
        aqueous_species : list of str
            Aqueous species in equilibrium with mineral
        pH_range : tuple, default (0, 14)
            pH range for calculation
        T : float, default 298.15
            Temperature in Kelvin
        P : float, default 1.0
            Pressure in bar
        resolution : int, default 100
            Grid resolution
            
        Returns
        -------
        dict
            Solubility diagram results
        """
        
        pH_grid = np.linspace(pH_range[0], pH_range[1], resolution)
        
        # Calculate dissolution reaction
        dissolution_reactions = self._get_dissolution_reactions(mineral, aqueous_species)
        
        # Calculate solubility at each pH
        total_solubility = np.zeros_like(pH_grid)
        species_concentrations = {sp: np.zeros_like(pH_grid) for sp in aqueous_species}
        
        for i, pH in enumerate(pH_grid):
            # Calculate equilibrium with mineral
            try:
                # This would involve solving the dissolution equilibrium
                # For now, use placeholder calculations
                for j, species in enumerate(aqueous_species):
                    # Simplified pH dependence
                    if '+' in species:  # Cation
                        conc = 1e-6 * 10**(-pH)
                    elif '-' in species:  # Anion
                        conc = 1e-6 * 10**(pH - 14)
                    else:  # Neutral
                        conc = 1e-6
                    
                    species_concentrations[species][i] = conc
                    total_solubility[i] += conc
                    
            except Exception as e:
                warnings.warn(f"Solubility calculation failed at pH {pH}: {e}")
                total_solubility[i] = np.nan
                for species in aqueous_species:
                    species_concentrations[species][i] = np.nan
        
        return {
            'pH': pH_grid,
            'total_solubility': total_solubility,
            'species_concentrations': species_concentrations,
            'mineral': mineral
        }
    
    def plot_predominance(self, diagram_data: Dict[str, Any], 
                         title: Optional[str] = None,
                         figsize: Tuple[int, int] = (10, 8)):
        """
        Plot predominance diagram.
        
        Parameters
        ----------
        diagram_data : dict
            Data from predominance_diagram()
        title : str, optional
            Plot title
        figsize : tuple, default (10, 8)
            Figure size
            
        Returns
        -------
        matplotlib Figure or None
            The plotted figure (if matplotlib available)
        """
        
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available - cannot create plot")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        pH = diagram_data['pH']
        predominant = diagram_data['predominant']
        species_names = diagram_data['species_names']
        
        if 'pe' in diagram_data:
            # 2D diagram (pH-pe)
            pe = diagram_data['pe']
            pH_mesh, pe_mesh = np.meshgrid(pH, pe)
            
            # Create color map for species
            n_species = len(species_names)
            colors = plt.cm.tab10(np.linspace(0, 1, n_species))
            
            contour = ax.contourf(pH_mesh, pe_mesh, predominant, 
                                levels=np.arange(n_species + 1) - 0.5, 
                                colors=colors[:n_species])
            
            ax.set_xlabel('pH')
            ax.set_ylabel('pe')
            
            # Create legend
            handles = [plt.Rectangle((0,0),1,1, color=colors[i]) 
                      for i in range(n_species)]
            ax.legend(handles, species_names, loc='center left', 
                     bbox_to_anchor=(1, 0.5))
            
        else:
            # 1D diagram (pH only)
            # Convert to step plot showing predominance regions
            pH_edges = np.diff(pH)
            predominant_species = [species_names[i] for i in predominant]
            
            # Find boundaries
            boundaries = []
            current_species = predominant[0]
            
            for i, species_idx in enumerate(predominant[1:], 1):
                if species_idx != current_species:
                    boundaries.append(pH[i])
                    current_species = species_idx
            
            ax.step(pH, [species_names[i] for i in predominant], where='post')
            ax.set_xlabel('pH')
            ax.set_ylabel('Predominant Species')
            
            # Add boundary lines
            for boundary in boundaries:
                ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.7)
        
        if title:
            ax.set_title(title)
        
        plt.tight_layout()
        return fig
    
    def _find_element_species(self, element: str) -> List[str]:
        """Find all species in database containing the specified element."""
        
        if thermo.obigt is None:
            raise ValueError("OBIGT database not loaded")
        
        # Search for species containing the element in their formula
        element_species = []
        
        for _, row in thermo.obigt.iterrows():
            formula = str(row['formula'])
            if element in formula:
                element_species.append(row['name'])
        
        return element_species
    
    def _calculate_point_speciation(self, species: List[str], basis_species: List[str],
                                   pH: float, pe: float, T: float, P: float,
                                   ionic_strength: float) -> Dict[str, Any]:
        """Calculate speciation at a single point."""
        
        # This would involve full equilibrium calculation
        # For now, return simplified results
        
        activities = {}
        for sp in species:
            # Simplified activity calculation based on pH and pe
            if '+' in sp:  # Cation
                activity = 1e-6 * 10**(-pH) * 10**(pe)
            elif '-' in sp:  # Anion  
                activity = 1e-6 * 10**(pH - 14) * 10**(-pe)
            else:  # Neutral
                activity = 1e-6
            
            activities[sp] = max(activity, 1e-20)  # Avoid zero activities
        
        return {'activities': activities}
    
    def _find_formed_species(self, basis_species: List[str]) -> List[str]:
        """Find species that can be formed from the given basis species."""
        
        # Placeholder - would search database for formation reactions
        return basis_species  # Simplified
    
    def _calculate_formation_energy(self, species: str, activities: Dict[str, float],
                                   T: float, P: float) -> float:
        """Calculate formation energy for a species."""
        
        # Placeholder - would calculate actual formation reaction energy
        return np.random.random()  # Simplified
    
    def _get_dissolution_reactions(self, mineral: str, aqueous_species: List[str]) -> Dict:
        """Get dissolution reactions for a mineral."""
        
        # Placeholder - would look up actual dissolution reactions
        return {}


# Global speciation engine instance
_speciation_engine = SpeciationEngine()


def diagram(basis: Optional[Union[str, List[str]]] = None,
           species: Optional[Union[str, List[str]]] = None,
           balance: Optional[Union[str, int]] = None,
           loga_balance: Optional[Union[float, List[float]]] = None,
           T: float = 298.15, P: float = 1.0,
           res: int = 256,
           **kwargs) -> Dict[str, Any]:
    """
    Create chemical activity or predominance diagram.
    
    Parameters
    ----------
    basis : str, list, or None
        Basis species for diagram axes  
    species : str, list, or None
        Species to include in diagram
    balance : str, int, or None
        Balanced component
    loga_balance : float, list, or None  
        Log activities for balance
    T : float, default 298.15
        Temperature in Kelvin
    P : float, default 1.0
        Pressure in bar
    res : int, default 256
        Diagram resolution
    **kwargs
        Additional parameters for diagram calculation
        
    Returns
    -------
    dict
        Diagram calculation results
    """
    
    # This is a placeholder for the main diagram function
    # It would coordinate with the basis system and species definitions
    
    if basis is None or species is None:
        raise ValueError("Both basis and species must be specified")
    
    if isinstance(basis, str):
        basis = [basis]
    if isinstance(species, str):
        species = [species]
    
    # Create simple predominance diagram
    result = _speciation_engine.predominance_diagram(
        element=basis[0].split('+')[0].split('-')[0],  # Extract element
        basis_species=basis,
        T=T, P=P,
        resolution=res
    )
    
    return result


def mosaic(bases: List[str], 
          T: float = 298.15, P: float = 1.0,
          resolution: int = 256) -> Dict[str, Any]:
    """
    Create mosaic diagram showing stability fields.
    
    Parameters
    ----------
    bases : list of str
        Two basis species defining the diagram
    T : float, default 298.15
        Temperature in Kelvin  
    P : float, default 1.0
        Pressure in bar
    resolution : int, default 256
        Diagram resolution
        
    Returns
    -------
    dict
        Mosaic diagram results
    """
    
    if len(bases) != 2:
        raise ValueError("Exactly two basis species required for mosaic diagram")
    
    return _speciation_engine.mosaic_diagram(
        bases[0], bases[1],
        range1=(-12, 0), range2=(-12, 0),  # Default activity ranges
        T=T, P=P, resolution=resolution
    )