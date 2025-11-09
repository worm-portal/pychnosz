"""
Basis species management module.

This module provides Python equivalents of the R functions in basis.R:
- basis(): Set up and manage basis species for thermodynamic calculations
- Basis species validation and stoichiometric matrix construction
- Buffer system support and preset basis definitions

Author: CHNOSZ Python port
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple
import warnings

from .thermo import thermo
from .info import info, find_species
from ..utils.formula import makeup


class BasisError(Exception):
    """Exception raised for basis-related errors."""
    pass


def basis(species: Optional[Union[str, int, List[Union[str, int]]]] = None,
          state: Optional[Union[str, List[str]]] = None,
          logact: Optional[Union[float, List[float]]] = None,
          delete: bool = False,
          add: bool = False,
          messages: bool = True,
          global_state: bool = True) -> Optional[pd.DataFrame]:
    """
    Set up the basis species of a thermodynamic system.

    Parameters
    ----------
    species : str, int, list, or None
        Species name(s), formula(s), or index(es), or preset keyword.
        If None, returns current basis definition.
    state : str, list of str, or None
        Physical state(s) for the species
    logact : float, list of float, or None
        Log activities for the basis species
    delete : bool, default False
        If True, delete the basis definition
    add : bool, default False
        If True, add to existing basis instead of replacing
    messages : bool, default True
        If True, print informational messages about species lookup
        If False, suppress all output (equivalent to R's suppressMessages())
    global_state : bool, default True
        If True, store basis definition in global thermo().basis (default behavior)
        If False, return basis definition without storing globally (local state)

    Returns
    -------
    pd.DataFrame or None
        Basis species definition DataFrame, or None if deleted

    Examples
    --------
    >>> # Set up a simple basis
    >>> basis(["H2O", "CO2", "NH3"], logact=[0, -3, -4])

    >>> # Use a preset basis
    >>> basis("CHNOS")

    >>> # Add species to existing basis
    >>> basis("Fe2O3", add=True)

    >>> # Delete basis
    >>> basis(delete=True)

    >>> # Suppress messages
    >>> basis("CHNOS", messages=False)
    """
    thermo_obj = thermo()
    
    # Get current basis
    old_basis = thermo_obj.basis
    
    # Delete basis if requested
    if delete or species == "":
        thermo_obj.basis = None
        thermo_obj.species = None
        return old_basis
    
    # Return current basis if no species specified
    if species is None:
        return old_basis
    
    # Handle empty species list
    if isinstance(species, list) and len(species) == 0:
        raise ValueError("species argument is empty")
    
    # Check for preset keywords
    if isinstance(species, str) and species in _get_preset_basis_keywords():
        return preset_basis(species, messages=messages, global_state=global_state)

    # Ensure species names are unique
    if isinstance(species, list):
        if len(set([str(s) for s in species])) != len(species):
            raise ValueError("species names are not unique")

    # Process arguments
    species, state, logact = _process_basis_arguments(species, state, logact)

    # Handle special transformations
    species, logact = _handle_special_species(species, logact)

    # Check if we're modifying existing basis species
    if (old_basis is not None and not add and
        _all_species_in_basis(species, old_basis)):
        if state is not None or logact is not None:
            return mod_basis(species, state, logact, messages=messages)

    # Create new basis definition or add to existing
    if logact is None:
        logact = [0.0] * len(species)

    # Get species indices
    ispecies = _get_species_indices(species, state, messages=messages)
    
    # Handle adding to existing basis
    if add and old_basis is not None:
        # Check for duplicates
        existing_indices = old_basis['ispecies'].tolist()
        for i, idx in enumerate(ispecies):
            if idx in existing_indices:
                sp_name = species[i] if isinstance(species[i], str) else str(species[i])
                raise BasisError(f"Species {sp_name} is already in the basis definition")
        
        # Append to existing basis
        ispecies = existing_indices + ispecies
        logact = old_basis['logact'].tolist() + logact
    
    # Create new basis
    new_basis = put_basis(ispecies, logact, global_state=global_state)

    # Only update global species list if using global state
    if global_state:
        # Handle species list when adding
        if add and thermo_obj.species is not None:
            _update_species_for_added_basis(old_basis, new_basis)
        else:
            # Clear species since basis changed
            from .species import species as species_func
            species_func(delete=True)

    return new_basis


def _process_basis_arguments(species, state, logact):
    """Process and validate basis function arguments."""
    # Convert single values to lists
    if not isinstance(species, list):
        species = [species]
    
    # Handle argument swapping for compatibility with R version
    # If logact looks like states (strings), swap them
    if logact is not None:
        if isinstance(logact, list) and len(logact) > 0 and isinstance(logact[0], str):
            state, logact = logact, state
        elif isinstance(logact, str):
            state, logact = logact, state
    # If state is numeric, treat it as logact (like R CHNOSZ)
    elif state is not None:
        if isinstance(state, (int, float)):
            state, logact = None, state
        elif isinstance(state, list) and len(state) > 0 and isinstance(state[0], (int, float)):
            state, logact = None, state
    
    # Ensure consistent lengths
    n_species = len(species)
    if state is not None:
        if isinstance(state, str):
            state = [state] * n_species
        else:
            state = list(state)[:n_species]  # Truncate if too long
            state.extend([state[-1]] * (n_species - len(state)))  # Extend if too short
    
    if logact is not None:
        if isinstance(logact, (int, float)):
            logact = [float(logact)] * n_species
        else:
            logact = list(logact)[:n_species]
            logact.extend([0.0] * (n_species - len(logact)))
    
    return species, state, logact


def _handle_special_species(species, logact):
    """Handle special species transformations (pH, pe, Eh)."""
    new_species = []
    new_logact = logact.copy() if logact else [0.0] * len(species)
    
    for i, sp in enumerate(species):
        if sp == "pH":
            new_logact[i] = -new_logact[i]
            new_species.append("H+")
        elif sp == "pe":
            new_logact[i] = -new_logact[i]
            new_species.append("e-")
        elif sp == "Eh":
            # Convert Eh to pe (simplified - assumes 25°C)
            new_logact[i] = -_convert_eh_to_pe(new_logact[i])
            new_species.append("e-")
        else:
            new_species.append(sp)
    
    return new_species, new_logact


def _convert_eh_to_pe(eh_value):
    """Convert Eh to pe (simplified for 25°C)."""
    # This is a simplified conversion - full implementation would
    # use proper temperature-dependent conversion
    return eh_value / 0.05916  # Approximate conversion at 25°C


def _all_species_in_basis(species, basis_df):
    """Check if all species are already in the basis definition."""
    if basis_df is None:
        return False
    
    basis_formulas = basis_df.index.tolist()
    basis_indices = basis_df['ispecies'].tolist()
    
    for sp in species:
        if isinstance(sp, str):
            if sp not in basis_formulas:
                return False
        elif isinstance(sp, int):
            if sp not in basis_indices:
                return False
    
    return True


def _get_species_indices(species, state, messages=True):
    """Get species indices for basis species."""
    ispecies = []

    for i, sp in enumerate(species):
        if isinstance(sp, int):
            # Already an index
            ispecies.append(sp)
        else:
            # Look up by name/formula
            sp_state = state[i] if state and i < len(state) else None
            try:
                idx = find_species(sp, sp_state, messages=messages)
                ispecies.append(idx)
            except ValueError:
                available = f"({sp_state})" if sp_state else ""
                raise BasisError(f"Species not available: {sp}{available}")

    return ispecies


def put_basis(ispecies: List[int], logact: List[float], global_state: bool = True) -> pd.DataFrame:
    """
    Create and validate a basis species definition.
    
    Parameters
    ----------
    ispecies : list of int
        Species indices in thermo().obigt
    logact : list of float
        Log activities for the basis species
    global_state : bool, default True
        If True, store in global thermo().basis (default)
        If False, return without storing globally

    Returns
    -------
    pd.DataFrame
        Validated basis definition
        
    Raises
    ------
    BasisError
        If the basis is invalid (non-square or singular matrix)
    """
    thermo_obj = thermo()
    obigt = thermo_obj.obigt
    
    if obigt is None:
        raise RuntimeError("Thermodynamic database not initialized")
    
    # Get species information
    states = [obigt.iloc[i-1]['state'] for i in ispecies]
    formulas = [obigt.iloc[i-1]['formula'] for i in ispecies]
    
    # Create stoichiometric matrix
    comp_matrix = _make_composition_matrix(ispecies, formulas)
    
    # Validate matrix
    n_species, n_elements = comp_matrix.shape
    if n_species > n_elements:
        if 'Z' in comp_matrix.columns:
            raise BasisError("the number of basis species is greater than the number of elements and charge")
        else:
            raise BasisError("the number of basis species is greater than the number of elements")
    elif n_species < n_elements:
        if 'Z' in comp_matrix.columns:
            raise BasisError("the number of basis species is less than the number of elements and charge")
        else:
            raise BasisError("the number of basis species is less than the number of elements")
    
    # Check if matrix is invertible
    try:
        np.linalg.inv(comp_matrix.values)
    except np.linalg.LinAlgError:
        raise BasisError("singular stoichiometric matrix")
    
    # Create basis DataFrame
    basis_data = comp_matrix.copy()
    basis_data['ispecies'] = ispecies
    basis_data['logact'] = logact
    basis_data['state'] = states
    
    # Set row names to formulas, handling electron specially
    rownames = []
    for formula in formulas:
        if formula == "(Z-1)":
            rownames.append("e-")
        else:
            rownames.append(formula)
    
    basis_data.index = rownames

    # Store in thermo system only if using global state
    if global_state:
        thermo_obj.basis = basis_data

    return basis_data


def _make_composition_matrix(ispecies: List[int], formulas: List[str]) -> pd.DataFrame:
    """
    Create elemental composition matrix for basis species.
    
    Parameters
    ----------
    ispecies : list of int
        Species indices
    formulas : list of str
        Chemical formulas
        
    Returns
    -------
    pd.DataFrame
        Composition matrix with elements as columns
    """
    # Get elemental makeup for each species
    compositions = []
    all_elements = set()
    
    for formula in formulas:
        comp = makeup(formula)
        compositions.append(comp)
        all_elements.update(comp.keys())
    
    # Create matrix with all elements
    all_elements = sorted(list(all_elements))
    comp_matrix = pd.DataFrame(index=range(len(formulas)), columns=all_elements)
    
    for i, comp in enumerate(compositions):
        for element in all_elements:
            comp_matrix.loc[i, element] = comp.get(element, 0)
    
    return comp_matrix.astype(float)


def mod_basis(species: Union[str, int, List[Union[str, int]]],
              state: Optional[Union[str, List[str]]] = None,
              logact: Optional[Union[float, List[float]]] = None,
              messages: bool = True) -> pd.DataFrame:
    """
    Modify states or log activities of existing basis species.

    Parameters
    ----------
    species : str, int, or list
        Basis species to modify (by formula or index)
    state : str, list of str, or None
        New state(s) or buffer name(s)
    logact : float, list of float, or None
        New log activity values
    messages : bool, default True
        If True, print informational messages

    Returns
    -------
    pd.DataFrame
        Updated basis definition

    Raises
    ------
    BasisError
        If basis not defined or species not found
    """
    thermo_obj = thermo()
    
    if thermo_obj.basis is None:
        raise BasisError("basis is not defined")
    
    # Ensure arguments are lists
    if not isinstance(species, list):
        species = [species]
    if state is not None and not isinstance(state, list):
        state = [state]
    if logact is not None and not isinstance(logact, list):
        logact = [logact]
    
    # Process each species
    for i, sp in enumerate(species):
        # Find basis species index
        if isinstance(sp, int):
            # Match by species index
            try:
                basis_idx = thermo_obj.basis[thermo_obj.basis['ispecies'] == sp].index[0]
            except IndexError:
                raise BasisError(f"{sp} is not a species index of one of the basis species")
        else:
            # Match by formula
            try:
                basis_idx = thermo_obj.basis.loc[sp].name
                basis_idx = sp  # Use the formula as index
            except KeyError:
                raise BasisError(f"{sp} is not a formula of one of the basis species")
        
        # Modify state
        if state is not None and i < len(state):
            new_state = state[i]
            
            # Check if it's a buffer name
            if _is_buffer_name(new_state):
                _validate_buffer_compatibility(new_state, basis_idx, messages=messages)
                thermo_obj.basis.loc[basis_idx, 'logact'] = new_state
            else:
                # Find species in new state
                current_species = thermo_obj.basis.loc[basis_idx, 'ispecies']
                species_name = thermo_obj.obigt.iloc[current_species-1]['name']
                species_formula = thermo_obj.obigt.iloc[current_species-1]['formula']
                
                # Try to find by name first, then by formula
                try:
                    new_ispecies = find_species(species_name, new_state, messages=messages)
                except ValueError:
                    try:
                        new_ispecies = find_species(species_formula, new_state, messages=messages)
                    except ValueError:
                        name_text = species_name if species_name == species_formula else f"{species_name} or {species_formula}"
                        raise BasisError(f"state or buffer '{new_state}' not found for {name_text}")
                
                # Update basis
                thermo_obj.basis.loc[basis_idx, 'ispecies'] = new_ispecies
                thermo_obj.basis.loc[basis_idx, 'state'] = new_state
        
        # Modify log activity
        if logact is not None and i < len(logact):
            thermo_obj.basis.loc[basis_idx, 'logact'] = logact[i]
    
    return thermo_obj.basis


def _is_buffer_name(name: str) -> bool:
    """Check if name corresponds to a buffer system."""
    thermo_obj = thermo()
    if thermo_obj.buffer is None:
        return False
    
    return name in thermo_obj.buffer['name'].values if 'name' in thermo_obj.buffer.columns else False


def _validate_buffer_compatibility(buffer_name: str, basis_idx: str, messages: bool = True) -> None:
    """Validate that buffer species are compatible with current basis."""
    thermo_obj = thermo()

    # Get buffer species
    buffer_data = thermo_obj.buffer[thermo_obj.buffer['name'] == buffer_name]

    for _, buffer_row in buffer_data.iterrows():
        species_name = buffer_row.get('species', '')
        species_state = buffer_row.get('state', '')

        try:
            ispecies = find_species(species_name, species_state, messages=messages)
            species_makeup = makeup(thermo_obj.obigt.iloc[ispecies-1]['formula'])

            # Check if all elements are in basis
            basis_elements = set(thermo_obj.basis.columns) - {'ispecies', 'logact', 'state'}
            species_elements = set(species_makeup.keys())

            missing_elements = species_elements - basis_elements
            if missing_elements:
                raise BasisError(f"the elements '{', '.join(missing_elements)}' of species "
                               f"'{species_name}' in buffer '{buffer_name}' are not in the basis")
        except ValueError:
            pass  # Skip if species not found


def _update_species_for_added_basis(old_basis: pd.DataFrame, new_basis: pd.DataFrame) -> None:
    """Update species list when basis species are added."""
    thermo_obj = thermo()
    
    if thermo_obj.species is None:
        return
    
    n_old = len(old_basis)
    n_new = len(new_basis)
    n_species = len(thermo_obj.species)
    
    # Create new stoichiometric matrix with zeros for added basis species
    old_stoich = thermo_obj.species.iloc[:, :n_old].values
    new_cols = np.zeros((n_species, n_new - n_old))
    new_stoich = np.hstack([old_stoich, new_cols])
    
    # Create new species DataFrame
    stoich_df = pd.DataFrame(new_stoich, columns=new_basis.index)
    other_cols = thermo_obj.species.iloc[:, n_old:]
    new_species = pd.concat([stoich_df, other_cols], axis=1)
    
    thermo_obj.species = new_species


def preset_basis(key: Optional[str] = None, messages: bool = True, global_state: bool = True) -> Union[List[str], pd.DataFrame]:
    """
    Load a preset basis definition by keyword.

    Parameters
    ----------
    key : str or None
        Preset keyword. If None, returns available keywords.
    messages : bool, default True
        If True, print informational messages
    global_state : bool, default True
        If True, store in global thermo().basis (default)
        If False, return without storing globally

    Returns
    -------
    list of str or pd.DataFrame
        Available keywords or basis definition

    Examples
    --------
    >>> # List available presets
    >>> preset_basis()

    >>> # Load CHNOS basis
    >>> preset_basis("CHNOS")
    """
    keywords = _get_preset_basis_keywords()
    
    if key is None:
        return keywords
    
    if key not in keywords:
        raise ValueError(f"{key} is not a keyword for preset basis species")

    # Clear existing basis only if using global state
    if global_state:
        basis(delete=True)
    
    # Define preset species
    species_map = {
        "CHNOS": ["CO2", "H2O", "NH3", "H2S", "oxygen"],
        "CHNOS+": ["CO2", "H2O", "NH3", "H2S", "oxygen", "H+"],
        "CHNOSe": ["CO2", "H2O", "NH3", "H2S", "e-", "H+"],
        "CHNOPS+": ["CO2", "H2O", "NH3", "H3PO4", "H2S", "oxygen", "H+"],
        "CHNOPSe": ["CO2", "H2O", "NH3", "H3PO4", "H2S", "e-", "H+"],
        "MgCHNOPS+": ["Mg+2", "CO2", "H2O", "NH3", "H3PO4", "H2S", "oxygen", "H+"],
        "MgCHNOPSe": ["Mg+2", "CO2", "H2O", "NH3", "H3PO4", "H2S", "e-", "H+"],
        "FeCHNOS": ["Fe2O3", "CO2", "H2O", "NH3", "H2S", "oxygen"],
        "FeCHNOS+": ["Fe2O3", "CO2", "H2O", "NH3", "H2S", "oxygen", "H+"],
        "QEC4": ["glutamine", "glutamic acid", "cysteine", "H2O", "oxygen"],
        "QEC": ["glutamine", "glutamic acid", "cysteine", "H2O", "oxygen"],
        "QEC+": ["glutamine", "glutamic acid", "cysteine", "H2O", "oxygen", "H+"],
        "QCa": ["glutamine", "cysteine", "acetic acid", "H2O", "oxygen"],
        "QCa+": ["glutamine", "cysteine", "acetic acid", "H2O", "oxygen", "H+"]
    }
    
    species = species_map[key]
    logact = _preset_logact(species)

    # Special case for QEC4
    if key == "QEC4":
        logact[:3] = [-4.0] * 3

    return basis(species, logact=logact, messages=messages, global_state=global_state)


def _get_preset_basis_keywords() -> List[str]:
    """Get list of available preset basis keywords."""
    return [
        "CHNOS", "CHNOS+", "CHNOSe", "CHNOPS+", "CHNOPSe",
        "MgCHNOPS+", "MgCHNOPSe", "FeCHNOS", "FeCHNOS+",
        "QEC4", "QEC", "QEC+", "QCa", "QCa+"
    ]


def _preset_logact(species: List[str]) -> List[float]:
    """Get preset log activities for basis species."""
    # Standard log activities for common species
    standard_logact = {
        "H2O": 0.0,
        "CO2": -3.0,
        "NH3": -4.0,
        "H2S": -7.0,
        "oxygen": -80.0,
        "H+": -7.0,
        "e-": -7.0,
        "Fe2O3": 0.0,
        # QEC amino acids (from Dick, 2017)
        "glutamine": -3.2,
        "glutamic acid": -4.5,
        "cysteine": -3.6
    }
    
    logact = []
    for sp in species:
        if sp in standard_logact:
            logact.append(standard_logact[sp])
        else:
            logact.append(-3.0)  # Default for unmatched species
    
    return logact


# Convenience functions
def get_basis() -> Optional[pd.DataFrame]:
    """
    Get current basis definition.
    
    Returns
    -------
    pd.DataFrame or None
        Current basis definition
    """
    return thermo().basis


def is_basis_defined() -> bool:
    """
    Check if basis is currently defined.
    
    Returns
    -------
    bool
        True if basis is defined
    """
    return thermo().basis is not None


def basis_elements() -> Optional[np.ndarray]:
    """
    Get basis elements matrix.
    
    Returns
    -------
    np.ndarray or None
        Transposed basis composition matrix
    """
    basis_df = get_basis()
    if basis_df is None:
        return None
    
    # Get elemental composition columns
    element_cols = [col for col in basis_df.columns 
                   if col not in ['ispecies', 'logact', 'state']]
    
    return basis_df[element_cols].values.T


def swap_basis(species1: Union[str, int], species2: Union[str, int]) -> pd.DataFrame:
    """
    Swap one basis species for another.
    
    Parameters
    ----------
    species1 : str or int
        Current basis species to replace
    species2 : str or int
        New species to add to basis
        
    Returns
    -------
    pd.DataFrame
        Updated basis definition
        
    Raises
    ------
    BasisError
        If operation is not possible
    """
    thermo_obj = thermo()
    
    if thermo_obj.basis is None:
        raise BasisError("basis is not defined")
    
    # This would require solving for the new basis coefficients
    # Full implementation would be more complex
    raise NotImplementedError("swap_basis not yet implemented")


# Export main functions
__all__ = [
    'basis', 'mod_basis', 'put_basis', 'preset_basis',
    'get_basis', 'is_basis_defined', 'basis_elements',
    'BasisError'
]