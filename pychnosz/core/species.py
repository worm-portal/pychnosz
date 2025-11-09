"""
Formed species management module.

This module provides Python equivalents of the R functions in species.R:
- species(): Define and manage species of interest for thermodynamic calculations
- Formation reaction calculations from basis species
- Species list management and validation

Author: CHNOSZ Python port
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Any, Tuple
import warnings

from .thermo import thermo
from .info import info, find_species
from .basis import get_basis, is_basis_defined
from ..utils.formula import makeup, species_basis


class SpeciesError(Exception):
    """Exception raised for species-related errors."""
    pass


def species(species: Optional[Union[str, int, List[Union[str, int]], pd.Series]] = None,
            state: Optional[Union[str, List[str]]] = None,
            delete: bool = False,
            add: bool = False,
            index_return: bool = False,
            global_state: bool = True,
            basis: Optional[pd.DataFrame] = None,
            messages: bool = True) -> Optional[Union[pd.DataFrame, List[int]]]:
    """
    Define species of interest for thermodynamic calculations.

    Parameters
    ----------
    species : str, int, list, pd.Series, or None
        Species name(s), formula(s), or index(es).
        Can also be a pandas Series (e.g., from retrieve()).
        If None, returns current species definition.
    state : str, list of str, or None
        Physical state(s) for the species
    delete : bool, default False
        If True, delete species (all if species is None)
    add : bool, default False
        If True, add to existing species instead of replacing
    index_return : bool, default False
        If True, return species indices instead of DataFrame
    global_state : bool, default True
        If True, store species in global thermo().species (default behavior)
        If False, return species definition without storing globally (local state)
    basis : pd.DataFrame, optional
        Basis species definition to use (if not using global basis)
        Required when global_state=False and basis is not defined globally
    messages : bool, default True
        If True, print informational messages

    Returns
    -------
    pd.DataFrame, list of int, or None
        Species definition DataFrame or indices, or None if deleted

    Examples
    --------
    >>> # Define species of interest
    >>> species(["CO2", "HCO3-", "CO3-2"])

    >>> # Add more species
    >>> species(["CH4", "C2H4"], add=True)

    >>> # Delete specific species
    >>> species(["CO2"], delete=True)

    >>> # Delete all species
    >>> species(delete=True)

    >>> # Use output from retrieve()
    >>> zn_species = retrieve("Zn", ["O", "H"], state="aq")
    >>> species(zn_species)
    """
    thermo_obj = thermo()

    # Handle pandas Series (e.g., from retrieve())
    if isinstance(species, pd.Series):
        # Extract the integer indices from the Series values
        species = species.values.tolist()

    # Handle NA species
    if species is pd.NA or species is np.nan:
        raise SpeciesError("'species' is NA")
    
    # Handle deletion
    if delete:
        return _delete_species(species, thermo_obj)
    
    # Return current species if no arguments
    if species is None and state is None:
        if index_return:
            if thermo_obj.species is not None:
                return list(range(1, len(thermo_obj.species) + 1))
            else:
                return []
        return thermo_obj.species
    
    # Use all species indices if species is None but state is given
    if species is None and thermo_obj.species is not None:
        species = list(range(1, len(thermo_obj.species) + 1))
    
    # Process state argument
    state = _process_state_argument(state)
    
    # Make species and state same length
    species, state = _match_argument_lengths(species, state)
    
    # Handle numeric state (treat as logact)
    logact = None
    if state is not None and len(state) > 0:
        if isinstance(state[0], (int, float)):
            logact = [float(s) for s in state]
            state = None
        elif _can_be_numeric(state[0]):
            logact = [float(s) for s in state]
            state = None
    
    # Handle species-state combinations for proteins
    if state is not None:
        species, state = _handle_protein_naming(species, state, thermo_obj)
    
    # Process species argument
    iOBIGT = None
    if isinstance(species[0], str):
        # Check if species are in current definition
        if thermo_obj.species is not None:
            existing_indices = _match_existing_species(species, thermo_obj.species)
            if all(idx is not None for idx in existing_indices) and logact is not None:
                # Update activities of existing species
                # Update activities of existing species directly
                species_indices = [i+1 for i in existing_indices]  # Convert to 1-based
                return _update_existing_species(species_indices, None, logact, index_return, thermo_obj)
        
        # Look up species in database
        iOBIGT = _lookup_species_indices(species, state, messages)
        
    else:
        # Handle numeric species
        if thermo_obj.species is not None:
            max_current = len(thermo_obj.species)
            if all(isinstance(s, int) and s <= max_current for s in species):
                # Referring to existing species
                return _update_existing_species(species, state, logact, index_return, thermo_obj)
        
        # Referring to OBIGT indices
        iOBIGT = species
    
    # Create or modify species definition
    if iOBIGT is not None:
        return _create_species_definition(iOBIGT, state, logact, add, index_return, thermo_obj, global_state, basis)
    else:
        return _update_existing_species(species, state, logact, index_return, thermo_obj)


def _delete_species(species: Optional[Union[str, int, List]], thermo_obj) -> Optional[pd.DataFrame]:
    """Delete species from the current definition."""
    if species is None:
        # Delete all species
        thermo_obj.species = None
        return None
    
    if thermo_obj.species is None:
        raise SpeciesError("nonexistent species definition")
    
    # Ensure species is a list
    if not isinstance(species, list):
        species = [species]
    
    # Find species to delete
    indices_to_delete = []
    for sp in species:
        if isinstance(sp, str):
            # Match by name
            matches = thermo_obj.species[thermo_obj.species['name'] == sp].index.tolist()
        elif isinstance(sp, int):
            # Match by row number (1-based)
            if 1 <= sp <= len(thermo_obj.species):
                matches = [sp - 1]  # Convert to 0-based
            else:
                matches = []
        else:
            matches = []
        
        if matches:
            indices_to_delete.extend(matches)
        else:
            warnings.warn(f"species: {sp} not present, so cannot be deleted")
    
    # Remove duplicates and sort
    indices_to_delete = sorted(set(indices_to_delete))
    
    if indices_to_delete:
        # Delete species
        thermo_obj.species = thermo_obj.species.drop(indices_to_delete).reset_index(drop=True)
        
        if len(thermo_obj.species) == 0:
            thermo_obj.species = None
    
    return thermo_obj.species


def _process_state_argument(state) -> Optional[List]:
    """Process state argument into consistent format."""
    if state is None:
        return None
    
    if isinstance(state, str):
        return [state]
    elif isinstance(state, (list, tuple)):
        return list(state)
    else:
        return [state]


def _match_argument_lengths(species, state) -> Tuple[List, Optional[List]]:
    """Ensure species and state arguments have compatible lengths."""
    if not isinstance(species, list):
        species = [species]
    
    if state is not None:
        if len(species) > len(state):
            # Extend state to match species length
            state = state * ((len(species) // len(state)) + 1)
            state = state[:len(species)]
        elif len(state) > len(species):
            # Extend species to match state length
            species = species * ((len(state) // len(species)) + 1)
            species = species[:len(state)]
    
    return species, state


def _can_be_numeric(value) -> bool:
    """Check if value can be converted to numeric."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def _handle_protein_naming(species: List, state: List, thermo_obj) -> Tuple[List, List]:
    """Handle protein naming convention (species_organism)."""
    if state is None:
        return species, state
    
    # Get all available states
    all_states = thermo_obj.obigt['state'].unique().tolist()
    
    # Check if states look like actual states or should be treated as suffixes
    if not all(s in all_states or _can_be_numeric(s) for s in state):
        # Treat as protein organism suffixes
        new_species = []
        for sp, st in zip(species, state):
            if '_' not in str(sp) and not _can_be_numeric(st):
                new_species.append(f"{sp}_{st}")
            else:
                new_species.append(sp)
        
        # Use default state for proteins
        default_state = thermo_obj.get_option('state', 'aq')
        state = [default_state] * len(species)
        species = new_species
    
    return species, state


def _match_existing_species(species: List[str], species_df: pd.DataFrame) -> List[Optional[int]]:
    """Match species names to existing species definition."""
    indices = []
    for sp in species:
        matches = species_df[species_df['name'] == sp].index.tolist()
        indices.append(matches[0] if matches else None)
    return indices


def _lookup_species_indices(species: List[str], state: Optional[List[str]], messages: bool = True) -> List[int]:
    """Look up species indices in the OBIGT database."""
    iOBIGT = []
    
    for i, sp in enumerate(species):
        sp_state = state[i] if state and i < len(state) else None
        
        try:
            # Use info function to find species
            idx = info(sp, sp_state, messages=messages)
            if pd.isna(idx):
                raise SpeciesError(f"species not available: {sp}")
            iOBIGT.append(idx)
        except Exception:
            raise SpeciesError(f"species not available: {sp}")
    
    return iOBIGT


def _update_existing_species(species: List[int], state, logact, index_return: bool, 
                           thermo_obj) -> Union[pd.DataFrame, List[int]]:
    """Update activities or states of existing species."""
    if thermo_obj.species is None:
        raise SpeciesError("no species definition exists")
    
    # Validate species indices
    max_species = len(thermo_obj.species)
    species_indices = []
    for sp in species:
        if isinstance(sp, int) and 1 <= sp <= max_species:
            species_indices.append(sp - 1)  # Convert to 0-based
        else:
            raise SpeciesError(f"invalid species index: {sp}")
    
    # Return without changes if no updates requested
    if state is None and logact is None:
        if index_return:
            return [i + 1 for i in species_indices]  # Convert back to 1-based
        else:
            return thermo_obj.species.iloc[species_indices]
    
    # Update log activities
    if logact is not None:
        for i, idx in enumerate(species_indices):
            if i < len(logact):
                thermo_obj.species.loc[idx, 'logact'] = logact[i]
    
    # Update states
    if state is not None:
        _update_species_states(species_indices, state, thermo_obj)
    
    if index_return:
        return [i + 1 for i in species_indices]
    else:
        # Return full species definition like R CHNOSZ
        return thermo_obj.species


def _update_species_states(species_indices: List[int], states: List[str], thermo_obj) -> None:
    """Update states of existing species."""
    for i, idx in enumerate(species_indices):
        if i >= len(states):
            break
            
        new_state = states[i]
        current_row = thermo_obj.species.iloc[idx]
        species_name = current_row['name']
        current_formula = thermo_obj.obigt.iloc[current_row['ispecies']-1]['formula']
        
        # Find species in new state
        try:
            # First try by name
            if '_' in species_name:  # Protein
                new_ispecies = find_species(species_name, new_state)
            else:
                # Try name first, then formula
                try:
                    new_ispecies = find_species(species_name, new_state)
                except ValueError:
                    new_ispecies = find_species(current_formula, new_state)
            
            # Update species data
            thermo_obj.species.loc[idx, 'ispecies'] = new_ispecies
            thermo_obj.species.loc[idx, 'state'] = new_state
            thermo_obj.species.loc[idx, 'name'] = thermo_obj.obigt.iloc[new_ispecies-1]['name']
            
        except ValueError:
            warnings.warn(f"can't update state of species {idx+1} to {new_state}", 
                         category=UserWarning)


def _create_species_definition(iOBIGT: List[int], state, logact, add: bool,
                             index_return: bool, thermo_obj, global_state: bool = True,
                             basis_df: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, List[int]]:
    """Create new species definition from OBIGT indices."""
    # Use provided basis or get from global state
    if basis_df is None:
        if not is_basis_defined():
            raise SpeciesError("basis species are not defined")
        basis_df = get_basis()

    # Calculate formation reactions with the provided basis
    formation_coeffs = species_basis(iOBIGT, basis_df=basis_df)

    # Get species information
    species_states = []
    species_names = []

    for idx in iOBIGT:
        obigt_row = thermo_obj.obigt.iloc[idx - 1]  # Convert to 0-based
        species_states.append(obigt_row['state'])
        species_names.append(obigt_row['name'])

    # Set default log activities
    if logact is None:
        logact = []
        for state_val in species_states:
            if state_val == 'aq':
                logact.append(-3.0)
            else:
                logact.append(0.0)

    # Create new species DataFrame
    basis_formulas = basis_df.index.tolist()
    
    # Build stoichiometric part
    stoich_data = {}
    for i, formula in enumerate(basis_formulas):
        stoich_data[formula] = formation_coeffs[:, i]
    
    # Add other columns
    new_data = pd.DataFrame(stoich_data)
    new_data['ispecies'] = iOBIGT
    new_data['logact'] = logact
    new_data['state'] = species_states
    new_data['name'] = species_names

    # Handle adding vs replacing
    if global_state:
        # Use global state
        if thermo_obj.species is None or not add:
            # Create new or replace existing
            thermo_obj.species = new_data
            species_indices = list(range(len(new_data)))
        else:
            # Add to existing - check for duplicates
            existing_indices = set(thermo_obj.species['ispecies'].tolist())
            new_indices = []
            rows_to_add = []

            for i, idx in enumerate(iOBIGT):
                if idx not in existing_indices:
                    new_indices.append(len(thermo_obj.species) + len(rows_to_add))
                    rows_to_add.append(new_data.iloc[i])

            if rows_to_add:
                # Add new rows
                new_rows_df = pd.DataFrame(rows_to_add)
                thermo_obj.species = pd.concat([thermo_obj.species, new_rows_df],
                                             ignore_index=True)

            # Find all species indices (including existing ones)
            species_indices = []
            for idx in iOBIGT:
                match_idx = thermo_obj.species[thermo_obj.species['ispecies'] == idx].index[0]
                species_indices.append(match_idx)

        # Reset index to ensure continuous numbering
        if thermo_obj.species is not None:
            thermo_obj.species.reset_index(drop=True, inplace=True)

        # Return results
        if index_return:
            return [i + 1 for i in species_indices]  # Convert to 1-based
        else:
            return thermo_obj.species
    else:
        # Local state - just return the dataframe
        if index_return:
            return list(range(1, len(new_data) + 1))
        else:
            return new_data


# Convenience functions
def get_species() -> Optional[pd.DataFrame]:
    """
    Get current species definition.
    
    Returns
    -------
    pd.DataFrame or None
        Current species definition
    """
    return thermo().species


def is_species_defined() -> bool:
    """
    Check if species are currently defined.
    
    Returns
    -------
    bool
        True if species are defined
    """
    return thermo().species is not None


def n_species() -> int:
    """
    Get number of defined species.
    
    Returns
    -------
    int
        Number of defined species
    """
    species_df = get_species()
    return len(species_df) if species_df is not None else 0


def species_names() -> List[str]:
    """
    Get names of defined species.
    
    Returns
    -------
    list of str
        Species names
    """
    species_df = get_species()
    if species_df is not None:
        return species_df['name'].tolist()
    else:
        return []


def species_formulas() -> List[str]:
    """
    Get formulas of defined species.
    
    Returns
    -------
    list of str
        Species formulas
    """
    thermo_obj = thermo()
    species_df = get_species()
    
    if species_df is not None and thermo_obj.obigt is not None:
        formulas = []
        for idx in species_df['ispecies']:
            formula = thermo_obj.obigt.iloc[idx - 1]['formula']
            formulas.append(formula)
        return formulas
    else:
        return []


def species_states() -> List[str]:
    """
    Get states of defined species.
    
    Returns
    -------
    list of str
        Species states
    """
    species_df = get_species()
    if species_df is not None:
        return species_df['state'].tolist()
    else:
        return []


def find_species_index(name: str) -> int:
    """
    Find index of species in current definition.
    
    Parameters
    ----------
    name : str
        Species name to find
        
    Returns
    -------
    int
        Species index (1-based) in current definition
        
    Raises
    ------
    SpeciesError
        If species not found
    """
    species_df = get_species()
    if species_df is None:
        raise SpeciesError("no species definition exists")
    
    matches = species_df[species_df['name'] == name].index.tolist()
    if not matches:
        raise SpeciesError(f"species '{name}' not found in current definition")
    
    return matches[0] + 1  # Convert to 1-based


# Export main functions
__all__ = [
    'species', 'get_species', 'is_species_defined', 'n_species',
    'species_names', 'species_formulas', 'species_states',
    'find_species_index', 'SpeciesError'
]