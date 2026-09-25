"""
Species database lookup and information retrieval module.

This module provides Python equivalents of the R functions in info.R:
- info(): Search for species by name, formula, or index
- species information retrieval and validation
- database summarization and query functions

Author: CHNOSZ Python port
"""

import pandas as pd
import numpy as np
from typing import Union, List, Optional, Dict, Any
import warnings
import re
import io
from contextlib import redirect_stdout

from .thermo import thermo
from ..utils.formula import makeup, as_chemical_formula


def info(species: Optional[Union[str, int, List[Union[str, int]], pd.Series]] = None,
         state: Optional[Union[str, List[str]]] = None,
         check_it: bool = True,
         messages: bool = True,
         check_protein: bool = True,
         E_units: Union[bool, str, None] = False) -> Union[pd.DataFrame, int, List[int], None]:
    """
    Search for species in the thermodynamic database.

    Parameters
    ----------
    species : str, int, list of str/int, pd.Series, or None
        Species name, formula, abbreviation, or OBIGT index.
        Can also be a pandas Series (e.g., from retrieve()).
        If None, returns summary information about the database.
    state : str, list of str, or None
        Physical state(s) to match ('aq', 'cr', 'gas', 'liq')
    check_it : bool, default True
        Whether to perform consistency checks on thermodynamic data
    messages : bool, default True
        Whether to print informational messages
    check_protein : bool, default True
        Whether to look up unmatched names in thermo().protein and add their
        group-additivity properties to OBIGT (e.g. LYSC_CHICK, H2O_RESIDUE)
    E_units : False, 'cal', or 'J', default False
        Energy units for the returned thermodynamic properties and equation-of-state
        parameters. The default (False) returns each entry in the units given by its
        own E_units column, as R CHNOSZ does. Set to 'cal' or 'J' to convert all
        entries to those units, which makes entries stored in different units
        directly comparable. Only affects the DataFrame returned for numeric
        'species'; V (cm3 mol-1), Z, and the CGL 'lambda' exponent are never
        converted. Consistency checks (check_it) are always done in the units of
        the database entry, so any messages they print are unaffected.

    Returns
    -------
    pd.DataFrame, int, list of int, or None
        - If species is None: prints database summary, returns None
        - If species is numeric: returns DataFrame with species data
        - If species is string: returns species index(es) or NA if not found

    Examples
    --------
    >>> # Get database summary
    >>> info()

    >>> # Find species index
    >>> info("H2O")

    >>> # Get species data by index
    >>> info(1)

    >>> # Search with specific state
    >>> info("CO2", "aq")

    >>> # Use output from retrieve()
    >>> zn_species = retrieve("Zn", ["O", "H"], state="aq")
    >>> info(zn_species)

    >>> # Compare two entries that are stored in different energy units
    >>> info(info(["diketopiperazine", "resorcinol"]), E_units="J")
    """
    thermo_obj = thermo()

    # Validate the requested energy units before doing any work
    E_units = _normalize_E_units(E_units)

    # Initialize database if needed
    if not thermo_obj.is_initialized():
        thermo_obj.reset()

    # Return database summary if no species specified
    if species is None:
        return _print_database_summary(thermo_obj, messages)

    # Handle pandas Series (e.g., from retrieve())
    if isinstance(species, pd.Series):
        # Extract the integer indices from the Series values
        indices = species.values.tolist()
        return _info_numeric(indices, thermo_obj, check_it, messages, E_units)

    # Handle numeric species indices
    if isinstance(species, (int, list)) and all(isinstance(s, int) for s in (species if isinstance(species, list) else [species])):
        return _info_numeric(species, thermo_obj, check_it, messages, E_units)

    # Handle string species names/formulas
    if isinstance(species, (str, list)):
        return _info_character(species, state, thermo_obj, messages, check_protein)

    raise ValueError(f"Invalid species type: {type(species)}")


def _print_database_summary(thermo_obj, messages: bool = True) -> None:
    """Print summary information about the thermodynamic database."""
    obigt = thermo_obj.obigt
    if obigt is None:
        if messages:
            print("Database not initialized")
        return

    if not messages:
        return

    # Count species by state
    aq_count = len(obigt[obigt['state'] == 'aq'])
    total_count = len(obigt)

    print(f"info: thermo().obigt has {aq_count} aqueous, {total_count} total species")

    # Count other data
    refs_count = len(thermo_obj.refs) if thermo_obj.refs is not None else 0
    elements_count = len(thermo_obj.element) if thermo_obj.element is not None else 0

    buffer_count = 0
    if thermo_obj.buffer is not None:
        buffer_count = len(thermo_obj.buffer['name'].unique()) if 'name' in thermo_obj.buffer.columns else 0

    print(f"number of literature sources: {refs_count}, elements: {elements_count}, buffers: {buffer_count}")

    protein_count = 0
    organism_count = 0
    if thermo_obj.protein is not None:
        protein_count = len(thermo_obj.protein)
        if 'organism' in thermo_obj.protein.columns:
            organism_count = len(thermo_obj.protein['organism'].unique())

    print(f"number of proteins in thermo().protein is {protein_count} from {organism_count} organisms")


def _info_numeric(species: Union[int, List[int]], thermo_obj, check_it: bool, messages: bool = True,
                  E_units: Optional[str] = None) -> pd.DataFrame:
    """
    Retrieve species information by numeric index.

    Parameters
    ----------
    species : int or list of int
        Species index(es) in thermo().obigt
    thermo_obj : ThermoSystem
        The thermodynamic system object
    check_it : bool
        Whether to perform data consistency checks
    messages : bool, default True
        Whether to print informational messages
    E_units : 'cal', 'J', or None
        Convert the returned properties to these energy units (None = leave each
        entry in the units of its own E_units column)

    Returns
    -------
    pd.DataFrame
        Species thermodynamic data
    """
    obigt = thermo_obj.obigt
    if obigt is None:
        raise RuntimeError("Thermodynamic database not initialized")

    # Ensure species is a list
    if isinstance(species, int):
        species = [species]

    # Validate indices
    max_index = len(obigt)
    for idx in species:
        if idx < 1 or idx > max_index:
            raise IndexError(f"Species index {idx} not found in thermo().obigt (1-{max_index})")

    # Get species data (convert from 1-based to 0-based indexing)
    results = []
    for idx in species:
        species_data = _get_species_data(idx - 1, obigt, check_it, messages, E_units)
        results.append(species_data)

    # Combine results
    result_df = pd.concat(results, ignore_index=True)
    return result_df


def _info_character(species: Union[str, List[str]],
                   state: Optional[Union[str, List[str]]],
                   thermo_obj,
                   messages: bool = True,
                   check_protein: bool = True) -> Union[int, List[int]]:
    """
    Search for species by name, formula, or abbreviation.

    Parameters
    ----------
    species : str or list of str
        Species name(s), formula(s), or abbreviation(s) to search for
    state : str, list of str, or None
        Physical state(s) to match
    thermo_obj : ThermoSystem
        The thermodynamic system object
    check_protein : bool, default True
        Whether to look up unmatched names in thermo().protein and add their
        group-additivity properties to OBIGT. Set False to avoid infinite
        recursion when called from mod_OBIGT() (as in R's info.character()).

    Returns
    -------
    int or list of int
        Species index(es) or NA if not found
    """
    obigt = thermo_obj.obigt
    if obigt is None:
        raise RuntimeError("Thermodynamic database not initialized")
    
    # Ensure species is a list
    if isinstance(species, str):
        species = [species]
        single_result = True
    else:
        single_result = False
    
    # Handle state argument
    if state is not None:
        if isinstance(state, str):
            state = [state] * len(species)
        elif len(state) != len(species):
            # Expand state to match species length
            state = state * ((len(species) // len(state)) + 1)
            state = state[:len(species)]
    
    results = []
    for i, sp in enumerate(species):
        sp_state = state[i] if state is not None else None
        result = _find_species_index(sp, sp_state, obigt, messages)

        # No match in OBIGT: unless it's a protein, which we add to OBIGT
        if pd.isna(result) and check_protein:
            result = _add_protein_to_obigt(sp, sp_state, thermo_obj, messages)
            # OBIGT grew, so use the updated table for any remaining species
            obigt = thermo_obj.obigt

        # Show approximate matches if exact match not found and not a protein
        if pd.isna(result) and '_' not in sp:
            _info_approx(sp, sp_state, obigt, messages)

        results.append(result)
    
    if single_result:
        return results[0]
    else:
        return results


def _add_protein_to_obigt(species: str, state: Optional[str], thermo_obj,
                          messages: bool = True) -> Union[int, float]:
    """
    Look up a name in thermo().protein and add its properties to OBIGT.

    Proteins are named protein_organism (e.g. LYSC_CHICK, or H2O_RESIDUE for
    the residue species used by affinity(iprotein = ...)). They are not stored
    in OBIGT; their thermodynamic properties are calculated on demand by group
    additivity and appended to OBIGT. Mirrors the check.protein branch of R's
    info.character().

    Returns the new OBIGT index, or NaN if the name is not a known protein.
    """
    # Imported here to avoid a circular import at module load
    from ..biomolecules.proteins import pinfo, protein_OBIGT
    from ..data.mod_obigt import mod_OBIGT

    if thermo_obj.protein is None:
        return np.nan

    ip = pinfo(species)
    # pinfo() returns NaN (or an array of them) when the name is not a protein
    if isinstance(ip, (list, np.ndarray)):
        ip = ip[0] if len(ip) > 0 else np.nan
    if ip is None or pd.isna(ip):
        return np.nan

    # Here we use a default state from thermo().opt['state']
    if state is None:
        state = thermo_obj.opt.get('state', 'aq')

    # Add up protein properties, then append them to OBIGT.
    # protein_OBIGT() reports the protein it found (as in R); the messages from
    # mod_OBIGT() are suppressed, matching R's suppressMessages(mod.OBIGT(eos)).
    if messages:
        eos = protein_OBIGT(int(ip), state=state)
    else:
        with redirect_stdout(io.StringIO()):
            eos = protein_OBIGT(int(ip), state=state)
    eos_args = {col: eos[col].tolist() for col in eos.columns}
    with redirect_stdout(io.StringIO()):
        nrows = mod_OBIGT(eos_args)
    if isinstance(nrows, list):
        nrows = nrows[-1]
    return nrows


def _find_species_index(species: str, state: Optional[str], obigt: pd.DataFrame, messages: bool = True) -> Union[int, float]:
    """
    Find exact match for species in the database.
    
    Parameters
    ----------
    species : str
        Species name, formula, or abbreviation
    state : str or None
        Physical state to match
    obigt : pd.DataFrame
        The OBIGT database
        
    Returns
    -------
    int or np.nan
        Species index (1-based) or NaN if not found
    """
    # Find matches for species name, abbreviation, or formula
    matches = (
        (obigt['name'] == species) |
        (obigt['abbrv'] == species) |
        (obigt['formula'] == species)
    )
    
    # Handle NaN values in abbrv column
    matches = matches.fillna(False)
    
    if not matches.any():
        # Check if it's a protein (would be handled elsewhere)
        return np.nan
    
    # Get matching indices
    matching_indices = obigt.index[matches].tolist()
    
    # Filter by state if specified
    if state is not None:
        # Special handling for H2O: 'aq' retrieves 'liq'
        if species in ['H2O', 'water'] and state == 'aq':
            state = 'liq'
        
        state_matches = obigt.loc[matching_indices, 'state'] == state
        matching_indices = [idx for idx, match in zip(matching_indices, state_matches) if match]
        
        if not matching_indices:
            # Requested state not available
            available_states = obigt.loc[matches, 'state'].unique()
            state_text = "', '".join(available_states)
            verb = "is" if len(available_states) == 1 else "are"
            if messages:
                print(f"info_character: requested state '{state}' for {species} "
                      f"but only '{state_text}' {verb} available")
            
            # Special warning for methane
            if species == 'methane' and state == 'aq':
                warnings.warn("'methane' is not an aqueous species; use 'CH4' instead\n"
                             "To revert to the old behavior, run mod_OBIGT(info('CH4'), name='methane')")
            
            return np.nan
    
    if len(matching_indices) == 1:
        # Index is already 1-based (shifted in obigt.py during data loading)
        return matching_indices[0]
    elif len(matching_indices) > 1:
        # Multiple matches - prefer exact name match
        exact_name_matches = obigt.loc[matching_indices, 'name'] == species
        exact_indices = [idx for idx, match in zip(matching_indices, exact_name_matches) if match]

        if len(exact_indices) == 1:
            result_index = exact_indices[0]
        else:
            # Return first match
            result_index = matching_indices[0]

        # Inform user about multiple states
        if messages:
            _report_multiple_matches(species, result_index, matching_indices, obigt, messages=messages)

        # Index is already 1-based (shifted in obigt.py during data loading)
        return result_index
    
    return np.nan


def _report_multiple_matches(species: str, selected_index: int, all_indices: List[int], obigt: pd.DataFrame, messages: bool = True):
    """Report information about multiple matches for a species."""
    selected_state = obigt.loc[selected_index, 'state']
    other_indices = [idx for idx in all_indices if idx != selected_index]
    other_states = obigt.loc[other_indices, 'state'].tolist()
    
    # Handle polymorphic transitions
    trans_states = ['cr2', 'cr3', 'cr4', 'cr5', 'cr6', 'cr7', 'cr8', 'cr9']
    is_trans = [state in trans_states for state in other_states]
    
    trans_text = ""
    if selected_state == 'cr':
        n_trans = sum(is_trans)
        if n_trans == 1:
            trans_text = f" with {n_trans} polymorphic transition"
        elif n_trans > 1:
            trans_text = f" with {n_trans} polymorphic transitions"
    
    # For non-aqueous species, show substance names
    selected_name = obigt.loc[selected_index, 'name']
    name_text = ""
    if selected_state != 'aq' and species != selected_name:
        name_text = f" [{selected_name}]"
    
    # Show other available states
    other_states = [state for state, trans in zip(other_states, is_trans) if not trans]
    if selected_state != 'aq':
        # Replace state with name for isomers in same state
        for i, (idx, state) in enumerate(zip(other_indices, obigt.loc[other_indices, 'state'])):
            if state == selected_state:
                other_states[i] = obigt.loc[idx, 'name']
    
    other_text = ""
    unique_others = list(set(other_states))
    if len(unique_others) == 1:
        other_text = f"; also available in {unique_others[0]}"
    elif len(unique_others) > 1:
        other_text = f"; also available in {', '.join(unique_others)}"
    
    if (trans_text or other_text) and messages:
        start_text = f"info_character: found {species}({selected_state}){name_text}"
        print(f"{start_text}{trans_text}{other_text}")


def _info_approx(species: str, state: Optional[str], obigt: pd.DataFrame, messages: bool = True) -> List[int]:
    """
    Find approximate matches for species name.

    Parameters
    ----------
    species : str
        Species name to search for
    state : str or None
        Physical state to filter by
    obigt : pd.DataFrame
        The OBIGT database

    Returns
    -------
    list of int
        Approximate match indices
    """
    # Simple approximate matching - find species containing the search term
    if state is not None:
        search_data = obigt[obigt['state'] == state]
    else:
        search_data = obigt

    approx_matches = []

    # Look for partial matches in name, abbrv, and formula
    for col in ['name', 'abbrv', 'formula']:
        if col in search_data.columns:
            mask = search_data[col].str.contains(species, case=False, na=False, regex=False)
            matches = search_data.index[mask].tolist()
            approx_matches.extend(matches)

    approx_matches = list(set(approx_matches))  # Remove duplicates

    if not messages:
        return approx_matches

    if approx_matches:
        if len(approx_matches) == 1:
            idx = approx_matches[0]
            species_info = _format_species_info(idx, obigt)
            print(f"info_approx: '{species}' is similar to {species_info}")
        else:
            max_show = 100
            n_show = min(len(approx_matches), max_show)
            ext_text = f" (showing first {max_show})" if len(approx_matches) > max_show else ""
            print(f"info_approx: '{species}' is ambiguous; has approximate matches to "
                  f"{len(approx_matches)} species{ext_text}:")

            # Get unique names (to avoid showing duplicates for polymorphs)
            unique_names = []
            for idx in approx_matches[:n_show]:
                name = obigt.loc[idx, 'name']
                if name not in unique_names:
                    unique_names.append(name)
                    print(f"  {name}")
    else:
        print(f"info_approx: '{species}' has no approximate matches")

    return approx_matches


def _format_species_info(index: int, obigt: pd.DataFrame, with_source: bool = True) -> str:
    """
    Format species information for display.
    
    Parameters
    ----------
    index : int
        Species index in obigt DataFrame
    obigt : pd.DataFrame
        The OBIGT database
    with_source : bool
        Whether to include source information
        
    Returns
    -------
    str
        Formatted species information string
    """
    row = obigt.loc[index]
    name = row['name']
    formula = row['formula']
    state = row['state']
    
    info_text = f"{name} [{formula}({state})]"
    
    if with_source:
        source_parts = []
        if 'ref1' in row and pd.notna(row['ref1']):
            source_parts.append(str(row['ref1']))
        if 'ref2' in row and pd.notna(row['ref2']):
            source_parts.append(str(row['ref2']))
        if 'date' in row and pd.notna(row['date']):
            source_parts.append(str(row['date']))
        
        if source_parts:
            info_text += f" ({', '.join(source_parts)})"
    
    return info_text


def _get_species_data(index: int, obigt: pd.DataFrame, check_it: bool, messages: bool = True,
                      E_units: Optional[str] = None) -> pd.DataFrame:
    """
    Get and validate species thermodynamic data.

    Parameters
    ----------
    index : int
        Species index (0-based) in obigt DataFrame
    obigt : pd.DataFrame
        The OBIGT database
    check_it : bool
        Whether to perform consistency checks
    messages : bool, default True
        Whether to print informational messages
    E_units : 'cal', 'J', or None
        Convert the returned properties to these energy units (None = no conversion)

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with species data (22 columns matching R)
    """
    # Get species data
    species_data = obigt.iloc[index:index+1].copy()

    # Remove scaling factors on EOS parameters (equivalent to OBIGT2eos)
    species_data = _remove_scaling_factors(species_data)

    # Check for missing model
    if pd.isna(species_data.iloc[0]['model']):
        species_name = species_data.iloc[0]['name']
        species_state = species_data.iloc[0]['state']
        raise ValueError(f"Species has NA model: {species_name}({species_state})")

    # Get the model for column selection (preserve case)
    model = str(species_data.iloc[0]['model'])

    # Berman minerals are fully implemented via the Berman() function
    # The cgl() function automatically calls Berman() when model="Berman"
    # No special handling needed here in info()

    # Fill in missing G, H, or S values
    if check_it:
        species_data = _check_and_fill_ghs(species_data, messages)
        species_data = _check_eos_parameters(species_data, messages)

    # Return only the 22 columns that R returns (matching R's info() behavior)
    # R uses different EOS column names depending on the model:
    # - HKF/DEW: a1, a2, a3, a4, c1, c2, omega, Z
    # - CGL and others: a, b, c, d, e, f, lambda, T

    # Base columns (first 14)
    base_columns = ['name', 'abbrv', 'formula', 'state', 'ref1', 'ref2', 'date',
                    'model', 'E_units', 'G', 'H', 'S', 'Cp', 'V']

    # EOS columns depend on model
    if model in ['HKF', 'DEW']:
        # HKF/DEW use: a1, a2, a3, a4, c1, c2, omega, Z
        eos_columns = ['a1', 'a2', 'a3', 'a4', 'c1', 'c2', 'omega', 'Z']
    else:
        # CGL and others use: a, b, c, d, e, f, lambda, T
        eos_columns = ['a', 'b', 'c', 'd', 'e', 'f', 'lambda', 'T']

    r_columns = base_columns + eos_columns

    # Select only columns that exist (for compatibility)
    available_cols = [col for col in r_columns if col in species_data.columns]
    species_data = species_data[available_cols].copy()

    # Convert energy units last, so that the consistency checks above are done in
    # the units of the database entry (as in R)
    if E_units is not None:
        species_data = _convert_E_units(species_data, E_units, model)

    return species_data


def _normalize_E_units(E_units: Union[bool, str, None]) -> Optional[str]:
    """
    Validate the 'E_units' argument of info() and return 'cal', 'J', or None.

    False and None both mean "no conversion"; 'cal' and 'J' are matched
    case-insensitively, as in R's E.units().
    """
    if E_units is None or E_units is False:
        return None
    if not isinstance(E_units, str):
        raise ValueError("info: 'E_units' must be False, 'cal', or 'J'")
    units = E_units.lower()
    if units == 'cal':
        return 'cal'
    if units == 'j':
        return 'J'
    raise ValueError(f"info: units of energy must be either cal or J, not '{E_units}'")


def _convert_E_units(species_data: pd.DataFrame, E_units: str, model: str) -> pd.DataFrame:
    """
    Convert the energy-dependent columns of an info() row to the given units.

    This mirrors the toJoules branch of R's OBIGT2eos() (util.data.R), which
    converts G, H, S, Cp and the first six EOS parameters, plus the seventh
    (omega) for aqueous species only. V (cm3 mol-1), the CGL 'lambda' exponent,
    the CGL transition temperature, and Z are unit-independent, so they are left
    alone. Here the conversion runs in either direction so that entries stored in
    cal and in J can be compared side by side.

    Parameters
    ----------
    species_data : pd.DataFrame
        Single-row DataFrame as returned by _get_species_data(), with EOS columns
        already renamed and unscaled
    E_units : 'cal' or 'J'
        Target energy units
    model : str
        The thermodynamic model of this entry ('HKF', 'DEW', 'AD', 'CGL', ...)

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with converted values and an updated E_units column
    """
    # Imported here to avoid a circular import at module load
    from ..utils.units import convert

    data = species_data.copy()
    irow = data.index[0]
    current = data.at[irow, 'E_units']
    # Nothing to do if the units are unknown or already the requested ones
    if pd.isna(current) or str(current) == E_units:
        return data

    # The EOS parameters that carry energy units, in the order used by R's
    # positional conversion of OBIGT columns 15:20 (always) and 21 (aq only)
    if model in ('HKF', 'DEW'):
        eos_cols = ['a1', 'a2', 'a3', 'a4', 'c1', 'c2']
        eos_aq_col = 'omega'
    elif model == 'AD':
        eos_cols = ['a', 'b', 'xi']
        eos_aq_col = None
    else:
        # CGL and others: a-f are heat capacity coefficients; 'lambda' is an exponent
        eos_cols = ['a', 'b', 'c', 'd', 'e', 'f']
        eos_aq_col = None

    cols = ['G', 'H', 'S', 'Cp'] + eos_cols
    if eos_aq_col is not None and data.at[irow, 'state'] == 'aq':
        cols.append(eos_aq_col)

    for col in cols:
        if col in data.columns and pd.notna(data.at[irow, col]):
            data.at[irow, col] = float(convert(data.at[irow, col], E_units))

    data.at[irow, 'E_units'] = E_units
    return data


def _remove_scaling_factors(species_data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove scaling factors from equation-of-state parameters.
    
    This mirrors the R CHNOSZ OBIGT2eos() function behavior:
    - Applies scaling factors to HKF and DEW species parameters
    - Changes column names from CSV format (a1.a) to EOS format (a1)
    """
    data = species_data.copy()
    
    model = str(data.iloc[0]['model'])
    
    # HKF and DEW models have scaling factors (mirroring R line 425)
    if model in ['HKF', 'DEW']:
        # Original CSV column names and their scaled equivalents
        csv_cols = ['a1.a', 'a2.b', 'a3.c', 'a4.d', 'c1.e', 'c2.f', 'omega.lambda', 'z.T']
        eos_cols = ['a1', 'a2', 'a3', 'a4', 'c1', 'c2', 'omega', 'Z']

        # Scaling factors from R: 10^c(-1, 2, 0, 4, 0, 4, 5, 0)
        scaling_factors = [0.1, 100, 1, 10000, 1, 10000, 100000, 1]

        # Apply scaling and rename columns - always create all 8 columns
        for i, (csv_col, eos_col) in enumerate(zip(csv_cols, eos_cols)):
            if csv_col in data.columns:
                # Apply scaling factor
                scaled_value = data[csv_col] * scaling_factors[i]
                # Add new column with EOS name
                data[eos_col] = scaled_value
            else:
                # Column doesn't exist, set to NaN
                data[eos_col] = np.nan
        
        # Also change column names for non-HKF species following R behavior
        # This is done in OBIGT2eos lines 429-431
        
    elif model == 'AD':
        # For AD species, rename columns and set some to NA (R line 427)
        csv_cols = ['a1.a', 'a2.b', 'a3.c', 'a4.d', 'c1.e', 'c2.f', 'omega.lambda', 'z.T']
        ad_cols = ['a', 'b', 'xi', 'XX1', 'XX2', 'XX3', 'XX4', 'Z']
        
        for csv_col, ad_col in zip(csv_cols, ad_cols):
            if csv_col in data.columns:
                data[ad_col] = data[csv_col]
                # Set unused columns to NA (columns 18-21 in R indexing)
                if ad_col in ['XX1', 'XX2', 'XX3', 'XX4']:
                    data[ad_col] = np.nan
    
    else:
        # For CGL and other models, use generic names (R line 431)
        csv_cols = ['a1.a', 'a2.b', 'a3.c', 'a4.d', 'c1.e', 'c2.f', 'omega.lambda', 'z.T']  
        cgl_cols = ['a', 'b', 'c', 'd', 'e', 'f', 'lambda', 'T']
        
        for csv_col, cgl_col in zip(csv_cols, cgl_cols):
            if csv_col in data.columns:
                data[cgl_col] = data[csv_col]
    
    return data


def _check_and_fill_ghs(species_data: pd.DataFrame, messages: bool = True) -> pd.DataFrame:
    """Check and fill missing G, H, S values."""
    data = species_data.copy()

    # Check if exactly one of G, H, S is missing
    ghs_cols = ['G', 'H', 'S']
    row = data.iloc[0]

    missing = [pd.isna(row[col]) for col in ghs_cols if col in row]
    n_missing = sum(missing)

    if n_missing == 1:
        # Calculate missing value from the other two
        formula = row['formula']
        G = row.get('G', np.nan)
        H = row.get('H', np.nan)
        S = row.get('S', np.nan)
        E_units = row.get('E_units', 'J')

        try:
            # This would use the GHS function from formula utilities
            from ..utils.formula import calculate_ghs
            calculated = calculate_ghs(formula, G=G, H=H, S=S, E_units=E_units)

            # Fill in the missing value
            # (index by label, not position: OBIGT row indices are 1-based labels)
            missing_col = ghs_cols[missing.index(True)]
            data.at[data.index[0], missing_col] = calculated[missing_col]

            if messages:
                print(f"info_numeric: {missing_col} of {row['name']}({row['state']}) is NA; "
                      f"set to {calculated[missing_col]:.2f} {E_units} mol-1")

        except Exception:
            # If calculation fails, leave as NaN
            pass

    return data


def _check_eos_parameters(species_data: pd.DataFrame, messages: bool = True) -> pd.DataFrame:
    """
    Check equation-of-state parameters for consistency.

    This function implements the EOS parameter checking from R's check.EOS function,
    calculating Cp and V from EOS parameters when they are NA in the database.

    Parameters
    ----------
    species_data : pd.DataFrame
        Single-row DataFrame with species data
    messages : bool, default True
        Whether to print informational messages

    Returns
    -------
    pd.DataFrame
        Species data with filled-in Cp and V values (if they were NA)
    """
    data = species_data.copy()
    model = str(data.iloc[0]['model'])
    state = data.iloc[0]['state']

    # Check for HKF and DEW aqueous species
    if model in ['HKF', 'DEW']:
        # Temperature for calculations (Tr = 298.15 K)
        Tr = 298.15
        Theta = 228  # K

        # Get species properties
        name = data.iloc[0]['name']
        E_units = data.iloc[0]['E_units']

        # Check and calculate Cp if it's NA
        if pd.isna(data.iloc[0].get('Cp')):
            # Extract EOS parameters
            c1 = data.iloc[0].get('c1', np.nan)
            c2 = data.iloc[0].get('c2', np.nan)
            omega = data.iloc[0].get('omega', np.nan)

            # Check if we have all required parameters
            if not (pd.isna(c1) or pd.isna(c2) or pd.isna(omega)):
                # Choose value of X consistent with SUPCRT92 or DEW
                if model == 'HKF':
                    X = -3.055586E-7
                elif model == 'DEW':
                    X = -3.09E-7

                # Calculate Cp from EOS parameters
                # Cp = c1 + c2/(Tr-Theta)^2 + omega*Tr*X
                calcCp = c1 + c2 / ((Tr - Theta) ** 2) + omega * Tr * X

                # Fill in the NA value
                data.at[data.index[0], 'Cp'] = calcCp
                if messages:
                    print(f"info.numeric: Cp° of {name}({state}) is NA; set by EOS parameters to {calcCp:.2f} {E_units} K-1 mol-1")

        # Check and calculate V if it's NA (only for aqueous species)
        if pd.isna(data.iloc[0].get('V')):
            # Extract EOS parameters
            a1 = data.iloc[0].get('a1', np.nan)
            a2 = data.iloc[0].get('a2', np.nan)
            a3 = data.iloc[0].get('a3', np.nan)
            a4 = data.iloc[0].get('a4', np.nan)
            omega = data.iloc[0].get('omega', np.nan)

            # Check if we have all required parameters
            if not (pd.isna(a1) or pd.isna(a2) or pd.isna(a3) or pd.isna(a4) or pd.isna(omega)):
                # Choose value of Q consistent with SUPCRT92 or DEW
                if model == 'HKF':
                    Q = 0.00002775729
                elif model == 'DEW':
                    Q = 0.0000005903 * 41.84

                # Calculate V from EOS parameters
                # V = 41.84*a1 + 41.84*a2/2601 + (41.84*a3 + 41.84*a4/2601)/(Tr-Theta) - Q*omega
                calcV = (41.84 * a1 + 41.84 * a2 / 2601 +
                        (41.84 * a3 + 41.84 * a4 / 2601) / (Tr - Theta) -
                        Q * omega)

                # Convert from J to cal if needed
                if E_units == 'J':
                    # Import convert function here to avoid circular import
                    from ..utils.units import convert
                    calcV = convert(calcV, 'cal', messages=messages)

                # Fill in the NA value
                data.at[data.index[0], 'V'] = calcV
                if messages:
                    print(f"info.numeric: V° of {name}({state}) is NA; set by EOS parameters to {calcV:.2f} cm3 mol-1")

    return data


# Convenience functions for common operations
def find_species(name: str, state: Optional[str] = None, messages: bool = True) -> int:
    """
    Find a single species index by name.

    Parameters
    ----------
    name : str
        Species name, formula, or abbreviation
    state : str, optional
        Physical state
    messages : bool, default True
        If True, print informational messages

    Returns
    -------
    int
        Species index (1-based)

    Raises
    ------
    ValueError
        If species not found or multiple matches
    """
    result = info(name, state, messages=messages)

    if pd.isna(result):
        raise ValueError(f"Species '{name}' not found")

    if isinstance(result, list):
        if len(result) > 1:
            raise ValueError(f"Multiple matches found for '{name}'")
        result = result[0]

    return int(result)


def get_species_data(species: Union[str, int], state: Optional[str] = None, messages: bool = True,
                     E_units: Union[bool, str, None] = False) -> pd.DataFrame:
    """
    Get complete thermodynamic data for a species.

    Parameters
    ----------
    species : str or int
        Species name/formula or index
    state : str, optional
        Physical state
    messages : bool, default True
        Display messages?
    E_units : False, 'cal', or 'J', default False
        Convert the properties to these energy units; see info()

    Returns
    -------
    pd.DataFrame
        Species thermodynamic data
    """
    if isinstance(species, str):
        species = find_species(species, state)

    return info(species, messages=messages, E_units=E_units)


def list_species(pattern: Optional[str] = None, state: Optional[str] = None) -> pd.DataFrame:
    """
    List species matching criteria.
    
    Parameters
    ----------
    pattern : str, optional
        Pattern to match in species names
    state : str, optional
        Physical state to filter by
        
    Returns
    -------
    pd.DataFrame
        Matching species information
    """
    thermo_obj = thermo()
    if not thermo_obj.is_initialized():
        thermo_obj.reset()
    
    obigt = thermo_obj.obigt.copy()
    
    # Filter by state
    if state is not None:
        obigt = obigt[obigt['state'] == state]
    
    # Filter by pattern
    if pattern is not None:
        mask = obigt['name'].str.contains(pattern, case=False, na=False)
        obigt = obigt[mask]
    
    # Return relevant columns
    columns = ['name', 'formula', 'state', 'ref1', 'model']
    return obigt[columns].reset_index(drop=True)