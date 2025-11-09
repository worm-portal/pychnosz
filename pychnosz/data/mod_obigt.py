"""
Implementation of mod_OBIGT() function for Python CHNOSZ.

This function modifies or adds entries to the thermodynamic database,
mimicking the behavior of R CHNOSZ mod.OBIGT() function.
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional
import warnings

from ..core.thermo import thermo
from ..core.info import info
from ..utils.formula import makeup


def mod_OBIGT(*args, zap: bool = False, **kwargs) -> Union[int, List[int]]:
    """
    Add or modify species in the thermodynamic database.

    This function replicates the behavior of R CHNOSZ mod.OBIGT() by allowing
    modification of existing species or addition of new species to thermo().obigt.

    Parameters
    ----------
    *args : int, str, list, or dict
        If first argument is numeric: species index or indices to modify
        If first argument is str: species name(s) to modify or add
        If first argument is list/dict: contains all parameters
    zap : bool, default False
        If True, clear all properties except state and model before updating
    **kwargs : any
        Named properties to set (e.g., G=-100, S=50, formula="H2O")
        Special properties: name, state, formula, model, E_units

    Returns
    -------
    int or list of int
        Species index or indices that were modified/added

    Examples
    --------
    >>> import pychnosz
    >>> pychnosz.reset()
    >>> # Add new species
    >>> i = pychnosz.mod_OBIGT("myspecies", formula="C2H6", G=-100, S=50)

    >>> # Modify existing species
    >>> i = pychnosz.mod_OBIGT("water", state="liq", G=-56690)

    >>> # Modify by index
    >>> i_h2o = pychnosz.info("water", "liq")
    >>> i = pychnosz.mod_OBIGT(i_h2o, G=-56690)

    >>> # Add multiple species
    >>> i = pychnosz.mod_OBIGT(["X", "Y"], formula=["C12", "C13"], state=["aq", "cr"])

    Notes
    -----
    This function modifies the thermo() object in place.
    The behavior exactly matches R CHNOSZ mod.OBIGT().
    """

    # Get the thermo system
    thermo_sys = thermo()

    # Ensure the thermodynamic system is initialized
    if not thermo_sys.is_initialized() or thermo_sys.obigt is None:
        raise RuntimeError("Thermodynamic system not initialized. Run reset() first.")

    # Process arguments
    # If called with a dict as first arg (like R's list)
    if len(args) == 1 and isinstance(args[0], dict):
        params = args[0].copy()
    elif len(args) > 0:
        # First positional argument could be species index or name
        first_arg = args[0]
        params = kwargs.copy()

        # Check if first argument is numeric (species index/indices)
        if isinstance(first_arg, (int, np.integer)):
            params['_index'] = first_arg
        elif isinstance(first_arg, (list, tuple)) and len(first_arg) > 0:
            if isinstance(first_arg[0], (int, np.integer)):
                params['_index'] = list(first_arg)
            else:
                # First arg is list of names
                params['name'] = list(first_arg)
        else:
            # First arg is species name
            # If first arg name is not in kwargs, it's the species name
            if 'name' not in params:
                params['name'] = first_arg
    else:
        params = kwargs.copy()

    # Validate we have at least a name/index and one property
    if '_index' not in params and 'name' not in params:
        raise ValueError("Please supply at least a species name and a property to update")

    # Check that we have at least one property
    # When using index: exclude _index and state from property count
    # When using name: exclude name and state from property count (name is identifier, not property)
    if '_index' in params:
        property_keys = set(params.keys()) - {'_index', 'state'}
    else:
        property_keys = set(params.keys()) - {'name', 'state'}

    if len(property_keys) == 0:
        raise ValueError("Please supply at least a species name and a property to update")

    # Get species indices
    if '_index' in params:
        # Working with indices
        ispecies_input = params['_index']
        if not isinstance(ispecies_input, list):
            ispecies_input = [ispecies_input]
        del params['_index']

        # Get species names from indices
        speciesname = []
        for idx in ispecies_input:
            sp_info = info(idx)
            speciesname.append(sp_info['name'].iloc[0] if isinstance(sp_info, pd.DataFrame) else sp_info['name'])

        ispecies = ispecies_input
    else:
        # Working with names
        names = params.get('name')
        if not isinstance(names, list):
            names = [names]

        states = params.get('state')
        if states is not None and not isinstance(states, list):
            states = [states]

        speciesname = names

        # Find species indices
        ispecies = []
        for i, name in enumerate(names):
            state = states[i] if states and i < len(states) else None
            try:
                if state:
                    idx = info(name, state)
                else:
                    idx = info(name)

                # info() returns an int if found
                if isinstance(idx, (int, np.integer)):
                    ispecies.append(int(idx))
                else:
                    # Not found
                    ispecies.append(None)
            except:
                # Species doesn't exist - will be added
                ispecies.append(None)

    # Convert params to DataFrame format
    # Handle list values vs single values
    nspecies = len(ispecies)
    param_df = {}
    for key, value in params.items():
        if isinstance(value, list):
            if len(value) != nspecies:
                raise ValueError(f"Length of '{key}' ({len(value)}) doesn't match number of species ({nspecies})")
            param_df[key] = value
        else:
            param_df[key] = [value] * nspecies

    # Create DataFrame of arguments
    args_df = pd.DataFrame(param_df)

    # Get column names of OBIGT (handle split names with ".")
    obigt_cols = thermo_sys.obigt.columns.tolist()

    # Map parameter names to column names (handle dot notation)
    # e.g., "E.units" can be accessed as "E_units"
    col_mapping = {}
    for col in obigt_cols:
        col_mapping[col] = col
        col_mapping[col.replace('_', '.')] = col
        # Also map first part before dot
        if '_' in col:
            col_mapping[col.split('_')[0]] = col

    # Determine which columns we're updating
    icol = []
    icol_names = []
    for key in args_df.columns:
        if key in col_mapping:
            icol_names.append(col_mapping[key])
            icol.append(obigt_cols.index(col_mapping[key]))
        else:
            raise ValueError(f"Property '{key}' not in thermo$OBIGT")

    # Separate new species from existing ones
    inew = [i for i, idx in enumerate(ispecies) if idx is None]
    iold = [i for i, idx in enumerate(ispecies) if idx is not None]

    result_indices = []

    # Add new species
    if len(inew) > 0:
        # Create blank rows
        newrows = pd.DataFrame(index=range(len(inew)), columns=obigt_cols)
        newrows[:] = np.nan

        # Set defaults
        default_state = thermo_sys.opt.get('state', 'aq')
        default_units = thermo_sys.opt.get('E.units', 'J')

        newrows['state'] = default_state
        newrows['E_units'] = default_units

        # Set formula from name if not provided
        for i, idx in enumerate(inew):
            if 'formula' in args_df.columns:
                newrows.at[i, 'formula'] = args_df.iloc[idx]['formula']
            else:
                newrows.at[i, 'formula'] = args_df.iloc[idx]['name']

        # Fill in provided columns
        for i, idx in enumerate(inew):
            for col_name in icol_names:
                if col_name in args_df.columns:
                    newrows.at[i, col_name] = args_df.at[idx, col_name]

        # Guess model from state
        for i in range(len(newrows)):
            if pd.isna(newrows.iloc[i]['model']):
                if newrows.iloc[i]['state'] == 'aq':
                    newrows.at[i, 'model'] = 'HKF'
                else:
                    newrows.at[i, 'model'] = 'CGL'

        # Validate formulas
        for i in range(len(newrows)):
            formula = newrows.iloc[i]['formula']
            try:
                makeup(formula)
            except Exception as e:
                warnings.warn("Please supply a valid chemical formula as the species name or in the 'formula' argument")
                raise e

        # Add to OBIGT
        ntotal_before = len(thermo_sys.obigt)
        thermo_sys.obigt = pd.concat([thermo_sys.obigt, newrows], ignore_index=True)

        # Reset index to 1-based
        thermo_sys.obigt.index = range(1, len(thermo_sys.obigt) + 1)

        # Update ispecies for new entries
        for i, idx in enumerate(inew):
            new_idx = ntotal_before + i + 1
            if idx < len(ispecies):
                ispecies[idx] = new_idx
            result_indices.append(new_idx)

            # Print message
            name = newrows.iloc[i]['name']
            state = newrows.iloc[i]['state']
            model = newrows.iloc[i]['model']
            e_units = newrows.iloc[i]['E_units']
            print(f"mod_OBIGT: added {name}({state}) with {model} model and energy units of {e_units}")

    # Modify existing species
    if len(iold) > 0:
        for i in iold:
            idx = ispecies[i]

            # Get old values
            oldprop = thermo_sys.obigt.loc[idx, icol_names].copy()
            state = thermo_sys.obigt.loc[idx, 'state']
            model = thermo_sys.obigt.loc[idx, 'model']

            # If zap, clear all values except state and model
            if zap:
                thermo_sys.obigt.loc[idx, :] = np.nan
                thermo_sys.obigt.loc[idx, 'state'] = state
                thermo_sys.obigt.loc[idx, 'model'] = model

            # Get new properties
            newprop = args_df.iloc[i][icol_names].copy()

            # Check if there's any change
            # Compare values element-wise, treating NaN as equal to NaN
            has_change = False
            for col in icol_names:
                old_val = oldprop[col] if col in oldprop.index else np.nan
                new_val = newprop[col] if col in newprop.index else np.nan

                # Check if both are NaN
                if pd.isna(old_val) and pd.isna(new_val):
                    continue
                # Check if one is NaN and other is not
                elif pd.isna(old_val) or pd.isna(new_val):
                    has_change = True
                    break
                # Check if values are different
                elif old_val != new_val:
                    has_change = True
                    break

            if not has_change:
                # No change
                print(f"mod_OBIGT: no change for {speciesname[i]}({state})")
            else:
                # Update the data
                for col_name in icol_names:
                    if col_name in args_df.columns:
                        thermo_sys.obigt.loc[idx, col_name] = args_df.iloc[i][col_name]

                print(f"mod_OBIGT: updated {speciesname[i]}({state})")

            result_indices.append(idx)

    # Return indices
    if len(result_indices) == 1:
        return result_indices[0]
    return result_indices
