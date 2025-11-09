"""
Implementation of add_OBIGT() function for Python CHNOSZ.

This function loads external OBIGT database files and replaces or adds
entries to the currently loaded thermodynamic database, mimicking the
behavior of R CHNOSZ add.OBIGT() function.
"""

import pandas as pd
import numpy as np
import os
from typing import Union, List, Optional
import warnings

from ..core.thermo import thermo


def add_OBIGT(file: Union[str, pd.DataFrame], force: bool = True, messages: bool = True) -> List[int]:
    """
    Add or replace entries in the thermodynamic database from external files or DataFrames.

    This function replicates the behavior of R CHNOSZ add.OBIGT() by loading
    CSV files from inst/extdata/OBIGT/ or accepting pandas DataFrames directly,
    and replacing entries with matching names.

    Parameters
    ----------
    file : str or pd.DataFrame
        Either:
        - Name of the database file to load (e.g., "SUPCRT92")
          The function will look for file.csv in inst/extdata/OBIGT/
        - Full path to a CSV file
        - A pandas DataFrame containing OBIGT data
    force : bool, default True
        If True, proceed even if some species are not found
    messages : bool, default True
        If True, print informational messages about additions/replacements
        If False, suppress all output (equivalent to R's suppressMessages())

    Returns
    -------
    list of int
        List of species indices (1-based) that were added or replaced

    Examples
    --------
    >>> import pychnosz
    >>> import pandas as pd
    >>>
    >>> # Example 1: Load from file name
    >>> pychnosz.reset()
    >>> indices = pychnosz.add_OBIGT("SUPCRT92")
    >>>
    >>> # Example 2: Load from DataFrame
    >>> thermo_df = pd.read_csv("thermodata.csv")
    >>> indices = pychnosz.add_OBIGT(thermo_df)
    >>>
    >>> # Example 3: Suppress messages
    >>> indices = pychnosz.add_OBIGT(thermo_df, messages=False)

    Notes
    -----
    This function modifies the thermo() object in place, replacing entries
    with matching names and adding new entries for species not in the database.
    The behavior exactly matches R CHNOSZ add.OBIGT().
    """

    # Get the thermo system
    thermo_sys = thermo()

    # Ensure the thermodynamic system is initialized
    if not thermo_sys.is_initialized() or thermo_sys.obigt is None:
        thermo_sys.reset()

    # Handle DataFrame input
    if isinstance(file, pd.DataFrame):
        new_data = file.copy()
        file_path = "<DataFrame>"
        file_basename = None
    else:
        # Handle string file path
        # If file is not an existing path, look for it in OBIGT directories
        if not os.path.exists(file):
            if not file.endswith('.csv'):
                file_to_find = file + '.csv'
            else:
                file_to_find = file

            # Look for the file in the OBIGT data directory
            # Use package-relative path
            base_paths = [
                os.path.join(os.path.dirname(__file__), 'extdata', 'OBIGT'),
            ]

            file_path = None
            for base_path in base_paths:
                potential_path = os.path.join(base_path, file_to_find)
                if os.path.exists(potential_path):
                    file_path = potential_path
                    break

            if file_path is None:
                raise FileNotFoundError(f"Could not find OBIGT file: {file}")
        else:
            # Use the file path as provided
            file_path = file

        # Extract the basename for source_file column
        file_basename = os.path.basename(file_path)

        # Read the CSV file
        try:
            new_data = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading {file_path}: {e}")

    if new_data.empty:
        raise ValueError(f"No data found in {file_path if isinstance(file, str) else 'DataFrame'}")

    # Validate columns before processing
    # Get the current OBIGT database to determine required columns
    to1 = thermo_sys.obigt

    # Define core required columns that all species must have
    # These are the fundamental columns needed for thermodynamic calculations
    # Model-specific columns (logK*, T*, P*, etc.) are optional
    core_required_columns = [
        'name', 'abbrv', 'formula', 'state', 'ref1', 'ref2', 'date', 'E_units',
        'G', 'H', 'S', 'Cp', 'V',
        'a1.a', 'a2.b', 'a3.c', 'a4.d', 'c1.e', 'c2.f', 'omega.lambda', 'z.T'
    ]

    # The 'model' column is optional and will be auto-generated if missing
    # Filter to only include columns that exist in current OBIGT (for compatibility)
    required_columns = [col for col in core_required_columns if col in to1.columns]

    # Check for missing required columns
    missing_columns = [col for col in required_columns if col not in new_data.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in input data: {', '.join(missing_columns)}. "
            f"Please ensure the CSV file contains all necessary OBIGT database columns."
        )

    # Special handling for 'model' column
    if 'model' not in new_data.columns:
        # Create model column with proper values
        new_data = new_data.copy()  # Make a copy to avoid SettingWithCopyWarning

        # Assign model based on state:
        # - aqueous species (state == 'aq') get 'HKF'
        # - non-aqueous species get 'CGL'
        new_data['model'] = new_data['state'].apply(lambda x: 'HKF' if x == 'aq' else 'CGL')

        # Issue a warning to inform the user
        warnings.warn(
            "The 'model' column was not found in the input data. "
            "Auto-generating 'model' column: 'HKF' for aqueous species (state='aq'), "
            "'CGL' for all other species.",
            UserWarning
        )

    # Get energy units from the file (all unique values)
    # Match R's behavior: unique values joined with " and "
    if 'E_units' in new_data.columns:
        unique_units = new_data['E_units'].dropna().unique().tolist()
        # Filter out non-energy unit values like "CGL" (which is a model, not energy unit)
        # Valid energy units are typically "cal" and "J"
        energy_unit_names = [str(u) for u in unique_units if str(u) in ['cal', 'J']]
        # Join in the order they appear in the file (matching R's paste(unique(...), collapse = " and "))
        energy_units_str = ' and '.join(energy_unit_names) if energy_unit_names else 'cal'
    else:
        energy_units_str = 'cal'

    # Create identifier strings for matching (name + state)
    id1 = to1['name'].astype(str) + ' ' + to1['state'].astype(str)
    id2 = new_data['name'].astype(str) + ' ' + new_data['state'].astype(str)

    # Track the indices we've modified/added
    inew = []

    # Check which entries in new_data exist in current database
    # does_exist is a boolean array indicating which id2 entries are in id1
    does_exist = id2.isin(id1.values)

    # Get the indices in to1 where matches exist (matching R's match(id2, id1))
    # This gives us the positions in to1 for each id2 element
    ispecies_exist = []
    for i, id_val in enumerate(id2):
        if does_exist.iloc[i]:
            # Find the index in to1 where this matches
            match_idx = id1[id1 == id_val].index[0]
            ispecies_exist.append(match_idx)
        else:
            ispecies_exist.append(None)

    nexist = sum(does_exist)

    # Check if new_data has columns that to1 doesn't have, and add them
    # Use object dtype for new columns to match pandas default behavior and avoid FutureWarning
    for col in new_data.columns:
        if col not in to1.columns:
            # Determine dtype from new_data
            dtype = new_data[col].dtype
            # Use object dtype for string columns to avoid dtype incompatibility
            if dtype == object or pd.api.types.is_string_dtype(dtype):
                to1[col] = pd.Series(dtype=object)
            else:
                to1[col] = np.nan

    if force:
        # Replace existing entries
        if nexist > 0:
            # Update rows in to1 for species that exist
            for i, idx in enumerate(ispecies_exist):
                if idx is not None:
                    # Replace the row in to1 with data from new_data
                    for col in new_data.columns:
                        # col should now be in to1 since we added missing columns above
                        to1.loc[idx, col] = new_data.iloc[i][col]
                    # Set source_file for replaced entries
                    if file_basename is not None:
                        to1.loc[idx, 'source_file'] = file_basename

            # Add these indices to inew
            inew.extend([idx for idx in ispecies_exist if idx is not None])

            # Remove existing entries from new_data (to2 <- to2[!does.exist, ])
            to2 = new_data[~does_exist].copy()
        else:
            to2 = new_data.copy()
    else:
        # Ignore any new entries that already exist
        to2 = new_data[~does_exist].copy()
        nexist = 0

    # Add new entries
    if len(to2) > 0:
        # Store the starting index for new additions
        len_id1 = len(id1)

        # Ensure new entries have all required columns
        # Make a proper copy to avoid SettingWithCopyWarning
        to2 = to2.copy()
        for col in to1.columns:
            if col not in to2.columns:
                to2[col] = np.nan

        # Set source_file for new entries
        if file_basename is not None:
            to2['source_file'] = file_basename

        # Reorder columns to match current OBIGT
        to2 = to2.reindex(columns=to1.columns)

        # Add to the database
        # Use concat with explicit future behavior to avoid FutureWarning
        to1 = pd.concat([to1, to2], ignore_index=True, sort=False)

        # Add new indices: (length(id1)+1):nrow(to1)
        new_indices = list(range(len_id1 + 1, len(to1) + 1))
        inew.extend(new_indices)

    # Reset rownames to 1:nrow (matching R's rownames(thermo$OBIGT) <- 1:nrow(thermo$OBIGT))
    to1.index = range(1, len(to1) + 1)

    # Update the thermo system with modified database
    thermo_sys.obigt = to1

    # Update formula_ox if the column exists in the database
    if 'formula_ox' in to1.columns:
        # Create a DataFrame with name and formula_ox columns
        # Keep the same index as the obigt DataFrame (1-based)
        formula_ox_df = pd.DataFrame({
            'name': to1['name'],
            'formula_ox': to1['formula_ox']
        })
        # Preserve the 1-based index
        formula_ox_df.index = to1.index
        thermo_sys.formula_ox = formula_ox_df
    else:
        # If formula_ox column doesn't exist, set to None
        thermo_sys.formula_ox = None

    # Print summary (matching R CHNOSZ output)
    if messages:
        print(f"add_OBIGT: read {len(new_data)} rows; made {nexist} replacements, {len(to2) if len(to2) > 0 else 0} additions [energy units: {energy_units_str}]")

    return inew


def list_OBIGT_files() -> List[str]:
    """
    List available OBIGT database files.
    
    Returns
    -------
    list of str
        List of available .csv files in the OBIGT directory
    """
    
    # Use package-relative path
    base_paths = [
        os.path.join(os.path.dirname(__file__), 'extdata', 'OBIGT'),
    ]
    
    files = []
    for base_path in base_paths:
        if os.path.exists(base_path):
            csv_files = [f[:-4] for f in os.listdir(base_path) if f.endswith('.csv')]
            files.extend(csv_files)
            break
    
    return sorted(list(set(files)))  # Remove duplicates and sort


def reset_OBIGT() -> None:
    """
    Reset OBIGT database to default state.
    
    This function reloads the default thermodynamic database,
    removing any modifications made by add_OBIGT().
    """
    from ..utils.reset import reset
    reset()
    print("OBIGT database reset to default state")