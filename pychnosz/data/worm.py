"""
WORM database loader for CHNOSZ.

This module provides functionality to load the Water-Organic-Rock-Microbe (WORM)
thermodynamic database from the WORM-db GitHub repository.

Reference: https://github.com/worm-portal/WORM-db
"""

import pandas as pd
from io import StringIO
from urllib.request import urlopen
from typing import Optional, Tuple
import warnings

from ..core.thermo import thermo
from .add_obigt import add_OBIGT


def can_connect_to(url: str, timeout: int = 5) -> bool:
    """
    Check if a URL is reachable.

    Parameters
    ----------
    url : str
        The URL to check
    timeout : int, default 5
        Connection timeout in seconds

    Returns
    -------
    bool
        True if URL is reachable, False otherwise
    """
    try:
        from urllib.request import Request
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def download_worm_data(url: str) -> Optional[pd.DataFrame]:
    """
    Download WORM database from URL.

    Parameters
    ----------
    url : str
        URL to the WORM CSV file

    Returns
    -------
    pd.DataFrame or None
        DataFrame containing WORM data, or None if download fails
    """
    try:
        from urllib.request import Request
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req, timeout=30) as webpage:
            content = webpage.read().decode('utf-8')
        return pd.read_csv(StringIO(content), sep=",")
    except Exception as e:
        warnings.warn(f"Failed to download WORM data from {url}: {e}")
        return None


def load_WORM(keep_default: bool = False, messages: bool = True) -> bool:
    """
    Load the WORM (Water-Organic-Rock-Microbe) thermodynamic database.

    This function downloads and loads the WORM database from the WORM-db GitHub
    repository. By default, it replaces the OBIGT database with WORM data,
    keeping only water, H+, and e- from the original database.

    Parameters
    ----------
    keep_default : bool, default False
        If False, replace OBIGT with minimal species (water, H+, e-) before
        loading WORM. If True, add WORM species to the existing OBIGT database.
    messages : bool, default True
        Whether to print informational messages

    Returns
    -------
    bool
        True if WORM database was loaded successfully, False otherwise

    Examples
    --------
    >>> import pychnosz
    >>> pychnosz.reset()
    >>> # Load WORM database (replaces default OBIGT)
    >>> pychnosz.load_WORM()
    >>>
    >>> # Load WORM database while keeping default OBIGT species
    >>> pychnosz.reset()
    >>> pychnosz.load_WORM(keep_default=True)

    Notes
    -----
    The WORM database is downloaded from:
    - Species data: https://github.com/worm-portal/WORM-db/master/wrm_data_latest.csv
    - References: https://github.com/worm-portal/WORM-db/master/references.csv

    This feature is exclusive to the Python version of CHNOSZ.
    """

    # WORM database URLs
    url_data = "https://raw.githubusercontent.com/worm-portal/WORM-db/master/wrm_data_latest.csv"
    url_refs = "https://raw.githubusercontent.com/worm-portal/WORM-db/master/references.csv"

    # Name for source_file column
    worm_source_name = "wrm_data_latest.csv"

    # Check if we can connect to the WORM database
    if not can_connect_to(url_data):
        if messages:
            print("load_WORM: could not reach WORM database repository")
        return False

    # Download WORM species data
    worm_data = download_worm_data(url_data)
    if worm_data is None:
        if messages:
            print("load_WORM: failed to download WORM species data")
        return False

    # Get the thermodynamic system
    thermo_sys = thermo()

    if not keep_default:
        # Keep only essential species (water, H+, e-)
        from ..core.info import info
        try:
            # Get indices for essential species
            essential_species = []
            for species in ["water", "H+", "e-"]:
                idx = info(species)
                if idx is not None:
                    if isinstance(idx, (list, tuple)):
                        essential_species.extend(idx)
                    else:
                        essential_species.append(idx)

            if essential_species:
                # Keep only essential species
                minimal_obigt = thermo_sys.obigt.loc[essential_species].copy()
                thermo_sys.obigt = minimal_obigt
        except Exception as e:
            if messages:
                print(f"load_WORM: warning - error keeping essential species: {e}")

    # Add WORM species data (suppress add_OBIGT messages)
    try:
        # Add source_file column to worm_data before adding
        worm_data['source_file'] = worm_source_name

        indices = add_OBIGT(worm_data, messages=False)
    except Exception as e:
        if messages:
            print(f"load_WORM: failed to add WORM species: {e}")
        return False

    # Try to download and load WORM references
    if can_connect_to(url_refs):
        worm_refs = download_worm_data(url_refs)
        if worm_refs is not None:
            # Replace refs with WORM refs
            thermo_sys.refs = worm_refs

    # Update formula_ox if it exists in WORM data
    # This is already handled by add_OBIGT, but we ensure it's set correctly
    if 'formula_ox' in thermo_sys.obigt.columns:
        formula_ox_df = pd.DataFrame({
            'name': thermo_sys.obigt['name'],
            'formula_ox': thermo_sys.obigt['formula_ox']
        })
        formula_ox_df.index = thermo_sys.obigt.index
        thermo_sys.formula_ox = formula_ox_df

    # Print single summary message
    if messages:
        final_obigt = thermo_sys.obigt
        total_species = len(final_obigt)
        aqueous_species = len(final_obigt[final_obigt['state'] == 'aq'])
        print(f"The WORM thermodynamic database has been loaded: {aqueous_species} aqueous, {total_species} total species")

    return True


def reset_WORM(messages: bool = True) -> None:
    """
    Initialize the thermodynamic system with the WORM database.

    This is a convenience function that combines reset() and load_WORM().
    It initializes the system and loads the WORM database in one step.

    Parameters
    ----------
    messages : bool, default True
        Whether to print informational messages

    Examples
    --------
    >>> import pychnosz
    >>> # Initialize with WORM database
    >>> pychnosz.reset_WORM()

    Notes
    -----
    This is equivalent to:
        pychnosz.reset()
        pychnosz.load_WORM()
    """
    from ..utils.reset import reset

    # Reset the system first
    reset(messages=messages)

    # Load WORM database
    success = load_WORM(keep_default=False, messages=messages)

    if not success:
        if messages:
            print("reset_WORM: falling back to default OBIGT database")
