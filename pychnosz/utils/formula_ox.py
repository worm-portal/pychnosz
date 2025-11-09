"""
Formula oxidation state utilities for CHNOSZ.

This module provides functions for working with chemical formulas that include
element oxidation states, specifically for use with the WORM thermodynamic database.
"""

from typing import Union, Dict, List
import pandas as pd

from ..core.thermo import thermo
from ..core.info import info


def get_formula_ox(name: Union[str, int]) -> Dict[str, float]:
    """
    Get quantities of elements and their oxidation states in a chemical compound.

    This function only works when a thermodynamic database with the 'formula_ox'
    column is loaded (e.g., the WORM database). For example, an input of "magnetite"
    would return the following: {'Fe+3': 2.0, 'Fe+2': 1.0, 'O-2': 4.0}.

    Parameters
    ----------
    name : str or int
        The name or database index of the chemical species of interest. Example:
        "magnetite" or 738.

    Returns
    -------
    dict
        A dictionary where each key represents an element in a specific
        oxidation state, and its value is the number of that element in the
        chemical species' formula.

    Raises
    ------
    TypeError
        If input is not a string or integer.
    AttributeError
        If the WORM thermodynamic database is not loaded (no formula_ox attribute).
    ValueError
        If the species is not found in the database or does not have oxidation
        state information.

    Examples
    --------
    >>> import pychnosz
    >>> # Load the WORM database
    >>> pychnosz.thermo("WORM")
    >>> # Get formula with oxidation states for magnetite
    >>> pychnosz.get_formula_ox("magnetite")
    {'Fe+3': 2.0, 'Fe+2': 1.0, 'O-2': 4.0}
    >>> # Can also use species index
    >>> pychnosz.get_formula_ox(738)
    {'Fe+3': 2.0, 'Fe+2': 1.0, 'O-2': 4.0}

    Notes
    -----
    This function requires the wormutils package to be installed for parsing
    the formula_ox strings. Install it with: pip install wormutils
    """

    # Import parse_formula_ox from wormutils
    try:
        from wormutils import parse_formula_ox
    except ImportError:
        raise ImportError(
            "The wormutils package is required to use get_formula_ox(). "
            "Install it with: pip install wormutils"
        )

    # Validate input type
    if not isinstance(name, str) and not isinstance(name, int):
        raise TypeError(
            "Must provide input as a string (chemical species name) or "
            "an integer (chemical species index)."
        )

    # Get the thermo system
    thermo_sys = thermo()

    # Convert index to name if necessary
    if isinstance(name, int):
        species_info = info(name, messages=False)
        if species_info is None or len(species_info) == 0:
            raise ValueError(f"Species index {name} not found in the database.")
        name = species_info.name.iloc[0]

    # Check if formula_ox exists in thermo()
    if not hasattr(thermo_sys, 'formula_ox') or thermo_sys.formula_ox is None:
        raise AttributeError(
            "The 'formula_ox' attribute is not available. "
            "This function only works when the WORM thermodynamic database "
            "is loaded. Load it with: pychnosz.thermo('WORM')"
        )

    df = thermo_sys.formula_ox

    # Check if the species name exists in the database
    if name not in list(df["name"]):
        raise ValueError(
            f"The species '{name}' was not found in the loaded thermodynamic database."
        )

    # Get the formula_ox string for this species
    try:
        formula_ox_str = df[df["name"] == name]["formula_ox"].iloc[0]
    except (KeyError, IndexError):
        raise ValueError(
            f"The species '{name}' does not have elemental oxidation states "
            "given in the 'formula_ox' column of the loaded thermodynamic database."
        )

    # Check if formula_ox is valid (not NaN or empty)
    if formula_ox_str is None or (isinstance(formula_ox_str, float) and pd.isna(formula_ox_str)) or formula_ox_str == "":
        raise ValueError(
            f"The species '{name}' does not have elemental oxidation states "
            "given in the 'formula_ox' column of the loaded thermodynamic database."
        )

    # Parse the formula_ox string and return
    return parse_formula_ox(formula_ox_str)


def get_n_element_ox(names: Union[str, int, List[Union[str, int]], pd.Series],
                     element_ox: str,
                     binary: bool = False) -> List[Union[float, bool]]:
    """
    Get the number of an element of a chosen oxidation state in chemical species formulas.

    This function only works when a thermodynamic database with the 'formula_ox'
    column is loaded (e.g., the WORM database).

    If binary is False, returns a list containing the number of the chosen
    element and oxidation state in the chemical species. For example, how many
    ferrous irons are in the formulae of hematite, fayalite, and magnetite,
    respectively?

    >>> get_n_element_ox(names=["hematite", "fayalite", "magnetite"],
    ...                  element_ox="Fe+2",
    ...                  binary=False)
    [0, 2.0, 1.0]

    If binary is True, returns a list of whether or not ferrous iron is in their
    formulas:

    >>> get_n_element_ox(names=["hematite", "fayalite", "magnetite"],
    ...                  element_ox="Fe+2",
    ...                  binary=True)
    [False, True, True]

    Parameters
    ----------
    names : str, int, list of str/int, or pd.Series
        The name or database index of a chemical species, or a list of
        names or indices. Can also be a pandas Series (e.g., from retrieve()).
        Example: ["hematite", "fayalite", "magnetite"] or [788, 782, 798].
    element_ox : str
        An element with a specific oxidation state. For example: "Fe+2" for
        ferrous iron.
    binary : bool, default False
        Should the output list show True/False for presence or absence of the
        element defined by `element_ox`? By default, this parameter is set to
        False so the output list shows quantities of the element instead.

    Returns
    -------
    list of float or list of bool
        A list containing quantities of the chosen element oxidation state in
        the formulas of the chemical species (if `binary=False`) or whether the
        chosen element oxidation state is present in the formulae (if `binary=True`).

    Raises
    ------
    AttributeError
        If the WORM thermodynamic database is not loaded (no formula_ox attribute).
    ValueError
        If a species is not found in the database or does not have oxidation
        state information.

    Examples
    --------
    >>> import pychnosz
    >>> # Load the WORM database
    >>> pychnosz.thermo("WORM")
    >>> # Get counts of Fe+2 in several minerals
    >>> pychnosz.get_n_element_ox(["hematite", "fayalite", "magnetite"], "Fe+2")
    [0, 2.0, 1.0]
    >>> # Get binary presence/absence
    >>> pychnosz.get_n_element_ox(["hematite", "fayalite", "magnetite"], "Fe+2", binary=True)
    [False, True, True]
    >>> # Can also use with retrieve()
    >>> r = pychnosz.retrieve("Fe", ["Si", "O", "H"], state=["cr"])
    >>> pychnosz.get_n_element_ox(r, "Fe+2")
    [1, 0, 0, 2.0, 1, 0, 1, 3.0, 1, 3.0, 0, 7.0]

    Notes
    -----
    This function requires the wormutils package to be installed for parsing
    the formula_ox strings. Install it with: pip install wormutils
    """

    # Handle pandas Series (e.g., from retrieve())
    if isinstance(names, pd.Series):
        # Convert Series to list of indices
        names = names.values.tolist()
    # Handle single name/index
    elif not isinstance(names, list):
        names = [names]

    # Get the count of element_ox for each species
    n_list = []
    for name in names:
        # Get the formula_ox dictionary for this species
        formula_ox_dict = get_formula_ox(name)
        # Get the count of element_ox (default to 0 if not present)
        count = formula_ox_dict.get(element_ox, 0)
        n_list.append(count)

    # Convert to binary if requested
    if binary:
        out_list = [True if n != 0 else False for n in n_list]
    else:
        out_list = n_list

    return out_list
