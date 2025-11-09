"""
Species retrieval by element composition.

This module provides Python equivalents of the R functions in retrieve.R:
- retrieve(): Retrieve species containing specified elements

Author: CHNOSZ Python port
"""

import pandas as pd
import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any
import warnings
import re

from .thermo import thermo
from ..utils.formula import makeup, i2A


def retrieve(elements: Optional[Union[str, List[str], Tuple[str]]] = None,
            ligands: Optional[Union[str, List[str], Tuple[str]]] = None,
            state: Optional[Union[str, List[str], Tuple[str]]] = None,
            T: Optional[Union[float, List[float]]] = None,
            P: Union[str, float, List[float]] = "Psat",
            add_charge: bool = True,
            hide_groups: bool = True,
            messages: bool = True) -> pd.Series:
    """
    Retrieve species containing specified elements.

    Parameters
    ----------
    elements : str, list of str, or tuple of str, optional
        Elements in a chemical system. If `elements` is a string, retrieve
        species containing that element.

        E.g., `retrieve("Au")` will return all species containing Au.

        If `elements` is a list, retrieve species that have all of the elements
        in the list.

        E.g., `retrieve(["Au", "Cl"])` will return all species that have both
        Au and Cl.

        If `elements` is a tuple, retrieve species relevant to the system,
        including charged species.

        E.g., `retrieve(("Au", "Cl"))` will return species that have Au
        and/or Cl, including charged species, but no other elements.

    ligands : str, list of str, or tuple of str, optional
        Elements present in any ligands. This affects the species search:
        - If ligands is a state ('cr', 'liq', 'gas', 'aq'), use that as the state filter
        - Otherwise, include elements in the system defined by ligands

    state : str, list of str, or tuple of str, optional
        Filter the result on these state(s) ('aq', 'cr', 'gas', 'liq').

    T : float or list of float, optional
        Temperature (K) for filtering species with non-NA Gibbs energy.

    P : str, float, or list of float, default "Psat"
        Pressure for Gibbs energy calculation. Default is "Psat" (saturation).

    add_charge : bool, default True
        For chemical systems (tuple input), automatically include charge (Z).

    hide_groups : bool, default True
        Exclude group species (names in brackets like [CH2]).

    messages : bool, default True
        Print informational messages. If False, suppress messages about
        updating the stoichiometric matrix and other information.

    Returns
    -------
    pd.Series
        Series of species indices (1-based) with chemical formulas as index.
        This behaves like R's named vector - you can access by name or position.
        Names are chemical formulas (or 'e-' for electrons).
        Values are species indices that match the criteria.

    Examples
    --------
    >>> # All species containing Au
    >>> retrieve("Au")

    >>> # All species that have both Au and Cl
    >>> retrieve(["Au", "Cl"])

    >>> # Au-Cl system: species with Au and/or Cl, including charged species
    >>> retrieve(("Au", "Cl"))

    >>> # All Au-bearing species in the Au-Cl system
    >>> retrieve("Au", ("Cl",))

    >>> # All uncharged Au-bearing species in the Au-Cl system
    >>> retrieve("Au", ("Cl",), add_charge=False)

    >>> # Minerals in the system SiO2-MgO-CaO-CO2
    >>> retrieve(("Si", "Mg", "Ca", "C", "O"), state="cr")

    Notes
    -----
    This function uses 1-based indexing to match R CHNOSZ conventions.
    The returned indices are labels that can be used with .loc[], not positions.
    """
    # Empty argument handling
    if elements is None:
        return pd.Series([], dtype=int)

    thermo_obj = thermo()

    # Initialize database if needed
    if not thermo_obj.is_initialized():
        thermo_obj.reset()

    ## Stoichiometric matrix
    # Get stoichiometric matrix from thermo object
    stoich = _get_or_update_stoich(thermo_obj, messages=messages)

    ## Generate error for missing element(s)
    allelements = []
    if elements is not None:
        if isinstance(elements, (list, tuple)):
            allelements.extend(elements)
        else:
            allelements.append(elements)
    if ligands is not None:
        if isinstance(ligands, (list, tuple)):
            allelements.extend(ligands)
        else:
            allelements.append(ligands)

    not_present = [elem for elem in allelements if elem not in stoich.columns and elem != "all"]
    if not_present:
        if len(not_present) == 1:
            raise ValueError(f'"{not_present[0]}" is not an element that is present in any species in the database')
        else:
            raise ValueError(f'"{", ".join(not_present)}" are not elements that are present in any species in the database')

    ## Handle 'ligands' argument
    if ligands is not None:
        # If 'ligands' is cr, liq, gas, or aq, use that as the state
        if ligands in ['cr', 'liq', 'gas', 'aq']:
            state = ligands
            ispecies = retrieve(elements, add_charge=add_charge, messages=messages)
        else:
            # Include the element in the system defined by the ligands list
            # Convert ligands to tuple if it's a string or list
            if isinstance(ligands, str):
                ligands_tuple = (ligands,)
            elif isinstance(ligands, list):
                ligands_tuple = tuple(ligands)
            else:
                ligands_tuple = ligands

            # Combine elements with ligands
            if isinstance(elements, str):
                combined = (elements,) + ligands_tuple
            elif isinstance(elements, list):
                combined = tuple(elements) + ligands_tuple
            else:
                combined = elements + ligands_tuple

            # Call retrieve() for each argument and take the intersection
            r1 = retrieve(elements, add_charge=add_charge, messages=messages)
            r2 = retrieve(combined, add_charge=add_charge, messages=messages)
            ispecies = np.intersect1d(r1, r2)
    else:
        ## Species identification
        ispecies_list = []

        # Determine if elements is a tuple (chemical system)
        is_system = isinstance(elements, tuple)

        # Convert single string to list for iteration
        if isinstance(elements, str):
            elements_iter = [elements]
        else:
            elements_iter = list(elements)

        # Automatically add charge to a system
        if add_charge and is_system and "Z" not in elements_iter:
            elements_iter.append("Z")

        # Proceed element-by-element
        for element in elements_iter:
            if element == "all":
                ispecies_list.append(np.array(thermo_obj.obigt.index.tolist()))
            else:
                # Identify the species that have the element
                has_element = (stoich[element] != 0)
                ispecies_list.append(np.array(stoich.index[has_element].tolist()))

        # Now we have a list of ispecies (one array for each element)
        # What we do next depends on whether the argument is a tuple or not
        if is_system:
            # For a chemical system, all species are included that do not contain any other elements
            ispecies = np.unique(np.concatenate(ispecies_list))

            # Get columns not in elements
            other_columns = [col for col in stoich.columns if col not in elements_iter]

            if other_columns:
                # Check which species have other elements
                otherstoich = stoich.loc[ispecies, other_columns]
                iother = (otherstoich != 0).any(axis=1)
                ispecies = ispecies[~iother.values]
        else:
            # Get species that have all the elements; the species must be present in each array
            # This is the intersection of all arrays
            ispecies = ispecies_list[0]
            for arr in ispecies_list[1:]:
                ispecies = np.intersect1d(ispecies, arr)

    # Exclude groups
    if hide_groups:
        obigt = thermo_obj.obigt
        names = obigt.loc[ispecies, 'name'].values
        is_group = np.array([bool(re.match(r'^\[.*\]$', str(name))) for name in names])
        ispecies = ispecies[~is_group]

    # Filter on state
    if state is not None:
        obigt = thermo_obj.obigt

        # Ensure state is a list
        if isinstance(state, str):
            state_list = [state]
        elif isinstance(state, tuple):
            state_list = list(state)
        else:
            state_list = state

        species_states = obigt.loc[ispecies, 'state'].values
        istate = np.array([s in state_list for s in species_states])
        ispecies = ispecies[istate]

    # Require non-NA Delta G0 at specific temperature
    if T is not None:
        from .subcrt import subcrt
        # Suppress warnings and (optionally) messages
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = subcrt(ispecies.tolist(), T=T, P=P, messages=False, show=False)
                if result is not None and 'out' in result:
                    G_values = []
                    for species_out in result['out']:
                        if isinstance(species_out, dict) and 'G' in species_out:
                            G = species_out['G']
                            if isinstance(G, (list, np.ndarray)):
                                G_values.append(G[0] if len(G) > 0 else np.nan)
                            else:
                                G_values.append(G)
                        else:
                            G_values.append(np.nan)

                    # Filter out species with NA G values
                    has_G = np.array([not pd.isna(g) for g in G_values])
                    ispecies = ispecies[has_G]
            except:
                # If subcrt fails, keep all species
                pass

    # Create a pandas Series with formula names (R-style named vector)
    obigt = thermo_obj.obigt
    formulas = obigt.loc[ispecies, 'formula'].values

    # Use e- instead of (Z-1) for electron
    formulas = np.array([f if f != '(Z-1)' else 'e-' for f in formulas])

    # Return empty Series if nothing found
    if len(ispecies) == 0:
        return pd.Series([], dtype=int)

    # Create a pandas Series with formulas as index (R-style named vector)
    # This allows both named access (result["Au"]) and positional access (result[0])
    result = pd.Series(ispecies, index=formulas)

    return result


def _get_or_update_stoich(thermo_obj, messages: bool = True) -> pd.DataFrame:
    """
    Get or update the stoichiometric matrix.

    This function manages the stoichiometric matrix cache, updating it
    when the OBIGT database changes.

    Parameters
    ----------
    thermo_obj : ThermoSystem
        The thermodynamic system object
    messages : bool, default True
        Print informational messages about updating the stoichiometric matrix

    Returns
    -------
    pd.DataFrame
        Stoichiometric matrix with species indices as index and elements as columns
    """
    obigt = thermo_obj.obigt
    if obigt is None:
        raise RuntimeError("Thermodynamic database not initialized")

    formula = obigt['formula']

    # Check if we have a cached stoichiometric DataFrame
    # We'll store it as a private attribute _stoich_df
    if not hasattr(thermo_obj, '_stoich_df'):
        thermo_obj._stoich_df = None
        thermo_obj._stoich_df_formulas = None

    stoich_df = thermo_obj._stoich_df
    stoich_df_formulas = thermo_obj._stoich_df_formulas

    # Check if stoichiometric matrix needs updating
    if stoich_df is None or stoich_df_formulas is None or not np.array_equal(stoich_df_formulas, formula.values):
        # Update needed
        if messages:
            print("retrieve: updating stoichiometric matrix")

        # Calculate stoichiometry for all formulas
        # Use makeup to get stoichiometric matrix
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Get makeup for all formulas
            makeups = []
            all_elements = set()

            for f in formula:
                try:
                    m = makeup(str(f))
                    if m is not None:
                        makeups.append(m)
                        all_elements.update(m.keys())
                    else:
                        makeups.append({})
                except:
                    makeups.append({})

            # Sort elements for consistent column order
            all_elements = sorted(list(all_elements))

            # Build stoichiometric matrix
            stoich_data = []
            for m in makeups:
                row = [m.get(elem, 0) for elem in all_elements]
                stoich_data.append(row)

            # Create DataFrame with species indices as index (matching obigt.index)
            stoich_df = pd.DataFrame(stoich_data, columns=all_elements, index=obigt.index)

        # Store the stoichiometric matrix
        thermo_obj._stoich_df = stoich_df
        thermo_obj._stoich_df_formulas = formula.values.copy()

    return stoich_df


__all__ = ['retrieve']
