"""
Reaction balancing utilities.

This module provides functions to balance chemical reactions using basis species,
without calculating thermodynamic properties. This is more efficient than subcrt()
when only the reaction stoichiometry is needed.
"""

import numpy as np
import pandas as pd
from typing import List, Union, Optional, Tuple

from ..core.thermo import thermo
from ..core.info import info
from ..utils.formula import makeup


def balance_reaction(species: Union[str, List[str], int, List[int]],
                    coeff: Union[int, float, List[Union[int, float]]],
                    state: Optional[Union[str, List[str]]] = None,
                    basis: Optional[pd.DataFrame] = None,
                    messages: bool = False) -> Optional[Tuple[List, List]]:
    """
    Balance a chemical reaction using basis species.

    This function checks if a reaction is balanced and, if not, attempts to
    balance it by adding basis species. Unlike subcrt(), this function only
    performs the balancing calculation without computing thermodynamic properties,
    making it much more efficient for reaction generation.

    Parameters
    ----------
    species : str, int, list of str, or list of int
        Species names or indices in the reaction
    coeff : int, float, or list
        Stoichiometric coefficients for the species
    state : str, list of str, or None
        Physical states for species (optional)
    basis : pd.DataFrame, optional
        Basis species definition to use. If None, uses global basis from thermo()
    messages : bool
        Whether to print informational messages

    Returns
    -------
    tuple or None
        If reaction is balanced or can be balanced:
            (balanced_species, balanced_coeffs) where both are lists
        If reaction cannot be balanced:
            None

    Examples
    --------
    >>> import chnosz
    >>> pychnosz.reset()
    >>> pychnosz.basis(['H2O', 'H+', 'Fe+2'])
    >>> # Balance reaction for Fe+3
    >>> species, coeffs = balance_reaction('Fe+3', [-1])
    >>> print(f"Species: {species}")
    >>> print(f"Coefficients: {coeffs}")
    """

    # Convert inputs to lists
    if not isinstance(species, list):
        species = [species]
    if not isinstance(coeff, list):
        coeff = [coeff]
    if state is not None and not isinstance(state, list):
        state = [state]

    # Validate lengths
    if len(species) != len(coeff):
        raise ValueError("Length of species and coeff must match")

    # Get basis definition
    thermo_sys = thermo()
    if basis is None:
        if hasattr(thermo_sys, 'basis') and thermo_sys.basis is not None:
            basis = thermo_sys.basis
        else:
            raise RuntimeError("Basis species not defined. Call pychnosz.basis() first.")

    # Look up species indices
    ispecies = []
    for i, sp in enumerate(species):
        if isinstance(sp, (int, np.integer)):
            ispecies.append(int(sp))
        else:
            sp_state = state[i] if state and i < len(state) else None
            sp_idx = info(sp, sp_state, messages=messages)
            if sp_idx is None or (isinstance(sp_idx, float) and np.isnan(sp_idx)):
                raise ValueError(f"Species not found: {sp}")
            ispecies.append(sp_idx)

    # Calculate mass balance
    try:
        mass_balance = makeup(ispecies, coeff, sum_formulas=True)

        # Check if balanced
        tolerance = 1e-6
        unbalanced_elements = {elem: val for elem, val in mass_balance.items()
                             if abs(val) > tolerance}

        if not unbalanced_elements:
            # Already balanced
            if messages:
                print("Reaction is already balanced")
            return (species, coeff)

        # Reaction is unbalanced - try to balance using basis species
        missing_composition = {elem: -val for elem, val in unbalanced_elements.items()}

        if messages:
            print("Reaction is not balanced; missing composition:")
            elem_names = list(missing_composition.keys())
            elem_values = list(missing_composition.values())
            print(" ".join(elem_names))
            print(" ".join([f"{val:.4f}" for val in elem_values]))

        # Get basis element columns
        basis_elements = [col for col in basis.columns
                        if col not in ['ispecies', 'logact', 'state']]

        # Check if all missing elements are in basis
        missing_elements = set(missing_composition.keys())
        if not missing_elements.issubset(set(basis_elements)):
            if messages:
                print(f"Cannot balance: elements {missing_elements - set(basis_elements)} not in basis")
            return None

        # Calculate coefficients for missing composition from basis species
        missing_matrix = np.zeros((1, len(basis_elements)))
        for i, elem in enumerate(basis_elements):
            missing_matrix[0, i] = missing_composition.get(elem, 0)

        # Get basis matrix
        basis_matrix = basis[basis_elements].values.T  # Transpose: (elements × basis_species)

        try:
            # Try to find simple integer solutions first
            basis_coeffs = _find_simple_integer_solution(
                basis_matrix.T,
                missing_matrix.flatten(),
                basis['ispecies'].tolist(),
                missing_composition
            )

            if basis_coeffs is None:
                # Fall back to linear algebra solution
                basis_coeffs = np.linalg.solve(basis_matrix, missing_matrix.T).flatten()

                # Apply zapsmall equivalent (digits=7)
                basis_coeffs = np.around(basis_coeffs, decimals=7)

                # Clean up very small numbers
                basis_coeffs[np.abs(basis_coeffs) < 1e-7] = 0

            # Get non-zero coefficients and corresponding basis species
            nonzero_indices = np.abs(basis_coeffs) > 1e-6
            if not np.any(nonzero_indices):
                if messages:
                    print("No basis species needed to balance (coefficients are zero)")
                return (species, coeff)

            # Get basis species info
            basis_indices = basis['ispecies'].values[nonzero_indices]
            basis_coeffs_nz = basis_coeffs[nonzero_indices]

            # Create new species list and coefficients
            new_species = list(species) + [int(idx) for idx in basis_indices]
            new_coeff = list(coeff) + list(basis_coeffs_nz)

            if messages:
                print("Balanced reaction by adding basis species:")
                for sp_idx, cf in zip(basis_indices, basis_coeffs_nz):
                    sp_name = thermo_sys.obigt.loc[int(sp_idx)]['name']
                    print(f"  {cf:.4f} {sp_name}")

            # CRITICAL: Consolidate duplicate species by summing coefficients
            # This prevents infinite recursion and matches subcrt's behavior
            consolidated_species = []
            consolidated_coeffs = []

            # Convert all species to indices for consolidation
            species_indices = []
            for sp in new_species:
                if isinstance(sp, (int, np.integer)):
                    species_indices.append(int(sp))
                else:
                    sp_idx = info(sp, None, messages=False)
                    if sp_idx is None or (isinstance(sp_idx, float) and np.isnan(sp_idx)):
                        # Keep as string if not found
                        species_indices.append(sp)
                    else:
                        species_indices.append(sp_idx)

            # Group by species index and sum coefficients
            species_coeff_map = {}
            for sp_idx, coeff in zip(species_indices, new_coeff):
                if sp_idx in species_coeff_map:
                    species_coeff_map[sp_idx] += coeff
                else:
                    species_coeff_map[sp_idx] = coeff

            # Remove species with zero coefficient (cancelled out)
            for sp_idx, coeff in species_coeff_map.items():
                if abs(coeff) > tolerance:
                    consolidated_species.append(sp_idx)
                    consolidated_coeffs.append(coeff)

            # Now check if consolidated reaction is balanced
            # If not, recursively balance again
            try:
                final_mass_balance = makeup(consolidated_species, consolidated_coeffs, sum_formulas=True)
                final_unbalanced = {elem: val for elem, val in final_mass_balance.items()
                                   if abs(val) > tolerance}

                if final_unbalanced:
                    # Still unbalanced after consolidation - recursively balance
                    if messages:
                        print(f"After consolidation, reaction still unbalanced: {final_unbalanced}")
                        print(f"Attempting recursive balance...")
                    return balance_reaction(consolidated_species, consolidated_coeffs, state=None,
                                           basis=basis, messages=messages)
                else:
                    # Balanced! Return consolidated result
                    if messages:
                        print(f"Reaction balanced after consolidation")
                    return (consolidated_species, consolidated_coeffs)

            except Exception as e:
                # If check fails, return consolidated result anyway
                if messages:
                    print(f"Could not verify final balance: {e}")
                return (consolidated_species, consolidated_coeffs)

        except np.linalg.LinAlgError:
            if messages:
                print("Cannot balance: singular basis matrix")
            return None

    except Exception as e:
        if messages:
            print(f"Error checking reaction balance: {e}")
            import traceback
            traceback.print_exc()
        return None


def _find_simple_integer_solution(basis_matrix, missing_vector, basis_species_indices, missing_composition):
    """
    Find simple integer solutions for basis species coefficients.

    This tries to match R CHNOSZ behavior by preferring simple integer combinations
    like 1 H2O + 1 NH3 over complex fractional solutions.
    """
    n_species = len(basis_species_indices)

    # Try single species solutions first (coefficient 1-3)
    for i in range(n_species):
        for coeff in [1, 2, 3, -1, -2, -3]:
            test_coeffs = np.zeros(n_species)
            test_coeffs[i] = coeff
            result = basis_matrix @ test_coeffs
            if np.allclose(result, missing_vector, atol=1e-10):
                return test_coeffs

    # Try two-species solutions (coefficients ±1, ±2 each)
    for i in range(n_species):
        for j in range(i+1, n_species):
            for coeff1 in [1, 2, -1, -2]:
                for coeff2 in [1, 2, -1, -2]:
                    test_coeffs = np.zeros(n_species)
                    test_coeffs[i] = coeff1
                    test_coeffs[j] = coeff2
                    result = basis_matrix @ test_coeffs
                    if np.allclose(result, missing_vector, atol=1e-10):
                        return test_coeffs

    # Try three-species solutions (coefficient ±1 each)
    for i in range(n_species):
        for j in range(i+1, n_species):
            for k in range(j+1, n_species):
                for coeff1 in [1, -1]:
                    for coeff2 in [1, -1]:
                        for coeff3 in [1, -1]:
                            test_coeffs = np.zeros(n_species)
                            test_coeffs[i] = coeff1
                            test_coeffs[j] = coeff2
                            test_coeffs[k] = coeff3
                            result = basis_matrix @ test_coeffs
                            if np.allclose(result, missing_vector, atol=1e-10):
                                return test_coeffs

    return None  # No simple solution found


def format_reaction(species: List[Union[str, int]], coeffs: List[float]) -> str:
    """
    Format a reaction as a string for EQ3/6 input.

    Parameters
    ----------
    species : list
        Species names or indices
    coeffs : list
        Stoichiometric coefficients

    Returns
    -------
    str
        Formatted reaction string like "-1.0000 Fe+3 1.0000 Fe+2 0.2500 O2(g)"
    """
    thermo_sys = thermo()
    parts = []

    for sp, coeff in zip(species, coeffs):
        # Get species name if we have an index
        if isinstance(sp, (int, np.integer)):
            sp_name = thermo_sys.obigt.loc[int(sp)]['name']
        else:
            sp_name = sp

        # Replace 'water' with 'H2O' for EQ3 compatibility
        if sp_name == 'water':
            sp_name = 'H2O'

        parts.append(f"{coeff:.4f}")
        parts.append(sp_name)

    return " ".join(parts)


__all__ = ['balance_reaction', 'format_reaction']
