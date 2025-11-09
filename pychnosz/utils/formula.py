"""
Chemical formula parsing and manipulation utilities.

This module provides Python equivalents of the R functions in makeup.R and util.formula.R:
- makeup(): Parse chemical formulas and return elemental composition
- Formula validation and parsing
- Molecular weight and entropy calculations
- Stoichiometric matrix operations

Author: CHNOSZ Python port
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
import re
import warnings

from ..core.thermo import thermo


class FormulaError(Exception):
    """Exception raised for formula parsing errors."""
    pass


def makeup(formula: Union[str, int, List[Union[str, int]]], 
           multiplier: Union[float, List[float]] = 1.0,
           sum_formulas: bool = False,
           count_zero: bool = False) -> Union[Dict[str, float], List[Dict[str, float]]]:
    """
    Return elemental makeup (counts) of chemical formula(s).
    
    Handles formulas with parenthetical subformulas, suffixed formulas,
    charges, and fractional coefficients.
    
    Parameters
    ----------
    formula : str, int, or list
        Chemical formula(s) or species index(es)
    multiplier : float or list of float
        Multiplier(s) to apply to formula coefficients
    sum_formulas : bool
        If True, return sum of all formulas
    count_zero : bool
        If True, include zero counts for all elements appearing in any formula
        
    Returns
    -------
    dict or list of dict
        Elemental composition(s) as {element: count} dictionaries
        
    Examples
    --------
    >>> makeup("H2O")
    {'H': 2, 'O': 1}
    
    >>> makeup("Ca(OH)2")
    {'Ca': 1, 'O': 2, 'H': 2}
    
    >>> makeup(["H2O", "CO2"])
    [{'H': 2, 'O': 1}, {'C': 1, 'O': 2}]
    """
    # Handle matrix input
    if isinstance(formula, np.ndarray) and formula.ndim == 2:
        return [makeup(formula[i, :]) for i in range(formula.shape[0])]
    
    # Handle named numeric objects (return unchanged)
    if isinstance(formula, dict) and all(isinstance(k, str) for k in formula.keys()):
        return formula
    
    # Handle list of named objects
    if isinstance(formula, list) and len(formula) > 0:
        if isinstance(formula[0], dict) and all(isinstance(k, str) for k in formula[0].keys()):
            return formula
    
    # Prepare multiplier
    if not isinstance(multiplier, list):
        multiplier = [multiplier]
    
    # Handle multiple formulas
    if isinstance(formula, list):
        if len(multiplier) != 1 and len(multiplier) != len(formula):
            raise ValueError("multiplier does not have length = 1 or length = number of formulas")
        
        if len(multiplier) == 1:
            multiplier = multiplier * len(formula)
        
        # Get formulas for any species indices
        formula = get_formula(formula)
        
        results = []
        for i, f in enumerate(formula):
            result = makeup(f, multiplier[i])
            results.append(result)
        
        # Handle sum_formulas option
        if sum_formulas:
            all_elements = set()
            for result in results:
                if result is not None:
                    all_elements.update(result.keys())
            
            summed = {}
            for element in all_elements:
                summed[element] = sum(result.get(element, 0) for result in results if result is not None)
            return summed
        
        # Handle count_zero option
        elif count_zero:
            # Get all elements appearing in any formula
            all_elements = set()
            for result in results:
                if result is not None:
                    all_elements.update(result.keys())
            
            # Add zero counts for missing elements
            complete_results = []
            for result in results:
                if result is None:
                    complete_result = {element: np.nan for element in all_elements}
                else:
                    complete_result = {element: result.get(element, 0) for element in all_elements}
                complete_results.append(complete_result)
            
            return complete_results
        
        return results
    
    # Handle single formula
    if isinstance(formula, int):
        # Get formula from species index
        thermo_obj = thermo()
        if thermo_obj.obigt is not None:
            # Use .loc for label-based indexing (species indices are 1-based labels)
            if formula in thermo_obj.obigt.index:
                formula = thermo_obj.obigt.loc[formula, 'formula']
            else:
                raise FormulaError(f"Species index {formula} not found in OBIGT database")
        else:
            raise FormulaError("Thermodynamic database not initialized")
    
    if formula is None or pd.isna(formula):
        return None
    
    # Parse single formula
    try:
        result = _parse_formula(str(formula))
        
        # Apply multiplier
        if multiplier[0] != 1.0:
            result = {element: count * multiplier[0] for element, count in result.items()}
        
        # Validate elements
        _validate_elements(result)
        
        return result
    
    except Exception as e:
        raise FormulaError(f"Error parsing formula '{formula}': {e}")


def _parse_formula(formula: str) -> Dict[str, float]:
    """Parse a single chemical formula string."""
    # Handle charge first
    charge_info = _count_charge(formula)
    uncharged_formula = charge_info['uncharged']
    charge = charge_info['Z']
    
    # Add explicit charge if present
    if charge != 0:
        uncharged_formula += f"Z{charge}"
    
    # Check for subformulas (parentheses, *, :)
    if re.search(r'[()*:]', uncharged_formula):
        return _parse_complex_formula(uncharged_formula)
    else:
        return _count_elements(uncharged_formula)


def _count_charge(formula: str) -> Dict[str, Any]:
    """Extract charge from formula."""
    Z = 0
    uncharged = formula
    
    # Look for charge at end: +, -, +n, -n
    charge_match = re.search(r'([+-])(\d*\.?\d*)$', formula)
    if charge_match:
        sign = 1 if charge_match.group(1) == '+' else -1
        magnitude_str = charge_match.group(2)
        
        if magnitude_str == '':
            magnitude = 1
        else:
            magnitude = float(magnitude_str)
        
        Z = sign * magnitude
        uncharged = formula[:charge_match.start()]
    
    return {'Z': Z, 'uncharged': uncharged}


def _count_elements(formula: str) -> Dict[str, float]:
    """Count elements in a simple chemical formula."""
    if pd.isna(formula) or formula == '':
        return {}
    
    # Regular expression for element symbol and coefficient
    element_pattern = r'([A-Z][a-z]*)([+-]?\d*\.?\d*)'
    
    # Validate formula format
    if not re.match(r'^(' + element_pattern + r')+$', formula):
        raise FormulaError(f"'{formula}' is not a simple chemical formula")
    
    elements = {}
    
    # Find all element-coefficient pairs
    matches = re.findall(element_pattern, formula)
    
    for element, coeff_str in matches:
        if coeff_str == '' or coeff_str == '+':
            coeff = 1.0
        elif coeff_str == '-':
            coeff = -1.0
        else:
            coeff = float(coeff_str)
        
        # Sum if element appears multiple times
        elements[element] = elements.get(element, 0) + coeff
    
    return elements


def _parse_complex_formula(formula: str) -> Dict[str, float]:
    """Parse formula with parentheses and/or suffixes."""
    subformulas = _count_formulas(formula)
    
    total_elements = {}
    
    for subformula, count in subformulas.items():
        if subformula:  # Skip empty subformulas
            sub_elements = _count_elements(subformula)
            
            # Add weighted contribution
            for element, element_count in sub_elements.items():
                total_elements[element] = total_elements.get(element, 0) + element_count * count
    
    return total_elements


def _count_formulas(formula: str) -> Dict[str, float]:
    """Count subformulas in a complex chemical formula."""
    subformulas = {}
    remaining = formula
    
    # Handle parenthetical terms: Ca(OH)2
    while '(' in remaining:
        # Find matching parentheses
        open_pos = remaining.find('(')
        if open_pos == -1:
            break
        
        close_pos = remaining.find(')', open_pos)
        if close_pos == -1:
            raise FormulaError("Unpaired parentheses in formula")
        
        # Extract subformula
        subformula = remaining[open_pos + 1:close_pos]
        
        # Look for coefficient after closing parenthesis
        after_close = remaining[close_pos + 1:]
        coeff_match = re.match(r'^([+-]?\d*\.?\d*)', after_close)
        
        if coeff_match and coeff_match.group(1):
            coeff_str = coeff_match.group(1)
            if coeff_str in ['+', '']:
                coeff = 1.0
            elif coeff_str == '-':
                coeff = -1.0
            else:
                coeff = float(coeff_str)
            coeff_end = coeff_match.end()
        else:
            coeff = 1.0
            coeff_end = 0
        
        # Add to subformulas
        subformulas[subformula] = subformulas.get(subformula, 0) + coeff
        
        # Remove processed part
        remaining = remaining[:open_pos] + remaining[close_pos + 1 + coeff_end:]
    
    # Handle suffixed terms: CaSO4*2H2O or CaSO4:2H2O
    for separator in ['*', ':']:
        if separator in remaining:
            parts = remaining.split(separator)
            main_part = parts[0]
            
            for i in range(1, len(parts)):
                suffix_part = parts[i]
                
                # Look for leading coefficient
                coeff_match = re.match(r'^([+-]?\d*\.?\d*)', suffix_part)
                if coeff_match and coeff_match.group(1):
                    coeff_str = coeff_match.group(1)
                    if coeff_str in ['+', '']:
                        coeff = 1.0
                    elif coeff_str == '-':
                        coeff = -1.0
                    else:
                        coeff = float(coeff_str)
                    subformula = suffix_part[coeff_match.end():]
                else:
                    coeff = 1.0
                    subformula = suffix_part
                
                if subformula:
                    subformulas[subformula] = subformulas.get(subformula, 0) + coeff
            
            remaining = main_part
            break
    
    # Add remaining main formula
    if remaining.strip():
        subformulas[remaining.strip()] = subformulas.get(remaining.strip(), 0) + 1
    
    return subformulas


def _validate_elements(composition: Dict[str, float]) -> None:
    """Validate that elements exist in the thermodynamic database."""
    thermo_obj = thermo()
    if thermo_obj.element is not None:
        known_elements = set(thermo_obj.element['element'].tolist())
        unknown_elements = set(composition.keys()) - known_elements - {'Z'}
        
        if unknown_elements:
            warnings.warn(f"element(s) not in thermo().element: {' '.join(unknown_elements)}")


def get_formula(formula: Union[str, int, List[Union[str, int]]]) -> Union[str, List[str]]:
    """
    Get chemical formulas for species indices or return formula strings.
    
    Parameters
    ----------
    formula : str, int, or list
        Chemical formula(s) or species index(es)
        
    Returns
    -------
    str or list of str
        Chemical formula(s)
    """
    # Handle single values
    if not isinstance(formula, list):
        formula = [formula]
        single_result = True
    else:
        single_result = False
    
    results = []
    thermo_obj = thermo()
    
    for f in formula:
        if isinstance(f, str):
            # Already a formula
            results.append(f)
        elif isinstance(f, int):
            # Species index - look up formula
            if thermo_obj.obigt is not None:
                # Use .loc for label-based indexing (species indices are 1-based labels)
                if f in thermo_obj.obigt.index:
                    formula_str = thermo_obj.obigt.loc[f, 'formula']
                    results.append(formula_str)
                else:
                    raise FormulaError(f"Species index {f} not found in OBIGT database")
            else:
                raise FormulaError("Thermodynamic database not initialized")
        else:
            # Try to convert to string
            results.append(str(f))
    
    if single_result:
        return results[0]
    else:
        return results


def as_chemical_formula(makeup_dict: Union[Dict[str, float], pd.DataFrame], 
                       drop_zero: bool = True) -> Union[str, List[str]]:
    """
    Convert elemental makeup to chemical formula string(s).
    
    Parameters
    ----------
    makeup_dict : dict or DataFrame
        Elemental composition(s)
    drop_zero : bool
        Whether to exclude zero coefficients
        
    Returns
    -------
    str or list of str
        Chemical formula string(s)
    """
    if isinstance(makeup_dict, pd.DataFrame):
        # Handle matrix of compositions
        results = []
        for i in range(len(makeup_dict)):
            row_dict = makeup_dict.iloc[i].to_dict()
            formula = _dict_to_formula(row_dict, drop_zero)
            results.append(formula)
        return results
    else:
        # Handle single composition
        return _dict_to_formula(makeup_dict, drop_zero)


def _dict_to_formula(composition: Dict[str, float], drop_zero: bool) -> str:
    """Convert single composition dictionary to formula string."""
    if drop_zero:
        composition = {k: v for k, v in composition.items() if v != 0}
    
    # Put Z (charge) at the end
    elements = [k for k in composition.keys() if k != 'Z']
    if 'Z' in composition:
        elements.append('Z')
    
    formula_parts = []
    
    for element in elements:
        count = composition[element]
        
        if element == 'Z':
            # Handle charge
            if count < 0:
                formula_parts.append(f"{count}")
            elif count > 0:
                formula_parts.append(f"+{count}")
            # count == 0 is omitted
        else:
            # Handle regular elements
            if count == 1:
                formula_parts.append(element)
            elif count == -1:
                formula_parts.append(f"{element}-1")
            else:
                formula_parts.append(f"{element}{count}")
    
    formula = ''.join(formula_parts)
    
    # Handle special case of negative coefficient at end without charge
    if 'Z' not in composition and len(elements) > 0:
        last_element = elements[-1]
        if composition[last_element] < 0:
            formula += "+0"
    
    return formula


def mass(formula: Union[str, int, List[Union[str, int]]]) -> Union[float, List[float]]:
    """
    Calculate molecular mass of chemical formula(s).
    
    Parameters
    ----------
    formula : str, int, or list
        Chemical formula(s) or species index(es)
        
    Returns
    -------
    float or list of float
        Molecular mass(es) in g/mol
    """
    thermo_obj = thermo()
    if thermo_obj.element is None:
        raise RuntimeError("Element data not available")
    
    # Convert to stoichiometric matrix
    compositions = makeup(formula, count_zero=False)
    if not isinstance(compositions, list):
        compositions = [compositions]
    
    masses = []
    
    for comp in compositions:
        if comp is None:
            masses.append(np.nan)
            continue
        
        total_mass = 0.0
        for element, count in comp.items():
            if element == 'Z':
                continue  # Charge has no mass
            
            # Look up element mass
            element_data = thermo_obj.element[thermo_obj.element['element'] == element]
            if len(element_data) == 0:
                raise FormulaError(f"Element {element} not found in element database")
            
            element_mass = element_data.iloc[0]['mass']
            total_mass += count * element_mass
        
        masses.append(total_mass)
    
    if len(masses) == 1:
        return masses[0]
    else:
        return masses


def entropy(formula: Union[str, int, List[Union[str, int]]]) -> Union[float, List[float]]:
    """
    Calculate standard molal entropy of elements in chemical formulas.
    
    Parameters
    ----------
    formula : str, int, or list
        Chemical formula(s) or species index(es)
        
    Returns
    -------
    float or list of float
        Standard entropy(ies) in J/(mol*K)
    """
    thermo_obj = thermo()
    if thermo_obj.element is None:
        raise RuntimeError("Element data not available")
    
    # Convert to stoichiometric matrix
    compositions = makeup(formula, count_zero=False)
    if not isinstance(compositions, list):
        compositions = [compositions]
    
    entropies = []
    
    for comp in compositions:
        if comp is None:
            entropies.append(np.nan)
            continue
        
        total_entropy = 0.0
        has_na = False
        
        for element, count in comp.items():
            
            # Look up element entropy
            element_data = thermo_obj.element[thermo_obj.element['element'] == element]
            if len(element_data) == 0:
                warnings.warn(f"Element {element} not available in thermo().element")
                has_na = True
                continue
            
            element_s = element_data.iloc[0]['s']
            element_n = element_data.iloc[0]['n']
            
            if pd.isna(element_s) or pd.isna(element_n):
                has_na = True
                continue
            
            # Entropy per atom
            entropy_per_atom = element_s / element_n
            total_entropy += count * entropy_per_atom
        
        if has_na and total_entropy == 0:
            entropies.append(np.nan)
        else:
            # Convert to Joules (assuming input is in cal)
            entropies.append(total_entropy * 4.184)  # cal to J conversion
    
    if len(entropies) == 1:
        return entropies[0]
    else:
        return entropies


def species_basis(species: Union[List[int], np.ndarray],
                  makeup_matrix: Optional[np.ndarray] = None,
                  basis_df: Optional[pd.DataFrame] = None) -> np.ndarray:
    """
    Calculate coefficients for formation reactions from basis species.

    Parameters
    ----------
    species : list of int or array
        Species indices in thermo().obigt
    makeup_matrix : array, optional
        Pre-calculated makeup matrix
    basis_df : pd.DataFrame, optional
        Basis definition to use (if not using global basis)

    Returns
    -------
    np.ndarray
        Formation reaction coefficients matrix
    """
    from ..core.basis import basis_elements, get_basis

    # Follow R CHNOSZ species.basis algorithm exactly
    from ..core.thermo import thermo

    # Get basis dataframe
    if basis_df is None:
        basis_df = get_basis()
        if basis_df is None:
            raise RuntimeError("Basis species not defined")

    # Get basis element names
    basis_element_names = [col for col in basis_df.columns
                          if col not in ['ispecies', 'logact', 'state']]

    # Calculate basis elements matrix from basis_df
    element_cols = [col for col in basis_df.columns
                   if col not in ['ispecies', 'logact', 'state']]
    bmat = basis_df[element_cols].values.T

    # basis_elements() already returns transposed matrix (equivalent to R tbmat)
    tbmat = bmat

    # Get thermo object for species lookup
    thermo_obj = thermo()

    # Initialize result matrix
    n_species = len(species)
    n_basis = len(basis_element_names)
    formation_coeffs = np.zeros((n_species, n_basis))

    # Process each species individually (following R apply logic)
    for i, sp_idx in enumerate(species):
        # Get species makeup (equivalent to R mkp <- as.matrix(sapply(makeup(species), c)))
        formula = thermo_obj.obigt.iloc[sp_idx - 1]['formula']
        sp_makeup = makeup([formula], count_zero=True)[0]

        # Convert makeup to array ordered by elements present in species
        sp_elements = list(sp_makeup.keys())
        sp_values = np.array(list(sp_makeup.values()))

        # Find positions of species elements in basis elements (R ielem <- match)
        # All species elements must be in basis
        missing_elements = []
        for elem in sp_elements:
            if elem not in basis_element_names:
                missing_elements.append(elem)
        if missing_elements:
            raise RuntimeError(f"element(s) not in the basis: {' '.join(missing_elements)}")

        # Find positions of basis elements in species elements (R jelem <- match)
        jelem = []
        for elem in basis_element_names:
            try:
                jelem.append(sp_elements.index(elem))
            except ValueError:
                jelem.append(None)  # NA in R

        # Reorder species matrix to match basis elements (R mkp <- mkp[jelem, , drop = FALSE])
        sp_makeup_ordered = np.zeros(len(basis_element_names))
        for j, pos in enumerate(jelem):
            if pos is not None:
                sp_makeup_ordered[j] = sp_values[pos]
            # else remains 0 (equivalent to R mkp[ina, ] <- 0)

        # Solve linear system: tbmat @ coeffs = sp_makeup_ordered
        # This is equivalent to R solve(tbmat, x)
        try:
            coeffs = np.linalg.solve(tbmat, sp_makeup_ordered)
        except np.linalg.LinAlgError:
            raise RuntimeError(f"Singular basis matrix for species {sp_idx}")

        # Apply R zapsmall equivalent (digits=7)
        coeffs = np.around(coeffs, decimals=7)

        # Clean up very small numbers
        coeffs[np.abs(coeffs) < 1e-7] = 0

        formation_coeffs[i, :] = coeffs

    return formation_coeffs


def calculate_ghs(formula: str, G: float = np.nan, H: float = np.nan, 
                 S: float = np.nan, T: float = 298.15, 
                 E_units: str = "J") -> Dict[str, float]:
    """
    Calculate missing G, H, or S from the other two values.
    
    Parameters
    ----------
    formula : str
        Chemical formula
    G : float
        Gibbs energy of formation
    H : float
        Enthalpy of formation  
    S : float
        Standard entropy
    T : float
        Temperature in K
    E_units : str
        Energy units ("J" or "cal")
        
    Returns
    -------
    dict
        Dictionary with G, H, S values
    """
    # Calculate elemental entropy
    Se = entropy(formula)
    if E_units == "cal":
        Se = Se / 4.184  # Convert J to cal
    
    # Calculate missing value
    if pd.isna(G):
        G = H - T * (S - Se)
    elif pd.isna(H):
        H = G + T * (S - Se)
    elif pd.isna(S):
        S = (H - G) / T + Se
    
    return {"G": G, "H": H, "S": S}


def ZC(formula: Union[str, int, List[Union[str, int]]]) -> Union[float, List[float]]:
    """
    Calculate average oxidation state of carbon in chemical formulas.
    
    Parameters
    ----------
    formula : str, int, or list
        Chemical formula(s) or species index(es)
        
    Returns
    -------
    float or list of float
        Average oxidation state(s) of carbon
    """
    # Get elemental compositions
    compositions = makeup(formula, count_zero=False)
    if not isinstance(compositions, list):
        compositions = [compositions]
    
    results = []
    
    # Nominal charges of elements
    known_elements = ['H', 'N', 'O', 'S', 'Z']
    charges = [-1, 3, 2, 2, 1]
    
    for comp in compositions:
        if comp is None or 'C' not in comp:
            results.append(np.nan)
            continue
        
        # Calculate total charge from known elements
        total_charge = 0
        unknown_elements = []
        
        for element, count in comp.items():
            if element == 'C':
                continue
            elif element in known_elements:
                idx = known_elements.index(element)
                total_charge += count * charges[idx]
            else:
                unknown_elements.append(element)
        
        if unknown_elements:
            warnings.warn(f"element(s) {' '.join(unknown_elements)} not in "
                         f"{' '.join(known_elements)} so not included in ZC calculation")
        
        # Calculate carbon oxidation state
        n_carbon = comp['C']
        zc = total_charge / n_carbon
        results.append(zc)
    
    if len(results) == 1:
        return results[0]
    else:
        return results


# Convenience functions for stoichiometric operations
def i2A(formula: Union[str, List[str], Dict[str, float]]) -> np.ndarray:
    """
    Convert formula(s) to stoichiometric matrix.
    
    Parameters
    ----------
    formula : str, list, or dict
        Chemical formula(s) or composition
        
    Returns
    -------
    np.ndarray
        Stoichiometric matrix with elements as columns
    """
    if isinstance(formula, np.ndarray):
        return formula
    elif isinstance(formula, dict) and all(isinstance(k, str) for k in formula.keys()):
        # Single composition dictionary
        return np.array([[formula.get(k, 0) for k in sorted(formula.keys())]])
    
    # Get compositions with zero padding
    compositions = makeup(formula, count_zero=True)
    if not isinstance(compositions, list):
        compositions = [compositions]
    
    # Get all elements
    all_elements = set()
    for comp in compositions:
        if comp is not None:
            all_elements.update(comp.keys())
    
    all_elements = sorted(list(all_elements))
    
    # Build matrix
    matrix = np.zeros((len(compositions), len(all_elements)))
    for i, comp in enumerate(compositions):
        if comp is not None:
            for j, element in enumerate(all_elements):
                matrix[i, j] = comp.get(element, 0)
    
    return matrix


# Export main functions
__all__ = [
    'makeup', 'get_formula', 'as_chemical_formula',
    'mass', 'entropy', 'species_basis', 'calculate_ghs', 'ZC', 'i2A',
    'FormulaError'
]