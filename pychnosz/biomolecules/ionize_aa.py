"""
Amino acid ionization calculations for CHNOSZ.

This module calculates ionization properties of proteins based on
amino acid composition.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional

from ..core.thermo import thermo
from ..core.subcrt import subcrt
from ..core.info import info
from ..utils.units import convert


def ionize_aa(aa: pd.DataFrame,
              property: str = "Z",
              T: Union[float, np.ndarray] = 25.0,
              P: Union[float, str, np.ndarray] = "Psat",
              pH: Union[float, np.ndarray] = 7.0,
              ret_val: Optional[str] = None,
              suppress_Cys: bool = False) -> pd.DataFrame:
    """
    Calculate additive ionization properties of proteins.

    This function calculates the net charge or other ionization properties
    of proteins based on amino acid composition at specified T, P, and pH.

    Parameters
    ----------
    aa : DataFrame
        Amino acid composition data
    property : str, default "Z"
        Property to calculate:
        - "Z": net charge
        - "A": chemical affinity
        - Other subcrt properties (G, H, S, Cp, V)
    T : float or array, default 25.0
        Temperature in degrees Celsius
    P : float, str, or array, default "Psat"
        Pressure in bar, or "Psat" for saturation
    pH : float or array, default 7.0
        pH value(s)
    ret_val : str, optional
        Return value type:
        - "pK": return pK values
        - "alpha": return degree of formation
        - "aavals": return amino acid values
        - None: return ionization property (default)
    suppress_Cys : bool, default False
        Suppress cysteine ionization

    Returns
    -------
    DataFrame
        Ionization properties

    Examples
    --------
    >>> from pychnosz import *
    >>> aa = pinfo(pinfo("LYSC_CHICK"))
    >>> Z = ionize_aa(aa, pH=7.0)
    """
    # Ensure inputs are arrays
    T = np.atleast_1d(T)
    if isinstance(P, str):
        P = np.array([P] * len(T))
    else:
        P = np.atleast_1d(P)
    pH_arr = np.atleast_1d(pH)

    # Get maximum length and replicate arrays
    lmax = max(len(T), len(P), len(pH_arr))
    T = np.resize(T, lmax)
    if isinstance(P[0], str):
        P = np.array([P[0]] * lmax)
    else:
        P = np.resize(P, lmax)
    pH_arr = np.resize(pH_arr, lmax)

    # Turn pH into a matrix with as many columns as ionizable groups (9)
    pH_matrix = np.tile(pH_arr[:, np.newaxis], (1, 9))

    # Charges for ionizable groups
    charges = np.array([-1, -1, -1, 1, 1, 1, -1, 1, -1])
    charges_matrix = np.tile(charges, (lmax, 1))

    # The ionizable groups
    neutral = ["[Cys]", "[Asp]", "[Glu]", "[His]", "[Lys]", "[Arg]", "[Tyr]", "[AABB]", "[AABB]"]
    charged = ["[Cys-]", "[Asp-]", "[Glu-]", "[His+]", "[Lys+]", "[Arg+]", "[Tyr-]", "[AABB+]", "[AABB-]"]

    # Get row numbers in OBIGT
    ineutral = [info(g, "aq") for g in neutral]
    icharged = [info(g, "aq") for g in charged]

    # Get unique T, P combinations
    pTP = [f"{t}_{p}" for t, p in zip(T, P)]
    unique_pTP = []
    seen = set()
    indices = []
    for i, tp in enumerate(pTP):
        if tp not in seen:
            unique_pTP.append(i)
            seen.add(tp)
        indices.append(list(seen).index(tp))

    # Determine which property to calculate
    sprop = ["G", property] if property not in ["A", "Z"] else ["G"]

    # Convert T to Kelvin for subcrt
    TK = convert(T, "K")

    # Call subcrt for unique T, P combinations
    unique_T = TK[unique_pTP]
    unique_P = P[unique_pTP]

    all_species = ineutral + icharged
    sout = subcrt(all_species, T=unique_T, P=unique_P, property=sprop, convert=False)

    # Extract G values
    Gs = np.zeros((len(unique_pTP), len(all_species)))
    for i, spec_idx in enumerate(all_species):
        if isinstance(sout['out'], dict):
            # Single species result
            Gs[:, i] = sout['out']['G']
        else:
            # Multiple species result
            Gs[:, i] = sout['out'][i]['G'].values

    # Gibbs energy difference for each group
    DG = Gs[:, 9:18] - Gs[:, 0:9]

    # Build matrix for all T, P values (including duplicates)
    DG_full = DG[indices, :]

    # Calculate pK values
    DG_full = DG_full * charges
    pK = np.zeros_like(DG_full)
    for i in range(pK.shape[1]):
        pK[:, i] = convert(DG_full[:, i], "logK", T=TK)

    # Return pK if requested
    if ret_val == "pK":
        return pd.DataFrame(pK, columns=charged)

    # Calculate alpha (degree of formation)
    alpha = 1 / (1 + 10 ** (charges_matrix * (pH_matrix - pK)))

    # Suppress cysteine ionization if requested
    if suppress_Cys:
        alpha[:, 0] = 0

    # Return alpha if requested
    if ret_val == "alpha":
        return pd.DataFrame(alpha, columns=charged)

    # Calculate amino acid values
    if property == "Z":
        aavals = charges_matrix.copy()
    elif property == "A":
        aavals = -charges_matrix * (pH_matrix - pK)
    else:
        # Extract property values from subcrt output
        prop_vals = np.zeros((len(unique_pTP), len(all_species)))
        for i, spec_idx in enumerate(all_species):
            if isinstance(sout['out'], dict):
                prop_vals[:, i] = sout['out'][property]
            else:
                prop_vals[:, i] = sout['out'][i][property].values

        # Build matrix for all T, P values
        prop_vals_full = prop_vals[indices, :]

        # Property difference for each group
        aavals = prop_vals_full[:, 9:18] - prop_vals_full[:, 0:9]

    # Return aavals if requested
    if ret_val == "aavals":
        return pd.DataFrame(aavals, columns=charged)

    # Contribution from each group
    aavals = aavals * alpha

    # Get counts of ionizable groups from aa
    # Columns: Cys, Asp, Glu, His, Lys, Arg, Tyr, chains, chains
    ionize_cols = ["Cys", "Asp", "Glu", "His", "Lys", "Arg", "Tyr", "chains", "chains"]
    aa_counts = aa[ionize_cols].values.astype(float)

    # Calculate total ionization property
    out = np.dot(aavals, aa_counts.T)

    # Create DataFrame
    result = pd.DataFrame(out)

    return result
