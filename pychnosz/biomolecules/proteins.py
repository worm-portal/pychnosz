"""
Protein functions for CHNOSZ.

This module implements protein-related functions from CHNOSZ including
add_protein, protein_length, protein_formula, protein_OBIGT, and protein_basis.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List
import warnings

from ..core.thermo import thermo
from ..utils.formula import i2A, as_chemical_formula, species_basis
from ..biomolecules.ionize_aa import ionize_aa


def pinfo(protein: Union[str, int, pd.DataFrame, List],
          organism: Optional[str] = None,
          residue: bool = False,
          regexp: bool = False) -> Union[pd.DataFrame, np.ndarray, int]:
    """
    Get protein information from thermo().protein.

    This function retrieves protein data from the thermodynamic database.
    The behavior depends on the input type:
    - DataFrame: returns the DataFrame (possibly per residue)
    - int or list of ints: returns rows from thermo().protein
    - str: searches for protein by name, returns row number(s)

    Parameters
    ----------
    protein : str, int, DataFrame, or list
        Protein identifier(s) or data
    organism : str, optional
        Organism identifier (used with protein name)
    residue : bool, default False
        Return per-residue amino acid composition
    regexp : bool, default False
        Use regular expression matching for protein search

    Returns
    -------
    DataFrame, array, or int
        Protein information or row numbers

    Examples
    --------
    >>> # Get protein by name
    >>> iprotein = pinfo("LYSC_CHICK")
    >>> # Get protein data by row number
    >>> protein_data = pinfo(iprotein)
    """
    t_p = thermo().protein

    if t_p is None:
        raise RuntimeError("Protein database not loaded. Run reset() first.")

    # If input is a DataFrame, return it (possibly per residue)
    if isinstance(protein, pd.DataFrame):
        out = protein.copy()
        if residue:
            # Normalize by total amino acid count (columns 5:25)
            row_sums = out.iloc[:, 5:25].sum(axis=1)
            out.iloc[:, 4:24] = out.iloc[:, 4:24].div(row_sums, axis=0)
        return out

    # If input is numeric, get rows from thermo().protein
    if isinstance(protein, (int, np.integer)):
        protein = [protein]

    if isinstance(protein, (list, np.ndarray)) and all(isinstance(x, (int, np.integer)) for x in protein):
        # Get amino acid counts
        iproteins = list(range(len(t_p)))
        # Replace invalid indices with NaN
        protein_clean = [p if p in iproteins else np.nan for p in protein]
        # Filter out NaN values for indexing
        valid_indices = [p for p in protein_clean if not np.isnan(p)]

        if not valid_indices:
            return pd.DataFrame()

        out = t_p.iloc[valid_indices].copy()

        # Compute per-residue counts if requested
        if residue:
            row_sums = out.iloc[:, 5:25].sum(axis=1)
            out.iloc[:, 4:24] = out.iloc[:, 4:24].div(row_sums, axis=0)

        return out

    # If input is string or list of strings, search for protein
    if isinstance(protein, str):
        protein = [protein]

    if isinstance(protein, list) and all(isinstance(x, str) for x in protein):
        # Search for protein or protein_organism in thermo().protein
        t_p_names = t_p['protein'] + '_' + t_p['organism']

        if regexp:
            # Use regular expression matching
            matches = []
            for prot in protein:
                iprotein = t_p['protein'].str.contains(prot, regex=True, na=False)
                if organism is not None:
                    iorganism = t_p['organism'].str.contains(organism, regex=True, na=False)
                    iprotein = iprotein & iorganism
                indices = np.where(iprotein)[0]
                if len(indices) > 0:
                    matches.extend(indices.tolist())
                else:
                    matches.append(np.nan)

            if len(matches) == 1:
                if np.isnan(matches[0]):
                    return np.nan
                return int(matches[0])
            return np.array(matches)
        else:
            # Exact matching
            if organism is None:
                my_names = protein
            else:
                my_names = [f"{p}_{organism}" for p in protein]

            # Find matches
            matches = []
            for name in my_names:
                idx = np.where(t_p_names == name)[0]
                if len(idx) > 0:
                    matches.append(idx[0])
                else:
                    matches.append(np.nan)

            if len(matches) == 1:
                if np.isnan(matches[0]):
                    return np.nan
                return int(matches[0])
            return np.array(matches)

    raise TypeError(f"Unsupported protein type: {type(protein)}")


def add_protein(aa: pd.DataFrame, as_residue: bool = False) -> np.ndarray:
    """
    Add protein amino acid compositions to thermo().protein.

    Parameters
    ----------
    aa : DataFrame
        DataFrame with protein amino acid compositions.
        Must have same columns as thermo().protein
    as_residue : bool, default False
        Normalize amino acid counts by protein length

    Returns
    -------
    array
        Row numbers of added/updated proteins in thermo().protein

    Examples
    --------
    >>> import pandas as pd
    >>> from pychnosz import *
    >>> aa = pd.read_csv("POLG.csv")
    >>> iprotein = add_protein(aa)
    """
    t = thermo()

    if t.protein is None:
        raise RuntimeError("Protein database not loaded. Run reset() first.")

    # Check that columns match
    if list(aa.columns) != list(t.protein.columns):
        raise ValueError("'aa' does not have the same columns as thermo().protein")

    # Check that new protein IDs are unique
    po = aa['protein'] + '_' + aa['organism']
    idup = po.duplicated()
    if idup.any():
        dup_proteins = po[idup].unique()
        raise ValueError(f"some protein IDs are duplicated: {' '.join(dup_proteins)}")

    # Normalize by protein length if as_residue = True
    if as_residue:
        pl = protein_length(aa)
        aa.iloc[:, 4:24] = aa.iloc[:, 4:24].div(pl, axis=0)

    # Find any protein IDs that are already present
    ip = pinfo(po.tolist())
    if isinstance(ip, (int, np.integer)):
        ip = np.array([ip])
    elif not isinstance(ip, np.ndarray):
        ip = np.array([ip])

    ip_present = ~np.isnan(ip)

    # Now we're ready to go
    tp_new = t.protein.copy()

    # Add new proteins
    if not all(ip_present):
        new_proteins = aa[~ip_present].copy()
        tp_new = pd.concat([tp_new, new_proteins], ignore_index=True)

    # Update existing proteins
    if any(ip_present):
        valid_ip = ip[ip_present].astype(int)
        tp_new.iloc[valid_ip] = aa[ip_present].values

    # Update the protein database
    tp_new.reset_index(drop=True, inplace=True)
    t.protein = tp_new

    # Return the new row numbers
    ip_new = pinfo(po.tolist())
    if isinstance(ip_new, (int, np.integer)):
        ip_new = np.array([ip_new])

    # Print messages
    n_added = sum(~ip_present)
    n_replaced = sum(ip_present)

    if n_added > 0:
        print(f"add_protein: added {n_added} new protein(s) to thermo().protein")
    if n_replaced > 0:
        print(f"add_protein: replaced {n_replaced} existing protein(s) in thermo().protein")

    return ip_new


def protein_length(protein: Union[int, List[int], pd.DataFrame],
                   organism: Optional[str] = None) -> Union[int, np.ndarray]:
    """
    Calculate the length(s) of proteins.

    Parameters
    ----------
    protein : int, list of int, or DataFrame
        Protein identifier(s) or amino acid composition data
    organism : str, optional
        Organism identifier (used with protein number)

    Returns
    -------
    int or array
        Protein length(s) in amino acid residues

    Examples
    --------
    >>> iprotein = pinfo("LYSC_CHICK")
    >>> length = protein_length(iprotein)
    """
    # Get amino acid composition
    aa = pinfo(pinfo(protein, organism))

    if isinstance(aa, pd.DataFrame):
        # Use sum on the columns containing amino acid counts (columns 5:25)
        pl = aa.iloc[:, 5:25].sum(axis=1).values
        return pl
    else:
        return 0


def group_formulas() -> pd.DataFrame:
    """
    Return chemical formulas of amino acid residues.

    This function returns a DataFrame with the chemical formulas of
    H2O, the 20 amino acid sidechain groups, and the unfolded protein
    backbone group [UPBB].

    Returns
    -------
    DataFrame
        Chemical formulas with elements C, H, N, O, S as columns
        and residues as rows
    """
    # Chemical formulas as a numpy array
    # Rows: water, [Ala], [Cys], [Asp], [Glu], [Phe], [Gly], [His], [Ile], [Lys], [Leu],
    #       [Met], [Asn], [Pro], [Gln], [Arg], [Ser], [Thr], [Val], [Trp], [Tyr], [UPBB]
    # Columns: C, H, N, O, S
    A = np.array([
        [0, 2, 0, 1, 0],      # H2O
        [1, 3, 0, 0, 0],      # [Ala]
        [1, 3, 0, 0, 1],      # [Cys]
        [2, 3, 0, 2, 0],      # [Asp]
        [3, 5, 0, 2, 0],      # [Glu]
        [7, 7, 0, 0, 0],      # [Phe]
        [0, 1, 0, 0, 0],      # [Gly]
        [4, 5, 2, 0, 0],      # [His]
        [4, 9, 0, 0, 0],      # [Ile]
        [4, 10, 1, 0, 0],     # [Lys]
        [4, 9, 0, 0, 0],      # [Leu]
        [3, 7, 0, 0, 1],      # [Met]
        [2, 4, 1, 1, 0],      # [Asn]
        [3, 5, 0, 0, 0],      # [Pro]
        [3, 6, 1, 1, 0],      # [Gln]
        [4, 10, 3, 0, 0],     # [Arg]
        [1, 3, 0, 1, 0],      # [Ser]
        [2, 5, 0, 1, 0],      # [Thr]
        [3, 7, 0, 0, 0],      # [Val]
        [9, 8, 1, 0, 0],      # [Trp]
        [7, 7, 0, 1, 0],      # [Tyr]
        [2, 2, 1, 1, 0]       # [UPBB]
    ])

    rownames = ['H2O', '[Ala]', '[Cys]', '[Asp]', '[Glu]', '[Phe]', '[Gly]',
                '[His]', '[Ile]', '[Lys]', '[Leu]', '[Met]', '[Asn]', '[Pro]',
                '[Gln]', '[Arg]', '[Ser]', '[Thr]', '[Val]', '[Trp]', '[Tyr]',
                '[UPBB]']

    # Add [UPBB] to the sidechain groups to get residues
    out = A.copy()
    # Add [UPBB] (last row) to each sidechain group (rows 1-20)
    out[1:21, :] = out[1:21, :] + A[21, :]

    # Create DataFrame
    df = pd.DataFrame(out[0:21, :],
                     index=rownames[0:21],
                     columns=['C', 'H', 'N', 'O', 'S'])

    return df


def protein_formula(protein: Union[int, List[int], pd.DataFrame],
                   organism: Optional[str] = None,
                   residue: bool = False) -> pd.DataFrame:
    """
    Calculate chemical formulas of proteins.

    Parameters
    ----------
    protein : int, list of int, or DataFrame
        Protein identifier(s) or amino acid composition data
    organism : str, optional
        Organism identifier (used with protein number)
    residue : bool, default False
        Return per-residue formula

    Returns
    -------
    DataFrame
        Chemical formulas with elements C, H, N, O, S as columns

    Examples
    --------
    >>> iprotein = pinfo("LYSC_CHICK")
    >>> formula = protein_formula(iprotein)
    """
    # Get amino acid composition
    aa = pinfo(pinfo(protein, organism))

    if not isinstance(aa, pd.DataFrame):
        raise TypeError("Could not retrieve protein data")

    # Get group formulas
    rf = group_formulas()

    # Matrix multiplication: amino acid counts * residue formulas
    # Columns 5:25 contain amino acid counts (excluding chains column at 4)
    # We need to add H2O (chains column) separately
    aa_counts = aa.iloc[:, 5:25].values.astype(float)
    chains = aa.iloc[:, 4].values.astype(float)
    rf_values = rf.iloc[1:, :].values.astype(float)  # Skip H2O row, use amino acid residues
    rf_H2O = rf.iloc[0, :].values.astype(float)  # H2O row

    # Calculate protein formula: amino acids + H2O for chains
    out = np.dot(aa_counts, rf_values) + np.outer(chains, rf_H2O)

    # Normalize by residue if requested
    if residue:
        row_sums = aa.iloc[:, 5:25].sum(axis=1).values
        out = out / row_sums[:, np.newaxis]

    # Create DataFrame with protein names as index
    protein_names = aa['protein'] + '_' + aa['organism']
    # Make names unique if there are duplicates
    if protein_names.duplicated().any():
        counts = {}
        unique_names = []
        for name in protein_names:
            if name in counts:
                counts[name] += 1
                unique_names.append(f"{name}.{counts[name]}")
            else:
                counts[name] = 0
                unique_names.append(name)
        protein_names = unique_names

    result = pd.DataFrame(out,
                         index=protein_names,
                         columns=['C', 'H', 'N', 'O', 'S'])

    return result


def protein_OBIGT(protein: Union[int, List[int], pd.DataFrame],
                 organism: Optional[str] = None,
                 state: Optional[str] = None) -> pd.DataFrame:
    """
    Calculate protein properties using group additivity.

    This function calculates thermodynamic properties of proteins
    from amino acid composition using the group additivity approach.

    Parameters
    ----------
    protein : int, list of int, or DataFrame
        Protein identifier(s) or amino acid composition data
    organism : str, optional
        Organism identifier
    state : str, optional
        Physical state ('aq' or 'cr'). If None, uses thermo().opt['state']

    Returns
    -------
    DataFrame
        Thermodynamic properties in OBIGT format

    Examples
    --------
    >>> iprotein = pinfo("LYSC_CHICK")
    >>> props = protein_OBIGT(iprotein)
    """
    # Get amino acid composition
    aa = pinfo(pinfo(protein, organism))

    if not isinstance(aa, pd.DataFrame):
        raise TypeError("Could not retrieve protein data")

    # Get state
    if state is None:
        state = thermo().opt.get('state', 'aq')

    # The names of the protein backbone groups depend on the state
    # [UPBB] for aq or [PBB] for cr
    if state == 'aq':
        bbgroup = 'UPBB'
    else:
        bbgroup = 'PBB'

    # Names of the AABB, sidechain and protein backbone groups
    aa_cols = aa.columns[5:25].tolist()  # Get amino acid column names
    groups = ['AABB'] + aa_cols + [bbgroup]

    # Put brackets around the group names
    groups = [f"[{g}]" for g in groups]

    # The row numbers of the groups in thermo().OBIGT
    from ..core.info import info

    groups_state = [f"{g}" for g in groups]
    obigt = thermo().obigt

    # Find groups in OBIGT
    igroup = []
    for group_name in groups_state:
        # Search for the group with the specified state
        matches = obigt[(obigt['name'] == group_name) & (obigt['state'] == state)]
        if len(matches) > 0:
            igroup.append(matches.index[0])
        else:
            # Try without brackets if not found
            group_alt = group_name.strip('[]')
            matches = obigt[(obigt['name'] == group_alt) & (obigt['state'] == state)]
            if len(matches) > 0:
                igroup.append(matches.index[0])
            else:
                raise ValueError(f"Group {group_name} not found in OBIGT for state {state}")

    # The properties are in columns 9:21 of thermo().OBIGT (G, H, S, Cp, V, etc.)
    # Column indices: G=9, H=10, S=11, Cp=12, V=13, a1.a=14, a2.b=15, a3.c=16, a4.d=17, c1.e=18, c2.f=19, omega.lambda=20, z.T=21
    groupprops = obigt.loc[igroup, obigt.columns[9:22]]

    # The elements in each of the groups
    groupelements = i2A(igroup)

    results = []

    # Process each protein
    for idx in range(len(aa)):
        aa_row = aa.iloc[idx]

        # Numbers of groups: chains [=AABB], sidechains, protein backbone
        nchains = float(aa_row.iloc[4])  # chains column
        length = float(aa_row.iloc[5:25].sum())  # sum of amino acids
        npbb = length - nchains

        # Create ngroups array
        ngroups = np.array([nchains] + aa_row.iloc[5:25].tolist() + [npbb], dtype=float)

        # Calculate thermodynamic properties by group additivity
        eos = (groupprops.values * ngroups[:, np.newaxis]).sum(axis=0)

        # Calculate formula
        f_in = (groupelements.values * ngroups[:, np.newaxis]).sum(axis=0).round(3)

        # Remove elements that don't appear
        element_names = groupelements.columns
        f_dict = {elem: f_in[i] for i, elem in enumerate(element_names) if f_in[i] != 0}

        # Turn it into a formula string
        f = as_chemical_formula(f_dict)

        # Species name
        name = f"{aa_row['protein']}_{aa_row['organism']}"

        # Print message
        print(f"protein_OBIGT: found {name} ({f}, {round(length, 3)} residues)")

        ref = aa_row['ref']

        # Include 'model' column
        model = 'HKF' if state == 'aq' else 'CGL'

        # Create header
        header = {
            'name': name,
            'abbrv': None,
            'formula': f,
            'state': state,
            'ref1': ref,
            'ref2': None,
            'date': None,
            'model': model,
            'E_units': 'cal'
        }

        # Combine header and eos
        eosout = {**header, **dict(zip(groupprops.columns, eos))}
        results.append(eosout)

    # Convert to DataFrame
    out = pd.DataFrame(results)
    out.reset_index(drop=True, inplace=True)

    return out


def protein_basis(protein: Union[int, List[int], pd.DataFrame],
                 T: float = 25.0,
                 normalize: bool = False) -> pd.DataFrame:
    """
    Calculate coefficients of basis species in protein formation reactions.

    Parameters
    ----------
    protein : int, list of int, or DataFrame
        Protein identifier(s) or amino acid composition data
    T : float, default 25.0
        Temperature in degrees Celsius
    normalize : bool, default False
        Normalize by protein length

    Returns
    -------
    DataFrame
        Coefficients of basis species

    Examples
    --------
    >>> from pychnosz import *
    >>> basis("CHNOSe")
    >>> iprotein = pinfo("LYSC_CHICK")
    >>> coeffs = protein_basis(iprotein)
    """
    # Get amino acid composition
    aa = pinfo(pinfo(protein))

    if not isinstance(aa, pd.DataFrame):
        raise TypeError("Could not retrieve protein data")

    # Get protein formulas
    pf = protein_formula(aa)

    # Calculate coefficients of basis species in formation reactions
    sb = species_basis(pf)

    # Calculate ionization states if H+ is a basis species
    t = thermo()
    if t.basis is not None:
        basis_species = t.basis.index.tolist()
        if 'H+' in basis_species:
            iHplus = basis_species.index('H+')
            pH = -t.basis.loc['H+', 'logact']
            Z = ionize_aa(aa, T=T, pH=pH).iloc[0, :]
            sb.iloc[:, iHplus] = sb.iloc[:, iHplus] + Z.values

    # Normalize by length if requested
    if normalize:
        plen = protein_length(aa)
        sb = sb.div(plen, axis=0)

    return sb
