"""
Affinity calculation module.

This module provides Python equivalents of the R functions in affinity.R:
- affinity(): Calculate chemical affinities of formation reactions
- Energy calculation utilities and argument processing
- Variable expansion and multi-dimensional calculations

Author: CHNOSZ Python port
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Any, Tuple
import warnings

from .thermo import thermo
from .basis import get_basis, is_basis_defined
from .species import get_species, is_species_defined
from .subcrt import subcrt


class AffinityError(Exception):
    """Exception raised for affinity-related errors."""
    pass


# F / (ln(10) * R), so that pe = Eh * _PE_PER_VOLT_TIMES_T / T(K).
# R and F are the values used by R CHNOSZ convert(): R in kJ/(mol K), F in kJ/(V mol).
_PE_PER_VOLT_TIMES_T = 96.4935 / (np.log(10) * 0.00831470)

# Arguments of affinity() that don't define a dimension of the calculation
_NON_DIM_ARGS = ('what', 'property', 'exceed_Ttr', 'exceed_rhomin',
                 'return_buffer', 'balance')


def _n_values(value) -> int:
    """Number of values in an argument, as R's length() sees it.

    A string ("Psat") is one value in R, not one per character.
    """
    if isinstance(value, str) or not hasattr(value, '__len__'):
        return 1
    return len(value)


def affinity(messages: bool = True, basis: Optional[pd.DataFrame] = None,
             species: Optional[pd.DataFrame] = None, iprotein: Optional[Union[int, List[int], np.ndarray]] = None,
             loga_protein: Union[float, List[float]] = 0.0,
             transect: Optional[bool] = None, **kwargs) -> Dict[str, Any]:
    """
    Calculate affinities of formation reactions.

    This function calculates chemical affinities for the formation reactions of
    species of interest from user-selected basis species. The affinities are
    calculated as A/2.303RT where A is the chemical affinity.

    Parameters
    ----------
    messages : bool, default True
        Whether to print informational messages
    basis : pd.DataFrame, optional
        Basis species definition to use (if not using global basis)
    species : pd.DataFrame, optional
        Species definition to use (if not using global species)
    iprotein : int, list of int, or array, optional
        Build proteins from residues (row numbers in thermo().protein)
    loga_protein : float or list of float, default 0.0
        Activity of proteins (log scale)
    transect : bool, optional
        Whether the variables define a transect (a sequence of points) rather
        than a grid. If None (default), a transect is inferred when any
        variable has more than 3 values, as in R CHNOSZ.
    **kwargs : dict
        Variable arguments defining calculation conditions:
        - Basis species names (e.g., CO2=[-60, 20, 5]): Variable basis species activities
        - T : float or list, Temperature in °C
        - P : float or list, Pressure in bar
        - property : str, Property to calculate ("A", "logK", "G", etc.)
        - exceed_Ttr : bool, Allow extrapolation beyond transition temperatures
        - exceed_rhomin : bool, Allow calculations below minimum water density
        - return_buffer : bool, Return buffer activities
        - balance : str, Balance method for protein buffers

    Returns
    -------
    dict
        Dictionary containing:
        - fun : str, Function name ("affinity")
        - args : dict, Arguments used in calculation
        - sout : dict, Subcrt calculation results
        - property : str, Property calculated
        - basis : pd.DataFrame, Basis species definition
        - species : pd.DataFrame, Species of interest definition
        - T : float or array, Temperature(s) in Kelvin
        - P : float or array, Pressure(s) in bar
        - vars : list, Variable names
        - vals : dict, Variable values
        - values : dict, Calculated affinity values by species

    Examples
    --------
    >>> import pychnosz
    >>> pychnosz.reset()
    >>> pychnosz.basis(["CO2", "H2O", "NH3", "H2S", "H+", "O2"])
    >>> pychnosz.species(["glycine", "tyrosine", "serine", "methionine"])
    >>> result = pychnosz.affinity(CO2=[-60, 20, 5], T=350, P=2000)
    >>> print(result['values'][1566])  # Glycine affinities

    >>> # With proteins
    >>> import pandas as pd
    >>> aa = pd.read_csv("POLG.csv")
    >>> iprotein = pychnosz.add_protein(aa)
    >>> pychnosz.basis("CHNOSe")
    >>> a = pychnosz.affinity(iprotein=iprotein, pH=[2, 14], Eh=[-1, 1])

    Notes
    -----
    This implementation maintains complete fidelity to R CHNOSZ affinity():
    - Identical argument processing including dynamic basis species parameters
    - Same variable expansion and multi-dimensional calculations
    - Exact energy() function behavior for property calculations
    - Identical output structure and formatting
    - Support for protein calculations via iprotein parameter
    """

    # Get thermo object for protein handling
    thermo_obj = thermo()

    # Handle iprotein parameter
    ires = None
    original_species = None
    if iprotein is not None:
        # Convert to array
        if isinstance(iprotein, (int, np.integer)):
            iprotein = np.array([iprotein])
        elif isinstance(iprotein, list):
            iprotein = np.array(iprotein)

        # Check all proteins are available
        if np.any(np.isnan(iprotein)):
            raise AffinityError("`iprotein` has some NA values")
        if thermo_obj.protein is None or not np.all(iprotein < len(thermo_obj.protein)):
            raise AffinityError("some value(s) of `iprotein` are not rownumbers of thermo().protein")

        # Add protein residues to the species list
        # Amino acids in 3-letter code
        aminoacids_3 = ["Ala", "Cys", "Asp", "Glu", "Phe", "Gly", "His", "Ile", "Lys", "Leu",
                        "Met", "Asn", "Pro", "Gln", "Arg", "Ser", "Thr", "Val", "Trp", "Tyr"]

        # Use _RESIDUE notation (matches R CHNOSZ affinity.R line 84)
        resnames_residue = ["H2O_RESIDUE"] + [f"{aa}_RESIDUE" for aa in aminoacids_3]

        # Save original species
        from .species import species as species_func
        original_species = get_species() if is_species_defined() else None

        # Residue activities set to zero; account for protein activities later
        # (R: species(resprot, 0), where the numeric second argument is logact)
        species_func(resnames_residue, 0, add=True, messages=messages)

        # Get indices of residues in species list
        species_df_temp = get_species()
        ires = []
        for name in resnames_residue:
            idx = np.where(species_df_temp['name'] == name)[0]
            if len(idx) > 0:
                ires.append(idx[0])
        ires = np.array(ires)

    # Check if basis and species are defined (use provided or global)
    if basis is None:
        if not is_basis_defined():
            raise AffinityError("basis species are not defined")
        basis_df = get_basis()
    else:
        basis_df = basis

    if species is None:
        if not is_species_defined():
            raise AffinityError("species are not defined")
        species_df = get_species()
    else:
        species_df = species

    # Process arguments
    args_orig = dict(kwargs)

    # Handle argument recall (if first argument is previous affinity result)
    if len(args_orig) > 0:
        first_key = list(args_orig.keys())[0]
        first_value = args_orig[first_key]
        if (isinstance(first_value, dict) and
            first_value.get('fun') == 'affinity'):
            # Update arguments from previous result
            aargs = first_value.get('args', {})
            # Update with new arguments (skip the first one)
            new_args = dict(list(args_orig.items())[1:])
            aargs.update(new_args)
            return affinity(**aargs)

    # Process energy arguments
    args = energy_args(args_orig, messages, basis_df=basis_df, transect=transect)

    # Get property to calculate
    property_name = args.get('what', 'A')

    # Get thermo data
    thermo_obj = thermo()
    # basis_df and species_df are already set above

    # Determine if we need specific property calculation
    if property_name and property_name != 'A':
        # Calculate specific property using energy function
        energy_result = energy(
            what=property_name,
            vars=args['vars'],
            vals=args['vals'],
            lims=args['lims'],
            T=args['T'],
            P=args['P'],
            IS=args.get('IS', 0),
            exceed_Ttr=kwargs.get('exceed_Ttr', True),
            exceed_rhomin=kwargs.get('exceed_rhomin', False),
            basis_df=basis_df,
            species_df=species_df,
            messages=messages,
            transect=args['transect']
        )
        affinity_values = energy_result['a']
        energy_sout = energy_result['sout']
    else:
        # Calculate affinities (A/2.303RT)
        energy_result = energy(
            what='A',
            vars=args['vars'],
            vals=args['vals'],
            lims=args['lims'],
            T=args['T'],
            P=args['P'],
            IS=args.get('IS', 0),
            exceed_Ttr=kwargs.get('exceed_Ttr', True),
            exceed_rhomin=kwargs.get('exceed_rhomin', False),
            basis_df=basis_df,
            species_df=species_df,
            messages=messages,
            transect=args['transect']
        )
        affinity_values = energy_result['a']
        energy_sout = energy_result['sout']

        # Deal with affinities of protein ionization here, as R does inside
        # energy()'s A(). Any species whose name is protein_organism (including
        # the residues added for iprotein) gets the affinity of its ionization
        # added to that of its formation reaction.
        affinity_values = _ionize_protein_species(
            affinity_values, species_df, basis_df, args, messages
        )

    # Handle protein affinity calculations if iprotein was provided
    if iprotein is not None and ires is not None:
        # Calculate protein affinities from residue affinities using group additivity
        # Normalize loga_protein to match number of proteins
        if isinstance(loga_protein, (int, float)):
            loga_protein_arr = np.full(len(iprotein), loga_protein)
        else:
            loga_protein_arr = np.array(loga_protein)
            if len(loga_protein_arr) < len(iprotein):
                loga_protein_arr = np.resize(loga_protein_arr, len(iprotein))

        # Calculate affinity for each protein
        protein_affinities = {}

        for ip, iprot in enumerate(iprotein):
            # Get protein amino acid composition from thermo().protein
            # Columns 4:25 are chains + the 20 amino acids (R's tpext, columns 5:25),
            # matching the 21 residue species (H2O_RESIDUE + one per amino acid)
            protein_row = thermo_obj.protein.iloc[iprot]
            aa_counts = protein_row.iloc[4:25].values.astype(float)

            # Calculate protein affinity by summing residue affinities weighted by composition
            # affinity_values keys are ispecies indices
            # Get the ispecies for each residue
            species_df_current = get_species()
            residue_ispecies = species_df_current.iloc[ires]['ispecies'].values

            # Initialize protein affinity with same shape as residue affinities
            first_residue_key = residue_ispecies[0]
            if first_residue_key in affinity_values:
                template_affinity = affinity_values[first_residue_key]
                protein_affinity = np.zeros_like(template_affinity)

                # Sum up contributions from all residues
                for i, res_ispecies in enumerate(residue_ispecies):
                    if res_ispecies in affinity_values:
                        residue_contrib = affinity_values[res_ispecies] * aa_counts[i]
                        protein_affinity = protein_affinity + residue_contrib

                # Subtract protein activity
                protein_affinity = protein_affinity - loga_protein_arr[ip]

                # Use negative index to denote protein (matches R CHNOSZ convention)
                protein_key = -(iprot + 1)  # Negative of (row number + 1)
                protein_affinities[protein_key] = protein_affinity

        # Replace affinity_values with protein affinities
        affinity_values = protein_affinities

        # Calculate stoichiometric coefficients for proteins using matrix multiplication
        # This matches R CHNOSZ: protbasis <- t(t((resspecies[ires, 1:nrow(thermo$basis)])) %*% t((thermo$protein[iprotein, 5:25])))
        # IMPORTANT: Get the species list BEFORE deletion
        species_df_with_residues = get_species()

        # Extract basis species coefficients from residue species (rows = residues, cols = basis species)
        # ires contains indices of residues in the species list
        # We need the columns corresponding to basis species
        basis_cols = list(basis_df.index)  # e.g., ['CO2', 'H2O', 'NH3', 'H2S', 'e-', 'H+']

        # Create residue coefficient matrix (n_residues x n_basis)
        # resspecies[ires, 1:nrow(thermo$basis)] in R
        res_coeffs = species_df_with_residues.iloc[ires][basis_cols].values.astype(float)

        # Get amino acid composition matrix (n_proteins x n_residues)
        # thermo$protein[iprotein, 5:25] in R (columns 5-25 contain chains and 20 amino acids)
        # In Python (0-indexed): columns 4:25 contain chains and 20 amino acids
        aa_composition = []
        for iprot in iprotein:
            protein_row = thermo_obj.protein.iloc[iprot]
            # Columns 4:25 contain: chains, Ala, Cys, Asp, Glu, Phe, Gly, His, Ile, Lys, Leu,
            #                       Met, Asn, Pro, Gln, Arg, Ser, Thr, Val, Trp, Tyr
            aa_counts = protein_row.iloc[4:25].values.astype(float)
            aa_composition.append(aa_counts)
        aa_composition = np.array(aa_composition)  # Shape: (n_proteins, 21)

        # Matrix multiplication: (n_proteins x 21) @ (21 x n_basis) = (n_proteins x n_basis)
        # Note: res_coeffs has shape (21, n_basis) - first row is H2O, next 20 are amino acids
        # R code: t(t(resspecies) %*% t(protein)) means: (n_basis x n_residues) @ (n_residues x n_proteins) = (n_basis x n_proteins)
        # Then transpose to get (n_proteins x n_basis)
        # In Python: (n_proteins x n_residues) @ (n_residues x n_basis) = (n_proteins x n_basis)
        protein_coeffs = aa_composition @ res_coeffs  # Shape: (n_proteins, n_basis)

        # Delete residue species from species list now that we have the coefficients
        from .species import species as species_func
        # ires is 0-based (it indexes species_df with .iloc), but species(delete=)
        # takes 1-based row numbers, as in R
        species_func((ires + 1).tolist(), delete=True, messages=False)

        if original_species is not None:
            # Restore original species (but we've already calculated, so just update species_df)
            pass

        # Create DataFrame for proteins with basis species coefficients
        species_data = {}

        # Add basis species columns
        for j, basis_sp in enumerate(basis_cols):
            species_data[basis_sp] = protein_coeffs[:, j]

        # Add metadata columns
        protein_names = []
        protein_ispecies = []

        for iprot in iprotein:
            prot_row = thermo_obj.protein.iloc[iprot]
            # Names are stored unescaped, as in R; _format_chemname() escapes
            # underscores when rendering labels in math mode
            protein_names.append(f"{prot_row['protein']}_{prot_row['organism']}")
            protein_ispecies.append(-(iprot + 1))  # Negative index

        species_data['ispecies'] = protein_ispecies
        species_data['logact'] = loga_protein_arr[:len(iprotein)]
        species_data['state'] = ['aq'] * len(iprotein)
        species_data['name'] = protein_names

        species_df = pd.DataFrame(species_data)

    # Process temperature and pressure for output
    T_out = args['T']
    P_out = args['P']
    vars_list = args['vars']
    vals_dict = {}

    # Convert variable names and values for output
    # Important: Keep vars_list with actual basis species names (H+, e-) for internal use
    # but create display versions in vals_dict with user-friendly names (pH, pe, Eh)
    vars_list_display = vars_list.copy()
    for i, var in enumerate(vars_list):
        # Handle pH, pe, Eh conversions for output
        if var == 'H+' and 'pH' in args_orig:
            vars_list_display[i] = 'pH'
            vals_dict['pH'] = [-val for val in args['vals'][i]]
        elif var == 'e-' and 'pe' in args_orig:
            vars_list_display[i] = 'pe'
            vals_dict['pe'] = [-val for val in args['vals'][i]]
        elif var == 'e-' and 'Eh' in args_orig:
            vars_list_display[i] = 'Eh'
            # Rebuild the Eh values in volts from the original argument, as R does:
            # args['vals'] holds log(a_e-) over all the dimensions (the conversion
            # is temperature-dependent), which is not the 1-D axis wanted here
            Eharg = args_orig['Eh']
            if not hasattr(Eharg, '__len__'):
                Eh_vals = np.array([float(Eharg)])
            elif args['transect'] or len(Eharg) > 3:
                # On a transect the values are the points, not a range
                Eh_vals = np.asarray(Eharg, dtype=float)
            elif len(Eharg) == 3:
                Eh_vals = np.linspace(Eharg[0], Eharg[1], int(Eharg[2]))
            elif len(Eharg) == 2:
                Eh_vals = np.linspace(Eharg[0], Eharg[1], 256)
            else:
                Eh_vals = np.asarray(Eharg, dtype=float)
            vals_dict['Eh'] = Eh_vals
        else:
            vals_dict[var] = args['vals'][i]

    # Keep vars_list as-is (with basis species names) for internal calculations
    # vars_list_display will be used for output only

    # Check if T or P are variables
    if 'T' in vars_list:
        T_out = []  # Variable T
        # Convert back to Celsius for output
        T_vals = vals_dict['T']
        vals_dict['T'] = [T - 273.15 for T in T_vals]
    else:
        # Convert to Kelvin for output (matching R)
        T_out = args['T']

    if 'P' in vars_list:
        P_out = []  # Variable P
    else:
        P_out = args['P']

    # Build output dictionary matching R CHNOSZ structure
    result = {
        'fun': 'affinity',
        'args': {
            **args_orig,
            'property': property_name,
            'exceed_Ttr': kwargs.get('exceed_Ttr', False),
            'exceed_rhomin': kwargs.get('exceed_rhomin', False),
            'return_buffer': kwargs.get('return_buffer', False),
            'balance': kwargs.get('balance', 'PBB')
        },
        'sout': energy_sout,
        'property': property_name,
        'basis': basis_df,
        'species': species_df,
        'T': T_out,
        'P': P_out,
        'vars': vars_list_display,  # Use display version with 'Eh', 'pH', 'pe' for output
        'vals': vals_dict,
        'values': affinity_values
    }

    return result


def _ionize_protein_species(affinity_values: Dict[Any, Any], species_df: pd.DataFrame,
                            basis_df: pd.DataFrame, args: Dict[str, Any],
                            messages: bool = True) -> Dict[Any, Any]:
    """
    Add the affinity of ionization to protein species in the species list.

    Port of the protein ionization block of R CHNOSZ energy()'s A() function.
    Applies to any species named protein_organism, which includes both proteins
    added with species() and the residues added by affinity(iprotein = ...).
    """
    from ..biomolecules.proteins import pinfo

    # Ionization needs H+ in the basis to be meaningful
    if 'H+' not in basis_df.index:
        return affinity_values

    # Which species are proteins
    names = list(species_df['name'])
    ip_all = [pinfo(n) for n in names]
    isprotein = [not (v is None or (np.ndim(v) == 0 and pd.isna(v))) for v in ip_all]
    if not any(isprotein):
        return affinity_values

    if not thermo().opt.get('ionize.aa', True):
        if messages:
            print("affinity: NOT ionizing proteins because thermo().opt['ionize.aa'] is False")
        return affinity_values

    if messages:
        print("affinity: ionizing proteins ...")

    # The rownumbers in thermo().protein of the proteins in the species list
    ip = [int(v) for v, isp in zip(ip_all, isprotein) if isp]
    # as float in case the logact column is character mode due to a buffer definition
    pH = -float(basis_df.loc['H+', 'logact'])

    A_ionization = _ionization_affinity(
        pinfo(ip), args['vars'], args['vals'],
        T=args['T'], P=args['P'], pH=pH,
        transect=args.get('transect', False)
    )

    # Add it to the affinities of formation reactions of the non-ionized proteins
    keys = list(affinity_values.keys())
    j = 0
    for i, isp in enumerate(isprotein):
        if not isp:
            continue
        if i < len(keys):
            affinity_values[keys[i]] = affinity_values[keys[i]] + A_ionization[j]
        j += 1

    return affinity_values


def _ionization_affinity(aa: pd.DataFrame, vars: List[str], vals: List[Any],
                         T: Any = 298.15, P: Any = "Psat", pH: float = 7.0,
                         transect: bool = False) -> List[np.ndarray]:
    """
    Build a list of values of A/2.303RT of protein ionization, one per protein.

    Port of R CHNOSZ A.ionization() in util.affinity.R. Ionization affinity
    depends only on T, P and pH, so the values are calculated on the T-P-pH
    grid and then grown and permuted into the dimensions of all the variables
    (e.g. the values are constant along an Eh or logfO2 axis).

    Returns
    -------
    list of ndarray
        Ionization affinity for each protein, shaped like the affinity grid
    """
    from ..biomolecules.ionize_aa import ionize_aa

    # Start from the constant values, overridden by any that are variables.
    # T arrives in Kelvin; ionize_aa() wants degrees C
    T_vals = np.atleast_1d(np.asarray(T, dtype=float)) - 273.15
    P_vals = P if isinstance(P, str) else np.atleast_1d(np.asarray(P, dtype=float))
    pH_vals = np.atleast_1d(np.asarray(pH, dtype=float))

    iT = vars.index("T") if "T" in vars else None
    iP = vars.index("P") if "P" in vars else None
    iHplus = vars.index("H+") if "H+" in vars else None

    if iT is not None:
        T_vals = np.atleast_1d(np.asarray(vals[iT], dtype=float)) - 273.15
    if iP is not None:
        P_vals = np.atleast_1d(np.asarray(vals[iP], dtype=float))
    if iHplus is not None:
        pH_vals = -np.atleast_1d(np.asarray(vals[iHplus], dtype=float))

    # "Psat" is carried through as a scalar string; it has length 1
    P_len = 1 if isinstance(P_vals, str) else len(P_vals)

    if transect:
        # Single points, not a grid
        TPpH_T, TPpH_P, TPpH_pH = T_vals, P_vals, pH_vals
    else:
        # Make a grid of all combinations, with T, P, pH in the order they
        # show up in vars (variables first, then the constants)
        order = sorted(
            range(3),
            key=lambda k: ({0: iT, 1: iP, 2: iHplus}[k]
                           if {0: iT, 1: iP, 2: iHplus}[k] is not None else np.inf)
        )
        if iT is None and iP is None and iHplus is None:
            order = [0, 1, 2]

        axes = {0: T_vals, 1: (np.array([0.0]) if isinstance(P_vals, str) else P_vals), 2: pH_vals}
        # expand.grid varies the first argument fastest
        grids = np.meshgrid(*[axes[k] for k in order], indexing="ij")
        flat = [g.flatten(order="F") for g in grids]
        by_axis = {k: flat[i] for i, k in enumerate(order)}
        TPpH_T = by_axis[0]
        TPpH_P = P_vals if isinstance(P_vals, str) else by_axis[1]
        TPpH_pH = by_axis[2]

        # The dimensions of T-P-pH, dropping any that aren't in vars
        TPpH_dim_by_var = {}
        if iT is not None:
            TPpH_dim_by_var[iT] = len(T_vals)
        if iP is not None:
            TPpH_dim_by_var[iP] = P_len
        if iHplus is not None:
            TPpH_dim_by_var[iHplus] = len(pH_vals)
        TPpH_dim = [TPpH_dim_by_var[k] for k in sorted(TPpH_dim_by_var)]

        # The dimensions of the other vars, in the order they appear in vars
        iother = [i for i, v in enumerate(vars) if v not in ("T", "P", "H+")]
        other_dim = []
        for i in iother:
            v = np.asarray(vals[i])
            if vars[i] == "e-" and v.ndim > 1:
                # Values of pe were calculated in all dimensions; recover the
                # original length of the Eh variable by removing the other dims
                edim = list(v.shape)
                for d in TPpH_dim + other_dim:
                    if d in edim:
                        edim.remove(d)
                other_dim.append(edim[0] if edim else v.size)
            else:
                other_dim.append(v.size if v.ndim <= 1 else max(v.shape))

        # The permutation vector: T-P-pH dimensions come first, then the others
        allvars = [v for i, v in enumerate(vars) if i not in iother] + [vars[i] for i in iother]
        perm = [allvars.index(v) for v in vars]

    # Calculate the values of A/2.303RT as a function of T-P-pH
    A = ionize_aa(aa, property="A", T=TPpH_T, P=TPpH_P, pH=TPpH_pH)
    A = np.asarray(A)

    out = []
    for i in range(A.shape[1]):
        thisA = A[:, i]
        if transect:
            out.append(thisA)
            continue

        # Apply the dimensions of T-P-pH
        tpph_dim = TPpH_dim if len(TPpH_dim) > 0 else [1]
        thisA = thisA.reshape(tpph_dim, order="F")

        # Grow into the dimensions of all vars
        alldim = list(tpph_dim) + list(other_dim) if len(TPpH_dim) > 0 else list(other_dim)
        if len(alldim) == 0:
            alldim = [1]
        # R's array() recycles the values to fill the larger shape
        thisA = np.resize(thisA.flatten(order="F"), int(np.prod(alldim))).reshape(alldim, order="F")

        # Permute to put the dimensions in the same order as the variables
        thisA = np.transpose(thisA, axes=perm)
        out.append(thisA)

    return out


def energy_args(args: Dict[str, Any], messages: bool = True, basis_df: Optional[pd.DataFrame] = None,
                transect: Optional[bool] = None) -> Dict[str, Any]:
    """
    Process arguments for energy calculations.

    Converts variable arguments into consistent format for multi-dimensional
    calculations, handling T, P, IS and basis species variables.

    Parameters
    ----------
    args : dict
        Raw arguments from affinity() call
    transect : bool, optional
        Whether the variables define a transect rather than a grid. If None,
        a transect is inferred when any variable has more than 3 values.

    Returns
    -------
    dict
        Processed arguments with consistent variable structure
    """

    thermo_obj = thermo()
    if basis_df is None:
        basis_df = get_basis()

    # Do the variables specify a transect? Inputs are like [x1, x2, res], which
    # expand to a grid axis, but more than 3 values are taken as the points of a
    # transect instead (R: transect <- any(transect, any(sapply(args, length) > 3)))
    transect = bool(transect) or any(
        _n_values(v) > 3 for k, v in args.items() if k not in _NON_DIM_ARGS
    )

    # Default values
    T = 298.15
    P = "Psat"
    IS = 0
    T_is_var = P_is_var = IS_is_var = False

    # Process T, P, IS arguments
    if 'T' in args:
        T = args['T']
        if hasattr(T, '__len__') and len(T) > 1:
            T_is_var = True
        # Convert to Kelvin if needed (assuming Celsius input)
        if T_is_var:
            if transect:
                # Every value is a point of the transect, not a range
                T = np.asarray(T, dtype=float) + 273.15
            elif isinstance(T, (list, tuple)):
                # Handle [T1, T2, npoints] format or [T1, T2] (default to 256 points)
                if len(T) == 3:
                    T = np.linspace(T[0] + 273.15, T[1] + 273.15, int(T[2]))
                elif len(T) == 2:
                    # Default resolution: 256 points (R CHNOSZ standard)
                    T = np.linspace(T[0] + 273.15, T[1] + 273.15, 256)
                else:
                    T = np.array(T) + 273.15
            else:
                T = T + 273.15
        else:
            T = T + 273.15

    if 'P' in args:
        P = args['P']
        if hasattr(P, '__len__') and not isinstance(P, str) and len(P) > 1:
            P_is_var = True
        if P_is_var:
            if transect:
                P = np.asarray(P, dtype=float)
            elif isinstance(P, (list, tuple)):
                if len(P) == 3:
                    P = np.linspace(P[0], P[1], int(P[2]))
                elif len(P) == 2:
                    # Default resolution: 256 points (R CHNOSZ standard)
                    P = np.linspace(P[0], P[1], 256)

    if 'IS' in args:
        IS = args['IS']
        if hasattr(IS, '__len__') and len(IS) > 1:
            IS_is_var = True
            if transect:
                IS = np.asarray(IS, dtype=float)
            elif isinstance(IS, (list, tuple)):
                if len(IS) == 3:
                    IS = np.linspace(IS[0], IS[1], int(IS[2]))
                elif len(IS) == 2:
                    # Default resolution: 256 points (R CHNOSZ standard)
                    IS = np.linspace(IS[0], IS[1], 256)

    # Print status messages
    if messages:
        if not T_is_var:
            T_celsius = T - 273.15 if isinstance(T, (int, float)) else T[0] - 273.15
            print(f'affinity: temperature is {T_celsius:.0f} ºC')

        if not P_is_var:
            if P == "Psat":
                print("affinity: pressure is Psat")
            else:
                print(f'affinity: pressure is {P} bar')

        if not IS_is_var and IS != 0:
            print(f'affinity: ionic strength is {IS}')

    # Default property
    what = 'A'
    if 'what' in args:
        what = args['what']

    # Process variable arguments
    # Preserve the order in which variables were specified (R CHNOSZ compatibility)
    vars_list = []
    vals_list = []
    lims_list = []

    # Track which T/P/IS are variables and process them in the order they appear in args
    tps_vars = {'T': (T_is_var, T), 'P': (P_is_var, P), 'IS': (IS_is_var, IS)}

    def _message_var(arg_name, n, lo, hi):
        """Report the identity, range and units of a variable, as R does.

        The name shown is the one the user gave, so pH/pe/Eh are reported as
        such; only a name that is itself a basis species gets wrapped in
        log10(a_) (or log10(f_) for a gas).
        """
        nametxt = arg_name
        if arg_name in basis_names:
            if basis_df.loc[arg_name, 'state'] == 'gas':
                nametxt = f'log10(f_{arg_name})'
            else:
                nametxt = f'log10(a_{arg_name})'
        unittxt = ''
        if arg_name == 'T':
            unittxt = ' K'
        elif arg_name == 'P':
            unittxt = ' bar'
        elif arg_name == 'Eh':
            unittxt = ' V'
        print(f'affinity: variable {len(vars_list)} is {nametxt} at {n} values '
              f'from {lo} to {hi}{unittxt}')

    def _append_TPIS(arg_name):
        """Append a variable T, P or IS to vars/vals/lims."""
        var_value = tps_vars[arg_name][1]

        vars_list.append(arg_name)
        vals_list.append(var_value)

        if transect:
            # The values are the points of the transect; the limits are their range
            lims_list.append([var_value.min(), var_value.max(), len(var_value)])
        elif isinstance(args[arg_name], (list, tuple)):
            if len(args[arg_name]) == 3:
                # User specified [min, max, npoints]
                if arg_name == 'T':
                    lims_list.append([args[arg_name][0] + 273.15, args[arg_name][1] + 273.15, args[arg_name][2]])
                else:
                    lims_list.append([args[arg_name][0], args[arg_name][1], args[arg_name][2]])
            elif len(args[arg_name]) == 2:
                # User specified [min, max], default to 256 points
                if arg_name == 'T':
                    lims_list.append([args[arg_name][0] + 273.15, args[arg_name][1] + 273.15, 256])
                else:
                    lims_list.append([args[arg_name][0], args[arg_name][1], 256])
            else:
                # User provided explicit array of values
                lims_list.append([var_value.min(), var_value.max(), len(var_value)])
        else:
            lims_list.append([var_value.min(), var_value.max(), len(var_value)])

        if messages:
            lim = lims_list[-1]
            _message_var(arg_name, int(lim[2]), lim[0], lim[1])

    # Process T, P, IS and basis species variables in a single pass, so the
    # dimensions follow the order the user gave the arguments in, as in R
    basis_names = basis_df.index.tolist()

    # Positions in vars_list of any Eh variables, converted to log(a_e-) below
    Eh_var_indices = []

    for arg_name, arg_value in args.items():
        # Skip non-variables and non-dimension arguments
        if arg_name in _NON_DIM_ARGS:
            continue
        if arg_name in ['T', 'P', 'IS']:
            # Constant T/P/IS don't define a dimension (R cleans these out)
            if tps_vars[arg_name][0]:
                _append_TPIS(arg_name)
            continue

        # Handle pH -> H+, pe -> e-, Eh -> e-
        var_name = arg_name
        var_values = arg_value

        if arg_name in ('pH', 'pe'):
            # pH = -log(a_H+) and pe = -log(a_e-), so negate to get the logact.
            # For a transect every value is a point; otherwise only the first two
            # are limits (the third is the resolution and must not be negated)
            var_name = 'H+' if arg_name == 'pH' else 'e-'
            if transect:
                var_values = -np.asarray(var_values, dtype=float)
            elif hasattr(var_values, '__len__'):
                if len(var_values) == 3:
                    var_values = np.linspace(-var_values[0], -var_values[1], int(var_values[2]))
                elif len(var_values) == 2:
                    var_values = [-v for v in var_values]
                else:
                    # Single value in a list
                    var_values = np.array([-var_values[0]])
            else:
                # Scalar value
                var_values = np.array([-var_values])
        elif arg_name == 'Eh':
            var_name = 'e-'
            # Keep the Eh values in volts for now. The conversion to log(a_e-)
            # depends on temperature, so it's done below once all the variables
            # are known and Eh and T can both be expanded to the full grid
            # (as R does after energy.args() assembles the variables)
            i_Eh_var = len(vars_list)
            Eh_var_indices.append(i_Eh_var)
            if transect:
                var_values = np.asarray(var_values, dtype=float)
            elif hasattr(var_values, '__len__') and len(var_values) >= 2:
                if len(var_values) == 3:
                    # [Eh1, Eh2, npoints] format
                    var_values = np.linspace(var_values[0], var_values[1], int(var_values[2]))
                else:
                    # [Eh1, Eh2] format - default to 256 points like R
                    var_values = np.linspace(var_values[0], var_values[1], 256)
            else:
                # Single value
                var_values = np.atleast_1d(np.asarray(var_values, dtype=float))

        # Stop if the argument doesn't correspond to a basis species, T, P or IS
        if var_name not in basis_names:
            raise AffinityError(f"{arg_name} is not one of T, P, or IS, and does not match any basis species")

        vars_list.append(var_name)

        if transect:
            # Every value is a point of the transect, not a range
            vals_array = np.atleast_1d(np.asarray(var_values, dtype=float))
            lims_list.append([vals_array.min(), vals_array.max(), len(vals_array)])
        elif isinstance(var_values, (list, tuple)):
            if len(var_values) == 3:
                # [min, max, npoints] format
                vals_array = np.linspace(var_values[0], var_values[1], int(var_values[2]))
                lims_list.append(list(var_values))
            elif len(var_values) == 2:
                # [min, max] format - default to 256 points (R CHNOSZ behavior)
                vals_array = np.linspace(var_values[0], var_values[1], 256)
                lims_list.append([var_values[0], var_values[1], 256])
            else:
                # Explicit array of values
                vals_array = np.atleast_1d(np.asarray(var_values, dtype=float))
                lims_list.append([vals_array.min(), vals_array.max(), len(vals_array)])
        else:
            # Single value, or values already expanded above (pH, pe, Eh)
            vals_array = np.atleast_1d(np.asarray(var_values, dtype=float))
            lims_list.append([vals_array[0], vals_array[-1], len(vals_array)])

        vals_list.append(vals_array)

        if messages:
            # The range is reported in the units the user gave, so pH values are
            # shown as pH rather than as the log(a_H+) stored in vals (R's lims.orig)
            orig = np.atleast_1d(np.asarray(arg_value, dtype=float))
            if transect or len(orig) == 1:
                lo, hi = orig.min(), orig.max()
            else:
                lo, hi = orig[0], orig[1]
            _message_var(arg_name, len(vals_array), lo, hi)

    if transect:
        # Every variable has to supply a point for each position along the transect
        n_points = {len(np.atleast_1d(v)) for v in vals_list}
        if len(n_points) > 1:
            raise AffinityError("variables define a transect but their lengths are not all equal")

    # Convert Eh (volts) to log(a_e-), which depends on temperature:
    #   pe = Eh * F / (ln(10) * R * T) and log(a_e-) = -pe
    # R expands both Eh and T over all the dimensions and converts elementwise,
    # so with variable T the result is an array, not one value per Eh
    if Eh_var_indices:
        # A transect has a single dimension: every variable supplies one value
        # per point, so Eh and T pair up directly (R: dim.fun() uses idim <- 1)
        if transect:
            grid_shape = (int(lims_list[0][2]),)
        else:
            grid_shape = tuple(int(lim[2]) for lim in lims_list)

        def _on_grid(values, axis):
            if transect:
                return np.broadcast_to(np.asarray(values, dtype=float), grid_shape)
            shape = [1] * len(grid_shape)
            shape[axis] = len(np.atleast_1d(values))
            return np.broadcast_to(np.asarray(values, dtype=float).reshape(shape), grid_shape)

        # Temperature over all the dimensions (a constant if T is not a variable)
        if 'T' in vars_list:
            iT = vars_list.index('T')
            T_grid = _on_grid(vals_list[iT], iT)
        else:
            T_grid = np.full(grid_shape, float(T) if isinstance(T, (int, float)) else float(np.atleast_1d(T)[0]))

        for i_Eh in Eh_var_indices:
            Eh_grid = _on_grid(vals_list[i_Eh], i_Eh)
            vals_list[i_Eh] = -Eh_grid * (_PE_PER_VOLT_TIMES_T / T_grid)

    return {
        'what': what,
        'vars': vars_list,
        'vals': vals_list,
        'lims': lims_list,
        'T': T,
        'P': P,
        'IS': IS,
        'transect': transect
    }


def energy(what: str, vars: List[str], vals: List, lims: List,
           T: Union[float, np.ndarray] = 298.15,
           P: Union[float, str] = "Psat",
           IS: float = 0,
           sout: Optional[Dict] = None,
           exceed_Ttr: bool = True,
           exceed_rhomin: bool = False,
           basis_df: Optional[pd.DataFrame] = None,
           species_df: Optional[pd.DataFrame] = None,
           messages: bool = True,
           transect: bool = False) -> Dict[str, Any]:
    """
    Calculate energy properties over multiple dimensions.

    This is the core calculation function that handles multi-dimensional
    property calculations for basis and formed species.

    Parameters
    ----------
    what : str
        Property to calculate ("A", "logK", "G", "H", etc.)
    vars : list of str
        Variable names
    vals : list of arrays
        Variable values
    lims : list of limits
        Variable limits [min, max, npoints]
    T : float or array
        Temperature(s) in Kelvin
    P : float or str
        Pressure(s) in bar or "Psat"
    IS : float
        Ionic strength
    sout : dict, optional
        Pre-calculated subcrt results
    exceed_Ttr : bool
        Allow extrapolation beyond transitions
    exceed_rhomin : bool
        Allow below minimum density
    transect : bool, default False
        If True the variables define a sequence of points rather than a grid,
        so the result has one dimension no matter how many variables there are

    Returns
    -------
    dict
        Dictionary with 'sout' (subcrt results) and 'a' (property values)
    """

    # Get system data
    thermo_obj = thermo()
    if basis_df is None:
        basis_df = get_basis()
    if species_df is None:
        species_df = get_species()

    n_basis = len(basis_df)
    n_species = len(species_df)

    # Determine array dimensions
    if len(vars) == 0:
        mydim = [1]
    elif transect:
        # All the variables step together along a single dimension
        n_points = [lim[2] for lim in lims]
        if min(n_points) != max(n_points):
            raise AffinityError("variables define a transect but their lengths are not all equal")
        mydim = [n_points[0]]
    else:
        mydim = [lim[2] for lim in lims]

    # The dimensions follow the order the variables were given in, which is not
    # necessarily T-first, so values are placed on their own axis explicitly
    # rather than relying on numpy's trailing-axis broadcasting
    def _var_on_grid(values, axis):
        """Broadcast one variable's values along its own axis of the full grid."""
        values = np.asarray(values)
        # On a transect every variable already has one value per point
        if transect:
            return np.asarray(values, dtype=float).reshape(mydim).copy()
        # Eh was already converted to log(a_e-) over all the dimensions
        if values.shape == tuple(mydim):
            return values.copy()
        shape = [1] * len(mydim)
        shape[axis] = len(np.atleast_1d(values))
        # np.broadcast_to returns a read-only view, so copy: callers accumulate in place
        return np.broadcast_to(values.reshape(shape), tuple(mydim)).copy()

    # Which of T, P, IS are variables, and which one subcrt() makes a grid over.
    # As in R, a grid is only needed when two of them vary, and IS is always the
    # grid variable when it is one of them (the only one subcrt() can grid over)
    subcrt_vars = [v for v in vars if v in ('T', 'P', 'IS')]
    if len(subcrt_vars) > 2:
        raise AffinityError("only up to 2 of P,T,IS are supported")
    if len(subcrt_vars) > 1 and not transect:
        grid_var = 'IS' if 'IS' in subcrt_vars else subcrt_vars[0]
    else:
        grid_var = None

    def _TPIS_on_grid(flat_vals):
        """Place values that vary only with T/P/IS onto the full variable grid.

        subcrt() returns them for a grid of the T/P/IS variables alone, with the
        grid variable varying slowest. That is both a subset of the dimensions
        and in a different order than the user's arguments.
        """
        arr = np.asarray(flat_vals)
        if transect:
            # subcrt() was called with the transect's T, P and IS paired up, so
            # it already returned one value per point
            return arr
        if len(mydim) <= 1 or arr.ndim != 1:
            return arr
        if not subcrt_vars:
            return arr
        # The grid variable varies slowest in subcrt()'s output
        sub_vars = ([grid_var] + [v for v in subcrt_vars if v != grid_var]
                    if grid_var is not None else list(subcrt_vars))
        sub_shape = [len(np.atleast_1d(vals[vars.index(v)])) for v in sub_vars]
        if arr.size != int(np.prod(sub_shape)):
            return arr
        arr = arr.reshape(sub_shape)
        # Reorder the subcrt axes into the order the user gave the variables in
        arr = np.transpose(arr, axes=np.argsort([vars.index(v) for v in sub_vars]))
        # Insert length-1 axes for the variables these values don't depend on
        shape = [1] * len(mydim)
        for v in sub_vars:
            shape[vars.index(v)] = len(np.atleast_1d(vals[vars.index(v)]))
        # Copy because np.broadcast_to returns a read-only view
        return np.broadcast_to(arr.reshape(shape), tuple(mydim)).copy()

    # Prepare subcrt call
    if what in ['G', 'H', 'S', 'Cp', 'V', 'E', 'kT', 'logK'] or what == 'A':
        # Need to call subcrt for thermodynamic properties

        # Prepare species list (basis + formed species)
        all_species = basis_df['ispecies'].tolist() + species_df['ispecies'].tolist()

        # Prepare T, P, IS for subcrt (convert T from Kelvin to Celsius)
        subcrt_T = T - 273.15 if isinstance(T, (int, float)) else T - 273.15
        subcrt_P = P
        subcrt_IS = IS

        # Handle variable T, P, IS
        if 'T' in vars:
            # T in vals is already in Kelvin, convert to Celsius for subcrt
            T_vals = vals[vars.index('T')]
            subcrt_T = T_vals - 273.15 if isinstance(T_vals, (int, float)) else T_vals - 273.15
        if 'P' in vars:
            subcrt_P = vals[vars.index('P')]
        if 'IS' in vars:
            subcrt_IS = vals[vars.index('IS')]

        # Call subcrt
        # Skip sout calculation for affinity (what=='A') since the affinity block
        # has its own optimized batch subcrt call
        if sout is None and what != 'A':
            try:
                sout_result = subcrt(
                    species=all_species,
                    T=subcrt_T,
                    P=subcrt_P,
                    IS=subcrt_IS,
                    property='logK',
                    grid=grid_var,
                    exceed_Ttr=exceed_Ttr,
                    exceed_rhomin=exceed_rhomin,
                    messages=messages,
                    show=False
                )
                sout_data = sout_result.out

            except Exception as e:
                warnings.warn(f"subcrt calculation failed: {e}")
                # Create dummy sout data
                n_conditions = np.prod(mydim) if len(mydim) > 0 else 1
                sout_data = pd.DataFrame({
                    'T': np.full(n_conditions, T if isinstance(T, (int, float)) else T[0]) - 273.15,
                    'P': np.full(n_conditions, 1.0 if P == "Psat" else (P if isinstance(P, (int, float)) else P[0])),
                    'logK': np.full(n_conditions, np.nan)
                })
        else:
            sout_data = sout

    # Calculate the requested property
    if what == 'A':
        # Calculate affinities A/2.303RT following R CHNOSZ logic exactly
        affinity_values = {}

        # Get basis and species information
        basis_names = basis_df.index.tolist()
        n_conditions = np.prod(mydim) if len(mydim) > 0 else 1

        # Create activity arrays for each basis species using multi-dimensional grid expansion
        # This implements R's expand.grid functionality using numpy.meshgrid
        logact_basis_arrays = {}

        if len(vars) > 1:
            # Multi-dimensional case: each basis variable varies along its own
            # axis of the grid, at the position where the user gave it
            for var_idx, var_name in enumerate(vars):
                if var_name in basis_names:
                    logact_basis_arrays[var_name] = _var_on_grid(np.array(vals[var_idx]), var_idx)

        # Handle all basis species (variables and fixed)
        for j, basis_name in enumerate(basis_names):
            if basis_name in vars and basis_name not in logact_basis_arrays:
                # Single variable case
                var_idx = vars.index(basis_name)
                logact_basis_arrays[basis_name] = np.array(vals[var_idx])
            elif basis_name not in logact_basis_arrays:
                # Fixed activity from basis definition - broadcast to full grid
                basis_logact = basis_df.iloc[j]['logact']
                try:
                    logact_val = float(basis_logact)
                except (ValueError, TypeError):
                    logact_val = 0.0

                if len(mydim) > 1:
                    # Multi-dimensional: broadcast scalar to full grid shape
                    logact_basis_arrays[basis_name] = np.full(mydim, logact_val)
                else:
                    # Single dimension
                    logact_basis_arrays[basis_name] = np.full(n_conditions, logact_val)

        # For affinities, we need logK of balanced formation reactions
        # Optimize by calling subcrt once for all basis + non-basis species
        # to get logK of formation from elements, then calculate formation from basis
        formation_logK = {}

        # Convert T from Kelvin back to Celsius for subcrt (subcrt expects Celsius)
        T_celsius = T - 273.15

        # Get all unique species (basis + formed species) using ispecies indices
        # to avoid redundant info_character lookups
        basis_ispecies_list = basis_df['ispecies'].tolist()
        species_ispecies_list = species_df['ispecies'].tolist()
        all_species_indices = list(dict.fromkeys(basis_ispecies_list + species_ispecies_list))

        # Create mapping from names to ispecies indices
        # Note: multiple names (e.g., "Fe" and "iron") can map to the same ispecies
        basis_names_list = basis_names  # Already defined at line 548
        species_names_list = species_df['name'].tolist()

        # Build a name->ispecies mapping
        name_to_ispecies = {}
        for name, ispec in zip(basis_names_list, basis_ispecies_list):
            name_to_ispecies[name] = ispec
        for name, ispec in zip(species_names_list, species_ispecies_list):
            name_to_ispecies[name] = ispec

        # Build ispecies->result_index mapping for batch result access
        ispecies_to_result_idx = {ispec: idx for idx, ispec in enumerate(all_species_indices)}

        # All unique names (may have duplicates that refer to same ispecies)
        all_species_names = list(dict.fromkeys(basis_names_list + species_names_list))

        # Single batch subcrt call to get logK of formation from elements for all species
        # Use ispecies indices to avoid redundant lookups
        try:
            # P and IS come from vals when they are variables; T_celsius already does
            batch_P = vals[vars.index('P')] if 'P' in vars else P
            batch_IS = vals[vars.index('IS')] if 'IS' in vars else IS

            batch_result = subcrt(all_species_indices, property="logK", T=T_celsius, P=batch_P,
                                  IS=batch_IS, grid=grid_var, messages=messages, show=False)

            # Extract logK values from batch result
            # batch_result.out is a dict with 'species_data' list
            # When T/P are variable, each species_data DataFrame has multiple rows
            species_logK_from_elements = {}
            if isinstance(batch_result.out, dict) and 'species_data' in batch_result.out:
                # Map each name to its data using the ispecies->result_idx mapping
                for sp_name in all_species_names:
                    ispec = name_to_ispecies[sp_name]
                    result_idx = ispecies_to_result_idx[ispec]
                    sp_data = batch_result.out['species_data'][result_idx]

                    if 'logK' in sp_data.columns:
                        # Get all logK values (may be array if T/P variable)
                        # Use .copy() to ensure array is writable (numpy 2.x returns read-only views)
                        logK_vals = sp_data['logK'].values.copy()
                        # Handle NaN values by keeping them as nan (they will propagate to affinity)
                        # DO NOT replace nan with 0.0 as this causes incorrect affinity calculations
                        # logK_vals = np.where(np.isnan(logK_vals), 0.0, logK_vals)

                        # logK varies only with T/P/IS; put it on the full grid
                        logK_vals = _TPIS_on_grid(logK_vals)

                        species_logK_from_elements[sp_name] = logK_vals
                    else:
                        # No logK column - use zeros
                        n_rows = len(sp_data)
                        if len(mydim) > 1 and n_rows == np.prod(mydim):
                            species_logK_from_elements[sp_name] = np.zeros(mydim)
                        else:
                            species_logK_from_elements[sp_name] = np.zeros(n_rows)
            elif isinstance(batch_result.out, pd.DataFrame):
                # Single species case - result.out is a DataFrame directly
                sp_data = batch_result.out
                sp_name = all_species_names[0]
                if 'logK' in sp_data.columns:
                    # Use .copy() to ensure array is writable (numpy 2.x returns read-only views)
                    logK_vals = sp_data['logK'].values.copy()
                    # Handle NaN values by keeping them as nan (they will propagate to affinity)
                    # DO NOT replace nan with 0.0 as this causes incorrect affinity calculations
                    # logK_vals = np.where(np.isnan(logK_vals), 0.0, logK_vals)

                    # logK varies only with T/P/IS; put it on the full grid
                    logK_vals = _TPIS_on_grid(logK_vals)

                    species_logK_from_elements[sp_name] = logK_vals
                else:
                    n_rows = len(sp_data)
                    if len(mydim) > 1 and n_rows == np.prod(mydim):
                        species_logK_from_elements[sp_name] = np.zeros(mydim)
                    else:
                        species_logK_from_elements[sp_name] = np.zeros(n_rows)
            else:
                # Fallback if structure is different
                for sp_name in all_species_names:
                    if len(mydim) > 1:
                        species_logK_from_elements[sp_name] = np.zeros(mydim)
                    else:
                        species_logK_from_elements[sp_name] = np.array([0.0])

            # Now calculate formation logK from basis species for each formed species
            for i in range(n_species):
                species_idx = species_df.iloc[i]['ispecies']
                species_name = species_df.iloc[i]['name']

                # Check if this species is also a basis species
                is_basis_species = species_idx in basis_df['ispecies'].values

                if is_basis_species:
                    # Species is in the basis - formation from basis is trivial
                    formation_logK[species_idx] = 0.0
                else:
                    # Calculate formation logK from basis using stoichiometry
                    # The species() coefficients represent: species = basis_products - basis_reactants
                    # For logK from elements: logK_formation = logK_species - sum(coeff_i * logK_basis_i)
                    logK_formation_val = species_logK_from_elements.get(species_name, 0.0)

                    # Subtract contribution from basis species
                    for basis_name in basis_names_list:
                        coeff = species_df.iloc[i][basis_name]
                        basis_logK = species_logK_from_elements.get(basis_name, 0.0)
                        logK_formation_val -= coeff * basis_logK

                    formation_logK[species_idx] = logK_formation_val

        except Exception as e:
            warnings.warn(f"Batch subcrt call failed, falling back to individual calls: {e}")
            # Fallback to old method if batch call fails
            for i in range(n_species):
                species_idx = species_df.iloc[i]['ispecies']
                is_basis_species = species_idx in basis_df['ispecies'].values

                if is_basis_species:
                    formation_logK[species_idx] = 0.0
                else:
                    try:
                        species_name = species_df.iloc[i]['name']
                        formation_result = subcrt([species_name], [1], T=T_celsius, P=P, messages=messages, show=False)

                        # Handle both single DataFrame and dict of DataFrames
                        if hasattr(formation_result, 'out'):
                            if isinstance(formation_result.out, dict) and 'species_data' in formation_result.out:
                                # Multiple conditions (T/P arrays) - result.out is a dict
                                sp_data = formation_result.out['species_data'][0]
                                if 'logK' in sp_data.columns:
                                    logK_vals = sp_data['logK'].values
                                    # Keep nan values as is
                                    # logK_vals = np.where(np.isnan(logK_vals), 0.0, logK_vals)
                                    logK_val = logK_vals
                                else:
                                    logK_val = np.zeros(len(sp_data))
                            elif isinstance(formation_result.out, pd.DataFrame):
                                # Single condition - result.out is a DataFrame
                                if 'logK' in formation_result.out.columns:
                                    logK_val = formation_result.out['logK'].values
                                    # Keep nan values as is
                                    # logK_val = np.where(np.isnan(logK_val), 0.0, logK_val)
                                else:
                                    logK_val = 0.0
                            else:
                                logK_val = 0.0
                        else:
                            logK_val = 0.0
                        formation_logK[species_idx] = logK_val
                    except Exception as e2:
                        warnings.warn(f"Could not get formation logK for species {species_idx}: {e2}")
                        formation_logK[species_idx] = 0.0

        # Calculate affinities for each formed species
        for i in range(n_species):
            species_idx = species_df.iloc[i]['ispecies']

            # Get the formation reaction logK (already balanced)
            logK_formation = formation_logK[species_idx]

            # Get formation reaction stoichiometry from species DataFrame
            # These are the stoichiometric coefficients from the balanced reaction
            formation_coeffs = {}
            for basis_name in basis_names:
                formation_coeffs[basis_name] = species_df.iloc[i][basis_name]

            # Calculate logQ using R CHNOSZ logic:
            # logQ = +1 * logact_species + sum(-coeff_i * logact_basis_i)
            # Species gets +1 coefficient (product), all basis species get negative coefficients (reactants)

            # Species activity (always +1 coefficient on product side)
            species_logact = species_df.iloc[i]['logact']
            try:
                species_logact_val = float(species_logact)
            except (ValueError, TypeError):
                species_logact_val = 0.0

            # Start with species contribution: +1 * logact_species
            # Create array with proper dimensions to match the grid
            if len(mydim) > 1:
                logQ_arrays = np.full(mydim, species_logact_val)
            else:
                logQ_arrays = np.full(n_conditions, species_logact_val)

            # Add contributions from all basis species: -coeff_i * logact_basis_i
            for basis_name in formation_coeffs:
                coeff = formation_coeffs[basis_name]
                logact_array = logact_basis_arrays[basis_name]
                # DEBUG
                if False and species_idx == 763:  # ethanol
                    print(f"  Basis {basis_name}: coeff={coeff}, logact_array[0]={logact_array[0] if hasattr(logact_array, '__getitem__') else logact_array}")
                # All basis species contributions are negative (reactant side)
                logQ_arrays += (-coeff) * logact_array

            # Calculate affinity: A/2.303RT = logK - logQ
            # Handle shape broadcasting when logK varies along fewer dimensions than logQ
            # This happens when we have basis variables (e.g., H2S) and subcrt variables (e.g., T)
            # logK only varies with subcrt variables (T, P, IS) but logQ varies with all variables
            if isinstance(logK_formation, np.ndarray) and isinstance(logQ_arrays, np.ndarray):
                if logK_formation.shape != logQ_arrays.shape:
                    # logK varies only with T/P/IS, so put it on those axes of the
                    # grid rather than guessing from the length (which is ambiguous
                    # when two variables have the same number of values)
                    if len(mydim) > 1 and logK_formation.ndim == 1:
                        logK_formation = _TPIS_on_grid(logK_formation)

            affinity_array = logK_formation - logQ_arrays

            # DEBUG: Check first value
            if False:  # Set to True for debugging
                if hasattr(affinity_array, '__getitem__'):
                    print(f"\nDEBUG affinity for species {species_idx}:")
                    print(f"  logK_formation[0] = {logK_formation[0] if hasattr(logK_formation, '__getitem__') else logK_formation}")
                    print(f"  logQ_arrays[0] = {logQ_arrays[0] if hasattr(logQ_arrays, '__getitem__') else logQ_arrays}")
                    print(f"  affinity_array[0] = {affinity_array[0]}")

            # Store result with proper dimensions
            # Keep array structure if we have multiple variables, even if n_conditions == 1
            # This ensures diagram() can detect the correct dimensionality (matching R behavior)
            if n_conditions == 1 and len(mydim) <= 1:
                # True scalar case: no variables or single variable with 1 point
                affinity_values[species_idx] = affinity_array.item() if hasattr(affinity_array, 'item') else affinity_array
            else:
                # Multi-dimensional case: preserve array structure
                # Array already has correct shape from meshgrid
                affinity_values[species_idx] = affinity_array

        return {
            'sout': sout_data,
            'a': affinity_values
        }

    elif what == 'logK':
        # Extract logK values from subcrt results
        logK_values = {}

        for i in range(n_species):
            species_idx = species_df.iloc[i]['ispecies']

            if hasattr(sout_data, 'iloc') and len(sout_data) > n_basis + i:
                logK_val = sout_data.iloc[n_basis + i]['logK'] if 'logK' in sout_data.columns else np.nan
            else:
                logK_val = np.nan

            # Expand to proper dimensions
            if np.prod(mydim) > 1:
                logK_values[species_idx] = np.full(mydim, logK_val)
            else:
                logK_values[species_idx] = logK_val

        return {
            'sout': sout_data,
            'a': logK_values
        }

    else:
        # Other thermodynamic properties
        prop_values = {}

        for i in range(n_species):
            species_idx = species_df.iloc[i]['ispecies']

            if hasattr(sout_data, 'iloc') and len(sout_data) > n_basis + i:
                prop_val = sout_data.iloc[n_basis + i][what] if what in sout_data.columns else np.nan
            else:
                prop_val = np.nan

            # Expand to proper dimensions
            if np.prod(mydim) > 1:
                prop_values[species_idx] = np.full(mydim, prop_val)
            else:
                prop_values[species_idx] = prop_val

        return {
            'sout': sout_data,
            'a': prop_values
        }


# Export main functions
__all__ = [
    'affinity', 'energy_args', 'energy', 'AffinityError'
]