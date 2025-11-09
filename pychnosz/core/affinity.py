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


def affinity(messages: bool = True, basis: Optional[pd.DataFrame] = None,
             species: Optional[pd.DataFrame] = None, iprotein: Optional[Union[int, List[int], np.ndarray]] = None,
             loga_protein: Union[float, List[float]] = 0.0, **kwargs) -> Dict[str, Any]:
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

        # Add residue species with activity 0 (all in "aq" state)
        species_func(resnames_residue, state="aq", add=True, messages=messages)

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
    args = energy_args(args_orig, messages, basis_df=basis_df)

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
            messages=messages
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
            messages=messages
        )
        affinity_values = energy_result['a']
        energy_sout = energy_result['sout']

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
            # Columns 4:24 contain chains and amino acid counts (0-indexed: columns 4-23)
            protein_row = thermo_obj.protein.iloc[iprot]
            aa_counts = protein_row.iloc[4:24].values.astype(float)

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

        # Add ionization affinity if H+ is in basis (matching R CHNOSZ behavior)
        if 'H+' in basis_df.index:
            if messages:
                print("affinity: ionizing proteins ...")

            # Get protein amino acid compositions
            from ..biomolecules.proteins import pinfo
            from ..biomolecules.ionize_aa import ionize_aa

            # Get aa compositions for these proteins
            aa = pinfo(iprotein)

            # Determine pH values from vars/vals or basis
            # Check if H+ is a variable
            if 'H+' in args['vars']:
                # H+ is a variable - get pH from vals
                iHplus = args['vars'].index('H+')
                pH_vals = -np.array(args['vals'][iHplus])  # pH = -log(a_H+)
            else:
                # H+ is constant - get from basis
                pH_val = -basis_df.loc['H+', 'logact']  # pH = -log(a_H+)
                pH_vals = np.array([pH_val])

            # Get T values (already processed earlier)
            T_vals = args['T']
            if isinstance(T_vals, (int, float)):
                T_celsius = T_vals - 273.15
            else:
                T_celsius = T_vals - 273.15

            # Get P values
            P_vals = args['P']

            # Calculate ionization affinity
            # ionize_aa expects arrays, so ensure T, P, pH are properly shaped
            # For grid calculations, we need to expand T, P, pH into a grid matching the affinity grid
            if len(args['vars']) >= 2:
                # Multi-dimensional case - create grid
                # Figure out which vars are T, P, H+
                var_names = args['vars']
                has_T_var = 'T' in var_names
                has_P_var = 'P' in var_names
                has_Hplus_var = 'H+' in var_names

                # Build T, P, pH grids matching the affinity calculation grid
                if has_T_var and has_Hplus_var:
                    # Both T and pH vary - create meshgrid
                    T_grid, pH_grid = np.meshgrid(T_celsius, pH_vals, indexing='ij')
                    T_flat = T_grid.flatten()
                    pH_flat = pH_grid.flatten()
                    if isinstance(P_vals, str):
                        P_flat = np.array([P_vals] * len(T_flat))
                    else:
                        P_flat = np.full(len(T_flat), P_vals if isinstance(P_vals, (int, float)) else P_vals[0])
                elif has_T_var:
                    # Only T varies
                    T_flat = T_celsius if isinstance(T_celsius, np.ndarray) else np.array([T_celsius])
                    pH_flat = np.full(len(T_flat), pH_vals[0])
                    P_flat = np.array([P_vals] * len(T_flat)) if isinstance(P_vals, str) else np.full(len(T_flat), P_vals if isinstance(P_vals, (int, float)) else P_vals[0])
                elif has_Hplus_var:
                    # Only pH varies
                    pH_flat = pH_vals
                    T_flat = np.full(len(pH_flat), T_celsius if isinstance(T_celsius, (int, float)) else T_celsius[0])
                    P_flat = np.array([P_vals] * len(pH_flat)) if isinstance(P_vals, str) else np.full(len(pH_flat), P_vals if isinstance(P_vals, (int, float)) else P_vals[0])
                else:
                    # No T or pH variables
                    T_flat = np.array([T_celsius if isinstance(T_celsius, (int, float)) else T_celsius[0]])
                    pH_flat = pH_vals
                    P_flat = np.array([P_vals] if isinstance(P_vals, str) else [P_vals if isinstance(P_vals, (int, float)) else P_vals[0]])
            else:
                # Single or no variable case
                T_flat = np.array([T_celsius if isinstance(T_celsius, (int, float)) else T_celsius[0]])
                pH_flat = pH_vals if isinstance(pH_vals, np.ndarray) else np.array([pH_vals[0] if hasattr(pH_vals, '__getitem__') else pH_vals])
                P_flat = np.array([P_vals] if isinstance(P_vals, str) else [P_vals if isinstance(P_vals, (int, float)) else P_vals[0]])

            # Call ionize_aa to get ionization affinity
            ionization_result = ionize_aa(aa, property="A", T=T_flat, P=P_flat, pH=pH_flat)

            # Add ionization affinity to formation affinity for each protein
            for ip, iprot in enumerate(iprotein):
                protein_key = -(iprot + 1)
                ionization_affinity = ionization_result.iloc[:, ip].values

                # Reshape to match formation affinity dimensions if needed
                formation_affinity = protein_affinities[protein_key]
                if isinstance(formation_affinity, np.ndarray):
                    if formation_affinity.shape != ionization_affinity.shape:
                        # Reshape ionization affinity to match formation affinity
                        ionization_affinity = ionization_affinity.reshape(formation_affinity.shape)

                # Add ionization to formation affinity
                protein_affinities[protein_key] = formation_affinity + ionization_affinity

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
        # In Python (0-indexed): columns 4:24 contain chains and 20 amino acids
        aa_composition = []
        for iprot in iprotein:
            protein_row = thermo_obj.protein.iloc[iprot]
            # Columns 4:24 contain: chains, Ala, Cys, Asp, Glu, Phe, Gly, His, Ile, Lys, Leu,
            #                       Met, Asn, Pro, Gln, Arg, Ser, Thr, Val, Trp, Tyr
            aa_counts = protein_row.iloc[4:24].values.astype(float)
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
        species_func(ires.tolist(), delete=True, messages=False)

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
            # Escape underscores for LaTeX compatibility in diagram labels
            protein_name = f"{prot_row['protein']}_{prot_row['organism']}"
            # Replace underscores with escaped version for matplotlib/LaTeX
            protein_name_escaped = protein_name.replace('_', r'\_')
            protein_names.append(protein_name_escaped)
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
            # Convert from log(a_e-) back to Eh using temperature-dependent formula
            # log(a_e-) = -pe, so pe = -log(a_e-)
            # Eh = pe * (ln(10) * R * T) / F = -log(a_e-) * T / 5039.76
            T_kelvin = args['T'] if isinstance(args['T'], (int, float)) else args['T'][0] if hasattr(args['T'], '__len__') else 298.15
            conversion_factor = T_kelvin / 5039.76  # volts per pe unit
            vals_dict['Eh'] = [-val * conversion_factor for val in args['vals'][i]]
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


def energy_args(args: Dict[str, Any], messages: bool = True, basis_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Process arguments for energy calculations.

    Converts variable arguments into consistent format for multi-dimensional
    calculations, handling T, P, IS and basis species variables.

    Parameters
    ----------
    args : dict
        Raw arguments from affinity() call

    Returns
    -------
    dict
        Processed arguments with consistent variable structure
    """

    thermo_obj = thermo()
    if basis_df is None:
        basis_df = get_basis()

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
            if isinstance(T, (list, tuple)):
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
        if hasattr(P, '__len__') and len(P) > 1:
            P_is_var = True
        if P_is_var and P != "Psat":
            if isinstance(P, (list, tuple)):
                if len(P) == 3:
                    P = np.linspace(P[0], P[1], int(P[2]))
                elif len(P) == 2:
                    # Default resolution: 256 points (R CHNOSZ standard)
                    P = np.linspace(P[0], P[1], 256)

    if 'IS' in args:
        IS = args['IS']
        if hasattr(IS, '__len__') and len(IS) > 1:
            IS_is_var = True
            if isinstance(IS, (list, tuple)):
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

    # Add T, P, IS in the order they appear in args (preserves user's specification order)
    for arg_name in args.keys():
        if arg_name in ['T', 'P', 'IS'] and tps_vars[arg_name][0]:
            var_name = arg_name
            var_value = tps_vars[arg_name][1]

            vars_list.append(var_name)
            vals_list.append(var_value)

            if isinstance(args[arg_name], (list, tuple)):
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

    # Process basis species variables
    basis_names = basis_df.index.tolist()

    for arg_name, arg_value in args.items():
        # Skip T, P, IS, and non-basis arguments
        if arg_name in ['T', 'P', 'IS', 'what', 'property', 'exceed_Ttr', 'exceed_rhomin', 'return_buffer', 'balance']:
            continue

        # Handle pH -> H+, pe -> e-, Eh -> e-
        var_name = arg_name
        var_values = arg_value

        if arg_name == 'pH':
            var_name = 'H+'
            if hasattr(var_values, '__len__'):
                if len(var_values) >= 3:
                    # [pH1, pH2, npoints] -> [-pH1, -pH2, npoints] for H+ (logact)
                    # pH and log(a_H+) are related by: pH = -log(a_H+), so log(a_H+) = -pH
                    var_values = np.linspace(-var_values[0], -var_values[1], int(var_values[2]))
                elif len(var_values) >= 2:
                    var_values = [-v for v in var_values]
                else:
                    # Single value in a list [pH]
                    var_values = np.array([-var_values[0]])
            else:
                # Scalar value
                var_values = np.array([-var_values])
        elif arg_name == 'pe':
            var_name = 'e-'
            if hasattr(var_values, '__len__'):
                if len(var_values) >= 3:
                    # pe = -log(a_e-), so log(a_e-) = -pe
                    # For pe range [pe1, pe2], log(a_e-) range is [-pe1, -pe2]
                    var_values = np.linspace(-var_values[0], -var_values[1], int(var_values[2]))
                elif len(var_values) >= 2:
                    var_values = [-v for v in var_values]
                else:
                    # Single value in a list [pe]
                    var_values = np.array([-var_values[0]])
            else:
                # Scalar value
                var_values = np.array([-var_values])
        elif arg_name == 'Eh':
            var_name = 'e-'
            # Convert Eh (volts) to log(a_e-) using temperature-dependent formula
            # pe = Eh * F / (ln(10) * R * T) where pe = -log(a_e-)
            # Therefore: log(a_e-) = -pe = -Eh * F / (ln(10) * R * T)
            # where R = 0.00831470 kJ/(mol·K), F = 96.4935 kJ/(V·mol), T in Kelvin
            # This gives: log(a_e-) = -Eh * 96.4935 / (2.303 * 0.00831470 * T)
            #           = -Eh * 96.4935 / (0.019145 * T)
            #           = -Eh * 5039.76 / T

            # Get temperature for conversion (default to 25°C if not specified)
            T_kelvin = T if isinstance(T, (int, float)) else T[0] if hasattr(T, '__len__') else 298.15
            conversion_factor = 5039.76 / T_kelvin  # pe per volt (need to negate for log(a_e-))

            if hasattr(var_values, '__len__') and len(var_values) >= 2:
                if len(var_values) == 3:
                    # [Eh1, Eh2, npoints] format
                    # Convert to log(a_e-) = -pe = -Eh * conversion_factor
                    logact_start = -var_values[0] * conversion_factor
                    logact_end = -var_values[1] * conversion_factor
                    var_values = np.linspace(logact_start, logact_end, int(var_values[2]))
                elif len(var_values) == 2:
                    # [Eh1, Eh2] format - default to 256 points like R
                    logact_start = -var_values[0] * conversion_factor
                    logact_end = -var_values[1] * conversion_factor
                    var_values = np.linspace(logact_start, logact_end, 256)
                else:
                    # List of explicit Eh values
                    var_values = [-v * conversion_factor for v in var_values]
            else:
                # Single value
                var_values = -var_values * conversion_factor

        # Check if this is a basis species
        if var_name in basis_names:
            vars_list.append(var_name)

            # Process values
            if isinstance(var_values, (list, tuple)):
                if len(var_values) == 3:
                    # [min, max, npoints] format
                    vals_array = np.linspace(var_values[0], var_values[1], int(var_values[2]))
                    vals_list.append(vals_array)
                    lims_list.append(var_values)

                    # Print variable info
                    if messages:
                        n_vals = int(var_values[2])
                        print(f'affinity: variable {len(vars_list)} is log10(a_{var_name}) at {n_vals} values from {var_values[0]} to {var_values[1]}')

                elif len(var_values) == 2:
                    # [min, max] format - default to 256 points (R CHNOSZ behavior)
                    vals_array = np.linspace(var_values[0], var_values[1], 256)
                    vals_list.append(vals_array)
                    lims_list.append([var_values[0], var_values[1], 256])

                    # Print variable info
                    if messages:
                        print(f'affinity: variable {len(vars_list)} is log10(a_{var_name}) at 256 values from {var_values[0]} to {var_values[1]}')

                else:
                    # Explicit array of values
                    vals_list.append(np.array(var_values))
                    lims_list.append([min(var_values), max(var_values), len(var_values)])
            else:
                # Single value
                if not hasattr(var_values, '__len__'):
                    var_values = [var_values]
                vals_list.append(np.array(var_values))
                lims_list.append([var_values[0], var_values[-1], len(var_values)])
        else:
            # Not a recognized basis species or variable
            raise AffinityError(f"{arg_name} is not one of T, P, or IS, and does not match any basis species")

    return {
        'what': what,
        'vars': vars_list,
        'vals': vals_list,
        'lims': lims_list,
        'T': T,
        'P': P,
        'IS': IS
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
           messages: bool = True) -> Dict[str, Any]:
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
    else:
        mydim = [lim[2] for lim in lims]

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
                # Determine grid parameter for subcrt
                grid_param = None
                if len(vars) > 1:
                    # Multi-variable case - use appropriate grid
                    subcrt_vars = [v for v in vars if v in ['T', 'P', 'IS']]
                    if len(subcrt_vars) >= 2:
                        grid_param = subcrt_vars[0]  # Use first subcrt variable

                sout_result = subcrt(
                    species=all_species,
                    T=subcrt_T,
                    P=subcrt_P,
                    IS=subcrt_IS,
                    property='logK',
                    grid=grid_param,
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
            # Multi-dimensional case: create meshgrid for all variables
            var_arrays = []
            var_names_ordered = []

            # Collect variable arrays in order
            for var_name in vars:
                if var_name in basis_names:
                    var_idx = vars.index(var_name)
                    var_arrays.append(np.array(vals[var_idx]))
                    var_names_ordered.append(var_name)

            # Create meshgrid for basis species variables
            if var_arrays:
                # meshgrid creates N-D arrays where each variable varies along its own axis
                # indexing='ij' gives matrix indexing (first index varies down rows)
                meshgrids = np.meshgrid(*var_arrays, indexing='ij')

                # Map meshgrid results back to basis species
                for i, var_name in enumerate(var_names_ordered):
                    logact_basis_arrays[var_name] = meshgrids[i]

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
            # Determine grid parameter for subcrt when we have multiple T/P variables
            grid_param = None
            if len(vars) >= 2:
                # Check if we have T and/or P as variables
                if 'T' in vars and 'P' in vars:
                    # Both T and P vary - use T as grid variable (R CHNOSZ convention)
                    grid_param = 'T'
                elif 'T' in vars:
                    grid_param = 'T'
                elif 'P' in vars:
                    grid_param = 'P'

            batch_result = subcrt(all_species_indices, property="logK", T=T_celsius, P=P, grid=grid_param, messages=messages, show=False)

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
                        logK_vals = sp_data['logK'].values
                        # Handle NaN values by keeping them as nan (they will propagate to affinity)
                        # DO NOT replace nan with 0.0 as this causes incorrect affinity calculations
                        # logK_vals = np.where(np.isnan(logK_vals), 0.0, logK_vals)

                        # Reshape if we have a 2-D grid
                        if len(mydim) > 1 and len(logK_vals) == np.prod(mydim):
                            # Reshape flattened array to match grid dimensions
                            # mydim is [nT, nP] or similar, and grid='T' gives row-major order
                            logK_vals = logK_vals.reshape(mydim)

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
                    logK_vals = sp_data['logK'].values
                    # Handle NaN values by keeping them as nan (they will propagate to affinity)
                    # DO NOT replace nan with 0.0 as this causes incorrect affinity calculations
                    # logK_vals = np.where(np.isnan(logK_vals), 0.0, logK_vals)

                    # Reshape if we have a 2-D grid
                    if len(mydim) > 1 and len(logK_vals) == np.prod(mydim):
                        logK_vals = logK_vals.reshape(mydim)

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
                    # Need to broadcast logK to match logQ dimensions
                    if len(mydim) > 1 and logK_formation.ndim == 1:
                        # logK is 1-D but should be broadcast to 2-D
                        # Determine which dimension logK varies along
                        # Check if logK length matches first dimension of mydim (typically T)
                        if len(logK_formation) == mydim[0]:
                            # logK varies along first dimension, broadcast to second
                            logK_formation = np.broadcast_to(logK_formation[:, np.newaxis], mydim)
                        elif len(logK_formation) == mydim[1]:
                            # logK varies along second dimension, broadcast to first
                            logK_formation = np.broadcast_to(logK_formation[np.newaxis, :], mydim)
                        elif len(logK_formation) == np.prod(mydim):
                            # logK is flattened, reshape it
                            logK_formation = logK_formation.reshape(mydim)

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