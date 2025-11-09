"""
Berman mineral equations of state implementation.

Calculate thermodynamic properties of minerals using equations from:
Berman, R. G. (1988) Internally-consistent thermodynamic data for minerals
   in the system Na2O-K2O-CaO-MgO-FeO-Fe2O3-Al2O3-SiO2-TiO2-H2O-CO2.
   J. Petrol. 29, 445-522. https://doi.org/10.1093/petrology/29.2.445

This is a 1:1 Python replica of CHNOSZ-main/R/Berman.R
"""

import numpy as np
import pandas as pd
import math
from typing import Union, List, Optional
import warnings

from ..core.thermo import thermo


def Berman(name: str, T: Union[float, List[float]] = 298.15, P: Union[float, List[float]] = 1, 
           check_G: bool = False, calc_transition: bool = True, calc_disorder: bool = True) -> pd.DataFrame:
    """
    Calculate thermodynamic properties of minerals using Berman equations.
    
    Parameters
    ----------
    name : str
        Name of the mineral
    T : float or list, optional
        Temperature in Kelvin (default: 298.15)
    P : float or list, optional  
        Pressure in bar (default: 1)
    check_G : bool, optional
        Check consistency of G in data file (default: False)
    calc_transition : bool, optional
        Calculate polymorphic transition contributions (default: True)
    calc_disorder : bool, optional
        Calculate disorder contributions (default: True)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns T, P, G, H, S, Cp, V
    """
    
    # Reference temperature and pressure
    Pr = 1
    Tr = 298.15
    
    # Make T and P the same length
    if isinstance(T, (int, float)):
        T = [T]
    if isinstance(P, (int, float)):
        P = [P]

    # Convert to list if numpy array (to avoid element-wise multiplication bug)
    if isinstance(T, np.ndarray):
        T = T.tolist()
    if isinstance(P, np.ndarray):
        P = P.tolist()

    ncond = max(len(T), len(P))
    T = np.array(T * (ncond // len(T) + 1), dtype=float)[:ncond]
    P = np.array(P * (ncond // len(P) + 1), dtype=float)[:ncond]
    
    # Get parameters in the Berman equations
    # Start with thermodynamic parameters provided with CHNOSZ
    thermo_sys = thermo()
    if thermo_sys.Berman is None:
        raise RuntimeError("Berman data not loaded. Please run pychnosz.reset() first.")
    
    dat = thermo_sys.Berman.copy()
    
    # TODO: Handle user-supplied data file (thermo()$opt$Berman)
    # For now, just use the default data
    
    # Remove duplicates (only the first, i.e. most recent entry is kept)
    dat = dat.drop_duplicates(subset=['name'], keep='first')
    
    # Remove the multipliers on volume parameters
    vcols = ['v1', 'v2', 'v3', 'v4']  # columns with v1, v2, v3, v4
    multexp = [5, 5, 5, 8]
    for i, col in enumerate(vcols):
        if col in dat.columns:
            dat[col] = dat[col] / (10 ** multexp[i])
    
    # Which row has data for this mineral?
    matching_rows = dat[dat['name'] == name]
    if len(matching_rows) == 0:
        raise ValueError(f"Data for {name} not available in Berman database")
    
    dat_mineral = matching_rows.iloc[0]
    
    # Extract parameters for easier access
    GfPrTr = dat_mineral['GfPrTr']
    HfPrTr = dat_mineral['HfPrTr'] 
    SPrTr = dat_mineral['SPrTr']
    VPrTr = dat_mineral['VPrTr']
    
    k0 = dat_mineral['k0']
    k1 = dat_mineral['k1'] 
    k2 = dat_mineral['k2']
    k3 = dat_mineral['k3']
    k4 = dat_mineral['k4'] if not pd.isna(dat_mineral['k4']) else 0
    k5 = dat_mineral['k5'] if not pd.isna(dat_mineral['k5']) else 0
    k6 = dat_mineral['k6'] if not pd.isna(dat_mineral['k6']) else 0
    
    v1 = dat_mineral['v1'] if not pd.isna(dat_mineral['v1']) else 0
    v2 = dat_mineral['v2'] if not pd.isna(dat_mineral['v2']) else 0
    v3 = dat_mineral['v3'] if not pd.isna(dat_mineral['v3']) else 0
    v4 = dat_mineral['v4'] if not pd.isna(dat_mineral['v4']) else 0
    
    # Transition parameters
    Tlambda = dat_mineral['Tlambda'] if not pd.isna(dat_mineral['Tlambda']) else None
    Tref = dat_mineral['Tref'] if not pd.isna(dat_mineral['Tref']) else None
    dTdP = dat_mineral['dTdP'] if not pd.isna(dat_mineral['dTdP']) else None
    l1 = dat_mineral['l1'] if not pd.isna(dat_mineral['l1']) else None
    l2 = dat_mineral['l2'] if not pd.isna(dat_mineral['l2']) else None
    
    # Disorder parameters
    Tmin = dat_mineral['Tmin'] if not pd.isna(dat_mineral['Tmin']) else None
    Tmax = dat_mineral['Tmax'] if not pd.isna(dat_mineral['Tmax']) else None
    d0 = dat_mineral['d0'] if not pd.isna(dat_mineral['d0']) else None
    d1 = dat_mineral['d1'] if not pd.isna(dat_mineral['d1']) else None
    d2 = dat_mineral['d2'] if not pd.isna(dat_mineral['d2']) else None
    d3 = dat_mineral['d3'] if not pd.isna(dat_mineral['d3']) else None
    d4 = dat_mineral['d4'] if not pd.isna(dat_mineral['d4']) else None
    Vad = dat_mineral['Vad'] if not pd.isna(dat_mineral['Vad']) else None
    
    # Get the entropy of the elements using the chemical formula
    # Get formula from OBIGT and calculate using entropy() function like in R CHNOSZ
    SPrTr_elements = 0
    if thermo_sys.obigt is not None:
        obigt_match = thermo_sys.obigt[thermo_sys.obigt['name'] == name]
        if len(obigt_match) > 0:
            formula = obigt_match.iloc[0]['formula']
            # Import entropy function and calculate SPrTr_elements properly
            from ..utils.formula import entropy
            SPrTr_elements = entropy(formula)
    
    # Check that G in data file follows Benson-Helgeson convention
    if check_G and not pd.isna(GfPrTr):
        GfPrTr_calc = HfPrTr - Tr * (SPrTr - SPrTr_elements)
        Gdiff = GfPrTr_calc - GfPrTr
        if abs(Gdiff) >= 1000:
            warnings.warn(f"{name}: GfPrTr(calc) - GfPrTr(table) is too big! == {round(Gdiff)} J/mol")
    
    ### Thermodynamic properties ###
    # Calculate Cp and V (Berman, 1988 Eqs. 4 and 5)
    # k4, k5, k6 terms from winTWQ documentation (doi:10.4095/223425)
    Cp = k0 + k1 * T**(-0.5) + k2 * T**(-2) + k3 * T**(-3) + k4 * T**(-1) + k5 * T + k6 * T**2
    
    P_Pr = P - Pr
    T_Tr = T - Tr
    V = VPrTr * (1 + v1 * T_Tr + v2 * T_Tr**2 + v3 * P_Pr + v4 * P_Pr**2)
    
    # Calculate Ha (symbolically integrated using sympy - expressions not simplified)
    intCp = (T*k0 - Tr*k0 + k2/Tr - k2/T + k3/(2*Tr**2) - k3/(2*T**2) + 2.0*k1*T**0.5 - 2.0*k1*Tr**0.5 + 
             k4*np.log(T) - k4*np.log(Tr) + k5*T**2/2 - k5*Tr**2/2 - k6*Tr**3/3 + k6*T**3/3)
    
    intVminusTdVdT = (-VPrTr + P*(VPrTr + VPrTr*v4 - VPrTr*v3 - Tr*VPrTr*v1 + VPrTr*v2*Tr**2 - VPrTr*v2*T**2) +
                      P**2*(VPrTr*v3/2 - VPrTr*v4) + VPrTr*v3/2 - VPrTr*v4/3 + Tr*VPrTr*v1 + 
                      VPrTr*v2*T**2 - VPrTr*v2*Tr**2 + VPrTr*v4*P**3/3)
    
    Ha = HfPrTr + intCp + intVminusTdVdT
    
    # Calculate S (also symbolically integrated)
    intCpoverT = (k0*np.log(T) - k0*np.log(Tr) - k3/(3*T**3) + k3/(3*Tr**3) + k2/(2*Tr**2) - k2/(2*T**2) + 
                  2.0*k1*Tr**(-0.5) - 2.0*k1*T**(-0.5) + k4/Tr - k4/T + T*k5 - Tr*k5 + k6*T**2/2 - k6*Tr**2/2)
    
    intdVdT = -VPrTr*(v1 + v2*(-2*Tr + 2*T)) + P*VPrTr*(v1 + v2*(-2*Tr + 2*T))
    
    S = SPrTr + intCpoverT - intdVdT
    
    # Calculate Ga --> Berman-Brown convention (DG = DH - T*S, no S(element))
    Ga = Ha - T * S
    
    ### Polymorphic transition properties ###
    if (Tlambda is not None and Tref is not None and 
        not pd.isna(Tlambda) and not pd.isna(Tref) and 
        np.any(T > Tref) and calc_transition):
        
        # Starting transition contributions are 0
        Cptr = np.zeros(ncond)
        Htr = np.zeros(ncond)
        Str = np.zeros(ncond)
        
        # Eq. 9: Tlambda at P
        Tlambda_P = Tlambda + dTdP * (P - 1)
        
        # Eq. 8a: Cp at P
        Td = Tlambda - Tlambda_P
        Tprime = T + Td
        
        # With the condition that Tref < Tprime < Tlambda(1bar)
        iTprime = (Tref < Tprime) & (Tprime < Tlambda)
        # Handle NA values
        iTprime = iTprime & ~np.isnan(Tprime)
        
        if np.any(iTprime):
            Tprime_valid = Tprime[iTprime]
            Cptr[iTprime] = Tprime_valid * (l1 + l2 * Tprime_valid)**2
        
        # We got Cp, now calculate the integrations for H and S
        iTtr = T > Tref
        if np.any(iTtr):
            Ttr = T[iTtr].copy()
            Tlambda_P_tr = Tlambda_P[iTtr].copy()
            Td_tr = Td[iTtr] if hasattr(Td, '__len__') else np.full_like(Ttr, Td)
            
            # Handle NA values
            Tlambda_P_tr[np.isnan(Tlambda_P_tr)] = np.inf
            
            # The upper integration limit is Tlambda_P
            Ttr[Ttr >= Tlambda_P_tr] = Tlambda_P_tr[Ttr >= Tlambda_P_tr]
            
            # Derived variables
            tref = Tref - Td_tr
            x1 = l1**2 * Td_tr + 2 * l1 * l2 * Td_tr**2 + l2**2 * Td_tr**3
            x2 = l1**2 + 4 * l1 * l2 * Td_tr + 3 * l2**2 * Td_tr**2
            x3 = 2 * l1 * l2 + 3 * l2**2 * Td_tr
            x4 = l2**2
            
            # Eqs. 10, 11, 12
            Htr[iTtr] = (x1 * (Ttr - tref) + x2/2 * (Ttr**2 - tref**2) + 
                        x3/3 * (Ttr**3 - tref**3) + x4/4 * (Ttr**4 - tref**4))
            Str[iTtr] = (x1 * (np.log(Ttr) - np.log(tref)) + x2 * (Ttr - tref) + 
                        x3/2 * (Ttr**2 - tref**2) + x4/3 * (Ttr**3 - tref**3))
        
        Gtr = Htr - T * Str
        
        # Apply the transition contributions
        Ga = Ga + Gtr
        Ha = Ha + Htr
        S = S + Str
        Cp = Cp + Cptr
    
    ### Disorder thermodynamic properties ###
    if (Tmin is not None and Tmax is not None and 
        not pd.isna(Tmin) and not pd.isna(Tmax) and 
        np.any(T > Tmin) and calc_disorder):
        
        # Starting disorder contributions are 0
        Cpds = np.zeros(ncond)
        Hds = np.zeros(ncond)
        Sds = np.zeros(ncond)
        Vds = np.zeros(ncond)
        
        # The lower integration limit is Tmin
        iTds = T > Tmin
        if np.any(iTds):
            Tds = T[iTds].copy()
            # The upper integration limit is Tmax
            Tds[Tds > Tmax] = Tmax
            
            # Ber88 Eqs. 15, 16, 17
            Cpds[iTds] = d0 + d1*Tds**(-0.5) + d2*Tds**(-2) + d3*Tds + d4*Tds**2
            Hds[iTds] = (d0*(Tds - Tmin) + d1*(Tds**0.5 - Tmin**0.5)/0.5 +
                        d2*(Tds**(-1) - Tmin**(-1))/(-1) + d3*(Tds**2 - Tmin**2)/2 + d4*(Tds**3 - Tmin**3)/3)
            Sds[iTds] = (d0*(np.log(Tds) - np.log(Tmin)) + d1*(Tds**(-0.5) - Tmin**(-0.5))/(-0.5) +
                        d2*(Tds**(-2) - Tmin**(-2))/(-2) + d3*(Tds - Tmin) + d4*(Tds**2 - Tmin**2)/2)
        
        # Eq. 18; we can't do this if Vad == 0 (dolomite and gehlenite)
        if Vad is not None and not pd.isna(Vad) and Vad != 0:
            Vds = Hds / Vad
        
        # Include the Vds term with Hds
        Hds = Hds + Vds * (P - Pr)
        
        # Disordering properties above Tmax (Eq. 20)
        ihigh = T > Tmax
        if np.any(ihigh):
            Hds[ihigh] = Hds[ihigh] - (T[ihigh] - Tmax) * Sds[ihigh]
        
        Gds = Hds - T * Sds
        
        # Apply the disorder contributions
        Ga = Ga + Gds
        Ha = Ha + Hds
        S = S + Sds
        V = V + Vds
        Cp = Cp + Cpds
    
    ### (for testing) Use G = H - TS to check that integrals for H and S are written correctly
    Ga_fromHminusTS = Ha - T * S
    if not np.allclose(Ga_fromHminusTS, Ga, atol=1e-6):
        raise RuntimeError(f"{name}: incorrect integrals detected using DG = DH - T*S")
    
    ### Thermodynamic and unit conventions used in SUPCRT ###
    # Use entropy of the elements in calculation of G --> Benson-Helgeson convention (DG = DH - T*DS)
    Gf = Ga + Tr * SPrTr_elements
    
    # The output will just have "G" and "H"
    G = Gf
    H = Ha
    
    # Convert J/bar to cm^3/mol
    V = V * 10
    
    return pd.DataFrame({
        'T': T,
        'P': P, 
        'G': G,
        'H': H,
        'S': S,
        'Cp': Cp,
        'V': V
    })