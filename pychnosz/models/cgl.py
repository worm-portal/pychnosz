"""
CGL (Crystalline, Gas, Liquid) equation of state implementation.

This module implements equations of state for crystalline, gas, and liquid species
(except liquid water), based on the tested functions from HKF_cgl.py.

References:
- Helgeson, H.C. et al. (1978). Summary and critique of the thermodynamic properties
  of rock-forming minerals. Am. J. Sci. 278-A.
- Berman, R.G. (1988). Internally-consistent thermodynamic data for minerals in the
  system Na2O-K2O-CaO-MgO-FeO-Fe2O3-Al2O3-SiO2-TiO2-H2O-CO2. J. Petrol.
- R CHNOSZ cgl.R implementation
"""

import pandas as pd
import numpy as np
import math
import copy
import warnings

def convert_cm3bar(value):
    return value*4.184 * 10

# CHNOSZ/cgl.R
# calculate standard thermodynamic properties of non-aqueous species
# 20060729 jmd

def cgl(property = None, parameters = None, T = 298.15, P = 1):
    # calculate properties of crystalline, liquid (except H2O) and gas species
    Tr = 298.15
    Pr = 1

    # Convert T and P to arrays for vectorized operations
    T = np.atleast_1d(T)
    P = np.atleast_1d(P)

    # make T and P equal length
    if P.size < T.size:
        P = np.full_like(T, P[0] if P.size == 1 else P)
    if T.size < P.size:
        T = np.full_like(P, T[0] if T.size == 1 else T)

    n_conditions = T.size
    # initialize output dict
    out_dict = dict()
    # loop over each species
    
    # Iterate over each row by position to handle duplicate indices properly
    for i in range(len(parameters)):
        # Get the index label for this row
        k = parameters.index[i]
        # Get the row data by position (iloc) to avoid duplicate index issues
        PAR = parameters.iloc[i]

        if PAR["state"] == "aq":
            # For aqueous species processed by CGL, return NaN
            # (they should be processed by HKF instead)
            out_dict[k] = {p:float('NaN') for p in property}
        else:

            # OBIGT database stores G, H, S in calories (E_units = "cal")
            # CGL calculations use calories (integrals intCpdT, intCpdlnT, intVdP are in cal)
            # Results are output in calories and converted to J in subcrt.py at line 959

            # Parameter scaling - SUPCRT92 data is already in correct units
            # PAR["a2.b"] = copy.copy(PAR["a2.b"]*10**-3)
            # PAR["a3.c"] = copy.copy(PAR["a3.c"]*10**5) 
            # PAR["c1.e"] = copy.copy(PAR["c1.e"]*10**-5)

            # Check if this is a Berman mineral (columns 9-21 are all NA in R indexing)
            # In Python/pandas, we check the relevant thermodynamic parameter columns
            # NOTE: A mineral is only Berman if it LACKS standard thermodynamic data (G,H,S)
            # If G,H,S are present, use regular CGL even if heat capacity coefficients are all zero
            berman_cols = ['a1.a', 'a2.b', 'a3.c', 'a4.d', 'c1.e', 'c2.f', 'omega.lambda', 'z.T']
            has_standard_thermo = pd.notna(PAR.get('G', np.nan)) and pd.notna(PAR.get('H', np.nan)) and pd.notna(PAR.get('S', np.nan))
            all_coeffs_zero_or_na = all(pd.isna(PAR.get(col, np.nan)) or PAR.get(col, 0) == 0 for col in berman_cols)
            is_berman_mineral = all_coeffs_zero_or_na and not has_standard_thermo

            if is_berman_mineral:
                # Use Berman equations (parameters not in thermo()$OBIGT)
                from .berman import Berman
                try:
                    # Berman is already vectorized - pass T and P arrays directly
                    properties_df = Berman(PAR["name"], T=T, P=P)
                    # Extract the requested properties as arrays
                    values = {}
                    for prop in property:
                        if prop in properties_df.columns:
                            # Get all values as an array
                            prop_values = properties_df[prop].values

                            # IMPORTANT: Berman function returns values in J/mol (Joules)
                            # but CGL returns values in cal/mol (calories)
                            # Convert Berman results from J/mol to cal/mol for consistency
                            # Energy properties that need conversion: G, H, S, Cp
                            # Volume (V) and other properties don't need conversion
                            energy_props = ['G', 'H', 'S', 'Cp']
                            if prop in energy_props:
                                # Convert J/mol to cal/mol by dividing by 4.184
                                prop_values = prop_values / 4.184

                            values[prop] = prop_values
                        else:
                            values[prop] = np.full(n_conditions, float('NaN'))
                except Exception as e:
                    # If Berman calculation fails, fall back to NaN arrays
                    values = {prop: np.full(n_conditions, float('NaN')) for prop in property}
            else:
                # Use regular CGL equations
                
                # in CHNOSZ, we have
                # 1 cm^3 bar --> convert(1, "calories") == 0.02390057 cal
                # but REAC92D.F in SUPCRT92 uses
                cm3bar_to_cal = 0.023901488 # cal
                # start with NA values
                values = dict()
                # a test for availability of heat capacity coefficients (a, b, c, d, e, f)
                # based on the column assignments in thermo()$OBIGT

                # Check for heat capacity coefficients, handling both NaN and non-numeric values
                # Heat capacity coefficients are at positions 14-19 (a1.a through c2.f)
                # Position 13 is V (volume), not a heat capacity coefficient
                has_hc_coeffs = False
                try:
                    hc_values = list(PAR.iloc[14:20])
                    has_hc_coeffs = any([pd.notna(p) and p != 0 for p in hc_values if pd.api.types.is_numeric_dtype(type(p))])

                    # DEBUG
                    if False and PAR["name"] == "rhomboclase":
                        print(f"DEBUG for rhomboclase:")
                        print(f"  hc_values (iloc[14:20]): {hc_values}")
                        print(f"  has_hc_coeffs: {has_hc_coeffs}")
                except Exception as e:
                    has_hc_coeffs = False

                if has_hc_coeffs:
                    # we have at least one of the heat capacity coefficients;
                    # zero out any NA's in the rest (leave lambda and T of transition (columns 20-21) alone)
                    for i in range(14, 20):
                        if pd.isna(PAR.iloc[i]) or not pd.api.types.is_numeric_dtype(type(PAR.iloc[i])):
                            PAR.iloc[i] = 0.0
                    # calculate the heat capacity and its integrals (vectorized)
                    Cp = PAR["a1.a"] + PAR["a2.b"]*T + PAR["a3.c"]*T**-2 + PAR["a4.d"]*T**-0.5 + PAR["c1.e"]*T**2 + PAR["c2.f"]*T**PAR["omega.lambda"]
                    intCpdT = PAR["a1.a"]*(T - Tr) + PAR["a2.b"]*(T**2 - Tr**2)/2 + PAR["a3.c"]*(1/T - 1/Tr)/-1 + PAR["a4.d"]*(T**0.5 - Tr**0.5)/0.5 + PAR["c1.e"]*(T**3-Tr**3)/3
                    intCpdlnT = PAR["a1.a"]*np.log(T / Tr) + PAR["a2.b"]*(T - Tr) + PAR["a3.c"]*(T**-2 - Tr**-2)/-2 + PAR["a4.d"]*(T**-0.5 - Tr**-0.5)/-0.5  + PAR["c1.e"]*(T**2 - Tr**2)/2

                    # do we also have the lambda parameter (Cp term with adjustable exponent on T)?
                    if pd.notna(PAR["omega.lambda"]) and PAR["omega.lambda"] != 0:
                        # equations for lambda adapted from Helgeson et al., 1998 (doi:10.1016/S0016-7037(97)00219-6)
                        if PAR["omega.lambda"] == -1:
                            intCpdT = intCpdT + PAR["c2.f"]*np.log(T/Tr)
                        else:
                            intCpdT = intCpdT - PAR["c2.f"]*( T**(PAR["omega.lambda"] + 1) - Tr**(PAR["omega.lambda"] + 1) ) / (PAR["omega.lambda"] + 1)
                        intCpdlnT = intCpdlnT + PAR["c2.f"]*(T**PAR["omega.lambda"] - Tr**PAR["omega.lambda"]) / PAR["omega.lambda"]

                else:
                    # use constant heat capacity if the coefficients are not available (vectorized)
                    # If Cp is NA/NaN, use 0 (matching R CHNOSZ behavior)
                    Cp_value = PAR["Cp"] if pd.notna(PAR["Cp"]) else 0.0
                    Cp = np.full(n_conditions, Cp_value)
                    intCpdT = Cp_value*(T - Tr)
                    intCpdlnT = Cp_value*np.log(T / Tr)
                    # in case Cp is listed as NA, set the integrals to 0 at Tr
                    at_Tr = (T == Tr)
                    intCpdT = np.where(at_Tr, 0, intCpdT)
                    intCpdlnT = np.where(at_Tr, 0, intCpdlnT)


                # volume and its integrals (vectorized)
                if PAR["name"] in ["quartz", "coesite"]:
                    # volume calculations for quartz and coesite
                    qtz = quartz_coesite(PAR, T, P)
                    V = qtz["V"]
                    intVdP = qtz["intVdP"]
                    intdVdTdP = qtz["intdVdTdP"]

                else:
                    # for other minerals, volume is constant (Helgeson et al., 1978)
                    V = np.full(n_conditions, PAR["V"])
                    # if the volume is NA, set its integrals to zero
                    if pd.isna(PAR["V"]):
                        intVdP = np.zeros(n_conditions)
                        intdVdTdP = np.zeros(n_conditions)
                    else:
                        intVdP = PAR["V"]*(P - Pr) * cm3bar_to_cal
                        intdVdTdP = np.zeros(n_conditions)

                # get the values of each of the requested thermodynamic properties (vectorized)
                for i,prop in enumerate(property):
                    if prop == "Cp": values["Cp"] = Cp
                    if prop == "V": values["V"] = V
                    if prop == "E": values["E"] = np.full(n_conditions, float('NaN'))
                    if prop == "kT": values["kT"] = np.full(n_conditions, float('NaN'))
                    if prop == "G":
                        # calculate S * (T - Tr), but set it to 0 at Tr (in case S is NA)
                        Sterm = PAR["S"]*(T - Tr)
                        Sterm = np.where(T == Tr, 0, Sterm)

                        # DEBUG
                        if False and PAR["name"] == "iron" and PAR.get("state") == "cr4":
                            print(f"DEBUG G calculation for {PAR['name']} {PAR.get('state', 'unknown')}:")
                            print(f"  PAR['G'] = {PAR['G']}")
                            print(f"  PAR['S'] = {PAR['S']}")
                            print(f"  model = {PAR.get('model', 'unknown')}")
                            print(f"  Sterm[0] = {Sterm[0] if hasattr(Sterm, '__len__') else Sterm}")
                            print(f"  intCpdT[0] = {intCpdT[0] if hasattr(intCpdT, '__len__') else intCpdT}")
                            print(f"  T[0]*intCpdlnT[0] = {(T[0]*intCpdlnT[0]) if hasattr(intCpdlnT, '__len__') else T*intCpdlnT}")
                            print(f"  intVdP[0] = {intVdP[0] if hasattr(intVdP, '__len__') else intVdP}")
                            G_calc = PAR['G'] - Sterm + intCpdT - T*intCpdlnT + intVdP
                            print(f"  G[0] (before subcrt conversion) = {G_calc[0] if hasattr(G_calc, '__len__') else G_calc}")

                        values["G"] = PAR["G"] - Sterm + intCpdT - T*intCpdlnT + intVdP
                    if prop == "H":
                        values["H"] = PAR["H"] + intCpdT + intVdP - T*intdVdTdP
                    if prop == "S": values["S"] = PAR["S"] + intCpdlnT - intdVdTdP

            out_dict[k] = values # species have to be numbered instead of named because of name repeats (e.g., cr polymorphs)

    return out_dict


### unexported function ###

# calculate GHS and V corrections for quartz and coesite 20170929
# (these are the only mineral phases for which SUPCRT92 uses an inconstant volume)
def quartz_coesite(PAR, T, P):
    # the corrections are 0 for anything other than quartz and coesite
    if not PAR["name"] in ["quartz", "coesite"]:
        n = T.size if isinstance(T, np.ndarray) else 1
        return(dict(G=np.zeros(n), H=np.zeros(n), S=np.zeros(n), V=np.zeros(n)))

    # Vectorized version
    T = np.atleast_1d(T)
    P = np.atleast_1d(P)

    # Tr, Pr and TtPr (transition temperature at Pr)
    Pr = 1      # bar
    Tr = 298.15 # K
    TtPr = 848  # K
    # constants from SUP92D.f
    aa = 549.824
    ba = 0.65995
    ca = -0.4973e-4
    VPtTta = 23.348
    VPrTtb = 23.72
    Stran = 0.342
    # constants from REAC92D.f
    VPrTra = 22.688 # VPrTr(a-quartz)
    Vdiff = 2.047   # VPrTr(a-quartz) - VPrTr(coesite)
    k = 38.5       # dPdTtr(a/b-quartz)
    #k <- 38.45834    # calculated in CHNOSZ: dPdTtr(info("quartz"))
    # code adapted from REAC92D.f
    qphase = PAR["state"].replace("cr", "")

    if qphase == "2":
        Pstar = P.copy()
        Sstar = np.zeros_like(T)
        V = np.full_like(T, VPrTtb)
    else:
        Pstar = Pr + k * (T - TtPr)
        Sstar = np.full_like(T, Stran)
        V = VPrTra + ca*(P-Pr) + (VPtTta - VPrTra - ca*(P-Pr))*(T-Tr) / (TtPr + (P-Pr)/k - Tr)

    # Apply condition: if T < TtPr
    below_transition = T < TtPr
    Pstar = np.where(below_transition, Pr, Pstar)
    Sstar = np.where(below_transition, 0, Sstar)

    if PAR["name"] == "coesite":
        VPrTra = VPrTra - Vdiff
        VPrTtb = VPrTtb - Vdiff
        V = V - Vdiff

    cm3bar_to_cal = 0.023901488

    # Vectorized log calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        log_term = np.log((aa + P/k) / (aa + Pstar/k))
        log_term = np.where(np.isfinite(log_term), log_term, 0)

    GVterm = cm3bar_to_cal * (VPrTra * (P - Pstar) + VPrTtb * (Pstar - Pr) - \
        0.5 * ca * (2 * Pr * (P - Pstar) - (P**2 - Pstar**2)) - \
        ca * k * (T - Tr) * (P - Pstar) + \
        k * (ba + aa * ca * k) * (T - Tr) * log_term)
    SVterm = cm3bar_to_cal * (-k * (ba + aa * ca * k) * log_term + ca * k * (P - Pstar)) - Sstar

    # note the minus sign on "SVterm" in order that intdVdTdP has the correct sign
    return dict(intVdP=GVterm, intdVdTdP=-SVterm, V=V)


def lambda_transition(T: np.ndarray, T_lambda: float, lambda_val: float, 
                     sigma: float = 50.0):
    """
    Calculate lambda transition contributions to thermodynamic properties.
    
    Parameters
    ----------
    T : array
        Temperature in Kelvin
    T_lambda : float
        Lambda transition temperature in Kelvin
    lambda_val : float
        Lambda transition parameter
    sigma : float
        Width parameter for transition (default: 50 K)
        
    Returns
    -------
    dict
        Dictionary with lambda contributions to Cp, H, S, G
    """
    
    # Gaussian approximation for lambda transition
    gaussian = np.exp(-(T - T_lambda)**2 / (2 * sigma**2))
    
    # Heat capacity contribution
    Cp_lambda = lambda_val * gaussian
    
    # Enthalpy contribution (integrated Cp)
    H_lambda = np.where(T > T_lambda, 
                       lambda_val * sigma * np.sqrt(2*np.pi) * 0.5 * (1 + np.tanh((T-T_lambda)/sigma)),
                       0)
    
    # Entropy contribution (integrated Cp/T)
    S_lambda = np.where(T > T_lambda, lambda_val / T_lambda, 0)
    
    # Gibbs energy contribution
    G_lambda = H_lambda - T * S_lambda
    
    return {
        'Cp': Cp_lambda,
        'H': H_lambda, 
        'S': S_lambda,
        'G': G_lambda
    }


def berman_properties(T: np.ndarray, P: np.ndarray, parameters: pd.Series):
    """
    Calculate properties using Berman (1988) equations for minerals.
    
    This is a simplified version of the Berman model - full implementation would
    include all coefficients and corrections from Berman (1988).
    """
    
    # Standard state values
    H0 = parameters.get('H', 0.0)
    S0 = parameters.get('S', 0.0) 
    V0 = parameters.get('V', 0.0)
    
    # Berman heat capacity coefficients (k0-k3)
    k0 = parameters.get('k0', parameters.get('Cp', 0.0))
    k1 = parameters.get('k1', 0.0)
    k2 = parameters.get('k2', 0.0) 
    k3 = parameters.get('k3', 0.0)
    
    # Heat capacity: Cp = k0 + k1/T^0.5 + k2/T^2 + k3/T^3
    Cp_calc = k0 + k1 / np.sqrt(T) + k2 / T**2 + k3 / T**3
    
    # Integrate for H and S (simplified)
    Tr = 298.15
    
    # Enthalpy integration (approximate)
    H_calc = H0 + k0 * (T - Tr)
    
    # Entropy integration (approximate) 
    S_calc = S0 + k0 * np.log(T/Tr)
    
    # Gibbs energy
    G_calc = H_calc - T * S_calc
    
    # Volume (assume constant)
    V_calc = np.full_like(T, V0)
    
    return {
        'G': G_calc,
        'H': H_calc,
        'S': S_calc, 
        'Cp': Cp_calc,
        'V': V_calc
    }