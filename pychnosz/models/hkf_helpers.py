"""
Helper functions for HKF calculations, integrated from HKF_cgl.py.

This module contains the tested helper functions from the HKF_cgl.py notebook,
including calc_logK, calc_G_TP, G2logK, dissrxn2logK, and OBIGT2eos functions.
"""

import pandas as pd
import numpy as np
import math
import copy
import warnings
from .hkf import hkf
from .cgl import cgl
from .water import water
from ..utils.formula import entropy

def calc_logK(OBIGT_df, Tc, P, TP_i, water_model):
    
    OBIGT_TP, rows_added = calc_G_TP(OBIGT_df, Tc, P, water_model)
    
    dissrxn2logK_out = []
    for i in OBIGT_TP.index:
        dissrxn2logK_out.append(dissrxn2logK(OBIGT_TP, i, Tc))
    assert len(dissrxn2logK_out) == OBIGT_TP.shape[0]
    
    OBIGT_TP['dissrxn_logK_'+str(TP_i)] = dissrxn2logK_out
    
    # remove any rows added by calc_G_TP
    OBIGT_TP.drop(OBIGT_TP.tail(rows_added).index, inplace = True)
    
    return OBIGT_TP


def calc_G_TP(OBIGT, Tc, P, water_model):
    
    aq_out, H2O_Pt = hkf(property=["G"], parameters=OBIGT,
                         T=273.15+Tc, P=P, contrib=["n", "s", "o"],
                         H2O_props=["rho"], water_model=water_model)
    
    cgl_out = cgl(property=["G"], parameters=OBIGT, T=273.15+Tc, P=P)
    
    aq_col = pd.DataFrame.from_dict(aq_out, orient="index")
    cgl_col = pd.DataFrame.from_dict(cgl_out, orient="index")

    G_TP_df = pd.concat([aq_col, cgl_col], axis=1)
    G_TP_df.columns = ['aq','cgl']
    
    OBIGT["G_TP"] = G_TP_df['aq'].combine_first(G_TP_df['cgl'])
    
    rows_added = 0

    # add a row for water
    if "H2O" not in list(OBIGT["name"]):
        # Set the water model (without printing messages)
        water(water_model, messages=False)
        # water() returns a scalar when called with single property and scalar T, P
        # The result is in J/mol, need to convert to cal/mol by dividing by 4.184
        G_water = water("G", T=Tc+273.15, P=P, messages=False)
        # Handle both scalar and DataFrame returns
        if isinstance(G_water, pd.DataFrame):
            G_water_cal = G_water.iloc[0]["G"] / 4.184
        else:
            G_water_cal = float(G_water) / 4.184
        OBIGT = pd.concat([OBIGT, pd.DataFrame({"name": "H2O", "tag": "nan", "G_TP": G_water_cal}, index=[OBIGT.shape[0]])], ignore_index=True)
        rows_added += 1

    # add a row for protons
    if "H+" not in list(OBIGT["name"]):
        OBIGT = pd.concat([OBIGT, pd.DataFrame({"name": "H+", "tag": "nan", "G_TP": 0}, index=[OBIGT.shape[0]])], ignore_index=True)
        rows_added += 1

    return OBIGT, rows_added
    


def _extract_scalar(val):
    """
    Extract Python scalar from pandas/numpy value.
    Handles numpy scalars, 0-d arrays, and Python scalars.
    """
    return val.item() if hasattr(val, 'item') else val


def G2logK(G, Tc):
    # Gas constant R is in cal/mol K
    return G / (-math.log(10) * 1.9872 * (273.15+Tc))


def dissrxn2logK(OBIGT, i, Tc):
    
    this_dissrxn = OBIGT.iloc[i, OBIGT.columns.get_loc('dissrxn')]
    
    if this_dissrxn == "nan":
        this_dissrxn = OBIGT.iloc[i, OBIGT.columns.get_loc('regenerate_dissrxn')]
    
#     print(OBIGT["name"][i], this_dissrxn)
    
    try:
        this_dissrxn = this_dissrxn.strip()
        split_dissrxn = this_dissrxn.split(" ")
    except:
        return float('NaN')
    
    
    
    coeff = [float(n) for n in split_dissrxn[::2]]
    species = split_dissrxn[1::2]
    try:
        G = sum([c * _extract_scalar(OBIGT.loc[OBIGT["name"]==sp, "G_TP"].values[0]) for c,sp in zip(coeff, species)])
    except:
        G_list = []
        for ii, sp in enumerate(species):
            G_TP = OBIGT.loc[OBIGT["name"]==sp, "G_TP"]
            if len(G_TP) == 1:
                G_list.append(coeff[ii] * _extract_scalar(OBIGT.loc[OBIGT["name"]==sp, "G_TP"].values[0]))
            else:
                ### check valid polymorph T

                # get polymorph entries of OBIGT that match mineral
                poly_df = copy.copy(OBIGT.loc[OBIGT["name"]==sp,:])
                # ensure polymorph df is sorted according to cr, cr2, cr3... etc.
                poly_df = poly_df.sort_values("state")

                z_Ts = list(poly_df.loc[poly_df["name"]==sp, "z.T"])

                last_t = float('-inf')
                appended=False
                for iii,t in enumerate(z_Ts):

                    if Tc+273.15 > last_t and Tc+273.15 < t:
                        G_list.append(coeff[ii] * _extract_scalar(poly_df.loc[poly_df["name"]==sp, "G_TP"].values[iii]))
                        appended=True
                    if not appended and z_Ts[-1] == t:
                        G_list.append(coeff[ii] * _extract_scalar(poly_df.loc[poly_df["name"]==sp, "G_TP"].values[iii]))
                    last_t = t

        G = sum(G_list)

    return G2logK(G, Tc)


def convert_cm3bar(value):
    return value*4.184 * 10


def OBIGT2eos(OBIGT, fixGHS=True, tocal=True, messages=True):
    """
    Convert OBIGT dataframe to equation of state parameters.

    This function processes the OBIGT thermodynamic database to prepare it for
    equation of state calculations. It handles energy unit conversions and
    optionally fills in missing G, H, or S values.

    Parameters
    ----------
    OBIGT : pd.DataFrame
        OBIGT thermodynamic database
    fixGHS : bool, default True
        Fill in one missing value among G, H, S using thermodynamic relations
    tocal : bool, default True
        Convert energy units from Joules to calories
    messages : bool, default True
        Print informational messages (currently not used, reserved for future)

    Returns
    -------
    pd.DataFrame
        Modified OBIGT dataframe with converted parameters
    """
    OBIGT_out = OBIGT.copy()

    # Get column indices for named columns (to handle varying column positions)
    G_idx = OBIGT.columns.get_loc('G')
    H_idx = OBIGT.columns.get_loc('H')
    S_idx = OBIGT.columns.get_loc('S')
    Cp_idx = OBIGT.columns.get_loc('Cp')
    V_idx = OBIGT.columns.get_loc('V')
    omega_lambda_idx = OBIGT.columns.get_loc('omega.lambda')

    for i in range(0, OBIGT.shape[0]):

        # we only convert omega for aqueous species, not lambda for cgl species
        if tocal and OBIGT.iloc[i, :]["E_units"] == "J" and OBIGT.iloc[i, :]["state"] == "aq":
            # Convert G, H, S, Cp
            OBIGT_out.iloc[i, G_idx:Cp_idx+1] = OBIGT.iloc[i, G_idx:Cp_idx+1]/4.184
            # Convert V through omega (includes omega for aq species)
            OBIGT_out.iloc[i, V_idx:omega_lambda_idx+1] = OBIGT.iloc[i, V_idx:omega_lambda_idx+1]/4.184
            OBIGT_out.iloc[i, OBIGT.columns.get_loc('E_units')] = "cal"

        elif tocal and OBIGT.iloc[i, :]["E_units"] == "J":
            # Convert G, H, S, Cp
            OBIGT_out.iloc[i, G_idx:Cp_idx+1] = OBIGT.iloc[i, G_idx:Cp_idx+1]/4.184
            # Convert V through c2.f (exclude omega.lambda for non-aq species)
            OBIGT_out.iloc[i, V_idx:omega_lambda_idx] = OBIGT.iloc[i, V_idx:omega_lambda_idx]/4.184
            OBIGT_out.iloc[i, OBIGT.columns.get_loc('E_units')] = "cal"

        # fill in one of missing G, H, S
        # for use esp. by subcrt because NA for one of G, H or S
        # will preclude calculations at high T
        if fixGHS:
            # which entries are missing just one
            GHS_values = [OBIGT.iloc[i, G_idx], OBIGT.iloc[i, H_idx], OBIGT.iloc[i, S_idx]]
            imiss = [pd.isna(v) for v in GHS_values]
            if sum(imiss) == 1:

                ii = imiss.index(True)

                if ii == 0:  # G is missing
                    H = OBIGT_out.iloc[i, H_idx]
                    S = OBIGT_out.iloc[i, S_idx]
                    Selem = entropy(OBIGT_out.iloc[i, OBIGT_out.columns.get_loc('formula')])
                    T = 298.15
                    G = H - T*(S - Selem)
                    OBIGT_out.iloc[i, G_idx] = G
                elif ii == 1:  # H is missing
                    G = OBIGT_out.iloc[i, G_idx]
                    S = OBIGT_out.iloc[i, S_idx]
                    Selem = entropy(OBIGT_out.iloc[i, OBIGT_out.columns.get_loc('formula')])
                    T = 298.15
                    H = G + T*(S - Selem)
                    OBIGT_out.iloc[i, H_idx] = H
                elif ii == 2:  # S is missing
                    G = OBIGT_out.iloc[i, G_idx]
                    H = OBIGT_out.iloc[i, H_idx]
                    Selem = entropy(OBIGT_out.iloc[i, OBIGT_out.columns.get_loc('formula')])
                    T = 298.15
                    S = Selem + (H - G)/T
                    OBIGT_out.iloc[i, S_idx] = S

    return OBIGT_out