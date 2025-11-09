"""
Unit conversion utilities for CHNOSZ.

This module provides functions for converting between different units commonly
used in geochemical thermodynamics, such as temperature (C/K), pressure (bar/MPa),
energy (cal/J), and electrochemical potentials (Eh/pe).

Based on R CHNOSZ util.units.R
"""

import numpy as np
from typing import Union, List, Optional
import warnings

from ..core.thermo import thermo
from ..core.subcrt import subcrt


def convert(value: Union[float, np.ndarray, List[float]],
            units: str,
            T: Union[float, np.ndarray] = 298.15,
            P: Union[float, np.ndarray] = 1,
            pH: Union[float, np.ndarray] = 7,
            logaH2O: Union[float, np.ndarray] = 0,
            messages: bool = True) -> Union[float, np.ndarray]:
    """
    Convert values to the specified units.

    This function converts thermodynamic values between different units commonly
    used in geochemistry.

    Parameters
    ----------
    value : float, ndarray, or list
        Value(s) to convert
    units : str
        Target units. Options include:
        - Temperature: 'C', 'K'
        - Energy: 'J', 'cal'
        - Pressure: 'bar', 'MPa'
        - Thermodynamic: 'G', 'logK'
        - Electrochemical: 'Eh', 'pe', 'E0', 'logfO2'
        - Volume: 'cm3bar', 'joules'
    T : float or ndarray, default 298.15
        Temperature in K (for Eh/pe/logK conversions)
    P : float or ndarray, default 1
        Pressure in bar (for E0/logfO2 conversions)
    pH : float or ndarray, default 7
        pH value (for E0/logfO2 conversions)
    logaH2O : float or ndarray, default 0
        Log activity of water (for E0/logfO2 conversions)
    messages : bool, default True
        Whether to print informational messages

    Returns
    -------
    float or ndarray
        Converted value(s)

    Examples
    --------
    >>> convert(25, 'K')  # Convert 25°C to K
    298.15
    >>> convert(1.0, 'pe', T=298.15)  # Convert 1V Eh to pe
    16.9
    """

    if value is None:
        return None

    # Convert to numpy array for uniform handling
    value = np.asarray(value)
    T = np.asarray(T)
    P = np.asarray(P)
    pH = np.asarray(pH)
    logaH2O = np.asarray(logaH2O)

    units = units.lower()

    # Temperature conversions (C <-> K)
    if units in ['c', 'k']:
        CK = 273.15
        if units == 'k':
            return value + CK
        if units == 'c':
            return value - CK

    # Energy conversions (J <-> cal)
    elif units in ['j', 'cal']:
        Jcal = 4.184
        if units == 'j':
            return value * Jcal
        if units == 'cal':
            return value / Jcal

    # Gibbs energy <-> logK conversions
    elif units in ['g', 'logk']:
        # Gas constant (J K^-1 mol^-1)
        R = 8.314463  # NIST value
        if units == 'logk':
            return value / (-np.log(10) * R * T)
        if units == 'g':
            return value * (-np.log(10) * R * T)

    # Volume conversions (cm3bar <-> joules)
    elif units in ['cm3bar', 'joules']:
        if units == 'cm3bar':
            return value * 10
        if units == 'joules':
            return value / 10

    # Electrochemical potential conversions (Eh <-> pe)
    elif units in ['eh', 'pe']:
        R = 0.00831470  # Gas constant in kJ K^-1 mol^-1
        F = 96.4935     # Faraday constant in kJ V^-1 mol^-1
        if units == 'pe':
            return value * F / (np.log(10) * R * T)
        if units == 'eh':
            return value * (np.log(10) * R * T) / F

    # Pressure conversions (bar <-> MPa)
    elif units in ['bar', 'mpa']:
        barmpa = 10
        if units == 'mpa':
            return value / barmpa
        if units == 'bar':
            return value * barmpa

    # Eh <-> logfO2 conversions
    elif units in ['e0', 'logfo2']:
        # Calculate equilibrium constant for: H2O = 1/2 O2 + 2 H+ + 2 e-
        # Handle P="Psat" case (pass it directly to subcrt)
        # Check if P is a string (including numpy string types)
        P_is_psat = False
        if isinstance(P, (str, np.str_)):
            P_is_psat = str(P).lower() == 'psat'
        elif isinstance(P, (list, tuple)):
            # P is a list/tuple - check if it's a single-element string
            if len(P) == 1 and isinstance(P[0], (str, np.str_)):
                P_is_psat = str(P[0]).lower() == 'psat'
        elif isinstance(P, np.ndarray):
            # P is a numpy array
            if P.ndim == 0:
                # Scalar array - check if it's a string
                try:
                    if isinstance(P.item(), (str, np.str_)):
                        P_is_psat = str(P.item()).lower() == 'psat'
                except (ValueError, AttributeError):
                    pass
            elif P.size == 1:
                # Single-element array - check if it's a string
                try:
                    if isinstance(P.flat[0], (str, np.str_)):
                        P_is_psat = str(P.flat[0]).lower() == 'psat'
                except (ValueError, AttributeError, IndexError):
                    pass

        if P_is_psat:
            P_arg = 'Psat'
            T_arg = np.atleast_1d(T)
            if len(T_arg) == 1:
                T_arg = float(T_arg[0])
            else:
                T_arg = T_arg.tolist()
        else:
            # Convert T and P to proper format for subcrt
            T_vals = np.atleast_1d(T)
            P_vals = np.atleast_1d(P)

            # subcrt needs lists for multiple T/P values
            if len(T_vals) > 1 or len(P_vals) > 1:
                T_arg = T_vals.tolist() if len(T_vals) > 1 else float(T_vals[0])
                P_arg = P_vals.tolist() if len(P_vals) > 1 else float(P_vals[0])
            else:
                T_arg = float(T_vals[0])
                P_arg = float(P_vals[0])

        supcrt_out = subcrt(['H2O', 'oxygen', 'H+', 'e-'],
                           [-1, 0.5, 2, 2],
                           T=T_arg, P=P_arg, convert=False, messages=messages, show=False)

        # Extract logK values
        if hasattr(supcrt_out.out, 'logK'):
            logK = supcrt_out.out.logK
        else:
            logK = supcrt_out.out['logK']

        # Convert to numpy array
        logK = np.asarray(logK)

        if units == 'logfo2':
            # Convert Eh to logfO2
            pe_value = convert(value, 'pe', T=T, messages=messages)
            return 2 * (logK + logaH2O + 2*pH + 2*pe_value)
        if units == 'e0':
            # Convert logfO2 to Eh
            pe_value = (-logK - 2*pH + value/2 - logaH2O) / 2
            return convert(pe_value, 'Eh', T=T, messages=messages)

    else:
        warnings.warn(f"convert: no conversion to {units} found")
        return value


def envert(value: Union[float, np.ndarray, List[float]],
           units: str) -> Union[float, np.ndarray]:
    """
    Convert values to the specified units from those given in thermo()$opt.

    This function is used internally to convert from the user's preferred units
    (stored in thermo().opt) to standard internal units.

    Parameters
    ----------
    value : float, ndarray, or list
        Value(s) to convert
    units : str
        Target units ('C', 'K', 'bar', 'MPa', 'J', 'cal')

    Returns
    -------
    float or ndarray
        Converted value(s)
    """

    if not isinstance(value, (int, float, np.ndarray, list)):
        return value

    value = np.asarray(value)

    # Check if first element is numeric
    if value.size > 0 and not np.issubdtype(value.dtype, np.number):
        return value

    units = units.lower()
    opt = thermo().opt

    # Temperature conversions
    if units in ['c', 'k', 't.units']:
        if units == 'c' and opt['T.units'] == 'K':
            return convert(value, 'c')
        if units == 'k' and opt['T.units'] == 'C':
            return convert(value, 'k')

    # Energy conversions
    if units in ['j', 'cal', 'e.units']:
        if units == 'j' and opt['E.units'] == 'cal':
            return convert(value, 'j')
        if units == 'cal' and opt['E.units'] == 'J':
            return convert(value, 'cal')

    # Pressure conversions
    if units in ['bar', 'mpa', 'p.units']:
        if units == 'mpa' and opt['P.units'] == 'bar':
            return convert(value, 'mpa')
        if units == 'bar' and opt['P.units'] == 'MPa':
            return convert(value, 'bar')

    return value
