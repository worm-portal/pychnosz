# pychnosz/core/nonideal.py
# Ported from CHNOSZ/R/nonideal.R
# First version of function in R: 20080308 jmd
# Added Helgeson method 20171012
# Python port: 2026

import numpy as np
from scipy.interpolate import PPoly
from typing import Optional, Union, List, Dict


# Gas constant - https://physics.nist.gov/cgi-bin/cuu/Value?r
R = 8.314463  # J K^-1 mol^-1


def _fmm_spline(x, y):
    """
    Create cubic spline matching R's splinefun(method='fmm').

    Direct port of R's fmm_spline() C function from
    src/library/stats/src/splines.c (Forsythe, Malcolm, Moler 1977).

    Parameters
    ----------
    x : array_like
        Knot positions (must be increasing).
    y : array_like
        Values at knots.

    Returns
    -------
    PPoly
        Piecewise polynomial callable matching R's splinefun output.
    """
    x = np.asarray(x, dtype=float).copy()
    y = np.asarray(y, dtype=float).copy()
    n = len(x)

    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    if n < 2:
        raise ValueError("need at least 2 data points")

    if n < 3:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        b[0] = b[1] = slope
        coeffs = np.array([[0], [0], [b[0]], [y[0]]])
        return PPoly(coeffs, x)

    nm1 = n - 1

    # Set up tridiagonal system
    # d = offdiagonal (h values), b = diagonal, c = right hand side
    d[0] = x[1] - x[0]
    c[1] = (y[1] - y[0]) / d[0]
    for i in range(1, n - 1):
        d[i] = x[i + 1] - x[i]
        b[i] = 2.0 * (d[i - 1] + d[i])
        c[i + 1] = (y[i + 1] - y[i]) / d[i]
        c[i] = c[i + 1] - c[i]

    # End conditions: third derivatives at endpoints obtained from divided differences
    b[0] = -d[0]
    b[n - 1] = -d[nm1 - 1]
    c[0] = 0.0
    c[n - 1] = 0.0
    if n > 3:
        c[0] = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0])
        c[n - 1] = c[nm1 - 1] / (x[n - 1] - x[n - 3]) - c[nm1 - 2] / (x[nm1 - 1] - x[n - 4])
        c[0] = c[0] * d[0] ** 2 / (x[3] - x[0])
        c[n - 1] = -c[n - 1] * d[nm1 - 1] ** 2 / (x[n - 1] - x[n - 4])

    # Gaussian elimination
    for i in range(1, n):
        t = d[i - 1] / b[i - 1]
        b[i] = b[i] - t * d[i - 1]
        c[i] = c[i] - t * c[i - 1]

    # Backward substitution
    c[n - 1] = c[n - 1] / b[n - 1]
    for i in range(nm1 - 1, -1, -1):
        c[i] = (c[i] - d[i] * c[i + 1]) / b[i]

    # c[i] is now the sigma[i] of the text
    # Compute polynomial coefficients
    b[n - 1] = (y[n - 1] - y[n - 2]) / d[nm1 - 1] + d[nm1 - 1] * (c[n - 2] + 2.0 * c[n - 1])
    for i in range(nm1):
        b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i])
        d[i] = (c[i + 1] - c[i]) / d[i]
        c[i] = 3.0 * c[i]
    c[n - 1] = 3.0 * c[n - 1]
    d[n - 1] = d[nm1 - 1]

    # Build PPoly: coefficients in descending power order for (x - x[i]) basis
    coeffs = np.zeros((4, n - 1))
    for i in range(n - 1):
        coeffs[0, i] = d[i]      # x^3
        coeffs[1, i] = c[i]      # x^2
        coeffs[2, i] = b[i]      # x^1
        coeffs[3, i] = y[i]      # x^0

    return PPoly(coeffs, x)


def bgamma(TC=25, P=1):
    """
    Calculate the extended term parameter (b_gamma) for the Debye-Huckel equation.

    Uses spline interpolation of data from:
    - Helgeson, 1969 (doi:10.2475/ajs.267.7.729)
    - Helgeson et al., 1981 (doi:10.2475/ajs.281.10.1249)
    - Manning et al., 2013 (doi:10.2138/rmg.2013.75.5)

    Parameters
    ----------
    TC : float or array_like
        Temperature in degrees Celsius.
    P : float or array_like
        Pressure in bar.

    Returns
    -------
    numpy.ndarray
        b_gamma values at the given T, P conditions.
    """
    T = np.atleast_1d(np.asarray(TC, dtype=float))
    P = np.atleast_1d(np.asarray(P, dtype=float))

    # Are we at a pre-fitted constant pressure?
    uP = np.unique(P)
    is1 = len(uP) == 1 and uP[0] == 1 and np.all(T == 25)
    is500 = len(uP) == 1 and uP[0] == 500
    is1000 = len(uP) == 1 and uP[0] == 1000
    is2000 = len(uP) == 1 and uP[0] == 2000
    is3000 = len(uP) == 1 and uP[0] == 3000
    is4000 = len(uP) == 1 and uP[0] == 4000
    is5000 = len(uP) == 1 and uP[0] == 5000
    is10000 = len(uP) == 1 and uP[0] == 10000
    is20000 = len(uP) == 1 and uP[0] == 20000
    is30000 = len(uP) == 1 and uP[0] == 30000
    is40000 = len(uP) == 1 and uP[0] == 40000
    is50000 = len(uP) == 1 and uP[0] == 50000
    is60000 = len(uP) == 1 and uP[0] == 60000
    isoP = (is1 or is500 or is1000 or is2000 or is3000 or is4000 or is5000 or
            is10000 or is20000 or is30000 or is40000 or is50000 or is60000)

    # Build isobaric spline functions as needed
    # Values for Bdot x 100 from Helgeson (1969), Figure (P = Psat)
    if not isoP:
        T0 = np.array([23.8, 49.4, 98.9, 147.6, 172.6, 197.1, 222.7, 248.1, 268.7])
        B0 = np.array([4.07, 4.27, 4.30, 4.62, 4.86, 4.73, 4.09, 3.61, 1.56]) / 100
        S0 = _fmm_spline(T0, B0)

    # Values for bgamma x 100 from Helgeson et al., 1981 Table 27
    if is500 or not isoP:
        T0_5 = np.arange(0, 401, 25, dtype=float)
        B0_5 = np.array([5.6, 7.1, 7.8, 8.0, 7.8, 7.5, 7.0, 6.4, 5.7, 4.8, 3.8, 2.6, 1.0, -1.2, -4.1, -8.4, -15.2]) / 100
        S0_5 = _fmm_spline(T0_5, B0_5)
        if is500:
            return np.asarray(S0_5(T), dtype=float)

    if is1000 or not isoP:
        T1 = np.arange(0, 501, 25, dtype=float)
        B1 = np.array([6.6, 7.7, 8.7, 8.3, 8.2, 7.9, 7.5, 7.0, 6.5, 5.9, 5.2, 4.4, 3.5, 2.5, 1.1, -0.6, -2.8, -5.7, -9.3, -13.7, -19.2]) / 100
        S1 = _fmm_spline(T1, B1)
        if is1000:
            return np.asarray(S1(T), dtype=float)

    if is2000 or not isoP:
        # 550 and 600 degC points from Manning et al., 2013 Fig. 11
        T2 = np.append(np.arange(0, 501, 25, dtype=float), [550, 600])
        B2 = np.array([7.4, 8.3, 8.8, 8.9, 8.9, 8.7, 8.5, 8.1, 7.8, 7.4, 7.0, 6.6, 6.2, 5.8, 5.2, 4.6, 3.8, 2.9, 1.8, 0.5, -1.0, -3.93, -4.87]) / 100
        S2 = _fmm_spline(T2, B2)
        if is2000:
            return np.asarray(S2(T), dtype=float)

    if is3000 or not isoP:
        T3 = np.arange(0, 501, 25, dtype=float)
        B3 = np.array([6.5, 8.3, 9.2, 9.6, 9.7, 9.6, 9.4, 9.3, 9.2, 9.0, 8.8, 8.6, 8.3, 8.1, 7.8, 7.5, 7.1, 6.6, 6.0, 5.4, 4.8]) / 100
        S3 = _fmm_spline(T3, B3)
        if is3000:
            return np.asarray(S3(T), dtype=float)

    if is4000 or not isoP:
        T4 = np.arange(0, 501, 25, dtype=float)
        B4 = np.array([4.0, 7.7, 9.5, 10.3, 10.7, 10.8, 10.8, 10.8, 10.7, 10.6, 10.5, 10.4, 10.3, 10.2, 10.0, 9.8, 9.6, 9.3, 8.9, 8.5, 8.2]) / 100
        S4 = _fmm_spline(T4, B4)
        if is4000:
            return np.asarray(S4(T), dtype=float)

    if is5000 or not isoP:
        # 550 and 600 degC points from Manning et al., 2013 Fig. 11
        T5 = np.append(np.arange(0, 501, 25, dtype=float), [550, 600])
        B5 = np.array([0.1, 6.7, 9.6, 11.1, 11.8, 12.2, 12.4, 12.4, 12.4, 12.4, 12.4, 12.3, 12.3, 12.2, 12.1, 11.9, 11.8, 11.5, 11.3, 11.0, 10.8, 11.2, 12.52]) / 100
        S5 = _fmm_spline(T5, B5)
        if is5000:
            return np.asarray(S5(T), dtype=float)

    # 10, 20, and 30 kb points from Manning et al., 2013 Fig. 11
    # Here, one control point at 25 degC is added to make the splines curve down at low T
    if is10000 or not isoP:
        T10 = np.array([25] + list(np.arange(300, 1001, 50, dtype=float)))
        B10 = np.array([12, 17.6, 17.8, 18, 18.2, 18.9, 21, 23.3, 26.5, 28.8, 31.4, 34.1, 36.5, 39.2, 41.6, 44.1]) / 100
        S10 = _fmm_spline(T10, B10)
        if is10000:
            return np.asarray(S10(T), dtype=float)

    if is20000 or not isoP:
        T20 = np.array([25] + list(np.arange(300, 1001, 50, dtype=float)))
        B20 = np.array([16, 21.2, 21.4, 22, 22.4, 23.5, 26.5, 29.2, 32.6, 35.2, 38.2, 41.4, 44.7, 47.7, 50.5, 53.7]) / 100
        S20 = _fmm_spline(T20, B20)
        if is20000:
            return np.asarray(S20(T), dtype=float)

    if is30000 or not isoP:
        T30 = np.array([25] + list(np.arange(300, 1001, 50, dtype=float)))
        B30 = np.array([19, 23.9, 24.1, 24.6, 25.2, 26.7, 30.3, 32.9, 36.5, 39.9, 43, 46.4, 49.8, 53.2, 56.8, 60]) / 100
        S30 = _fmm_spline(T30, B30)
        if is30000:
            return np.asarray(S30(T), dtype=float)

    # 40-60 kb points extrapolated from 10-30 kb points of Manning et al., 2013
    if is40000 or not isoP:
        T40 = np.arange(300, 1001, 50, dtype=float)
        B40 = np.array([25.8, 26, 26.4, 27.2, 28.9, 33, 35.5, 39.2, 43.2, 46.4, 49.9, 53.4, 57.1, 61.2, 64.4]) / 100
        S40 = _fmm_spline(T40, B40)
        if is40000:
            return np.asarray(S40(T), dtype=float)

    if is50000 or not isoP:
        T50 = np.arange(300, 1001, 50, dtype=float)
        B50 = np.array([27.1, 27.3, 27.7, 28.5, 30.5, 34.8, 37.3, 41.1, 45.5, 48.7, 52.4, 55.9, 59.8, 64.3, 67.5]) / 100
        S50 = _fmm_spline(T50, B50)
        if is50000:
            return np.asarray(S50(T), dtype=float)

    if is60000 or not isoP:
        T60 = np.arange(300, 1001, 50, dtype=float)
        B60 = np.array([28, 28.2, 28.6, 29.5, 31.6, 36.1, 38.6, 42.5, 47.1, 50.4, 54.1, 57.6, 61.6, 66.5, 69.7]) / 100
        S60 = _fmm_spline(T60, B60)
        if is60000:
            return np.asarray(S60(T), dtype=float)

    if is1:
        # Fast path for 25 degC and 1 bar
        return np.full_like(T, 0.041)

    # General case: interpolate across T and P
    # Make T and P the same length
    ncond = max(len(T), len(P))
    T = np.resize(T, ncond)
    P = np.resize(P, ncond)

    result = np.empty(ncond)
    lastT = None
    ST = None

    for i in range(ncond):
        # Fast path: skip splines at 25 degC and 1 bar
        if T[i] == 25 and P[i] == 1:
            result[i] = 0.041
        else:
            if T[i] != lastT:
                # Get the spline fits from particular pressures for each T
                if T[i] >= 700:
                    PT = np.array([10000, 20000, 30000, 40000, 50000, 60000], dtype=float)
                    B = np.array([float(S10(T[i])), float(S20(T[i])), float(S30(T[i])),
                                  float(S40(T[i])), float(S50(T[i])), float(S60(T[i]))])
                elif T[i] >= 600:
                    PT = np.array([2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 60000], dtype=float)
                    B = np.array([float(S2(T[i])), float(S3(T[i])), float(S4(T[i])), float(S5(T[i])),
                                  float(S10(T[i])), float(S20(T[i])), float(S30(T[i])),
                                  float(S40(T[i])), float(S50(T[i])), float(S60(T[i]))])
                elif T[i] >= 500:
                    PT = np.array([1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 60000], dtype=float)
                    B = np.array([float(S1(T[i])), float(S2(T[i])), float(S3(T[i])), float(S4(T[i])),
                                  float(S5(T[i])), float(S10(T[i])), float(S20(T[i])), float(S30(T[i])),
                                  float(S40(T[i])), float(S50(T[i])), float(S60(T[i]))])
                elif T[i] >= 400:
                    PT = np.array([500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 60000], dtype=float)
                    B = np.array([float(S0_5(T[i])), float(S1(T[i])), float(S2(T[i])), float(S3(T[i])),
                                  float(S4(T[i])), float(S5(T[i])), float(S10(T[i])), float(S20(T[i])),
                                  float(S30(T[i])), float(S40(T[i])), float(S50(T[i])), float(S60(T[i]))])
                elif T[i] >= 300:
                    # Here the lowest P is Psat
                    PT = np.array([86, 500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000, 60000], dtype=float)
                    B = np.array([float(S0(T[i])), float(S0_5(T[i])), float(S1(T[i])), float(S2(T[i])),
                                  float(S3(T[i])), float(S4(T[i])), float(S5(T[i])), float(S10(T[i])),
                                  float(S20(T[i])), float(S30(T[i])), float(S40(T[i])), float(S50(T[i])),
                                  float(S60(T[i]))])
                elif T[i] >= 200:
                    # Drop highest pressures because we get into ice
                    PT = np.array([16, 500, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000], dtype=float)
                    B = np.array([float(S0(T[i])), float(S0_5(T[i])), float(S1(T[i])), float(S2(T[i])),
                                  float(S3(T[i])), float(S4(T[i])), float(S5(T[i])), float(S10(T[i])),
                                  float(S20(T[i])), float(S30(T[i])), float(S40(T[i]))])
                elif T[i] >= 100:
                    PT = np.array([1, 500, 1000, 2000, 3000, 4000, 5000, 10000, 20000], dtype=float)
                    B = np.array([float(S0(T[i])), float(S0_5(T[i])), float(S1(T[i])), float(S2(T[i])),
                                  float(S3(T[i])), float(S4(T[i])), float(S5(T[i])), float(S10(T[i])),
                                  float(S20(T[i]))])
                elif T[i] >= 0:
                    PT = np.array([1, 500, 1000, 2000, 3000, 4000, 5000], dtype=float)
                    B = np.array([float(S0(T[i])), float(S0_5(T[i])), float(S1(T[i])), float(S2(T[i])),
                                  float(S3(T[i])), float(S4(T[i])), float(S5(T[i]))])

                # Make a new spline as a function of pressure at this T
                ST = _fmm_spline(PT, B)
                # Remember this T; if it's the same as the next one, we won't re-make the spline
                lastT = T[i]

            result[i] = float(ST(P[i]))

    return result


def Bdot_fn(TC):
    """
    Calculate the B-dot parameter as a function of temperature.

    This is a simpler alternative to bgamma(), used for the "Bdot" nonideal method.
    Data from Helgeson (1969).

    Parameters
    ----------
    TC : float or array_like
        Temperature in degrees Celsius.

    Returns
    -------
    numpy.ndarray
        B-dot values.
    """
    TC = np.atleast_1d(np.asarray(TC, dtype=float))
    x = np.array([25, 50, 100, 150, 200, 250, 300], dtype=float)
    y = np.array([0.0418, 0.0439, 0.0468, 0.0479, 0.0456, 0.0348, 0])
    S = _fmm_spline(x, y)
    result = np.asarray(S(TC), dtype=float)
    result[TC > 300] = 0
    return result


def _Helgeson(prop, Z, I, T, A_DH, B_DH, acirc, m_star, bgamma_val):
    """
    Debye-Huckel equation with b_gamma or B-dot extended term parameter.

    From Helgeson, 1969.

    Parameters
    ----------
    prop : str
        Property to calculate: "loggamma" or "G".
    Z : float or array
        Charge of species.
    I : float or array
        Ionic strength.
    T : float or array
        Temperature in Kelvin.
    A_DH : float or array
        Debye-Huckel A parameter.
    B_DH : float or array
        Debye-Huckel B parameter.
    acirc : float
        Ion size parameter in cm.
    m_star : float or array
        Total molality of all dissolved species.
    bgamma_val : float or array
        b_gamma or B-dot parameter.

    Returns
    -------
    float or numpy.ndarray
        loggamma or G correction values.
    """
    loggamma = (-A_DH * Z**2 * I**0.5 / (1 + acirc * B_DH * I**0.5)
                - np.log10(1 + 0.0180153 * m_star)
                + bgamma_val * I)
    if prop == "loggamma":
        return loggamma
    elif prop == "G":
        return R * T * np.log(10) * loggamma
    return 0.0


def _Setchenow(prop, I, T, m_star, bgamma_val):
    """
    Setchenow equation with b_gamma or B-dot extended term parameter.

    From Shvarov and Bastrakov, 1999.

    Parameters
    ----------
    prop : str
        Property to calculate: "loggamma" or "G".
    I : float or array
        Ionic strength.
    T : float or array
        Temperature in Kelvin.
    m_star : float or array
        Total molality of all dissolved species.
    bgamma_val : float or array
        b_gamma or B-dot parameter.

    Returns
    -------
    float or numpy.ndarray
        loggamma or G correction values.
    """
    loggamma = -np.log10(1 + 0.0180153 * m_star) + bgamma_val * I
    if prop == "loggamma":
        return loggamma
    elif prop == "G":
        return R * T * np.log(10) * loggamma
    return 0.0


def nonideal(species, speciesprops, IS, T, P, A_DH=None, B_DH=None, m_star=None, method=None):
    """
    Generate nonideal contributions to thermodynamic properties of aqueous species.

    Ported from CHNOSZ nonideal.R. Calculates activity coefficient corrections
    using Helgeson (B-dot/bgamma) or Setchenow equations.

    Parameters
    ----------
    species : list of int
        OBIGT indices (1-based) of aqueous species.
    speciesprops : dict
        Dictionary mapping species index (0-based position) to property dict.
        Each property dict maps property name ('G', 'H', 'S', 'Cp') to numpy arrays.
        This is modified in-place.
    IS : float or array_like
        Ionic strength (mol/kg).
    T : array_like
        Temperature in Kelvin.
    P : array_like
        Pressure in bar.
    A_DH : array_like, optional
        Debye-Huckel A parameter (needed for Bdot/bgamma methods).
    B_DH : array_like, optional
        Debye-Huckel B parameter (needed for Bdot/bgamma methods).
    m_star : array_like, optional
        Total molality of all dissolved species. If None, taken equal to IS.
    method : str, optional
        Nonideal method. One of "Bdot", "Bdot0", "bgamma", "bgamma0".
        If None, uses thermo().opt['nonideal'].

    Returns
    -------
    dict
        Modified speciesprops with corrections applied and 'loggam' added.
    """
    from .thermo import thermo
    from .info import info as info_fn
    from ..utils.formula import makeup

    thermo_sys = thermo()

    if method is None:
        method = thermo_sys.opt.get('nonideal', 'Bdot')

    # Validate method
    valid_methods = ('Bdot', 'Bdot0', 'bgamma', 'bgamma0')
    if method not in valid_methods:
        raise ValueError(f"invalid nonideal method: {method}. Must be one of {valid_methods}")

    IS = np.atleast_1d(np.asarray(IS, dtype=float))
    T = np.atleast_1d(np.asarray(T, dtype=float))
    P = np.atleast_1d(np.asarray(P, dtype=float))

    if A_DH is not None:
        A_DH = np.atleast_1d(np.asarray(A_DH, dtype=float))
    if B_DH is not None:
        B_DH = np.atleast_1d(np.asarray(B_DH, dtype=float))

    if m_star is None:
        m_star = IS

    # Get species charges from formulas
    obigt = thermo_sys.obigt
    Z = np.zeros(len(species))
    formulas = []
    for idx, sp in enumerate(species):
        formula = obigt.loc[sp, 'formula']
        formulas.append(formula)
        # Get charge using makeup - look for 'Z' in the elemental composition
        try:
            mkp = makeup(formula)
            if 'Z' in mkp:
                Z[idx] = mkp['Z']
        except Exception:
            pass

    # Get ion size parameters (acirc)
    if method.startswith('Bdot'):
        # "ion size parameter" from HCh package (Shvarov and Bastrakov, 1999)
        # based on Table 2.7 of Garrels and Christ, 1965
        bdot_acirc = thermo_sys.bdot_acirc if thermo_sys.bdot_acirc else {}
        acirc = np.array([bdot_acirc.get(f, 4.5) for f in formulas])
        # Convert to cm
        acirc = acirc * 1e-8
    elif method.startswith('bgamma'):
        # "distance of closest approach" of ions in NaCl solutions (HKF81 Table 2)
        acirc = np.full(len(species), 3.72e-8)

    # Get b_gamma or B-dot values
    if method == 'bgamma':
        # Convert T from Kelvin to Celsius for bgamma()
        TC = T - 273.15
        bgamma_val = bgamma(TC, P)
    elif method == 'Bdot':
        TC = T - 273.15
        bgamma_val = Bdot_fn(TC)
    elif method in ('Bdot0', 'bgamma0'):
        bgamma_val = 0.0

    # Get indices for H+ and e- to keep their activity coefficients at unity
    iH = info_fn("H+")
    ie = info_fn("e-")
    ideal_H = thermo_sys.opt.get('ideal.H', True)
    ideal_e = thermo_sys.opt.get('ideal.e', True)

    # Setchenow method option
    setchenow_opt = thermo_sys.opt.get('Setchenow', 'bgamma0')

    icharged = []
    ineutral = []

    for idx, sp in enumerate(species):
        # Skip H+ and e- if ideal options are set
        if sp == iH and ideal_H:
            continue
        if sp == ie and ideal_e:
            continue

        didcharged = False
        didneutral = False

        if Z[idx] == 0:
            # Neutral species: use Setchenow equation
            for pname in ('G', 'H', 'S', 'Cp'):
                if pname not in speciesprops[idx]:
                    continue
                if setchenow_opt == 'bgamma':
                    speciesprops[idx][pname] = speciesprops[idx][pname] + _Setchenow(pname, IS, T, m_star, bgamma_val)
                    didneutral = True
                elif setchenow_opt == 'bgamma0':
                    speciesprops[idx][pname] = speciesprops[idx][pname] + _Setchenow(pname, IS, T, m_star, 0.0)
                    didneutral = True
        else:
            # Charged species: use Helgeson equation
            for pname in ('G', 'H', 'S', 'Cp'):
                if pname not in speciesprops[idx]:
                    continue
                speciesprops[idx][pname] = speciesprops[idx][pname] + _Helgeson(
                    pname, Z[idx], IS, T, A_DH, B_DH, acirc[idx], m_star, bgamma_val)
                didcharged = True

        # Append loggam
        if didcharged:
            speciesprops[idx]['loggam'] = _Helgeson(
                "loggamma", Z[idx], IS, T, A_DH, B_DH, acirc[idx], m_star, bgamma_val)
            icharged.append(formulas[idx])
        if didneutral:
            if setchenow_opt == 'bgamma':
                speciesprops[idx]['loggam'] = _Setchenow("loggamma", IS, T, m_star, bgamma_val)
            else:
                speciesprops[idx]['loggam'] = _Setchenow("loggamma", IS, T, m_star, 0.0)
            ineutral.append(formulas[idx])

    # Print messages matching R CHNOSZ behavior
    mettext = f"{method} equation"
    if method == "Bdot0":
        mettext = "B-dot equation (B-dot = 0)"
    if icharged:
        print(f"nonideal: calculations for {', '.join(icharged)} ({mettext})")
    if ineutral:
        print(f"nonideal: calculations for {', '.join(ineutral)} (Setchenow equation)")

    return speciesprops
