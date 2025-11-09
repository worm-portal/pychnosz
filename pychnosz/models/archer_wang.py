"""
Archer & Wang (1990) dielectric constant correlation for water.

This module implements the accurate dielectric constant calculation
used in the original R version of CHNOSZ.

Reference:
Archer, D. G., and Wang, P. (1990) The dielectric constant of water 
and Debye-Hückel limiting law slopes. Journal of Physical and Chemical 
Reference Data, 19, 371-411.
"""

import numpy as np
import warnings
from typing import Union


def water_AW90(T: Union[float, np.ndarray] = 298.15, 
               rho: Union[float, np.ndarray] = 1000.0, 
               P: Union[float, np.ndarray] = 0.1) -> Union[float, np.ndarray]:
    """
    Calculate dielectric constant of water using Archer & Wang (1990) correlation.
    
    This is a direct Python translation of the R function water.AW90() from
    the original CHNOSZ package.
    
    Parameters
    ----------
    T : float or array
        Temperature in Kelvin
    rho : float or array  
        Density in kg/m³
    P : float or array
        Pressure in MPa
        
    Returns
    -------
    float or array
        Dielectric constant (dimensionless)
        
    Examples
    --------
    >>> # Water at 25°C, 1000 kg/m³, 0.1 MPa
    >>> eps = water_AW90(298.15, 1000.0, 0.1)
    >>> print(f"Dielectric constant: {eps:.1f}")  # Should be ~78.4
    """
    
    # Convert inputs to arrays
    T = np.atleast_1d(np.asarray(T, dtype=float))
    rho = np.atleast_1d(np.asarray(rho, dtype=float))  
    P = np.atleast_1d(np.asarray(P, dtype=float))
    
    # Make all arrays the same length
    max_len = max(len(T), len(rho), len(P))
    if len(T) < max_len:
        T = np.resize(T, max_len)
    if len(rho) < max_len:
        rho = np.resize(rho, max_len)
    if len(P) < max_len:
        P = np.resize(P, max_len)
    
    # Table 2 coefficients from Archer & Wang (1990)
    b = np.array([
        -4.044525E-2, 103.6180,    75.32165,
        -23.23778,    -3.548184,   -1246.311,
        263307.7,     -6.928953E-1, -204.4473
    ])
    
    # Physical constants
    alpha = 18.1458392E-30  # polarizability, m³
    mu = 6.1375776E-30      # dipole moment, C·m  
    N_A = 6.0221367E23      # Avogadro's number, mol⁻¹
    k = 1.380658E-23        # Boltzmann constant, J·K⁻¹
    M = 0.0180153           # molar mass of water, kg/mol
    rho_0 = 1000.0          # reference density, kg/m³
    epsilon_0 = 8.8541878E-12  # permittivity of vacuum, C²·J⁻¹·m⁻¹
    
    # Initialize output
    epsilon = np.full_like(T, np.nan)
    
    for i in range(len(T)):
        T_i = T[i]
        rho_i = rho[i]
        P_i = P[i]
        
        # Skip invalid conditions
        if np.isnan(T_i) or np.isnan(rho_i) or np.isnan(P_i):
            continue
        if T_i <= 0 or rho_i <= 0 or P_i < 0:
            continue
            
        # Equation 3: rho function 
        def rhofun():
            return (b[0]*P_i/T_i + 
                    b[1]/np.sqrt(T_i) + 
                    b[2]/(T_i-215) + 
                    b[3]/np.sqrt(T_i-215) + 
                    b[4]/(T_i-215)**0.25 + 
                    np.exp(b[5]/T_i + b[6]/T_i**2 + b[7]*P_i/T_i + b[8]*P_i/T_i**2))
        
        # g function
        def gfun():
            return rhofun() * rho_i/rho_0 + 1.0
        
        # mu function  
        def mufun():
            return gfun() * mu**2
        
        # Right-hand side of Equation 1
        V_m = M / rho_i  # molar volume, m³/mol
        epsfun_rhs = N_A * (alpha + mufun()/(3*epsilon_0*k*T_i)) / (3*V_m)
        
        # Solve quadratic equation (Equation 1 rearranged)
        # Original: (ε-1)(2ε+1)/(9ε) = rhs
        # Rearranged to: 2ε² - (9*rhs + 1)ε + 1 = 0
        # Using quadratic formula with positive root
        discriminant = (9*epsfun_rhs + 1)**2 + 8
        if discriminant < 0:
            warnings.warn(f'water_AW90: negative discriminant at T={T_i:.1f} K, '
                         f'rho={rho_i:.0f} kg/m3', stacklevel=2)
            continue
            
        epsilon_calc = (9*epsfun_rhs + 1 + np.sqrt(discriminant)) / 4.0
        
        # Check for reasonable result
        if epsilon_calc < 1.0 or epsilon_calc > 200.0:
            warnings.warn(f'water_AW90: unrealistic dielectric constant {epsilon_calc:.1f} '
                         f'at T={T_i:.1f} K, rho={rho_i:.0f} kg/m3', stacklevel=2)
            continue
            
        epsilon[i] = epsilon_calc
    
    # Return scalar if input was scalar
    if len(epsilon) == 1:
        return epsilon[0]
    else:
        return epsilon


if __name__ == "__main__":
    # Test the function
    print("Testing Archer & Wang (1990) dielectric constant correlation")
    print("=" * 60)
    
    # Test conditions from R CHNOSZ
    test_conditions = [
        (298.15, 997.0, 0.1),      # 25°C, ~997 kg/m³, 0.1 MPa
        (373.15, 958.0, 0.1),      # 100°C
        (273.15, 1000.0, 0.1),     # 0°C  
        (473.15, 800.0, 1.0),      # 200°C, 1 MPa
    ]
    
    for T, rho, P in test_conditions:
        eps = water_AW90(T, rho, P)
        print(f"T = {T:6.1f} K, ρ = {rho:6.0f} kg/m³, P = {P:5.1f} MPa: ε = {eps:6.1f}")
    
    # Test array input
    print("\nTesting array input:")
    T_array = np.array([273.15, 298.15, 373.15])
    rho_array = np.array([1000.0, 997.0, 958.0]) 
    P_array = np.array([0.1, 0.1, 0.1])
    
    eps_array = water_AW90(T_array, rho_array, P_array)
    for i in range(len(T_array)):
        print(f"T = {T_array[i]:6.1f} K: ε = {eps_array[i]:6.1f}")