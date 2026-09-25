"""
IAPWS-95 water model implementation.

This module implements the IAPWS-95 formulation for the thermodynamic properties
of ordinary water substance for general and scientific use. This is the international
standard for water properties.

This implementation exactly matches the R CHNOSZ package, with identical coefficients
and derivative calculations. No shortcuts or approximations - full fidelity to Wagner & Pruss (2002).

References:
- Wagner, W., & Pruß, A. (2002). The IAPWS formulation 1995 for the thermodynamic
  properties of ordinary water substance for general and scientific use.
  Journal of Physical and Chemical Reference Data, 31(2), 387-535.
- Fernández, D. P., et al. (1997). A formulation for the static permittivity of
  water and steam at temperatures from 238 K to 873 K at pressures up to 1200 MPa.
  Journal of Physical and Chemical Reference Data, 26(4), 1125-1166.
- R CHNOSZ package IAPWS95.R implementation
"""

import numpy as np
from typing import Union, List, Optional, Dict, Any
import warnings
from scipy.optimize import brentq

from .archer_wang import water_AW90


class AccurateIAPWS95Water:
    """
    Accurate IAPWS-95 water model implementation matching R CHNOSZ exactly.
    
    This class provides thermodynamic properties of water using the IAPWS-95
    formulation with exact coefficients and derivative calculations from the
    Wagner & Pruss (2002) specification as implemented in R CHNOSZ.
    """
    
    def __init__(self):
        """Initialize IAPWS95 water model with exact constants."""
        # Physical constants (exactly matching R CHNOSZ)
        self.R = 0.46151805  # kJ/(kg·K) - Specific gas constant for water
        self.MW = 18.015268  # g/mol - Molecular weight
        
        # Critical constants (exactly matching R CHNOSZ)
        self.Tc = 647.096    # K - Critical temperature
        self.rhoc = 322.0    # kg/m³ - Critical density
        
        # Triple point constants
        self.Tt = 273.16     # K - Triple point temperature
        
        # Initialize coefficients exactly as in R CHNOSZ
        self._init_coefficients()
    
    def _init_coefficients(self):
        """Initialize coefficients for IAPWS-95 fundamental equation (exact R match)."""
        # Ideal gas coefficients (Table 6.1 Wagner & Pruss 2002, R CHNOSZ lines 114-117)
        self.n_ideal = np.array([
            -8.32044648201, 6.6832105268, 3.00632, 0.012436,
            0.97315, 1.27950, 0.96956, 0.24873
        ])
        
        self.gamma_ideal = np.array([
            np.nan, np.nan, np.nan, 1.28728967,
            3.53734222, 7.74073708, 9.24437796, 27.5075105
        ])
        
        # Residual part coefficients (Table 6.2 Wagner & Pruss 2002, R CHNOSZ lines 134-171)
        # c coefficients
        c_list = [np.nan]*7 + [1]*15 + [2]*20 + [3]*4 + [4] + [6]*4 + [np.nan]*5
        self.c_res = np.array(c_list)
        
        # d coefficients  
        self.d_res = np.array([
            1,1,1,2,2,3,4,1,1,1,2,2,3,4,
            4,5,7,9,10,11,13,15,1,2,2,2,3,4,
            4,4,5,6,6,7,9,9,9,9,9,10,10,12,
            3,4,4,5,14,3,6,6,6,3,3,3,np.nan,np.nan
        ])
        
        # t coefficients
        self.t_res = np.array([
            -0.5,0.875,1,0.5,0.75,0.375,1,4,6,12,1,5,4,2,
            13,9,3,4,11,4,13,1,7,1,9,10,10,3,
            7,10,10,6,10,10,1,2,3,4,8,6,9,8,
            16,22,23,23,10,50,44,46,50,0,1,4,np.nan,np.nan
        ])
        
        # n coefficients (exact values from R CHNOSZ)
        self.n_res = np.array([
            0.12533547935523E-1, 0.78957634722828E1 ,-0.87803203303561E1 ,
            0.31802509345418   ,-0.26145533859358   ,-0.78199751687981E-2,
            0.88089493102134E-2,-0.66856572307965   , 0.20433810950965   ,
           -0.66212605039687E-4,-0.19232721156002   ,-0.25709043003438   ,
            0.16074868486251   ,-0.40092828925807E-1, 0.39343422603254E-6,
           -0.75941377088144E-5, 0.56250979351888E-3,-0.15608652257135E-4,
            0.11537996422951E-8, 0.36582165144204E-6,-0.13251180074668E-11,
           -0.62639586912454E-9,-0.10793600908932   , 0.17611491008752E-1,
            0.22132295167546   ,-0.40247669763528   , 0.58083399985759   ,
            0.49969146990806E-2,-0.31358700712549E-1,-0.74315929710341   ,
            0.47807329915480   , 0.20527940895948E-1,-0.13636435110343   ,
            0.14180634400617E-1, 0.83326504880713E-2,-0.29052336009585E-1,
            0.38615085574206E-1,-0.20393486513704E-1,-0.16554050063734E-2,
            0.19955571979541E-2, 0.15870308324157E-3,-0.16388568342530E-4,
            0.43613615723811E-1, 0.34994005463765E-1,-0.76788197844621E-1,
            0.22446277332006E-1,-0.62689710414685E-4,-0.55711118565645E-9,
           -0.19905718354408   , 0.31777497330738   ,-0.11841182425981   ,
           -0.31306260323435E2 , 0.31546140237781E2 ,-0.25213154341695E4 ,
           -0.14874640856724   , 0.31806110878444
        ])
        
        # Additional coefficients for complex terms (R CHNOSZ lines 162-171)
        alpha_list = [np.nan]*51 + [20,20,20,np.nan,np.nan]
        self.alpha_res = np.array(alpha_list)
        
        beta_list = [np.nan]*51 + [150,150,250,0.3,0.3]
        self.beta_res = np.array(beta_list)
        
        gamma_list = [np.nan]*51 + [1.21,1.21,1.25,np.nan,np.nan]
        self.gamma_res = np.array(gamma_list)
        
        epsilon_list = [np.nan]*51 + [1,1,1,np.nan,np.nan]
        self.epsilon_res = np.array(epsilon_list)
        
        a_list = [np.nan]*54 + [3.5,3.5]
        self.a_res = np.array(a_list)
        
        b_list = [np.nan]*54 + [0.85,0.95]
        self.b_res = np.array(b_list)
        
        B_list = [np.nan]*54 + [0.2,0.2]
        self.B_res = np.array(B_list)
        
        C_list = [np.nan]*54 + [28,32]
        self.C_res = np.array(C_list)
        
        D_list = [np.nan]*54 + [700,800]
        self.D_res = np.array(D_list)
        
        A_list = [np.nan]*54 + [0.32,0.32]
        self.A_res = np.array(A_list)
        
        # Index ranges (from R CHNOSZ Table 6.5)
        self.i1 = np.arange(0, 7)      # 1:7 in R (0-based in Python)
        self.i2 = np.arange(7, 51)     # 8:51 in R
        self.i3 = np.arange(51, 54)    # 52:54 in R  
        self.i4 = np.arange(54, 56)    # 55:56 in R
        
    def _phi_ideal(self, delta: float, tau: float, derivative: str = 'phi') -> float:
        """
        Calculate ideal gas part of dimensionless Helmholtz energy and derivatives.
        
        Exact implementation matching R CHNOSZ IAPWS95.idealgas function.
        """
        if derivative == 'phi':
            # Equation 6.5 from Wagner & Pruss 2002
            result = (np.log(delta) + self.n_ideal[0] + self.n_ideal[1]*tau + 
                     self.n_ideal[2]*np.log(tau))
            
            # Sum term with exponentials
            for i in range(3, 8):  # n[4:8] in R (indices 3:7 in Python)
                gamma_val = self.gamma_ideal[i]
                if not np.isnan(gamma_val):
                    result += self.n_ideal[i] * np.log(1 - np.exp(-gamma_val*tau))
            
            return result
            
        elif derivative == 'phi.delta':
            return 1.0/delta
            
        elif derivative == 'phi.delta.delta':
            return -1.0/(delta**2)
            
        elif derivative == 'phi.tau':
            result = self.n_ideal[1] + self.n_ideal[2]/tau
            
            # Sum term with exponentials and gamma
            for i in range(3, 8):
                gamma_val = self.gamma_ideal[i]
                if not np.isnan(gamma_val):
                    exp_term = np.exp(-gamma_val*tau)
                    result += self.n_ideal[i] * gamma_val * ((1-exp_term)**(-1) - 1)
            
            return result
            
        elif derivative == 'phi.tau.tau':
            result = -self.n_ideal[2]/(tau**2)
            
            # Sum term with exponentials
            for i in range(3, 8):
                gamma_val = self.gamma_ideal[i]
                if not np.isnan(gamma_val):
                    exp_term = np.exp(-gamma_val*tau)
                    result -= (self.n_ideal[i] * gamma_val**2 * exp_term * 
                              (1-exp_term)**(-2))
            
            return result
            
        elif derivative == 'phi.delta.tau':
            return 0.0
            
        elif derivative == 'phi.tau.tau.tau':
            # Third derivative with respect to tau
            result = 2*self.n_ideal[2]/(tau**3)
            
            # Sum term with exponentials
            for i in range(3, 8):
                gamma_val = self.gamma_ideal[i]
                if not np.isnan(gamma_val):
                    exp_term = np.exp(-gamma_val*tau)
                    result += (self.n_ideal[i] * gamma_val**3 * exp_term * 
                              (1-exp_term)**(-3) * (2*exp_term - 1))
            
            return result
            
        elif derivative == 'phi.delta.delta.delta':
            # Third derivative with respect to delta
            return 2.0/(delta**3)
            
        elif derivative == 'phi.delta.tau.tau':
            # Mixed derivative: d³φ⁰/dδdτ²
            return 0.0
            
        elif derivative == 'phi.delta.delta.tau':
            # Mixed derivative: d³φ⁰/dδ²dτ
            return 0.0
            
        else:
            raise ValueError(f"Unknown derivative: {derivative}")
    
    def _delta_function(self, i: int, delta: float, tau: float) -> float:
        """Delta function for complex terms (R CHNOSZ Delta function)."""
        theta = self._theta_function(i, delta, tau)
        B_val = self.B_res[i]
        a_val = self.a_res[i]
        return theta**2 + B_val * ((delta-1)**2)**a_val
    
    def _theta_function(self, i: int, delta: float, tau: float) -> float:
        """Theta function for complex terms (R CHNOSZ Theta function)."""
        A_val = self.A_res[i]
        beta_val = self.beta_res[i]
        return (1-tau) + A_val * ((delta-1)**2)**(1/(2*beta_val))
    
    def _psi_function(self, i: int, delta: float, tau: float) -> float:
        """Psi function for complex terms (R CHNOSZ Psi function)."""
        C_val = self.C_res[i]
        D_val = self.D_res[i]
        return np.exp(-C_val*(delta-1)**2 - D_val*(tau-1)**2)
    
    def _delta_derivatives(self, i: int, delta: float, tau: float) -> Dict[str, float]:
        """Calculate Delta function derivatives (matching R CHNOSZ exactly)."""
        theta = self._theta_function(i, delta, tau)
        A_val = self.A_res[i]
        B_val = self.B_res[i]
        a_val = self.a_res[i]
        b_val = self.b_res[i]
        beta_val = self.beta_res[i]
        
        # dDelta/ddelta
        dDelta_ddelta = ((delta-1) * 
                        (A_val*theta*2/beta_val*((delta-1)**2)**(1/(2*beta_val)-1) +
                         2*B_val*a_val*((delta-1)**2)**(a_val-1)))
        
        # d²Delta/ddelta² (handle division by zero when delta ≈ 1)
        if abs(delta - 1) < 1e-15:
            # Use L'Hôpital's rule or limit behavior
            d2Delta_ddelta2 = (4*B_val*a_val*(a_val-1) + 
                              2*A_val**2*(1/beta_val)**2 + 
                              A_val*theta*4/beta_val*(1/(2*beta_val)-1))
        else:
            d2Delta_ddelta2 = (1/(delta-1)*dDelta_ddelta + (delta-1)**2 * (
                4*B_val*a_val*(a_val-1)*((delta-1)**2)**(a_val-2) + 
                2*A_val**2*(1/beta_val)**2 * (((delta-1)**2)**(1/(2*beta_val)-1))**2 + 
                A_val*theta*4/beta_val*(1/(2*beta_val)-1) * 
                ((delta-1)**2)**(1/(2*beta_val)-2)))
        
        # Delta^b derivatives
        delta_func = self._delta_function(i, delta, tau)
        dDelta_bi_ddelta = b_val * delta_func**(b_val-1) * dDelta_ddelta
        d2Delta_bi_ddelta2 = (b_val * (delta_func**(b_val-1) * d2Delta_ddelta2 + 
                             (b_val-1) * delta_func**(b_val-2) * dDelta_ddelta**2))
        
        # Tau derivatives
        dDelta_bi_dtau = -2*theta*b_val*delta_func**(b_val-1)
        d2Delta_bi_dtau2 = (2*b_val*delta_func**(b_val-1) + 
                           4*theta**2*b_val*(b_val-1)*delta_func**(b_val-2))
        
        # Mixed derivative
        d2Delta_bi_ddelta_dtau = (-A_val*b_val*2/beta_val*delta_func**(b_val-1)*(delta-1) *
                                 ((delta-1)**2)**(1/(2*beta_val)-1) - 
                                 2*theta*b_val*(b_val-1)*delta_func**(b_val-2)*dDelta_ddelta)
        
        return {
            'dDelta_ddelta': dDelta_ddelta,
            'd2Delta_ddelta2': d2Delta_ddelta2,
            'dDelta_bi_ddelta': dDelta_bi_ddelta,
            'd2Delta_bi_ddelta2': d2Delta_bi_ddelta2,
            'dDelta_bi_dtau': dDelta_bi_dtau,
            'd2Delta_bi_dtau2': d2Delta_bi_dtau2,
            'd2Delta_bi_ddelta_dtau': d2Delta_bi_ddelta_dtau
        }
    
    def _psi_derivatives(self, i: int, delta: float, tau: float) -> Dict[str, float]:
        """Calculate Psi function derivatives (matching R CHNOSZ exactly)."""
        C_val = self.C_res[i]
        D_val = self.D_res[i]
        psi = self._psi_function(i, delta, tau)
        
        return {
            'dPsi_ddelta': -2*C_val*(delta-1)*psi,
            'd2Psi_ddelta2': (2*C_val*(delta-1)**2 - 1) * 2*C_val*psi,
            'dPsi_dtau': -2*D_val*(tau-1)*psi,
            'd2Psi_dtau2': (2*D_val*(tau-1)**2 - 1) * 2*D_val*psi,
            'd2Psi_ddelta_dtau': 4*C_val*D_val*(delta-1)*(tau-1)*psi
        }
    
    def _phi_residual(self, delta: float, tau: float, derivative: str = 'phi') -> float:
        """
        Calculate residual part of dimensionless Helmholtz energy and derivatives.
        
        Exact implementation matching R CHNOSZ IAPWS95.residual function.
        """
        if derivative == 'phi':
            # Four terms as in R CHNOSZ phi function (lines 201-206)
            term1 = np.sum(self.n_res[self.i1] * delta**self.d_res[self.i1] * tau**self.t_res[self.i1])
            
            term2 = np.sum(self.n_res[self.i2] * delta**self.d_res[self.i2] * tau**self.t_res[self.i2] *
                          np.exp(-delta**self.c_res[self.i2]))
            
            term3 = 0.0
            for i in self.i3:
                alpha_val = self.alpha_res[i]
                beta_val = self.beta_res[i]
                epsilon_val = self.epsilon_res[i]
                gamma_val = self.gamma_res[i]
                if not (np.isnan(alpha_val) or np.isnan(beta_val)):
                    term3 += (self.n_res[i] * delta**self.d_res[i] * tau**self.t_res[i] *
                             np.exp(-alpha_val*(delta-epsilon_val)**2 - beta_val*(tau-gamma_val)**2))
            
            term4 = 0.0
            for i in self.i4:
                if not np.isnan(self.b_res[i]):
                    delta_func = self._delta_function(i, delta, tau)
                    psi_val = self._psi_function(i, delta, tau)
                    term4 += self.n_res[i] * delta_func**self.b_res[i] * delta * psi_val
            
            return term1 + term2 + term3 + term4
            
        elif derivative == 'phi.delta':
            # phi.delta implementation (R CHNOSZ lines 208-214)
            term1 = np.sum(self.n_res[self.i1] * self.d_res[self.i1] * 
                          delta**(self.d_res[self.i1]-1) * tau**self.t_res[self.i1])
            
            term2 = np.sum(self.n_res[self.i2] * np.exp(-delta**self.c_res[self.i2]) *
                          (delta**(self.d_res[self.i2]-1) * tau**self.t_res[self.i2] *
                           (self.d_res[self.i2] - self.c_res[self.i2] * delta**self.c_res[self.i2])))
            
            term3 = 0.0
            for i in self.i3:
                alpha_val = self.alpha_res[i]
                beta_val = self.beta_res[i]
                epsilon_val = self.epsilon_res[i]
                gamma_val = self.gamma_res[i]
                if not (np.isnan(alpha_val) or np.isnan(beta_val)):
                    exp_term = np.exp(-alpha_val*(delta-epsilon_val)**2 - beta_val*(tau-gamma_val)**2)
                    term3 += (self.n_res[i] * delta**self.d_res[i] * tau**self.t_res[i] * exp_term *
                             (self.d_res[i]/delta - 2*alpha_val*(delta-epsilon_val)))
            
            term4 = 0.0
            for i in self.i4:
                if not np.isnan(self.b_res[i]):
                    delta_func = self._delta_function(i, delta, tau)
                    psi_val = self._psi_function(i, delta, tau)
                    psi_derivs = self._psi_derivatives(i, delta, tau)
                    delta_derivs = self._delta_derivatives(i, delta, tau)
                    
                    term4 += (self.n_res[i] * 
                             (delta_func**self.b_res[i] * (psi_val + delta*psi_derivs['dPsi_ddelta']) +
                              delta_derivs['dDelta_bi_ddelta'] * delta * psi_val))
            
            return term1 + term2 + term3 + term4
            
        elif derivative == 'phi.delta.delta':
            # phi.delta.delta implementation (R CHNOSZ lines 216-224)
            term1 = np.sum(self.n_res[self.i1] * self.d_res[self.i1] * (self.d_res[self.i1]-1) * 
                          delta**(self.d_res[self.i1]-2) * tau**self.t_res[self.i1])
            
            term2 = 0.0
            for i in self.i2:
                d_val = self.d_res[i]
                c_val = self.c_res[i]
                exp_term = np.exp(-delta**c_val)
                factor = ((d_val - c_val*delta**c_val) * (d_val - 1 - c_val*delta**c_val) - 
                         c_val**2 * delta**c_val)
                term2 += (self.n_res[i] * exp_term * delta**(d_val-2) * tau**self.t_res[i] * factor)
            
            term3 = 0.0
            for i in self.i3:
                alpha_val = self.alpha_res[i]
                beta_val = self.beta_res[i]
                epsilon_val = self.epsilon_res[i]
                gamma_val = self.gamma_res[i]
                if not (np.isnan(alpha_val) or np.isnan(beta_val)):
                    d_val = self.d_res[i]
                    exp_term = np.exp(-alpha_val*(delta-epsilon_val)**2 - beta_val*(tau-gamma_val)**2)
                    factor = (-2*alpha_val*delta**d_val + 4*alpha_val**2*delta**d_val*(delta-epsilon_val)**2 -
                             4*d_val*alpha_val*delta**(d_val-1)*(delta-epsilon_val) + 
                             d_val*(d_val-1)*delta**(d_val-2))
                    term3 += self.n_res[i] * tau**self.t_res[i] * exp_term * factor
            
            term4 = 0.0
            for i in self.i4:
                if not np.isnan(self.b_res[i]):
                    delta_func = self._delta_function(i, delta, tau)
                    psi_val = self._psi_function(i, delta, tau)
                    psi_derivs = self._psi_derivatives(i, delta, tau)
                    delta_derivs = self._delta_derivatives(i, delta, tau)
                    
                    term4 += (self.n_res[i] * 
                             (delta_func**self.b_res[i] * 
                              (2*psi_derivs['dPsi_ddelta'] + delta*psi_derivs['d2Psi_ddelta2']) +
                              2*delta_derivs['dDelta_bi_ddelta'] * 
                              (psi_val + delta*psi_derivs['dPsi_ddelta']) +
                              delta_derivs['d2Delta_bi_ddelta2'] * delta * psi_val))
            
            return term1 + term2 + term3 + term4
            
        elif derivative == 'phi.tau':
            # phi.tau implementation (R CHNOSZ lines 226-231)
            term1 = np.sum(self.n_res[self.i1] * self.t_res[self.i1] * 
                          delta**self.d_res[self.i1] * tau**(self.t_res[self.i1]-1))
            
            term2 = np.sum(self.n_res[self.i2] * self.t_res[self.i2] * 
                          delta**self.d_res[self.i2] * tau**(self.t_res[self.i2]-1) * 
                          np.exp(-delta**self.c_res[self.i2]))
            
            term3 = 0.0
            for i in self.i3:
                alpha_val = self.alpha_res[i]
                beta_val = self.beta_res[i]
                epsilon_val = self.epsilon_res[i]
                gamma_val = self.gamma_res[i]
                if not (np.isnan(alpha_val) or np.isnan(beta_val)):
                    exp_term = np.exp(-alpha_val*(delta-epsilon_val)**2 - beta_val*(tau-gamma_val)**2)
                    term3 += (self.n_res[i] * delta**self.d_res[i] * tau**self.t_res[i] * exp_term *
                             (self.t_res[i]/tau - 2*beta_val*(tau-gamma_val)))
            
            term4 = 0.0
            for i in self.i4:
                if not np.isnan(self.b_res[i]):
                    psi_val = self._psi_function(i, delta, tau)
                    psi_derivs = self._psi_derivatives(i, delta, tau)
                    delta_derivs = self._delta_derivatives(i, delta, tau)
                    
                    term4 += (self.n_res[i] * delta * 
                             (delta_derivs['dDelta_bi_dtau'] * psi_val +
                              self._delta_function(i, delta, tau)**self.b_res[i] * psi_derivs['dPsi_dtau']))
            
            return term1 + term2 + term3 + term4
            
        elif derivative == 'phi.tau.tau':
            # phi.tau.tau implementation (R CHNOSZ lines 233-239)
            term1 = np.sum(self.n_res[self.i1] * self.t_res[self.i1] * (self.t_res[self.i1]-1) *
                          delta**self.d_res[self.i1] * tau**(self.t_res[self.i1]-2))
            
            term2 = np.sum(self.n_res[self.i2] * self.t_res[self.i2] * (self.t_res[self.i2]-1) *
                          delta**self.d_res[self.i2] * tau**(self.t_res[self.i2]-2) *
                          np.exp(-delta**self.c_res[self.i2]))
            
            term3 = 0.0
            for i in self.i3:
                alpha_val = self.alpha_res[i]
                beta_val = self.beta_res[i]
                epsilon_val = self.epsilon_res[i]
                gamma_val = self.gamma_res[i]
                if not (np.isnan(alpha_val) or np.isnan(beta_val)):
                    exp_term = np.exp(-alpha_val*(delta-epsilon_val)**2 - beta_val*(tau-gamma_val)**2)
                    tau_factor = (self.t_res[i]/tau - 2*beta_val*(tau-gamma_val))
                    term3 += (self.n_res[i] * delta**self.d_res[i] * tau**self.t_res[i] * exp_term *
                             (tau_factor**2 - self.t_res[i]/tau**2 - 2*beta_val))
            
            term4 = 0.0
            for i in self.i4:
                if not np.isnan(self.b_res[i]):
                    delta_func = self._delta_function(i, delta, tau)
                    psi_val = self._psi_function(i, delta, tau)
                    psi_derivs = self._psi_derivatives(i, delta, tau)
                    delta_derivs = self._delta_derivatives(i, delta, tau)
                    
                    term4 += (self.n_res[i] * delta * 
                             (delta_derivs['d2Delta_bi_dtau2'] * psi_val +
                              2*delta_derivs['dDelta_bi_dtau'] * psi_derivs['dPsi_dtau'] +
                              delta_func**self.b_res[i] * psi_derivs['d2Psi_dtau2']))
            
            return term1 + term2 + term3 + term4
            
        elif derivative == 'phi.delta.tau':
            # phi.delta.tau implementation (R CHNOSZ lines 241-248)
            term1 = np.sum(self.n_res[self.i1] * self.d_res[self.i1] * self.t_res[self.i1] *
                          delta**(self.d_res[self.i1]-1) * tau**(self.t_res[self.i1]-1))
            
            term2 = np.sum(self.n_res[self.i2] * self.t_res[self.i2] *
                          delta**(self.d_res[self.i2]-1) * tau**(self.t_res[self.i2]-1) *
                          (self.d_res[self.i2] - self.c_res[self.i2]*delta**self.c_res[self.i2]) *
                          np.exp(-delta**self.c_res[self.i2]))
            
            term3 = 0.0
            for i in self.i3:
                alpha_val = self.alpha_res[i]
                beta_val = self.beta_res[i]
                epsilon_val = self.epsilon_res[i]
                gamma_val = self.gamma_res[i]
                if not (np.isnan(alpha_val) or np.isnan(beta_val)):
                    exp_term = np.exp(-alpha_val*(delta-epsilon_val)**2 - beta_val*(tau-gamma_val)**2)
                    delta_factor = (self.d_res[i]/delta - 2*alpha_val*(delta-epsilon_val))
                    tau_factor = (self.t_res[i]/tau - 2*beta_val*(tau-gamma_val))
                    term3 += (self.n_res[i] * delta**self.d_res[i] * tau**self.t_res[i] * exp_term *
                             delta_factor * tau_factor)
            
            term4 = 0.0
            for i in self.i4:
                if not np.isnan(self.b_res[i]):
                    delta_func = self._delta_function(i, delta, tau)
                    psi_val = self._psi_function(i, delta, tau)
                    psi_derivs = self._psi_derivatives(i, delta, tau)
                    delta_derivs = self._delta_derivatives(i, delta, tau)
                    
                    term4 += (self.n_res[i] *
                             (delta_func**self.b_res[i] * 
                              (psi_derivs['dPsi_dtau'] + delta*psi_derivs['d2Psi_ddelta_dtau']) +
                              delta*delta_derivs['dDelta_bi_ddelta']*psi_derivs['dPsi_dtau'] +
                              delta_derivs['dDelta_bi_dtau'] * 
                              (psi_val + delta*psi_derivs['dPsi_ddelta']) +
                              delta_derivs['d2Delta_bi_ddelta_dtau']*delta*psi_val))
            
            return term1 + term2 + term3 + term4
            
            
            
            
        else:
            raise ValueError(f"Unknown derivative: {derivative}")
    
    def calculate_IAPWS95_property(self, property_name: str, T: float, rho: float) -> float:
        """
        Calculate individual IAPWS95 property (matching R CHNOSZ IAPWS95 function).
        
        Parameters
        ----------
        property_name : str
            Property name (matching R CHNOSZ property names)
        T : float
            Temperature in K
        rho : float
            Density in kg/m³
            
        Returns
        -------
        float
            Property value
        """
        # Calculate dimensionless variables (Equation 6.4)
        delta = rho / self.rhoc
        tau = self.Tc / T
        
        # Property calculations matching R CHNOSZ (lines 32-81)
        if property_name.lower() == 'p':
            # Pressure in MPa (R line 34: x*rho*R*T/1000)
            x = 1 + delta * self._phi_residual(delta, tau, 'phi.delta')
            return x * rho * self.R * T / 1000.0
            
        elif property_name.lower() == 's':
            # Entropy in kJ/(kg·K) (R lines 36-38)
            phi_ideal = self._phi_ideal(delta, tau, 'phi')
            phi_residual = self._phi_residual(delta, tau, 'phi')
            phi_tau_ideal = self._phi_ideal(delta, tau, 'phi.tau')
            phi_tau_residual = self._phi_residual(delta, tau, 'phi.tau')
            x = tau * (phi_tau_ideal + phi_tau_residual) - phi_ideal - phi_residual
            return x * self.R
            
        elif property_name.lower() == 'u':
            # Internal energy in kJ/kg (R lines 40-42)
            phi_tau_ideal = self._phi_ideal(delta, tau, 'phi.tau')
            phi_tau_residual = self._phi_residual(delta, tau, 'phi.tau')
            x = tau * (phi_tau_ideal + phi_tau_residual)
            return x * self.R * T
            
        elif property_name.lower() == 'h':
            # Enthalpy in kJ/kg (R lines 44-46)
            phi_tau_ideal = self._phi_ideal(delta, tau, 'phi.tau')
            phi_tau_residual = self._phi_residual(delta, tau, 'phi.tau')
            phi_delta_residual = self._phi_residual(delta, tau, 'phi.delta')
            x = 1 + tau * (phi_tau_ideal + phi_tau_residual) + delta * phi_delta_residual
            return x * self.R * T
            
        elif property_name.lower() == 'g':
            # Gibbs energy in kJ/kg (R lines 48-50)
            phi_ideal = self._phi_ideal(delta, tau, 'phi')
            phi_residual = self._phi_residual(delta, tau, 'phi')
            phi_delta_residual = self._phi_residual(delta, tau, 'phi.delta')
            x = 1 + phi_ideal + phi_residual + delta * phi_delta_residual
            return x * self.R * T

        elif property_name.lower() == 'a':
            # Helmholtz energy in kJ/kg: a = R*T*(phi_ideal + phi_residual)
            phi_ideal = self._phi_ideal(delta, tau, 'phi')
            phi_residual = self._phi_residual(delta, tau, 'phi')
            x = phi_ideal + phi_residual
            return x * self.R * T

        elif property_name.lower() == 'cv':
            # Isochoric heat capacity in kJ/(kg·K) (R lines 52-54)
            phi_tau_tau_ideal = self._phi_ideal(delta, tau, 'phi.tau.tau')
            phi_tau_tau_residual = self._phi_residual(delta, tau, 'phi.tau.tau')
            x = -tau**2 * (phi_tau_tau_ideal + phi_tau_tau_residual)
            return x * self.R
            
        elif property_name.lower() == 'cp':
            # Isobaric heat capacity in kJ/(kg·K) (R lines 56-60)
            phi_tau_tau_ideal = self._phi_ideal(delta, tau, 'phi.tau.tau')
            phi_tau_tau_residual = self._phi_residual(delta, tau, 'phi.tau.tau')
            phi_delta_residual = self._phi_residual(delta, tau, 'phi.delta')
            phi_delta_tau_residual = self._phi_residual(delta, tau, 'phi.delta.tau')
            phi_delta_delta_residual = self._phi_residual(delta, tau, 'phi.delta.delta')
            
            term1 = -tau**2 * (phi_tau_tau_ideal + phi_tau_tau_residual)
            term2 = ((1 + delta*phi_delta_residual - delta*tau*phi_delta_tau_residual)**2 /
                    (1 + 2*delta*phi_delta_residual + delta**2*phi_delta_delta_residual))
            x = term1 + term2
            return x * self.R
            
        elif property_name.lower() == 'w':
            # Speed of sound in m/s (R lines 71-75)
            phi_tau_tau_ideal = self._phi_ideal(delta, tau, 'phi.tau.tau')
            phi_tau_tau_residual = self._phi_residual(delta, tau, 'phi.tau.tau')
            phi_delta_residual = self._phi_residual(delta, tau, 'phi.delta')
            phi_delta_tau_residual = self._phi_residual(delta, tau, 'phi.delta.tau')
            phi_delta_delta_residual = self._phi_residual(delta, tau, 'phi.delta.delta')
            
            x = (1 + 2*delta*phi_delta_residual + delta**2*phi_delta_delta_residual - 
                ((1 + delta*phi_delta_residual - delta*tau*phi_delta_tau_residual)**2 /
                 (tau**2 * (phi_tau_tau_ideal + phi_tau_tau_residual))))
            return np.sqrt(x * self.R * T)
            
            
            
            
            
            
            
        else:
            raise ValueError(f"Unknown property: {property_name}")
    
    def calculate(self, 
                  properties: Union[str, List[str]], 
                  T: Union[float, np.ndarray] = 298.15,
                  rho: Union[float, np.ndarray] = 1000.0) -> Union[float, np.ndarray, Dict[str, Any]]:
        """
        Calculate water properties using accurate IAPWS-95.
        
        Parameters
        ----------
        properties : str or list of str
            Property or list of properties to calculate
        T : float or array
            Temperature in Kelvin
        rho : float or array
            Density in kg/m³
            
        Returns
        -------
        float, array, or dict
            Calculated properties
        """
        # Handle input types
        if isinstance(properties, str):
            properties = [properties]
            single_prop = True
        else:
            single_prop = False
        
        # Convert inputs to arrays
        T = np.atleast_1d(np.asarray(T, dtype=float))
        rho = np.atleast_1d(np.asarray(rho, dtype=float))
        
        # Ensure same length
        max_len = max(len(T), len(rho))
        if len(T) < max_len:
            T = np.resize(T, max_len)
        if len(rho) < max_len:
            rho = np.resize(rho, max_len)
        
        # Calculate properties
        results = {}
        
        for prop in properties:
            prop_values = np.full(max_len, np.nan)
            
            for i in range(max_len):
                if not (np.isnan(T[i]) or np.isnan(rho[i]) or T[i] <= 0 or rho[i] <= 0):
                    try:
                        prop_values[i] = self.calculate_IAPWS95_property(prop, T[i], rho[i])
                    except Exception:
                        prop_values[i] = np.nan
            
            results[prop] = prop_values if len(prop_values) > 1 else prop_values[0]
        
        # Return results
        if single_prop:
            return results[properties[0]]
        else:
            return results


# Create module-level instance
accurate_iapws95 = AccurateIAPWS95Water()


def _WP02_auxiliary_accurate(property_type: str, T: Union[float, np.ndarray]) -> np.ndarray:
    """
    Auxiliary equations for liquid-vapor phase boundary (exact R CHNOSZ match).
    
    From Wagner and Pruss, 2002, exactly matching R CHNOSZ util.water.R
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    result = np.full_like(T, np.nan)
    
    # Critical point constants (exactly matching R CHNOSZ)
    T_critical = 647.096  # K
    P_critical = 22.064   # MPa
    rho_critical = 322.0  # kg/m³
    
    # Only calculate for valid temperatures below critical point
    valid = (T > 0) & (T < T_critical)
    T_valid = T[valid]
    
    if property_type == "P.sigma":
        # Vapor pressure (R CHNOSZ lines 13-25)
        V = 1 - T_valid / T_critical
        a1, a2, a3, a4, a5, a6 = -7.85951783, 1.84408259, -11.7866497, 22.6807411, -15.9618719, 1.80122502
        
        ln_P_sigma_P_critical = (T_critical / T_valid * 
            (a1*V + a2*V**1.5 + a3*V**3 + a4*V**3.5 + a5*V**4 + a6*V**7.5))
        P_sigma = P_critical * np.exp(ln_P_sigma_P_critical)  # MPa
        result[valid] = P_sigma
        
    elif property_type == "rho.liquid":
        # Saturated liquid density (R CHNOSZ lines 27-37)
        V = 1 - T_valid / T_critical
        b1, b2, b3, b4, b5, b6 = 1.99274064, 1.09965342, -0.510839303, -1.75493479, -45.5170352, -6.74694450E5
        
        rho_liquid = rho_critical * (1 + b1*V**(1/3) + b2*V**(2/3) + b3*V**(5/3) + 
                                   b4*V**(16/3) + b5*V**(43/3) + b6*V**(110/3))
        result[valid] = rho_liquid
        
    elif property_type == "rho.vapor":
        # Saturated vapor density (R CHNOSZ lines 38+)
        V = 1 - T_valid / T_critical
        c1, c2, c3, c4, c5, c6 = -2.03150240, -2.68302940, -5.38626492, -17.2991605, -44.7586581, -63.9201063
        
        ln_rho_vapor_rho_critical = (c1*V**(1/3) + c2*V**(2/3) + c3*V**(4/3) + 
                                   c4*V**(3) + c5*V**(37/6) + c6*V**(71/6))
        rho_vapor = rho_critical * np.exp(ln_rho_vapor_rho_critical)
        result[valid] = rho_vapor
        
    return result


def rho_IAPWS95_accurate(T: Union[float, np.ndarray], P: Union[float, np.ndarray], 
                        state: str = "", trace: int = 0) -> np.ndarray:
    """
    Return density in kg/m³ corresponding to given pressure (bar) and temperature (K).

    Port of the R CHNOSZ function rho.IAPWS95() (util.water.R). The root of
    P(T, rho) - P is bracketed exactly as in R, including R's uniroot()
    behavior of extending the search interval in the direction indicated by
    `extendInt` ("upX": the function increases with rho, so the lower bound
    is decreased while f(lower) > 0 and the upper bound is increased while
    f(upper) < 0). This is what allows the liquid root to be found for
    supercooled water (T below 273.15 K), where the saturated-liquid density
    estimate can lie on the wrong side of the root.

    Parameters
    ----------
    T : float or array
        Temperature in K
    P : float or array
        Pressure in bar
    state : str
        "" (determine the stable phase near the saturation curve by comparing
        Gibbs energies), "liquid", or "vapor"
    trace : int
        If > 0, print information about the root search
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    P = np.atleast_1d(np.asarray(P, dtype=float))

    # Ensure T and P have same length
    if len(P) < len(T):
        P = np.resize(P, len(T))
    elif len(T) < len(P):
        T = np.resize(T, len(P))

    rho = np.full_like(T, np.nan)

    # Critical point constants
    T_critical = 647.096  # K
    P_critical = 22.064   # MPa

    # Convert pressure from bar to MPa (matching R code line 60)
    P_MPa = P / 10.0

    for i in range(len(T)):
        if np.isnan(T[i]) or np.isnan(P[i]) or T[i] <= 0 or P[i] <= 0:
            continue

        # Function to find zero: P_calculated - P_target = 0
        def dP(rho_guess, Ti=T[i], Pi=P_MPa[i]):
            if rho_guess <= 0:
                return np.nan
            try:
                return accurate_iapws95.calculate_IAPWS95_property('P', Ti, rho_guess) - Pi
            except Exception:
                return np.nan

        try:
            Psat = _WP02_auxiliary_accurate("P.sigma", T[i])[0]  # MPa

            if T[i] > T_critical:
                # Above critical temperature
                interval = [0.1, 1.0]
                extendInt = "upX"
                if trace > 0:
                    print("supercritical (T) ", end="")
            elif P_MPa[i] > P_critical:
                # Above critical pressure
                rho_sat = _WP02_auxiliary_accurate("rho.liquid", T[i])[0]
                interval = [rho_sat, rho_sat + 1.0]
                extendInt = "upX"
                if trace > 0:
                    print("supercritical (P) ", end="")
            elif P_MPa[i] <= 0.9999 * Psat:
                # Steam
                rho_sat = _WP02_auxiliary_accurate("rho.vapor", T[i])[0]
                interval = [rho_sat * 0.1, rho_sat]
                extendInt = "upX"
                if trace > 0:
                    print("steam ", end="")
            elif P_MPa[i] >= 1.00005 * Psat:
                # Water
                rho_sat = _WP02_auxiliary_accurate("rho.liquid", T[i])[0]
                interval = [rho_sat, rho_sat + 1.0]
                extendInt = "upX"
                if trace > 0:
                    print("water ", end="")
            elif state not in ["liquid", "vapor"]:
                # We're close to the saturation curve;
                # calculate rho and G for liquid and vapor and return rho for the stable phase
                if trace > 0:
                    print("close to saturation; trying liquid and vapor")
                rho_liquid = rho_IAPWS95_accurate(T[i], P[i], state="liquid", trace=trace)[0]
                rho_vapor = rho_IAPWS95_accurate(T[i], P[i], state="vapor", trace=trace)[0]
                G_liquid = accurate_iapws95.calculate_IAPWS95_property('G', T[i], rho_liquid)
                G_vapor = accurate_iapws95.calculate_IAPWS95_property('G', T[i], rho_vapor)
                if G_liquid < G_vapor:
                    rho[i] = rho_liquid
                    if trace > 0:
                        print(f"G.liquid({G_liquid}) < G.vapor({G_vapor})")
                else:
                    rho[i] = rho_vapor
                    if trace > 0:
                        print(f"G.vapor({G_vapor}) < G.liquid ({G_liquid})")
                continue
            else:
                # We are looking at a specific state
                if trace > 0:
                    print(f"specified state: {state}  ", end="")
                if state == "vapor":
                    rho0 = _WP02_auxiliary_accurate("rho.vapor", T[i])[0]
                else:
                    rho0 = _WP02_auxiliary_accurate("rho.liquid", T[i])[0]
                # A too-big range may cause problems e.g.
                # interval <- c(rho0*0.9, rho0*1.1) fails for T=253.15, P=1
                interval = [rho0 * 0.95, rho0 * 1.05]
                # If P on the initial interval are both higher or lower than target P,
                # set the direction of interval extension
                P_init = [dP(interval[0]), dP(interval[1])]
                if all(p < 0 for p in P_init):
                    extendInt = "downX"
                elif all(p > 0 for p in P_init):
                    extendInt = "upX"
                else:
                    extendInt = "yes"

            if trace > 0:
                print(f"T={T[i]} P={P[i]} rho=[{interval[0]},{interval[1]}]")

            try:
                rho[i] = _uniroot(dP, interval[0], interval[1], extendInt=extendInt, trace=trace)
            except Exception as e:
                warnings.warn(f"rho_IAPWS95_accurate: problems finding density at {T[i]} K and {P[i]} bar",
                              stacklevel=2)
                if trace > 0:
                    print(f"root search failed: {e}")
                rho[i] = np.nan

        except Exception as e:
            warnings.warn(f"rho_IAPWS95_accurate: problems finding density at {T[i]} K and {P[i]} bar",
                          stacklevel=2)
            if trace > 0:
                print(f"phase identification failed: {e}")
            rho[i] = np.nan

    return rho


# Alias matching the R CHNOSZ function name
rho_IAPWS95 = rho_IAPWS95_accurate


def _uniroot(f, lower: float, upper: float, extendInt: str = "no", trace: int = 0,
             maxiter: int = 1000, xtol: float = 1e-10) -> float:
    """
    Find a root of f on [lower, upper], extending the interval as R's
    uniroot(extendInt = ...) does when the initial interval does not bracket
    a root.

    extendInt = "no": require a sign change on the initial interval.
    extendInt = "yes": extend both ends until a sign change is found.
    extendInt = "upX": f is increasing; decrease `lower` while f(lower) > 0 and
                       increase `upper` while f(upper) < 0.
    extendInt = "downX": f is decreasing; the opposite of "upX".
    The step sizes start at 0.01 * max(1e-4, |x|) and double each iteration,
    exactly as in R.
    """
    f_lower = f(lower)
    f_upper = f(upper)

    def Delta(u):
        return 0.01 * max(1e-4, abs(u))

    if extendInt != "no":
        delta_l = Delta(lower)
        delta_u = Delta(upper)
        it = 0
        if extendInt == "yes":
            while np.isnan(f_lower) or np.isnan(f_upper) or f_lower * f_upper > 0:
                it += 1
                if it > maxiter:
                    raise RuntimeError("no sign change found in %d iterations" % maxiter)
                lower -= delta_l
                upper += delta_u
                f_lower = f(lower)
                f_upper = f(upper)
                delta_l *= 2
                delta_u *= 2
        else:
            Sig = 1.0 if extendInt == "upX" else -1.0
            while np.isnan(f_lower) or Sig * f_lower > 0:
                it += 1
                if it > maxiter:
                    raise RuntimeError("no sign change found in %d iterations" % maxiter)
                lower -= delta_l
                f_lower = f(lower)
                delta_l *= 2
            while np.isnan(f_upper) or Sig * f_upper < 0:
                it += 1
                if it > maxiter:
                    raise RuntimeError("no sign change found in %d iterations" % maxiter)
                upper += delta_u
                f_upper = f(upper)
                delta_u *= 2
        if trace > 0 and it > 0:
            print(f"search extended to [{lower},{upper}] in {it} steps")

    if np.isnan(f_lower) or np.isnan(f_upper) or f_lower * f_upper > 0:
        raise RuntimeError(f"f() values at end points not of opposite sign: [{lower}, {upper}]")
    if f_lower == 0:
        return lower
    if f_upper == 0:
        return upper
    return brentq(f, lower, upper, xtol=xtol, rtol=1e-10)


def water_IAPWS95_accurate(properties: Union[str, List[str]], 
                          T: Union[float, np.ndarray] = 298.15,
                          P: Union[float, np.ndarray] = 1.0,
                          rho: Optional[Union[float, np.ndarray]] = None) -> Union[float, np.ndarray, Dict[str, Any]]:
    """
    Calculate water properties using accurate IAPWS-95 implementation.
    
    This function provides an accurate implementation of IAPWS-95 that matches
    the R CHNOSZ package exactly, with no shortcuts or approximations.
    
    Parameters
    ----------
    properties : str or list of str
        Property or properties to calculate ('P', 'S', 'U', 'H', 'G', 'Cv', 'Cp', 'w', 'rho')
    T : float or array
        Temperature in Kelvin
    P : float or array
        Pressure in bar (used to calculate density if rho not provided)
    rho : float or array, optional
        Density in kg/m³. If provided, used directly; if not, calculated from T,P
        
    Returns
    -------
    float, array, or dict
        Calculated water properties
        
    Examples
    --------
    >>> # Single property with T,P
    >>> p = water_IAPWS95_accurate('P', T=298.15, P=1.0)
    >>> 
    >>> # Single property with T,rho
    >>> p = water_IAPWS95_accurate('P', T=298.15, rho=997.0)
    >>> 
    >>> # Multiple properties
    >>> props = water_IAPWS95_accurate(['rho', 'Cp'], T=298.15, P=1.0)
    """
    # Handle input types
    if isinstance(properties, str):
        properties = [properties]
        single_prop = True
    else:
        single_prop = False
    
    # Convert inputs
    T = np.atleast_1d(np.asarray(T, dtype=float))
    
    if rho is None:
        # Calculate density from T,P
        P = np.atleast_1d(np.asarray(P, dtype=float))
        rho_calc = rho_IAPWS95_accurate(T, P)
    else:
        # Use provided density
        rho_calc = np.atleast_1d(np.asarray(rho, dtype=float))
        P = np.atleast_1d(np.asarray(P, dtype=float))
    
    # Ensure same length
    max_len = max(len(T), len(rho_calc))
    if len(T) < max_len:
        T = np.resize(T, max_len)
    if len(rho_calc) < max_len:
        rho_calc = np.resize(rho_calc, max_len)
    
    # Calculate properties
    results = {}
    
    # Reference state correction constants (from R water.R lines 187-194)
    # Convert to SUPCRT reference state at the triple point
    # difference = SUPCRT - IAPWS ( + entropy in G )
    M = 18.015268  # g/mol, molar mass of water
    Tr = 298.15    # Reference temperature
    cal_to_J = 4.184  # Conversion factor from cal to J
    
    # Pre-calculate reference corrections (from R)
    dH = (-68316.76 - 451.75437) * cal_to_J  # J/mol
    dS = (16.7123 - 1.581072) * cal_to_J     # J/mol/K  
    dU = (-67434.5 - 451.3229) * cal_to_J    # J/mol
    dA_base = (-55814.06 + 20.07376) * cal_to_J  # J/mol
    
    for prop in properties:
        if prop.lower() == 'rho':
            # Return density
            results[prop] = rho_calc if len(rho_calc) > 1 else rho_calc[0]
        else:
            # Calculate other properties
            prop_values = np.full(max_len, np.nan)
            
            for i in range(max_len):
                if not (np.isnan(T[i]) or np.isnan(rho_calc[i]) or T[i] <= 0 or rho_calc[i] <= 0):
                    try:
                        # Get raw IAPWS95 value in kJ/kg (specific units)
                        raw_value = accurate_iapws95.calculate_IAPWS95_property(prop, T[i], rho_calc[i])
                        
                        # Convert to J/mol (molar units) and apply reference state corrections
                        if prop.lower() == 'g':
                            # Gibbs energy: IAPWS95("g")*M + dG
                            dG = (-56687.71 + 19.64228 - dS * (T[i] - Tr)) * cal_to_J
                            prop_values[i] = raw_value * M + dG
                        elif prop.lower() == 'h':
                            # Enthalpy: IAPWS95("h")*M + dH
                            prop_values[i] = raw_value * M + dH
                        elif prop.lower() == 'u':
                            # Internal energy: IAPWS95("u")*M + dU
                            prop_values[i] = raw_value * M + dU
                        elif prop.lower() == 'a':
                            # Helmholtz energy: IAPWS95("a")*M + dA
                            dA = dA_base - dS * (T[i] - Tr)
                            prop_values[i] = raw_value * M + dA
                        elif prop.lower() == 's':
                            # Entropy: IAPWS95("s")*M + dS
                            prop_values[i] = raw_value * M + dS
                        elif prop.lower() in ['cv', 'cp']:
                            # Heat capacities: just convert to molar units (no reference correction)
                            prop_values[i] = raw_value * M
                        else:
                            # Other properties (P, w, etc.): use as-is or convert units as needed
                            if prop.lower() == 'w':
                                # Speed of sound: convert m/s to cm/s (factor of 100)
                                prop_values[i] = raw_value * 100
                            elif prop.lower() == 'p':
                                # Pressure: convert from MPa to bar (factor of 10)
                                prop_values[i] = raw_value * 10
                            else:
                                # Other properties: return as-is
                                prop_values[i] = raw_value
                                
                    except Exception:
                        prop_values[i] = np.nan
            
            results[prop] = prop_values if len(prop_values) > 1 else prop_values[0]
    
    # Return results
    if single_prop:
        return results[properties[0]]
    else:
        return results


# ========================================================================
# INTERFACE LAYER - Provides standardized API and unit conversion
# ========================================================================

class IAPWS95Water:
    """
    IAPWS-95 water model interface class.

    This class provides thermodynamic properties of water using the IAPWS-95
    formulation based on a fundamental equation for the Helmholtz free energy.

    This interface handles unit conversions and provides a standardized API
    that matches other water models in the package.
    """

    def __init__(self):
        """Initialize IAPWS95 water model."""
        pass

    def available_properties(self) -> List[str]:
        """
        Get list of available properties.

        Returns
        -------
        List[str]
            List of available property names
        """
        return list(IAPWS95_AVAILABLE_PROPERTIES)

    def calculate(self,
                  properties: Union[str, List[str]],
                  T: Union[float, np.ndarray] = 298.15,
                  P: Union[float, np.ndarray] = 100.0) -> Union[float, np.ndarray, Dict[str, Any]]:
        """
        Calculate water properties using IAPWS-95.

        Parameters
        ----------
        properties : str or list of str
            Property or list of properties to calculate
        T : float or array
            Temperature in Kelvin
        P : float or array
            Pressure in kPa

        Returns
        -------
        float, array, or dict
            Calculated properties
        """
        # Convert pressure from kPa to bar for the accurate implementation
        if isinstance(P, (int, float)):
            P_bar = P / 100.0
        else:
            P_bar = np.asarray(P) / 100.0

        # Use the accurate implementation (defined above in this file)
        return water_IAPWS95_accurate(properties, T=T, P=P_bar)


# Create module-level instance for backward compatibility
iapws95_water = IAPWS95Water()


# ========================================================================
# R CHNOSZ water.IAPWS95() equivalent
# ========================================================================

# Properties available from water.IAPWS95() in R CHNOSZ (water.R)
IAPWS95_AVAILABLE_PROPERTIES = [
    "A", "G", "S", "U", "H", "Cv", "Cp",
    "Speed", "epsilon",
    "YBorn", "QBorn", "XBorn", "NBorn", "UBorn",
    "V", "rho", "Psat", "de.dT", "de.dP", "pressure",
    "A_DH", "B_DH",
]

# Properties that R CHNOSZ water.IAPWS95() returns as NA with a warning
# (they are available from SUPCRT92 but not from IAPWS-95 in CHNOSZ)
IAPWS95_NA_PROPERTIES = ["E", "kT", "alpha", "daldT", "beta"]

# Molar mass of water used in R CHNOSZ water.IAPWS95()
_M_H2O = 18.015268  # g mol-1


def Psat_IAPWS95(T: Union[float, np.ndarray], Psat_floor: Union[float, None] = 1.0) -> np.ndarray:
    """
    Liquid-vapor saturation pressure (bar) as used by R CHNOSZ water.IAPWS95():
    the Wagner and Pruss (2002) auxiliary equation, plus 0.1 bar for values
    above 1 bar (for stability of the Born functions), floored at
    `Psat_floor` bar (use None to get the unfloored values).
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    P = _WP02_auxiliary_accurate("P.sigma", T)  # MPa
    P = np.where(P > 0.1, P + 0.01, P)
    if Psat_floor is not None:
        floor_MPa = Psat_floor / 10.0
        P = np.where(P < floor_MPa, floor_MPa, P)
    return P * 10.0  # bar


def water_IAPWS95(property: Union[str, List[str], None] = None,
                  T: Union[float, np.ndarray] = 298.15,
                  P: Union[float, np.ndarray, str] = 1.0,
                  Psat_floor: Union[float, None] = 1.0,
                  state: Optional[str] = None) -> Union[float, np.ndarray, Dict[str, Any], List[str]]:
    """
    Calculate thermodynamic and electrostatic properties of water with the
    IAPWS-95 formulation. Port of the R CHNOSZ function water.IAPWS95().

    Parameters
    ----------
    property : str, list of str, or None
        Property or properties to calculate. If None, the list of available
        properties is returned. Available: A, G, S, U, H, Cv, Cp (J/mol or
        J/mol/K, SUPCRT92 reference state), Speed (cm/s), epsilon
        (dielectric constant, Archer and Wang 1990), YBorn (K^-1), QBorn
        (bar^-1), XBorn (K^-2), NBorn (bar^-2), UBorn (bar^-1 K^-1), V
        (cm3/mol), rho (kg/m3), Psat (bar), de.dT, de.dP, pressure (bar),
        A_DH (kg^0.5 mol^-0.5) and B_DH (kg^0.5 mol^-0.5 cm^-1).
        E, kT, alpha, daldT and beta are returned as NaN with a warning,
        as in R CHNOSZ.
    T : float or array
        Temperature in Kelvin
    P : float, array, or "Psat"
        Pressure in bar, or "Psat" for the liquid-vapor saturation pressure
    Psat_floor : float or None
        Minimum value of Psat in bar (default 1)
    state : str, optional
        Phase to use when T and P are close to the saturation curve
        ("liquid" or "vapor"); defaults to the thermo option "IAPWS.sat"
        (normally "liquid").

    Returns
    -------
    float, array, or dict
        A single value (or array) for one property, or a dictionary of
        arrays for several properties.

    Notes
    -----
    Because the IAPWS-95 equation of state extrapolates into the supercooled
    liquid region, properties can be calculated below 273.15 K (down to about
    243 K at 1 bar), which SUPCRT92 cannot do. The Born functions are
    obtained by numerical differentiation of the dielectric constant, exactly
    as in R CHNOSZ.
    """
    if property is None:
        return list(IAPWS95_AVAILABLE_PROPERTIES)

    if isinstance(property, str):
        properties = [property]
        single_prop = True
    else:
        properties = list(property)
        single_prop = False

    unknown = [p for p in properties
               if p not in IAPWS95_AVAILABLE_PROPERTIES and p not in IAPWS95_NA_PROPERTIES]
    if len(unknown) > 0:
        raise ValueError(f"water_IAPWS95: property(s) not available: {' '.join(unknown)}")

    T = np.atleast_1d(np.asarray(T, dtype=float))

    if state is None:
        try:
            from ..core.thermo import thermo
            state = thermo().get_option("IAPWS.sat", "liquid")
        except Exception:
            state = "liquid"
    if state is None:
        state = ""

    # Psat
    if isinstance(P, str):
        if P != "Psat":
            raise ValueError(f"water_IAPWS95: unrecognized pressure '{P}'")
        P = Psat_IAPWS95(T, Psat_floor)
    else:
        P = np.atleast_1d(np.asarray(P, dtype=float))
        if len(P) < len(T):
            P = np.resize(P, len(T))
        elif len(T) < len(P):
            T = np.resize(T, len(P))
    n = len(T)
    P_MPa = P / 10.0

    # Reference state corrections to the SUPCRT92 convention at the triple
    # point (difference = SUPCRT - IAPWS, with the entropy term in G and A).
    # These reproduce R CHNOSZ exactly: there, dS is converted to J first and
    # the dS*(T - Tr) term is then included inside the cal-to-J conversion of
    # dG and dA, so the same order of operations is kept here.
    Tr = 298.15
    cal_to_J = 4.184
    dH = (-68316.76 - 451.75437) * cal_to_J
    dS = (16.7123 - 1.581072) * cal_to_J
    dG = (-56687.71 + 19.64228 - dS * (T - Tr)) * cal_to_J
    dU = (-67434.5 - 451.3229) * cal_to_J
    dA = (-55814.06 + 20.07376 - dS * (T - Tr)) * cal_to_J

    # Densities are needed for everything except Psat
    rho = None
    if properties != ["Psat"]:
        rho = rho_IAPWS95_accurate(T, P, state=state)

    def eos(prop):
        """Specific (per kg) IAPWS-95 property at each T, rho."""
        out = np.full(n, np.nan)
        for i in range(n):
            if np.isnan(T[i]) or np.isnan(rho[i]) or T[i] <= 0 or rho[i] <= 0:
                continue
            try:
                out[i] = accurate_iapws95.calculate_IAPWS95_property(prop, T[i], rho[i])
            except Exception:
                out[i] = np.nan
        return out

    def eps_at(Tv, Pv_bar):
        """Dielectric constant at arbitrary T (K) and P (bar) via IAPWS-95 density
        and Archer and Wang (1990)."""
        Tv = np.atleast_1d(np.asarray(Tv, dtype=float))
        Pv_bar = np.atleast_1d(np.asarray(Pv_bar, dtype=float))
        rho_v = rho_IAPWS95_accurate(Tv, Pv_bar, state=state)
        return np.atleast_1d(water_AW90(T=Tv, rho=rho_v, P=Pv_bar / 10.0))

    # epsilon is computed once and reused
    my_epsilon = None
    if any(p in ["epsilon", "A_DH", "B_DH"] for p in properties):
        my_epsilon = np.atleast_1d(water_AW90(T=T, rho=rho, P=P_MPa))

    results = {}
    for prop in properties:
        if prop in IAPWS95_NA_PROPERTIES:
            # R CHNOSZ: "water.IAPWS95: values of <prop> are NA"
            warnings.warn(f"water_IAPWS95: values of {prop} are NA", stacklevel=2)
            val = np.full(n, np.nan)
        elif prop == "Psat":
            val = np.array(P, dtype=float)
        elif prop == "rho":
            val = np.array(rho, dtype=float)
        elif prop == "V":
            val = _M_H2O * 1000.0 / rho
        elif prop == "pressure":
            val = eos("p") * 10.0
        elif prop == "S":
            val = eos("s") * _M_H2O + dS
        elif prop == "U":
            val = eos("u") * _M_H2O + dU
        elif prop == "A":
            val = eos("a") * _M_H2O + dA
        elif prop == "H":
            val = eos("h") * _M_H2O + dH
        elif prop == "G":
            val = eos("g") * _M_H2O + dG
        elif prop == "Cv":
            val = eos("cv") * _M_H2O
        elif prop == "Cp":
            val = eos("cp") * _M_H2O
        elif prop == "Speed":
            val = eos("w") * 100.0  # m/s to cm/s
        elif prop == "epsilon":
            val = np.array(my_epsilon, dtype=float)
        elif prop == "A_DH":
            val = 1.8246e6 * (rho / 1000.0) ** 0.5 / (my_epsilon * T) ** 1.5
        elif prop == "B_DH":
            val = 50.29e8 * (rho / 1000.0) ** 0.5 / (my_epsilon * T) ** 0.5
        elif prop == "de.dT":
            # NOTE: as in R CHNOSZ, the pressure is passed to water.AW90 in bar here
            val = np.full(n, np.nan)
            for i in range(n):
                dt = 0.001
                rho_v = rho_IAPWS95_accurate([T[i] - dt, T[i] + dt], [P[i], P[i]], state=state)
                e = np.atleast_1d(water_AW90(T=np.array([T[i] - dt, T[i] + dt]), rho=rho_v, P=np.array([P[i], P[i]])))
                val[i] = (e[1] - e[0]) / (2 * dt)
        elif prop == "de.dP":
            # NOTE: as in R CHNOSZ, the pressure is passed to water.AW90 in bar here
            val = np.full(n, np.nan)
            for i in range(n):
                dp = 0.001
                rho_v = rho_IAPWS95_accurate([T[i], T[i]], [P[i] - dp, P[i] + dp], state=state)
                e = np.atleast_1d(water_AW90(T=np.array([T[i], T[i]]), rho=rho_v, P=np.array([P[i] - dp, P[i] + dp])))
                val[i] = (e[1] - e[0]) / (2 * dp)
        elif prop == "QBorn":
            val = np.full(n, np.nan)
            for i in range(n):
                dp = 0.01
                e = eps_at([T[i], T[i]], [P[i] - dp, P[i] + dp])
                val[i] = -(1 / e[1] - 1 / e[0]) / (2 * dp)
        elif prop == "NBorn":
            val = np.full(n, np.nan)
            for i in range(n):
                dp = 0.01
                e = eps_at([T[i], T[i], T[i]], [P[i] - dp, P[i], P[i] + dp])
                val[i] = (-(1 / e[2] - 1 / e[1]) / dp + (1 / e[1] - 1 / e[0]) / dp) / dp
        elif prop == "YBorn":
            val = np.full(n, np.nan)
            for i in range(n):
                dt = 0.001
                e = eps_at([T[i] - dt, T[i] + dt], [P[i], P[i]])
                val[i] = -(1 / e[1] - 1 / e[0]) / (2 * dt)
        elif prop == "XBorn":
            val = np.full(n, np.nan)
            for i in range(n):
                dt = 0.001
                e = eps_at([T[i] - dt, T[i], T[i] + dt], [P[i], P[i], P[i]])
                val[i] = (-(1 / e[2] - 1 / e[1]) / dt + (1 / e[1] - 1 / e[0]) / dt) / dt
        elif prop == "UBorn":
            val = np.full(n, np.nan)
            for i in range(n):
                dt = 0.001
                dp = 0.001
                e1 = eps_at([T[i] - dt, T[i] - dt], [P[i] - dp, P[i] + dp])
                e2 = eps_at([T[i] + dt, T[i] + dt], [P[i] - dp, P[i] + dp])
                p1 = -(1 / e1[1] - 1 / e1[0]) / (2 * dp)
                p2 = -(1 / e2[1] - 1 / e2[0]) / (2 * dp)
                val[i] = (p2 - p1) / (2 * dt)
        else:  # pragma: no cover - guarded by the check above
            raise ValueError(f"water_IAPWS95: property not available: {prop}")

        val = np.asarray(val, dtype=float)
        results[prop] = val if n > 1 else float(val[0])

    if single_prop:
        return results[properties[0]]
    return results


# Alias for consistency with naming conventions
water_iapws95 = water_IAPWS95


# The main API functions are already defined above as top-level functions
# No need to redefine them here