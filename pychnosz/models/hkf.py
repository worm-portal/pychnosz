"""
HKF (Helgeson-Kirkham-Flowers) equation of state implementation.

This module implements the revised HKF equations for calculating thermodynamic
properties of aqueous species, based on the tested functions from HKF_cgl.py.

References:
- Shock, E.L. et al. (1992). Calculation of the thermodynamic properties of 
  aqueous species at high pressures and temperatures. J. Chem. Soc. Faraday Trans.
- Johnson, J.W. et al. (1992). SUPCRT92: A software package for calculating the 
  standard molal thermodynamic properties. Computers & Geosciences.
- R CHNOSZ hkf.R implementation
"""

import pandas as pd
import numpy as np
import math
import copy
import warnings
from .water import water

def convert_cm3bar(value):
    return value*4.184 * 10

def gfun(rhohat, Tc, P, alpha, daldT, beta):
    ## g and f functions for describing effective electrostatic radii of ions
    ## split from hkf() 20120123 jmd
    ## based on equations in
    ## Shock EL, Oelkers EH, Johnson JW, Sverjensky DA, Helgeson HC, 1992
    ## Calculation of the Thermodynamic Properties of Aqueous Species at High Pressures
    ## and Temperatures: Effective Electrostatic Radii, Dissociation Constants and
    ## Standard Partial Molal Properties to 1000 degrees C and 5 kbar
    ## J. Chem. Soc. Faraday Trans., 88(6), 803-826  doi:10.1039/FT9928800803
    # rhohat - density of water in g/cm3
    # Tc - temperature in degrees Celsius
    # P - pressure in bars

    # Vectorized version - handle both scalars and arrays
    rhohat = np.atleast_1d(rhohat)
    Tc = np.atleast_1d(Tc)
    P = np.atleast_1d(P)
    alpha = np.atleast_1d(alpha)
    daldT = np.atleast_1d(daldT)
    beta = np.atleast_1d(beta)

    # Broadcast to same shape
    shape = np.broadcast_shapes(rhohat.shape, Tc.shape, P.shape, alpha.shape, daldT.shape, beta.shape)
    rhohat = np.broadcast_to(rhohat, shape)
    Tc = np.broadcast_to(Tc, shape)
    P = np.broadcast_to(P, shape)
    alpha = np.broadcast_to(alpha, shape)
    daldT = np.broadcast_to(daldT, shape)
    beta = np.broadcast_to(beta, shape)

    # Initialize output arrays
    g = np.zeros(shape)
    dgdT = np.zeros(shape)
    d2gdT2 = np.zeros(shape)
    dgdP = np.zeros(shape)

    # only rhohat less than 1 will give results other than zero
    mask = rhohat < 1
    if not np.any(mask):
        return {"g": g, "dgdT": dgdT, "d2gdT2": d2gdT2, "dgdP": dgdP}

    # eta in Eq. 1
    eta = 1.66027E5
    # Table 3
    ag1 = -2.037662
    ag2 = 5.747000E-3
    ag3 = -6.557892E-6
    bg1 = 6.107361
    bg2 = -1.074377E-2
    bg3 = 1.268348E-5

    # Work only with masked values
    Tc_m = Tc[mask]
    P_m = P[mask]
    rhohat_m = rhohat[mask]
    alpha_m = alpha[mask]
    daldT_m = daldT[mask]
    beta_m = beta[mask]

    # Eq. 25
    ag = ag1 + ag2 * Tc_m + ag3 * Tc_m ** 2
    # Eq. 26
    bg = bg1 + bg2 * Tc_m + bg3 * Tc_m ** 2
    # Eq. 24
    g_m = ag * (1 - rhohat_m) ** bg

    # Table 4
    af1 = 0.3666666E2
    af2 = -0.1504956E-9
    af3 = 0.5017997E-13

    # Eq. 33
    f = ( ((Tc_m - 155) / 300) ** 4.8 + af1 * ((Tc_m - 155) / 300) ** 16 ) * \
        ( af2 * (1000 - P_m) ** 3 + af3 * (1000 - P_m) ** 4 )

    # limits of the f function (region II of Fig. 6)
    ifg = (Tc_m > 155) & (P_m < 1000) & (Tc_m < 355)

    # Eq. 32 - apply f correction where ifg is True
    # Check for complex values
    f_is_real = ~np.iscomplex(f)
    apply_f = ifg & f_is_real
    g_m = np.where(apply_f, g_m - f.real, g_m)

    # at P > 6000 bar (in DEW calculations), g is zero 20170926
    g_m = np.where(P_m > 6000, 0, g_m)

    ## now we have g at P, T
    # put the results in their right place (where rhohat < 1)
    g[mask] = g_m
    
    ## the rest is to get its partial derivatives with pressure and temperature
    ## after Johnson et al., 1992
    # alpha - coefficient of isobaric expansivity (K^-1)
    # daldT - temperature derivative of coefficient of isobaric expansivity (K^-2)
    # beta - coefficient of isothermal compressibility (bar^-1)

    # Eqn. 76
    d2fdT2 = (0.0608/300*((Tc_m-155)/300)**2.8 + af1/375*((Tc_m-155)/300)**14) * (af2*(1000-P_m)**3 + af3*(1000-P_m)**4)
    # Eqn. 75
    dfdT = (0.016*((Tc_m-155)/300)**3.8 + 16*af1/300*((Tc_m-155)/300)**15) * \
        (af2*(1000-P_m)**3 + af3*(1000-P_m)**4)
    # Eqn. 74
    dfdP = -(((Tc_m-155)/300)**4.8 + af1*((Tc_m-155)/300)**16) * \
        (3*af2*(1000-P_m)**2 + 4*af3*(1000-P_m)**3)
    d2bdT2 = 2 * bg3  # Eqn. 73
    d2adT2 = 2 * ag3  # Eqn. 72
    dbdT = bg2 + 2*bg3*Tc_m  # Eqn. 71
    dadT = ag2 + 2*ag3*Tc_m  # Eqn. 70

    # Convert complex to NaN
    d2fdT2 = np.where(np.iscomplex(d2fdT2), np.nan, np.real(d2fdT2))
    dfdT = np.where(np.iscomplex(dfdT), np.nan, np.real(dfdT))
    dfdP = np.where(np.iscomplex(dfdP), np.nan, np.real(dfdP))

    # Initialize derivative arrays for masked region
    dgdT_m = np.zeros_like(g_m)
    d2gdT2_m = np.zeros_like(g_m)
    dgdP_m = np.zeros_like(g_m)

    # Calculate derivatives where alpha and daldT are not NaN
    alpha_valid = ~np.isnan(alpha_m) & ~np.isnan(daldT_m)
    if np.any(alpha_valid):
        # Work with valid subset
        av_idx = alpha_valid
        bg_av = bg[av_idx]
        rhohat_av = rhohat_m[av_idx]
        alpha_av = alpha_m[av_idx]
        daldT_av = daldT_m[av_idx]
        g_av = g_m[av_idx]
        ag_av = ag[av_idx]
        Tc_av = Tc_m[av_idx]
        dbdT_av = dbdT[av_idx]
        dadT_av = dadT[av_idx]

        # Handle log of (1-rhohat) safely
        with np.errstate(divide='ignore', invalid='ignore'):
            log_term = np.log(1 - rhohat_av)
            log_term = np.where(np.isfinite(log_term), log_term, 0)

        # Eqn. 69
        dgadT = bg_av*rhohat_av*alpha_av*(1-rhohat_av)**(bg_av-1) + log_term*g_av/ag_av*dbdT_av
        D = rhohat_av

        # transcribed from SUPCRT92/reac92.f
        dDdT = -D * alpha_av
        dDdTT = -D * (daldT_av - alpha_av**2)
        Db = (1-D)**bg_av
        dDbdT = -bg_av*(1-D)**(bg_av-1)*dDdT + log_term*Db*dbdT_av
        dDbdTT = -(bg_av*(1-D)**(bg_av-1)*dDdTT + (1-D)**(bg_av-1)*dDdT*dbdT_av + \
            bg_av*dDdT*(-(bg_av-1)*(1-D)**(bg_av-2)*dDdT + log_term*(1-D)**(bg_av-1)*dbdT_av)) + \
            log_term*(1-D)**bg_av*d2bdT2 - (1-D)**bg_av*dbdT_av*dDdT/(1-D) + log_term*dbdT_av*dDbdT
        d2gdT2_calc = ag_av*dDbdTT + 2*dDbdT*dadT_av + Db*d2adT2

        # Apply f correction where ifg is True
        ifg_av = ifg[av_idx]
        d2fdT2_av = d2fdT2[av_idx]
        dfdT_av = dfdT[av_idx]
        d2gdT2_calc = np.where(ifg_av, d2gdT2_calc - d2fdT2_av, d2gdT2_calc)

        dgdT_calc = g_av/ag_av*dadT_av + ag_av*dgadT  # Eqn. 67
        dgdT_calc = np.where(ifg_av, dgdT_calc - dfdT_av, dgdT_calc)

        dgdT_m[av_idx] = dgdT_calc
        d2gdT2_m[av_idx] = d2gdT2_calc

    # Calculate dgdP where beta is not NaN
    beta_valid = ~np.isnan(beta_m)
    if np.any(beta_valid):
        bv_idx = beta_valid
        bg_bv = bg[bv_idx]
        rhohat_bv = rhohat_m[bv_idx]
        beta_bv = beta_m[bv_idx]
        g_bv = g_m[bv_idx]

        dgdP_calc = -bg_bv*rhohat_bv*beta_bv*g_bv*(1-rhohat_bv)**-1  # Eqn. 66
        ifg_bv = ifg[bv_idx]
        dfdP_bv = dfdP[bv_idx]
        dgdP_calc = np.where(ifg_bv, dgdP_calc - dfdP_bv, dgdP_calc)
        dgdP_m[bv_idx] = dgdP_calc

    # Put results back into full arrays
    dgdT[mask] = dgdT_m
    d2gdT2[mask] = d2gdT2_m
    dgdP[mask] = dgdP_m

    return {"g": g, "dgdT": dgdT, "d2gdT2": d2gdT2, "dgdP": dgdP}

def hkf(property=None, parameters=None, T=298.15, P=1,
    contrib = ["n", "s", "o"], H2O_props=["rho"], water_model="SUPCRT92"):
    # calculate G, H, S, Cp, V, kT, and/or E using
    # the revised HKF equations of state
    # H2O_props - H2O properties needed for subcrt() output
    # constants
    Tr = 298.15 # K
    Pr = 1      # bar
    Theta = 228 # K
    Psi = 2600  # bar

    # Convert T and P to arrays for vectorized operations
    T = np.atleast_1d(T)
    P = np.atleast_1d(P)

    # DEBUG
    if False:
        print(f"\nDEBUG HKF input:")
        print(f"  T (K): {T}")
        print(f"  P (bar): {P}")

    # make T and P equal length
    if P.size < T.size:
        P = np.full_like(T, P[0] if P.size == 1 else P)
    if T.size < P.size:
        T = np.full_like(P, T[0] if T.size == 1 else T)

    n_conditions = T.size
    
    # GB conversion note: handle error messages later
#     # nonsolvation, solvation, and origination contribution
#     notcontrib <- ! contrib %in% c("n", "s", "o")
#     if(TRUE %in% notcontrib) stop(paste("contrib must be in c('n', 's', 'o); got", c2s(contrib[notcontrib])))
    
    # get water properties
    # rho - for subcrt() output and g function
    # Born functions and epsilon - for HKF calculations
    H2O_props += ["QBorn", "XBorn", "YBorn", "epsilon"]

    if water_model == "SUPCRT92":
      # using H2O92D.f from SUPCRT92: alpha, daldT, beta - for partial derivatives of omega (g function)
      H2O_props += ["alpha", "daldT", "beta"]
    
    elif water_model == "IAPWS95":
      # using IAPWS-95: NBorn, UBorn - for compressibility, expansibility
      H2O_props += ["alpha", "daldT", "beta", "NBorn", "UBorn"]
    
    elif water_model == "DEW":
      # using DEW model: get beta to calculate dgdP
      H2O_props += ["alpha", "daldT", "beta"]

    # DEBUG: Print T and P being passed to water
    if False:
        print(f"DEBUG HKF calling water():")
        print(f"  T type: {type(T)}, T: {T}")
        print(f"  P type: {type(P)}, P: {P}")
        print(f"  H2O_props: {H2O_props}")

    H2O_PrTr = water(H2O_props, T=Tr, P=Pr)
    H2O_PT = water(H2O_props, T=T, P=P)

    # DEBUG: Print what water returned
    if False:
        print(f"DEBUG HKF water() returned:")
        print(f"  H2O_PT type: {type(H2O_PT)}")
        if isinstance(H2O_PT, dict):
            print(f"  H2O_PT keys: {H2O_PT.keys()}")
            print(f"  epsilon: {H2O_PT.get('epsilon', 'NOT FOUND')}")

    # Handle dict output from water function
    def get_water_prop(water_dict, prop):
        """Helper function to get water property from dict or DataFrame"""
        if isinstance(water_dict, dict):
            return water_dict[prop]
        else:
            return water_dict.loc["1", prop]

    # Get epsilon values and handle potential zeros
    epsilon_PT = get_water_prop(H2O_PT, "epsilon")
    epsilon_PrTr = get_water_prop(H2O_PrTr, "epsilon")

    # Check for zero or very small epsilon values and warn
    if np.any(epsilon_PT == 0) or np.any(np.abs(epsilon_PT) < 1e-10):
        warnings.warn(f"HKF: epsilon at P,T is zero or very small: {epsilon_PT}. H2O_PT keys: {H2O_PT.keys() if isinstance(H2O_PT, dict) else 'not dict'}")

    with np.errstate(divide='ignore', invalid='ignore'):
        ZBorn = -1 / epsilon_PT
        ZBorn_PrTr = -1 / epsilon_PrTr
    
    # a class to store the result
    out_dict = {} # dictionary to store output
    
    for k in parameters.index:
        
        if parameters["state"][k] != "aq":
            out_dict[k] = {p:float('NaN') for p in property}
        else:
            sp = parameters["name"][k]

            # loop over each species
            PAR = copy.copy(parameters.loc[k, :])

            PAR["a1.a"] = copy.copy(PAR["a1.a"]*10**-1)
            PAR["a2.b"] = copy.copy(PAR["a2.b"]*10**2)
            PAR["a4.d"] = copy.copy(PAR["a4.d"]*10**4)
            PAR["c2.f"] = copy.copy(PAR["c2.f"]*10**4)
            PAR["omega.lambda"] = copy.copy(PAR["omega.lambda"]*10**5)

            # substitute Cp and V for missing EoS parameters
            # here we assume that the parameters are in the same position as in thermo()$OBIGT
            # we don't need this if we're just looking at solvation properties (Cp_s_var, V_s_var)

            # GB conversion note: this block checks various things about EOS parameters.
            # for now, just set hasEOS to True
            hasEOS = True # delete this once the following block is converted to python
    #         if "n" in contrib:
    #             # put the heat capacity in for c1 if both c1 and c2 are missing
    #             if all(is.na(PAR[, 18:19])):
    #                 PAR[, 18] = PAR["Cp"]
    #             # put the volume in for a1 if a1, a2, a3 and a4 are missing
    #             if all(is.na(PAR[, 14:17])):
    #                 PAR[, 14] = convert(PAR["V"], "calories")
    #             # test for availability of the EoS parameters
    #             hasEOS = any(!is.na(PAR[, 14:21]))
    #             # if at least one of the EoS parameters is available, zero out any NA's in the rest
    #             if hasEOS:
    #                 PAR[, 14:21][, is.na(PAR[, 14:21])] = 0

            # compute values of omega(P,T) from those of omega(Pr,Tr)
            # using g function etc. (Shock et al., 1992 and others)
            omega = PAR["omega.lambda"]  # omega_PrTr
            # its derivatives are zero unless the g function kicks in
            dwdP = np.zeros(n_conditions)
            dwdT = np.zeros(n_conditions)
            d2wdT2 = np.zeros(n_conditions)
            Z = PAR["z.T"]

            omega_PT = np.full(n_conditions, PAR["omega.lambda"])
            if Z != 0 and Z != "NA" and PAR["name"] != "H+":
                # compute derivatives of omega: g and f functions (Shock et al., 1992; Johnson et al., 1992)
                rhohat = get_water_prop(H2O_PT, "rho")/1000  # just converting kg/m3 to g/cm3

                # temporarily filter out Python's warnings about dividing by zero, which is possible
                # with the equations in the gfunction
                # Possible complex output is acounted for in gfun().
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    g = gfun(rhohat, T-273.15, P, get_water_prop(H2O_PT, "alpha"), get_water_prop(H2O_PT, "daldT"), get_water_prop(H2O_PT, "beta"))

                # after SUPCRT92/reac92.f
                eta = 1.66027E5
                reref = (Z**2) / (omega/eta + Z/(3.082 + 0))
                re = reref + abs(Z) * g["g"]
                omega_PT = eta * (Z**2/re - Z/(3.082 + g["g"]))
                Z3 = abs(Z**3)/re**2 - Z/(3.082 + g["g"])**2
                Z4 = abs(Z**4)/re**3 - Z/(3.082 + g["g"])**3
                dwdP = (-eta * Z3 * g["dgdP"])
                dwdT = (-eta * Z3 * g["dgdT"])
                d2wdT2 = (2 * eta * Z4 * g["dgdT"]**2 - eta * Z3 * g["d2gdT2"])

            # loop over each property
            w = float('NaN')
            for i,PROP in enumerate(property) :

                # over nonsolvation, solvation, or origination contributions - vectorized
                hkf_p = np.zeros(n_conditions)

                for icontrib in contrib :
                    # various contributions to the properties
                    if icontrib == "n":
                        # nonsolvation ghs equations
                        if PROP == "H":
                            p_c = PAR["c1.e"]*(T-Tr) - PAR["c2.f"]*(1/(T-Theta)-1/(Tr-Theta))
                            p_a = PAR["a1.a"]*(P-Pr) + PAR["a2.b"]*np.log((Psi+P)/(Psi+Pr)) + \
                              ((2*T-Theta)/(T-Theta)**2)*(PAR["a3.c"]*(P-Pr)+PAR["a4.d"]*np.log((Psi+P)/(Psi+Pr)))
                            p = p_c + p_a
                        elif PROP == "S":
                            p_c = PAR["c1.e"]*np.log(T/Tr) - \
                              (PAR["c2.f"]/Theta)*( 1/(T-Theta)-1/(Tr-Theta) + \
                              np.log( (Tr*(T-Theta))/(T*(Tr-Theta)) )/Theta )
                            p_a = (T-Theta)**(-2)*(PAR["a3.c"]*(P-Pr)+PAR["a4.d"]*np.log((Psi+P)/(Psi+Pr)))
                            p = p_c + p_a
                        elif PROP == "G":
                            p_c = -PAR["c1.e"]*(T*np.log(T/Tr)-T+Tr) - \
                              PAR["c2.f"]*( (1/(T-Theta)-1/(Tr-Theta))*((Theta-T)/Theta) - \
                              (T/Theta**2)*np.log((Tr*(T-Theta))/(T*(Tr-Theta))) )
                            p_a = PAR["a1.a"]*(P-Pr) + PAR["a2.b"]*np.log((Psi+P)/(Psi+Pr)) + \
                              (PAR["a3.c"]*(P-Pr) + PAR["a4.d"]*np.log((Psi+P)/(Psi+Pr)))/(T-Theta)
                            p = p_c + p_a
                            # at Tr,Pr, if the origination contribution is not NA, ensure the solvation contribution is 0, not NA
                            if not np.isnan(PAR["G"]):
                                p = np.where((T==Tr) & (P==Pr), 0, p)
                        # nonsolvation cp v kt e equations
                        elif PROP == "Cp":
                            p = PAR["c1.e"] + PAR["c2.f"] * ( T - Theta ) ** (-2)
                        elif PROP == "V":
                            p = convert_cm3bar(PAR["a1.a"]) + \
                              convert_cm3bar(PAR["a2.b"]) / (Psi + P) + \
                              (convert_cm3bar(PAR["a3.c"]) + convert_cm3bar(PAR["a4.d"]) / (Psi + P)) / (T - Theta)
#                         elif PROP == "kT":
#                             p = (convert(PAR["a2.b"], "cm3bar") + \
#                               convert(PAR["a4.d"], "cm3bar") / (T - Theta)) * (Psi + P) ** (-2)
#                         elif PROP == "E":
#                             p = convert( - (PAR["a3.c"] + PAR["a4.d"] / convert((Psi + P), "calories")) * \
#                               (T - Theta) ** (-2), "cm3bar")
                        else:
                            print("BAD")

                    if icontrib == "s":
                        # solvation ghs equations
                        if PROP == "G":
                            p = -omega_PT*(ZBorn+1) + omega*(ZBorn_PrTr+1) + omega*get_water_prop(H2O_PrTr, "YBorn")*(T-Tr)
                            # at Tr,Pr, if the origination contribution is not NA, ensure the solvation contribution is 0, not NA
                            if(np.isnan(PAR["G"])):
                                p = np.where((T==Tr) & (P==Pr), 0, p)
                        if PROP == "H":
                            p = -omega_PT*(ZBorn+1) + omega_PT*T*get_water_prop(H2O_PT, "YBorn") + T*(ZBorn+1)*dwdT + \
                                   omega*(ZBorn_PrTr+1) - omega*Tr*get_water_prop(H2O_PrTr, "YBorn")
                        if PROP == "S":
                            p = omega_PT*get_water_prop(H2O_PT, "YBorn") + (ZBorn+1)*dwdT - omega*get_water_prop(H2O_PrTr, "YBorn")
                        # solvation cp v kt e equations
                        if PROP == "Cp":
                            p = omega_PT*T*get_water_prop(H2O_PT, "XBorn") + 2*T*get_water_prop(H2O_PT, "YBorn")*dwdT + T*(ZBorn+1)*d2wdT2
                        if PROP == "V":
                            term1 = -convert_cm3bar(omega_PT) * get_water_prop(H2O_PT, "QBorn")
                            term2 = convert_cm3bar(dwdP) * (-ZBorn - 1)
                            p = term1 + term2

                            # DEBUG
                            if False:
                                print(f"\nDEBUG solvation V terms:")
                                print(f"  omega_PT: {omega_PT}")
                                print(f"  QBorn: {get_water_prop(H2O_PT, 'QBorn')}")
                                print(f"  dwdP: {dwdP}")
                                print(f"  ZBorn: {ZBorn}")
                                print(f"  term1 (-ω*QBorn): {term1}")
                                print(f"  term2 (dwdP*(-Z-1)): {term2}")
                                print(f"  total p: {p}")
                        # TODO: the partial derivatives of omega are not included here here for kt and e
                        # (to do it, see p. 820 of SOJ+92 ... but kt requires d2wdP2 which we don"t have yet)
                        if PROP == "kT":
                            p = convert_cm3bar(omega) * get_water_prop(H2O_PT, "NBorn")
                        if PROP == "E":
                            p = -convert_cm3bar(omega) * get_water_prop(H2O_PT, "UBorn")

                    if icontrib == "o":
                        # origination ghs equations
                        if PROP == "G":
                            p = PAR["G"] - PAR["S"] * (T-Tr)
                            # don"t inherit NA from PAR$S at Tr
                            p = np.where(T == Tr, PAR["G"], p)
                        elif PROP == "H":
                            p = np.full(n_conditions, PAR["H"])
                        elif PROP == "S":
                            p = np.full(n_conditions, PAR["S"])
                        # origination eos equations (Cp, V, kT, E): senseless
                        else:
                            p = np.zeros(n_conditions)

                    # accumulate the contribution
                    hkf_p = hkf_p + p

                    # DEBUG
                    if False and PROP == "V":
                        print(f"\nDEBUG HKF V calculation (species {k}, contrib={icontrib}):")
                        print(f"  T: {T}")
                        print(f"  P: {P}")
                        print(f"  contribution p: {p}")
                        print(f"  accumulated hkf_p: {hkf_p}")

                # species have to be numbered (k) instead of named because of name repeats in db (e.g., cr polymorphs)
                if i > 0:
                    out_dict[k][PROP] = hkf_p
                else:
                    out_dict[k] = {PROP:hkf_p}

                # DEBUG
                if False and PROP == "V":
                    print(f"\nDEBUG HKF final V for species {k}: {hkf_p}")

    return(out_dict, H2O_PT)


def calculate_born_functions(T: np.ndarray, P) -> dict:
    """Calculate Born functions needed for HKF solvation calculations."""
    
    # Get water properties needed for Born functions
    water_props = water(["epsilon", "rho"], T=T, P=P)
    
    # Basic Born functions (simplified - full implementation would include all derivatives)
    epsilon = water_props['epsilon']
    rho = water_props['rho']
    
    # Dielectric constant derivatives (simplified approximations)
    deps_dT = -np.gradient(epsilon) / np.gradient(T) if len(T) > 1 else np.zeros_like(T)
    deps_dP = np.zeros_like(T)  # Would need pressure derivative
    
    # Born functions (simplified)
    QBorn = 1.0 / epsilon
    XBorn = deps_dT / epsilon**2
    YBorn = np.gradient(XBorn) / np.gradient(T) if len(T) > 1 else np.zeros_like(T)
    NBorn = deps_dP / epsilon**2
    UBorn = np.zeros_like(T)  # Second derivative term
    
    return {
        'QBorn': QBorn,
        'XBorn': XBorn, 
        'YBorn': YBorn,
        'NBorn': NBorn,
        'UBorn': UBorn
    }