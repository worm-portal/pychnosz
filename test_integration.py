#!/usr/bin/env python
"""
Integration test for pychnosz package.

Tests critical functions used by downstream packages like aqequil.
"""

import sys

def test_import():
    """Test basic import."""
    print("=" * 60)
    print("Test 1: Import pychnosz")
    print("=" * 60)
    
    try:
        import pychnosz
        print("[OK] Successfully imported pychnosz")
        return True
    except Exception as e:
        print(f"[FAIL] Failed to import pychnosz: {e}")
        return False


def test_fortran_interface():
    """Test Fortran water properties interface."""
    print("\n" + "=" * 60)
    print("Test 2: Fortran interface (H2O properties)")
    print("=" * 60)
    
    try:
        from pychnosz.fortran import get_h2o92_interface
        h2o = get_h2o92_interface()
        props = h2o.calculate_properties(298.15, 1.0)
        
        if 'rho' not in props:
            print("[FAIL] No 'rho' in properties")
            return False
            
        print(f"[OK] Water density at 298.15 K, 1 bar: {props['rho']} g/cm³")
        return True
    except Exception as e:
        print(f"[FAIL] Fortran interface error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hkf_helpers():
    """Test HKF helper functions (used by aqequil)."""
    print("\n" + "=" * 60)
    print("Test 3: HKF helpers (subcrt function)")
    print("=" * 60)

    try:
        # Test the subcrt function which internally uses calc_logK and dissrxn2logK
        # This is the main function that aqequil actually uses
        from pychnosz import subcrt

        print("Testing subcrt with quartz at elevated temperature...")

        # Calculate properties at T=100°C, P=1 bar
        # This internally calls calc_logK which calls dissrxn2logK
        # If there's a pandas/numpy compatibility issue, it will fail here
        result = subcrt(["quartz"], T=100, P=1, exceed_Ttr=True)

        print(f"[OK] subcrt calculation completed successfully")

        # Test with a simple aqueous species
        print("Testing subcrt with aqueous species...")
        result2 = subcrt(["H2O", "H+", "OH-"], T=[25, 100], P=1)

        print(f"[OK] Aqueous species calculation completed successfully")

        return True

    except TypeError as e:
        if "only 0-dimensional arrays" in str(e):
            print(f"[FAIL] Pandas/numpy compatibility error: {e}")
            print("[FAIL] This is the bug that affects aqequil!")
            import traceback
            traceback.print_exc()
            return False
        else:
            raise
    except Exception as e:
        print(f"[FAIL] HKF helpers error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mosaic():
    """Test mosaic() against reference values from R CHNOSZ 2.2.0.

    The Cu-Cl aqueous complexes were updated from AZ01 to TTA+23 in the OBIGT
    database bundled here, so their reference affinities were regenerated with
    R CHNOSZ 2.2.0 running on the updated parameters.
    """
    print("\n" + "=" * 60)
    print("Test 4: mosaic (Cu-S-Cl-H2O)")
    print("=" * 60)

    try:
        import numpy as np
        import pychnosz as pc

        # Reference values from R CHNOSZ 2.2.0 with the current OBIGT database,
        # printed in column-major order.  Keyed by species name rather than by
        # OBIGT index, because indices shift whenever OBIGT is updated.
        ref_values = {
            "CuCl": [-8.861376, -8.861376, -8.861376, 1.790766, 1.790766,
                     1.790766, 12.442908, 12.442908, 12.442908],
            "CuCl2-": [-8.415097, -8.415097, -8.415097, 2.237045, 2.237045,
                       2.237045, 12.889187, 12.889187, 12.889187],
            "chalcocite": [-20.181588, -8.263455, -4.665054, 1.122696,
                           -20.590192, -68.577609, -40.958802, -84.503044,
                           -132.490460],
            "tenorite": [-32.278660, -20.278660, -8.278660, -10.974377,
                         1.025623, 13.025623, 10.329907, 22.329907, 34.329907],
            "copper": [0.0] * 9,
        }
        ref_predominant = [5, 5, 5, 2, 2, 4, 2, 4, 4]
        ref_bases_predominant = [1, 1, 4, 1, 4, 4, 3, 4, 4]

        pc.reset(messages=False)
        pc.basis(["Cu", "H2S", "Cl-", "H2O", "H+", "e-"], messages=False)
        pc.basis("H2S", -6)
        pc.basis("Cl-", -1)
        pc.species(["CuCl", "CuCl2-"], messages=False)
        sp = pc.species(["chalcocite", "tenorite", "copper"], add=True,
                        messages=False)
        # Map species name -> OBIGT index used as the key of A_species['values'].
        ispecies = dict(zip(sp['name'], sp['ispecies']))

        m = pc.mosaic(["H2S", "HS-", "HSO4-", "SO4-2"],
                      pH=[0, 12, 3], Eh=[-1, 1, 3], T=200, messages=False)

        worst = 0.0
        for name, ref in ref_values.items():
            got = np.asarray(m['A_species']['values'][ispecies[name]],
                             dtype=float).ravel(order='F')
            worst = max(worst, float(np.max(np.abs(got - np.asarray(ref)))))
        if worst > 1e-4:
            print(f"[FAIL] affinities differ from R by {worst:.3e}")
            return False
        print(f"[OK] affinities match R CHNOSZ (max diff {worst:.2e})")

        d = pc.diagram(m['A_species'], plot_it=False, messages=False)
        got = np.asarray(d['predominant']).ravel(order='F').astype(int).tolist()
        if got != ref_predominant:
            print(f"[FAIL] predominance {got} != R {ref_predominant}")
            return False
        print("[OK] species predominance matches R CHNOSZ")

        db = pc.diagram(m['A_bases'], plot_it=False, messages=False)
        got = np.asarray(db['predominant']).ravel(order='F').astype(int).tolist()
        if got != ref_bases_predominant:
            print(f"[FAIL] bases predominance {got} != R {ref_bases_predominant}")
            return False
        print("[OK] basis species predominance matches R CHNOSZ")

        return True

    except Exception as e:
        print(f"[FAIL] mosaic error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_equilibrate_mosaic():
    """Test equilibrate() on mosaic() output against R CHNOSZ 2.2.0."""
    print("\n" + "=" * 60)
    print("Test 5: equilibrate on mosaic output")
    print("=" * 60)

    try:
        import numpy as np
        import pychnosz as pc

        # Reference loga.equil from R CHNOSZ 2.2.0 with the current OBIGT
        # database (TTA+23 Cu-Cl complexes), in column-major order.
        # The basis species (H2S, HS-, HSO4-, SO4-2) are prepended to the
        # formed species by the mosaic branch of equilibrate().
        ref = {
            "H2S": [-6.000000, -6.000900, -7.337348, -14.483466, -6.000000,
                    -20.295786, -59.700433, -99.700305, -69.385782,
                    -105.512921, -144.917568, -184.917440],
            "HS-": [-12.683099, -8.683999, -6.020447, -9.166565, -12.683099,
                    -22.978885, -58.383532, -94.383404, -76.068881,
                    -108.196020, -143.600667, -179.600539],
            "HSO4-": [-113.048503, -77.049403, -42.385850, -13.531969,
                      -27.831368, -6.127154, -9.531800, -13.531673,
                      -6.000015, -6.127154, -9.531800, -13.531673],
            "SO4-2": [-117.516830, -77.517730, -38.854177, -6.000296,
                      -32.299695, -6.595481, -6.000128, -6.000000,
                      -10.468342, -6.595481, -6.000128, -6.000000],
            "CuCl": [-999.0] * 4 + [-4.308069, -4.301052, -999.0, -999.0,
                                    -4.060824] + [-999.0] * 3,
            "CuCl2-": [-999.0] * 4 + [-3.861790, -3.854774, -999.0, -999.0,
                                      -3.614546] + [-999.0] * 3,
            "chalcocite": [-999.0] * 12,
            "tenorite": [-999.0] * 6 + [-3.000005, -3.000000, -999.0,
                                        -3.000000, -3.000000, -3.000000],
            "copper": [-3.000000, -3.000001, -3.000000, -3.000000]
                      + [-999.0] * 8,
        }
        ref_predominant = [9, 9, 9, 9, 6, 6, 8, 8, 6, 8, 8, 8]

        pc.reset(messages=False)
        pc.basis(["Cu", "H2S", "Cl-", "H2O", "H+", "e-"], messages=False)
        pc.basis("H2S", -6)
        pc.basis("Cl-", -1)
        pc.species(["CuCl", "CuCl2-"], messages=False)
        pc.species(["chalcocite", "tenorite", "copper"], add=True, messages=False)

        m = pc.mosaic(["H2S", "HS-", "HSO4-", "SO4-2"],
                      pH=[0, 12, 4], Eh=[-1, 1, 3], T=200, messages=False)
        e = pc.equilibrate(m, loga_balance=-3, messages=False)

        names = list(e['species']['name'])
        if names != list(ref):
            print(f"[FAIL] species {names} != R {list(ref)}")
            return False
        print(f"[OK] combined species list matches R ({len(names)} species)")

        worst = 0.0
        for i, name in enumerate(names):
            got = np.asarray(e['loga_equil'][i], dtype=float).ravel(order='F')
            worst = max(worst, float(np.max(np.abs(got - np.asarray(ref[name])))))
        if worst > 1e-4:
            print(f"[FAIL] loga_equil differs from R by {worst:.3e}")
            return False
        print(f"[OK] loga_equil matches R CHNOSZ (max diff {worst:.2e})")

        d = pc.diagram(e, plot_it=False, messages=False)
        got = np.asarray(d['predominant']).ravel(order='F').astype(int).tolist()
        if got != ref_predominant:
            print(f"[FAIL] predominance {got} != R {ref_predominant}")
            return False
        print("[OK] predominance of combined object matches R CHNOSZ")

        return True

    except Exception as e:
        print(f"[FAIL] equilibrate/mosaic error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ZC_oxidation_states():
    """Test ZC() default values and user-supplied oxidation states."""
    print("\n" + "=" * 60)
    print("Test 6: ZC with user-supplied oxidation states")
    print("=" * 60)

    try:
        import warnings
        import numpy as np
        import pychnosz as pc

        # Defaults must be unchanged (these also match R CHNOSZ ZC())
        defaults = {"CH4": -4.0, "CO2": 4.0, "C6H12O6": 0.0,
                    "CH3COO-": 0.0, "C2H6O": -2.0, "CH2O": 0.0}
        for formula, want in defaults.items():
            got = pc.ZC(formula)
            if not np.isclose(got, want):
                print(f"[FAIL] ZC({formula!r}) = {got}, expected {want}")
                return False
        if pc.ZC(["CH4", "CO2"]) != [-4.0, 4.0]:
            print("[FAIL] ZC() on a list of formulas")
            return False
        print("[OK] default oxidation states unchanged")

        # Supplying an element that has no default, including a fake one
        cases = [
            (("C10H14N5O7P",), {"P": 5}, 1.0),      # AMP, phosphate P
            (("C10H14N5O7Xx",), {"Xx": 5}, 1.0),    # fake element for P
            (("C10H12N5O6P-2",), {"P": 5}, 0.8),    # charged species
            (("CH3SO3H",), {"S": 5}, -3.0),         # override the S default
            (("C6H12O6",), {"O": -1}, -1.0),        # override the O default
            (("C10H14N5O7PXx",), {"P": 5, "Xx": 0}, 1.0),
        ]
        for args, kwargs, want in cases:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                got = pc.ZC(*args, **kwargs)
            if not np.isclose(got, want):
                print(f"[FAIL] ZC({args[0]!r}, {kwargs}) = {got}, expected {want}")
                return False
        print("[OK] user-supplied oxidation states applied")

        # An element with no oxidation state is dropped with a warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            got = pc.ZC("C10H14N5O7P")
            if not np.isclose(got, 1.5):
                print(f"[FAIL] ZC without P = {got}, expected 1.5")
                return False
            if not any("oxidation state" in str(x.message) for x in w):
                print("[FAIL] no warning for the dropped element")
                return False
        print("[OK] elements without an oxidation state warn and are dropped")

        # Bad element symbols are rejected rather than silently ignored
        for kwargs in [{"p": 5}, {"PP": 5}, {"C": 0}, {"Z": 1}]:
            try:
                pc.ZC("C10H14N5O7P", **kwargs)
            except ValueError:
                continue
            print(f"[FAIL] ZC() accepted invalid argument {kwargs}")
            return False
        print("[OK] invalid oxidation state arguments raise ValueError")

        return True

    except Exception as e:
        print(f"[FAIL] ZC error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ratlab():
    """Test that ratlab() and ratlab_html() produce charge-balanced ratios."""
    print("\n" + "=" * 60)
    print("Test 7: ratlab() and ratlab_html() activity ratios")
    print("=" * 60)

    try:
        import re
        from pychnosz.utils.formula import makeup
        from pychnosz.utils import expression as expr

        # The exponents keep the ratio charge-balanced, so the signs of the
        # charges matter (as in R CHNOSZ ratlab()). An anion over H+ becomes a
        # product: the ratio for HCO3- over H+ is (a HCO3-)(a H+).
        pairs = [("K+", "H+"), ("Ca+2", "H+"), ("HCO3-", "H+"),
                 ("SO4-2", "H+"), ("Mg+2", "Ca+2"), ("HCO3-", "Ca+2"),
                 ("Ca+2", "SO4-2"), ("Fe+2", "Cu+")]

        def net_charge(label, top, bottom, product_op, quotient_op, exp_pattern):
            """Net charge of a ratio label, or None if it cannot be parsed."""
            if not (label.startswith("log(") and label.endswith(")")):
                print(f"[FAIL] {label!r} is not of the form log(...)")
                return None
            inner = label[4:-1]

            # A product multiplies the two activities, a quotient divides them
            if re.search(product_op, inner):
                terms, sign = re.split(product_op, inner, maxsplit=1), 1
            elif re.search(quotient_op, inner):
                terms, sign = re.split(quotient_op, inner, maxsplit=1), -1
            else:
                print(f"[FAIL] no ratio operator in {label!r}")
                return None
            if len(terms) != 2:
                print(f"[FAIL] {label!r} does not have two activity terms")
                return None

            # An activity without an exponent is raised to the first power
            exps = []
            for term in terms:
                found = re.search(exp_pattern, term)
                exps.append(int(found.group(1)) if found else 1)

            Z_top = makeup(top).get("Z", 0)
            Z_bottom = makeup(bottom).get("Z", 0)
            return exps[0] * Z_top + sign * exps[1] * Z_bottom

        # log(a<sup>2</sup><sub>Ca<sup>+2</sup></sub>/...)
        html_exp = r"^[am]<sup>(-?\d+)</sup><sub>"
        # log($a_{Ca^{+2}}^{2}$ / ...)
        latex_exp = r"\^\{(-?\d+)\}\$$"

        for top, bottom in pairs:
            label = expr.ratlab(top, bottom)
            Z = net_charge(label, top, bottom, r" \$\\cdot\$ ", " / ", latex_exp)
            if Z != 0:
                print(f"[FAIL] ratlab({top!r}, {bottom!r}) = {label!r} "
                      f"has net charge {Z}, expected 0")
                return False
        print("[OK] ratlab() ratios are charge-balanced")

        if not expr._HTML_DEPS_AVAILABLE:
            print("[SKIP] ratlab_html() needs WORMutils and chemparse")
        else:
            for top, bottom in pairs:
                label = expr.ratlab_html(top, bottom)
                # A "/" inside a closing HTML tag is not the ratio operator
                Z = net_charge(label, top, bottom, "·", r"/(?=[am])", html_exp)
                if Z != 0:
                    print(f"[FAIL] ratlab_html({top!r}, {bottom!r}) = {label!r} "
                          f"has net charge {Z}, expected 0")
                    return False
            print("[OK] ratlab_html() ratios are charge-balanced")

            # Guard the case that AqEquil reaction path diagrams depend on
            if "·" not in expr.ratlab_html("HCO3-", "H+"):
                print("[FAIL] ratlab_html('HCO3-') should be a product")
                return False
            print("[OK] an anion over H+ is written as a product")

        return True

    except Exception as e:
        print(f"[FAIL] ratlab error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PYCHNOSZ INTEGRATION TEST SUITE")
    print("=" * 60)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print("=" * 60)
    
    tests = [
        ("Import", test_import),
        ("Fortran Interface", test_fortran_interface),
        ("HKF Helpers", test_hkf_helpers),
        ("Mosaic", test_mosaic),
        ("Equilibrate Mosaic", test_equilibrate_mosaic),
        ("ZC Oxidation States", test_ZC_oxidation_states),
        ("Activity Ratio Labels", test_ratlab),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[FAIL] Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    print("=" * 60)
    
    # Return exit code
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n[PASS] All tests passed!")
        return 0
    else:
        print("\n[FAIL] Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
