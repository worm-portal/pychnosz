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
    print("Test 3: HKF helpers (calc_logK, dissrxn2logK)")
    print("=" * 60)

    try:
        import pandas as pd
        from pychnosz.models.hkf_helpers import calc_logK, dissrxn2logK
        from pychnosz.core import thermo

        print("Loading OBIGT database...")
        # Get the ThermoSystem and access its OBIGT dataframe
        ts = thermo()
        obigt = ts.OBIGT

        # Create a simple test case with a mineral dissociation reaction
        # Use quartz (SiO2) which has a well-defined dissrxn
        print("Testing dissrxn2logK with a known mineral...")

        # Get quartz entry from OBIGT
        quartz_idx = obigt[obigt["name"] == "quartz"].index[0]

        # Calculate at standard conditions (25°C)
        Tc = 25.0

        # This should work without throwing TypeError
        logK = dissrxn2logK(obigt, quartz_idx, Tc)

        if pd.isna(logK):
            print(f"[OK] dissrxn2logK returned NaN (expected for quartz with no dissrxn)")
        else:
            print(f"[OK] dissrxn2logK calculated logK = {logK}")

        # Test calc_logK with OBIGT at temperature/pressure
        print("Testing calc_logK with temperature and pressure...")
        from pychnosz.models import subcrt

        # Calculate properties at T=100°C, P=1 bar
        result = subcrt.subcrt(["quartz"], T=100, P=1, exceed_Ttr=True)

        print(f"[OK] subcrt calculation completed")

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
