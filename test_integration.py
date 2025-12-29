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
