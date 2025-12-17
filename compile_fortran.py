#!/usr/bin/env python
"""
Compile Fortran source code to shared library for the current platform.

This script is called during the wheel build process to compile the H2O92
Fortran subroutine into a platform-specific shared library.
"""

import os
import sys
import subprocess
import shutil
import platform
import tempfile
from pathlib import Path


def find_gfortran():
    """Find gfortran compiler on the system."""
    # Check if explicit gfortran path is provided via environment variable
    explicit_path = os.environ.get('GFORTRAN_PATH')
    if explicit_path and os.path.isfile(explicit_path):
        print(f"Using explicit gfortran path from GFORTRAN_PATH: {explicit_path}")
        return explicit_path

    # Try common gfortran locations
    candidates = ['gfortran', 'gfortran-11', 'gfortran-12', 'gfortran-13', 'gfortran-14']

    for compiler in candidates:
        compiler_path = shutil.which(compiler)
        if compiler_path:
            return compiler_path

    raise RuntimeError(
        "gfortran compiler not found. Please install gfortran:\n"
        "  Ubuntu/Debian: sudo apt-get install gfortran\n"
        "  macOS: brew install gcc\n"
        "  Windows: Install MinGW-w64 with gfortran support"
    )


def compile_fortran():
    """Compile H2O92D.f.orig to a shared library."""
    # Determine paths
    script_dir = Path(__file__).parent
    source_file = script_dir / "pychnosz" / "data" / "extdata" / "src" / "H2O92D.f.orig"
    output_dir = script_dir / "pychnosz" / "fortran"

    # Verify source file exists
    if not source_file.exists():
        raise FileNotFoundError(f"Fortran source not found: {source_file}")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output library name based on platform
    if sys.platform == "win32":
        lib_name = "h2o92.dll"
    elif sys.platform == "darwin":
        lib_name = "h2o92.dylib"
    else:  # Linux and other Unix-like
        lib_name = "h2o92.so"

    output_file = output_dir / lib_name

    # Find gfortran
    gfortran = find_gfortran()
    print(f"Using compiler: {gfortran}")
    print(f"Compiling {source_file.name} -> {lib_name}")

    # Detect macOS architecture for cross-compilation
    macos_arch = None
    if sys.platform == "darwin":
        # Check environment variable set by cibuildwheel
        archflags = os.environ.get('ARCHFLAGS', '')
        print(f"ARCHFLAGS environment variable: '{archflags}'")

        if 'x86_64' in archflags:
            macos_arch = 'x86_64'
        elif 'arm64' in archflags:
            macos_arch = 'arm64'
        else:
            # Fallback to machine architecture
            machine = platform.machine()
            macos_arch = 'arm64' if machine == 'arm64' else 'x86_64'
        print(f"macOS target architecture: {macos_arch}")
        print(f"Current machine architecture: {platform.machine()}")

    # gfortran doesn't recognize .f.orig extension, so copy to temp file with .f extension
    with tempfile.NamedTemporaryFile(suffix='.f', delete=False) as tmp_f:
        temp_source = Path(tmp_f.name)

    try:
        # Copy source to temp file with .f extension
        shutil.copy2(source_file, temp_source)
        print(f"Using temporary source file: {temp_source}")

        # Compile command
        # -shared: create a shared library
        # -fPIC: position-independent code (required for shared libraries)
        # -O2: optimization level 2
        # -o: output file
        cmd = [
            gfortran,
            "-shared",
            "-fPIC",
            "-O2",
            str(temp_source),
            "-o", str(output_file)
        ]

        # Platform-specific flags
        if sys.platform == "darwin":
            # macOS requires -dynamiclib for shared libraries
            cmd[1] = "-dynamiclib"
            # Add architecture flag for cross-compilation
            if macos_arch:
                cmd.insert(2, f"-arch")
                cmd.insert(3, macos_arch)
        elif sys.platform == "win32":
            # Windows: statically link all MinGW runtime libraries
            # This ensures the DLL has no external dependencies
            cmd.insert(2, "-static-libgfortran")
            cmd.insert(3, "-static-libgcc")
            cmd.insert(4, "-static")
            cmd.insert(5, "-lquadmath")

        print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            if result.stdout:
                print(result.stdout)

            if result.stderr:
                print(result.stderr, file=sys.stderr)

            print(f"[OK] Successfully compiled {lib_name}")
            print(f"  Output: {output_file}")
            print(f"  Size: {output_file.stat().st_size / 1024:.1f} KB")

            # Verify architecture on macOS
            if sys.platform == "darwin":
                print(f"Verifying architecture of compiled library...")
                try:
                    arch_check = subprocess.run(
                        ["lipo", "-info", str(output_file)],
                        capture_output=True,
                        text=True
                    )
                    print(f"  lipo output: {arch_check.stdout.strip()}")
                except FileNotFoundError:
                    print(f"  Warning: lipo not found, skipping architecture check")

            return True

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Compilation failed!", file=sys.stderr)
            print(f"Exit code: {e.returncode}", file=sys.stderr)
            if e.stdout:
                print(f"stdout: {e.stdout}", file=sys.stderr)
            if e.stderr:
                print(f"stderr: {e.stderr}", file=sys.stderr)
            raise RuntimeError(f"Failed to compile Fortran library: {e}")

    finally:
        # Clean up temporary file
        if temp_source.exists():
            temp_source.unlink()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Compiling Fortran library for pychnosz")
    print("=" * 60)
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")

    try:
        compile_fortran()
        print("=" * 60)
        print("Compilation successful!")
        print("=" * 60)
        return 0
    except Exception as e:
        print("=" * 60, file=sys.stderr)
        print(f"ERROR: {e}", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
