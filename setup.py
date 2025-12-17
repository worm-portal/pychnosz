"""
Setup script for pychnosz.

This setup.py is needed to force the creation of platform-specific wheels
since we include compiled Fortran binaries (.so/.dll/.dylib).
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """
    Distribution which always forces a binary package with platform name.

    This ensures that wheels are built as platform-specific (e.g.,
    pychnosz-1.0.0-py3-none-linux_x86_64.whl) rather than pure Python
    wheels (py3-none-any.whl).
    """

    def has_ext_modules(self):
        """
        Tell setuptools this distribution has extension modules.
        This forces platform-specific wheel names.
        """
        return True


if __name__ == "__main__":
    setup(
        distclass=BinaryDistribution,
    )
