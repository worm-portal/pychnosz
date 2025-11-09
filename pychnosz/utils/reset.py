"""
Reset function for initializing the CHNOSZ thermodynamic system.

This provides the reset() function that initializes/resets the global
thermodynamic system, equivalent to reset() in the R version.
"""

from ..core.thermo import get_thermo_system


def reset(messages: bool = True):
    """
    Initialize or reset the CHNOSZ thermodynamic system.

    This function initializes the global thermodynamic system by loading
    all thermodynamic data files, setting up the OBIGT database, and
    preparing the system for calculations.

    This is equivalent to the reset() function in the R version of CHNOSZ.

    Parameters
    ----------
    messages : bool, default True
        Whether to print informational messages

    Examples
    --------
    >>> import pychnosz
    >>> pychnosz.reset()  # Initialize the system
    reset: thermodynamic system initialized
    """
    thermo_system = get_thermo_system()
    thermo_system.reset(messages=messages)