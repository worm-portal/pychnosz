"""
Expression utilities for formatted labels and text.

This module provides Python equivalents of the R functions in util.expression.R:
- ratlab(): Create formatted text for activity ratios
- expr_species(): Format chemical formula for display
- syslab(): Create formatted text for thermodynamic systems

Author: CHNOSZ Python port
"""

import re
from typing import Optional
from .formula import makeup

# Optional imports for ratlab_html
try:
    from wormutils import chemlabel
    from chemparse import parse_formula
    _HTML_DEPS_AVAILABLE = True
except ImportError:
    _HTML_DEPS_AVAILABLE = False


def ratlab(top: str = "K+", bottom: str = "H+", molality: bool = False,
           reverse_charge: bool = False) -> str:
    """
    Create formatted text label for activity ratio.

    This function generates a LaTeX-formatted string suitable for use as
    axis labels in matplotlib plots, showing the ratio of activities of
    two ions raised to appropriate powers based on their charges.

    Parameters
    ----------
    top : str, default "K+"
        Chemical formula for the numerator ion
    bottom : str, default "H+"
        Chemical formula for the denominator ion
    molality : bool, default False
        If True, use 'm' (molality) instead of 'a' (activity)
    reverse_charge : bool, default False
        If True, reverse charge order in formatting (e.g., "Fe+3" becomes "Fe^{3+}")
        If False, keep original order (e.g., "Fe+3" becomes "Fe^{+3}")

    Returns
    -------
    str
        LaTeX-formatted string for the activity ratio label

    Examples
    --------
    >>> ratlab("K+", "H+")
    'log($a_{K^{+}}$ / $a_{H^{+}}$)'

    >>> ratlab("Ca+2", "H+")
    'log($a_{Ca^{+2}}$ / $a_{H^{+}}^{2}$)'

    >>> ratlab("Ca+2", "H+", reverse_charge=True)
    'log($a_{Ca^{2+}}$ / $a_{H^{+}}^{2}$)'

    >>> ratlab("Mg+2", "Ca+2")
    'log($a_{Mg^{+2}}$ / $a_{Ca^{+2}}$)'

    Notes
    -----
    The exponents are determined by the charges of the ions to maintain
    charge balance in the ratio. For example, for Ca+2/H+, the H+ term
    is squared because Ca has a +2 charge.

    The output format is compatible with matplotlib's LaTeX rendering.
    In R CHNOSZ, this uses plotmath expressions; here we use LaTeX strings
    that matplotlib can render.
    """
    # Get the charges of the ions
    makeup_top = makeup(top)
    makeup_bottom = makeup(bottom)

    Z_top = makeup_top.get('Z', 0)
    Z_bottom = makeup_bottom.get('Z', 0)

    # The exponents for charge balance
    # If top has charge +2 and bottom has +1, bottom gets exponent 2
    exp_bottom = abs(Z_top)
    exp_top = abs(Z_bottom)

    # Format exponents (don't show if = 1)
    exp_top_str = "" if exp_top == 1 else f"^{{{int(exp_top)}}}"
    exp_bottom_str = "" if exp_bottom == 1 else f"^{{{int(exp_bottom)}}}"

    # Format the ion formulas for display
    top_formatted = _format_species_latex(top, reverse_charge=reverse_charge)
    bottom_formatted = _format_species_latex(bottom, reverse_charge=reverse_charge)

    # Choose activity or molality symbol
    a = "m" if molality else "a"

    # Build the expression
    # Format: log(a_top^exp / a_bottom^exp)
    numerator = f"${a}_{{{top_formatted}}}{exp_top_str}$"
    denominator = f"${a}_{{{bottom_formatted}}}{exp_bottom_str}$"

    label = f"log({numerator} / {denominator})"

    return label


def ratlab_html(top: str = "K+", bottom: str = "H+", molality: bool = False) -> str:
    """
    Create HTML-formatted text label for activity ratio (for Plotly/HTML rendering).

    This function generates an HTML-formatted string suitable for use with
    Plotly interactive plots, showing the ratio of activities of two ions
    raised to appropriate powers based on their charges.

    This is a companion function to ratlab() which produces LaTeX format for
    matplotlib. Use ratlab_html() when creating labels for diagram(..., interactive=True).

    Parameters
    ----------
    top : str, default "K+"
        Chemical formula for the numerator ion
    bottom : str, default "H+"
        Chemical formula for the denominator ion
    molality : bool, default False
        If True, use 'm' (molality) instead of 'a' (activity)

    Returns
    -------
    str
        HTML-formatted string for the activity ratio label

    Examples
    --------
    >>> ratlab_html("K+", "H+")
    'log(a<sub>K<sup>+</sup></sub>/a<sub>H<sup>+</sup></sub>)'

    >>> ratlab_html("Ca+2", "H+")
    'log(a<sub>Ca<sup>2+</sup></sub>/a<sup>2</sup><sub>H<sup>+</sup></sub>)'

    >>> ratlab_html("Mg+2", "Ca+2")
    'log(a<sub>Mg<sup>2+</sup></sub>/a<sub>Ca<sup>2+</sup></sub>)'

    Notes
    -----
    The exponents are determined by the charges of the ions to maintain
    charge balance in the ratio. For example, for Ca+2/H+, the H+ term
    is squared because Ca has a +2 charge.

    The output format uses HTML tags (<sub>, <sup>) compatible with Plotly.
    For matplotlib plots with LaTeX rendering, use ratlab() instead.

    Requires: WORMutils (for chemlabel) and chemparse (for parse_formula)

    See Also
    --------
    ratlab : LaTeX version for matplotlib
    """
    if not _HTML_DEPS_AVAILABLE:
        raise ImportError(
            "ratlab_html() requires 'WORMutils' and 'chemparse' packages.\n"
            "Install with: pip install WORMutils chemparse"
        )

    # Parse the formulas to get charges
    top_formula = parse_formula(top)
    if "+" in top_formula.keys():
        top_charge = top_formula["+"]
    elif "-" in top_formula.keys():
        top_charge = -top_formula["-"]
    else:
        raise ValueError("Cannot create an ion ratio involving one or more neutral species.")

    bottom_formula = parse_formula(bottom)
    if "+" in bottom_formula.keys():
        bottom_charge = bottom_formula["+"]
    elif "-" in bottom_formula.keys():
        bottom_charge = -bottom_formula["-"]
    else:
        raise ValueError("Cannot create an ion ratio involving one or more neutral species.")

    # Convert to integers if whole numbers
    if top_charge.is_integer():
        top_charge = int(top_charge)

    if bottom_charge.is_integer():
        bottom_charge = int(bottom_charge)

    # The exponents for charge balance
    # If top has charge +2 and bottom has +1, bottom gets exponent 2
    exp_bottom = abs(top_charge)
    exp_top = abs(bottom_charge)

    # Format exponents as superscripts (don't show if = 1)
    if exp_top != 1:
        top_exp_str = "<sup>" + str(exp_top) + "</sup>"
    else:
        top_exp_str = ""

    if exp_bottom != 1:
        bottom_exp_str = "<sup>" + str(exp_bottom) + "</sup>"
    else:
        bottom_exp_str = ""

    # Choose activity or molality symbol
    if molality:
        sym = "m"
    else:
        sym = "a"

    # Format the chemical formulas with chemlabel
    top_formatted = chemlabel(top)
    bottom_formatted = chemlabel(bottom)

    # Build the HTML expression
    # Format: log(a_top^exp / a_bottom^exp)
    return f"log({sym}{top_exp_str}<sub>{top_formatted}</sub>/{sym}{bottom_exp_str}<sub>{bottom_formatted}</sub>)"


def _format_species_latex(formula: str, reverse_charge: bool = False) -> str:
    """
    Format a chemical formula for LaTeX rendering.

    This converts a chemical formula like "Ca+2" to LaTeX format.

    Parameters
    ----------
    formula : str
        Chemical formula
    reverse_charge : bool, default False
        If True, reverse charge order (e.g., "Fe+3" becomes "Fe^{3+}")
        If False, keep original order (e.g., "Fe+3" becomes "Fe^{+3}")

    Returns
    -------
    str
        LaTeX-formatted formula
    """
    # Handle charge at the end
    # Look for patterns like +, -, +2, -2, +3, etc.
    charge_match = re.search(r'([+-])(\d*)$', formula)

    if charge_match:
        sign = charge_match.group(1)
        magnitude = charge_match.group(2)

        # Get the base formula (without charge)
        base = formula[:charge_match.start()]

        # Format the charge
        if magnitude == '' or magnitude == '1':
            # Single charge: Ca+ or Ca-
            charge_str = f"^{{{sign}}}"
        else:
            # Multiple charges: Ca+2 can be Ca^{+3} or Ca^{3+}
            if reverse_charge:
                # Reversed: magnitude first, then sign (e.g., Ca^{2+})
                charge_str = f"^{{{magnitude}{sign}}}"
            else:
                # Original order: sign first, then magnitude (e.g., Ca^{+2})
                charge_str = f"^{{{sign}{magnitude}}}"

        # Add subscripts for numbers in the base formula
        base_formatted = _add_subscripts(base)

        return f"{base_formatted}{charge_str}"
    else:
        # No charge, just add subscripts
        return _add_subscripts(formula)


def _add_subscripts(formula: str) -> str:
    """
    Add LaTeX subscripts for numbers in a chemical formula.

    Parameters
    ----------
    formula : str
        Chemical formula without charge

    Returns
    -------
    str
        Formula with subscripts in LaTeX format
    """
    # Replace numbers following letters with subscripts
    # H2O becomes H_{2}O, CO2 becomes CO_{2}
    result = re.sub(r'([A-Z][a-z]?)(\d+)', r'\1_{\2}', formula)
    return result


def syslab(system: list = None, dash: str = "-") -> str:
    """
    Create formatted text for thermodynamic system.

    This generates a label showing the components of a thermodynamic system,
    separated by dashes (or other separator).

    Parameters
    ----------
    system : list of str, optional
        List of component formulas. Default: ["K2O", "Al2O3", "SiO2", "H2O"]
    dash : str, default "-"
        Separator between components

    Returns
    -------
    str
        LaTeX-formatted string for the system label

    Examples
    --------
    >>> syslab(["K2O", "Al2O3", "SiO2", "H2O"])
    '$K_{2}O-Al_{2}O_{3}-SiO_{2}-H_{2}O$'

    >>> syslab(["CaO", "MgO", "SiO2"], dash="–")
    '$CaO–MgO–SiO_{2}$'
    """
    if system is None:
        system = ["K2O", "Al2O3", "SiO2", "H2O"]

    # Format each component
    formatted_components = []
    for component in system:
        formatted = _add_subscripts(component)
        formatted_components.append(formatted)

    # Join with separator
    label = dash.join(formatted_components)

    # Wrap in LaTeX math mode
    return f"${label}$"


def syslab_html(system: list = None, dash: str = "-") -> str:
    """
    Create HTML-formatted text for thermodynamic system (for Plotly).

    This generates a label showing the components of a thermodynamic system,
    separated by dashes (or other separator), using HTML formatting compatible
    with Plotly instead of LaTeX.

    Parameters
    ----------
    system : list of str, optional
        List of component formulas. Default: ["K2O", "Al2O3", "SiO2", "H2O"]
    dash : str, default "-"
        Separator between components

    Returns
    -------
    str
        HTML-formatted string for the system label

    Examples
    --------
    >>> syslab_html(["K2O", "Al2O3", "SiO2", "H2O"])
    'K<sub>2</sub>O-Al<sub>2</sub>O<sub>3</sub>-SiO<sub>2</sub>-H<sub>2</sub>O'

    >>> syslab_html(["CaO", "MgO", "SiO2"], dash="–")
    'CaO–MgO–SiO<sub>2</sub>'

    Notes
    -----
    Use this function instead of syslab() when creating titles for interactive
    (Plotly) diagrams. The HTML formatting is compatible with Plotly's rendering.

    Requires: WORMutils (for chemlabel)
    """
    if not _HTML_DEPS_AVAILABLE:
        raise ImportError(
            "syslab_html() requires 'WORMutils' package.\n"
            "Install with: pip install WORMutils"
        )

    if system is None:
        system = ["K2O", "Al2O3", "SiO2", "H2O"]

    # Format each component using HTML via chemlabel
    formatted_components = []
    for component in system:
        formatted = chemlabel(component)
        formatted_components.append(formatted)

    # Join with separator (no HTML wrapper needed)
    label = dash.join(formatted_components)

    return label


def expr_species(formula: str, state: Optional[str] = None, use_state: bool = False) -> str:
    """
    Format a chemical species formula for display.

    This is a simplified version that returns LaTeX-formatted strings
    suitable for matplotlib. The R version returns plotmath expressions.

    Parameters
    ----------
    formula : str
        Chemical formula
    state : str, optional
        Physical state (aq, cr, gas, liq)
    use_state : bool, default False
        Whether to include state in the formatted output

    Returns
    -------
    str
        LaTeX-formatted formula string

    Examples
    --------
    >>> expr_species("H2O")
    '$H_{2}O$'

    >>> expr_species("Ca+2")
    '$Ca^{2+}$'

    >>> expr_species("SO4-2")
    '$SO_{4}^{2-}$'
    """
    formatted = _format_species_latex(formula)

    if use_state and state:
        # Add state subscript
        return f"${formatted}_{{{state}}}$"
    else:
        return f"${formatted}$"


def describe_property(property: list = None, value: list = None,
                     digits: int = 0, oneline: bool = False,
                     ret_val: bool = False) -> list:
    """
    Create formatted text describing thermodynamic properties and their values.

    This function generates formatted strings for displaying property-value pairs
    in legends, typically for temperature, pressure, and other conditions.

    Parameters
    ----------
    property : list of str
        Property names (e.g., ["T", "P"])
    value : list
        Property values (e.g., [300, 1000])
    digits : int, default 0
        Number of decimal places to display
    oneline : bool, default False
        If True, combine all properties on one line (not implemented)
    ret_val : bool, default False
        If True, return only values with units (not property names)

    Returns
    -------
    list of str
        Formatted property descriptions

    Examples
    --------
    >>> describe_property(["T", "P"], [300, 1000])
    ['$T$ = 300 °C', '$P$ = 1000 bar']

    >>> describe_property(["T"], [25], digits=1)
    ['$T$ = 25.0 °C']

    Notes
    -----
    This is used to create legend entries showing the conditions
    used in thermodynamic calculations.
    """
    if property is None or value is None:
        raise ValueError("property or value is None")

    descriptions = []

    for i in range(len(property)):
        prop = property[i]
        val = value[i]

        # Get property symbol
        if prop == "T":
            prop_str = "$T$"
            if val == "Psat" or val == "NA":
                val_str = "$P_{sat}$"
            else:
                val_formatted = format(round(float(val), digits), f'.{digits}f')
                val_str = f"{val_formatted} °C"
        elif prop == "P":
            prop_str = "$P$"
            if val == "Psat" or val == "NA":
                val_str = "$P_{sat}$"
            else:
                val_formatted = format(round(float(val), digits), f'.{digits}f')
                val_str = f"{val_formatted} bar"
        elif prop == "pH":
            prop_str = "pH"
            val_formatted = format(round(float(val), digits), f'.{digits}f')
            val_str = val_formatted
        elif prop == "Eh":
            prop_str = "Eh"
            val_formatted = format(round(float(val), digits), f'.{digits}f')
            val_str = f"{val_formatted} V"
        elif prop == "IS":
            prop_str = "$IS$"
            val_formatted = format(round(float(val), digits), f'.{digits}f')
            val_str = val_formatted
        else:
            prop_str = f"${prop}$"
            val_formatted = format(round(float(val), digits), f'.{digits}f')
            val_str = val_formatted

        if ret_val:
            descriptions.append(val_str)
        else:
            descriptions.append(f"{prop_str} = {val_str}")

    return descriptions


def describe_basis(ibasis: list = None, digits: int = 1,
                  oneline: bool = False, molality: bool = False,
                  use_pH: bool = True) -> list:
    """
    Create formatted text describing basis species activities/fugacities.

    This function generates formatted strings for displaying the chemical
    activities or fugacities of basis species, typically for plot legends.

    Parameters
    ----------
    ibasis : list of int, optional
        Indices of basis species to describe (1-based). If None, describes all.
    digits : int, default 1
        Number of decimal places to display
    oneline : bool, default False
        If True, combine all species on one line (not fully implemented)
    molality : bool, default False
        If True, use molality (m) instead of activity (a)
    use_pH : bool, default True
        If True, display H+ as pH instead of log a_H+

    Returns
    -------
    list of str
        Formatted basis species descriptions

    Examples
    --------
    >>> from pychnosz.core.basis import basis
    >>> basis(["H2O", "H+", "O2"], [-10, -7, -80])
    >>> describe_basis([2, 3])
    ['pH = 7.0', 'log $f_{O_2}$ = -80.0']

    >>> describe_basis()  # All basis species
    ['log $a_{H_2O}$ = -10.0', 'pH = 7.0', 'log $f_{O_2}$ = -80.0']

    Notes
    -----
    This is used to create legend entries showing the basis species
    activities used in thermodynamic calculations.
    """
    from ..core.basis import get_basis

    basis_df = get_basis()
    if basis_df is None:
        raise RuntimeError("Basis species are not defined")

    # Default to all basis species
    if ibasis is None:
        ibasis = list(range(1, len(basis_df) + 1))

    # Convert to 0-based indexing
    ibasis_0 = [i - 1 for i in ibasis]

    descriptions = []

    for i in ibasis_0:
        species_name = basis_df.index[i]
        state = basis_df.iloc[i]['state']
        logact = basis_df.iloc[i]['logact']

        # Check if logact is numeric
        try:
            logact_val = float(logact)
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False

        if is_numeric:
            # Handle H+ specially with pH
            if species_name == "H+" and use_pH:
                pH_val = -logact_val
                val_formatted = format(round(pH_val, digits), f'.{digits}f')
                descriptions.append(f"pH = {val_formatted}")
            else:
                # Format the activity/fugacity
                val_formatted = format(round(logact_val, digits), f'.{digits}f')

                # Determine if it's activity or fugacity based on state
                if state in ['aq', 'liq', 'cr']:
                    a_or_f = "a" if not molality else "m"
                else:
                    a_or_f = "f"

                # Format the species name
                species_formatted = _format_species_latex(species_name)

                descriptions.append(f"log ${a_or_f}_{{{species_formatted}}}$ = {val_formatted}")
        else:
            # Non-numeric value (buffer)
            if species_name == "H+" and use_pH:
                descriptions.append(f"pH = {logact}")
            else:
                # For buffers, just show the buffer name
                if state in ['aq', 'liq', 'cr']:
                    a_or_f = "a" if not molality else "m"
                else:
                    a_or_f = "f"

                species_formatted = _format_species_latex(species_name)
                descriptions.append(f"${a_or_f}_{{{species_formatted}}}$ = {logact}")

    return descriptions


def describe_property_html(property: list = None, value: list = None,
                           digits: int = 0, oneline: bool = False,
                           ret_val: bool = False) -> list:
    """
    Create HTML-formatted text describing thermodynamic properties (for Plotly).

    This function generates HTML-formatted strings for displaying thermodynamic
    properties and their values, typically for plot legends in interactive diagrams.

    Parameters
    ----------
    property : list of str
        Property names (e.g., ["T", "P"])
    value : list
        Property values
    digits : int, default 0
        Number of decimal places to display
    oneline : bool, default False
        If True, format on one line (not implemented)
    ret_val : bool, default False
        If True, return only values without property names

    Returns
    -------
    list of str
        HTML-formatted property descriptions

    Examples
    --------
    >>> describe_property_html(["T", "P"], [300, 1000])
    ['<i>T</i> = 300 °C', '<i>P</i> = 1000 bar']

    Notes
    -----
    Use this instead of describe_property() when creating legends for
    interactive (Plotly) diagrams.
    """
    if property is None or value is None:
        raise ValueError("property or value is None")

    descriptions = []

    for i in range(len(property)):
        prop = property[i]
        val = value[i]

        # Get property symbol (HTML format)
        if prop == "T":
            prop_str = "<i>T</i>"
            if val == "Psat" or val == "NA":
                val_str = "<i>P</i><sub>sat</sub>"
            else:
                val_formatted = format(round(float(val), digits), f'.{digits}f')
                val_str = f"{val_formatted} °C"
        elif prop == "P":
            prop_str = "<i>P</i>"
            if val == "Psat" or val == "NA":
                val_str = "<i>P</i><sub>sat</sub>"
            else:
                val_formatted = format(round(float(val), digits), f'.{digits}f')
                val_str = f"{val_formatted} bar"
        elif prop == "pH":
            prop_str = "pH"
            val_formatted = format(round(float(val), digits), f'.{digits}f')
            val_str = val_formatted
        elif prop == "Eh":
            prop_str = "Eh"
            val_formatted = format(round(float(val), digits), f'.{digits}f')
            val_str = f"{val_formatted} V"
        elif prop == "IS":
            prop_str = "<i>IS</i>"
            val_formatted = format(round(float(val), digits), f'.{digits}f')
            val_str = val_formatted
        else:
            prop_str = f"<i>{prop}</i>"
            val_formatted = format(round(float(val), digits), f'.{digits}f')
            val_str = val_formatted

        if ret_val:
            descriptions.append(val_str)
        else:
            descriptions.append(f"{prop_str} = {val_str}")

    return descriptions


def describe_basis_html(ibasis: list = None, digits: int = 1,
                        oneline: bool = False, molality: bool = False,
                        use_pH: bool = True) -> list:
    """
    Create HTML-formatted text describing basis species (for Plotly).

    This function generates HTML-formatted strings for displaying the chemical
    activities or fugacities of basis species, typically for plot legends in
    interactive diagrams.

    Parameters
    ----------
    ibasis : list of int, optional
        Indices of basis species to describe (1-based). If None, describes all.
    digits : int, default 1
        Number of decimal places to display
    oneline : bool, default False
        If True, combine all species on one line (not fully implemented)
    molality : bool, default False
        If True, use molality (m) instead of activity (a)
    use_pH : bool, default True
        If True, display H+ as pH instead of log a_H+

    Returns
    -------
    list of str
        HTML-formatted basis species descriptions

    Examples
    --------
    >>> from pychnosz.core.basis import basis
    >>> basis(["H2O", "H+", "O2"], [-10, -7, -80])
    >>> describe_basis_html([2, 3])
    ['pH = 7.0', 'log <i>f</i><sub>O<sub>2</sub></sub> = -80.0']

    >>> describe_basis_html([4])  # CO2
    ['log <i>f</i><sub>CO<sub>2</sub></sub> = -1.0']

    Notes
    -----
    Use this instead of describe_basis() when creating legends for
    interactive (Plotly) diagrams.
    """
    if not _HTML_DEPS_AVAILABLE:
        raise ImportError(
            "describe_basis_html() requires 'WORMutils' package.\n"
            "Install with: pip install WORMutils"
        )

    from ..core.basis import get_basis

    basis_df = get_basis()
    if basis_df is None:
        raise RuntimeError("Basis species are not defined")

    # Default to all basis species
    if ibasis is None:
        ibasis = list(range(1, len(basis_df) + 1))

    # Convert to 0-based indexing
    ibasis_0 = [i - 1 for i in ibasis]

    descriptions = []

    for i in ibasis_0:
        species_name = basis_df.index[i]
        state = basis_df.iloc[i]['state']
        logact = basis_df.iloc[i]['logact']

        # Check if logact is numeric
        try:
            logact_val = float(logact)
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False

        if is_numeric:
            # Handle H+ specially with pH
            if species_name == "H+" and use_pH:
                pH_val = -logact_val
                val_formatted = format(round(pH_val, digits), f'.{digits}f')
                descriptions.append(f"pH = {val_formatted}")
            else:
                # Format the activity/fugacity
                val_formatted = format(round(logact_val, digits), f'.{digits}f')

                # Determine if it's activity or fugacity based on state
                if state in ['aq', 'liq', 'cr']:
                    a_or_f = "a" if not molality else "m"
                else:
                    a_or_f = "f"

                # Format the species name using HTML
                species_formatted = chemlabel(species_name)

                descriptions.append(f"log <i>{a_or_f}</i><sub>{species_formatted}</sub> = {val_formatted}")
        else:
            # Non-numeric value (buffer)
            if species_name == "H+" and use_pH:
                descriptions.append(f"pH = {logact}")
            else:
                # For buffers, just show the buffer name
                if state in ['aq', 'liq', 'cr']:
                    a_or_f = "a" if not molality else "m"
                else:
                    a_or_f = "f"

                species_formatted = chemlabel(species_name)
                descriptions.append(f"<i>{a_or_f}</i><sub>{species_formatted}</sub> = {logact}")

    return descriptions


def add_legend(ax, labels: list = None, loc: str = 'best',
              frameon: bool = False, fontsize: float = 9, **kwargs):
    """
    Add a legend to a diagram with matplotlib or Plotly formatting.

    This is a convenience function that adds a legend with sensible
    defaults matching R CHNOSZ legend styling. Works with both matplotlib
    and Plotly figures.

    Parameters
    ----------
    ax : matplotlib.axes.Axes or plotly.graph_objs.Figure
        Axes/Figure object to add legend to. For interactive diagrams,
        pass the figure from d['fig'] or d['ax'].
    labels : list of str
        Legend labels (can be from describe_property, describe_basis, etc.)
    loc : str, default 'best'
        Legend location. Options: 'best', 'upper left', 'upper right',
        'lower left', 'lower right', 'right', 'center left', 'center right',
        'lower center', 'upper center', 'center'
        For Plotly: 'best' defaults to 'lower right'
    frameon : bool, default False
        Whether to draw a frame around the legend (R bty="n" equivalent)
    fontsize : float, default 9
        Font size for legend text (R cex=0.9 equivalent)
    **kwargs
        Additional arguments passed to matplotlib legend() or Plotly annotation

    Returns
    -------
    matplotlib.legend.Legend or plotly.graph_objs.Figure
        The legend object (matplotlib) or the figure (Plotly)

    Examples
    --------
    >>> from pychnosz.utils.expression import add_legend, describe_property
    >>> # Matplotlib diagram with plot_it=False
    >>> d1 = diagram(a, interactive=False, plot_it=False)
    >>> dprop = describe_property(["T", "P"], [300, 1000])
    >>> add_legend(d1['ax'], dprop, loc='lower right')
    >>> # Display the figure in Jupyter:
    >>> from IPython.display import display
    >>> display(d1['fig'])
    >>> # Or save it:
    >>> d1['fig'].savefig('diagram.png')

    >>> # Plotly diagram
    >>> d1 = diagram(a, interactive=True, plot_it=False)
    >>> dprop = describe_property(["T", "P"], [300, 1000])
    >>> add_legend(d1['fig'], dprop, loc='lower right')
    >>> d1['fig'].show()

    Notes
    -----
    Common R legend locations and their matplotlib equivalents:
    - "bottomright" → "lower right"
    - "topleft" → "upper left"
    - "topright" → "upper right"
    - "bottomleft" → "lower left"

    When using plot_it=False, you need to explicitly display the figure after
    adding legends. In Jupyter notebooks, use display(d['fig']) or d['fig'].show()
    for Plotly diagrams. Outside Jupyter, use plt.show() or save with d['fig'].savefig().
    """
    if labels is None:
        raise ValueError("labels must be provided")

    # Detect if this is a Plotly figure
    is_plotly = _is_plotly_figure(ax)

    if is_plotly:
        return _add_plotly_legend(ax, labels, loc, frameon, fontsize, **kwargs)
    else:
        return _add_matplotlib_legend(ax, labels, loc, frameon, fontsize, **kwargs)


def _is_plotly_figure(fig):
    """Check if object is a Plotly figure."""
    return hasattr(fig, 'add_annotation') and hasattr(fig, 'update_layout')


def _add_matplotlib_legend(ax, labels, loc, frameon, fontsize, **kwargs):
    """Add legend to matplotlib axes."""
    # Handle R-style location names
    loc_map = {
        'bottomright': 'lower right',
        'bottomleft': 'lower left',
        'topleft': 'upper left',
        'topright': 'upper right',
        'bottom': 'lower center',
        'top': 'upper center',
        'left': 'center left',
        'right': 'center right'
    }

    # Convert R-style location to matplotlib style
    if loc.lower() in loc_map:
        loc = loc_map[loc.lower()]

    # Create legend with text-only labels (no symbols)
    # This matches R's legend behavior when just providing character vectors
    # We need to create invisible handles for matplotlib to work properly
    from matplotlib.patches import Rectangle

    # Create invisible (alpha=0) dummy handles for each label
    handles = [Rectangle((0, 0), 1, 1, fc="white", ec="white", alpha=0)
               for _ in labels]

    # Create legend with invisible handles and no spacing
    legend = ax.legend(handles, labels, loc=loc, frameon=frameon,
                      fontsize=fontsize, handlelength=0, handletextpad=0,
                      **kwargs)

    return legend


def _add_plotly_legend(fig, labels, loc, frameon, fontsize, **kwargs):
    """Add legend-style annotation to Plotly figure."""
    # Handle R-style location names and map to Plotly coordinates
    loc_map = {
        'best': 'lower right',
        'bottomright': 'lower right',
        'bottomleft': 'lower left',
        'topleft': 'upper left',
        'topright': 'upper right',
        'bottom': 'lower center',
        'top': 'upper center',
        'left': 'center left',
        'right': 'center right',
        'lower right': 'lower right',
        'lower left': 'lower left',
        'upper left': 'upper left',
        'upper right': 'upper right',
        'lower center': 'lower center',
        'upper center': 'upper center',
        'center left': 'center left',
        'center right': 'center right',
        'center': 'center'
    }

    # Normalize location
    loc_normalized = loc_map.get(loc.lower(), 'lower right')

    # Map to Plotly anchor and position coordinates
    # Format: (x, y, xanchor, yanchor)
    plotly_positions = {
        'upper left': (0.02, 0.98, 'left', 'top'),
        'upper right': (0.98, 0.98, 'right', 'top'),
        'lower left': (0.02, 0.02, 'left', 'bottom'),
        'lower right': (0.98, 0.02, 'right', 'bottom'),
        'upper center': (0.5, 0.98, 'center', 'top'),
        'lower center': (0.5, 0.02, 'center', 'bottom'),
        'center left': (0.02, 0.5, 'left', 'middle'),
        'center right': (0.98, 0.5, 'right', 'middle'),
        'center': (0.5, 0.5, 'center', 'middle')
    }

    x, y, xanchor, yanchor = plotly_positions[loc_normalized]

    # Build legend text
    legend_text = '<br>'.join(labels)

    # Add annotation as legend
    fig.add_annotation(
        x=x,
        y=y,
        xref='paper',
        yref='paper',
        text=legend_text,
        showarrow=False,
        xanchor=xanchor,
        yanchor=yanchor,
        font=dict(size=fontsize),
        bgcolor='rgba(255, 255, 255, 0.8)' if not frameon else 'rgba(255, 255, 255, 1)',
        bordercolor='black' if frameon else 'rgba(0, 0, 0, 0)',
        borderwidth=1 if frameon else 0,
        borderpad=4,
        align='left'
    )

    return fig


def set_title(ax_or_fig, title: str, fontsize: float = 12, **kwargs):
    """
    Set title on a matplotlib axes or Plotly figure.

    This function provides a unified interface for setting titles on both
    matplotlib and Plotly plots, allowing seamless switching between
    interactive=True and interactive=False.

    Parameters
    ----------
    ax_or_fig : matplotlib.axes.Axes or plotly.graph_objs.Figure
        Axes or Figure object to set title on
    title : str
        The title text
    fontsize : float, default 12
        Font size for the title
    **kwargs
        Additional arguments passed to matplotlib set_title() or Plotly update_layout()

    Returns
    -------
    matplotlib.text.Text or plotly.graph_objs.Figure
        The title object (matplotlib) or the figure (Plotly)

    Examples
    --------
    >>> from pychnosz.utils.expression import set_title, syslab
    >>> # Matplotlib diagram
    >>> d1 = diagram(a, interactive=False, plot_it=False)
    >>> title_text = syslab(["H2O", "CO2", "CaO", "MgO", "SiO2"])
    >>> set_title(d1['ax'], title_text, fontsize=12)
    >>> # Display the figure in Jupyter:
    >>> from IPython.display import display
    >>> display(d1['fig'])

    >>> # Plotly diagram
    >>> d1 = diagram(a, interactive=True, plot_it=False)
    >>> title_text = syslab_html(["H2O", "CO2", "CaO", "MgO", "SiO2"])
    >>> set_title(d1['ax'], title_text, fontsize=12)
    >>> d1['fig'].show()

    Notes
    -----
    When using plot_it=False, you need to explicitly display the figure after
    setting the title. In Jupyter notebooks, use display(d['fig']) or d['fig'].show()
    for Plotly diagrams. Outside Jupyter, use plt.show() or save with d['fig'].savefig().
    """
    is_plotly = _is_plotly_figure(ax_or_fig)

    if is_plotly:
        # Plotly figure
        title_dict = {'text': title, 'x': 0.5, 'xanchor': 'center'}
        if fontsize:
            title_dict['font'] = {'size': fontsize}
        ax_or_fig.update_layout(title=title_dict, **kwargs)
        return ax_or_fig
    else:
        # Matplotlib axes
        return ax_or_fig.set_title(title, fontsize=fontsize, **kwargs)


# Export main functions
__all__ = ['ratlab', 'ratlab_html', 'expr_species', 'syslab', 'syslab_html',
           'describe_property', 'describe_property_html',
           'describe_basis', 'describe_basis_html',
           'add_legend', 'set_title']
