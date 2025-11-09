"""
Diagram module for plotting chemical activity and predominance diagrams.

This module provides Python equivalents of the R functions in diagram.R:
- diagram(): Plot equilibrium chemical activity and predominance diagrams
- Supporting utilities for 1D line plots and 2D predominance diagrams

Author: CHNOSZ Python port
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, List, Optional, Dict, Any, Tuple
import warnings
import copy
from ..utils.expression import _format_species_latex


def copy_plot(diagram_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a deep copy of a diagram result, allowing independent modification.

    This function addresses a fundamental limitation in Python plotting libraries:
    matplotlib figure and axes objects are mutable, so passing them between
    functions causes modifications to affect all references. This function
    creates a true deep copy that can be modified independently.

    Parameters
    ----------
    diagram_result : dict
        Result dictionary from diagram(), which may contain 'fig' and 'ax' keys

    Returns
    -------
    dict
        A deep copy of the diagram result with independent figure and axes objects

    Examples
    --------
    Manual copying workflow (advanced usage - normally use add_to parameter instead):

    >>> import pychnosz
    >>> # Create base plot (Plot A)
    >>> basis(['SiO2', 'Ca+2', 'Mg+2', 'CO2', 'H2O', 'O2', 'H+'])
    >>> species(['quartz', 'talc', 'chrysotile', 'forsterite'])
    >>> a = affinity(**{'Mg+2': [4, 10, 500], 'Ca+2': [5, 15, 500]})
    >>> plot_a = diagram(a, fill='terrain')
    >>>
    >>> # Manual approach: create copies first, then modify the axes directly
    >>> plot_a1 = copy_plot(plot_a)  # For modification 1
    >>> plot_a2 = copy_plot(plot_a)  # For modification 2
    >>> # ... then modify plot_a1['ax'] and plot_a2['ax'] directly
    >>>
    >>> # Recommended approach: use add_to parameter instead
    >>> # This automatically handles copying internally
    >>> basis('CO2', -1)
    >>> species(['calcite', 'dolomide'])
    >>> a2 = affinity(**{'Mg+2': [4, 10, 500], 'Ca+2': [5, 15, 500]})
    >>> plot_a1 = diagram(a2, type='saturation', add_to=plot_a, col='blue')
    >>> plot_a2 = diagram(a2, type='saturation', add_to=plot_a, col='red')
    >>> # Now you have three independent plots: plot_a, plot_a1, plot_a2

    Notes
    -----
    - This function uses copy.deepcopy() which works well for matplotlib figures
    - For very large plots, copying may be memory-intensive
    - Interactive plots (plotly) may not copy perfectly - test before relying on this
    - The copied plot is fully independent and can be saved, displayed, or modified
      without affecting the original

    Limitations
    -----------
    Python's matplotlib (unlike R's base graphics) uses mutable objects for plots.
    Without explicit copying, all references point to the same plot. This is a
    known limitation of matplotlib that this function works around.

    See Also
    --------
    diagram : Create plots that can be copied with this function
    """
    return copy.deepcopy(diagram_result)


def diagram(eout: Dict[str, Any],
            type: str = "auto",
            alpha: bool = False,
            balance: Optional[Union[str, float, List[float]]] = None,
            names: Optional[List[str]] = None,
            format_names: bool = True,
            xlab: Optional[str] = None,
            ylab: Optional[str] = None,
            xlim: Optional[List[float]] = None,
            ylim: Optional[List[float]] = None,
            col: Optional[Union[str, List[str]]] = None,
            col_names: Optional[Union[str, List[str]]] = None,
            lty: Optional[Union[str, int, List]] = None,
            lwd: Union[float, List[float]] = 1,
            cex: Union[float, List[float]] = 1.0,
            main: Optional[str] = None,
            fill: Optional[str] = None,
            fill_NA: str = "0.8",
            limit_water: Optional[bool] = None,
            plot_it: bool = True,
            add_to: Optional[Dict[str, Any]] = None,
            contour_method: Optional[Union[str, List[str]]] = "edge",
            messages: bool = True,
            interactive: bool = False,
            annotation: Optional[str] = None,
            annotation_coords: List[float] = [0, 0],
            width: int = 600,
            height: int = 520,
            save_as: Optional[str] = None,
            save_format: Optional[str] = None,
            save_scale: float = 1,
            normalize: Union[bool, List[bool]] = False,
            as_residue: bool = False,
            **kwargs) -> Dict[str, Any]:
    """
    Plot equilibrium chemical activity and predominance diagrams.

    This function creates plots from the output of affinity() or equilibrate().
    For 1D diagrams, it produces line plots showing how affinity or activity
    varies with a single variable. For 2D diagrams, it creates predominance
    field diagrams.

    Parameters
    ----------
    eout : dict
        Output from affinity() or equilibrate()
    type : str, default "auto"
        Type of diagram:
        - "auto" (default): Plot affinity values (A/2.303RT)
        - "loga.equil": Plot equilibrium activities from equilibrate()
        - "saturation": Draw affinity=0 contour lines (mineral saturation)
        - Basis species name (e.g., "O2", "H2O", "CO2"): Plot equilibrium
          log activity/fugacity of the specified basis species where affinity=0
          for each formed species. Useful for Eh-pH diagrams and showing
          oxygen/water fugacities at equilibrium.
    alpha : bool or str, default False
        Plot degree of formation instead of activities?
        If "balance", scale by balancing coefficients
    balance : str, float, or list of float, optional
        Balancing coefficients or method for balancing reactions
    names : list of str, optional
        Custom names for species (for labels)
    format_names : bool, default True
        Apply formatting to chemical formulas?
    xlab : str, optional
        Custom x-axis label
    ylab : str, optional
        Custom y-axis label
    xlim : list of float, optional
        X-axis limits [min, max]
    ylim : list of float, optional
        Y-axis limits [min, max]
    col : str or list of str, optional
        Line colors for 1-D plots and boundary lines in 2-D plots (matplotlib color specs)
    col_names : str or list of str, optional
        Text colors for field labels in 2-D plots (matplotlib color specs)
    lty : str, int, or list, optional
        Line styles (matplotlib linestyle specs)
    lwd : float or list of float, default 1
        Line widths for 1-D plots and boundary lines in 2-D predominance
        diagrams. Set to 0 to disable borders in 2-D diagrams. If fill is
        None and lwd > 0, uses white fill with black borders (R CHNOSZ default).
    cex : float or list of float, default 1.0
        Character expansion factor for text labels. Values > 1 make text larger,
        values < 1 make text smaller. Can be a single value or a list (one per species).
        Used for contour labels in type="saturation" plots.
    main : str, optional
        Plot title
    fill : str, optional
        Color palette for 2-D predominance diagrams. Can be any matplotlib
        colormap name (e.g., 'viridis', 'plasma', 'terrain', 'rainbow',
        'Set1', 'tab10', 'Pastel1'). If None, uses discrete colors from
        the default color cycle. Ignored for 1-D diagrams.
    fill_NA : str, default "0.8"
        Color for regions outside water stability limits (water instability regions).
        Matplotlib color specification (e.g., "0.8" for gray, "#CCCCCC").
        Set to "transparent" to disable shading. Default "0.8" matches R's "gray80".
    limit_water : bool, optional
        Whether to show water stability limits as shaded regions (default True for
        2-D diagrams). If True, also clips the diagram to the water stability region.
        Set to False to disable water stability shading.
    plot_it : bool, default True
        Display the plot?
    add_to : dict, optional
        A diagram result dictionary from a previous diagram() call. When provided,
        this plot will be AUTOMATICALLY COPIED and the new diagram will be added to
        the copy. This preserves the original plot while creating a modified version.
        The axes object is extracted from add_to['ax'].

        This parameter eliminates the need for a separate 'add' boolean - when
        add_to is provided, the function automatically operates in "add" mode.

        Example workflow:
        >>> plot_a = diagram(affinity1, fill='terrain')  # Create base plot
        >>> plot_a1 = diagram(affinity2, add_to=plot_a, col='blue')  # Copy and add
        >>> plot_a2 = diagram(affinity3, add_to=plot_a, col='red')   # Copy and add again
        >>> # plot_a remains unchanged, plot_a1 and plot_a2 are independent modifications
    contour_method : str or list of str, optional
        Method for labeling contour lines. Default "edge" labels at plot edges.
        Can be a single value (applied to all species) or a list (one per species).
        Set to None, NA, or "" to disable labels (only for type="saturation").
        In R CHNOSZ, different methods like "edge", "flattest", "simple" control
        label placement; in Python, this mainly controls whether labels are shown.
    interactive : bool, default False
        Create an interactive plot using Plotly instead of matplotlib?
        If True, calls diagram_interactive() with the appropriate parameters.
    annotation : str, optional
        For interactive plots only. Annotation text to add to the plot.
    annotation_coords : list of float, default [0, 0]
        For interactive plots only. Coordinates of annotation, where [0, 0] is
        bottom left and [1, 1] is top right.
    width : int, default 600
        For interactive plots only. Width of the plot in pixels.
    height : int, default 520
        For interactive plots only. Height of the plot in pixels.
    save_as : str, optional
        For interactive plots only. Provide a filename to save this figure.
        Filetype is determined by `save_format`.
    save_format : str, optional
        For interactive plots only. Desired format of saved or downloaded figure.
        Can be 'png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'eps', 'json', or 'html'.
        If 'html', an interactive plot will be saved.
    save_scale : float, default 1
        For interactive plots only. Multiply title/legend/axis/canvas sizes by
        this factor when saving the figure.
    **kwargs
        Additional arguments passed to matplotlib plotting functions

    Returns
    -------
    dict
        Dictionary containing:
        - plotvar : str, Variable that was plotted
        - plotvals : dict, Values that were plotted
        - names : list, Names used for labels
        - predominant : array or NA, Predominance matrix (for 2D)
        - balance : str or list, Balancing coefficients used
        - n.balance : list, Numerical balancing coefficients
        - ax : matplotlib.axes.Axes, The axes object used for plotting (if plot_it=True)
        - fig : matplotlib.figure.Figure, The figure object used for plotting (if plot_it=True)
        - All original eout contents

    Examples
    --------
    >>> import pychnosz
    >>> pychnosz.basis(["Fe2O3", "CO2", "H2O", "NH3", "H2S", "oxygen", "H+"],
    ...              [0, -3, 0, -4, -7, -80, -7])
    >>> pychnosz.species(["pyrite", "goethite"])
    >>> a = pychnosz.affinity(H2S=[-60, 20, 5], T=25, P=1)
    >>> d = diagram(a)

    Notes
    -----
    This implementation is based on R CHNOSZ diagram() function but adapted
    for Python's matplotlib plotting instead of R's base graphics. The key
    differences from diagram_from_WORM.py are:
    - Works directly with Python dict output from affinity() (no rpy2)
    - Uses matplotlib for 1D plots by default
    - Can optionally use plotly if requested
    """

    # Handle add_to parameter: automatically copy the provided plot
    # This extracts the axes object and creates an independent copy
    # When add_to is provided, we're in "add" mode
    ax = None
    add = add_to is not None
    plot_was_provided = add

    if add_to is not None:
        # Make a deep copy of the provided plot to preserve the original
        plot_copy = copy_plot(add_to)
        # Extract the axes from the copied plot
        if 'ax' in plot_copy:
            ax = plot_copy['ax']
        else:
            raise ValueError("The 'add_to' parameter must contain an 'ax' key (a diagram result dictionary)")

    # If interactive mode is requested, delegate to diagram_interactive
    if interactive:
        df, fig = diagram_interactive(
            eout=eout,
            type=type,
            main=main,
            borders=lwd,
            names=names,
            format_names=format_names,
            annotation=annotation,
            annotation_coords=annotation_coords,
            balance=balance,
            xlab=xlab,
            ylab=ylab,
            fill=fill,
            width=width,
            height=height,
            alpha=alpha,
            plot_it=plot_it,
            add=add,
            ax=ax,
            col=col,
            lty=lty,
            lwd=lwd,
            cex=cex,
            contour_method=contour_method,
            save_as=save_as,
            save_format=save_format,
            save_scale=save_scale,
            messages=messages
        )
        # Return in a format compatible with diagram's normal output
        # diagram_interactive returns (df, fig), wrap in a dict for consistency
        # Include eout data so water_lines() can access vars, vals, basis, etc.
        result = {
            **eout,  # Include all original eout data
            'df': df,
            'fig': fig,
            'ax': fig  # For compatibility, store fig in ax key for add=True workflow
        }
        return result

    # Check that eout is valid
    efun = eout.get('fun', '')
    if efun not in ['affinity', 'equilibrate', 'solubility']:
        raise ValueError("'eout' is not the output from affinity(), equilibrate(), or solubility()")

    # Determine if eout is from affinity() (as opposed to equilibrate())
    # Check for both Python naming (loga_equil) and R naming (loga.equil)
    eout_is_aout = 'loga_equil' not in eout and 'loga.equil' not in eout

    # Check if type is a basis species name
    plot_loga_basis = False
    if type not in ["auto", "saturation", "loga.equil", "loga_equil", "loga.balance", "loga_balance"]:
        # Check if type matches a basis species name
        if 'basis' in eout:
            basis_species = list(eout['basis'].index) if hasattr(eout['basis'], 'index') else []
            if type in basis_species:
                plot_loga_basis = True
                if alpha:
                    raise ValueError("equilibrium activities of basis species not available with alpha = TRUE")

    # Handle type="saturation" - requires affinity output
    if type == "saturation":
        if not eout_is_aout:
            raise ValueError("type='saturation' requires output from affinity(), not equilibrate()")
        # Set eout_is_aout flag
        eout_is_aout = True

    # Get number of dimensions
    # Handle both dict (affinity) and list (equilibrate) values structures
    if isinstance(eout['values'], dict):
        first_values = list(eout['values'].values())[0]
    elif isinstance(eout['values'], list):
        first_values = eout['values'][0]
    else:
        first_values = eout['values']

    if hasattr(first_values, 'shape'):
        nd = len(first_values.shape)
    elif hasattr(first_values, '__len__'):
        nd = 1
    else:
        nd = 0  # Single value

    # For affinity output, get balancing coefficients
    if eout_is_aout and type == "auto":
        n_balance, balance = _get_balance(eout, balance, messages)
    elif eout_is_aout and type == "saturation":
        # For saturation diagrams, use n_balance = 1 for all species (don't normalize by stoichiometry)
        if isinstance(eout['values'], dict):
            n_balance = [1] * len(eout['values'])
        elif isinstance(eout['values'], list):
            n_balance = [1] * len(eout['values'])
        else:
            n_balance = [1]
        if balance is None:
            balance = 1
    else:
        # For equilibrate output, use n_balance from equilibrate if available
        if 'n_balance' in eout:
            n_balance = eout['n_balance']
            balance = eout.get('balance', 1)
        else:
            if isinstance(eout['values'], dict):
                n_balance = [1] * len(eout['values'])
            elif isinstance(eout['values'], list):
                n_balance = [1] * len(eout['values'])
            else:
                n_balance = [1]
            if balance is None:
                balance = 1

    # Determine what to plot
    plotvals = {}
    plotvar = eout.get('property', 'A')

    # Calculate equilibrium log activity/fugacity of basis species
    if plot_loga_basis:
        # Find the index of the basis species
        basis_df = eout['basis']
        ibasis = list(basis_df.index).index(type)

        # Get the logarithm of activity used in the affinity calculation
        logact = basis_df.iloc[ibasis]['logact']

        # Check if logact is numeric
        try:
            loga_basis = float(logact)
        except (ValueError, TypeError):
            raise ValueError(f"the logarithm of activity for basis species {type} is not numeric - was a buffer selected?")

        # Get the reaction coefficients for this basis species
        # eout['species'] is a DataFrame with basis species as columns
        nu_basis = eout['species'].iloc[:, ibasis].values

        # Calculate the logarithm of activity where affinity = 0
        # loga_equilibrium = loga_basis - affinity / nu_basis
        plotvals = {}
        for i, (sp_idx, affinity_vals) in enumerate(eout['values'].items()):
            plotvals[sp_idx] = loga_basis - affinity_vals / nu_basis[i]

        plotvar = type

        # Set n_balance (not used for basis species plots, but needed for compatibility)
        n_balance = [1] * len(plotvals)
        if balance is None:
            balance = 1
    elif eout_is_aout:
        # Plot affinity values divided by balancing coefficients
        # DEBUG: Check balance application
        if False:  # Set to True for debugging
            print(f"\nDEBUG: Applying balance to affinity values")
            print(f"  n_balance: {n_balance}")

        # Handle dict-based values (from affinity)
        if isinstance(eout['values'], dict):
            for i, (species_idx, values) in enumerate(eout['values'].items()):
                if False:  # Set to True for debugging
                    print(f"  Species {i} (ispecies {species_idx}): values/n_balance[{i}]={n_balance[i]}")
                plotvals[species_idx] = values / n_balance[i]
        # Handle list-based values
        elif isinstance(eout['values'], list):
            for i, values in enumerate(eout['values']):
                species_idx = eout['species']['ispecies'].iloc[i]
                plotvals[species_idx] = values / n_balance[i]

        if plotvar == 'A':
            plotvar = 'A/(2.303RT)'
            if nd == 1:
                if messages:
                    print(f"diagram: plotting {plotvar} / n.balance")
    else:
        # Plot equilibrated activities
        # Check for both Python naming (loga_equil) and R naming (loga.equil)
        loga_equil_key = 'loga_equil' if 'loga_equil' in eout else 'loga.equil'
        loga_equil_list = eout[loga_equil_key]

        # For equilibrate output, keep plotvals as a dict with INTEGER indices as keys
        # This preserves the 1:1 correspondence with the species list, including duplicates
        # Do NOT use ispecies as keys because duplicates would overwrite each other
        if isinstance(loga_equil_list, list):
            for i, loga_val in enumerate(loga_equil_list):
                plotvals[i] = loga_val  # Use integer index, not ispecies
        else:
            # Already a dict
            plotvals = loga_equil_list

        plotvar = 'loga.equil'

    # Handle alpha (degree of formation)
    if alpha:
        # Convert to activities (remove logarithms)
        # Use numpy arrays for proper element-wise operations
        act_vals = {}
        for k, v in plotvals.items():
            if isinstance(v, np.ndarray):
                act_vals[k] = 10**v
            else:
                act_vals[k] = np.power(10, v)

        # Scale by balance if requested
        if alpha == "balance":
            species_keys = list(act_vals.keys())
            for i, k in enumerate(species_keys):
                act_vals[k] = act_vals[k] * n_balance[i]

        # Calculate sum of activities (element-wise for arrays)
        # Get the first value to determine shape
        first_val = list(act_vals.values())[0]
        if isinstance(first_val, np.ndarray):
            # Multi-dimensional case
            sum_act = np.zeros_like(first_val)
            for v in act_vals.values():
                sum_act = sum_act + v
        else:
            # Single value case
            sum_act = sum(act_vals.values())

        # Calculate alpha (fraction) - element-wise division
        plotvals = {k: v / sum_act for k, v in act_vals.items()}
        plotvar = "alpha"

    # Get species information for labels
    species_df = eout['species']
    if names is None:
        names = species_df['name'].tolist()

    # Format chemical names if requested
    if format_names and not alpha:
        names = [_format_chemname(name) for name in names]

    # Prepare for plotting
    if nd == 0:
        # 0-D: Bar plot (not implemented yet)
        raise NotImplementedError("0-D bar plots not yet implemented")

    elif nd == 1:
        # 1-D: Line plot
        result = _plot_1d(eout, plotvals, plotvar, names, n_balance, balance,
                       xlab, ylab, xlim, ylim, col, lty, lwd, main, add, plot_it, ax, width, height, plot_was_provided, **kwargs)

    elif nd == 2:
        # 2-D: Predominance diagram or saturation lines
        # Pass lty and cex through kwargs for saturation plots
        result = _plot_2d(eout, plotvals, plotvar, names, n_balance, balance,
                       xlab, ylab, xlim, ylim, col, col_names, fill, fill_NA, limit_water, lwd, main, add, plot_it, ax,
                       type, contour_method, messages, width, height, plot_was_provided, lty=lty, cex=cex, **kwargs)

    else:
        raise ValueError(f"Cannot create diagram with {nd} dimensions")

    # Handle Jupyter display behavior
    # When plot_it=True, we want the figure to display
    # When plot_it=False, we want to suppress display and close the figure
    if not plot_it and result is not None and 'fig' in result:
        # Close the figure to prevent auto-display in Jupyter
        # The figure is still in the result dict, so users can access it via result['fig']
        # but it won't be displayed automatically
        plt.close(result['fig'])
    elif plot_it and result is not None and 'fig' in result:
        # Try to use IPython display if available (for Jupyter notebooks)
        try:
            from IPython.display import display
            display(result['fig'])
        except (ImportError, NameError):
            # Not in IPython/Jupyter, regular matplotlib display
            pass

    return result


def _get_balance(eout: Dict[str, Any], balance: Optional[Union[str, float, List[float]]], messages: bool = True) -> Tuple[List[float], Union[str, int, List[float]]]:
    """
    Get balancing coefficients for formation reactions.

    This implements the R CHNOSZ balance() function logic for determining
    how to balance formation reactions when calculating diagrams.

    Parameters
    ----------
    eout : dict
        Output from affinity()
    balance : str, float, list of float, or None
        Balancing specification

    Returns
    -------
    tuple of (list of float, balance_name)
        - Balancing coefficients for each species
        - The balance identifier used
    """
    species_df = eout['species']
    basis_df = eout['basis']
    n_species = len(species_df)

    # Get basis species column names (exclude metadata columns)
    basis_cols = [col for col in species_df.columns
                 if col not in ['ispecies', 'name', 'state', 'logact']]

    if balance is None:
        # Auto-select using which_balance logic
        ibalance = _which_balance(species_df, basis_cols)
        if len(ibalance) == 0:
            raise ValueError("no basis species is present in all formation reactions")
        balance_col = basis_cols[ibalance[0]]
        n_balance = species_df[balance_col].tolist()
        if messages:
            print(f"balance: on moles of {balance_col} in formation reactions")
        balance = balance_col
    elif balance == 1 or balance == "1":
        # Balance on one mole of species (formula units)
        n_balance = [1] * n_species
        if messages:
            print("balance: on supplied numeric argument (1) [1 means balance on formula units]")
        balance = 1
    elif isinstance(balance, (int, float)):
        # Use a specific basis species by index
        if 0 < balance <= len(basis_cols):
            balance_col = basis_cols[int(balance) - 1]
            n_balance = species_df[balance_col].tolist()
            if messages:
                print(f"balance: on moles of {balance_col} in formation reactions")
            balance = balance_col
        else:
            warnings.warn(f"Balance index {balance} out of range, using 1")
            n_balance = [1] * n_species
            balance = 1
    elif isinstance(balance, str):
        # Use named basis species
        if balance in species_df.columns:
            n_balance = species_df[balance].tolist()
            if messages:
                print(f"balance: on moles of {balance} in formation reactions")
        else:
            warnings.warn(f"Balance species '{balance}' not found, using 1")
            n_balance = [1] * n_species
            balance = 1
    elif isinstance(balance, list):
        # Use provided list
        if len(balance) == n_species:
            n_balance = balance
            if messages:
                print(f"balance: on supplied numeric argument ({','.join(map(str, balance))})")
        else:
            warnings.warn(f"Balance list length ({len(balance)}) doesn't match species count ({n_species}), using 1")
            n_balance = [1] * n_species
            balance = 1
    else:
        n_balance = [1] * n_species
        balance = 1

    # Handle negative coefficients (make all positive if all negative)
    if all(x < 0 for x in n_balance):
        n_balance = [-x for x in n_balance]

    return n_balance, balance


def _which_balance(species_df: pd.DataFrame, basis_cols: List[str]) -> List[int]:
    """
    Find basis species present in all formation reactions.

    This implements R CHNOSZ which.balance() function.

    Parameters
    ----------
    species_df : pd.DataFrame
        Species dataframe with stoichiometric coefficients
    basis_cols : list of str
        Names of basis species columns

    Returns
    -------
    list of int
        Indices of basis species present in all reactions (0-indexed)
    """
    ib = []
    for i, col in enumerate(basis_cols):
        coeffs = species_df[col].values
        # Check if all coefficients are non-zero
        if np.all(coeffs != 0):
            ib.append(i)
    return ib


def _plot_1d(eout: Dict[str, Any],
             plotvals: Dict,
             plotvar: str,
             names: List[str],
             n_balance: List[float],
             balance: Optional[Union[str, float, List[float]]],
             xlab: Optional[str],
             ylab: Optional[str],
             xlim: Optional[List[float]],
             ylim: Optional[List[float]],
             col: Optional[Union[str, List[str]]],
             lty: Optional[Union[str, int, List]],
             lwd: Union[float, List[float]],
             main: Optional[str],
             add: bool,
             plot_it: bool,
             ax: Optional[Any],
             width: int = 600,
             height: int = 520,
             plot_was_provided: bool = False,
             **kwargs) -> Dict[str, Any]:
    """
    Create a 1-D line plot.

    Parameters
    ----------
    (See diagram() for parameter descriptions)

    Returns
    -------
    dict
        Output dictionary with plot data and metadata
    """

    # Get x-axis values
    xvar = eout['vars'][0]
    xvals = eout['vals'][xvar]

    # Convert to numpy array if needed
    if not isinstance(xvals, np.ndarray):
        xvals = np.array(xvals)

    # Set up axis labels
    if xlab is None:
        xlab = _axis_label(xvar, eout)

    if ylab is None:
        ylab = _axis_label(plotvar, eout)

    # Set up x limits
    if xlim is None:
        xlim = [xvals[0], xvals[-1]]

    # Set up colors and line styles
    n_species = len(plotvals)

    if col is None:
        # Use matplotlib default color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        col = [colors[i % len(colors)] for i in range(n_species)]
    elif isinstance(col, str):
        col = [col] * n_species
    else:
        col = list(col) * (n_species // len(col) + 1)
        col = col[:n_species]

    if lty is None:
        lty = ['-'] * n_species
    elif isinstance(lty, (str, int)):
        lty = [lty] * n_species
    else:
        lty = list(lty) * (n_species // len(lty) + 1)
        lty = lty[:n_species]

    if isinstance(lwd, (int, float)):
        lwd = [lwd] * n_species
    else:
        lwd = list(lwd) * (n_species // len(lwd) + 1)
        lwd = lwd[:n_species]

    # Convert numeric line styles to matplotlib styles
    lty_map = {1: '-', 2: '--', 3: '-.', 4: ':', 5: '-', 6: '--'}
    lty = [lty_map.get(lt, lt) if isinstance(lt, int) else lt for lt in lty]

    # Temporarily disable interactive mode if plot_it=False
    # This prevents Jupyter from auto-displaying the figure
    was_interactive = plt.isinteractive()
    if not plot_it:
        plt.ioff()

    # Convert width and height from pixels to inches for matplotlib
    # Use standard 96 DPI for consistency with web/screen displays
    dpi = 96
    figsize_inches = (width / dpi, height / dpi)

    # Create figure and axes (always, even if plot_it=False)
    # This allows the plot to be used with add_to parameter later
    fig = None
    ax_was_provided = ax is not None  # Track if ax was passed as parameter

    if ax is not None:
        # Use provided axes
        fig = ax.get_figure()
    elif not add:
        # Create new figure and axes with specified size
        fig, ax = plt.subplots(figsize=figsize_inches, dpi=dpi)
    else:
        # Try to get current axes, create new if none exists
        try:
            ax = plt.gca()
            fig = ax.get_figure()
        except:
            fig, ax = plt.subplots(figsize=figsize_inches, dpi=dpi)

    # Plot each species (always draw, regardless of plot_it)
    # plot_it only controls display, not drawing
    for i, (species_idx, yvals) in enumerate(plotvals.items()):
        # Convert to numpy array if needed
        if not isinstance(yvals, np.ndarray):
            yvals = np.array([yvals] * len(xvals))

        ax.plot(xvals, yvals,
               color=col[i],
               linestyle=lty[i],
               linewidth=lwd[i],
               label=names[i],
               **kwargs)

    # Set labels and limits
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    # Add legend
    ax.legend()

    # Add title
    if main is not None:
        ax.set_title(main)

    # Add grid
    ax.grid(True, alpha=0.3)

    if not add:
        plt.tight_layout()

    # Build output dictionary
    result = {
        **eout,
        'plotvar': plotvar,
        'plotvals': plotvals,
        'names': names,
        'predominant': np.nan,
        'balance': balance,
        'n.balance': n_balance
    }

    # Add figure and axes to output if they were created
    if fig is not None:
        if not ax_was_provided or plot_was_provided:
            result['ax'] = ax
            result['fig'] = fig

    # Always restore interactive mode to its original state
    if was_interactive and not plt.isinteractive():
        plt.ion()
    elif not was_interactive and plt.isinteractive():
        plt.ioff()

    return result


def _plot_2d(eout: Dict[str, Any],
            plotvals: Dict[int, np.ndarray],
            plotvar: str,
            names: List[str],
            n_balance: List[float],
            balance: Union[str, int, List[float]],
            xlab: Optional[str],
            ylab: Optional[str],
            xlim: Optional[List[float]],
            ylim: Optional[List[float]],
            col: Optional[Union[str, List[str]]],
            col_names: Optional[Union[str, List[str]]],
            fill: Optional[str],
            fill_NA: str,
            limit_water: Optional[bool],
            lwd: Union[float, List[float]],
            main: Optional[str],
            add: bool,
            plot_it: bool,
            ax: Optional[Any],
            type: str = "auto",
            contour_method: Optional[str] = "edge",
            messages: bool = True,
            width: int = 600,
            height: int = 520,
            plot_was_provided: bool = False,
            **kwargs) -> Dict[str, Any]:
    """
    Create a 2-D predominance diagram (internal function).

    This function determines which species has the maximum affinity at each
    point on a 2D grid and creates a predominance field diagram.

    Parameters
    ----------
    eout : dict
        Output from affinity()
    plotvals : dict
        Values to plot (affinity or equilibrium values)
    plotvar : str
        Variable being plotted
    names : list of str
        Species names for labels
    n_balance : list of float
        Balancing coefficients
    balance : str, int, or list
        Balance identifier
    xlab, ylab : str or None
        Axis labels
    xlim, ylim : list or None
        Axis limits
    col : str, list, or None
        Colors for boundary lines in 2D plots
    col_names : str, list, or None
        Colors for field labels (text) in 2D plots
    fill : str or None
        Matplotlib colormap name for coloring predominance fields
        (e.g., 'viridis', 'plasma', 'terrain', 'rainbow', 'Set1', 'tab10')
    lwd : float or list of float
        Line width for drawing boundaries between predominance fields.
        Set to 0 to disable borders.
    main : str or None
        Plot title
    add : bool
        Add to existing plot?
    plot_it : bool
        Display the plot?
    **kwargs
        Additional matplotlib arguments

    Returns
    -------
    dict
        Result dictionary with predominance information
    """

    # Get the two variables
    vars_list = eout['vars']
    if len(vars_list) != 2:
        raise ValueError(f"Expected 2 variables for 2-D plot, got {len(vars_list)}")

    # R CHNOSZ convention: first variable in affinity() → x-axis, second → y-axis
    # In the array: rows correspond to first var, columns to second var
    xvar = vars_list[0]  # First variable (rows in array) → x-axis
    yvar = vars_list[1]  # Second variable (cols in array) → y-axis

    # Get the values for each variable
    xvals = eout['vals'][xvar]
    yvals = eout['vals'][yvar]

    # Convert to numpy arrays if needed
    xvals = np.asarray(xvals)
    yvals = np.asarray(yvals)

    # Get axis labels
    if xlab is None:
        xlab = _axis_label(xvar, eout)
    if ylab is None:
        ylab = _axis_label(yvar, eout)

    # Handle saturation lines separately from predominance diagrams
    if type == "saturation":
        # Extract lty and cex from kwargs (they're in kwargs from diagram() call)
        lty_param = kwargs.pop('lty', None)
        cex_param = kwargs.pop('cex', 1.0)

        return _plot_saturation_2d(eout, plotvals, plotvar, names, n_balance, balance,
                                   xlab, ylab, xlim, ylim, col, lwd, lty_param, cex_param,
                                   main, add, plot_it, ax, contour_method, messages, width, height, plot_was_provided, **kwargs)

    # For non-saturation plots, remove lty and cex from kwargs to avoid passing to matplotlib
    kwargs.pop('lty', None)
    kwargs.pop('cex', None)

    # Print message about diagram method
    if messages:
        print(f"diagram: using maximum affinity method for 2-D diagram")

    # Stack all species values into a 3D array (species, rows, cols)
    # The plotvals dict can have two types of keys:
    # 1. Integer indices (0, 1, 2, ...) for equilibrate output - preserves duplicates
    # 2. ispecies values (1844, 1876, ...) for affinity output - unique species only
    species_keys = list(plotvals.keys())
    n_species = len(species_keys)

    # Check if species_keys are integer indices (equilibrate) or ispecies values (affinity)
    # Integer indices will be 0, 1, 2, ..., n-1 and directly map to names list
    # ispecies values are typically > 100 and need mapping
    species_df = eout['species']
    if all(isinstance(k, int) and k < len(names) for k in species_keys):
        # Integer indices: direct mapping to names
        predominant_to_names_idx = {i: species_keys[i] for i in range(n_species)}
    else:
        # ispecies values: need to find matching rows
        # Use first matching row (for consistency with affinity output)
        predominant_to_names_idx = {}
        for i, sp_idx in enumerate(species_keys):
            matching_rows = species_df[species_df['ispecies'] == sp_idx].index.tolist()
            if len(matching_rows) > 0:
                predominant_to_names_idx[i] = matching_rows[0]
            else:
                predominant_to_names_idx[i] = i  # Fallback

    # DEBUG: Print species_keys order
    # print(f"DEBUG: species_keys = {species_keys}")

    # Get dimensions from first species
    first_vals = plotvals[species_keys[0]]
    if len(first_vals.shape) != 2:
        raise ValueError(f"Expected 2-D array for each species, got shape {first_vals.shape}")

    # Array shape: (n_xvar, n_yvar) since first var → x-axis, second → y-axis
    # For our example: (n_T, n_P) = (3, 10)
    n_xvar, n_yvar = first_vals.shape

    # Stack all species affinities into a 3D array
    affinity_stack = np.zeros((n_species, n_xvar, n_yvar))
    for i, sp_idx in enumerate(species_keys):
        affinity_stack[i, :, :] = plotvals[sp_idx]

    # DEBUG: Check species order
    if False:  # Set to True for debugging
        print(f"\nDEBUG affinity_stack:")
        print(f"  n_species: {n_species}")
        print(f"  species_keys: {species_keys}")
        for i, sp_idx in enumerate(species_keys):
            print(f"  Stack position {i}: species {sp_idx}")

    # Find the species with maximum affinity at each point
    # predominant will have indices 0, 1, 2, ... for species
    predominant_indices = np.argmax(affinity_stack, axis=0)

    # Get the affinity values at predominant points
    predominant_values = np.max(affinity_stack, axis=0)

    # Convert indices to species indices (1-based for R compatibility)
    # In R, predominant contains 1, 2, 3, etc.
    predominant = predominant_indices + 1

    # Calculate water stability limits if requested (default True for 2-D diagrams)
    H2O_predominant = None
    if limit_water is None:
        limit_water = not add  # True unless adding to existing plot

    if limit_water:
        # Call water_lines with plot_it=False to get boundaries
        wl = water_lines(eout, plot_it=False, messages=messages)
        # Check if water_lines produced valid results
        if not (wl['y_oxidation'] is None or wl['y_reduction'] is None):
            # Create a copy of predominant matrix for water stability masking
            # Convert to float to allow NaN values
            H2O_predominant = predominant.astype(float).copy()

            # For each x-point, find y-values outside water stability limits
            for i in range(len(wl['xpoints'])):
                ymin = min(wl['y_oxidation'][i], wl['y_reduction'][i])
                ymax = max(wl['y_oxidation'][i], wl['y_reduction'][i])

                if not wl['swapped']:
                    # Normal orientation: x is first var, y is second var
                    # eout['vals'][yvar] contains the y-axis values
                    yvals = np.asarray(eout['vals'][yvar])
                    # Find indices where y is outside stability range
                    iNA = (yvals < ymin) | (yvals > ymax)
                    # Mark those regions as NA (using nan)
                    H2O_predominant[i, iNA] = np.nan
                else:
                    # Swapped: first var is y-axis
                    xvals = np.asarray(eout['vals'][xvar])
                    iNA = (xvals < ymin) | (xvals > ymax)
                    H2O_predominant[iNA, i] = np.nan

    # For plotting: transpose arrays so that x-axis is horizontal and y-axis is vertical
    # imshow expects (n_rows, n_cols) = (n_yaxis, n_xaxis)
    # Current shape is (n_xvar, n_yvar), so transpose to (n_yvar, n_xvar)
    predominant_indices_T = predominant_indices.T
    if H2O_predominant is not None:
        H2O_predominant_T = H2O_predominant.T
    else:
        H2O_predominant_T = None

    # Temporarily disable interactive mode if plot_it=False
    # This prevents Jupyter from auto-displaying the figure
    was_interactive = plt.isinteractive()
    if not plot_it:
        plt.ioff()

    # Convert width and height from pixels to inches for matplotlib
    # Use standard 96 DPI for consistency with web/screen displays
    dpi = 96
    figsize_inches = (width / dpi, height / dpi)

    # Create figure and axes (always, even if plot_it=False)
    # This allows the plot to be used with add_to parameter later
    fig = None
    ax_was_provided = ax is not None  # Track if ax was passed as parameter

    if ax is not None:
        # Use provided axes
        fig = ax.get_figure()
    elif not add:
        # Create new figure and axes with specified size
        fig, ax = plt.subplots(figsize=figsize_inches, dpi=dpi)
    else:
        # Try to get current axes, create new if none exists
        try:
            ax = plt.gca()
            fig = ax.get_figure()
        except:
            fig, ax = plt.subplots(figsize=figsize_inches, dpi=dpi)

    # When add=True, don't draw fill by default (to overlay on existing plot)
    # User can explicitly provide fill parameter to override this
    draw_fill = True
    if add and fill is None:
        draw_fill = False

    # Draw the plot content (always, regardless of plot_it)
    # plot_it only controls display, not drawing
    # Determine fill colors for predominance fields
    # Priority: fill parameter > default
    # R CHNOSZ default: white fill with black borders
    if fill is not None:
        # Use a matplotlib colormap
        try:
            import matplotlib.cm as cm
            cmap = cm.get_cmap(fill)
            # Sample the colormap evenly across species
            # Use range 0.1 to 0.9 to avoid very light/dark ends
            color_indices = np.linspace(0.1, 0.9, n_species)
            fill_colors = [cmap(idx) for idx in color_indices]
        except (ValueError, KeyError):
            warnings.warn(f"Colormap '{fill}' not found, using default colors")
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']
            fill_colors = [colors[i % len(colors)] for i in range(n_species)]
    elif fill is None and lwd > 0:
        # R CHNOSZ default behavior: white fill with black borders
        fill_colors = ['white'] * n_species
    else:
        # Use default matplotlib colors
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        fill_colors = [colors[i % len(colors)] for i in range(n_species)]

    # Determine boundary line colors (col parameter)
    if col is None:
        # Default to black for boundary lines
        boundary_colors = ['black'] * n_species
    elif isinstance(col, str):
        boundary_colors = [col] * n_species
    else:
        boundary_colors = list(col)
        if len(boundary_colors) < n_species:
            # Repeat colors if not enough provided
            boundary_colors = boundary_colors * (n_species // len(boundary_colors) + 1)
        boundary_colors = boundary_colors[:n_species]

    # Determine text label colors (col_names parameter)
    if col_names is None:
        # Default to black for text labels
        text_colors = ['black'] * n_species
    elif isinstance(col_names, str):
        text_colors = [col_names] * n_species
    else:
        text_colors = list(col_names)
        if len(text_colors) < n_species:
            # Repeat colors if not enough provided
            text_colors = text_colors * (n_species // len(text_colors) + 1)
        text_colors = text_colors[:n_species]

    # NOTE: Water instability shading is NOT drawn here in diagram()
    # It is only drawn when water_lines() is explicitly called
    # We keep H2O_predominant for the limit_water feature, but don't show gray shading

    # Draw filled predominance fields only if not adding to existing plot
    # (or if user explicitly provided fill parameter)
    if draw_fill:
        # Create a colored map showing predominance fields
        # Map each predominance index to its color
        # Shape is (n_yvar, n_xvar) after transpose
        colored_predominant = np.zeros((n_yvar, n_xvar, 3))  # RGB image
        from matplotlib.colors import to_rgb

        for i in range(n_species):
            mask = predominant_indices_T == i
            rgb = to_rgb(fill_colors[i])
            colored_predominant[mask] = rgb

        # Display the predominance map
        # imshow plots: rows → y-axis (vertical), cols → x-axis (horizontal)
        # extent sets the data coordinates: [x_start, x_end, y_start, y_end]
        # IMPORTANT: Use original order (xvals[0] to xvals[-1]), not min/max
        # This preserves axes with decreasing values (e.g., F- from -2 to -9)
        extent = [xvals[0], xvals[-1], yvals[0], yvals[-1]]

        # Expand extent if any dimension has identical limits to avoid matplotlib warning
        if extent[0] == extent[1]:
            x_range = abs(extent[0]) * 0.1 if extent[0] != 0 else 0.1
            extent[0] -= x_range
            extent[1] += x_range
        if extent[2] == extent[3]:
            y_range = abs(extent[2]) * 0.1 if extent[2] != 0 else 0.1
            extent[2] -= y_range
            extent[3] += y_range

        im = ax.imshow(colored_predominant, aspect='auto', origin='lower',
                      extent=extent, interpolation='nearest', **kwargs)

    # Add species labels at the center of their predominance regions
    for i in range(n_species):
        # Find all points where this species predominates (in transposed array)
        mask = predominant_indices_T == i
        if np.any(mask):
            # Get row and column indices from transposed array
            # rows correspond to y-axis values, cols to x-axis values
            rows, cols = np.where(mask)
            # Calculate mean position
            mean_row = rows.mean()
            mean_col = cols.mean()
            # Convert to data coordinates
            # cols index into xvals, rows index into yvals
            x_pos = xvals[int(mean_col)]
            y_pos = yvals[int(mean_row)]

            # Map predominant index to correct names index
            # This handles cases where duplicate species exist in the species list
            names_idx = predominant_to_names_idx[i]

            # Add text label with color from col_names
            ax.text(x_pos, y_pos, names[names_idx],
                   ha='center', va='center',
                   color=text_colors[i],
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.7),
                   fontsize=10, fontweight='bold')

    # Set labels and limits
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        # Preserve original axis direction (important for decreasing axes)
        x_min, x_max = xvals[0], xvals[-1]
        # Expand range if limits are identical to avoid matplotlib warning
        if x_min == x_max:
            x_range = abs(x_max) * 0.1 if x_max != 0 else 0.1
            x_min -= x_range
            x_max += x_range
        ax.set_xlim([x_min, x_max])

    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        # Preserve original axis direction (important for decreasing axes)
        y_min, y_max = yvals[0], yvals[-1]
        # Expand range if limits are identical to avoid matplotlib warning
        if y_min == y_max:
            y_range = abs(y_max) * 0.1 if y_max != 0 else 0.1
            y_min -= y_range
            y_max += y_range
        ax.set_ylim([y_min, y_max])

    # Add title
    if main is not None:
        ax.set_title(main)

    # Draw borders between predominance fields
    if lwd > 0:
        # Use matplotlib's contour function to draw boundaries between species
        # This matches R CHNOSZ's use of contourLines()
        # Following R CHNOSZ approach: loop over species and draw contours at level 0.5

        # Get unique species values (excluding any that don't appear)
        unique_species = np.unique(predominant_indices_T)
        unique_species = unique_species[~np.isnan(unique_species)]

        # Create meshgrid for contour (matches actual data coordinates)
        X, Y = np.meshgrid(xvals, yvals)

        # Loop over species (except the last one to avoid double-plotting)
        for i in range(len(unique_species) - 1):
            species_idx = int(unique_species[i])

            # Create a binary mask: 1 where this species predominates, 0 elsewhere
            z = (predominant_indices_T == species_idx).astype(float)

            # Draw contour at level 0.5 (boundary between 0 and 1)
            # This creates smooth boundaries following the actual grid
            try:
                line_color = boundary_colors[species_idx]
                cs = ax.contour(X, Y, z, levels=[0.5], colors=[line_color],
                               linewidths=lwd, zorder=10)
            except:
                pass  # Skip if contour can't be drawn (e.g., species doesn't appear)

    if not add:
        plt.tight_layout()

    # Don't close the figure when plot_it=False
    # The plt.ioff() above already prevents auto-display in Jupyter
    # This keeps the figure available for adding titles, legends, and later display
    # Users can display with: d['fig'].show() or display(d['fig']) in Jupyter

    # Build output dictionary (matching R CHNOSZ structure)
    result = {
        **eout,
        'plotvar': plotvar,
        'plotvals': plotvals,
        'names': names,
        'predominant': predominant,
        'predominant.values': predominant_values,
        'balance': balance,
        'n.balance': n_balance
    }

    # Add figure and axes to output if they were created
    if fig is not None:
        if not ax_was_provided or plot_was_provided:
            result['ax'] = ax
            result['fig'] = fig

    # Always restore interactive mode to its original state
    if was_interactive and not plt.isinteractive():
        plt.ion()
    elif not was_interactive and plt.isinteractive():
        plt.ioff()

    return result


def _plot_saturation_2d(eout: Dict[str, Any],
                        plotvals: Dict[int, np.ndarray],
                        plotvar: str,
                        names: List[str],
                        n_balance: List[float],
                        balance: Union[str, int, List[float]],
                        xlab: str,
                        ylab: str,
                        xlim: Optional[List[float]],
                        ylim: Optional[List[float]],
                        col: Optional[Union[str, List[str]]],
                        lwd: Union[float, List[float]],
                        lty: Optional[Union[str, int, List]],
                        cex: Union[float, List[float]],
                        main: Optional[str],
                        add: bool,
                        plot_it: bool,
                        ax: Optional[Any],
                        contour_method: Optional[Union[str, List[str]]],
                        messages: bool = True,
                        width: int = 600,
                        height: int = 520,
                        plot_was_provided: bool = False,
                        **kwargs) -> Dict[str, Any]:
    """
    Plot saturation lines (affinity=0 contours) for 2-D diagrams.

    This function draws contour lines where affinity = 0 for each species,
    indicating saturation boundaries (e.g., mineral precipitation thresholds).

    Parameters
    ----------
    (Most parameters are the same as _plot_2d)
    contour_method : str, list of str, or None
        Method for labeling contour lines. Can be a single value (applied to all species)
        or a list (one per species). If None, NA, or "", disable labels.
        Matplotlib doesn't support the same contour methods as R, so this mainly
        controls whether labels are drawn (any non-None/non-empty value enables labels).

    Returns
    -------
    dict
        Result dictionary
    """

    # Get the two variables
    vars_list = eout['vars']
    xvar = vars_list[0]
    yvar = vars_list[1]

    # Get the values for each variable
    xvals = np.asarray(eout['vals'][xvar])
    yvals = np.asarray(eout['vals'][yvar])

    species_keys = list(plotvals.keys())
    n_species = len(species_keys)

    if messages:
        print(f"diagram: plotting saturation lines for 2-D diagram")

    # Set up colors and line styles
    if col is None:
        # Use matplotlib default color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        col = [colors[i % len(colors)] for i in range(n_species)]
    elif isinstance(col, str):
        col = [col] * n_species
    else:
        col = list(col)
        if len(col) < n_species:
            col = col * (n_species // len(col) + 1)
        col = col[:n_species]

    if isinstance(lwd, (int, float)):
        lwd = [lwd] * n_species
    else:
        lwd = list(lwd)
        if len(lwd) < n_species:
            lwd = lwd * (n_species // len(lwd) + 1)
        lwd = lwd[:n_species]

    # Handle line styles (lty)
    if lty is None:
        lty = ['-'] * n_species
    elif isinstance(lty, (str, int)):
        lty = [lty] * n_species
    else:
        lty = list(lty)
        if len(lty) < n_species:
            lty = lty * (n_species // len(lty) + 1)
        lty = lty[:n_species]

    # Convert numeric line styles to matplotlib styles
    lty_map = {1: '-', 2: '--', 3: '-.', 4: ':', 5: '-', 6: '--'}
    lty = [lty_map.get(lt, lt) if isinstance(lt, int) else lt for lt in lty]

    # Handle text size (cex) for contour labels
    if isinstance(cex, (int, float)):
        cex_list = [cex] * n_species
    else:
        cex_list = list(cex)
        if len(cex_list) < n_species:
            cex_list = cex_list * (n_species // len(cex_list) + 1)
        cex_list = cex_list[:n_species]

    # Determine if labels should be drawn (per species)
    # Convert contour_method to a list (one per species)
    if contour_method is None or contour_method == "" or (isinstance(contour_method, str) and contour_method.upper() == "NA"):
        # No labels for any species
        drawlabels = [False] * n_species
    elif isinstance(contour_method, str):
        # Same method for all species
        drawlabels = [True] * n_species
    elif isinstance(contour_method, list):
        # Per-species methods
        if len(contour_method) != n_species:
            # Repeat/extend to match number of species
            contour_method_extended = list(contour_method) * (n_species // len(contour_method) + 1)
            contour_method_extended = contour_method_extended[:n_species]
        else:
            contour_method_extended = contour_method

        # Check each method to determine if labels should be drawn
        drawlabels = []
        for method in contour_method_extended:
            if method is None or method == "" or (isinstance(method, str) and method.upper() == "NA"):
                drawlabels.append(False)
            else:
                drawlabels.append(True)
    else:
        drawlabels = [True] * n_species

    # Temporarily disable interactive mode if plot_it=False
    # This prevents Jupyter from auto-displaying the figure
    was_interactive = plt.isinteractive()
    if not plot_it:
        plt.ioff()

    # Convert width and height from pixels to inches for matplotlib
    # Use standard 96 DPI for consistency with web/screen displays
    dpi = 96
    figsize_inches = (width / dpi, height / dpi)

    # Create figure and axes (always, even if plot_it=False)
    # This allows the plot to be used with add_to parameter later
    fig = None
    ax_was_provided = ax is not None  # Track if ax was passed as parameter

    if ax is not None:
        fig = ax.get_figure()
    elif not add:
        fig, ax = plt.subplots(figsize=figsize_inches, dpi=dpi)
    else:
        try:
            ax = plt.gca()
            fig = ax.get_figure()
        except:
            fig, ax = plt.subplots(figsize=figsize_inches, dpi=dpi)

    # Only do the actual plotting if plot_it=True
    # Draw the plot content (always, regardless of plot_it)
    # plot_it only controls display, not drawing
    if not add:
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim([xvals.min(), xvals.max()])

        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            ax.set_ylim([yvals.min(), yvals.max()])

        if main is not None:
            ax.set_title(main)

    # Plot saturation lines (affinity = 0 contours) for each species
    for i, sp_idx in enumerate(species_keys):
        zs = plotvals[sp_idx]

        # Skip plotting if this species has no possible saturation line
        if len(np.unique(zs)) == 1:
            if messages:
                print(f"diagram: no saturation line possible for {names[i]}")
            continue

        # Skip if line is outside the plot range
        if np.all(zs < 0) or np.all(zs > 0):
            if messages:
                print(f"diagram: beyond range for saturation line of {names[i]}")
            continue

        # Draw the contour line at affinity = 0
        # matplotlib's contour needs (X, Y, Z) where X and Y are meshgrids
        X, Y = np.meshgrid(xvals, yvals)
        # Transpose zs to match meshgrid orientation
        # plotvals has shape (n_xvar, n_yvar), but contour expects (n_yvar, n_xvar)
        zs_T = zs.T

        try:
            # Calculate font size from cex (matplotlib default is ~10pt)
            fontsize = 9 * cex_list[i]

            if drawlabels[i]:
                CS = ax.contour(X, Y, zs_T, levels=[0], colors=col[i],
                               linewidths=lwd[i], linestyles=lty[i], **kwargs)
                ax.clabel(CS, inline=True, fontsize=fontsize, fmt=names[i])
            else:
                ax.contour(X, Y, zs_T, levels=[0], colors=col[i],
                          linewidths=lwd[i], linestyles=lty[i], **kwargs)
        except Exception as e:
            warnings.warn(f"Could not plot contour for {names[i]}: {e}")

    if not add:
        plt.tight_layout()

    # Build output dictionary
    result = {
        **eout,
        'plotvar': plotvar,
        'plotvals': plotvals,
        'names': names,
        'predominant': np.nan,
        'balance': balance,
        'n.balance': n_balance
    }

    # Add figure and axes to output if they were created
    if fig is not None:
        if not ax_was_provided or plot_was_provided:
            result['ax'] = ax
            result['fig'] = fig

    # Always restore interactive mode to its original state
    if was_interactive and not plt.isinteractive():
        plt.ion()
    elif not was_interactive and plt.isinteractive():
        plt.ioff()

    return result


def _axis_label(var: str, eout: Dict[str, Any]) -> str:
    """
    Generate axis label for a variable.

    Parameters
    ----------
    var : str
        Variable name
    eout : dict
        Output from affinity()

    Returns
    -------
    str
        Formatted axis label
    """

    # Special cases
    if var == 'A/(2.303RT)':
        return r'A/(2.303RT)'
    elif var == 'alpha':
        return r'$\alpha$'
    elif var == 'loga.equil':
        return r'log activity'
    elif var == 'pH':
        return 'pH'
    elif var == 'pe':
        return 'pe'
    elif var == 'Eh':
        return 'Eh (V)'
    elif var == 'T':
        return 'Temperature (°C)'
    elif var == 'P':
        return 'Pressure (bar)'
    elif var == 'IS':
        return 'Ionic strength'
    else:
        # Check if it's a basis species
        basis_df = eout.get('basis')
        if basis_df is not None and var in basis_df.index:
            state = basis_df.loc[var, 'state']
            # Format the chemical formula with proper LaTeX subscripts and superscripts
            var_formatted = _format_species_latex(var)
            if state in ['aq', 'liq', 'cr']:
                return f'$\\log\\ a_{{{var_formatted}}}$'
            else:
                return f'$\\log\\ f_{{{var_formatted}}}$'
        return var


def _format_chemname(name: str) -> str:
    """
    Format a chemical formula for display in matplotlib.

    Uses LaTeX formatting for proper subscripts and superscripts.
    Delegates to _format_species_latex from expression.py for consistency.

    Parameters
    ----------
    name : str
        Chemical formula

    Returns
    -------
    str
        Formatted formula (using LaTeX for matplotlib)
    """
    # Use the centralized formatting function and wrap in math mode for matplotlib
    latex_formula = _format_species_latex(name)
    return f'${latex_formula}$'


def water_lines(eout: Dict[str, Any],
                which: Union[str, List[str]] = ['oxidation', 'reduction'],
                lty: Union[int, str] = 2,
                lwd: float = 1,
                col: Optional[str] = None,
                plot_it: bool = True,
                messages: bool = True) -> Dict[str, Any]:
    """
    Draw water stability limits for Eh-pH, logfO2-pH, logfO2-T or Eh-T diagrams.

    This function adds lines showing the oxidation and reduction stability limits
    of water to diagrams. Above the oxidation line, water breaks down to O2.
    Below the reduction line, water breaks down to H2.

    Parameters
    ----------
    eout : dict
        Output from affinity(), equilibrate(), or diagram()
    which : str or list of str, default ['oxidation', 'reduction']
        Which line(s) to draw: 'oxidation', 'reduction', or both
    lty : int or str, default 2
        Line style (matplotlib linestyle or numeric code)
    lwd : float, default 1
        Line width
    col : str, optional
        Line color (matplotlib color spec). If None, uses current foreground color
    plot_it : bool, default True
        Whether to plot the lines and display the figure. When True, the lines
        are added to the diagram and the figure is displayed (useful when the
        original diagram was created with plot_it=False). When False, only
        calculates and returns the water line coordinates without plotting.

    Returns
    -------
    dict
        Dictionary containing all keys from the input diagram (including 'fig', 'ax',
        'plotvar', 'plotvals', 'names', 'predominant', etc. if present) plus the
        following water line specific keys:
        - xpoints: x-axis values
        - y_oxidation: y values for oxidation line (or None)
        - y_reduction: y values for reduction line (or None)
        - swapped: whether axes were swapped

    Examples
    --------
    >>> # Add water lines to an existing displayed diagram
    >>> basis(["Fe+2", "SO4-2", "H2O", "H+", "e-"], [0, math.log10(3), math.log10(0.75), 999, 999])
    >>> species(["rhomboclase", "ferricopiapite", "hydronium jarosite", "goethite", "melanterite", "pyrite"])
    >>> a = affinity(pH=[-1, 4, 256], pe=[-5, 23, 256])
    >>> d = diagram(a, main="Fe-S-O-H, after Majzlan et al., 2006")
    >>> water_lines(d, lwd=2)

    >>> # Add water lines and display when diagram was created with plot_it=False
    >>> d = diagram(a, main="Fe-S-O-H", plot_it=False)
    >>> water_lines(d, lwd=2)  # This will display the figure with water lines

    Notes
    -----
    This function only works on diagrams with a redox variable (Eh, pe, O2, or H2)
    on one axis and pH, T, P, or another non-redox variable on the other axis.
    For 1-D diagrams, vertical lines are drawn.
    """

    # Import here to avoid circular imports
    from ..utils.units import convert, envert
    from ..core.subcrt import subcrt

    # Create a deep copy of the input to preserve all diagram information
    # This allows us to return all the original keys plus water line data
    result = copy_plot(eout)

    # Detect if this is a Plotly figure (interactive diagram)
    is_plotly = False
    if 'fig' in result and result['fig'] is not None:
        is_plotly = hasattr(result['fig'], 'add_trace') and hasattr(result['fig'], 'update_layout')

    # Ensure which is a list
    if isinstance(which, str):
        which = [which]

    # Get number of variables used in affinity()
    nvar1 = len(result['vars'])

    # Determine actual number of variables from array dimensions
    # Check both loga.equil (equilibrate output) and values (affinity output)
    if 'loga_equil' in result or 'loga.equil' in result:
        loga_key = 'loga_equil' if 'loga_equil' in result else 'loga.equil'
        first_val = result[loga_key][0] if isinstance(result[loga_key], list) else list(result[loga_key].values())[0]
    else:
        first_val = list(result['values'].values())[0] if isinstance(result['values'], dict) else result['values'][0]

    if hasattr(first_val, 'shape'):
        dim = first_val.shape
    elif hasattr(first_val, '__len__'):
        dim = (len(first_val),)
    else:
        dim = ()

    nvar2 = len(dim)

    # We only work on diagrams with 1 or 2 variables
    if nvar1 not in [1, 2] or nvar2 not in [1, 2]:
        result.update({'xpoints': None, 'y_oxidation': None, 'y_reduction': None, 'swapped': False})
        return result

    # Get variables from result
    vars_list = result['vars'].copy()

    # If needed, swap axes so redox variable is on y-axis
    # Also do this for 1-D diagrams
    if len(vars_list) == 1:
        vars_list.append('nothing')

    swapped = False
    if vars_list[1] in ['T', 'P', 'nothing']:
        vars_list = list(reversed(vars_list))
        vals_dict = {vars_list[0]: result['vals'][vars_list[0]]} if vars_list[0] != 'nothing' else {}
        if len(result['vars']) > 1:
            vals_dict[vars_list[1]] = result['vals'][vars_list[1]]
        swapped = True
    else:
        vals_dict = result['vals']

    xaxis = vars_list[0]
    yaxis = vars_list[1]
    xpoints = np.asarray(vals_dict[xaxis]) if xaxis in vals_dict else np.array([0])

    # Make xaxis "nothing" if it is not pH, T, or P
    # (so that horizontal water lines can be drawn for any non-redox variable on the x-axis)
    if xaxis not in ['pH', 'T', 'P']:
        xaxis = 'nothing'

    # T and P are constants unless they are plotted on one of the axes
    T = result['T']
    if vars_list[0] == 'T':
        T = envert(xpoints, 'K')
    P = result['P']
    if vars_list[0] == 'P':
        P = envert(xpoints, 'bar')

    # Handle the case where P is "Psat" - keep it as is for subcrt
    # (subcrt knows how to handle "Psat")

    # logaH2O is 0 unless given in result['basis']
    basis_df = result['basis']
    if 'H2O' in basis_df.index:
        logaH2O = float(basis_df.loc['H2O', 'logact'])
    else:
        logaH2O = 0

    # pH is 7 unless given in eout['basis'] or plotted on one of the axes
    if vars_list[0] == 'pH':
        pH = xpoints
    elif 'H+' in basis_df.index:
        minuspH = basis_df.loc['H+', 'logact']
        # Special treatment for non-numeric value (happens when a buffer is used)
        try:
            pH = -float(minuspH)
        except (ValueError, TypeError):
            pH = np.nan
    else:
        pH = 7

    # O2 state is gas unless given in eout['basis']
    O2state = 'gas'
    if 'O2' in basis_df.index:
        O2state = basis_df.loc['O2', 'state']

    # H2 state is gas unless given in eout['basis']
    H2state = 'gas'
    if 'H2' in basis_df.index:
        H2state = basis_df.loc['H2', 'state']

    # Where the calculated values will go
    y_oxidation = None
    y_reduction = None

    if xaxis in ['pH', 'T', 'P', 'nothing'] and yaxis in ['Eh', 'pe', 'O2', 'H2']:
        # Eh/pe/logfO2/logaO2/logfH2/logaH2 vs pH/T/P

        # Reduction line (H2O + e- = 1/2 H2 + OH-)
        if 'reduction' in which:
            logfH2 = logaH2O  # usually 0

            if yaxis == 'H2':
                # Calculate equilibrium constant for gas-aqueous conversion if needed
                logK = subcrt(['H2', 'H2'], [-1, 1], ['gas', H2state], T=T, P=P, convert=False, messages=messages, show=False).out['logK']
                # This is logfH2 if H2state == "gas", or logaH2 if H2state == "aq"
                logfH2 = logfH2 + logK
                # Broadcast to match xpoints length
                if isinstance(logfH2, (int, float)):
                    y_reduction = np.full_like(xpoints, logfH2)
                else:
                    logfH2_val = float(logfH2.iloc[0]) if hasattr(logfH2, 'iloc') else float(logfH2[0])
                    y_reduction = np.full_like(xpoints, logfH2_val)
            else:
                # Calculate logfO2 from H2O = 1/2 O2 + H2
                logK = subcrt(['H2O', 'O2', 'H2'], [-1, 0.5, 1], ['liq', O2state, 'gas'], T=T, P=P, convert=False, messages=messages, show=False).out['logK']
                # This is logfO2 if O2state == "gas", or logaO2 if O2state == "aq"
                logfO2 = 2 * (logK - logfH2 + logaH2O)

                if yaxis == 'O2':
                    # Broadcast to match xpoints length
                    if isinstance(logfO2, (int, float)):
                        y_reduction = np.full_like(xpoints, logfO2)
                    else:
                        logfO2_val = float(logfO2.iloc[0]) if hasattr(logfO2, 'iloc') else float(logfO2[0])
                        y_reduction = np.full_like(xpoints, logfO2_val)
                elif yaxis == 'Eh':
                    y_reduction = convert(logfO2, 'E0', T=T, P=P, pH=pH, logaH2O=logaH2O, messages=messages)
                elif yaxis == 'pe':
                    Eh_val = convert(logfO2, 'E0', T=T, P=P, pH=pH, logaH2O=logaH2O, messages=messages)
                    y_reduction = convert(Eh_val, 'pe', T=T, messages=messages)

        # Oxidation line (H2O = 1/2 O2 + 2H+ + 2e-)
        if 'oxidation' in which:
            logfO2 = logaH2O  # usually 0

            if yaxis == 'H2':
                # Calculate logfH2 from H2O = 1/2 O2 + H2
                logK = subcrt(['H2O', 'O2', 'H2'], [-1, 0.5, 1], ['liq', 'gas', H2state], T=T, P=P, convert=False, messages=messages, show=False).out['logK']
                # This is logfH2 if H2state == "gas", or logaH2 if H2state == "aq"
                logfH2 = logK - 0.5*logfO2 + logaH2O
                # Broadcast to match xpoints length
                if isinstance(logfH2, (int, float)):
                    y_oxidation = np.full_like(xpoints, logfH2)
                else:
                    logfH2_val = float(logfH2.iloc[0]) if hasattr(logfH2, 'iloc') else float(logfH2[0])
                    y_oxidation = np.full_like(xpoints, logfH2_val)
            else:
                # Calculate equilibrium constant for gas-aqueous conversion if needed
                logK = subcrt(['O2', 'O2'], [-1, 1], ['gas', O2state], T=T, P=P, convert=False, messages=messages, show=False).out['logK']
                # This is logfO2 if O2state == "gas", or logaO2 if O2state == "aq"
                logfO2 = logfO2 + logK

                if yaxis == 'O2':
                    # Broadcast to match xpoints length
                    if isinstance(logfO2, (int, float)):
                        y_oxidation = np.full_like(xpoints, logfO2)
                    else:
                        logfO2_val = float(logfO2.iloc[0]) if hasattr(logfO2, 'iloc') else float(logfO2[0])
                        y_oxidation = np.full_like(xpoints, logfO2_val)
                elif yaxis == 'Eh':
                    y_oxidation = convert(logfO2, 'E0', T=T, P=P, pH=pH, logaH2O=logaH2O, messages=messages)
                elif yaxis == 'pe':
                    Eh_val = convert(logfO2, 'E0', T=T, P=P, pH=pH, logaH2O=logaH2O, messages=messages)
                    y_oxidation = convert(Eh_val, 'pe', T=T, messages=messages)

    else:
        # Invalid axis combination
        result.update({'xpoints': xpoints, 'y_oxidation': None, 'y_reduction': None, 'swapped': swapped})
        return result

    # Route to Plotly or matplotlib implementation
    if is_plotly:
        return _water_lines_plotly(result, xpoints, y_oxidation, y_reduction, swapped,
                                  lty, lwd, col, plot_it)

    # Matplotlib implementation
    # Only draw water lines if eout already has an axes (meaning it's from a diagram)
    # If no axes, this is being called just for calculation (e.g., from within diagram())
    if 'ax' not in eout or eout['ax'] is None:
        # No axes to plot on - just return the calculated values
        result.update({'xpoints': xpoints, 'y_oxidation': y_oxidation, 'y_reduction': y_reduction, 'swapped': swapped})
        return result

    # Use the axes from result
    ax = result['ax']

    # First, shade the water-unstable regions with gray
    # This creates the same effect as R's fill.NA for H2O.predominant
    if y_oxidation is not None and y_reduction is not None:
        from matplotlib.colors import ListedColormap

        # Get current axis limits to create shading
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Create a high-resolution mesh for smooth shading
        n_points = 500
        if swapped:
            # When swapped, xpoints is on the y-axis
            y_mesh = np.linspace(ylim[0], ylim[1], n_points)
            x_mesh = np.linspace(xlim[0], xlim[1], n_points)
            X, Y = np.meshgrid(x_mesh, y_mesh)

            # For each y-value, determine if it's in water-unstable region
            # Interpolate oxidation and reduction values to the mesh
            y_ox_interp = np.interp(y_mesh, xpoints, y_oxidation)
            y_red_interp = np.interp(y_mesh, xpoints, y_reduction)

            # Create mask: unstable where x < min or x > max
            unstable = np.zeros_like(X, dtype=bool)
            for i in range(n_points):
                ymin = min(y_ox_interp[i], y_red_interp[i])
                ymax = max(y_ox_interp[i], y_red_interp[i])
                unstable[i, :] = (X[i, :] < ymin) | (X[i, :] > ymax)
        else:
            # Normal: xpoints on x-axis, y values on y-axis
            x_mesh = np.linspace(xlim[0], xlim[1], n_points)
            y_mesh = np.linspace(ylim[0], ylim[1], n_points)
            X, Y = np.meshgrid(x_mesh, y_mesh)

            # Interpolate oxidation and reduction values to the mesh
            y_ox_interp = np.interp(x_mesh, xpoints, y_oxidation)
            y_red_interp = np.interp(x_mesh, xpoints, y_reduction)

            # Create mask: unstable where y < min or y > max
            unstable = np.zeros_like(Y, dtype=bool)
            for i in range(n_points):
                ymin = min(y_ox_interp[i], y_red_interp[i])
                ymax = max(y_ox_interp[i], y_red_interp[i])
                unstable[:, i] = (Y[:, i] < ymin) | (Y[:, i] > ymax)

        # Create masked array for unstable regions
        import numpy.ma as ma
        unstable_mask = ma.masked_where(~unstable, np.ones_like(X))

        # Draw the shading with gray (matching R's gray80 = 0.8)
        fill_na_cmap = ListedColormap(['0.8'])
        extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
        ax.imshow(unstable_mask, aspect='auto', origin='lower',
                 extent=extent, interpolation='nearest',
                 cmap=fill_na_cmap, vmin=0, vmax=1, zorder=1)

    # Set line color
    if col is None:
        col = 'black'

    # Convert numeric line style to matplotlib style
    lty_map = {1: '-', 2: '--', 3: '-.', 4: ':', 5: '-', 6: '--'}
    if isinstance(lty, int):
        lty = lty_map.get(lty, '--')

    if swapped:
        if nvar1 == 1 or nvar2 == 2:
            # Add vertical lines on 1-D diagram
            if y_oxidation is not None and len(y_oxidation) > 0:
                ax.axvline(x=y_oxidation[0], linestyle=lty, linewidth=lwd, color=col)
            if y_reduction is not None and len(y_reduction) > 0:
                ax.axvline(x=y_reduction[0], linestyle=lty, linewidth=lwd, color=col)
        else:
            # xpoints above is really the ypoints
            if y_oxidation is not None:
                ax.plot(y_oxidation, xpoints, linestyle=lty, linewidth=lwd, color=col)
            if y_reduction is not None:
                ax.plot(y_reduction, xpoints, linestyle=lty, linewidth=lwd, color=col)
    else:
        if y_oxidation is not None:
            ax.plot(xpoints, y_oxidation, linestyle=lty, linewidth=lwd, color=col)
        if y_reduction is not None:
            ax.plot(xpoints, y_reduction, linestyle=lty, linewidth=lwd, color=col)

    # Update the figure and axes references in result to reflect the water lines
    fig = ax.get_figure()
    result['fig'] = fig
    result['ax'] = ax

    # Display the figure if plot_it=True
    # This allows water_lines() to display a figure that was created with plot_it=False
    if plot_it and fig is not None:
        try:
            from IPython.display import display
            display(fig)
        except (ImportError, NameError):
            # Not in IPython/Jupyter, matplotlib will handle display
            pass

    # Update result with water line data and return
    result.update({'xpoints': xpoints, 'y_oxidation': y_oxidation, 'y_reduction': y_reduction, 'swapped': swapped})
    return result


def _water_lines_plotly(eout: Dict[str, Any],
                        xpoints: np.ndarray,
                        y_oxidation: Optional[np.ndarray],
                        y_reduction: Optional[np.ndarray],
                        swapped: bool,
                        lty: Union[int, str],
                        lwd: float,
                        col: Optional[str],
                        plot_it: bool) -> Dict[str, Any]:
    """
    Add water stability lines to a Plotly interactive diagram.

    This helper function adds dashed lines showing water oxidation and reduction
    stability limits, plus gray shading for water-unstable regions, to an
    interactive Plotly diagram.
    """
    import plotly.graph_objects as go
    import numpy as np

    # Get the Plotly figure from eout
    fig = eout['fig']

    # Set line color (default to black)
    if col is None:
        col = 'black'

    # Convert numeric/matplotlib line styles to Plotly dash types
    lty_map = {
        1: 'solid', '-': 'solid',
        2: 'dash', '--': 'dash',
        3: 'dashdot', '-.': 'dashdot',
        4: 'dot', ':': 'dot',
        5: 'solid', 6: 'dash'
    }
    dash_style = lty_map.get(lty, 'dash') if (isinstance(lty, int) or lty in lty_map) else 'dash'

    # Get axis limits to determine shading extent
    # We need to extract axis ranges from the existing figure
    if fig.layout.xaxis.range:
        xlim = fig.layout.xaxis.range
    else:
        # If not set, estimate from data
        xlim = [xpoints.min(), xpoints.max()]

    if fig.layout.yaxis.range:
        ylim = fig.layout.yaxis.range
    else:
        # If not set, estimate from y_oxidation and y_reduction
        if y_oxidation is not None and y_reduction is not None:
            ylim = [min(y_oxidation.min(), y_reduction.min()),
                   max(y_oxidation.max(), y_reduction.max())]
        else:
            ylim = [0, 1]  # Fallback

    # Add gray shading for water-unstable regions
    if y_oxidation is not None and y_reduction is not None:
        # Create high-resolution mesh for smooth shading
        n_points = 200

        if swapped:
            # When swapped, xpoints is on y-axis, y values on x-axis
            # Create shading shapes for regions outside water stability

            # Upper unstable region (above oxidation line)
            y_mesh = np.linspace(ylim[0], ylim[1], n_points)
            x_ox_interp = np.interp(y_mesh, xpoints, y_oxidation)

            # Fill from oxidation line to right edge
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_ox_interp, [xlim[1]] * len(y_mesh), x_ox_interp[::-1]]),
                y=np.concatenate([y_mesh, y_mesh[::-1], y_mesh[::-1]]),
                fill='toself',
                fillcolor='rgba(128, 128, 128, 0.5)',  # Gray with transparency
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Lower unstable region (below reduction line)
            x_red_interp = np.interp(y_mesh, xpoints, y_reduction)

            # Fill from left edge to reduction line
            fig.add_trace(go.Scatter(
                x=np.concatenate([[xlim[0]] * len(y_mesh), x_red_interp[::-1], [xlim[0]] * len(y_mesh)]),
                y=np.concatenate([y_mesh, y_mesh[::-1], y_mesh[::-1]]),
                fill='toself',
                fillcolor='rgba(128, 128, 128, 0.5)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

        else:
            # Normal: xpoints on x-axis, y values on y-axis
            # Interpolate to create smooth shading boundaries
            x_mesh = np.linspace(xlim[0], xlim[1], n_points)
            y_ox_interp = np.interp(x_mesh, xpoints, y_oxidation)
            y_red_interp = np.interp(x_mesh, xpoints, y_reduction)

            # Upper unstable region (above oxidation line)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_mesh, x_mesh[::-1]]),
                y=np.concatenate([y_ox_interp, [ylim[1]] * len(x_mesh)]),
                fill='toself',
                fillcolor='rgba(128, 128, 128, 0.5)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

            # Lower unstable region (below reduction line)
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_mesh, x_mesh[::-1]]),
                y=np.concatenate([[ylim[0]] * len(x_mesh), y_red_interp[::-1]]),
                fill='toself',
                fillcolor='rgba(128, 128, 128, 0.5)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))

    # Add water stability lines
    if swapped:
        # When swapped: xpoints is on y-axis, y values on x-axis
        if y_oxidation is not None:
            fig.add_trace(go.Scatter(
                x=y_oxidation,
                y=xpoints,
                mode='lines',
                line=dict(color=col, width=lwd, dash=dash_style),
                name='H₂O oxidation limit',
                showlegend=False,
                hoverinfo='skip'
            ))
        if y_reduction is not None:
            fig.add_trace(go.Scatter(
                x=y_reduction,
                y=xpoints,
                mode='lines',
                line=dict(color=col, width=lwd, dash=dash_style),
                name='H₂O reduction limit',
                showlegend=False,
                hoverinfo='skip'
            ))
    else:
        # Normal orientation
        if y_oxidation is not None:
            fig.add_trace(go.Scatter(
                x=xpoints,
                y=y_oxidation,
                mode='lines',
                line=dict(color=col, width=lwd, dash=dash_style),
                name='H₂O oxidation limit',
                showlegend=False,
                hoverinfo='skip'
            ))
        if y_reduction is not None:
            fig.add_trace(go.Scatter(
                x=xpoints,
                y=y_reduction,
                mode='lines',
                line=dict(color=col, width=lwd, dash=dash_style),
                name='H₂O reduction limit',
                showlegend=False,
                hoverinfo='skip'
            ))

    # Update the figure reference in eout to reflect the water lines
    eout['fig'] = fig

    # Display the figure if plot_it=True
    if plot_it:
        try:
            from IPython.display import display
            display(fig)
        except (ImportError, NameError):
            # Not in IPython/Jupyter, use fig.show()
            fig.show()

    # Update eout with water line data and return
    eout.update({'xpoints': xpoints, 'y_oxidation': y_oxidation, 'y_reduction': y_reduction, 'swapped': swapped})
    return eout


def find_tp(predominant: np.ndarray) -> np.ndarray:
    """
    Find triple points in a predominance diagram.

    This function identifies the approximate positions of triple points
    (where three phases meet) in a 2-D predominance diagram by locating
    cells with the greatest number of different neighboring values.

    Parameters
    ----------
    predominant : np.ndarray
        Matrix of integers from diagram() output indicating which species
        predominates at each point. Should be a 2-D array where each value
        represents a different species/phase.

    Returns
    -------
    np.ndarray
        Array of shape (n, 2) where n is the number of triple points found.
        Each row contains [row_index, col_index] of a triple point location.
        Indices are 1-based to match R behavior.

    Examples
    --------
    >>> from pychnosz import *
    >>> reset()
    >>> basis(["corundum", "quartz", "oxygen"])
    >>> species(["kyanite", "sillimanite", "andalusite"])
    >>> a = affinity(T=[200, 900, 99], P=[0, 9000, 101], exceed_Ttr=True)
    >>> d = diagram(a)
    >>> tp = find_tp(d['predominant'])
    >>> # Get T and P at the triple point
    >>> Ttp = a['vals'][0][tp[0, 1] - 1]  # -1 for 0-based indexing
    >>> Ptp = a['vals'][1][::-1][tp[0, 0] - 1]  # reversed and -1

    Notes
    -----
    This is a Python translation of the R function find.tp() from CHNOSZ.
    The R version returns 1-based indices, and this Python version does too
    for consistency. When using these indices to access Python arrays,
    remember to subtract 1.

    The function works by:
    1. Rearranging the matrix as done by diagram() for plotting
    2. For each position, examining a 3x3 neighborhood
    3. Counting the number of unique values in that neighborhood
    4. Returning positions with the maximum count (typically 3 or more)
    """
    # Rearrange the matrix in the same way that diagram() does for 2-D predominance diagrams
    # R code: x <- t(x[, ncol(x):1])
    # This means: first reverse columns, then transpose
    x = np.transpose(predominant[:, ::-1])

    # Get all positions with valid values (> 0)
    valid_positions = np.argwhere(x > 0)

    if len(valid_positions) == 0:
        return np.array([])

    # For each position, count unique values in 3x3 neighborhood
    counts = []
    for pos in valid_positions:
        row, col = pos

        # Define the range to look at (3x3 except at edges)
        r1 = max(row - 1, 0)
        r2 = min(row + 1, x.shape[0] - 1)
        c1 = max(col - 1, 0)
        c2 = min(col + 1, x.shape[1] - 1)

        # Extract the neighborhood
        neighborhood = x[r1:r2+1, c1:c2+1]

        # Count unique values
        n_unique = len(np.unique(neighborhood))
        counts.append(n_unique)

    counts = np.array(counts)

    # Find positions with the maximum count
    max_count = np.max(counts)
    max_positions = valid_positions[counts == max_count]

    # Convert to 1-based indexing (to match R)
    # Return as [row, col] with 1-based indices
    result = max_positions + 1

    return result


def diagram_interactive(eout: Dict[str, Any],
                        type: str = "auto",
                        main: Optional[str] = None,
                        borders: Union[float, str] = 0,
                        names: Optional[List[str]] = None,
                        format_names: bool = True,
                        annotation: Optional[str] = None,
                        annotation_coords: List[float] = [0, 0],
                        balance: Optional[Union[str, float, List[float]]] = None,
                        xlab: Optional[str] = None,
                        ylab: Optional[str] = None,
                        fill: Optional[Union[str, List[str]]] = "viridis",
                        width: int = 600,
                        height: int = 520,
                        alpha: Union[bool, str] = False,
                        add: bool = False,
                        ax: Optional[Any] = None,
                        col: Optional[Union[str, List[str]]] = None,
                        lty: Optional[Union[str, int, List]] = None,
                        lwd: Union[float, List[float]] = 1,
                        cex: Union[float, List[float]] = 1.0,
                        contour_method: Optional[Union[str, List[str]]] = "edge",
                        messages: bool = True,
                        plot_it: bool = True,
                        save_as: Optional[str] = None,
                        save_format: Optional[str] = None,
                        save_scale: float = 1) -> Tuple[pd.DataFrame, Any]:
    """
    Create an interactive diagram using Plotly.

    This function produces interactive versions of the diagrams created by diagram(),
    using Plotly for interactivity. It accepts output from affinity() or equilibrate()
    and creates either 1D line plots or 2D predominance diagrams.

    Parameters
    ----------
    eout : dict
        Output from affinity() or equilibrate().
    main : str, optional
        Title of the plot.
    borders : float or str, default 0
        Controls boundary lines between regions in 2D predominance diagrams.
        - If numeric > 0: draws grid-aligned borders with specified thickness (pixels)
        - If "contour": draws smooth contour-based boundaries (like diagram())
        - If 0 or None: no borders drawn
    names : list of str, optional
        Names of species for activity lines or predominance fields.
    format_names : bool, default True
        Apply formatting to chemical formulas?
    annotation : str, optional
        Annotation to add to the plot.
    annotation_coords : list of float, default [0, 0]
        Coordinates of annotation, where 0,0 is bottom left and 1,1 is top right.
    balance : str or numeric, optional
        How to balance the transformations.
    xlab : str, optional
        Custom x-axis label.
    ylab : str, optional
        Custom y-axis label.
    fill : str or list of str, default "viridis"
        For 2D diagrams: colormap name (e.g., "viridis", "hot") or list of colors.
        For 1D diagrams: list of line colors.
    width : int, default 600
        Width of the plot in pixels.
    height : int, default 520
        Height of the plot in pixels.
    alpha : bool or str, default False
        For speciation diagrams, plot degree of formation instead of activities?
        If True, plots mole fractions. If "balance", scales by stoichiometry.
    messages : bool, default True
        Display messages?
    plot_it : bool, default True
        Show the plot?
    save_as : str, optional
        Provide a filename to save this figure. Filetype of saved figure is
        determined by save_format.
    save_format : str, default "png"
        Desired format of saved or downloaded figure. Can be 'png', 'jpg', 'jpeg',
        'webp', 'svg', 'pdf', 'eps', 'json', or 'html'. If 'html', an interactive
        plot will be saved.
    save_scale : float, default 1
        Multiply title/legend/axis/canvas sizes by this factor when saving.

    Returns
    -------
    tuple
        (df, fig) where df is a pandas DataFrame with the data and fig is the
        Plotly figure object.

    Examples
    --------
    1D diagram:
    >>> basis("CHNOS+")
    >>> species(info(["glycinium", "glycine", "glycinate"]))
    >>> a = affinity(pH=[0, 14])
    >>> e = equilibrate(a)
    >>> diagram_interactive(e, alpha=True)

    2D diagram:
    >>> basis(["Fe", "oxygen", "S2"])
    >>> species(["iron", "ferrous-oxide", "magnetite", "hematite", "pyrite", "pyrrhotite"])
    >>> a = affinity(S2=[-50, 0], O2=[-90, -10], T=200)
    >>> diagram_interactive(a, fill="hot")

    Notes
    -----
    This function requires plotly to be installed. Install with:
        pip install plotly

    The function adapts the pyCHNOSZ diagram_interactive() implementation
    to work with Python CHNOSZ's native data structures.
    """

    # Import plotly (lazy import to avoid dependency issues)
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        raise ImportError("diagram_interactive() requires plotly. Install with: pip install plotly")

    # Check that eout is valid
    efun = eout.get('fun', '')
    if efun not in ['affinity', 'equilibrate', 'solubility']:
        raise ValueError("'eout' is not the output from affinity(), equilibrate(), or solubility()")

    # Determine if this is affinity or equilibrate output
    calc_type = "a" if ('loga_equil' not in eout and 'loga.equil' not in eout) else "e"

    # Get basis species and their states
    basis_df = eout['basis']
    basis_sp = list(basis_df.index)
    basis_state = list(basis_df['state'])

    # Get variable names and values
    xyvars = eout['vars']
    xyvals_dict = eout['vals']
    # Convert vals dict to list format for easier access
    xyvals = [xyvals_dict[var] for var in xyvars]

    # Determine balance if not provided
    if balance is None or balance == "":
        # For saturation diagrams, use balance=1 (formula units) to match R behavior
        # This avoids issues when minerals don't have a common basis element
        if type == "saturation":
            balance = 1
            n_balance = [1] * len(eout['values'])
        else:
            # Call diagram with plot_it=False to get balance
            # Need to import matplotlib to close the figure afterward
            import matplotlib.pyplot as plt_temp
            temp_result = diagram(eout, messages=False, plot_it=False)
            balance = temp_result.get('balance', 1)
            n_balance = temp_result.get('n_balance', [1])
            # Close the matplotlib figure created by diagram() since we don't need it
            if 'fig' in temp_result and temp_result['fig'] is not None:
                plt_temp.close(temp_result['fig'])
    else:
        # Calculate n_balance from balance
        try:
            balance_float = float(balance)
            n_balance = [balance_float] * len(eout['values'])
        except (ValueError, TypeError):
            # balance is a string (element name)
            # Get species from eout instead of global state
            if 'species' in eout and eout['species'] is not None:
                sp_df = eout['species']
            else:
                # Fallback to global species if not in eout
                from .species import species as species_func
                sp_df = species_func()

            # Check if balance is a list (user-provided values) or a string (column name)
            if isinstance(balance, list):
                n_balance = balance
            elif balance in sp_df.columns:
                n_balance = list(sp_df[balance])
            else:
                n_balance = [1] * len(eout['values'])

    # Get output values
    if calc_type == "a":
        # handling output of affinity()
        out_vals = eout['values']
        out_units = "A/(2.303RT)"
    else:
        # handling output of equilibrate()
        loga_equil_key = 'loga_equil' if 'loga_equil' in eout else 'loga.equil'
        out_vals = eout[loga_equil_key]
        out_units = "log a"

    # Convert values to a list format
    if isinstance(out_vals, dict):
        nsp = len(out_vals)
        values_list = list(out_vals.values())
        species_indices = list(out_vals.keys())
    else:
        nsp = len(out_vals)
        values_list = out_vals
        species_indices = eout['species']['ispecies'].tolist()

    # Get species names
    from .info import info as info_func
    # Convert numpy types to Python types
    species_indices_py = [int(idx) for idx in species_indices]
    sp_info = info_func(species_indices_py, messages=False)
    sp_names = sp_info['name'].tolist()

    # Use custom names if provided
    if isinstance(names, list) and len(names) == len(sp_names):
        sp_names = names

    # Determine dimensions
    first_val = values_list[0]
    if hasattr(first_val, 'shape'):
        nd = len(first_val.shape)
    else:
        nd = 1 if hasattr(first_val, '__len__') else 0

    # Handle type="saturation" - plot contour lines where affinity=0
    if type == "saturation":
        if nd != 2:
            raise ValueError("type='saturation' requires 2-D diagram")
        if calc_type != "a":
            raise ValueError("type='saturation' requires output from affinity(), not equilibrate()")

        # Delegate to saturation plotting function
        return _plot_saturation_interactive(
            eout, values_list, sp_names, xyvars, xyvals,
            xlab, ylab, col, lwd, lty, cex, contour_method,
            main, add, ax, width, height, plot_it,
            save_as, save_format, save_scale, messages
        )

    # Build DataFrame
    if nd == 2:
        # 2D case - flatten the data
        xvals = xyvals[0]
        yvals = xyvals[1]
        xvar = xyvars[0]
        yvar = xyvars[1]

        # Flatten the data - transpose first so coordinates match
        # Original shape is (nx, ny) where nx=len(xvals), ny=len(yvals)
        # After transpose, shape is (ny, nx)
        # Flattening with C-order then gives: [row0, row1, ...] = [x-values at y[0], x-values at y[1], ...]
        flat_out_vals = []
        for v in values_list:
            # Transpose then flatten so coordinates align correctly
            flat_out_vals.append(v.T.flatten())
        df = pd.DataFrame(flat_out_vals, index=sp_names).T

        # Apply balance if needed
        if calc_type == "a":
            if isinstance(balance, str):
                # Get balance from species dataframe
                # Get species from eout instead of global state
                if 'species' in eout and eout['species'] is not None:
                    sp_df = eout['species']
                else:
                    # Fallback to global species if not in eout
                    from .species import species as species_func
                    sp_df = species_func()

                # Check if balance is a list (user-provided values) or a string (column name)
                if isinstance(balance, list):
                    n_balance = balance
                elif balance in sp_df.columns:
                    n_balance = list(sp_df[balance])
            # Divide by balance
            for i, sp in enumerate(sp_names):
                df[sp] = df[sp] / n_balance[i]

        # Find predominant species
        df["pred"] = df.idxmax(axis=1, skipna=True)
        df["prednames"] = df["pred"]

        # Add x and y coordinates
        # After transpose and flatten, data is ordered as:
        # [x0,y0], [x1,y0], ..., [xn,y0], [x0,y1], [x1,y1], ...
        xvals_full = list(xvals) * len(yvals)
        yvals_full = []
        for y in yvals:
            yvals_full.extend([y] * len(xvals))
        df[xvar] = xvals_full
        df[yvar] = yvals_full

    else:
        # 1D case
        xvar = xyvars[0]
        xvals = xyvals[0]

        flat_out_vals = []
        for v in values_list:
            flat_out_vals.append(v)
        df = pd.DataFrame(flat_out_vals, index=sp_names).T

        # Apply balance if needed
        if calc_type == "a":
            if isinstance(balance, str):
                # Get species from eout instead of global state
                if 'species' in eout and eout['species'] is not None:
                    sp_df = eout['species']
                else:
                    # Fallback to global species if not in eout
                    from .species import species as species_func
                    sp_df = species_func()

                # Check if balance is a list (user-provided values) or a string (column name)
                if isinstance(balance, list):
                    n_balance = balance
                elif balance in sp_df.columns:
                    n_balance = list(sp_df[balance])
            # Divide by balance
            for i, sp in enumerate(sp_names):
                df[sp] = df[sp] / n_balance[i]

        # Handle alpha (degree of formation)
        if alpha:
            df = df.apply(lambda x: 10**x)
            df = df[sp_names].div(df[sp_names].sum(axis=1), axis=0)

        df[xvar] = xvals

    # Create axis labels
    unit_dict = {"P": "bar", "T": "°C", "pH": "", "Eh": "volts", "IS": "mol/kg"}

    for i, s in enumerate(basis_sp):
        if basis_state[i] in ["aq", "liq", "cr"]:
            if format_names:
                unit_dict[s] = f"log <i>a</i><sub>{_format_html_species(s)}</sub>"
            else:
                unit_dict[s] = f"log <i>a</i><sub>{s}</sub>"
        else:
            if format_names:
                unit_dict[s] = f"log <i>f</i><sub>{_format_html_species(s)}</sub>"
            else:
                unit_dict[s] = f"log <i>f</i><sub>{s}</sub>"

    # Set x-axis label
    if not isinstance(xlab, str):
        xlab = xvar + ", " + unit_dict.get(xvar, "")
        if xvar == "pH":
            xlab = "pH"
        if xvar in basis_sp:
            xlab = unit_dict[xvar]

    # Create the plot
    if nd == 1:
        # 1D plot
        # Melt the dataframe for plotting
        df_melted = pd.melt(df, id_vars=[xvar], value_vars=sp_names, var_name='variable', value_name='value')

        # Format species names if requested
        if format_names:
            df_melted['variable'] = df_melted['variable'].apply(_format_html_species)

        # Set y-axis label
        if not isinstance(ylab, str):
            if alpha:
                ylab = "alpha"
            else:
                ylab = out_units

        fig = px.line(df_melted, x=xvar, y="value", color='variable',
                      template="simple_white", width=width, height=height,
                      labels={'value': ylab, xvar: xlab},
                      render_mode='svg')

        # Apply custom colors if provided
        if isinstance(fill, list):
            for i, color in enumerate(fill):
                if i < len(fig.data):
                    fig.data[i].line.color = color

        # Check for LaTeX format in axis labels
        if xlab and _detect_latex_format(xlab):
            warnings.warn(
                "LaTeX formatting detected in 'xlab' parameter. "
                "Plotly requires HTML format (<sub>, <sup>) instead of LaTeX ($, _, ^). "
                "For activity ratios, use ratlab_html() instead of ratlab().",
                UserWarning
            )
        if ylab and _detect_latex_format(ylab):
            warnings.warn(
                "LaTeX formatting detected in 'ylab' parameter. "
                "Plotly requires HTML format (<sub>, <sup>) instead of LaTeX ($, _, ^). "
                "For activity ratios, use ratlab_html() instead of ratlab().",
                UserWarning
            )

        fig.update_layout(xaxis_title=xlab,
                          yaxis_title=ylab,
                          legend_title=None)

        if isinstance(main, str):
            fig.update_layout(title={'text': main, 'x': 0.5, 'xanchor': 'center'})

        if isinstance(annotation, str):
            # Check for LaTeX format and warn user
            if _detect_latex_format(annotation):
                warnings.warn(
                    "LaTeX formatting detected in 'annotation' parameter. "
                    "Plotly requires HTML format (<sub>, <sup>) instead of LaTeX ($, _, ^). "
                    "For activity ratios, use ratlab_html() instead of ratlab().",
                    UserWarning
                )

            fig.add_annotation(
                x=annotation_coords[0],
                y=annotation_coords[1],
                text=annotation,
                showarrow=False,
                xref="paper",
                yref="paper",
                align='left',
                bgcolor="rgba(255, 255, 255, 0.5)")

        # Configure download button
        save_as_name, save_format_final = _save_figure(fig, save_as, save_format, save_scale,
                                                        plot_width=width, plot_height=height, ppi=1)

        config = {'displaylogo': False,
                  'modeBarButtonsToRemove': ['resetScale2d', 'toggleSpikelines'],
                  'toImageButtonOptions': {
                      'format': save_format_final,
                      'filename': save_as_name,
                      'height': height,
                      'width': width,
                      'scale': save_scale,
                  }}

        # Store config on figure so it persists when fig.show() is called later
        fig._config = fig._config | config

    else:
        # 2D plot
        # Map species names to numeric values
        mappings = {s: lab for s, lab in zip(sp_names, range(len(sp_names)))}
        df['pred'] = df['pred'].map(mappings).astype(int)

        # Reshape data
        # Data is flattened as [x0,y0], [x1,y0], ..., [xn,y0], [x0,y1], ...
        # Reshape to (ny, nx) for proper orientation in Plotly
        # Plotly expects data[i,j] to correspond to x[j], y[i]
        data = np.array(df['pred'])
        shape = (len(yvals), len(xvals))
        dmap = data.reshape(shape)

        data_names = np.array(df['prednames'])
        dmap_names = data_names.reshape(shape)

        # Set y-axis label
        if not isinstance(ylab, str):
            ylab = yvar + ", " + unit_dict.get(yvar, "")
            if yvar in basis_sp:
                ylab = unit_dict[yvar]
            if yvar == "pH":
                ylab = "pH"

        # Check for LaTeX format in axis labels (2D plot)
        if xlab and _detect_latex_format(xlab):
            warnings.warn(
                "LaTeX formatting detected in 'xlab' parameter. "
                "Plotly requires HTML format (<sub>, <sup>) instead of LaTeX ($, _, ^). "
                "For activity ratios, use ratlab_html() instead of ratlab().",
                UserWarning
            )
        if ylab and _detect_latex_format(ylab):
            warnings.warn(
                "LaTeX formatting detected in 'ylab' parameter. "
                "Plotly requires HTML format (<sub>, <sup>) instead of LaTeX ($, _, ^). "
                "For activity ratios, use ratlab_html() instead of ratlab().",
                UserWarning
            )

        # Create heatmap
        fig = px.imshow(dmap, width=width, height=height, aspect="auto",
                        labels={'x': xlab, 'y': ylab, 'color': "region"},
                        x=xvals, y=yvals, template="simple_white")

        fig.update(data=[{'customdata': dmap_names,
                          'hovertemplate': xlab + ': %{x}<br>' + ylab + ': %{y}<br>Region: %{customdata}<extra></extra>'}])

        # Set colormap
        if fill == 'none':
            colormap = [[0, 'white'], [1, 'white']]
        elif isinstance(fill, list):
            colmap_temp = []
            for i, v in enumerate(fill):
                colmap_temp.append([i / (len(fill) - 1) if len(fill) > 1 else 0, v])
            colormap = colmap_temp
        else:
            colormap = fill

        fig.update_traces(dict(showscale=False,
                               coloraxis=None,
                               colorscale=colormap),
                          selector={'type': 'heatmap'})

        fig.update_yaxes(autorange=True)

        if isinstance(main, str):
            fig.update_layout(title={'text': main, 'x': 0.5, 'xanchor': 'center'})

        # Add species labels
        for s in sp_names:
            if s in set(df["prednames"]):
                df_s = df.loc[df["prednames"] == s]
                namex = df_s[xvar].mean()
                namey = df_s[yvar].mean()

                if format_names:
                    annot_text = _format_html_species(s)
                else:
                    annot_text = str(s)

                fig.add_annotation(x=namex, y=namey,
                                   text=annot_text,
                                   bgcolor="rgba(255, 255, 255, 0.5)",
                                   showarrow=False)

        if isinstance(annotation, str):
            # Check for LaTeX format and warn user
            if _detect_latex_format(annotation):
                warnings.warn(
                    "LaTeX formatting detected in 'annotation' parameter. "
                    "Plotly requires HTML format (<sub>, <sup>) instead of LaTeX ($, _, ^). "
                    "For activity ratios, use ratlab_html() instead of ratlab().",
                    UserWarning
                )

            fig.add_annotation(
                x=annotation_coords[0],
                y=annotation_coords[1],
                text=annotation,
                showarrow=False,
                xref="paper",
                yref="paper",
                align='left',
                bgcolor="rgba(255, 255, 255, 0.5)")

        # Add borders if requested
        if borders == "contour":
            # Use contour-based boundaries (smooth, like diagram())
            # Draw boundaries using matplotlib contour extraction without filling

            # Get unique species (excluding any that don't appear)
            unique_species_names = sorted(df["prednames"].unique())

            # Create a temporary matplotlib figure to extract contour paths
            # We won't display it, just use it to calculate contours
            temp_fig, temp_ax = plt.subplots()

            # For each species, create a binary mask and extract contours
            for i, sp_name in enumerate(unique_species_names):
                # Create binary mask: 1 where this species predominates, 0 elsewhere
                z = (dmap_names == sp_name).astype(float)

                # Create meshgrid for contour
                X, Y = np.meshgrid(xvals, yvals)

                # Find contours at level 0.5 using matplotlib
                try:
                    cs = temp_ax.contour(X, Y, z, levels=[0.5])

                    # Extract the contour segments
                    # cs.allsegs is a list of lists: [level][segment]
                    for level_segs in cs.allsegs:
                        for segment in level_segs:
                            # segment is an (N, 2) array of (x, y) coordinates
                            # Add as a scatter trace with lines
                            fig.add_trace(
                                go.Scatter(
                                    x=segment[:, 0],
                                    y=segment[:, 1],
                                    mode='lines',
                                    line=dict(color='black', width=2),
                                    hoverinfo='skip',
                                    showlegend=False
                                )
                            )

                    # Clear the temp axes for next species
                    temp_ax.clear()
                except Exception as e:
                    if messages:
                        warnings.warn(f"Could not draw contour for {sp_name}: {e}")
                    pass  # Skip if contour can't be drawn

            # Close the temporary figure
            plt.close(temp_fig)

        elif isinstance(borders, (int, float)) and borders > 0:
            unique_x_vals = sorted(list(set(df[xvar])))
            unique_y_vals = sorted(list(set(df[yvar])))

            # Skip border drawing if there are fewer than 2 unique values
            # (single point or single line - no borders to draw between regions)
            if len(unique_x_vals) < 2 or len(unique_y_vals) < 2:
                if messages:
                    warnings.warn("Skipping border drawing: need at least 2 unique values in each dimension")
            else:
                def mov_mean(numbers, window_size=2):
                    moving_averages = []
                    for i in range(len(numbers) - window_size + 1):
                        window_average = sum(numbers[i:i + window_size]) / window_size
                        moving_averages.append(window_average)
                    return moving_averages

                x_mov_mean = mov_mean(unique_x_vals)
                y_mov_mean = mov_mean(unique_y_vals)

                x_plot_min = x_mov_mean[0] - (x_mov_mean[1] - x_mov_mean[0])
                y_plot_min = y_mov_mean[0] - (y_mov_mean[1] - y_mov_mean[0])

                x_plot_max = x_mov_mean[-1] + (x_mov_mean[1] - x_mov_mean[0])
                y_plot_max = y_mov_mean[-1] + (y_mov_mean[1] - y_mov_mean[0])

                x_vals_border = [x_plot_min] + x_mov_mean + [x_plot_max]
                y_vals_border = [y_plot_min] + y_mov_mean + [y_plot_max]

                # Find border lines
                def find_line(dmap, row_index):
                    return [i for i in range(len(dmap[row_index]) - 1) if dmap[row_index][i] != dmap[row_index][i + 1]]

                nrows, ncols = dmap.shape
                vlines = [find_line(dmap, row_i) for row_i in range(nrows)]

                dmap_transposed = dmap.transpose()
                nrows_t, ncols_t = dmap_transposed.shape
                hlines = [find_line(dmap_transposed, row_i) for row_i in range(nrows_t)]

                y_coord_list_vertical = []
                x_coord_list_vertical = []
                for i, row in enumerate(vlines):
                    for line in row:
                        x_coord_list_vertical += [x_vals_border[line + 1], x_vals_border[line + 1], np.nan]
                        y_coord_list_vertical += [y_vals_border[i], y_vals_border[i + 1], np.nan]

                y_coord_list_horizontal = []
                x_coord_list_horizontal = []
                for i, col in enumerate(hlines):
                    for line in col:
                        y_coord_list_horizontal += [y_vals_border[line + 1], y_vals_border[line + 1], np.nan]
                        x_coord_list_horizontal += [x_vals_border[i], x_vals_border[i + 1], np.nan]

                fig.add_trace(
                    go.Scatter(
                        mode='lines',
                        x=x_coord_list_horizontal,
                        y=y_coord_list_horizontal,
                        line={'width': borders, 'color': 'black'},
                        hoverinfo='skip',
                        showlegend=False))

                fig.add_trace(
                    go.Scatter(
                        mode='lines',
                        x=x_coord_list_vertical,
                        y=y_coord_list_vertical,
                        line={'width': borders, 'color': 'black'},
                        hoverinfo='skip',
                        showlegend=False))

                fig.update_yaxes(range=[min(yvals), max(yvals)], autorange=False, mirror=True)
                fig.update_xaxes(range=[min(xvals), max(xvals)], autorange=False, mirror=True)

        # Configure download button
        save_as_name, save_format_final = _save_figure(fig, save_as, save_format, save_scale,
                                                        plot_width=width, plot_height=height, ppi=1)

        config = {'displaylogo': False,
                  'modeBarButtonsToRemove': ['zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d',
                                             'autoScale2d', 'resetScale2d', 'toggleSpikelines',
                                             'hoverClosestCartesian', 'hoverCompareCartesian'],
                  'toImageButtonOptions': {
                      'format': save_format_final,
                      'filename': save_as_name,
                      'height': height,
                      'width': width,
                      'scale': save_scale,
                  }}

        # Store config on figure so it persists when fig.show() is called later
        fig._config = fig._config | config

    if plot_it:
        fig.show(config=config)

    return df, fig


def _detect_latex_format(text: str) -> bool:
    """
    Detect if a string contains LaTeX formatting (incompatible with Plotly).

    Parameters
    ----------
    text : str
        Text to check for LaTeX formatting

    Returns
    -------
    bool
        True if LaTeX formatting is detected (e.g., $...$, _{...}, ^{...})
    """
    import re
    # Check for common LaTeX patterns:
    # - Text wrapped in $ $
    # - LaTeX subscripts _{...}
    # - LaTeX superscripts ^{...}
    latex_patterns = [
        r'\$[^$]+\$',           # $...$
        r'_\{[^}]+\}',          # _{...}
        r'\^\{[^}]+\}'          # ^{...}
    ]

    for pattern in latex_patterns:
        if re.search(pattern, text):
            return True
    return False


def _format_html_species(formula: str) -> str:
    """
    Format a chemical formula for HTML rendering in Plotly.

    Converts chemical formulas like "H2O" to "H<sub>2</sub>O" and
    "Ca+2" to "Ca<sup>2+</sup>".

    Parameters
    ----------
    formula : str
        Chemical formula to format

    Returns
    -------
    str
        HTML-formatted formula
    """
    import re

    # Handle charge notation (e.g., +2, -1)
    # Match patterns like +2, -2, +, -
    charge_pattern = r'([+-])(\d*)'

    def format_charge(match):
        sign = match.group(1)
        num = match.group(2)
        if num == '' or num == '1':
            return f"<sup>{sign}</sup>"
        else:
            return f"<sup>{num}{sign}</sup>"

    # First handle charges at the end
    formula = re.sub(charge_pattern + r'$', format_charge, formula)

    # Handle subscript numbers (digits that aren't part of the charge)
    # Match digits that come after letters and aren't preceded by < or >
    def format_subscript(match):
        return f"<sub>{match.group(0)}</sub>"

    # Find all digits that should be subscripts
    # This matches digits that come after letters
    result = []
    i = 0
    while i < len(formula):
        if formula[i].isdigit() and i > 0 and formula[i-1].isalpha():
            # Start of a number sequence
            num_start = i
            while i < len(formula) and formula[i].isdigit():
                i += 1
            result.append(f"<sub>{formula[num_start:i]}</sub>")
        else:
            result.append(formula[i])
            i += 1

    return ''.join(result)


def _save_figure(fig, save_as, save_format, save_scale, plot_width, plot_height, ppi):
    """
    Save a Plotly figure to a file.

    Parameters
    ----------
    fig : plotly figure
        The figure to save
    save_as : str or None
        Filename (without extension) to save as
    save_format : str or None
        Format to save ('png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'eps', 'json', 'html')
    save_scale : float
        Scale factor for saving
    plot_width : int
        Width of the plot
    plot_height : int
        Height of the plot
    ppi : int
        Pixels per inch

    Returns
    -------
    tuple
        (save_as, save_format) - processed values for use in config
    """
    import plotly.io as pio

    valid_formats = ['png', 'jpg', 'jpeg', 'webp', 'svg', 'pdf', 'eps', 'json', 'html']

    if isinstance(save_format, str) and save_format not in valid_formats:
        raise ValueError(f"{save_format} is an unrecognized save format. "
                         f"Supported formats include: {', '.join(valid_formats)}")

    if isinstance(save_format, str) and save_as is not None:
        if not isinstance(save_as, str):
            save_as = "newplot"

        if save_format == "html":
            fig.write_html(save_as + ".html")
            print(f"Saved figure as {save_as}.html")
            save_format = 'png'
        elif save_format in ['pdf', 'eps', 'json']:
            pio.write_image(fig, save_as + "." + save_format, format=save_format,
                            scale=save_scale, width=plot_width * ppi, height=plot_height * ppi)
            print(f"Saved figure as {save_as}.{save_format}")
            save_format = "png"
        else:
            pio.write_image(fig, save_as + "." + save_format, format=save_format,
                            scale=save_scale, width=plot_width * ppi, height=plot_height * ppi)
            print(f"Saved figure as {save_as}.{save_format}")
    else:
        save_format = "png"

    return save_as, save_format


def _plot_saturation_interactive(eout, values_list, sp_names, xyvars, xyvals,
                                  xlab, ylab, col, lwd, lty, cex, contour_method,
                                  main, add, ax, width, height, plot_it,
                                  save_as, save_format, save_scale, messages):
    """
    Plot saturation lines (affinity=0 contours) for interactive 2-D diagrams using Plotly.

    This function draws contour lines where affinity = 0 for each species,
    indicating saturation boundaries (e.g., mineral precipitation thresholds).
    """
    import plotly.graph_objects as go
    import plotly.io as pio

    # Get x and y values
    xvals = xyvals[0]
    yvals = xyvals[1]
    xvar = xyvars[0]
    yvar = xyvars[1]

    n_species = len(sp_names)

    if messages:
        print(f"diagram: plotting saturation lines for interactive 2-D diagram")

    # Set up colors
    if col is None:
        # Use default Plotly colors
        default_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                         '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
        col = [default_colors[i % len(default_colors)] for i in range(n_species)]
    elif isinstance(col, str):
        col = [col] * n_species
    else:
        col = list(col)
        if len(col) < n_species:
            col = col * (n_species // len(col) + 1)
        col = col[:n_species]

    # Set up line widths
    if isinstance(lwd, (int, float)):
        lwd = [lwd] * n_species
    else:
        lwd = list(lwd)
        if len(lwd) < n_species:
            lwd = lwd * (n_species // len(lwd) + 1)
        lwd = lwd[:n_species]

    # Handle line styles (lty)
    if lty is None:
        lty = ['solid'] * n_species
    elif isinstance(lty, (str, int)):
        lty = [lty] * n_species
    else:
        lty = list(lty)
        if len(lty) < n_species:
            lty = lty * (n_species // len(lty) + 1)
        lty = lty[:n_species]

    # Convert numeric/matplotlib line styles to Plotly dash types
    lty_map = {
        1: 'solid', '-': 'solid',
        2: 'dash', '--': 'dash',
        3: 'dashdot', '-.': 'dashdot',
        4: 'dot', ':': 'dot',
        5: 'solid', 6: 'dash'
    }
    lty = [lty_map.get(lt, 'solid') if (isinstance(lt, int) or lt in lty_map) else 'solid' for lt in lty]

    # Handle contour label control
    if contour_method is None or contour_method == "" or (isinstance(contour_method, str) and contour_method.upper() == "NA"):
        show_labels = [False] * n_species
    elif isinstance(contour_method, str):
        show_labels = [True] * n_species
    elif isinstance(contour_method, list):
        if len(contour_method) != n_species:
            contour_method_extended = list(contour_method) * (n_species // len(contour_method) + 1)
            contour_method_extended = contour_method_extended[:n_species]
        else:
            contour_method_extended = contour_method

        show_labels = []
        for method in contour_method_extended:
            if method is None or method == "" or (isinstance(method, str) and method.upper() == "NA"):
                show_labels.append(False)
            else:
                show_labels.append(True)
    else:
        show_labels = [True] * n_species

    # Handle text size (cex) for contour labels
    if isinstance(cex, (int, float)):
        cex_list = [cex] * n_species
    else:
        cex_list = list(cex)
        if len(cex_list) < n_species:
            cex_list = cex_list * (n_species // len(cex_list) + 1)
        cex_list = cex_list[:n_species]

    # Base font size for labels (Plotly default ~12)
    base_font_size = 12
    font_sizes = [base_font_size * c for c in cex_list]

    # Create or get figure
    if add and ax is not None:
        # ax is actually the Plotly figure from previous call
        fig = ax
    else:
        # Create new figure
        fig = go.Figure()

    # Add contour lines for each species
    for i, sp_name in enumerate(sp_names):
        # Get affinity values for this species
        # values_list[i] has shape (nx, ny)
        affinity_2d = values_list[i]

        # Create contour trace for affinity=0 only
        # Use ncontours=1 with start=0 and end=0 to force a single contour at zero
        # Note: Plotly doesn't support custom text on contour lines, so we rely on the legend
        contour = go.Contour(
            x=xvals,
            y=yvals,
            z=affinity_2d.T,  # Transpose to match Plotly's expected orientation
            ncontours=1,  # Only generate one contour level
            contours=dict(
                coloring='lines',  # Draw only contour lines, not filled regions
                start=0,  # Start at 0
                end=0,    # End at 0
                showlabels=False  # Don't show "0" labels on contour lines
            ),
            line=dict(
                color=col[i],
                width=lwd[i],
                dash=lty[i]
            ),
            colorscale=[[0, col[i]], [1, col[i]]],  # Force uniform color
            showscale=False,
            hoverinfo='skip',
            name=sp_name,
            legendgroup=sp_name,
            showlegend=True
        )

        fig.add_trace(contour)

    # Set axis labels if not adding to existing plot
    if not add:
        # Create axis labels with proper units
        from .thermo import thermo as thermo_func
        basis_df = eout['basis']
        basis_sp = list(basis_df.index)
        basis_state = list(basis_df['state'])

        unit_dict = {"P": "bar", "T": "°C", "pH": "", "Eh": "volts", "IS": "mol/kg"}

        for i, s in enumerate(basis_sp):
            if basis_state[i] in ["aq", "liq", "cr"]:
                unit_dict[s] = f"log <i>a</i><sub>{s}</sub>"
            else:
                unit_dict[s] = f"log <i>f</i><sub>{s}</sub>"

        if not isinstance(xlab, str):
            xlab = xvar + ", " + unit_dict.get(xvar, "")
            if xvar == "pH":
                xlab = "pH"
            if xvar in basis_sp:
                xlab = unit_dict[xvar]

        if not isinstance(ylab, str):
            ylab = yvar + ", " + unit_dict.get(yvar, "")
            if yvar in basis_sp:
                ylab = unit_dict[yvar]
            if yvar == "pH":
                ylab = "pH"

        fig.update_xaxes(title_text=xlab)
        fig.update_yaxes(title_text=ylab)

        fig.update_layout(
            template="simple_white",
            width=width,
            height=height,
            showlegend=True
        )

        if isinstance(main, str):
            fig.update_layout(title={'text': main, 'x': 0.5, 'xanchor': 'center'})

    # Configure download button
    save_as_name, save_format_final = _save_figure(fig, save_as, save_format, save_scale,
                                                    plot_width=width, plot_height=height, ppi=1)

    config = {'displaylogo': False,
              'modeBarButtonsToRemove': ['resetScale2d', 'toggleSpikelines'],
              'toImageButtonOptions': {
                  'format': save_format_final,
                  'filename': save_as_name,
                  'height': height,
                  'width': width,
                  'scale': save_scale,
              }}

    fig._config = fig._config | config if hasattr(fig, '_config') else config

    # Show plot if requested
    if plot_it:
        fig.show(config=config)

    # Return empty DataFrame (saturation doesn't produce tabular data like predominance diagrams)
    df = pd.DataFrame()

    return df, fig


# Export main functions
__all__ = ['diagram', 'diagram_interactive', 'water_lines', 'find_tp']