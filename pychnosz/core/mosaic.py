"""
Mosaic diagram module.

This module provides a Python equivalent of the R function in mosaic.R:
- mosaic(): Calculate affinities of formation reactions while changing basis species

The affinities of the formation reactions of the species of interest are
calculated for every combination of the "candidate" basis species given in
'bases', then weighted by the equilibrium mole fractions (blend = True) or the
predominance (blend = False) of those candidate basis species.

Author: CHNOSZ Python port
"""

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Any, Sequence
import warnings

from .thermo import thermo
from .basis import get_basis, put_basis
from .species import get_species
from .info import info


class MosaicError(Exception):
    """Exception raised for mosaic-related errors."""
    pass


def mosaic(bases: Union[Sequence, Dict[str, Any]],
           blend: Union[bool, List[bool]] = True,
           stable: Optional[List[Optional[np.ndarray]]] = None,
           loga_aq: Optional[List[Optional[float]]] = None,
           messages: bool = True,
           **kwargs) -> Dict[str, Any]:
    """
    Calculate affinities with changing basis species.

    Chemical affinities are calculated for the formation reactions of the
    species of interest, considering every combination of the candidate basis
    species in ``bases``.  The affinities are then weighted by the equilibrium
    mole fractions of the candidate basis species (``blend=True``) or by which
    candidate basis species predominates (``blend=False``) and summed.  This
    produces a "mosaic" diagram, where the boundaries between the species of
    interest bend to follow the changing speciation of the basis species.

    Parameters
    ----------
    bases : list of str, list of list of str, or dict
        Candidate basis species.  A flat list (e.g. ``["H2S", "HS-", "HSO4-",
        "SO4-2"]``) defines a single group of changing basis species; a list of
        lists defines several independent groups.  The *first* species in each
        group must be present in the current basis definition -- it is the one
        that gets swapped out for the others.

        May also be the dict returned by a previous ``mosaic()`` call, in which
        case the remaining arguments update that call's arguments and the
        calculation is repeated (argument recall).
    blend : bool or list of bool, default True
        Calculate affinities using the equilibrium mole fractions of the
        candidate basis species?  If False, only the predominant candidate
        basis species in each group is used.  Can be a list with one value per
        group in ``bases``.
    stable : list, optional
        Pre-computed predominance matrices (as returned in the ``'predominant'``
        key of ``diagram()``), one per group in ``bases``.  Use ``None`` for a
        group that should be computed by ``mosaic()`` itself.  Supplying this
        implies ``blend=False`` behaviour for that group.  If ``stable`` has
        exactly two entries, the second group is reduced to only those
        candidate basis species that are predominant somewhere in
        ``stable[1]`` (mosaic stacking).
    loga_aq : list of float, optional
        Logarithm of activity of the *aqueous* mosaiced basis species, one
        value per group in ``bases`` (use ``None`` to leave a group alone).
        By default the activity from the starting basis definition is used.
    messages : bool, default True
        Whether to print informational messages.
    **kwargs
        Arguments for :func:`~pychnosz.core.affinity.affinity`, i.e. the
        plotting variables (``pH``, ``Eh``, ``T``, ``P``, basis species names,
        ...).

    Returns
    -------
    dict
        Dictionary containing:

        - ``fun`` : str, ``"mosaic"``
        - ``args`` : dict, all arguments used (for argument recall)
        - ``A_species`` : dict, ``affinity()``-like output holding the total
          affinities of the species of interest.  Also available under the
          R-style key ``'A.species'``.
        - ``A_bases`` : ``affinity()``-like output for the candidate basis
          species.  A single dict if ``bases`` was a flat list, otherwise a
          list with one entry per group.  Also under ``'A.bases'``.
        - ``E_bases`` : ``equilibrate()``-like output for each group of
          candidate basis species (empty list if ``blend=False``).  Also under
          ``'E.bases'``.

    Examples
    --------
    >>> import pychnosz as pc
    >>> pc.reset()
    >>> pc.basis(["Cu", "H2S", "Cl-", "H2O", "H+", "e-"])
    >>> pc.basis("H2S", -6)
    >>> pc.basis("Cl-", -1)
    >>> pc.species(["CuCl", "CuCl2-", "CuCl3-2", "CuCl+",
    ...             "CuCl2", "CuCl3-", "CuCl4-2"])
    >>> pc.species(["chalcocite", "tenorite", "cuprite", "copper"], add=True)
    >>> bases = ["H2S", "HS-", "HSO4-", "SO4-2"]
    >>> m = pc.mosaic(bases, pH=[0, 12], Eh=[-1, 1], T=200)
    >>> a = m['A_species']
    >>> d = pc.diagram(a, lwd=2)
    >>> pc.diagram(m['A_bases'], add_to=d, col=4, col_names=4)

    Notes
    -----
    This is a port of the R CHNOSZ ``mosaic()`` function.  Affinities agree
    with R CHNOSZ to within its printing precision, and predominance
    matrices are identical.  There are three intentional differences:

    1. The ``sout`` optimization (pre-computing ``subcrt()`` values once and
       reusing them for every combination) is not available because
       pychnosz's ``affinity()`` does not accept a ``sout`` argument; the
       thermodynamic properties are recalculated for each combination
       instead.  The numerical results are unaffected.
    2. ``loga_aq`` is honoured whether ``bases`` is a flat list or a list of
       groups.  In R, the flat-list backward-compatibility path silently
       drops ``loga_aq``.
    3. Argument recall preserves ``stable`` and ``loga_aq``, and keeps the
       shape of ``A_bases`` from the original call.  In R, the stored
       arguments hold only ``bases`` and ``blend``, so recall silently
       reverts ``stable``/``loga_aq`` to their defaults and returns
       ``A.bases`` as a list of groups even when the first call returned a
       single affinity output.
    """
    from .affinity import affinity
    from .equilibrate import equilibrate
    from .species import species as species_func

    # Get the arguments for affinity() before doing anything else
    affinityargs = dict(kwargs)

    # Argument recall: if the first argument is the result from a previous
    # mosaic() calculation, just update the remaining arguments
    if isinstance(bases, dict) and bases.get('fun') == 'mosaic':
        bargs = dict(bases.get('args', {}))
        # We can only update arguments for affinity()
        bargs.update(affinityargs)
        recall_bases = bargs.pop('bases')
        return mosaic(recall_bases,
                      blend=bargs.pop('blend', blend),
                      stable=bargs.pop('stable', stable),
                      loga_aq=bargs.pop('loga_aq', loga_aq),
                      messages=messages, **bargs)

    if 'sout' in affinityargs:
        raise MosaicError("'sout' is not supported by pychnosz's affinity(), "
                          "so it cannot be used with mosaic()")

    # Backward compatibility: bases can be a flat sequence instead of a list of lists
    bases_was_flat = not _is_list_of_groups(bases)
    if bases_was_flat:
        bases = [list(bases)]
    else:
        bases = [list(group) for group in bases]

    # Normalize 'stable' to a list with one (possibly None) entry per group
    if stable is None:
        stable = []
    else:
        stable = list(stable)

    if len(stable) == 2:
        # Use only predominant basis species for mosaic stacking
        stable2_orig = np.asarray(stable[1])
        stable2 = np.array(stable2_orig, dtype=float)
        # The first basis species should always be included
        # (because it has to be swapped out for the others)
        flat2 = np.concatenate(([1.0], stable2_orig.ravel().astype(float)))
        # R's sort() drops NA, which can appear where no species predominates
        istable2 = np.sort(np.unique(flat2[~np.isnan(flat2)]))
        for i, val in enumerate(istable2):
            stable2[stable2_orig == val] = i + 1
        bases[1] = [bases[1][int(k) - 1] for k in istable2]
        stable[1] = stable2

    # Save starting basis and species definition
    basis0 = get_basis()
    species0 = get_species()
    if basis0 is None:
        raise MosaicError("basis species are not defined")
    if species0 is None:
        raise MosaicError("species are not defined")
    basis0 = basis0.copy()
    species0 = species0.copy()

    # Get species indices of requested basis species
    ispecies = []
    for group in bases:
        idx = info(group, messages=False)
        if not isinstance(idx, list):
            idx = [idx]
        ispecies.append(idx)
    if any(pd.isna(i) for group in ispecies for i in group):
        raise MosaicError("one or more of the requested basis species is unavailable")

    # Identify starting basis species
    basis0_ispecies = list(basis0['ispecies'])
    ispecies0 = [group[0] for group in ispecies]
    ibasis0 = []
    for isp in ispecies0:
        ibasis0.append(basis0_ispecies.index(isp) if isp in basis0_ispecies else None)

    # Quit if starting basis species are not present
    if any(ib is None for ib in ibasis0):
        names0 = [group[0] for group, ib in zip(bases, ibasis0) if ib is None]
        raise MosaicError("the starting basis species do not have " + " and ".join(names0))

    obigt = thermo().obigt

    # Calculate affinities of the basis species themselves
    A_bases = []
    for i, group in enumerate(bases):
        if messages:
            print(f"mosaic: calculating affinities of basis species group {i + 1}: "
                  f"{' '.join(group)}")
        species_func(delete=True)
        mysp = species_func(group, messages=False)
        # Include only aq species in total activity
        is_aq = [state == "aq" for state in mysp['state']]
        # Use float() in case a buffer is active
        if any(is_aq):
            iaq = [j + 1 for j, aq in enumerate(is_aq) if aq]
            species_func(iaq, float(basis0['logact'].iloc[ibasis0[i]]), messages=False)
        A_bases.append(affinity(messages=False, **affinityargs))

    # Get all combinations of basis species (species indices in OBIGT)
    ind_mat = _expand_grid_indices([len(group) for group in ispecies])
    ncomb = len(ind_mat)

    allbases = []
    allbnames = []
    for row in ind_mat:
        thisbases = list(basis0_ispecies)
        thisbnames = list(basis0.index)
        for igroup, k in enumerate(row):
            thisbases[ibasis0[igroup]] = ispecies[igroup][k]
            thisbnames[ibasis0[igroup]] = bases[igroup][k]
        allbases.append(thisbases)
        allbnames.append(thisbnames)

    # Look for argument names for affinity() in starting basis species
    # (i.e., basis species that are variables on the diagram)
    matches_bnames = [name in allbnames[0] for name in affinityargs]
    argnames = list(affinityargs)
    ibnames = [allbnames[0].index(name)
               for name, m in zip(argnames, matches_bnames) if m]

    # Figure out the element to make labels (total C, total S, etc.)
    labels = None
    if any(matches_bnames):
        element_matrix = basis0.iloc[:, :len(basis0)]
        elements_in_basis0 = element_matrix.sum(axis=0)
        labelnames = [allbnames[0][ib] for ib in ibnames]
        labels = {}
        for name in labelnames:
            has_element = element_matrix.loc[name] > 0
            ielement = has_element & (elements_in_basis0 == 1)
            # Use the element or fallback to species name if element isn't found
            if ielement.any():
                labels[name] = str(element_matrix.columns[ielement][0])
            else:
                labels[name] = name

    # Calculate affinities of species for all combinations of basis species
    aff_species = [None] * ncomb
    if messages:
        print(f"mosaic: calculating affinities of species for all {ncomb} "
              "combinations of the basis species")
    # Run backwards so that we end up with the starting basis species
    for i in range(ncomb - 1, -1, -1):
        # Get default loga from starting basis species
        thislogact = [_as_float(la) for la in basis0['logact']]
        states = [obigt.loc[isp, 'state'] for isp in allbases[i]]
        # Use logact = 0 for solids
        for j, state in enumerate(states):
            if 'cr' in str(state):
                thislogact[j] = 0.0
        # Use loga_aq for log(activity) of mosaiced aqueous basis species
        if loga_aq is not None:
            if len(loga_aq) != len(ibasis0):
                raise MosaicError("'loga_aq' should have same length as 'bases'")
            for j, ib in enumerate(ibasis0):
                if loga_aq[j] is None or pd.isna(loga_aq[j]):
                    continue
                if 'aq' in str(states[ib]):
                    thislogact[ib] = float(loga_aq[j])
        put_basis(allbases[i], thislogact)
        # Load the formed species using the current basis
        species_func(delete=True)
        species_func(list(species0['ispecies']), list(species0['logact']), messages=False)

        # If mosaic() changes variables on the diagram, argument names for
        # affinity() also have to be changed
        myaffinityargs = dict(affinityargs)
        if any(matches_bnames):
            # At least one basis species in 'bases' is a variable on the diagram
            # Use the name of the current swapped-in basis species
            myaffinityargs = {}
            iname = 0
            for name, m in zip(argnames, matches_bnames):
                if m:
                    myaffinityargs[allbnames[i][ibnames[iname]]] = affinityargs[name]
                    iname += 1
                else:
                    myaffinityargs[name] = affinityargs[name]

        aff_species[i] = affinity(messages=False, **myaffinityargs)

    # Calculate equilibrium mole fractions for each group of basis species
    group_fraction = []
    blend_list = _rep(blend, len(A_bases))
    E_bases = []
    for i, A_base in enumerate(A_bases):
        this_stable = stable[i] if i < len(stable) else None
        if blend_list[i] and this_stable is None:
            base_values = _values_list(A_base)
            # This isn't needed (and doesn't work) if all the affinities are NA
            if not all(np.all(np.isnan(np.asarray(v, dtype=float))) for v in base_values):
                # When equilibrating the changing basis species, use a total activity
                # equal to the activity from the basis definition
                e = equilibrate(dict(A_base),
                                loga_balance=float(basis0['logact'].iloc[ibasis0[i]]),
                                messages=False)
                # Exponentiate to get activities then divide by total activity
                a_equil = [10.0 ** np.asarray(x, dtype=float) for x in e['loga_equil']]
                a_tot = np.sum(a_equil, axis=0)
                group_fraction.append([x / a_tot for x in a_equil])
                # Include the equilibrium activities in the output of this function
                E_bases.append(e)
            else:
                group_fraction.append(base_values)
        else:
            # For blend = False, we just look at whether a basis species
            # predominates within its group
            if this_stable is None:
                predom = _predominant(A_base, messages=False)
            else:
                # Get the stable species from the argument
                predom = this_stable
            predom = np.asarray(predom, dtype=float)
            # If a basis species predominates, it has a mole fraction of 1, or 0 otherwise
            group_fraction.append([(predom == (j + 1)).astype(float)
                                   for j in range(len(bases[i]))])

    # Loop over combinations of basis species
    for icomb, row in enumerate(ind_mat):
        aout = aff_species[icomb]
        keys = list(aout['species']['ispecies'])
        groupx = None
        # Loop over groups of changing basis species
        for igroup, k in enumerate(row):
            # Get mole fractions for this particular basis species
            basisx = np.asarray(group_fraction[igroup][k], dtype=float)
            # Loop over species
            with np.errstate(divide='ignore', invalid='ignore'):
                log_basisx = np.log10(basisx)
            for jspecies, key in enumerate(keys):
                # Get coefficient of this basis species in the formation reaction
                nbasis = aout['species'].iloc[jspecies, ibasis0[igroup]]
                # Adjust affinity of species for mole fractions
                # (i.e. lower activity) of basis species
                with np.errstate(invalid='ignore'):
                    aff_adjust = nbasis * log_basisx
                aff_adjust = np.asarray(aff_adjust, dtype=float)
                # Avoid infinite values (from log10(0))
                # np.where (rather than masked assignment) so that a 0-d value,
                # which a single-point grid produces, is handled like any other
                # shape.  R has no 0-d arrays, so there the mask is always valid.
                isfin = np.isfinite(aff_adjust)
                values = np.array(aout['values'][key], dtype=float, copy=True)
                with np.errstate(invalid='ignore'):
                    values = np.where(isfin, values + aff_adjust, values)
                aout['values'][key] = values
            # Multiply fractions of basis species from each group
            # to get overall fraction
            groupx = basisx if groupx is None else groupx * basisx
        # Multiply affinities by the mole fractions of basis species
        for key in keys:
            aout['values'][key] = aout['values'][key] * groupx

    # Get total affinities for the species
    A_species = dict(aff_species[0])
    A_species['values'] = {}
    for key in aff_species[0]['species']['ispecies']:
        # Sum the affinity contributions from each basis species
        A_species['values'][key] = np.sum(
            [np.asarray(aout['values'][key], dtype=float) for aout in aff_species],
            axis=0)

    # Insert custom labels
    A_species['labels'] = labels

    # Restore the starting basis and species definition
    put_basis(list(basis0['ispecies']), [_as_float(la) for la in basis0['logact']])
    species_func(delete=True)
    species_func(list(species0['ispecies']), list(species0['logact']), messages=False)

    # For argument recall, include all arguments in output
    allargs = {'bases': bases if not bases_was_flat else bases[0],
               'blend': blend, 'stable': stable, 'loga_aq': loga_aq}
    allargs.update(affinityargs)

    # Replace A_bases with a single affinity output for backwards compatibility
    A_bases_out = A_bases[0] if bases_was_flat else A_bases

    return {
        'fun': 'mosaic',
        'args': allargs,
        'A_species': A_species,
        'A.species': A_species,
        'A_bases': A_bases_out,
        'A.bases': A_bases_out,
        'E_bases': E_bases,
        'E.bases': E_bases,
    }


def _predominant(aout: Dict[str, Any], messages: bool = False) -> np.ndarray:
    """
    Find which species predominates at each point (maximum affinity method).

    This mirrors R CHNOSZ's ``which.pmax()`` applied to the balanced affinities,
    as diagram() does.  Unlike pychnosz's ``diagram()``, this works for any
    number of dimensions, which mosaic() needs for 1-D calculations.

    Returns a 1-based array of species positions, with NaN where any species
    has a missing affinity.
    """
    from .diagram import _get_balance

    n_balance, _ = _get_balance(aout, None, messages)
    values = [v / n for v, n in zip(_values_list(aout), n_balance)]

    stack = np.stack([np.atleast_1d(v) for v in values], axis=0)
    # Keep NAs out of the comparison, then mask them back in
    hasna = np.isnan(stack).any(axis=0)
    imax = np.nanargmax(np.where(np.isnan(stack), -np.inf, stack), axis=0) + 1.0
    imax[hasna] = np.nan
    return imax


def _is_list_of_groups(bases) -> bool:
    """Is 'bases' a list of groups (as opposed to a flat list of species)?"""
    if isinstance(bases, (str, bytes)):
        return False
    try:
        first = bases[0]
    except (TypeError, IndexError, KeyError):
        return False
    return isinstance(first, (list, tuple, np.ndarray, pd.Series))


def _expand_grid_indices(lengths: List[int]) -> List[List[int]]:
    """
    Enumerate index combinations in the order used by R's expand.grid().

    The first group varies fastest.  Returns 0-based indices.
    """
    total = int(np.prod(lengths)) if lengths else 0
    rows = []
    for r in range(total):
        rem = r
        row = []
        for n in lengths:
            row.append(rem % n)
            rem //= n
        rows.append(row)
    return rows


def _rep(x, length: int) -> List:
    """Recycle a scalar or sequence to the requested length (like R's rep())."""
    if isinstance(x, (list, tuple, np.ndarray)):
        x = list(x)
        if len(x) == 0:
            raise MosaicError("cannot recycle an empty value")
        return [x[i % len(x)] for i in range(length)]
    return [x] * length


def _as_float(value) -> float:
    """Convert a basis logact to float, tolerating buffer names."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _values_list(aout: Dict[str, Any]) -> List[np.ndarray]:
    """Get affinity values as a list ordered like aout['species']."""
    values = aout['values']
    if isinstance(values, dict):
        return [np.asarray(values[key], dtype=float)
                for key in aout['species']['ispecies']]
    return [np.asarray(v, dtype=float) for v in values]


__all__ = ['mosaic', 'MosaicError']
