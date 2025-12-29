"""
pyCHNOSZ: Thermodynamic Calculations and Diagrams for Geochemistry

An integrated set of tools for thermodynamic calculations in aqueous geochemistry
and geobiochemistry. Functions are provided for writing balanced reactions to form
species from user-selected basis species and for calculating the standard molal
properties of species and reactions, including the standard Gibbs energy and
equilibrium constant.

Python port of the CHNOSZ package for R. The original CHNOSZ package belongs to Dr. Jeffrey Dick.
"""

__version__ = "1.1.11"
__author__ = "Grayson Boyer"
__email__ = "gmboyer@asu.edu"

# Import main classes and functions
from .core.thermo import ThermoSystem, thermo
from .core.basis import basis
from .core.species import species
from .core.info import info
from .core.retrieve import retrieve
from .models.water import water
from .utils.reset import reset

# Import equation of state functions
from .models.hkf import hkf, gfun
from .models.cgl import cgl, quartz_coesite
from .models.hkf_helpers import calc_logK, calc_G_TP, G2logK, dissrxn2logK, OBIGT2eos

# Import implemented functions
from .core.subcrt import subcrt
from .core.balance import balance_reaction, format_reaction
from .data.add_obigt import add_OBIGT, list_OBIGT_files, reset_OBIGT
from .data.mod_obigt import mod_OBIGT
from .data.worm import load_WORM, reset_WORM
from .models.berman import Berman
from .utils.formula import makeup, mass, entropy, ZC
from .utils.formula_ox import get_formula_ox, get_n_element_ox
from .utils.expression import ratlab, ratlab_html, expr_species, syslab, syslab_html, describe_property, describe_property_html, describe_basis, describe_basis_html, add_legend, set_title
from .utils.units import convert, envert

# Implemented functions
from .core.affinity import affinity
from .core.diagram import diagram, diagram_interactive, water_lines, find_tp, copy_plot
from .core.equilibrate import equilibrate
from .core.unicurve import unicurve, univariant_TP

# Optional: animation requires plotly (install with: pip install pychnosz[interactive])
try:
    from .core.animation import animation
except ImportError:
    # plotly not installed, create stub function with helpful error message
    def animation(*args, **kwargs):
        raise ImportError(
            "The 'animation' function requires plotly, which is not installed. "
            "Install it with: pip install pychnosz[interactive]"
        )

# Protein functions
from .biomolecules.proteins import pinfo, add_protein, protein_length, protein_formula, protein_OBIGT, protein_basis, group_formulas
from .biomolecules.ionize_aa import ionize_aa

__all__ = [
    'ThermoSystem',
    'thermo',
    'basis',
    'species',
    'info',
    'retrieve',
    'water',
    'reset',
    'subcrt',
    'balance_reaction',
    'format_reaction',
    'affinity',
    'diagram',
    'diagram_interactive',
    'water_lines',
    'find_tp',
    'copy_plot',
    'equilibrate',
    'animation',
    'unicurve',
    'univariant_TP',
    'add_OBIGT',
    'mod_OBIGT',
    'list_OBIGT_files',
    'reset_OBIGT',
    'load_WORM',
    'reset_WORM',
    'Berman',
    'makeup',
    'mass',
    'entropy',
    'ZC',
    'get_formula_ox',
    'get_n_element_ox',
    'ratlab',
    'ratlab_html',
    'expr_species',
    'syslab',
    'syslab_html',
    'describe_property',
    'describe_property_html',
    'describe_basis',
    'describe_basis_html',
    'add_legend',
    'set_title',
    'hkf',
    'gfun',
    'cgl',
    'quartz_coesite',
    'calc_logK',
    'calc_G_TP',
    'G2logK',
    'dissrxn2logK',
    'OBIGT2eos',
    'convert',
    'envert',
    'pinfo',
    'add_protein',
    'protein_length',
    'protein_formula',
    'protein_OBIGT',
    'protein_basis',
    'group_formulas',
    'ionize_aa'
]