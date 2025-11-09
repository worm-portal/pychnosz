"""
Core ThermoSystem class for managing global thermodynamic state.

This class manages the global thermodynamic system state, similar to the
'thermo' object in the R version of CHNOSZ.
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
from pathlib import Path

from ..data.loader import DataLoader
from ..data.obigt import OBIGTDatabase


class ThermoSystem:
    """
    Global thermodynamic system manager for CHNOSZ.
    
    This class manages the thermodynamic database, basis species, 
    formed species, and calculation options - essentially serving
    as the global state container for all CHNOSZ calculations.
    """
    
    def __init__(self):
        """Initialize the thermodynamic system."""
        self._data_loader = DataLoader()
        self._obigt_db = None
        self._initialized = False
        
        # Core data containers (similar to R thermo object)
        self.opt: Dict[str, Any] = {}
        self.element: Optional[pd.DataFrame] = None
        self.obigt: Optional[pd.DataFrame] = None
        self.refs: Optional[pd.DataFrame] = None
        self.Berman: Optional[pd.DataFrame] = None
        self.buffer: Optional[pd.DataFrame] = None
        self.protein: Optional[pd.DataFrame] = None
        self.groups: Optional[pd.DataFrame] = None
        self.stoich: Optional[np.ndarray] = None
        self.stoich_formulas: Optional[np.ndarray] = None
        self.bdot_acirc: Optional[Dict[str, float]] = None
        self.formula_ox: Optional[pd.DataFrame] = None
        
        # System state
        self.basis: Optional[pd.DataFrame] = None
        self.species: Optional[pd.DataFrame] = None
        
        # Options and parameters
        self.opar: Dict[str, Any] = {}
        
    def reset(self, messages: bool = True) -> None:
        """
        Initialize/reset the thermodynamic system.

        This is equivalent to reset() in the R version, loading all
        the thermodynamic data and initializing the system.

        Parameters
        ----------
        messages : bool, default True
            Whether to print informational messages
        """
        try:
            # Load core data files
            self._load_options(messages)
            self._load_element_data(messages)
            self._load_berman_data(messages)
            self._load_buffer_data(messages)
            self._load_protein_data(messages)
            self._load_stoich_data(messages)
            self._load_bdot_data(messages)
            self._load_refs_data(messages)

            # Initialize OBIGT database
            self._obigt_db = OBIGTDatabase()
            self.obigt = self._obigt_db.get_combined_data()

            # Reset system state
            self.basis = None
            self.species = None
            self.opar = {}

            self._initialized = True
            if messages:
                print('reset: thermodynamic system initialized')

        except Exception as e:
            raise RuntimeError(f"Failed to initialize thermodynamic system: {e}")
    
    def _load_options(self, messages: bool = True) -> None:
        """Load default thermodynamic options."""
        try:
            opt_file = self._data_loader.get_data_path() / "thermo" / "opt.csv"
            if opt_file.exists():
                df = pd.read_csv(opt_file)
                # Convert to dictionary format (first row contains values)
                self.opt = dict(zip(df.columns, df.iloc[0]))
            else:
                # Default options if file not found
                self.opt = {
                    'E.units': 'J',
                    'T.units': 'C',
                    'P.units': 'bar',
                    'state': 'aq',
                    'water': 'SUPCRT92',
                    'G.tol': 100,
                    'Cp.tol': 1,
                    'V.tol': 1,
                    'varP': False,
                    'IAPWS.sat': 'liquid',
                    'paramin': 1000,
                    'ideal.H': True,
                    'ideal.e': True,
                    'nonideal': 'Bdot',
                    'Setchenow': 'bgamma0',
                    'Berman': np.nan,
                    'maxcores': 2,
                    'ionize.aa': True
                }
        except Exception as e:
            if messages:
                print(f"Warning: Could not load options: {e}")
            # Fallback to hardcoded defaults with critical unit options
            self.opt = {
                'E.units': 'J',
                'T.units': 'C',
                'P.units': 'bar',
                'state': 'aq',
                'water': 'SUPCRT92',
                'G.tol': 100,
                'Cp.tol': 1,
                'V.tol': 1,
                'varP': False,
                'IAPWS.sat': 'liquid',
                'paramin': 1000,
                'ideal.H': True,
                'ideal.e': True,
                'nonideal': 'Bdot',
                'Setchenow': 'bgamma0',
                'Berman': np.nan,
                'maxcores': 2,
                'ionize.aa': True
            }
    
    def _load_element_data(self, messages: bool = True) -> None:
        """Load element properties data."""
        try:
            self.element = self._data_loader.load_elements()
        except Exception as e:
            if messages:
                print(f"Warning: Could not load element data: {e}")
            self.element = None
    
    def _load_berman_data(self, messages: bool = True) -> None:
        """Load Berman mineral parameters from CSV files."""
        try:
            # Get path to Berman directory
            berman_path = self._data_loader.data_path / "Berman"

            if not berman_path.exists():
                if messages:
                    print(f"Warning: Berman directory not found: {berman_path}")
                self.Berman = None
                return

            # Find all CSV files in the directory
            csv_files = list(berman_path.glob("*.csv"))

            if not csv_files:
                if messages:
                    print(f"Warning: No CSV files found in {berman_path}")
                self.Berman = None
                return
            
            # Extract year from filename and sort in reverse chronological order (youngest first)
            # Following R logic: files <- rev(files[order(sapply(strsplit(files, "_"), "[", 2))])
            def extract_year(filepath):
                filename = filepath.name
                parts = filename.split('_')
                if len(parts) >= 2:
                    year_part = parts[1].replace('.csv', '')
                    try:
                        return int(year_part)
                    except ValueError:
                        return 0
                return 0
            
            # Sort files by year (youngest first)
            sorted_files = sorted(csv_files, key=extract_year, reverse=True)
            
            # Read parameters from each file
            berman_dfs = []
            for file_path in sorted_files:
                try:
                    df = pd.read_csv(file_path)
                    berman_dfs.append(df)
                except Exception as e:
                    print(f"Warning: Could not read Berman file {file_path}: {e}")
            
            # Combine all data frames (equivalent to do.call(rbind, Berman))
            if berman_dfs:
                self.Berman = pd.concat(berman_dfs, ignore_index=True)
                # Ensure all numeric columns are properly typed
                numeric_cols = ['GfPrTr', 'HfPrTr', 'SPrTr', 'VPrTr', 'k0', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6',
                               'v1', 'v2', 'v3', 'v4', 'Tlambda', 'Tref', 'dTdP', 'l1', 'l2', 'DtH', 'Tmax', 'Tmin',
                               'd0', 'd1', 'd2', 'd3', 'd4', 'Vad']
                for col in numeric_cols:
                    if col in self.Berman.columns:
                        self.Berman[col] = pd.to_numeric(self.Berman[col], errors='coerce')
            else:
                self.Berman = None
                
        except Exception as e:
            if messages:
                print(f"Warning: Could not load Berman data: {e}")
            self.Berman = None

    def _load_buffer_data(self, messages: bool = True) -> None:
        """Load buffer definitions."""
        try:
            self.buffer = self._data_loader.load_buffers()
        except Exception as e:
            if messages:
                print(f"Warning: Could not load buffer data: {e}")
            self.buffer = None

    def _load_protein_data(self, messages: bool = True) -> None:
        """Load protein composition data.""" 
        try:
            self.protein = self._data_loader.load_proteins()
        except Exception as e:
            if messages:
                print(f"Warning: Could not load protein data: {e}")
            self.protein = None

    def _load_stoich_data(self, messages: bool = True) -> None:
        """Load stoichiometric matrix data."""
        try:
            stoich_df = self._data_loader.load_stoich()
            if stoich_df is not None:
                # Extract formulas and convert to matrix
                self.stoich_formulas = stoich_df.iloc[:, 0].values
                self.stoich = stoich_df.iloc[:, 1:].values
            else:
                self.stoich_formulas = None
                self.stoich = None
        except Exception as e:
            if messages:
                print(f"Warning: Could not load stoichiometric data: {e}")
            self.stoich_formulas = None
            self.stoich = None

    def _load_bdot_data(self, messages: bool = True) -> None:
        """Load B-dot activity coefficient parameters."""
        try:
            bdot_file = self._data_loader.get_data_path() / "thermo" / "Bdot_acirc.csv"
            if bdot_file.exists():
                df = pd.read_csv(bdot_file)
                if len(df.columns) >= 2:
                    self.bdot_acirc = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
                else:
                    self.bdot_acirc = {}
            else:
                self.bdot_acirc = {}
        except Exception as e:
            if messages:
                print(f"Warning: Could not load B-dot data: {e}")
            self.bdot_acirc = {}

    def _load_refs_data(self, messages: bool = True) -> None:
        """Load references data."""
        try:
            self.refs = self._data_loader.load_refs()
        except Exception as e:
            if messages:
                print(f"Warning: Could not load refs data: {e}")
            self.refs = None
    
    def is_initialized(self) -> bool:
        """Check if the thermodynamic system is initialized."""
        return self._initialized
    
    def get_obigt_db(self) -> OBIGTDatabase:
        """Get the OBIGT database instance."""
        if not self._initialized:
            self.reset()
        return self._obigt_db
    
    def get_option(self, key: str, default: Any = None) -> Any:
        """Get a thermodynamic option value."""
        return self.opt.get(key, default)
    
    def set_option(self, key: str, value: Any) -> None:
        """Set a thermodynamic option value."""
        self.opt[key] = value
    
    def info(self) -> Dict[str, Any]:
        """Get information about the current thermodynamic system."""
        if not self._initialized:
            return {"status": "Not initialized"}
        
        info = {
            "status": "Initialized",
            "obigt_species": len(self.obigt) if self.obigt is not None else 0,
            "elements": len(self.element) if self.element is not None else 0,
            "berman_minerals": len(self.Berman) if self.Berman is not None else 0,
            "buffers": len(self.buffer) if self.buffer is not None else 0,
            "proteins": len(self.protein) if self.protein is not None else 0,
            "stoich_species": len(self.stoich_formulas) if self.stoich_formulas is not None else 0,
            "basis_species": len(self.basis) if self.basis is not None else 0,
            "formed_species": len(self.species) if self.species is not None else 0,
            "current_options": dict(self.opt)
        }
        return info
    
    def __repr__(self) -> str:
        """String representation of the thermodynamic system."""
        if not self._initialized:
            return "ThermoSystem(uninitialized)"

        info = self.info()
        return (f"ThermoSystem("
                f"obigt={info['obigt_species']} species, "
                f"basis={info['basis_species']}, "
                f"formed={info['formed_species']})")

    # R-style uppercase property aliases for compatibility
    @property
    def OBIGT(self):
        """Alias for obigt (R compatibility)."""
        # Auto-initialize if needed AND obigt is None (matches R behavior)
        if self.obigt is None and not self._initialized:
            self.reset(messages=True)
        return self.obigt

    @OBIGT.setter
    def OBIGT(self, value):
        """Setter for OBIGT (R compatibility)."""
        _set_obigt_data(self, value)


# Global instance (singleton pattern)
_thermo_system = None

def get_thermo_system() -> ThermoSystem:
    """Get the global thermodynamic system instance."""
    global _thermo_system
    if _thermo_system is None:
        _thermo_system = ThermoSystem()
    return _thermo_system

def _set_obigt_data(thermo_sys: ThermoSystem, obigt_df: pd.DataFrame) -> None:
    """
    Set OBIGT data with proper index normalization.

    This helper function ensures that when OBIGT is replaced, the DataFrame
    index is properly set to use 1-based indexing (matching R conventions).

    Parameters
    ----------
    thermo_sys : ThermoSystem
        The thermodynamic system object
    obigt_df : pd.DataFrame
        The new OBIGT DataFrame to set
    """
    # Make a copy to avoid modifying the original
    new_obigt = obigt_df.copy()

    # Ensure the index starts at 1 (R convention)
    # If the DataFrame has a default 0-based index, shift it to 1-based
    if new_obigt.index[0] == 0:
        new_obigt.index = new_obigt.index + 1

    # Set the OBIGT data
    thermo_sys.obigt = new_obigt

    # Try to load refs data if available
    # This matches R behavior where OBIGT() loads both OBIGT and refs
    try:
        refs_df = thermo_sys._data_loader.load_refs()
        thermo_sys.refs = refs_df
    except Exception:
        # If refs can't be loaded, just leave it as is
        pass


def thermo(*args, messages=True, **kwargs):
    """
    Access or modify the thermodynamic system data object.

    This function provides a convenient interface to get or set parts of the
    thermodynamic system, similar to R's par() function for graphics parameters.

    Parameters
    ----------
    *args : str or list of str
        Names of attributes to retrieve (e.g., "element", "opt$ideal.H")
        For nested access, use "$" notation (e.g., "opt$E.units")
        Special values:
        - "WORM": Load the WORM thermodynamic database (Python-exclusive feature)
    messages : bool, default True
        Whether to print informational messages during operations
    **kwargs : any
        Named arguments to set attributes (e.g., element=new_df, opt={'E.units': 'cal'})
        For nested attributes, use "$" in the name (e.g., **{"opt$ideal.H": False})

    Returns
    -------
    various
        - If no arguments: returns the ThermoSystem object
        - If single unnamed argument: returns the requested value
        - If multiple unnamed arguments: returns list of requested values
        - If named arguments: returns original values before modification

    Examples
    --------
    >>> import pychnosz
    >>> # Get the entire thermo object
    >>> ts = pychnosz.thermo()

    >>> # Get a specific attribute
    >>> elem = pychnosz.thermo("element")

    >>> # Get nested attribute
    >>> e_units = pychnosz.thermo("opt$E.units")

    >>> # Get multiple attributes
    >>> elem, buf = pychnosz.thermo("element", "buffer")

    >>> # Set an attribute
    >>> old_elem = pychnosz.thermo(element=new_element_df)

    >>> # Set nested attribute
    >>> old_units = pychnosz.thermo(**{"opt$ideal.H": False})

    >>> # Load WORM database (Python-exclusive feature)
    >>> pychnosz.thermo("WORM")

    >>> # Suppress messages
    >>> pychnosz.thermo("WORM", messages=False)

    Notes
    -----
    This function mimics the behavior of R CHNOSZ thermo() function,
    providing flexible access to the thermodynamic data object.

    The "WORM" special argument is a Python-exclusive feature that loads
    the Water-Organic-Rock-Microbe thermodynamic database from the
    WORM-db GitHub repository.
    """
    # Get the global thermo system
    thermo_sys = get_thermo_system()

    # If no arguments, return the entire object
    if len(args) == 0 and len(kwargs) == 0:
        return thermo_sys

    # Handle character vectors passed as args (like R's c("basis", "species"))
    # If all args are strings or lists of strings, flatten them
    flat_args = []
    for arg in args:
        if isinstance(arg, (list, tuple)) and all(isinstance(x, str) for x in arg):
            flat_args.extend(arg)
        else:
            flat_args.append(arg)
    args = flat_args

    # Prepare return values list
    return_values = []

    # Ensure system is initialized if needed (before accessing any properties)
    # This prevents auto-initialization from using hardcoded messages=True
    if not thermo_sys.is_initialized() and len(args) > 0:
        thermo_sys.reset(messages=messages)

    # Process unnamed arguments (getters)
    for arg in args:
        if not isinstance(arg, str):
            raise TypeError(f"Unnamed arguments must be strings, got {type(arg)}")

        # Special handling for "WORM" - load WORM database
        if arg.upper() == "WORM":
            from ..data.worm import load_WORM
            success = load_WORM(keep_default=False, messages=messages)
            return_values.append(success)
            continue

        # Parse the argument to get slots (handle nested access with $)
        slots = arg.split('$')

        # Get the value from thermo_sys
        value = thermo_sys
        for slot in slots:
            # Handle OBIGT case-insensitively (R uses uppercase, Python uses lowercase)
            slot_lower = slot.lower()
            if hasattr(value, slot_lower):
                value = getattr(value, slot_lower)
            elif hasattr(value, slot):
                value = getattr(value, slot)
            elif isinstance(value, dict) and slot in value:
                value = value[slot]
            else:
                raise AttributeError(f"Attribute '{arg}' not found in thermo object")

        return_values.append(value)

    # Process named arguments (setters)
    setter_returns = {}

    # Ensure system is initialized if needed (before setting any properties)
    if not thermo_sys.is_initialized() and len(kwargs) > 0:
        thermo_sys.reset(messages=messages)

    for key, new_value in kwargs.items():
        # Parse the key to get slots
        slots = key.split('$')

        # Get the original value before modification
        orig_value = thermo_sys
        for slot in slots:
            # Handle case-insensitive attribute access (for OBIGT, etc.)
            slot_lower = slot.lower()
            if hasattr(orig_value, slot_lower):
                orig_value = getattr(orig_value, slot_lower)
            elif hasattr(orig_value, slot):
                orig_value = getattr(orig_value, slot)
            elif isinstance(orig_value, dict) and slot in orig_value:
                orig_value = orig_value[slot]
            else:
                raise AttributeError(f"Attribute '{key}' not found in thermo object")

        setter_returns[key] = orig_value

        # Set the new value
        if len(slots) == 1:
            # Direct attribute
            # Special handling for OBIGT - normalize index and handle refs
            if slots[0].upper() == 'OBIGT':
                # Handle OBIGT replacement with proper index normalization
                _set_obigt_data(thermo_sys, new_value)
            else:
                # Use lowercase version if it exists (Python convention)
                slot_lower = slots[0].lower()
                if hasattr(thermo_sys, slot_lower):
                    setattr(thermo_sys, slot_lower, new_value)
                else:
                    setattr(thermo_sys, slots[0], new_value)
        elif len(slots) == 2:
            # Nested attribute (e.g., opt$ideal.H)
            parent = getattr(thermo_sys, slots[0])
            if isinstance(parent, dict):
                parent[slots[1]] = new_value
            else:
                setattr(parent, slots[1], new_value)
        else:
            # Deeper nesting (if needed)
            current = thermo_sys
            for i, slot in enumerate(slots[:-1]):
                if hasattr(current, slot):
                    current = getattr(current, slot)
                elif isinstance(current, dict) and slot in current:
                    current = current[slot]

            # Set the final value
            final_slot = slots[-1]
            if isinstance(current, dict):
                current[final_slot] = new_value
            else:
                setattr(current, final_slot, new_value)

    # Determine return value based on R's behavior
    if len(kwargs) > 0:
        # If we had setters, return the original values as a named dict
        # In R, setters always return a named list
        if len(args) == 0:
            # Only setters - return dict (named list in R)
            return setter_returns
        else:
            # Mix of getters and setters - return all original values
            combined = {}
            for i, arg in enumerate(args):
                combined[arg] = return_values[i]
            combined.update(setter_returns)
            return combined
    else:
        # Only getters
        # Single unnamed argument returns the value directly
        if len(return_values) == 1:
            return return_values[0]
        return return_values