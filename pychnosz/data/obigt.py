"""
OBIGT database access module.

This module provides a high-level interface to the OBIGT (Oelkers, Benezeth, 
and Isobaric Gas Thermodynamics) database, which contains thermodynamic 
parameters for chemical species.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from .loader import DataLoader


class OBIGTDatabase:
    """
    High-level interface to the OBIGT thermodynamic database.
    
    This class provides methods to access, search, and manipulate the
    thermodynamic data from the OBIGT database files.
    """
    
    def __init__(self, data_loader: Optional[DataLoader] = None):
        """
        Initialize the OBIGT database.
        
        Parameters:
        -----------
        data_loader : DataLoader, optional
            DataLoader instance to use. If None, creates a default loader.
        """
        if data_loader is None:
            from .loader import get_default_loader
            self.loader = get_default_loader()
        else:
            self.loader = data_loader
            
        # Cache for combined data
        self._combined_data = None
        self._species_index = None
        
        # Define the expected columns for OBIGT data
        self.obigt_columns = [
            'name', 'abbrv', 'formula', 'state', 'ref1', 'ref2', 'date', 'model',
            'E_units', 'G', 'H', 'S', 'Cp', 'V', 'a1.a', 'a2.b', 'a3.c', 'a4.d',
            'c1.e', 'c2.f', 'omega.lambda', 'z.T'
        ]
        
        # State classifications
        self.aqueous_states = ['aq']
        self.crystalline_states = ['cr']
        self.gas_states = ['gas']
        self.liquid_states = ['liq']
    
    def load_all_data(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load and combine all OBIGT data files.
        
        Parameters:
        -----------
        force_reload : bool, default False
            Force reloading of data even if cached
            
        Returns:
        --------
        pd.DataFrame
            Combined OBIGT database
        """
        if self._combined_data is not None and not force_reload:
            return self._combined_data.copy()
        
        # Load all OBIGT files
        obigt_files = self.loader.load_all_obigt_files()
        
        # Combine all files
        combined_data = []
        
        for filename, df in obigt_files.items():
            # Add source file information
            df_copy = df.copy()
            df_copy['source_file'] = filename
            combined_data.append(df_copy)
        
        # Concatenate all data
        self._combined_data = pd.concat(combined_data, ignore_index=True)

        # IMPORTANT: R uses 1-based indexing, so we need to shift the DataFrame index
        # to match R's row numbers. Row 0 in pandas should be row 1 in R.
        self._combined_data.index = self._combined_data.index + 1

        # Create species index for fast lookups
        self._create_species_index()
        
        return self._combined_data.copy()
    
    def get_combined_data(self) -> pd.DataFrame:
        """
        Get combined OBIGT thermodynamic data.
        
        Returns
        -------
        pd.DataFrame
            Combined OBIGT data with all species
        """
        if self._combined_data is not None:
            return self._combined_data.copy()
        
        try:
            # Try to load data normally first
            return self.load_all_data()
        except Exception as e:
            print(f"Warning: Could not load OBIGT data: {e}")
            # Create minimal fallback data for essential species
            return self._create_fallback_data()
    
    def _create_fallback_data(self) -> pd.DataFrame:
        """Create minimal fallback data for essential species."""
        
        # Essential species data (approximate values for basic functionality)
        fallback_data = {
            'name': ['water', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2'],
            'abbrv': ['H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2'],
            'formula': ['H2O', 'H+', 'OH-', 'CO2', 'HCO3-', 'CO3-2'],
            'state': ['liq', 'aq', 'aq', 'aq', 'aq', 'aq'],
            'G': [-56688.1, 0.0, -37595.0, -92307.0, -140314.0, -126172.0],
            'H': [-68317.0, 0.0, -54977.0, -98900.0, -165180.0, -161963.0],
            'S': [16.712, 0.0, -2.56, -39.75, 98.4, -50.0],
            'Cp': [18.0, 0.0, -36.4, 37.11, 25.0, -53.1],
            'V': [18.068, 0.0, -4.71, 34.0, 25.0, -6.0],
            'z.T': [0, 1, -1, 0, -1, -2],
            'ref1': ['', '', '', '', '', ''],
            'ref2': ['', '', '', '', '', ''],
            'date': ['', '', '', '', '', ''],
            'model': ['', '', '', '', '', ''],
            'E_units': ['', '', '', '', '', ''],
            'a1.a': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'a2.b': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'a3.c': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'a4.d': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'c1.e': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'c2.f': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            'omega.lambda': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
        
        df = pd.DataFrame(fallback_data)
        
        # Cache the fallback data
        self._combined_data = df
        self._create_species_index()
        
        return df.copy()
    
    def _create_species_index(self):
        """Create an index for fast species lookups."""
        if self._combined_data is None:
            return
            
        # Create multi-level index for name, formula, and state
        self._species_index = {}
        
        for idx, row in self._combined_data.iterrows():
            name = str(row.get('name', '')).strip()
            formula = str(row.get('formula', '')).strip()
            state = str(row.get('state', '')).strip()
            
            # Index by name
            if name and name not in self._species_index:
                self._species_index[name] = []
            if name:
                self._species_index[name].append(idx)
            
            # Index by formula
            formula_key = f"formula:{formula}"
            if formula and formula_key not in self._species_index:
                self._species_index[formula_key] = []
            if formula:
                self._species_index[formula_key].append(idx)
            
            # Index by name+state combination
            name_state_key = f"{name}({state})"
            if name and state and name_state_key not in self._species_index:
                self._species_index[name_state_key] = []
            if name and state:
                self._species_index[name_state_key].append(idx)
    
    def get_species(self, identifier: str, state: Optional[str] = None) -> pd.DataFrame:
        """
        Get species data by name, formula, or identifier.
        
        Parameters:
        -----------
        identifier : str
            Species name, formula, or identifier
        state : str, optional
            Physical state ('aq', 'cr', 'gas', 'liq')
            
        Returns:
        --------
        pd.DataFrame
            Matching species data
        """
        if self._combined_data is None:
            self.load_all_data()
        
        results = []
        
        # Try exact name match first
        if identifier in self._species_index:
            indices = self._species_index[identifier]
            for idx in indices:
                row = self._combined_data.iloc[idx]
                if state is None or str(row.get('state', '')).strip() == state:
                    results.append(row)
        
        # Try formula match
        formula_key = f"formula:{identifier}"
        if formula_key in self._species_index:
            indices = self._species_index[formula_key]
            for idx in indices:
                row = self._combined_data.iloc[idx]
                if state is None or str(row.get('state', '')).strip() == state:
                    results.append(row)
        
        # Try name+state combination
        if state:
            name_state_key = f"{identifier}({state})"
            if name_state_key in self._species_index:
                indices = self._species_index[name_state_key]
                for idx in indices:
                    results.append(self._combined_data.iloc[idx])
        
        # If no exact matches, try partial matching
        if not results:
            mask = self._combined_data['name'].str.contains(identifier, case=False, na=False) | \
                   self._combined_data['formula'].str.contains(identifier, case=False, na=False)
            
            if state:
                mask &= (self._combined_data['state'] == state)
            
            partial_matches = self._combined_data[mask]
            results = [row for _, row in partial_matches.iterrows()]
        
        if results:
            return pd.DataFrame(results).reset_index(drop=True)
        else:
            return pd.DataFrame(columns=self._combined_data.columns)
    
    def search_species(self, query: str, search_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Search for species using a text query.
        
        Parameters:
        -----------
        query : str
            Search query
        search_columns : List[str], optional
            Columns to search in. Default: ['name', 'formula', 'abbrv']
            
        Returns:
        --------
        pd.DataFrame
            Matching species data
        """
        if self._combined_data is None:
            self.load_all_data()
        
        if search_columns is None:
            search_columns = ['name', 'formula', 'abbrv']
        
        # Create search mask
        mask = pd.Series([False] * len(self._combined_data))
        
        for col in search_columns:
            if col in self._combined_data.columns:
                mask |= self._combined_data[col].str.contains(query, case=False, na=False)
        
        return self._combined_data[mask].reset_index(drop=True)
    
    def get_species_by_state(self, state: str) -> pd.DataFrame:
        """
        Get all species in a specific physical state.
        
        Parameters:
        -----------
        state : str
            Physical state ('aq', 'cr', 'gas', 'liq')
            
        Returns:
        --------
        pd.DataFrame
            Species data for the specified state
        """
        if self._combined_data is None:
            self.load_all_data()
        
        mask = self._combined_data['state'] == state
        return self._combined_data[mask].reset_index(drop=True)
    
    def get_aqueous_species(self) -> pd.DataFrame:
        """Get all aqueous species."""
        return self.get_species_by_state('aq')
    
    def get_crystalline_species(self) -> pd.DataFrame:
        """Get all crystalline species."""
        return self.get_species_by_state('cr')
    
    def get_gas_species(self) -> pd.DataFrame:
        """Get all gas species."""
        return self.get_species_by_state('gas')
    
    def get_liquid_species(self) -> pd.DataFrame:
        """Get all liquid species."""
        return self.get_species_by_state('liq')
    
    def get_species_by_elements(self, elements: List[str]) -> pd.DataFrame:
        """
        Get species containing specific elements.
        
        Parameters:
        -----------
        elements : List[str]
            List of element symbols
            
        Returns:
        --------
        pd.DataFrame
            Species containing the specified elements
        """
        if self._combined_data is None:
            self.load_all_data()
        
        # Create search pattern for elements
        pattern = '|'.join(elements)
        mask = self._combined_data['formula'].str.contains(pattern, case=False, na=False)
        
        return self._combined_data[mask].reset_index(drop=True)
    
    def get_thermodynamic_properties(self, species_data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract thermodynamic properties from species data.
        
        Parameters:
        -----------
        species_data : pd.DataFrame
            Species data from get_species or similar methods
            
        Returns:
        --------
        pd.DataFrame
            Thermodynamic properties (G, H, S, Cp, V, etc.)
        """
        thermo_columns = ['G', 'H', 'S', 'Cp', 'V', 'a1.a', 'a2.b', 'a3.c', 'a4.d', 
                         'c1.e', 'c2.f', 'omega.lambda', 'z.T']
        
        available_columns = [col for col in thermo_columns if col in species_data.columns]
        
        result = species_data[['name', 'formula', 'state'] + available_columns].copy()
        
        # Convert numeric columns to proper numeric types
        for col in available_columns:
            result[col] = pd.to_numeric(result[col], errors='coerce')
        
        return result
    
    def get_database_stats(self) -> Dict[str, Union[int, Dict[str, int]]]:
        """
        Get statistics about the database.
        
        Returns:
        --------
        Dict
            Database statistics including total species, states, etc.
        """
        if self._combined_data is None:
            self.load_all_data()
        
        stats = {
            'total_species': len(self._combined_data),
            'states': self._combined_data['state'].value_counts().to_dict(),
            'source_files': self._combined_data['source_file'].value_counts().to_dict(),
            'unique_names': self._combined_data['name'].nunique(),
            'unique_formulas': self._combined_data['formula'].nunique(),
        }
        
        return stats
    
    def validate_data(self) -> Dict[str, List]:
        """
        Validate the OBIGT database for common issues.
        
        Returns:
        --------
        Dict
            Validation results with issues found
        """
        if self._combined_data is None:
            self.load_all_data()
        
        issues = {
            'missing_names': [],
            'missing_formulas': [],
            'missing_states': [],
            'invalid_numeric_values': [],
            'duplicate_entries': []
        }
        
        # Check for missing critical fields
        missing_names = self._combined_data['name'].isna() | (self._combined_data['name'] == '')
        if missing_names.any():
            issues['missing_names'] = self._combined_data[missing_names].index.tolist()
        
        missing_formulas = self._combined_data['formula'].isna() | (self._combined_data['formula'] == '')
        if missing_formulas.any():
            issues['missing_formulas'] = self._combined_data[missing_formulas].index.tolist()
        
        missing_states = self._combined_data['state'].isna() | (self._combined_data['state'] == '')
        if missing_states.any():
            issues['missing_states'] = self._combined_data[missing_states].index.tolist()
        
        # Check for invalid numeric values in key thermodynamic properties
        numeric_columns = ['G', 'H', 'S', 'Cp']
        for col in numeric_columns:
            if col in self._combined_data.columns:
                numeric_data = pd.to_numeric(self._combined_data[col], errors='coerce')
                invalid_mask = numeric_data.isna() & self._combined_data[col].notna()
                if invalid_mask.any():
                    issues['invalid_numeric_values'].extend(
                        [(idx, col) for idx in self._combined_data[invalid_mask].index]
                    )
        
        # Check for potential duplicates
        duplicate_mask = self._combined_data.duplicated(subset=['name', 'formula', 'state'], keep=False)
        if duplicate_mask.any():
            issues['duplicate_entries'] = self._combined_data[duplicate_mask].index.tolist()
        
        return issues
    
    def export_to_csv(self, filename: str, species_filter: Optional[str] = None):
        """
        Export database or filtered data to CSV.
        
        Parameters:
        -----------
        filename : str
            Output filename
        species_filter : str, optional
            Filter to apply (state name like 'aq', 'cr', etc.)
        """
        if self._combined_data is None:
            self.load_all_data()
        
        data_to_export = self._combined_data
        
        if species_filter:
            if species_filter in ['aq', 'cr', 'gas', 'liq']:
                data_to_export = self.get_species_by_state(species_filter)
        
        data_to_export.to_csv(filename, index=False)


def get_default_obigt() -> OBIGTDatabase:
    """
    Get a default OBIGT database instance.
    
    Returns:
    --------
    OBIGTDatabase
        Default OBIGT database instance
    """
    return OBIGTDatabase()