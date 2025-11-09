"""
Data loader module for CHNOSZ thermodynamic database files.

This module provides utilities to load and manage the thermodynamic database
files from the R CHNOSZ package, converting them to pandas-compatible formats.
"""

import os
import pandas as pd
import lzma
import warnings
from pathlib import Path
from typing import Dict, Optional, Union, List


class DataLoader:
    """
    Main data loader class for CHNOSZ thermodynamic database files.
    
    This class handles loading of various data files from the CHNOSZ R package,
    including compressed files, and converts them to pandas DataFrames while
    preserving data integrity.
    """
    
    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        """
        Initialize the DataLoader.

        Parameters:
        -----------
        data_path : str or Path, optional
            Path to the CHNOSZ data directory. If None, will attempt to find
            the data/extdata directory relative to this file within the package.
        """
        if data_path is None:
            # Try to find the data directory relative to this file
            # We're now in pychnosz/data/, so extdata is in the same directory
            current_dir = Path(__file__).parent
            self.data_path = current_dir / "extdata"
        else:
            self.data_path = Path(data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_path}")

        self.obigt_path = self.data_path / "OBIGT"
        self.thermo_path = self.data_path / "thermo"

        # Cache for loaded data
        self._cache = {}
    
    def _read_csv_safe(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """
        Safely read a CSV file with appropriate error handling.
        
        Parameters:
        -----------
        filepath : Path
            Path to the CSV file
        **kwargs
            Additional arguments to pass to pd.read_csv
            
        Returns:
        --------
        pd.DataFrame
            Loaded DataFrame
        """
        try:
            # Handle potential encoding issues
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(filepath, encoding=encoding, **kwargs)
                    return df
                except UnicodeDecodeError:
                    continue
                    
            # If all encodings fail, try with error handling
            df = pd.read_csv(filepath, encoding='utf-8', errors='replace', **kwargs)
            warnings.warn(f"Used error replacement for file {filepath}")
            return df
            
        except Exception as e:
            raise IOError(f"Failed to read {filepath}: {str(e)}")
    
    def _read_compressed_csv(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """
        Read a compressed CSV file (e.g., .xz format).
        
        Parameters:
        -----------
        filepath : Path
            Path to the compressed CSV file
        **kwargs
            Additional arguments to pass to pd.read_csv
            
        Returns:
        --------
        pd.DataFrame
            Loaded DataFrame
        """
        if filepath.suffix == '.xz':
            with lzma.open(filepath, 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f, **kwargs)
                return df
        else:
            raise ValueError(f"Unsupported compression format: {filepath.suffix}")
    
    def load_obigt_file(self, filename: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Load a specific OBIGT database file.
        
        Parameters:
        -----------
        filename : str
            Name of the OBIGT file to load (e.g., 'inorganic_aq.csv')
        use_cache : bool, default True
            Whether to use cached data if available
            
        Returns:
        --------
        pd.DataFrame
            Loaded OBIGT data
        """
        cache_key = f"obigt_{filename}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        filepath = self.obigt_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"OBIGT file not found: {filepath}")
        
        # Load the data
        df = self._read_csv_safe(filepath)
        
        # Clean up column names (remove any whitespace)
        df.columns = df.columns.str.strip()
        
        # Cache the result
        if use_cache:
            self._cache[cache_key] = df.copy()
            
        return df
    
    def load_all_obigt_files(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Load all OBIGT database files in the same order as R CHNOSZ.
        
        This mirrors the exact loading order from R CHNOSZ/thermo.R OBIGT() function
        to ensure identical species indices between R and Python versions.
        
        Parameters:
        -----------
        use_cache : bool, default True
            Whether to use cached data if available
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary with filenames as keys and DataFrames as values, ordered like R CHNOSZ
        """
        obigt_files = {}
        
        if not self.obigt_path.exists():
            raise FileNotFoundError(f"OBIGT directory not found: {self.obigt_path}")
        
        # Use exact same order as R CHNOSZ (from thermo.R lines 63-67)
        # sources_aq <- paste0(c("H2O", "inorganic", "organic"), "_aq")
        # sources_cr <- paste0(c("Berman", "inorganic", "organic"), "_cr")
        # sources_liq <- paste0(c("organic"), "_liq")
        # sources_gas <- paste0(c("inorganic", "organic"), "_gas")
        # sources <- c(sources_aq, sources_cr, sources_gas, sources_liq)
        r_chnosz_order = [
            "H2O_aq.csv",
            "inorganic_aq.csv", 
            "organic_aq.csv",
            "Berman_cr.csv",
            "inorganic_cr.csv",
            "organic_cr.csv", 
            "inorganic_gas.csv",
            "organic_gas.csv",
            "organic_liq.csv"
        ]
        
        # Load files in R CHNOSZ order
        for filename in r_chnosz_order:
            file_path = self.obigt_path / filename
            if file_path.exists():
                obigt_files[filename] = self.load_obigt_file(filename, use_cache=use_cache)
            else:
                warnings.warn(f"OBIGT file not found: {filename}")
            
        return obigt_files
    
    def load_thermo_file(self, filename: str, use_cache: bool = True) -> pd.DataFrame:
        """
        Load a specific thermo database file.
        
        Parameters:
        -----------
        filename : str
            Name of the thermo file to load (e.g., 'element.csv', 'stoich.csv.xz')
        use_cache : bool, default True
            Whether to use cached data if available
            
        Returns:
        --------
        pd.DataFrame
            Loaded thermo data
        """
        cache_key = f"thermo_{filename}"
        
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        filepath = self.thermo_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Thermo file not found: {filepath}")
        
        # Handle compressed files
        if filepath.suffix == '.xz':
            df = self._read_compressed_csv(filepath)
        else:
            df = self._read_csv_safe(filepath)
        
        # Clean up column names
        df.columns = df.columns.str.strip()
        
        # Cache the result
        if use_cache:
            self._cache[cache_key] = df.copy()
            
        return df
    
    def load_elements(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load the elements data file.
        
        Parameters:
        -----------
        use_cache : bool, default True
            Whether to use cached data if available
            
        Returns:
        --------
        pd.DataFrame
            Elements data with columns: element, state, source, mass, s, n
        """
        return self.load_thermo_file('element.csv', use_cache=use_cache)
    
    def load_buffer(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load the buffer data file.
        
        Parameters:
        -----------
        use_cache : bool, default True
            Whether to use cached data if available
            
        Returns:
        --------
        pd.DataFrame
            Buffer data with columns: name, species, state, logact
        """
        return self.load_thermo_file('buffer.csv', use_cache=use_cache)
    
    def load_protein(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load the protein data file.
        
        Parameters:
        -----------
        use_cache : bool, default True
            Whether to use cached data if available
            
        Returns:
        --------
        pd.DataFrame
            Protein data with amino acid compositions
        """
        return self.load_thermo_file('protein.csv', use_cache=use_cache)
    
    def load_stoich(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load the stoichiometry data file (compressed).
        
        Parameters:
        -----------
        use_cache : bool, default True
            Whether to use cached data if available
            
        Returns:
        --------
        pd.DataFrame
            Stoichiometry matrix for all species
        """
        return self.load_thermo_file('stoich.csv.xz', use_cache=use_cache)
    
    def get_available_obigt_files(self) -> List[str]:
        """
        Get list of available OBIGT files.
        
        Returns:
        --------
        List[str]
            List of available OBIGT filenames
        """
        if not self.obigt_path.exists():
            return []
        
        return [f.name for f in self.obigt_path.glob("*.csv")]
    
    def get_available_thermo_files(self) -> List[str]:
        """
        Get list of available thermo files.
        
        Returns:
        --------
        List[str]
            List of available thermo filenames
        """
        if not self.thermo_path.exists():
            return []
        
        # Get both .csv and .csv.xz files
        csv_files = [f.name for f in self.thermo_path.glob("*.csv")]
        xz_files = [f.name for f in self.thermo_path.glob("*.csv.xz")]
        
        return sorted(csv_files + xz_files)
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache.clear()
    
    def get_cache_info(self) -> Dict[str, int]:
        """
        Get information about cached data.
        
        Returns:
        --------
        Dict[str, int]
            Dictionary with cache keys and DataFrame sizes
        """
        return {key: len(df) for key, df in self._cache.items()}
    
    def get_data_path(self) -> Path:
        """
        Get the data directory path.
        
        Returns
        -------
        Path
            Path to the data directory
        """
        return self.data_path
    
    def load_buffers(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load buffer data (alias for load_buffer for compatibility).
        
        Parameters
        ----------
        use_cache : bool, default True
            Whether to use cached data if available
            
        Returns
        -------
        pd.DataFrame
            Buffer data
        """
        try:
            return self.load_buffer(use_cache=use_cache)
        except Exception:
            # Return empty DataFrame if buffer data not available
            return pd.DataFrame(columns=['name', 'species', 'state', 'logact'])
    
    def load_proteins(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load protein data (alias for load_protein for compatibility).

        Parameters
        ----------
        use_cache : bool, default True
            Whether to use cached data if available

        Returns
        -------
        pd.DataFrame
            Protein data
        """
        try:
            return self.load_protein(use_cache=use_cache)
        except Exception:
            # Return empty DataFrame if protein data not available
            return pd.DataFrame(columns=['protein', 'organism', 'ref', 'abbrv', 'chains'])

    def load_refs(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load references data file.

        Parameters
        ----------
        use_cache : bool, default True
            Whether to use cached data if available

        Returns
        -------
        pd.DataFrame
            References data
        """
        try:
            return self.load_thermo_file('refs.csv', use_cache=use_cache)
        except Exception:
            # Return empty DataFrame if refs data not available
            return pd.DataFrame(columns=['key', 'author', 'year', 'citation'])


def get_default_loader() -> DataLoader:
    """
    Get a default DataLoader instance.
    
    Returns:
    --------
    DataLoader
        Default DataLoader instance
    """
    return DataLoader()