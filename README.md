# pyCHNOSZ

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11406142.svg)](https://doi.org/10.5281/zenodo.11406142)

Author: Dr. Grayson Boyer, GEOPIG Lab, Arizona State University

[CHNOSZ](https://www.chnosz.net/) is a package written by [Dr. Jeff Dick](https://chnosz.net/jeff/) for performing thermodynamic calculations in aqueous geochemistry and biogeochemistry. pyCHNOSZ is a Python conversion of CHNOSZ that allows these calculations to be carried out in Python 3 Jupyter notebooks.

## Features

The following CHNOSZ functions are supported in pyCHNOSZ:

- [info](https://chnosz.net/manual/info.html) - Search for chemical species by name or formula and retrieve their thermodynamic parameters.
- [add_OBIGT](https://chnosz.net/manual/add.OBIGT.html) - Add or overwrite species in the OBIGT thermodynamic database by supplying a comma separated value (csv) file with custom data.
- [mod_OBIGT](https://chnosz.net/manual/add.OBIGT.html) - Modify species in the OBIGT thermodynamic database. Optionally, supply a Pandas dataframe containing custom data.
- [reset](https://chnosz.net/manual/thermo.html) - reset data to default values.
- [subcrt](https://chnosz.net/manual/subcrt.html) - Calculate standard state partial molal thermodynamic properties of reactions at elevated temperatures and pressures.
- [basis](https://chnosz.net/manual/basis.html) - Define basis species of a chemical system.
- [species](https://chnosz.net/manual/species.html) - Define the species of interest in a system.
- [equilibrate](https://chnosz.net/manual/equilibrate.html) - Calculate equilibrium chemical activities of species from the affinities of formation of the species at unit activity.
- [affinity](https://chnosz.net/manual/affinity.html) - Calculate the chemical affinities of formation reactions of species.
- [diagram](https://chnosz.net/manual/diagram.html) - Plot equilibrium chemical activity (1-D speciation) or equal-activity (2-D predominance) diagrams as a function of chemical activities of basis species, temperature and/or pressure.
- And more. See the [documentation for pyCHNOSZ](https://worm-portal.asu.edu/pyCHNOSZ-docs/fun.html)


## Requirements

This package must be installed into an environment with an [R](https://www.r-project.org/) installation. See [these instructions](https://docs.anaconda.com/anaconda/user-guide/tasks/using-r-language/) for installing R with Anaconda. Additionally, the CHNOSZ package for R must be installed (see instructions below).

## Installation

### Installing pyCHNOSZ

Install pyCHNOSZ using pip:

```
$ pip install pychnosz
```

## Usage

Import pyCHNOSZ in your python code with:
```python
import pychnosz
```

In the following examples, pyCHNOSZ functions are imported with:
```python
from pychnosz import info, add_OBIGT, mod_OBIGT, reset, subcrt
```

### Search for chemical species

The `info()` function can be used to look up chemical species by name or formula. If names or formulas are provided, database index integers are returned. A list of integers will look up chemical species by index and return a table of thermodynamic properties. See the `info()` function's [original documentation](https://chnosz.net/manual/info.html) to learn more about what this function can do. A few examples are shown below.

Look up the database index value of Fe+2:

```python
info("Fe+2")
```

Look up multiple chemical species:

```python
info(["HCO3-", "CH4"])
```

Define chemical states:

```python
info(["HCO3-", "CH4"], state=["aq", "gas"])
```

Search species by index values to look up their thermodynamic parameters.

```python
info([13, 872])
```

Nest `info` functions to look up thermodynamic properties directly from names or formulas:

```python
info(info("Fe+2"))
```

### Add or replace thermodynamic data in the database

See the original R documentation for `add_OBIGT()` and `reset()` for basic useage. A few examples are given below.

Load the SUPCRT92 database.

```python
a = add_OBIGT("SUPCRT92")
```

The variable `a` is assigned a list of integers corresponding to the indices of chemical species that are added or replaced in the OBIGT thermodynamic database used by pyCHNOSZ.

It is possible to add a custom table of thermodynamic parameters.

```python
a = add_OBIGT("my_custom_entries.csv")
info(a) # confirm new entries have been added
```

The entries of `my_custom_entries.csv` would then be available for thermodynamic calculations with `subcrt()`, for example.

Reset thermodynamic database to its original configuration.

```python
reset()
```

Modify values in the thermodynamic database with `mod_OBIGT()`:

```python
mod_OBIGT("HCO3-", G = -140283.7, Cp = -9)
info(info("HCO3-"))
```

### Calculate thermodynamic properties of reactions

See the [original documentation](https://chnosz.net/manual/subcrt.html) for `subcrt()`. Useage in pyCHNOSZ is the same, except python lists are used in place of R's vectors. The function produces a dictionary of results stored in pandas dataframes. An example is shown below for the reaction H<sub>2 (aq)</sub> + 0.5 O<sub>2 (gas)</sub> = H<sub>2</sub>O<sub>(liq)</sub> at 30 and 50 degrees C and 100 bars pressure:

```python
subcrt(species=["H2", "O2", "H2O"], coeff=[-1.0, -0.5, 1.0],
       state=["aq", "gas", "liq"], T=[30, 50], P=100)
```

Output is a python dictionary of dataframes:
```
subcrt: 3 species at 2 values of T (ºC) and P (bar) (wet) [energy units: cal]

{'reaction':       coeff    name formula state  ispecies
 62     -1.0      H2      H2    aq      62.0
 2612   -0.5  oxygen      O2   gas    2612.0
 1       1.0   water     H2O   liq       1.0,
 'out':       T    P       rho       logK             G             H         S  \
 1  30.0  100  1.000017  43.855086 -60832.380282 -67420.887872 -21.89070   
 2  50.0  100  0.992305  40.834419 -60379.262657 -67882.530994 -23.36663   
 
           V         Cp  
 1 -7.494052 -24.126268  
 2 -8.259704 -20.941879  }
```

### More examples:

For more examples, check out the [pyCHNOSZ tutorial notebooks available from the WORM Library](https://github.com/worm-portal/WORM-Library/tree/master/2-Reaction-Properties).
