# CyMeP (Cyclone Metrics Package)
## CyMeP-Huracanpy
For the original CyMeP, see the [github repository](https://github.com/zarzycki/cymep) and [paper](https://doi.org/10.1175/JAMC-D-20-0149.1)

## General workflow
1. Create a YAML configuration file
2. Run cymep
3. Run cymep-graphics

## 0. Install cymep and dependencies

```bash
pip install .
```

## 1. Create a YAML configuration file
The easiest way to do this is to copy and modify `examples/example_config.yaml` which
contains all the options as well as descriptions. The options can be split up into two
parts, shown below. You can also run cymep on the example data by running the commands
in the examples folder.

### 1.1 Configure cymep options
The first part of the yaml file has options for how cymep performs the analysis

```yaml
# A tag to identify data and figures output by cymep using this configuration
filename_out: "rean_configs"

# Specify particular ocean basin or hemisphere
# Basins - NATL, ENP, CP, WNP, NI, MED, SI, AUS, SP, SA, SH
# Use "N" or "S" to specify hemisphere
# Leave empty for global
basin: "NATL"

# Length of side of each square gridbox used for spatial analysis in degrees
gridsize: 8.0

# Extra region around the gridbox used for spatial analysis if focused on a single basin
grid_buffer: 10.0

# Start and end year for overlapping interannual correlation analysis
styr: 1980
enyr: 2019

# Filter out years external to styr and enyr for all analysis?
# If False keep all data
truncate_years: False

# First and last month to include in climatology
# If enmon < stmon, it will overlap December/January (e.g. 11, 12, 1, 2)
stmon: 1
enmon: 12

# Threshold wind (in m/s) for ACE calculations.
# Leave empty for no threshold
THRESHOLD_ACE_WIND:
  
# Threshold pressure (in hPa) for PACE calculations.
# Leave empty for no threshold
THRESHOLD_PACE_PRES:

# Apply a wind-speed threshold of 17.5 to the reference set of tracks? (the first entry
# in datasets).
# False for no filter
do_special_filter_obs: True

# Fill missing data with observed pressure-wind curve?
# False leaves data as missing
do_fill_missing_pw: True

# Define maximum intensity location by minimum PSL?
# False uses maximum wind
do_defineMIbypres: False
```

### 1.2 Add dataset configurations
The second part of the yaml file is for specifying the datasets you want to analyse and
how to find/load them

```yaml
# The directory containing your tracks. Can be absolute or relative to where cymep is
# run
path_to_data: "trajs/"

# Units of SLP data on tracks
# cymep works with hPa, so specify "Pa" to divide by 100 when loaded or "hPa" for no
# change
slp_units: "Pa"

# Keywords passed to huracanpy.load() for all datasets
# Can also be specified for each dataset if they require additional keywords
load_keywords:
  tracker: tempestextremes
  tempest_extremes_unstructured: False
  variable_names:
    - slp
    - wind
    - unknown

# Specify the track data to apply the analysis to
# datasets is a dictionary mapping a name (used in the output files) for each dataset to
# a dictionary of options for each dataset
# - filename:       The file containing the set of tracks
# - load_keywords:  Any additional keywords that need to be passed to huracanpy.load()
#                   for this dataset
# - ensmembers:     Number of ensemble members included in the dataset
# - yearspermember: Number of years per ensemble member in the datasets
# - windcorrs:      Wind speed correction factor
datasets:
  OBS:
    filename: ibtracs-1980-2019-GLOB.v4.txt
    load_keywords: {}
    ensmembers: 1
    yearspermember: 40
    windcorrs: 1.0

  ERAI:
    filename: trajectories.txt.ERAI
    load_keywords: {}
    ensmembers: 1
    yearspermember: 39
    windcorrs: 1.0

...
```

**NOTE**: The first entry in datasets will be defined as the reference, so this should *always* be either observations or some sort of model/configuration control.

**NOTE**: The wind speed correction factor is a multiplier on the "wind" variable in the trajectory file to normalized from lowest model level to some reference height (e.g., lowest model level to 10m winds for TCs).

## 2. Run cymep

```bash
$> cymep config.yaml
```

This will produce a handful of netCDF files in `cymep-data/`.
- `"diags_{filename_out}_{basin}.nc"` The main set of output diagnostics
- `"means_{filename_out}_{basin}.nc"` Climatologies computed from the reference dataset
- `"storms_{filename_out}_{basin}_{dataset}.nc"` One file per dataset with metrics for each individual storm in the dataset

## 3. Run cymep-graphics

```bash
$> cymep-graphics config.yaml
```

This will produce a suite of figures in various subfolders within `cymep-figs/`.
- `line/`
- `spatial/`
- `tables/`
- `taylor/`
