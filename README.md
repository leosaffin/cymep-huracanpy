# CyMeP (Cyclone Metrics Package)

[![DOI](https://zenodo.org/badge/263738878.svg)](https://zenodo.org/badge/latestdoi/263738878)

[Zarzycki, C. M., Ullrich, P. A., and Reed, K. A. (2021). Metrics for evaluating tropical cyclones in climate data. *Journal of Applied Meteorology and Climatology*. doi: 10.1175/JAMC-D-20-0149.1.](https://doi.org/10.1175/JAMC-D-20-0149.1)

## General workflow
1. Create a YAML configuration file
2. Run cymep
3. Run cymep-graphics

## 0. Install cymep and dependencies

```bash
pip install .
```
**NOTE**: Currently will install the most recent revision of `huracanpy` directly from the master branch on github


## 1. Create a YAML configuration file
Example for `examples/example_config.yaml`

### 1.1 Configure options

```yaml
# cymep configuration
filename_out: "rean_configs"
basin: 1
gridsize: 8.0
styr: 1980
enyr: 2019
stmon: 1
enmon: 12
truncate_years: False
THRESHOLD_ACE_WIND: # wind speed (in m/s) to threshold ACE. None means off.
THRESHOLD_PACE_PRES:  # slp (in hPa) to threshold PACE. None means off.
do_special_filter_obs: True  # Special "if" block for first line (control)
do_fill_missing_pw: True
do_defineMIbypres: False
```
| Variable              | Value             | Description                                                                                                        |
|-----------------------|-------------------|--------------------------------------------------------------------------------------------------------------------|
| filename_out          | rean_configs      |                                                                                                                    |
| basin                 | 1                 | Set to negative to turn off filtering, otherwise specify particular ocean basin/hemisphere based on mask functions |
| gridsize              | 8.0               | Length of side of each square gridbox used for spatial analysis in degrees                                         |
| styr                  | 1980              | Start year for overlapping interannual correlation analysis                                                        |
| enyr                  | 2019              | End year for overlapping interannual correlation analysis                                                          |
| stmon                 | 1                 | First month to include in climatology                                                                              |
| enmon                 | 12                | Last month to include in climatology                                                                               |
| truncate_years        | False             | If `True` then filter out years external to styr and enyr. If `False` keep all data                                |
| THRESHOLD_ACE_WIND    | None              | Select threshold wind (in m/s) for ACE calculations (leave empty for no threshold)                                 |
| THRESHOLD_PACE_PRES   | None              | Select threshold SLP (in hPa) for PACE calculations (leave empty for no threshold)                                 |
| do_special_filter_obs | True              | Apply a wind-speed threshold of 17.5 to the reference set of tracks? (`False` for no filter)                       |
| do_fill_missing_pw    | True              | Fill missing data with observed pressure-wind curve? (`False` leaves data as missing)                              |
| do_defineMIbypres     | False             | Define maximum intensity location by PSL instead of wind? (`False` uses wind)                                      |

### 1.2 Add model configurations

```yaml
load_keywords:
  # Keywords passed to huracanpy.load(). Specified here for keywords used for every
  # model and individually for each model if they have additional keywords
  tracker: tempestextremes
  tempest_extremes_unstructured: False
  variable_names:
    - slp
    - wind
    - unknown
models:
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

| Variable       | Value                         | Description                                                                                                       |
|----------------|-------------------------------|-------------------------------------------------------------------------------------------------------------------|
|                | OBS                           | "Shortname" used for plotting, data output                                                                        | 
| filename       | ibtracs-1980-2019-GLOB.v4.txt | Trajectory file name                                                                                              |
| load_keyword   | Empty dictionary              | Keywords passed to [`huracanpy.load`](https://huracanpy.readthedocs.io/en/latest/api/loading.html#huracanpy.load) |
| ensmembers     | 1                             | Number of ensemble members included in trajectory file                                                            |
| yearspermember | 40                            | Number of years per ensemble member in trajectory file                                                            |
| windcorrs      | 1.0                           | Wind speed correction factor                                                                                      |

**NOTE**: The first entry will be defined as the reference, so this should *always* be either observations or some sort of model/configuration control.

**NOTE**: The wind speed correction factor is a multiplier on the "wind" variable in the trajectory file to normalized from lowest model level to some reference height (e.g., lowest model level to 10m winds for TCs).

## 2. Run cymep

```bash
$> cymep config.yaml
```

This will produce a handful of netCDF files in `cymep-data/`.
- `"diags_{filename_out}_{basin}.nc"` The main set of output diagnostics
- `"means_{filename_out}_{basin}.nc"` Climatologies computed from the reference model
- `"storms_{filename_out}_{basin}_{model}.nc"` One file per model with metrics for each individual storm in the dataset

## 3. Run cymep-graphics

```bash
$> cymep-graphics config.yaml
```

This will produce a suite of figures in various subfolders within `cymep-figs/`.
- `line/`
- `spatial/`
- `tables/`
- `taylor/`
