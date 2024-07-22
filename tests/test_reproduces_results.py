import pathlib

import numpy as np
import pandas as pd
import xarray as xr


old_names = dict(
    genesis_year="gYear",
    genesis_month="gMonth",
    genesis_day="gDay",
    genesis_hour="gHour",
    genesis_latitude="gLat",
    genesis_longitude="gLon",
    minimum_pressure="minPres",
    maximum_wind="maxWind",
    total_cyclone_days="TCD",
    accumulated_cyclone_energy="ACE",
    pressure_accumulated_cyclone_energy="PACE",
)


def test_all():
    # Spatial data
    ds_old = xr.open_dataset("../cymep/netcdf-files/netcdf_NATL_rean_configs_old.nc")
    ds_old = ds_old.isel(years=slice(80, 119+1))
    ds = xr.open_dataset("../cymep/netcdf-files/netcdf_NATL_rean_configs_new.nc")

    assert len(ds) == len(ds_old) - 1

    for variable in ds:
        print(variable)
        assert ds[variable].attrs == ds_old[variable].attrs
        if (ds[variable] != ds_old[variable]).all():
            nans = np.isnan(ds[variable].values)
            assert np.isnan(ds_old[variable].values[nans]).all()
            np.testing.assert_allclose(ds[variable].values[~nans], ds_old[variable].values[~nans], rtol=1e-4)

    assert (ds.model == ds_old.model_names.astype("U16")).all()

    for key in ds.attrs:
        print(key)
        if key not in ["history", "styr", "enyr"]:
            assert str(ds.attrs[key]) == ds_old.attrs[key]

    # Metrics by storm
    for csvfile in pathlib.Path("../cymep/csv-files").glob("storms_rean_configs_NATL_*_old.csv"):
        df_old = pd.read_csv(csvfile)
        ds_new = xr.open_dataset(str(csvfile).replace("_old.csv", ".nc"))

        for var in ds_new:
            print(var)
            np.testing.assert_allclose(ds_new[var], df_old[old_names[var]], atol=0.01)

    # Metrics by Model
    dfs_old = []
    for csvfile in pathlib.Path("../cymep/csv-files").glob("metrics_rean_configs_NATL_*_old.csv"):
        dfs_old.append(pd.read_csv(csvfile, index_col="Model"))
    df_old = pd.concat(dfs_old, axis=1)

    ds_new = xr.open_dataset("../cymep/csv-files/metrics_rean_configs_NATL.nc")
    assert len(ds_new) == len(df_old.columns)
    for var in ds_new:
        print(var)
        np.testing.assert_allclose(ds_new[var], df_old[var], rtol=1e-4)

    # Climatology metrics
    df_old = pd.read_csv("../cymep/csv-files/means_rean_configs_NATL_climo_mean_old.csv", index_col="Model")
    ds_new = xr.open_dataset("../cymep/csv-files/means_rean_configs_NATL_climo_mean.nc")
    assert len(ds_new) == len(df_old.columns)
    for var in ds_new:
        print(var)
        np.testing.assert_allclose(ds_new[var], df_old[var], rtol=1e-4)

