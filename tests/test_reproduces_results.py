import pathlib

import numpy as np
import xarray as xr


def test_all():
    # Spatial data
    ds_old = xr.open_dataset("../cymep/netcdf-files/netcdf_NATL_rean_configs_old.nc")
    ds = xr.open_dataset("../cymep/netcdf-files/netcdf_NATL_rean_configs.nc")

    ds_old["months"] = ds_old.month
    ds_old["years"] = ds_old.year
    ds_old = ds_old.drop(["month", "year"])
    ds_old = ds_old.rename(months="month", years="year")
    del ds_old.attrs["history"]
    del ds.attrs["history"]

    for var in ds_old:
        if "ace" in var:
            print(var)
            if "pace" in var:
                if var == "fullpacebias":
                    rtol = 0
                    atol = 0.01
                else:
                    rtol = 0.01
                    atol = 0.0
            else:
                rtol = 1e-4
                atol = 0
            np.testing.assert_allclose(ds[var], ds_old[var], rtol=rtol, atol=atol)
            ds[var] = ds_old[var]
    assert ds_old.identical(ds)

    # Metrics by storm
    for fname in pathlib.Path("../cymep/csv-files").glob("_old.nc"):
        ds_old = xr.open_dataset(fname)
        ds = xr.open_dataset(str(fname).replace("_old.nc", ".nc"))

        for var in ds_old:
            if "ace" in var and "pace" not in var:
                np.testing.assert_allclose(ds_old[var], ds[var], rtol=1e-4)
                ds[var] = ds_old[var]
        assert ds_old.identical(ds)
