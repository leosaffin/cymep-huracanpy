import pathlib

import xarray as xr


def test_all():
    # Spatial data
    ds_old = xr.open_dataset("../cymep/netcdf-files/netcdf_NATL_rean_configs_old.nc")
    ds = xr.open_dataset("../cymep/netcdf-files/netcdf_NATL_rean_configs.nc")

    del ds_old.attrs["history"]
    del ds.attrs["history"]
    assert ds_old.identical(ds)

    # Metrics by storm
    for fname in pathlib.Path("../cymep/csv-files").glob("_old.nc"):
        ds_old = xr.open_dataset(fname)
        ds = xr.open_dataset(str(fname).replace("_old.nc", ".nc"))
        assert ds_old.identical(ds)
