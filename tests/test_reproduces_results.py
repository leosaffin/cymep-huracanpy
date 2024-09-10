import os
import pathlib

import numpy as np
import xarray as xr

from cymep import cymep


def test_all():
    # Run cymep using the example data
    os.chdir("example/")
    remove_new_files()
    cymep.generate_diagnostics("example_config.yaml")

    # Check that newly generated files all match the old files
    for fname in pathlib.Path("cymep-data/").glob("*_old.nc"):
        ds_old = xr.open_dataset(fname)
        ds = xr.open_dataset(str(fname).replace("_old.nc", ".nc"))

        # The timestamp won't match, so remove this
        if "diags" in str(fname):
            del ds_old.attrs["history"]
            del ds.attrs["history"]

        for var in ds_old:
            print(var)
            if var != "dataset":
                np.testing.assert_allclose(ds[var].values, ds_old[var].values, rtol=1e-12)
        #assert ds_old.identical(ds)


    #remove_new_files()


def remove_new_files():
    for fname in pathlib.Path("cymep-data").glob("*.nc"):
        if "_old" not in str(fname):
            os.remove(str(fname))
