import os
import pathlib
import yaml

import numpy as np
import xarray as xr

from cymep import cymep


def test_all():
    # Run cymep using the example data
    os.chdir("example/")
    # remove_new_files()
    # with open("example_config.yaml") as f:
    #     configs = yaml.safe_load(f)
    # cymep.generate_diagnostics(configs)

    # Check that newly generated files all match the old files
    for fname in pathlib.Path("cymep-data/").glob("*_old.nc"):
        ds_old = xr.open_dataset(fname)
        ds = xr.open_dataset(
            str(fname)
            .replace("_old.nc", ".nc")
            .replace("cymep-data/", "cymep-data/NATL/")
        )

        # The timestamp won't match, so remove this
        if "diags" in str(fname):
            del ds_old.attrs["history"]
            del ds.attrs["history"]

        #xr.testing.assert_identical(ds, ds_old)
        for var in ds:
            print(var)
            if "time" in var:
                assert (ds[var] == ds_old[var]).all()
            else:
                np.testing.assert_allclose(ds[var], ds_old[var])

        for attr in ds.attrs:
            assert ds.attrs[attr] == ds_old.attrs[attr]

    remove_new_files()


def remove_new_files():
    for fname in pathlib.Path("cymep-data").glob("*.nc"):
        if "_old" not in str(fname):
            os.remove(str(fname))
