import os
import pathlib
import yaml

import xarray as xr

from cymep import cymep


def test_all():
    # Run cymep using the example data
    os.chdir("example/")
    remove_new_files()
    with open("example_config.yaml") as f:
        configs = yaml.safe_load(f)
    cymep.generate_diagnostics(configs)

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

        xr.testing.assert_identical(ds, ds_old)

    remove_new_files()


def remove_new_files():
    for fname in pathlib.Path("cymep-data").glob("*.nc"):
        if "_old" not in str(fname):
            os.remove(str(fname))
