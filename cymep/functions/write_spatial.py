from datetime import datetime
import os

import xarray as xr


def write_spatial_netcdf(
    spatialdict,
    permondict,
    peryrdict,
    taydict,
    modelsin,
    years,
    months,
    latin,
    lonin,
    globaldict,
):
    netcdfdir = "./netcdf-files/"
    os.makedirs(os.path.dirname(netcdfdir), exist_ok=True)
    netcdfile = (
        netcdfdir
        + "/netcdf_"
        + globaldict["strbasin"]
        + "_"
        + os.path.splitext(globaldict["csvfilename"])[0]
    )

    # open a netCDF file to write
    dsout = xr.Dataset(
        coords=dict(
            model=modelsin,
            lat=0.5 * (latin[:-1] + latin[1:]),
            lon=0.5 * (lonin[:-1] + lonin[1:]),
            month=months,
            year=years,
        )
    )

    # create latitude axis
    dsout["lat"].attrs = dict(
        standard_name="latitude",
        long_name="latitude",
        units="degrees_north",
        axis="Y",
    )

    dsout["lon"].attrs = dict(
        standard_name="longitude",
        long_name="longitude",
        units="degrees_east",
        axis="X",
    )

    # create variable arrays
    # Do spatial variables
    for vardict, coords in [
        (spatialdict, ["model", "lat", "lon"]),
        (permondict, ["model", "months"]),
        (peryrdict, ["model", "years"]),
        (taydict, "model")
    ]:
        for ii in vardict:
            dsout[ii] = (coords, vardict[ii])

    # today = datetime.today()
    dsout.attrs = globaldict
    dsout.attrs["description"] = "Coastal metrics processed data"
    dsout.attrs["history"] = "Created " + datetime.today().strftime("%Y-%m-%d-%H:%M:%S")

    dsout.to_netcdf(netcdfile + ".nc")


def write_single_csv(vardicts, modelsin, csvdir, csvname):
    # create variable array
    os.makedirs(os.path.dirname(csvdir), exist_ok=True)
    csvfilename = csvdir + "/" + csvname

    dsout = xr.Dataset(
        coords=dict(
            model=modelsin
        )
    )

    for vardict in vardicts:
        for varname in vardict:
            dsout[varname] = ("model", vardict[varname])

    dsout.to_netcdf(csvfilename.replace(".csv", ".nc"))
