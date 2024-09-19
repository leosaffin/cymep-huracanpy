from argparse import ArgumentParser
import pathlib

import yaml
from datetime import datetime

import numpy as np
import xarray as xr
from iris.analysis.cartography import wrap_lons

import huracanpy

from cymep.mask_tc import fill_missing_pressure_wind
from cymep.track_density import track_density, track_mean, track_minmax, create_grid
from cymep.pattern_cor import spatial_correlations, temporal_correlations, taylor_stats_ds


def initialise_arrays(datasets, years, months, denslon, denslat):
    data_vars = dict()
    for var in ["py_count", "py_tcd", "py_ace", "py_pace", "py_latgen", "py_lmi"]:
        data_vars[var] = (("dataset", "year"), np.full([len(datasets), len(years)], np.nan))

    for var in ["pm_count", "pm_tcd", "pm_ace", "pm_pace", "pm_lmi"]:
        data_vars[var] = (
            ("dataset", "month"),
            np.full((len(datasets), len(months)), np.nan),
        )

    for var in ["uclim_count", "uclim_tcd", "uclim_ace", "uclim_pace", "uclim_lmi"]:
        data_vars[var] = ("dataset", np.full(len(datasets), np.nan))

    for var in ["utc_tcd", "utc_ace", "utc_pace", "utc_latgen", "utc_lmi"]:
        data_vars[var] = ("dataset", np.full(len(datasets), np.nan))

    for var in [
        "fulldens",
        "fullpres",
        "fullwind",
        "fullgen",
        "fullace",
        "fullpace",
        "fulltcd",
        "fulltrackbias",
        "fullgenbias",
        "fullacebias",
        "fullpacebias",
    ]:
        data_vars[var] = (
            ("dataset", "lat", "lon"),
            np.empty((len(datasets), denslat.size - 1, denslon.size - 1)),
        )

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            dataset=datasets,
            lat=0.5 * (denslat[:-1] + denslat[1:]),
            lon=0.5 * (denslon[:-1] + denslon[1:]),
            month=months,
            year=years,
        ),
    )

    # create latitude axis
    ds_out.lat.attrs = dict(
        standard_name="latitude",
        long_name="latitude",
        units="degrees_north",
        axis="Y",
    )

    ds_out.lon.attrs = dict(
        standard_name="longitude",
        long_name="longitude",
        units="degrees_east",
        axis="X",
    )

    return ds_out


def filter_tracks(tracks, special_filter_obs, basin, months, years, truncate_years):
    # if "control" record and do_special_filter_obs = true, we can apply specific
    # criteria here to match objective tracks better
    # for example, ibtracs includes tropical depressions, eliminate these to get WMO
    # tropical storms > 17 m/s.
    if special_filter_obs:
        print("Doing special processing of control file")
        windthreshold = 17.5
        tracks = tracks.where(tracks.wind > windthreshold, drop=True)

    # Mask TCs for particular basin based on genesis location
    if basin is not None:
        if basin in ["N", "H"]:
            tracks["basin"] = huracanpy.utils.geography.get_hemisphere(tracks.lon, tracks.lat)
        else:
            tracks["basin"] = huracanpy.utils.geography.get_basin(tracks.lon, tracks.lat)

        # Determine if track is in basin by whether it has an Ocean point in the basin
        tracks["land"] = huracanpy.utils.geography.get_land_or_ocean(tracks.lon, tracks.lat)
        tracks_ = tracks.where((tracks.basin == basin) & (tracks.land == "Ocean"), drop=True)

        track_ids_to_keep = list(set(tracks_.track_id.values))
        tracks = tracks.where(tracks.track_id.isin(track_ids_to_keep), drop=True)

    # Mask TCs based on temporal characteristics
    origin = tracks.groupby("track_id").first()
    valid_times = origin.time.dt.month.isin(months)
    if truncate_years:
        valid_times = valid_times & origin.time.dt.year.isin(years)

    track_ids_to_keep = origin.track_id.where(valid_times)
    tracks = tracks.where(tracks.track_id.isin(track_ids_to_keep), drop=True)

    return tracks


def generate_diagnostics(config_filename):
    # Read in configuration file
    with open(config_filename) as f:
        configs = yaml.safe_load(f)

    # Get some useful global values based on input data
    datasets = list(configs["datasets"].keys())
    years = list(range(configs["styr"], configs["enyr"] + 1))
    if configs["enmon"] < configs["stmon"]:
        months = list(range(configs["stmon"], 12 + 1)) + list(
            range(1, configs["enmon"] + 1)
        )
    else:
        months = list(range(configs["stmon"], configs["enmon"] + 1))

    # Generate grid for spatial patterns
    denslon, denslat, denslatwgt, wrap_point = create_grid(
        configs["gridsize"], configs["basin"], configs["grid_buffer"]
    )

    # Initialize global numpy array/dicts
    ds_out = initialise_arrays(datasets, years, months, denslon, denslat)

    # Make path for output files
    filename_out = f"{configs['filename_out']}_{configs['basin']}"
    output_dir = pathlib.Path("cymep-data/")
    output_dir.mkdir(exist_ok=True)

    # Load and analyse each dataset
    input_dir = pathlib.Path(configs["path_to_data"])
    for ii, (dataset, dataset_config) in enumerate(configs["datasets"].items()):
        print("-----------------------------------------------------------------------")
        print(dataset_config["filename"])

        # Determine the number of years available in our dataset
        if configs["truncate_years"]:
            nmodyears = dataset_config["ensmembers"] * len(years)
        else:
            nmodyears = dataset_config["ensmembers"] * dataset_config["yearspermember"]

        # Extract trajectories from tempest file and assign to arrays
        tracks = huracanpy.load(
            str(input_dir / dataset_config["filename"]),
            **{**configs["load_keywords"], **dataset_config["load_keywords"]},
        )
        tracks["slp"] = tracks.slp / 100.0
        tracks["wind"] = tracks.wind * dataset_config["windcorrs"]
        tracks["lon"] = wrap_lons(tracks.lon, wrap_point, 360)

        # Fill in missing values of pressure and wind
        if configs["do_fill_missing_pw"]:
            fill_missing_pressure_wind(tracks)

        # Filter observational records
        tracks = filter_tracks(
            tracks,
            configs["do_special_filter_obs"] and ii == 0,
            configs["basin"],
            months,
            years,
            configs["truncate_years"],
        )

        # Add ACE and PACE to tracks
        tracks["ace"] = huracanpy.utils.ace.ace_by_point(
            tracks.wind, threshold=configs["THRESHOLD_ACE_WIND"],
        )

        # Calculate the coefficients of the fit for the reference data and then apply
        # these coefficients to the other datasets
        if ii == 0:
            tracks["pace"], pw_model = huracanpy.utils.ace.pace_by_point(
                tracks.slp,
                wind=tracks.wind,
                threshold_pressure=configs["THRESHOLD_PACE_PRES"],
            )
        else:
            tracks["pace"], _ = huracanpy.utils.ace.pace_by_point(
                tracks.slp,
                model=pw_model,
                threshold_pressure=configs["THRESHOLD_PACE_PRES"],
            )

        # Extract summary variables for each individual storm and save as netCDF
        storm_data = get_storm_data(tracks, configs["do_defineMIbypres"])
        storm_data.to_netcdf(output_dir / f"storms_{filename_out}_{dataset}.nc")

        # Bin storms per dataset per calendar month
        for jj, month in enumerate(months):
            is_month = storm_data.genesis_time.dt.month == month
            ds_out.pm_count[ii, jj] = np.count_nonzero(is_month) / nmodyears
            ds_out.pm_tcd[ii, jj] = storm_data.total_cyclone_days[is_month].sum() / nmodyears
            ds_out.pm_ace[ii, jj] = storm_data.accumulated_cyclone_energy[is_month].sum() / nmodyears
            ds_out.pm_pace[ii, jj] = storm_data.pressure_accumulated_cyclone_energy[is_month].sum() / nmodyears
            ds_out.pm_lmi[ii, jj] = storm_data.maximum_intensity_lat[is_month].mean()

        # Bin storms per dataset per calendar year
        year_start = storm_data.genesis_time.dt.year.min()
        year_end = storm_data.genesis_time.dt.year.max()
        for year in years:
            # Convert from year to zero indexing for numpy array
            yrix = (year - configs["styr"])
            if year_start <= year <= year_end:
                is_year = storm_data.genesis_time.dt.year == year
                ds_out.py_count[ii, yrix] = np.count_nonzero(is_year) / dataset_config["ensmembers"]
                ds_out.py_tcd[ii, yrix] = storm_data.total_cyclone_days[is_year].sum() / dataset_config["ensmembers"]
                ds_out.py_ace[ii, yrix] = storm_data.accumulated_cyclone_energy[is_year].sum() / dataset_config["ensmembers"]
                ds_out.py_pace[ii, yrix] = storm_data.pressure_accumulated_cyclone_energy[is_year].sum() / dataset_config["ensmembers"]
                ds_out.py_lmi[ii, yrix] = storm_data.maximum_intensity_lat[is_year].mean()
                ds_out.py_latgen[ii, yrix] = np.abs(storm_data.genesis_lat[is_year]).mean()

        # Calculate annual averages
        ds_out.uclim_count[ii] = ds_out.pm_count[ii, :].sum()
        ds_out.uclim_tcd[ii] = storm_data.total_cyclone_days.sum() / nmodyears
        ds_out.uclim_ace[ii] = storm_data.accumulated_cyclone_energy.sum() / nmodyears
        ds_out.uclim_pace[ii] = storm_data.pressure_accumulated_cyclone_energy.sum() / nmodyears
        ds_out.uclim_lmi[ii] = ds_out.py_lmi[ii, :].mean()

        # Calculate storm averages
        ds_out.utc_tcd[ii] = storm_data.total_cyclone_days.mean()
        ds_out.utc_ace[ii] = storm_data.accumulated_cyclone_energy.mean()
        ds_out.utc_pace[ii] = storm_data.pressure_accumulated_cyclone_energy.mean()
        ds_out.utc_lmi[ii] = storm_data.maximum_intensity_lat.mean()
        ds_out.utc_latgen[ii] = np.abs(storm_data.genesis_lat).mean()

        # Calculate spatial densities, integrals, and min/maxes
        trackdens = (
            track_density(tracks.lat.data, tracks.lon.data, denslat, denslon, False)
            / nmodyears
        )

        gendens = track_density(
            storm_data.genesis_lat.data,
            storm_data.genesis_lon.data,
            denslat,
            denslon,
            False
        ) / nmodyears

        tcddens = trackdens * 0.25

        acedens = (
            track_mean(
                tracks.lat.data,
                tracks.lon.data,
                denslat,
                denslon,
                tracks.ace.data,
                False,
                0,
            )
            / nmodyears
        )

        pacedens = (
            track_mean(
                tracks.lat.data,
                tracks.lon.data,
                denslat,
                denslon,
                tracks.pace.data,
                False,
                0,
            )
            / nmodyears
        )

        minpres = track_minmax(
            tracks.lat.data,
            tracks.lon.data,
            denslat,
            denslon,
            tracks.slp.data,
            min,
        )
        maxwind = track_minmax(
            tracks.lat.data,
            tracks.lon.data,
            denslat,
            denslon,
            tracks.wind.data,
            max,
        )

        # If there are no storms tracked in this particular dataset, set everything to NaN
        if np.nansum(trackdens) == 0:
            trackdens = float("NaN")
            pacedens = float("NaN")
            acedens = float("NaN")
            tcddens = float("NaN")
            gendens = float("NaN")
            minpres = float("NaN")
            maxwind = float("NaN")

        # Store this dataset's data in the master spatial array
        ds_out.fulldens[ii, :, :] = trackdens[:, :]
        ds_out.fullgen[ii, :, :] = gendens[:, :]
        ds_out.fullpace[ii, :, :] = pacedens[:, :]
        ds_out.fullace[ii, :, :] = acedens[:, :]
        ds_out.fulltcd[ii, :, :] = tcddens[:, :]
        ds_out.fullpres[ii, :, :] = minpres[:, :]
        ds_out.fullwind[ii, :, :] = maxwind[:, :]
        ds_out.fulltrackbias[ii, :, :] = trackdens[:, :] - ds_out.fulldens[0, :, :]
        ds_out.fullgenbias[ii, :, :] = gendens[:, :] - ds_out.fullgen[0, :, :]
        ds_out.fullacebias[ii, :, :] = acedens[:, :] - ds_out.fullace[0, :, :]
        ds_out.fullpacebias[ii, :, :] = pacedens[:, :] - ds_out.fullpace[0, :, :]

        print("-----------------------------------------------------------------------")

    # Back to the main program
    # Spatial correlation calculations
    rxy_ds = spatial_correlations(ds_out, denslatwgt)

    # Temporal correlation calculations
    corr_ds = temporal_correlations(ds_out)

    # Generate Taylor dict
    tay_ds = taylor_stats_ds(ds_out, denslatwgt)

    # ----------------------------------------------------------------------------------
    # Write output data
    # Calculate control interannual standard deviations
    stdy_ds = xr.Dataset(
        data_vars=dict(
            sdy_count=("dataset", [np.nanstd(ds_out.py_count[0, :])]),
            sdy_tcd=("dataset", [np.nanstd(ds_out.py_tcd[0, :])]),
            sdy_ace=("dataset", [np.nanstd(ds_out.py_ace[0, :])]),
            sdy_pace=("dataset", [np.nanstd(ds_out.py_pace[0, :])]),
            sdy_lmi=("dataset", [np.nanstd(ds_out.py_lmi[0, :])]),
            sdy_latgen=("dataset", [np.nanstd(ds_out.py_latgen[0, :])]),
        ),
        coords=dict(dataset=[datasets[0]]),
    )
    stdy_ds.to_netcdf(output_dir / f"means_{filename_out}.nc")

    # Write out primary stats files
    ds_out = xr.merge([rxy_ds, corr_ds, tay_ds, ds_out])

    # Write NetCDF file
    # Package a series of global package inputs for storage as NetCDF attributes
    ds_out.attrs = dict(
        strbasin=configs['basin'],
        do_special_filter_obs=str(configs["do_special_filter_obs"]),
        do_fill_missing_pw=str(configs["do_fill_missing_pw"]),
        truncate_years=str(configs["truncate_years"]),
        do_defineMIbypres=str(configs["do_defineMIbypres"]),
        gridsize=configs["gridsize"],
        description="Coastal metrics processed data",
        history="Created " + datetime.today().strftime("%Y-%m-%d-%H:%M:%S"),
    )

    ds_out.to_netcdf(output_dir / f"diags_{filename_out}.nc")


def get_storm_data(tracks, define_mi_by_pressure):
    track_groups = tracks.groupby("track_id")

    origin = track_groups.first()
    origin_vars = {key: f"genesis_{key}" for key in ["lon", "lat", "time"]}
    storm_data = origin[origin_vars.keys()].rename(**origin_vars)

    # Calculate LMI
    if define_mi_by_pressure:
        locMI = track_groups.map(lambda x: x.isel(record=x.slp.argmin()))
    else:
        locMI = track_groups.map(lambda x: x.isel(record=x.wind.argmax()))

    storm_data["maximum_intensity_lon"] = locMI.lon
    storm_data["maximum_intensity_lat"] = locMI.lat

    # Flip LMI sign in SH to report poleward values when averaging
    abs_lats = True
    if abs_lats:
        storm_data["maximum_intensity_lat"] = np.abs(locMI.lat)

    # Calculate storm-accumulated ACE and PACE
    storm_data["accumulated_cyclone_energy"] = track_groups.map(lambda x: x.ace.sum())
    storm_data["pressure_accumulated_cyclone_energy"] = track_groups.map(
        lambda x: x.pace.sum()
    )

    # Get maximum intensity and TCD
    storm_data["minimum_pressure"] = track_groups.map(lambda x: x.slp.min())
    storm_data["maximum_wind"] = track_groups.map(lambda x: x.wind.max())

    # Currently assuming 6-hourly timesteps
    # Taking "last - first" doesn't work due to gaps in data
    # storm_data["total_cyclone_days"] = track_groups.map(
    #     lambda x: ((x.time.max() - x.time.min()).astype(float) * 1e-9 / (3600 * 24))
    # )
    storm_data["total_cyclone_days"] = (
        "track_id",
        np.array([0.25 * len(track.time) for track_id, track in track_groups])
    )

    return storm_data


def main():
    parser = ArgumentParser()
    parser.add_argument("config_filename")

    args = parser.parse_args()

    generate_diagnostics(args.config_filename)


if __name__ == "__main__":
    main()
