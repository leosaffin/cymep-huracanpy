import os
import re
import yaml

import numpy as np
import xarray as xr
import scipy.stats as sps

import huracanpy

from functions.mask_tc import maskTC, getbasinmaskstr, fill_missing_pressure_wind
from functions.track_density import track_density, track_mean, track_minmax, create_grid
from functions.write_spatial import write_spatial_netcdf, write_single_csv
from functions.pattern_cor import pattern_cor, taylor_stats


def initialise_arrays(nfiles, nyears, nmonths, denslon, denslat):
    # Init per year arrays
    pydict = {
        var: np.full([nfiles, nyears], np.nan)
        for var in ["py_count", "py_tcd", "py_ace", "py_pace", "py_latgen", "py_lmi"]
    }

    # Init per month arrays
    pmdict = {
        var: np.full((nfiles, nmonths), np.nan)
        for var in ["pm_count", "pm_tcd", "pm_ace", "pm_pace", "pm_lmi"]
    }

    # Average by year arrays
    aydict = {
        var: np.full(nfiles, np.nan)
        for var in ["uclim_count", "uclim_tcd", "uclim_ace", "uclim_pace", "uclim_lmi"]
    }

    # Average by storm arrays
    asdict = {
        var: np.full(nfiles, np.nan)
        for var in ["utc_tcd", "utc_ace", "utc_pace", "utc_latgen", "utc_lmi"]
    }

    # generate master spatial arrays
    msdict = {
        var: np.empty((nfiles, denslat.size - 1, denslon.size - 1))
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
        ]
    }

    # Initialize dict
    rxydict = {
        var: np.empty(nfiles)
        for var in ["rxy_track", "rxy_gen", "rxy_u10", "rxy_slp", "rxy_ace", "rxy_pace"]
    }

    return pydict, pmdict, aydict, asdict, msdict, rxydict


def filter_tracks(tracks, special_filter_obs, basin, stmon, enmon, styr, enyr, truncate_years):
    # if "control" record and do_special_filter_obs = true, we can apply specific
    # criteria here to match objective tracks better
    # for example, ibtracs includes tropical depressions, eliminate these to get WMO
    # tropical storms > 17 m/s.
    if special_filter_obs:
        print("Doing special processing of control file")
        windthreshold = 17.5
        tracks = tracks.where(tracks.wind > windthreshold, drop=True)

    # Mask TCs for particular basin based on genesis location
    tracks_in_basin = []
    if basin > 0:
        for track_id, track in tracks.groupby("track_id"):
            if basin == 20 or basin == 21:
                test_basin = maskTC(track.lat[0], track.lon[0], dohemi=True)
            else:
                test_basin = maskTC(track.lat[0], track.lon[0])
            if test_basin == basin:
                tracks_in_basin.append(track_id)

    tracks = tracks.where(tracks.track_id.isin(tracks_in_basin), drop=True)

    # Mask TCs based on temporal characteristics
    tracks_to_keep = []
    for track_id, track in tracks.groupby("track_id"):
        maskoff = False
        orimon = track.time.dt.month[0]
        oriyear = track.time.dt.year[0]
        # End month < start month. Account for wrap around in years
        if enmon <= stmon:
            if enmon < orimon < stmon:
                maskoff = True
        else:
            if orimon < stmon or orimon > enmon:
                maskoff = True
        if truncate_years:
            if oriyear < styr or oriyear > enyr:
                maskoff = True

        if not maskoff:
            tracks_to_keep.append(track_id)

    tracks = tracks.where(tracks.track_id.isin(tracks_to_keep), drop=True)

    return tracks


def main():
    # Read in configuration file
    with open("example_config.yaml") as f:
        configs = yaml.safe_load(f)

    # Get some useful global values based on input data
    models = list(configs["models"].keys())
    nfiles = len(models)
    years = list(range(configs["styr"], configs["enyr"] + 1))
    nyears = len(years)
    if configs["enmon"] < configs["stmon"]:
        months = list(range(configs["stmon"], 12 + 1)) + list(range(1, configs["enmon"] + 1))
    else:
        months = list(range(configs["stmon"], configs["enmon"] + 1))
    nmonths = len(months)

    # Generate grid for spatial patterns
    lonstart = 0.0
    denslon, denslat = create_grid(configs["gridsize"], lonstart)
    denslatwgt = np.cos(np.deg2rad(0.5 * (denslat[:-1] + denslat[1:])))

    # Initialize global numpy array/dicts
    pydict, pmdict, aydict, asdict, msdict, rxydict = initialise_arrays(
        nfiles, nyears, nmonths, denslon, denslat
    )

    # Get basin string
    strbasin = getbasinmaskstr(configs["basin"])

    for ii, (model, model_config) in enumerate(configs["models"].items()):
        print("-----------------------------------------------------------------------")
        print(model_config["filename"])

        # Determine the number of model years available in our dataset
        if configs["truncate_years"]:
            # print("Truncating years from "+yearspermember(zz)+" to "+nyears)
            nmodyears = model_config["ensmembers"] * nyears
        else:
            # print("Using years per member of "+yearspermember(zz))
            nmodyears = model_config["ensmembers"] * model_config["yearspermember"]

        # Extract trajectories from tempest file and assign to arrays
        # USER_MODIFY
        tracks = huracanpy.load(
            "trajs/" + model_config["filename"],
            **{**configs["load_keywords"], **model_config["load_keywords"]}
        )
        tracks["slp"] = tracks.slp / 100.0
        tracks["wind"] = tracks.wind * model_config["windcorrs"]
        tracks["lon"] = tracks.lon % 360

        # Fill in missing values of pressure and wind
        if configs["do_fill_missing_pw"]:
            fill_missing_pressure_wind(tracks)

        # Filter observational records
        if configs["debug_level"] >= 1:
            print("DEBUG1: Storms originally: ", len(tracks.groupby("track_id")))
        tracks = filter_tracks(
            tracks, configs["do_special_filter_obs"] and ii == 0, configs["basin"],
            configs["stmon"], configs["enmon"], configs["styr"], configs["enyr"], configs["truncate_years"]
        )

        if configs["debug_level"] >= 1:
            print("DEBUG1: Storms after time filter: ", len(tracks.groupby("track_id")))
        #########################################

        # Initialize nan'ed arrays specific to this traj file
        nstorms = len(set(tracks.track_id.data))
        origin = tracks.groupby("track_id").map(lambda tr: tr.isel(record=0))
        xglon = origin.lon.data
        xglat = origin.lat.data
        xgmonth = origin.time.dt.month.data
        xgyear = origin.time.dt.year.data
        xgday = origin.time.dt.day.data
        xghour = origin.time.dt.hour.data
        xlatmi = np.empty(nstorms)
        xlonmi = np.empty(nstorms)

        xlatmi[:] = np.nan
        xlonmi[:] = np.nan

        # Calculate LMI
        for kk, (track_id, track) in enumerate(tracks.groupby("track_id")):
            if configs["do_defineMIbypres"]:
                locMI = track.slp.argmin()
            else:
                locMI = track.wind.argmax()
            xlatmi[kk] = track.lat.isel(record=locMI)
            xlonmi[kk] = track.lon.isel(record=locMI)

        # Flip LMI sign in SH to report poleward values when averaging
        abs_lats = True
        if abs_lats:
            xlatmi = np.absolute(xlatmi)
            # xglat  = np.absolute(xglat)

        # Calculate storm-accumulated ACE
        xace = huracanpy.diags.track_stats.ace_by_track(
            tracks, tracks.wind, threshold=configs["THRESHOLD_ACE_WIND"], keep_ace_by_point=True
        ).values

        # Calculate PACE
        # Calculate the coefficients of the fit for the reference data and then apply
        # these coefficients to the other datasets
        if ii == 0:
            xpace, pw_model = huracanpy.diags.track_stats.pace_by_track(
                tracks,
                tracks.slp,
                wind=tracks.wind,
                threshold_pressure=configs["THRESHOLD_PACE_PRES"],
                keep_pace_by_point=True,
            )
        else:
            xpace, _ = huracanpy.diags.track_stats.pace_by_track(
                tracks,
                tracks.slp,
                model=pw_model,
                threshold_pressure=configs["THRESHOLD_PACE_PRES"],
                keep_pace_by_point=True,
            )
        xpace = xpace.values

        # Get maximum intensity and TCD
        xmpres = np.empty(nstorms)
        xmwind = np.empty(nstorms)
        xtcd = np.empty(nstorms)
        for kk, (track_id, track) in enumerate(tracks.groupby("track_id")):
            xmpres[kk] = track.slp.data.min()
            xmwind[kk] = track.wind.data.max()
            xtcd[kk] = len(track.time) / 4

        # Print some CSV files with storm-level information from each dataset
        filtered_storm_data = xr.Dataset(
            data_vars=dict(
                genesis_year=("storm", xgyear),
                genesis_month=("storm", xgmonth),
                genesis_day=("storm", xgday),
                genesis_hour=("storm", xghour),
                genesis_latitude=("storm", xglat),
                genesis_longitude=("storm", xglon),
                minimum_pressure=("storm", xmpres),
                maximum_wind=("storm", xmwind),
                total_cyclone_days=("storm", xtcd),
                accumulated_cyclone_energy=("storm", xace),
                pressure_accumulated_cyclone_energy=("storm", xpace),
                model=[model] * len(xgyear),
            ),
            coords=dict(storm=np.arange(len(xgyear))),
        )

        os.makedirs(os.path.dirname("./csv-files/"), exist_ok=True)
        csvfilename_out = f"{configs['filename_out']}_{strbasin}"
        filtered_storm_data.to_netcdf(
            f"./csv-files/storms_{csvfilename_out}_{model}_output.nc"
        )

        # Bin storms per dataset per calendar month
        for jj, month in enumerate(months):
            pmdict["pm_count"][ii, jj] = np.count_nonzero(xgmonth == month) / nmodyears
            pmdict["pm_tcd"][ii, jj] = (
                np.nansum(np.where(xgmonth == month, xtcd, 0.0)) / nmodyears
            )
            pmdict["pm_ace"][ii, jj] = (
                np.nansum(np.where(xgmonth == month, xace, 0.0)) / nmodyears
            )
            pmdict["pm_pace"][ii, jj] = (
                np.nansum(np.where(xgmonth == month, xpace, 0.0)) / nmodyears
            )
            pmdict["pm_lmi"][ii, jj] = np.nanmean(
                np.where(xgmonth == month, xlatmi, float("NaN"))
            )

        # Bin storms per dataset per calendar year
        for year in years:
            yrix = year - configs["styr"]  # Convert from year to zero indexing for numpy array
            if np.nanmin(xgyear) <= year <= np.nanmax(xgyear):
                pydict["py_count"][ii, yrix] = (
                    np.count_nonzero(xgyear == year) / model_config["ensmembers"]
                )
                pydict["py_tcd"][ii, yrix] = (
                    np.nansum(np.where(xgyear == year, xtcd, 0.0)) / model_config["ensmembers"]
                )
                pydict["py_ace"][ii, yrix] = (
                    np.nansum(np.where(xgyear == year, xace, 0.0)) / model_config["ensmembers"]
                )
                pydict["py_pace"][ii, yrix] = (
                    np.nansum(np.where(xgyear == year, xpace, 0.0)) / model_config["ensmembers"]
                )
                pydict["py_lmi"][ii, yrix] = np.nanmean(
                    np.where(xgyear == year, xlatmi, float("NaN"))
                )
                pydict["py_latgen"][ii, yrix] = np.nanmean(
                    np.where(xgyear == year, np.absolute(xglat), float("NaN"))
                )

        # Calculate control interannual standard deviations
        if ii == 0:
            stdydict = {
                "sdy_count": [np.nanstd(pydict["py_count"][ii, :])],
                "sdy_tcd": [np.nanstd(pydict["py_tcd"][ii, :])],
                "sdy_ace": [np.nanstd(pydict["py_ace"][ii, :])],
                "sdy_pace": [np.nanstd(pydict["py_pace"][ii, :])],
                "sdy_lmi": [np.nanstd(pydict["py_lmi"][ii, :])],
                "sdy_latgen": [np.nanstd(pydict["py_latgen"][ii, :])],
            }

        # Calculate annual averages
        aydict["uclim_count"][ii] = np.nansum(pmdict["pm_count"][ii, :])
        aydict["uclim_tcd"][ii] = np.nansum(xtcd) / nmodyears
        aydict["uclim_ace"][ii] = np.nansum(xace) / nmodyears
        aydict["uclim_pace"][ii] = np.nansum(xpace) / nmodyears
        aydict["uclim_lmi"][ii] = np.nanmean(pydict["py_lmi"][ii, :])

        # Calculate storm averages
        asdict["utc_tcd"][ii] = np.nanmean(xtcd)
        asdict["utc_ace"][ii] = np.nanmean(xace)
        asdict["utc_pace"][ii] = np.nanmean(xpace)
        asdict["utc_lmi"][ii] = np.nanmean(xlatmi)
        asdict["utc_latgen"][ii] = np.nanmean(np.absolute(xglat))

        # Calculate spatial densities, integrals, and min/maxes
        trackdens = (
            track_density(tracks.lat.data, tracks.lon.data, denslat, denslon, False)
            / nmodyears
        )

        gendens = track_density(xglat, xglon, denslat, denslon, False) / nmodyears

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

        # Store this model's data in the master spatial array
        msdict["fulldens"][ii, :, :] = trackdens[:, :]
        msdict["fullgen"][ii, :, :] = gendens[:, :]
        msdict["fullpace"][ii, :, :] = pacedens[:, :]
        msdict["fullace"][ii, :, :] = acedens[:, :]
        msdict["fulltcd"][ii, :, :] = tcddens[:, :]
        msdict["fullpres"][ii, :, :] = minpres[:, :]
        msdict["fullwind"][ii, :, :] = maxwind[:, :]
        msdict["fulltrackbias"][ii, :, :] = (
            trackdens[:, :] - msdict["fulldens"][0, :, :]
        )
        msdict["fullgenbias"][ii, :, :] = gendens[:, :] - msdict["fullgen"][0, :, :]
        msdict["fullacebias"][ii, :, :] = acedens[:, :] - msdict["fullace"][0, :, :]
        msdict["fullpacebias"][ii, :, :] = pacedens[:, :] - msdict["fullpace"][0, :, :]

        print(
            "-------------------------------------------------------------------------"
        )

    # Back to the main program
    # Spatial correlation calculations
    for ii in range(nfiles):
        rxydict["rxy_track"][ii] = pattern_cor(
            msdict["fulldens"][0, :, :], msdict["fulldens"][ii, :, :], denslatwgt, 0
        )
        rxydict["rxy_gen"][ii] = pattern_cor(
            msdict["fullgen"][0, :, :], msdict["fullgen"][ii, :, :], denslatwgt, 0
        )
        rxydict["rxy_u10"][ii] = pattern_cor(
            msdict["fullwind"][0, :, :], msdict["fullwind"][ii, :, :], denslatwgt, 0
        )
        rxydict["rxy_slp"][ii] = pattern_cor(
            msdict["fullpres"][0, :, :], msdict["fullpres"][ii, :, :], denslatwgt, 0
        )
        rxydict["rxy_ace"][ii] = pattern_cor(
            msdict["fullace"][0, :, :], msdict["fullace"][ii, :, :], denslatwgt, 0
        )
        rxydict["rxy_pace"][ii] = pattern_cor(
            msdict["fullpace"][0, :, :], msdict["fullpace"][ii, :, :], denslatwgt, 0
        )

    # Temporal correlation calculations
    # Spearman Rank
    rsdict = {}
    for jj in pmdict:
        # Swap per month strings with corr prefix and init dict key
        repStr = re.sub("pm_", "rs_", jj)
        rsdict[repStr] = np.empty(nfiles)
        for ii in range(len(configs["models"])):
            # Create tmp vars and find nans
            tmpx = pmdict[jj][0, :]
            tmpy = pmdict[jj][ii, :]
            nas = np.logical_or(np.isnan(tmpx), np.isnan(tmpy))
            rsdict[repStr][ii], tmp = sps.spearmanr(tmpx[~nas], tmpy[~nas])

    # Pearson correlation
    rpdict = {}
    for jj in pmdict:
        # Swap per month strings with corr prefix and init dict key
        repStr = re.sub("pm_", "rp_", jj)
        rpdict[repStr] = np.empty(nfiles)
        for ii in range(len(configs["models"])):
            # Create tmp vars and find nans
            tmpx = pmdict[jj][0, :]
            tmpy = pmdict[jj][ii, :]
            nas = np.logical_or(np.isnan(tmpx), np.isnan(tmpy))
            rpdict[repStr][ii], tmp = sps.pearsonr(tmpx[~nas], tmpy[~nas])

    # Generate Taylor dict
    taydict = {}
    tayvars = [
        "tay_pc",
        "tay_ratio",
        "tay_bias",
        "tay_xmean",
        "tay_ymean",
        "tay_xvar",
        "tay_yvar",
        "tay_rmse",
    ]
    for x in tayvars:
        taydict[x] = np.empty(nfiles)

    # Calculate Taylor stats and put into taylor dict
    for ii in range(nfiles):
        ratio = taylor_stats(
            msdict["fulldens"][ii, :, :], msdict["fulldens"][0, :, :], denslatwgt, 0
        )
        for ix, x in enumerate(tayvars):
            # print(x+" "+str(ratio[ix]))
            taydict[x][ii] = ratio[ix]

    # Calculate special bias for Taylor diagrams
    taydict["tay_bias2"] = np.empty(nfiles)
    for ii in range(nfiles):
        taydict["tay_bias2"][ii] = 100.0 * (
            (aydict["uclim_count"][ii] - aydict["uclim_count"][0])
            / aydict["uclim_count"][0]
        )

    # Write out primary stats files
    write_single_csv(
        [rxydict, rsdict, rpdict, aydict, asdict],
        models,
        "./csv-files/",
        f"metrics_{csvfilename_out}.csv",
    )

    write_single_csv(
        [stdydict],
        [models[0]],
        "./csv-files/",
        f"means_{csvfilename_out}_climo_mean.csv",
    )

    # Package a series of global package inputs for storage as NetCDF attributes
    globaldict = dict(
        strbasin=strbasin,
        do_special_filter_obs=str(configs["do_special_filter_obs"]),
        do_fill_missing_pw=str(configs["do_fill_missing_pw"]),
        csvfilename=configs["filename_out"] + ".csv",
        truncate_years=str(configs["truncate_years"]),
        do_defineMIbypres=str(configs["do_defineMIbypres"]),
        gridsize=configs["gridsize"],
    )

    # Write NetCDF file
    write_spatial_netcdf(
        msdict,
        pmdict,
        pydict,
        taydict,
        models,
        years,
        months,
        denslat,
        denslon,
        globaldict,
    )


if __name__ == "__main__":
    main()
