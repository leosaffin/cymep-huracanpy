import os
import re
import numpy as np
import pandas as pd
import xarray as xr
import scipy.stats as sps
import warnings

import huracanpy

from functions.mask_tc import maskTC, getbasinmaskstr, fill_missing_pressure_wind
from functions.track_density import track_density, track_mean, track_minmax
from functions.write_spatial import write_spatial_netcdf, write_single_csv
from functions.pattern_cor import pattern_cor, taylor_stats

# --------------------------------------------------------------------------------------
# User settings

basin = 1
csvfilename = "rean_configs.csv"
gridsize = 8.0
styr = 1980
enyr = 2019
stmon = 1
enmon = 12
truncate_years = False
THRESHOLD_ACE_WIND = -1.0  # wind speed (in m/s) to threshold ACE. Negative means off.
THRESHOLD_PACE_PRES = -100.0  # slp (in hPa) to threshold PACE. Negative means off.
do_special_filter_obs = True  # Special "if" block for first line (control)
do_fill_missing_pw = True
do_defineMIbypres = False
debug_level = 0  # 0 = no debug, 1 = semi-verbose, 2 = very verbose

# --------------------------------------------------------------------------------------

# Constants
ms_to_kts = 1.94384449
pi = 3.141592653589793
deg2rad = pi / 180.0

# --------------------------------------------------------------------------------------


def initialise_arrays(nfiles, nyears, nmonths):
    # Init per year arrays
    pydict = {
        var: np.full([nfiles, nyears], np.nan) for var in
        ["py_count", "py_tcd", "py_ace", "py_pace", "py_latgen", "py_lmi"]
    }

    # Init per month arrays
    pmdict = {
        var: np.full((nfiles, nmonths), np.nan) for var in
        ["pm_count", "pm_tcd", "pm_ace", "pm_pace", "pm_lmi"]
    }

    # Average by year arrays
    aydict = {
        var: np.full(nfiles, np.nan) for var in
        ["uclim_count", "uclim_tcd", "uclim_ace", "uclim_pace", "uclim_lmi"]
    }

    # Average by storm arrays
    asdict = {
        var: np.full(nfiles, np.nan) for var in
        ["utc_tcd", "utc_ace", "utc_pace", "utc_latgen", "utc_lmi"]
    }

    return pydict, pmdict, aydict, asdict


def filter_tracks(tracks, special_filter_obs, basin):
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
    print("****** " + "./config-lists/" + csvfilename)

    # Read in configuration file and parse columns for each case
    # Ignore commented lines starting with !
    df = pd.read_csv("./config-lists/" + csvfilename, sep=",", comment="!", header=None)
    files = df.loc[:, 0]
    strs = np.array(df.loc[:, 1]).astype("U16")
    isUnstructStr = df.loc[:, 2]
    ensmembers = df.loc[:, 3]
    yearspermember = df.loc[:, 4]
    windcorrs = df.loc[:, 5]

    # Get some useful global values based on input data
    nfiles = len(files)
    years = list(range(styr, enyr+1))
    nyears = enyr - styr + 1
    if enmon < stmon:
        months = list(range(stmon, 12 + 1)) + list(range(1, enmon + 1))
    else:
        months = list(range(stmon, enmon + 1))
    nmonths = len(months)

    # Initialize global numpy array/dicts
    pydict, pmdict, aydict, asdict = initialise_arrays(nfiles, nyears, nmonths)

    # Get basin string
    strbasin = getbasinmaskstr(basin)

    for ii in range(len(files)):
        print("-----------------------------------------------------------------------")
        print(files[ii])

        if files[ii][0] == "/":
            print("First character is /, using absolute path")
            trajfile = files[ii]
        else:
            trajfile = "trajs/" + files[ii]

        # Determine the number of model years available in our dataset
        if truncate_years:
            # print("Truncating years from "+yearspermember(zz)+" to "+nyears)
            nmodyears = ensmembers[ii] * nyears
        else:
            # print("Using years per member of "+yearspermember(zz))
            nmodyears = ensmembers[ii] * yearspermember[ii]

        # Extract trajectories from tempest file and assign to arrays
        # USER_MODIFY
        tracks = huracanpy.load(
            trajfile,
            tracker="tempest",
            variable_names=["slp", "wind", "unknown"],
            tempest_extremes_unstructured=isUnstructStr[ii],
            tempest_extremes_header_str="start",
        )
        tracks["slp"] = tracks.slp / 100.0
        tracks["wind"] = tracks.wind * windcorrs[ii]

        # Fill in missing values of pressure and wind
        if do_fill_missing_pw:
            fill_missing_pressure_wind(tracks)

        # Filter observational records
        if debug_level >= 1:
            print("DEBUG1: Storms originally: ", len(tracks.groupby("track_id")))
        tracks = filter_tracks(tracks, do_special_filter_obs and ii == 0, basin)

        if debug_level >= 1:
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
            if do_defineMIbypres:
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

        # Calculate TC days at every valid track point
        # Assuming 6-hourly data
        xtcdpp = tracks.wind
        xtcdpp = np.where(~np.isnan(xtcdpp), 0.25, 0)

        # Calculate storm-accumulated ACE
        xacepp = 1.0e-4 * (ms_to_kts * tracks.wind) ** 2.0
        if THRESHOLD_ACE_WIND > 0:
            print("Thresholding ACE to only TCs > " + str(THRESHOLD_ACE_WIND) + " m/s")
            xacepp[tracks.wind < THRESHOLD_ACE_WIND] = np.nan

        tracks["xacepp"] = xacepp
        xace = np.empty(nstorms)
        for kk, (track_id, track) in enumerate(tracks.groupby("track_id")):
            xace[kk] = np.nansum(track.xacepp)

        # Calculate storm-accumulated PACE
        quadratic_fit = True
        calcPolyFitPACE = True
        xprestmp = tracks.slp.data

        # Threshold PACE if requested
        if THRESHOLD_PACE_PRES > 0:
            print("Thresholding PACE to only TCs < " + str(THRESHOLD_PACE_PRES) + " hPa")
            xprestmp = np.where(xprestmp > THRESHOLD_PACE_PRES, float("NaN"), xprestmp)

        xprestmp = np.ma.array(xprestmp, mask=np.isnan(xprestmp))
        warnings.filterwarnings("ignore")
        if quadratic_fit:
            if calcPolyFitPACE:
                # Here, we calculate a quadratic P/W fit based off of the "control"
                if ii == 0:
                    polyn = 2
                    xprestmp = np.ma.where(xprestmp < 1010.0, xprestmp, 1010.0)
                    xprestmp = 1010.0 - xprestmp
                    idx = np.isfinite(xprestmp) & np.isfinite(tracks.wind)
                    quad_a = np.polyfit(
                        xprestmp[idx].flatten(), tracks.wind[idx].data, polyn
                    )
            else:  # Use the coefficients from Z2021
                print("calcPolyFitPACE is False, using coefficients from Z2021")
                quad_a = np.array([-1.05371378e-03, 5.68356519e-01, 1.43290190e01])
            print("m/s")
            print(quad_a)
            print("kts")
            print(quad_a * ms_to_kts)
            xwindtmp = (
                quad_a[2]
                + quad_a[1] * (1010.0 - tracks.slp.data)
                + quad_a[0] * ((1010.0 - tracks.slp.data) ** 2)
            )
            xpacepp = 1.0e-4 * (ms_to_kts * xwindtmp) ** 2.0

            if debug_level >= 2:
                # Flatten the 2-D arrays
                xwindtmp_flat = xwindtmp.flatten()
                xwind_flat = track.wind.flatten()
                xpres_flat = track.slp.flatten()
                # Print the flattened values in sets of three
                for ss in range(len(xwindtmp_flat)):
                    if not (
                        np.isnan(xwindtmp_flat[ss])
                        and np.isnan(xwind_flat[ss])
                        and np.isnan(xpres_flat[ss])
                    ):
                        print("DEBUG2: ", xwindtmp_flat[ss], xwind_flat[ss], xpres_flat[ss])

        else:
            # Here, we apply a predetermined PW relationship from Holland
            xprestmp = np.ma.where(xprestmp < 1010.0, xprestmp, 1010.0)
            xpacepp = 1.0e-4 * (ms_to_kts * 2.3 * (1010.0 - xprestmp) ** 0.76) ** 2.0

        # Calculate PACE from xpacepp
        tracks["xpacepp"] = ("record", xpacepp)
        xpace = np.empty(nstorms)

        # Get maximum intensity and TCD
        xmpres = np.empty(nstorms)
        xmwind = np.empty(nstorms)
        xtcd = np.empty(nstorms)
        for kk, (track_id, track) in enumerate(tracks.groupby("track_id")):
            xpace[kk] = np.nansum(track.xpacepp)
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
                model=[strs[ii]] * len(xgyear),
            ),
            coords=dict(storm=np.arange(len(xgyear))),
        )

        os.makedirs(os.path.dirname("./csv-files/"), exist_ok=True)
        csvfilename_out = f"{os.path.splitext(csvfilename)[0]}_{strbasin}"
        filtered_storm_data.to_netcdf(
            f"./csv-files/storms_{csvfilename_out}_{strs[ii]}_output.nc"
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
            yrix = year - styr  # Convert from year to zero indexing for numpy array
            if np.nanmin(xgyear) <= year <= np.nanmax(xgyear):
                pydict["py_count"][ii, yrix] = (
                    np.count_nonzero(xgyear == year) / ensmembers[ii]
                )
                pydict["py_tcd"][ii, yrix] = (
                    np.nansum(np.where(xgyear == year, xtcd, 0.0)) / ensmembers[ii]
                )
                pydict["py_ace"][ii, yrix] = (
                    np.nansum(np.where(xgyear == year, xace, 0.0)) / ensmembers[ii]
                )
                pydict["py_pace"][ii, yrix] = (
                    np.nansum(np.where(xgyear == year, xpace, 0.0)) / ensmembers[ii]
                )
                pydict["py_lmi"][ii, yrix] = np.nanmean(
                    np.where(xgyear == year, xlatmi, float("NaN"))
                )
                pydict["py_latgen"][ii, yrix] = np.nanmean(
                    np.where(xgyear == year, np.absolute(xglat), float("NaN"))
                )

        # Calculate control interannual standard deviations
        if ii == 0:
            stdydict = {"sdy_count": [np.nanstd(pydict["py_count"][ii, :])],
                        "sdy_tcd": [np.nanstd(pydict["py_tcd"][ii, :])],
                        "sdy_ace": [np.nanstd(pydict["py_ace"][ii, :])],
                        "sdy_pace": [np.nanstd(pydict["py_pace"][ii, :])],
                        "sdy_lmi": [np.nanstd(pydict["py_lmi"][ii, :])],
                        "sdy_latgen": [np.nanstd(pydict["py_latgen"][ii, :])]}

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
        trackdens, denslat, denslon = track_density(
            gridsize, 0.0, tracks.lat.data, tracks.lon.data, False
        )
        trackdens = trackdens / nmodyears
        gendens, denslat, denslon = track_density(
            gridsize, 0.0, xglat, xglon, False
        )
        gendens = gendens / nmodyears
        tcddens, denslat, denslon = track_mean(
            gridsize, 0.0, tracks.lat.data, tracks.lon.data, xtcdpp, False, 0
        )
        tcddens = tcddens / nmodyears
        acedens, denslat, denslon = track_mean(
            gridsize, 0.0, tracks.lat.data, tracks.lon.data, xacepp.data, False, 0
        )
        acedens = acedens / nmodyears
        pacedens, denslat, denslon = track_mean(
            gridsize, 0.0, tracks.lat.data, tracks.lon.data, xpacepp.data, False, 0
        )
        pacedens = pacedens / nmodyears
        minpres, denslat, denslon = track_minmax(
            gridsize, 0.0, tracks.lat.data, tracks.lon.data, tracks.slp.data, "min", -1
        )
        maxwind, denslat, denslon = track_minmax(
            gridsize, 0.0, tracks.lat.data, tracks.lon.data, tracks.wind.data, "max", -1
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

        # If ii = 0, generate master spatial arrays
        if ii == 0:
            print("Generating cosine weights...")
            denslatwgt = np.cos(deg2rad * denslat)
            print("Generating master spatial arrays...")
            msdict = {}
            msvars = [
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
            for x in msvars:
                msdict[x] = np.empty((nfiles, denslat.size, denslon.size))

        # Store this model's data in the master spatial array
        msdict["fulldens"][ii, :, :] = trackdens[:, :]
        msdict["fullgen"][ii, :, :] = gendens[:, :]
        msdict["fullpace"][ii, :, :] = pacedens[:, :]
        msdict["fullace"][ii, :, :] = acedens[:, :]
        msdict["fulltcd"][ii, :, :] = tcddens[:, :]
        msdict["fullpres"][ii, :, :] = minpres[:, :]
        msdict["fullwind"][ii, :, :] = maxwind[:, :]
        msdict["fulltrackbias"][ii, :, :] = trackdens[:, :] - msdict["fulldens"][0, :, :]
        msdict["fullgenbias"][ii, :, :] = gendens[:, :] - msdict["fullgen"][0, :, :]
        msdict["fullacebias"][ii, :, :] = acedens[:, :] - msdict["fullace"][0, :, :]
        msdict["fullpacebias"][ii, :, :] = pacedens[:, :] - msdict["fullpace"][0, :, :]

        print("-------------------------------------------------------------------------")


    ### Back to the main program

    # for zz in pydict:
    #  print(pydict[zz])
    #  pydict[zz] = np.where( pydict[zz] <= 0.     , 0. , pydict[zz] )
    #  pydict[zz] = np.where( np.isnan(pydict[zz]) , 0. , pydict[zz] )
    #  pydict[zz] = np.where( np.isinf(pydict[zz]) , 0. , pydict[zz] )

    # Spatial correlation calculations

    ## Initialize dict
    rxydict = {}
    rxyvars = ["rxy_track", "rxy_gen", "rxy_u10", "rxy_slp", "rxy_ace", "rxy_pace"]
    for x in rxyvars:
        rxydict[x] = np.empty(nfiles)

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
        for ii in range(len(files)):
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
        for ii in range(len(files)):
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
        strs,
        "./csv-files/",
        f"metrics_{csvfilename_out}.csv",
    )

    write_single_csv(
        [stdydict],
        [strs[0]],
        "./csv-files/",
        f"means_{csvfilename_out}_climo_mean.csv",
    )

    # Package a series of global package inputs for storage as NetCDF attributes
    globaldict = dict(
        strbasin=strbasin,
        do_special_filter_obs=str(do_special_filter_obs),
        do_fill_missing_pw=str(do_fill_missing_pw),
        csvfilename=csvfilename,
        truncate_years=str(truncate_years),
        do_defineMIbypres=str(do_defineMIbypres),
        gridsize=gridsize,
    )

    # Write NetCDF file
    write_spatial_netcdf(
        msdict, pmdict, pydict, taydict, strs, years, months, denslat, denslon, globaldict
    )


if __name__ == "__main__":
    main()
