import huracanpy
import numpy as np
import pandas as pd


def fill_missing_pressure_wind(tracks):
    aaa = 2.3
    bbb = 1010.0
    ccc = 0.76

    # Fill in missing values of pressure/wind assuming the relationship from Holland
    # wind = 2.3 * (1010 - p) ** 0.76
    # first, when pressure is missing but wind exists
    fixes_1 = (tracks.slp < 0.0) & (tracks.wind > 0.0)
    tracks.slp[fixes_1] = bbb - (tracks.wind[fixes_1] / aaa) ** (1.0 / ccc)

    # next, when wind is missing but pressure exists
    fixes_2 = (tracks.wind < 0.0) & (tracks.slp > 0.0)
    tracks.wind[fixes_2] = aaa * (bbb - tracks.slp[fixes_2]) ** ccc

    # last, if both are missing, assume TD
    fixes_3 = tracks.slp < 0.0
    tracks.slp[fixes_3] = 1008.0
    tracks.wind[fixes_3] = 15.0

    print(
        f"Num fills for PW "
        f"{np.count_nonzero(fixes_1)}, "
        f"{np.count_nonzero(fixes_2)}, "
        f"{np.count_nonzero(fixes_3)}"
    )


def filter_tracks(tracks, special_filter_obs, basin, start_time, end_time, months, years, truncate_years):
    # if "control" record and do_special_filter_obs = true, we can apply specific
    # criteria here to match objective tracks better
    # for example, ibtracs includes tropical depressions, eliminate these to get WMO
    # tropical storms > 17 m/s.
    if special_filter_obs:
        print("Doing special processing of control file")
        windthreshold = 17.5
        tracks = tracks.isel(record=np.where(tracks.wind > windthreshold)[0])

    # Mask TCs for particular basin based on genesis location
    if basin.lower() != "global":
        if basin.lower() in ["n", "s"]:
            track_basin = huracanpy.info.hemisphere(tracks.lat)
        else:
            track_basin = huracanpy.info.basin(tracks.lon, tracks.lat)

        # Determine if track is in basin by whether it has an Ocean point in the basin
        ocean = huracanpy.info.is_ocean(tracks.lon, tracks.lat)
        track_ids_to_keep = np.unique(tracks.track_id[(track_basin == basin) & ocean])
        tracks = tracks.hrcn.sel_id(track_ids_to_keep)

    # Mask TCs based on temporal characteristics
    origin = tracks.hrcn.get_gen_vals()
    valid_times = origin.time.dt.month.isin(months)
    if truncate_years:
        valid_times = valid_times & origin.time.dt.year.isin(years)

    if start_time is not None:
        valid_times = valid_times & (origin.time >= pd.to_datetime(start_time))

    if end_time is not None:
        lysis = tracks.hrcn.get_apex_vals("time")
        valid_times = valid_times & (lysis.time < pd.to_datetime(end_time))

    track_ids_to_keep = origin.track_id[valid_times]
    tracks = tracks.hrcn.sel_id(track_ids_to_keep)

    return tracks
