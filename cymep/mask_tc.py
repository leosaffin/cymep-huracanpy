import numpy as np
import huracanpy


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
