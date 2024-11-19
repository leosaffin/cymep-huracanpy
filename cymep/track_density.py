import warnings

import shapely
from shapely.affinity import translate
import numpy as np

from huracanpy import basins


def create_grid(gridsize, basin, buffer, wrap_point=-180):
    if basin.lower() == "global":
        lonW, latS, lonE, latN = -180, -90, 180, 90
    elif basin.lower() == "n":
        lonW, latS, lonE, latN = -180, 0, 180, 90
    elif basin.lower() == "s":
        lonW, latS, lonE, latN = -180, -90, 180, 0
    else:
        try:
            lonW, latS, lonE, latN = (
                basins["WMO-TC"].loc[basin].geometry.exterior.bounds
            )
        except AttributeError:
            # If the basin crosses the dateline it will be specified as a multipolygon
            # In this case we don't want the exterior of this multipolygon we want to
            # shift our longitudes so that they cross the dateline
            wrap_point = 0
            geoms = basins["WMO-TC"].loc[basin].geometry.geoms
            geoms_translated = []
            for geom in geoms:
                if (np.array(geom.exterior.xy[0]) < wrap_point).all():
                    geoms_translated.append(translate(geom, xoff=360))
                elif (np.array(geom.exterior.xy[0]) < wrap_point).any():
                    raise ValueError(f"Can't merge geometry for {basin}")
                else:
                    geoms_translated.append(geom)

            geom = shapely.unary_union(geoms_translated)
            lonW, latS, lonE, latN = geom.exterior.bounds

        if buffer is not None:
            latS = max(latS - buffer, -90)
            latN = min(latN + buffer, 90)

            lonW = lonW - buffer
            lonE = lonE + buffer

            if lonE - lonW >= 360:
                warnings.warn(
                    f"basin={basin} and grid_buffer={buffer} covers all longitudes"
                )
                lonW = wrap_point
                lonE = wrap_point + 360

            elif lonW < wrap_point or lonE > wrap_point + 360:
                # Put the wrap point somewhere between lonW and lonE
                wrap_point = lonW

    dlat = gridsize
    dlon = gridsize

    nlat = int((latN - latS) / dlat) + 1
    mlon = int((lonE - lonW) / dlon) + 1

    lat = np.linspace(latS, latN, num=nlat)
    lon = np.linspace(lonW, lonE, num=mlon)

    weights = np.cos(np.deg2rad(0.5 * (lat[:-1] + lat[1:])))
    weights = np.expand_dims(weights, axis=1)
    weights = np.repeat(weights, mlon - 1, axis=1)

    return lon, lat, weights, wrap_point


def track_density(clat, clon, glat, glon, setzeros):
    countarr, y, x = np.histogram2d(clat, clon, bins=[glat, glon])

    print(f"count: min={int(np.nanmin(countarr))}   max={int(np.nanmax(countarr))}")
    print(f"count: sum={int(np.nansum(countarr))}")

    if setzeros:
        countarr = np.where(countarr == 0, np.nan, countarr)

    return countarr


def track_mean(clat, clon, glat, glon, cvar, meanornot, minhits):
    npoints = len(clat)
    out_of_bounds = (
        (clon < glon[0]) | (clon >= glon[-1]) | (clat < glat[0]) | (clat >= glat[-1])
    )

    clon = clon[~out_of_bounds]
    clat = clat[~out_of_bounds]
    cvar = cvar[~out_of_bounds]

    print(f"Track mean. Removed {npoints - len(clon)} points out of {npoints}")

    xidx = np.digitize(clon, glon) - 1
    yidx = np.digitize(clat, glat) - 1

    countarr = np.zeros((len(glat) - 1, len(glon) - 1))
    cumulative = np.zeros((len(glat) - 1, len(glon) - 1))

    # =================== Count data ==================================
    for nn, (jl, il) in enumerate(zip(yidx, xidx)):
        countarr[jl, il] = countarr[jl, il] + 1
        cumulative[jl, il] = cumulative[jl, il] + cvar[nn]

    # set to nan if cumulative less than the specified number of min hits
    cumulative = np.where(countarr < minhits, np.nan, cumulative)

    if meanornot:
        # Normalize by dividing by count
        countarr = np.where(countarr == 0, np.nan, countarr)
        cumulative = cumulative / countarr

    print(f"cumulative: min={np.nanmin(cumulative)}   max={np.nanmax(cumulative)}")
    print(f"cumulative: sum={np.nansum(cumulative)}")

    return cumulative


def track_minmax(clat, clon, glat, glon, cvar, statistic):
    npoints = len(clat)
    out_of_bounds = (
        (clon < glon[0]) | (clon >= glon[-1]) | (clat < glat[0]) | (clat >= glat[-1])
    )

    clon = clon[~out_of_bounds]
    clat = clat[~out_of_bounds]
    cvar = cvar[~out_of_bounds]

    print(f"Track minmax. Removed {npoints - len(clon)} points out of {npoints}")

    xidx = np.digitize(clon, glon) - 1
    yidx = np.digitize(clat, glat) - 1

    countarr = np.full((len(glat) - 1, len(glon) - 1), np.nan)

    # =================== Count data ==================================
    for nn, (jl, il) in enumerate(zip(yidx, xidx)):
        if np.isnan(countarr[jl, il]):
            countarr[jl, il] = cvar[nn]
        else:
            countarr[jl, il] = statistic(countarr[jl, il], cvar[nn])

    print(f"count: min={np.nanmin(countarr)}  max={np.nanmax(countarr)}")

    return countarr
