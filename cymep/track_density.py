import numpy as np


def create_grid(gridsize, lonstart):
    # =================== Create grid ==================================
    latS = -90.0
    latN = 90.0
    lonW = lonstart
    lonE = lonstart + 360.0

    dlat = gridsize
    dlon = gridsize

    nlat = int((latN - latS) / dlat) + 1
    mlon = int((lonE - lonW) / dlon) + 1

    lat = np.linspace(latS, latN, num=nlat)
    lon = np.linspace(lonW, lonE, num=mlon)

    return lon, lat


def track_density(clat, clon, glat, glon, setzeros):
    countarr, y, x = np.histogram2d(clat, clon, bins=[glat, glon])

    print(f"count: min={int(np.nanmin(countarr))}   max={int(np.nanmax(countarr))}")
    print(f"count: sum={int(np.nansum(countarr))}")

    if setzeros:
        countarr = np.where(countarr == 0, np.nan, countarr)

    return countarr


def track_mean(clat, clon, glat, glon, cvar, meanornot, minhits):
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
