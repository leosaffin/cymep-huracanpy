import numpy as np


def fill_missing_pressure_wind(tracks):
    aaa = 2.3
    bbb = 1010.0
    ccc = 0.76

    # first, when xpres is missing but xwind exists, try to fill in xpres
    numfixes_1 = np.count_nonzero((tracks.slp < 0.0) & (tracks.wind > 0.0))
    # xpres    = np.where(((xpres < 0.0) & (xwind > 0.0)),-1*((xwind/aaa)**(1./ccc)-bbb),xpres)
    tracks["slp"] = ("record", np.where(
        ((tracks.slp.data < 0.0) & (tracks.wind.data > 0.0)),
        -1 * (np.sign(tracks.wind.data / aaa) * (np.abs(tracks.wind.data / aaa)) ** (1.0 / ccc) - bbb),
        tracks.slp.data,
    ))
    # next, when xwind is missing but xpres exists, try to fill in xwind
    numfixes_2 = np.count_nonzero((tracks.wind < 0.0) & (tracks.slp > 0.0))
    # xwind    = np.where(((xwind < 0.0) & (xpres > 0.0)),aaa*(bbb - xpres)**ccc,xwind)
    tracks["wind"] = ("record", np.where(
        ((tracks.wind.data < 0.0) & (tracks.slp.data > 0.0)),
        aaa * np.sign(bbb - tracks.slp.data) * (np.abs(bbb - tracks.slp.data)) ** ccc,
        tracks.wind.data,
    ))
    # now if still missing assume TD
    numfixes_3 = np.count_nonzero((tracks.slp < 0.0))
    tracks.slp[tracks.slp < 0.0] = 1008.0
    tracks.wind[tracks.wind < 0.0] = 15.0
    print(
        "Num fills for PW "
        + str(numfixes_1)
        + " "
        + str(numfixes_2)
        + " "
        + str(numfixes_3)
    )


def maskTC(lat, lon, dohemi=False):
    # dohemi defaults to False, so basin-by-basin
    # however, if overrode with dohemi = True, will stratify by NHEMI and SHEMI

    # if lon is negative, switch to 0->360 convention
    if lon < 0.0:
        lon = lon + 360.0

    if dohemi:
        if lat >= 0.0:
            basin = 20
        else:
            basin = 21
    else:
        # Coefficients for calculating ATL/EPAC sloped line
        m = -0.58
        b = 0.0 - m * 295.0
        maxlat = 50.0

        if lat >= 0.0 and lat <= maxlat and lon > 257.0 and lon <= 359.0:
            funcval = m * lon + b
            if lat > funcval:
                basin = 1
            else:
                basin = 2
        elif lat >= 0.0 and lat <= maxlat and lon > 220.0 and lon <= 257.0:
            basin = 2
        elif lat >= 0.0 and lat <= maxlat and lon > 180.0 and lon <= 220.0:
            basin = 3
        elif lat >= 0.0 and lat <= maxlat and lon > 100.0 and lon <= 180.0:
            basin = 4
        elif lat >= 0.0 and lat <= maxlat and lon > 30.0 and lon <= 100.0:
            basin = 5
        elif lat < 0.0 and lat >= -maxlat and lon > 30.0 and lon <= 135.0:
            basin = 6
        elif lat < 0.0 and lat >= -maxlat and lon > 135.0 and lon <= 290.0:
            basin = 7
        else:
            basin = 0

    return basin


def getbasinmaskstr(gridchoice):
    if hasattr(gridchoice, "__len__"):
        if gridchoice[0] == 1:
            strbasin = "NHEMI"
        else:
            strbasin = "SHEMI"
    else:
        if gridchoice < 0:
            strbasin = "GLOB"
        else:
            if gridchoice == 1:
                strbasin = "NATL"
            elif gridchoice == 2:
                strbasin = "EPAC"
            elif gridchoice == 3:
                strbasin = "CPAC"
            elif gridchoice == 4:
                strbasin = "WPAC"
            elif gridchoice == 5:
                strbasin = "NIO"
            elif gridchoice == 6:
                strbasin = "SIO"
            elif gridchoice == 7:
                strbasin = "SPAC"
            elif gridchoice == 8:
                strbasin = "SATL"
            elif gridchoice == 9:
                strbasin = "FLA"
            elif gridchoice == 20:
                strbasin = "NHEMI"
            elif gridchoice == 21:
                strbasin = "SHEMI"
            else:
                strbasin = "NONE"

    return strbasin
