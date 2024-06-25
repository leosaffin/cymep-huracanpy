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
