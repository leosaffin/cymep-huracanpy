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
