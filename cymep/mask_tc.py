import numpy as np


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
