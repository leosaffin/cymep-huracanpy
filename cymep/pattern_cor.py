import numpy as np
import xarray as xr


def spatial_correlations(
    ms_ds, datasets, weights, names=dict(
        rxy_track="fulldens",
        rxy_gen="fullgen",
        rxy_u10="fullwind",
        rxy_slp="fullpres",
        rxy_ace="fullace",
        rxy_pace="fullpace",
    )
):
    nfiles = len(datasets)
    rxy_ds = xr.Dataset(coords=dict(dataset=datasets))

    for name_out, name_in in names.items():
        correlations = []
        x = ms_ds[name_in][0, :, :].values
        for ii in range(nfiles):
            y = ms_ds[name_in][ii, :, :].values

            if "wind" in name_in or "pres" in name_in:
                correlation = wpearsonr(*filter_nans([x, y], [x, y, weights]))
            else:
                correlation = wpearsonr(x, y, weights)
            correlations.append(correlation)

        rxy_ds[name_out] = ("dataset", np.array(correlations))

    return rxy_ds


def wpearsonr(x, y, weights):
    """Weighted pearson correlation coefficient
    """
    xanom = x - np.average(x, weights=weights)
    yanom = y - np.average(y, weights=weights)
    xycov = np.sum(weights * xanom * yanom)
    xanom2 = np.sum(weights * xanom ** 2)
    yanom2 = np.sum(weights * yanom ** 2)

    # Calculate coefficient
    return xycov / (np.sqrt(xanom2) * np.sqrt(yanom2))


def _filter(arrays_to_filter, to_keep):
    return [a.flatten()[to_keep] for a in arrays_to_filter]


def filter_nans(arrays_to_check, arrays_to_filter):
    nans = np.isnan(np.array(arrays_to_check)).any(axis=0).flatten()

    return _filter(arrays_to_filter, ~nans)


def filter_zeros(arrays_to_check, arrays_to_filter):
    # drop matching 0s, because model getting 0 when obs 0 is not interesting
    zeros = (np.array(arrays_to_check) == 0).all(axis=0).flatten()

    return _filter(arrays_to_filter, ~zeros)


def taylor_stats(x, y, weights):
    ## x is the test variable
    ## y is the reference variable (truth or control)
    ## w is weights, either a scalar, 1-D array (ex: Gaussian) or 2D lat/lon

    # Calculate pattern correlation
    pc = wpearsonr(x, y, weights)

    # Calculate averages, variance, and RMSE
    xmean = np.average(x, weights=weights)
    ymean = np.average(y, weights=weights)
    xvar = np.average((x - xmean)**2, weights=weights)
    yvar = np.average((y - ymean)**2, weights=weights)
    rmse = np.sqrt(np.average((x - y) ** 2, weights=weights))

    # Calculate bias
    bias = 100 * (xmean - ymean) / ymean

    # Calculate ratio and update RMSE
    ratio = np.sqrt(xvar / yvar)
    rmse = rmse / np.sqrt(yvar)

    return pc, ratio, bias, xmean, ymean, xvar, yvar, rmse
