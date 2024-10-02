import numpy as np
import xarray as xr
from scipy.stats import pearsonr, spearmanr


def spatial_correlations(ds, weights):
    rxy_ds = xr.Dataset(coords=dict(dataset=ds.dataset))

    for name in ds:
        if "full" in name and "bias" not in name:
            correlations = []
            x = ds[name][0, :, :].values
            for ii, dataset in enumerate(ds.dataset):
                y = ds[name][ii, :, :].values
                correlations.append(wpearsonr(*filter_nans([x, y], [x, y, weights])))

            rxy_ds[name.replace("full", "rxy_")] = ("dataset", np.array(correlations))

    return rxy_ds


def temporal_correlations(ds):
    nfiles = len(ds.dataset)
    corr_ds = xr.Dataset(coords=dict(dataset=ds.dataset))

    # Temporal correlation calculations
    for name in ds:
        if "pm_" in name or "py_" in name:
            if "pm_" in name:
                # Swap per month strings with corr prefix and init dict key
                # Spearman Rank
                repStr = name.replace("pm_", "spearman_month_")
                # Pearson correlation
                repStr_p = name.replace("pm_", "pearson_month_")

            if "py_" in name:
                repStr = name.replace("py_", "spearman_year_")
                # Pearson correlation
                repStr_p = name.replace("py_", "pearson_year_")

            corr_ds[repStr] = ("dataset", np.empty(nfiles))
            corr_ds[repStr_p] = ("dataset", np.empty(nfiles))
            for ii in range(nfiles):
                # Create tmp vars and find nans
                tmpx = ds[name][0, :].values
                tmpy = ds[name][ii, :].values
                tmpx, tmpy = filter_nans([tmpx, tmpy], [tmpx, tmpy])

                if len(tmpx) < 2 or len(tmpy) < 0:
                    corr_ds[repStr][ii] = np.nan
                    corr_ds[repStr_p][ii] = np.nan
                else:
                    corr_ds[repStr][ii], tmp = spearmanr(tmpx, tmpy)
                    corr_ds[repStr_p][ii], tmp = pearsonr(tmpx, tmpy)

    return corr_ds


def taylor_stats_ds(ds, weights):
    nfiles = len(ds.dataset)
    taylor_ds = xr.Dataset(coords=dict(dataset=ds.dataset))
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
        taylor_ds[x] = ("dataset", np.empty(nfiles))

    # Calculate Taylor stats and put into taylor dict
    for ii in range(nfiles):
        ratio = taylor_stats(
            ds.fulldens[ii, :, :].values, ds.fulldens[0, :, :].values, weights
        )
        for ix, x in enumerate(tayvars):
            taylor_ds[x][ii] = ratio[ix]

    # Calculate special bias for Taylor diagrams
    taylor_ds["tay_bias2"] = ("dataset", np.empty(nfiles))
    for ii in range(nfiles):
        taylor_ds.tay_bias2[ii] = 100.0 * (
            (ds.uclim_count[ii] - ds.uclim_count[0]) / ds.uclim_count[0]
        )

    return taylor_ds


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
