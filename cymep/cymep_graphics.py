from argparse import ArgumentParser
import pathlib
import yaml

import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from cartopy.crs import EqualEarth, PlateCarree

from huracanpy import basins


transform = PlateCarree()


def generate_plots(configs):
    # Make path for output files
    path = pathlib.Path("cymep-data") / configs["basin"]
    plot_path = pathlib.Path("cymep-figs") / configs["basin"]
    plot_path.mkdir(parents=True, exist_ok=True)
    (plot_path / "line").mkdir(exist_ok=True)
    (plot_path / "spatial").mkdir(exist_ok=True)
    (plot_path / "tables").mkdir(exist_ok=True)
    (plot_path / "taylor").mkdir(exist_ok=True)

    # Load in processed data
    filename = f"{configs['filename_out']}_{configs['basin']}"
    ds = xr.open_dataset(path / f"diags_{filename}.nc")

    projection = EqualEarth(central_longitude=ds.lon.values.mean())

    # Determine layout for groups of maps
    nsubplots = len(ds.dataset.values)
    nrows, ncols = optimum_layout(nsubplots)

    for var in ds:
        if "py_" in var or "pm_" in var:
            if "py_" in var:
                x = "year"
            elif "pm_" in var:
                x = "month"
            else:
                x = None

            for dataset in ds.dataset.values:
                ds_ = ds[var].sel(dataset=dataset)
                plt.plot(ds[x], ds_, label=dataset)

            plt.xlabel(x.capitalize())
            plt.ylabel(var)
            plt.legend()
            plt.savefig(plot_path / f"line/{var}_{filename}.png")
            plt.close()

        elif "full" in var:
            fig, axes = plt.subplots(
                nrows,
                ncols,
                sharex="all",
                sharey="all",
                figsize=(ncols * 3, nrows * 2 + 1),
                subplot_kw=dict(projection=projection),
            )
            axes = axes.flatten()
            if len(ds.dataset.values) != nrows * ncols:
                axes[-1].remove()

            if "bias" in var:
                vmax = np.abs(ds[var]).max()
                vmin = -vmax
                cmap = "seismic"

            else:
                vmin = ds[var].min()
                vmax = ds[var].max()

                if vmin == 0:
                    cmap = "cubehelix_r"
                else:
                    cmap = "plasma"

            for n, dataset in enumerate(ds.dataset.values):
                ds_ = ds[var].sel(dataset=dataset)
                im = axes[n].pcolormesh(
                    ds.lon,
                    ds.lat,
                    ds_,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    transform=transform,
                )
                axes[n].set_title(dataset)
                axes[n].coastlines()

                if n >= (nrows - 1) * ncols:
                    if n % ncols == 0:
                        axes[n].gridlines(draw_labels=["left", "bottom"])
                    else:
                        axes[n].gridlines(draw_labels=["bottom"])
                elif n % ncols == 0:
                    axes[n].gridlines(draw_labels=["left"])
                else:
                    axes[n].gridlines(draw_labels=False)

            fig.subplots_adjust(bottom=0.15)
            cax = fig.add_axes([0.1, 0.05, 0.8, 0.05])
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            if "units" in ds_.attrs:
                cbar.set_label(ds_.attrs["units"])
            fig.suptitle(var)
            plt.savefig(plot_path / f"spatial/{var}_{filename}.png")
            plt.close()

    for correlation in ["spearman", "pearson"]:
        for period in ["month", "year"]:
            repstr = f"{correlation}_{period}_"
            ds_ = ds[[var for var in ds if repstr in var]]
            ds_ = ds_.rename({var: var.replace(repstr, "") for var in ds_})
            fig, table = correlation_table(ds_, cmap="Blues_r")
            fig.suptitle(f"{correlation.capitalize()} correlation by {period}")
            plt.savefig(
                plot_path
                / f"tables/{period}ly_{correlation}_correlation_{filename}.png"
            )
            plt.close()

    fig, table = correlation_table(ds[[var for var in ds if "rxy_" in var]], cmap="Blues_r")
    fig.suptitle("Spatial correlation")
    plt.savefig(plot_path / f"tables/spatial_correlation_{filename}.png")
    plt.close()

    fig, table = correlation_table(
        ds[[var for var in ds if "uclim_" in var]], reference="OBS", cmap="coolwarm"
    )
    fig.suptitle("Climatological bias")
    plt.savefig(plot_path / f"tables/climatological_bias_{filename}.png")
    plt.close()

    fig, table = correlation_table(
        ds[[var for var in ds if "utc_" in var]], reference="OBS", cmap="coolwarm"
    )
    fig.suptitle("Storm mean bias")
    plt.savefig(plot_path / f"tables/storm_mean_bias_{filename}.png")
    plt.close()

    # TODO - Taylor diagram


def correlation_table(ds, cmap="viridis", reference=None):
    fig = plt.figure()

    rownames = ds.dataset.values
    colnames = list(ds)
    values = np.array([ds[var].values for var in ds]).T

    if reference is not None:
        reference = ds.sel(dataset=reference)
        reference_values = np.array([reference[var].values for var in ds])

        values[1:] -= reference_values

    colors = np.zeros([len(rownames), len(colnames), 4])
    cmap = plt.get_cmap(cmap)

    for n, col in enumerate(values[1:].T):
        if reference is None:
            vmin = col.min()
            vmax = 1
        else:
            vmax = np.abs(col).max()
            vmin = -vmax

        norm = matplotlib.colors.Normalize(vmin, vmax)
        scalarmap = matplotlib.cm.ScalarMappable(norm, cmap)

        colors[1:, n, :] = scalarmap.to_rgba(col)

    colors[0, :] = matplotlib.colors.to_rgba_array(["grey"] * len(colnames))

    values = [[f"{value:.2f}" for value in row] for row in values]

    table = plt.table(
        cellText=values,
        cellColours=colors,
        rowLabels=rownames,
        colLabels=colnames,
        bbox=[0, 0, 1, 1],
    )
    plt.gca().set_axis_off()

    fig.subplots_adjust(bottom=0.15)
    cax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
    cbar = fig.colorbar(scalarmap, cax=cax, orientation="horizontal")
    if reference is None:
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels(["Worse Performance", "Better Performance"])
    else:
        cbar.set_ticks([vmin, 0, vmax])
        cbar.set_ticklabels(["Low Bias", "No Bias", "High Bias"])

    return fig, table


def optimum_layout(nsubplots):
    # Given the number of suplots arrange into the closest to square pair of factors
    # e.g. 15 would be 5x3 (prefer more rows than columns)
    factors = _factors(nsubplots)

    # If the number of subplots is prime add an extra blank suplot and find the new
    # closest pair of factors
    # e.g. 7 would be 4x2
    if len(factors) == 1:
        factors = _factors(nsubplots + 1)
    ncols, nrows = factors[-1]

    return nrows, ncols


def _factors(y):
    return [(x, y // x) for x in range(1, int(np.floor(np.sqrt(y))) + 1) if y % x == 0]


def main():
    parser = ArgumentParser()
    parser.add_argument("config_filename")

    args = parser.parse_args()

    # Read in configuration file
    with open(args.config_filename) as f:
        configs = yaml.safe_load(f)

    if isinstance(configs["basin"], list):
        for basin in configs["basin"].copy():
            print(f"Running cymep-graphics for basin {basin}")
            configs["basin"] = basin
            generate_plots(configs)
    elif configs["basin"].lower() == "all":
        for basin in list(basins["WMO-TC"].index) + ["N", "S", "global"]:
            print(f"Running cymep-graphics for basin {basin}")
            configs["basin"] = basin
            generate_plots(configs)
    else:
        generate_plots(configs)


if __name__ == "__main__":
    main()
