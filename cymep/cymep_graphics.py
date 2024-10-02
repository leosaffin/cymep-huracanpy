from argparse import ArgumentParser
import pathlib
import yaml

import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
from cartopy.crs import EqualEarth, PlateCarree


transform = PlateCarree()


def generate_plots(config_filename):
    # Read in configuration file
    with open(config_filename) as f:
        configs = yaml.safe_load(f)

    # Make path for output files
    path = pathlib.Path("cymep-data")
    plot_path = pathlib.Path("cymep-figs")
    plot_path.mkdir(exist_ok=True)
    (plot_path / "line").mkdir(exist_ok=True)
    (plot_path / "spatial").mkdir(exist_ok=True)
    (plot_path / "tables").mkdir(exist_ok=True)
    (plot_path / "taylor").mkdir(exist_ok=True)

    # Load in processed data
    filename = f"{configs['filename_out']}_{configs['basin']}"
    ds = xr.open_dataset(path / f"diags_{filename}.nc")

    projection = EqualEarth(central_longitude=ds.lon.values.mean())

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
                len(ds.dataset.values) // 2,
                2,
                sharex="all", sharey="all", figsize=(8, 15), subplot_kw=dict(projection=projection))
            axes = axes.flatten()

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
                im = axes[n].pcolormesh(ds.lon, ds.lat, ds_, vmin=vmin, vmax=vmax, cmap=cmap, transform=transform)
                axes[n].set_title(dataset)
                axes[n].coastlines()
                axes[n].gridlines(draw_labels=["left", "bottom"])

    for correlation in ["spearman", "pearson"]:
        for period in ["month", "year"]:
            repstr = f"{correlation}_{period}_"
            ds_ = ds[[var for var in ds if repstr in var]]
            ds_ = ds_.rename({var: var.replace(repstr, "") for var in ds_})
            correlation_table(ds_, cmap="Blues_r")
            plt.savefig(plot_path / f"tables/{period}ly_{correlation}_correlation_{filename}.png")
            plt.close()

    correlation_table(ds[[var for var in ds if "rxy_" in var]], cmap="Blues_r")
    plt.savefig(plot_path / f"tables/spatial_correlation_{filename}.png")

    correlation_table(ds[[var for var in ds if "uclim_" in var]], reference="OBS", cmap="coolwarm")
    plt.savefig(plot_path / f"tables/climatological_bias_{filename}.png")

    correlation_table(ds[[var for var in ds if "utc_" in var]], reference="OBS", cmap="coolwarm")
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
    cax = fig.add_axes([0.2, 0.05, 0.6, 0.05])
    cbar = fig.colorbar(scalarmap, cax=cax, orientation="horizontal")
    if reference is None:
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels(["Worse Performance", "Better Performance"])
    else:
        cbar.set_ticks([vmin, 0, vmax])
        cbar.set_ticklabels(["Low Bias", "No Bias", "High Bias"])

    return table


def main():
    parser = ArgumentParser()
    parser.add_argument("config_filename")

    args = parser.parse_args()

    generate_plots(args.config_filename)


if __name__ == '__main__':
    main()
