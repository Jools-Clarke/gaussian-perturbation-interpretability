"""Fig 13: response-gradient maps of the CNN for SO2 (top) and CO2 (bottom), shared colour scale, with the global
metric of both gases over the training spectra between them and, dotted, the TauREx-III local metric of CO2 as a
visual guide (Sec. 8.2). f is the mean over 1 000 fixed dropout masks.

Reads: E3_CNN/outputs/response_gradient_fixed1000.csv, global_metric.csv, E2_Hat-P-26b/outputs/metrics.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import _CONFIG as config
from src import plotting as style

ROOT = Path(__file__).resolve().parents[1]
E2, E3 = ROOT / "E2_Hat-P-26b" / "outputs", ROOT / "E3_CNN" / "outputs"
VMAX = 0.3


def main():
    gradients = pd.read_csv(E3 / "response_gradient_fixed1000.csv")
    rho = pd.read_csv(E3 / "global_metric.csv")
    local = pd.read_csv(E2 / "metrics.csv")
    wl = rho["wl_um"].values
    assert np.allclose(local["wl_um"].values, wl, atol=1e-3)
    fig = plt.figure(figsize=(3.4, 4))
    grid = fig.add_gridspec(3, 2, width_ratios=[1, 0.04], height_ratios=[1, 0.7, 1], hspace=0.1, wspace=0.02)
    ax_rho = fig.add_subplot(grid[1, 0])
    ax_so2 = fig.add_subplot(grid[0, 0], sharex=ax_rho)
    ax_co2 = fig.add_subplot(grid[2, 0], sharex=ax_rho)

    for mol in ("CO2", "SO2"):
        r = rho[f"global_{mol}"].values
        ax_rho.fill_between(wl, 0, r, color=style.COLOURS[mol], alpha=0.1, zorder=2)
        ax_rho.plot(wl, r, color=style.COLOURS[mol], lw=1, zorder=3, label=rf"$\mathrm{{{style.TEX[mol]}}}$")
    ax_rho.plot(wl, local["local_CO2"].values, ":", color=style.COLOURS["CO2"], lw=1.1, alpha=0.75, zorder=2.5)
    ax_rho.axhline(0, color="0.4", lw=0.6, ls="--", zorder=1)
    ax_rho.set_ylabel(r"$\rho(\mathcal{D}_m, \log Y)$")
    leg = ax_rho.legend(ncol=2, loc="upper left", fontsize=7, handlelength=1.2, columnspacing=0.9, frameon=False)
    ax_rho.add_artist(leg)
    ax_rho.legend(handles=[Line2D([0], [0], color=style.COLOURS["CO2"], lw=1.1, ls=":", alpha=0.75,
                                  label=r"$\mathrm{CO_2}$ TauREx local")],
                  loc="upper left", bbox_to_anchor=(0.0, 0.8), fontsize=7, handlelength=1.6, frameon=False)
    ax_rho.set_ylim(-0.65, 0.8)
    for band in config.TARGET_BANDS.values():
        ax_rho.axvline(band, color="0.35", lw=0.7, ls=":", zorder=0)

    edges, columns = style.bin_edges_with_gaps(wl)
    xi_edges = np.union1d(gradients["xi_low"], gradients["xi_high"])
    for ax, mol, slot in ((ax_so2, "SO2", 0), (ax_co2, "CO2", 2)):
        gradient = gradients.query("gas == @mol").pivot(index="xi_low", columns="centre_um", values="gradient")
        assert np.allclose(gradient.columns.values, wl, atol=1e-3)
        mesh = ax.pcolormesh(edges, xi_edges, style.expand_columns(gradient.values, columns), cmap="PuOr", vmin=-VMAX,
                             vmax=VMAX, shading="flat", rasterized=True)
        ax.set_ylabel(r"$\xi$ ($\sigma$)")
        label = rf"$\partial\Delta\log\mathrm{{{style.TEX[mol]}}}/\partial\xi$" + "\n" + r"(dex $\sigma^{-1}$)"
        cbar = style.colorbar(mesh, fig.add_subplot(grid[slot, 1]), label)
        cbar.set_label(label, fontsize=7)
        cbar.ax.tick_params(labelsize=7)
    ax_so2.tick_params(labelbottom=False)
    ax_rho.tick_params(labelbottom=False)
    ax_co2.set_xlabel(r"Wavelength ($\mu$m)")
    ax_co2.set_xlim(edges[0], edges[-1])
    fig.align_ylabels([ax_rho, ax_so2, ax_co2])
    print("saved", style.save(fig, Path(__file__).stem, dpi=300))
    plt.close(fig)


if __name__ == "__main__":
    style.use(minor_ticks=True)
    main()
