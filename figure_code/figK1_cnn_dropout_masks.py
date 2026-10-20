"""Fig K1: response-gradient maps of the CNN for SO2 (a-c) and CO2 (e-g) with 1 000 fixed masks, 1 000 fresh masks
and 1 fresh mask per evaluation, and the importance in each case (d, h) (Sec. 8.2).

Reads: E3_CNN/outputs/response_gradient_{fixed1000,fresh1000,fresh1}.csv, importance.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _CONFIG as config
from src import plotting as style
from src.plotting import bin_edges_with_gaps, expand_columns
from figJ1_cnn_corner import LABELS

E3 = Path(__file__).resolve().parents[1] / "E3_CNN" / "outputs"
TEXTWIDTH_IN = 7.03  # RASTI two-column text width (508 pt)
TITLES = {"fixed1000": "1000 fixed masks", "fresh1000": "1000 fresh masks", "fresh1": "1 fresh mask"}
STYLES = {"fixed1000": dict(color="k", lw=1.1, marker="o", ms=2.2, zorder=4),
          "fresh1000": dict(color="#D55E00", lw=1.1, marker="o", ms=2.2, zorder=3),
          "fresh1": dict(color="0.65", lw=0.7, zorder=2)}
GRADIENTS = {mode: pd.read_csv(E3 / f"response_gradient_{mode}.csv") for mode in TITLES}
IMPORTANCE = pd.read_csv(E3 / "importance.csv")


def gradient(mode, gas):
    """(interval, centre) response gradient of one gas, intervals ascending in xi."""
    return GRADIENTS[mode].query("gas == @gas").pivot(index="xi_low", columns="centre_um", values="gradient")


def importance(mode, gas):
    return IMPORTANCE.query("mode == @mode and gas == @gas").sort_values("centre_um")["importance"].values


def figure():
    fixed = GRADIENTS["fixed1000"]
    wl = np.sort(fixed["centre_um"].unique())
    xi_edges = np.union1d(fixed["xi_low"], fixed["xi_high"])
    edges, columns = bin_edges_with_gaps(wl)

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 8.6))
    grid = fig.add_gridspec(5, 4, width_ratios=[1, 1, 1, 0.045], height_ratios=[1, 0.75, 0.12, 1, 0.75],
                            hspace=0.3, wspace=0.07)
    letters = iter("abcdefgh")
    for block, mol in enumerate(config.TARGET_BANDS):
        row, lab = 3 * block, LABELS[mol]
        vmax = np.nanpercentile(np.abs(gradient("fixed1000", mol).values), 99)
        for col, key in enumerate(TITLES):
            ax = fig.add_subplot(grid[row, col])
            mesh = ax.pcolormesh(edges, xi_edges, expand_columns(gradient(key, mol).values, columns),
                                 cmap="PuOr", vmin=-vmax, vmax=vmax, shading="flat", rasterized=True)
            for w in config.TARGET_BANDS.values():
                ax.axvline(w, color="k", ls=":", lw=0.6)
            ax.text(0.03, 0.96, f"({next(letters)}) {lab}, {TITLES[key]}", transform=ax.transAxes, ha="left", va="top",
                    fontsize=7, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
            ax.set_xticks([3.0, 3.5, 4.0, 4.5, 5.0])
            ax.tick_params(labelsize=7, labelleft=(col == 0))
            if col == 0:
                ax.set_ylabel(r"$\xi$ ($\sigma$)")
        cbar = fig.colorbar(mesh, cax=fig.add_subplot(grid[row, 3]))
        cbar.set_label(rf"$\partial\Delta\log${lab}$/\partial\xi$" + "\n" + r"(dex $\sigma^{-1}$)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax = fig.add_subplot(grid[row + 1, :3])
        for key in TITLES:
            ax.plot(wl, importance(key, mol), label=TITLES[key], **STYLES[key])
        ax.axhline(0, color="0.5", lw=0.5)
        for w in config.TARGET_BANDS.values():
            ax.axvline(w, color="k", ls=":", lw=0.6)
        lim = 1.5 * np.abs(importance("fixed1000", mol)).max()
        ax.set_ylim(-lim, lim)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylabel(f"importance, {lab}\n" + r"(dex $\sigma^{-1}$)")
        ax.text(0.01, 0.95, f"({next(letters)})", transform=ax.transAxes, ha="left", va="top", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=8)
        if block == 0:
            fig.legend(*ax.get_legend_handles_labels(), fontsize=8, ncol=3, loc="lower center",
                       bbox_to_anchor=(0.45, 0.895), frameon=False)
    return fig


if __name__ == "__main__":
    style.use()
    for mol in config.TARGET_BANDS:
        ref = importance("fixed1000", mol)
        for key in ("fresh1000", "fresh1"):
            grad_r = np.corrcoef(gradient(key, mol).values.ravel(), gradient("fixed1000", mol).values.ravel())[0, 1]
            print(f"{mol} {key}: r(importance) = {np.corrcoef(importance(key, mol), ref)[0, 1]:.2f}, "
                  f"r(gradient map) = {grad_r:.2f}")
    fig = figure()
    print("saved", style.save(fig, Path(__file__).stem, dpi=300, pad_inches=0.1))  # the original savefig default pad
    plt.close(fig)
