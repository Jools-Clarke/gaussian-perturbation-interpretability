"""Fig 11: HAT-P-26 b, changes relative to TauREx's best fit to x*, in sigma of each bin (Sec. 7.3.3): the +3 sigma
Gaussian perturbation at 4.29 um (grey), CO2 +-0.5 dex and H2O -0.5 dex alone, and TauREx's best fit to the perturbed
spectrum (black), which raises the band core through the scale height rather than through CO2.

Reads: E2_Hat-P-26b/outputs/co2_band_changes.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import _CONFIG as config
from src import plotting as style
from src.perturbation import gaussian_window

E2 = Path(__file__).resolve().parents[1] / "E2_Hat-P-26b" / "outputs"
TEXTWIDTH_IN = 7.03  # RASTI two-column text width
BUMP_XI = 3.0  # the perturbation of E2 step5 (co2_band_changes.csv)


def changes(ds):
    wl, sigma = ds["wl_um"].values, ds["sigma"].values
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 3.0))
    # the perturbation drawn smoothly: Eq. 11's bump on a fine grid, in units of the (interpolated) noise of each bin
    edges, _ = style.bin_edges_with_gaps(wl)
    core = int(np.argmin(np.abs(wl - config.TARGET_BANDS["CO2"])))
    fine = np.linspace(edges[0], edges[-1], 4000)
    bump = BUMP_XI * sigma[core] * gaussian_window(fine, wl[core], config.GAUSSIAN_FWHM_UM)
    ax.fill_between(fine, 0, bump / np.interp(fine, wl, sigma), color="0.82", lw=0,
                    label=rf"$+{BUMP_XI:g}\sigma$ Gaussian perturbation")
    ax.plot(wl, ds["taurex_plus3"].values, color="k", lw=1.4, zorder=5,
            label=r"Best-fitting TauREx-III model of the $+3\sigma$ spectrum")
    styles = {"co2_plus": (style.COLOURS["CO2"], "-", r"CO$_2$ $+0.5$ dex alone"),
              "co2_minus": (style.COLOURS["CO2"], "--", r"CO$_2$ $-0.5$ dex alone"),
              "h2o_minus": (style.COLOURS["H2O"], "--", r"H$_2$O $-0.5$ dex alone")}
    for change, (colour, ls, label) in styles.items():
        ax.plot(wl, ds[change].values, color=colour, ls=ls, lw=1.1, label=label)
    for w in config.TARGET_BANDS.values():
        ax.axvline(w, color="k", ls=":", lw=0.6)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_ylabel(r"$\Delta(R_p/R_s)^2$ ($\sigma$)")
    lines = [line.get_ydata() for line in ax.get_lines() if len(line.get_ydata()) == len(wl)]
    low, high = np.min(lines), max(np.max(lines), BUMP_XI)
    ax.set_ylim(low - 1.5, high + 0.35)  # room below the data for the legend
    ax.legend(fontsize=7, loc="lower center", ncol=3, borderaxespad=0.3, frameon=True, facecolor="white",
              edgecolor="none", framealpha=1.0)  # opaque so the band markers stop behind it
    ax.minorticks_on()
    print("saved", style.save(fig, Path(__file__).stem, dpi=300))
    plt.close(fig)


if __name__ == "__main__":
    style.use()
    changes(pd.read_csv(E2 / "co2_band_changes.csv"))
