"""Fig F1: how much of each HD 209458 b perturbation TauREx's best fit reproduces (Sec. 7.2.5). (a, b) the change in
the spectrum and in the best fit for the 1.46 um, +2 sigma naive and Gaussian runs; (c) the fraction fitted (Eq. 16)
per perturbed bin, median and interquartile range over the six magnitudes.

Reads: E1_HD-209458-b/outputs/fraction_fitted.csv, fraction_fitted_example.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from src import plotting as style
from src.plotting import edges, wavelength_axis

E1 = Path(__file__).resolve().parents[1] / "E1_HD-209458-b" / "outputs"
COLOURS = {"naive": "k", "gaussian": "firebrick"}
FRACTION = "fraction of the\nperturbation fitted"
YLABEL = r"$\Delta(R_\mathrm{p}/R_\mathrm{s})^2$ ($\sigma$)"


def example_panel(ax, ex, fraction, title, centres, colour):
    dx, dbf = ex["delta_spectrum_sigma"].values, ex["delta_best_fit_sigma"].values
    e = edges(centres)
    ax.bar(centres, dx, width=np.diff(e) * 0.9, color="0.82", edgecolor="none", label="change in the spectrum")
    ax.plot(centres, dbf, "-o", color=colour, lw=1.2, ms=2.5, label="change in the best fit")
    ax.axhline(0, color="0.6", lw=0.5)
    ax.set_ylabel(YLABEL)
    ax.text(0.015, 0.95, title, transform=ax.transAxes, va="top",
            fontsize=7, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    key = ex.iloc[0]
    return float(fraction.query("kernel == @key.kernel and centre_um == @key.centre_um and xi == @key.xi")["fraction"].iloc[0])


def main():
    style.use()
    fraction = pd.read_csv(E1 / "fraction_fitted.csv")
    example = pd.read_csv(E1 / "fraction_fitted_example.csv")
    centres = np.sort(fraction["centre_um"].unique())
    key = tuple(float(v) for v in example.iloc[0][["centre_um", "xi"]])

    fig = plt.figure(figsize=(3.4, 4.2))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1.1], hspace=0.08, figure=fig)
    ax_n = fig.add_subplot(gs[0]); ax_g = fig.add_subplot(gs[1], sharex=ax_n, sharey=ax_n)
    ax_f = fig.add_subplot(gs[2], sharex=ax_n)
    f_n = example_panel(ax_n, example[example["kernel"] == "naive"], fraction, "Naive", centres, COLOURS["naive"])
    f_g = example_panel(ax_g, example[example["kernel"] == "gaussian"], fraction, "Gaussian", centres, COLOURS["gaussian"])
    ax_n.legend(loc="upper left", bbox_to_anchor=(0.0, 0.84), fontsize=6, frameon=False, handlelength=1.4)
    ax_n.set_ylim(-0.6, 2.6)

    summary = {}
    for name, offset, marker, face in (("naive", -0.004, "o", "white"), ("gaussian", 0.004, "o", COLOURS["gaussian"])):
        frac = fraction[fraction["kernel"] == name].pivot(index="centre_um", columns="xi", values="fraction").values
        med, lo, hi = np.median(frac, axis=1), np.percentile(frac, 25, axis=1), np.percentile(frac, 75, axis=1)
        ax_f.vlines(centres + offset, lo, hi, color=COLOURS[name], lw=0.8)
        ax_f.plot(centres + offset, med, marker, ms=3.5, mec=COLOURS[name], mfc=face, ls="none")
        summary[name] = np.mean(frac), np.median(frac)
    ax_f.axvline(key[0], color="0.75", lw=0.6, ls="--", zorder=0)
    ax_f.set_ylabel(FRACTION)
    ax_f.legend(handles=[Line2D([], [], marker="o", ls="none", ms=3.5, mec=COLOURS["naive"], mfc="white", label="naive"),
                         Line2D([], [], marker="o", ls="none", ms=3.5, mec=COLOURS["gaussian"], mfc=COLOURS["gaussian"], label="Gaussian")],
                loc="upper left", fontsize=6, frameon=False, ncol=2, handletextpad=0.3)
    ax_f.set_ylim(-0.1, 1.25)
    for ax in (ax_n, ax_g):
        ax.tick_params(labelbottom=False)
    wavelength_axis(ax_f, centres)
    fig.align_ylabels([ax_n, ax_g, ax_f])
    print(f"example {key}: naive {f_n:.2f}, Gaussian {f_g:.2f}; mean (median) over all runs: "
          f"naive {summary['naive'][0]:.2f} ({summary['naive'][1]:.2f}), "
          f"Gaussian {summary['gaussian'][0]:.2f} ({summary['gaussian'][1]:.2f}); saved", style.save(fig, Path(__file__).stem, dpi=300))
    plt.close(fig)


if __name__ == "__main__":
    main()
