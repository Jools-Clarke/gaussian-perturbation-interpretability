"""Fig K1: the full posterior of the CNN retrieval of the simulated HAT-P-26 b spectrum (20 000 draws, Sec. 8.1):
1, 2, 3 sigma contours, median and 1 sigma above each column, input values in red. Its helpers also draw Fig 12's
inset.

Reads: E3_CNN/outputs/posterior_draws.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

import _CONFIG as config
from src import plotting as style

E3 = Path(__file__).resolve().parents[1] / "E3_CNN" / "outputs"
TEXTWIDTH_IN = 7.03  # RASTI two-column text width (508 pt)
ACCENT = "#B03A2E"  # brick red: the input values in the corner plots, the recovered spectrum in the spectrum plots
SIGMAS = (1, 2, 3)  # contours of the corner plots
ONE_SIGMA = 100 * norm.cdf([-1, 0, 1])  # percentiles: 1 sigma below, median, 1 sigma above
CONTOUR_GREYS = {1: "0.52", 2: "0.72", 3: "0.88"}
# gas labels of Figs 12, K1, L1: text with math subscripts (SO$_2$)
LABELS = {m: "".join(f"$_{c}$" if c.isdigit() else c for c in m) for m in style.TEX}


def contour_style(sigmas):
    """2D panels: black lines around grey regions holding the given sigmas of a Gaussian (1 - exp(-k^2 / 2) of the
    samples)."""
    return dict(bins=30, smooth=1.0, levels=1 - np.exp(-0.5 * np.array(sigmas) ** 2), color="k",
                plot_datapoints=False, plot_density=False, fill_contours=True,
                contourf_kwargs={"colors": ["none"] + [CONTOUR_GREYS[k] for k in sigmas[::-1]]},  # outside first
                contour_kwargs={"linewidths": 0.6})


def plot_ranges(samples, inputs):
    """(low, high) per parameter covering the central 99.8 per cent of the samples and the input value."""
    lo = np.minimum(np.percentile(samples, 0.1, axis=0), inputs)
    hi = np.maximum(np.percentile(samples, 99.9, axis=0), inputs)
    pad = 0.08 * (hi - lo)
    return list(zip(lo - pad, hi + pad))


def draw_corner(fig, samples, inputs, labels, titles=None, fontsize=8, labelpad=0.0):
    """corner.corner in the paper style on an empty figure, with the input values in ACCENT.

    titles: None, "value" (median and 1 sigma) or "full" (label = value).
    labelpad moves the axis labels away from the axes, in units of the panel size.
    """
    corner.corner(
        samples, fig=fig, labels=labels, truths=inputs, truth_color=ACCENT, range=plot_ranges(samples, inputs),
        quantiles=ONE_SIGMA / 100, max_n_ticks=3, labelpad=labelpad, hist_kwargs={"lw": 1.0},
        label_kwargs={"fontsize": fontsize}, **contour_style(SIGMAS),
    )
    n = len(labels)
    axes = np.array(fig.axes).reshape(n, n)
    for ax in axes.flat:
        ax.tick_params(labelsize=fontsize - 1)
    if titles:
        for i in range(n):
            low, mid, high = np.percentile(samples[:, i], ONE_SIGMA)
            value = f"${mid:.2f}^{{+{high - mid:.2f}}}_{{-{mid - low:.2f}}}$"
            axes[i, i].set_title(value if titles == "value" else f"{labels[i]} = {value}", fontsize=fontsize)
    return axes


if __name__ == "__main__":
    style.use()
    draws = pd.read_csv(E3 / "posterior_draws.csv")
    molecules = [c.removeprefix("log_") for c in draws.columns]
    inputs = np.array([config.HATP26B_LOG_VMR[m] for m in molecules])
    labels = [f"log {LABELS[m]}" for m in molecules]
    fig = plt.figure(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN))
    draw_corner(fig, draws.values, inputs, labels, titles="value", fontsize=7, labelpad=0.1)
    print("saved", style.save(fig, Path(__file__).stem, dpi=300, pad_inches=0.1))  # the original savefig default pad
    plt.close(fig)
