"""Figs I1-I4: full corner plots of the HAT-P-26 b retrievals at xi = -3 (purple), 0 (teal) and +3 sigma (yellow):
    I1 naive 4.041 um    I2 naive 4.293 um    I3 Gaussian 4.041 um    I4 Gaussian 4.293 um
Weighted KDE contours and marginals, as Fig 9.

Reads: E2_Hat-P-26b/outputs/runs.csv, posteriors/*.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import gaussian_kde

from fig09_hatp26b_posteriors import MAGNITUDES, finite_samples, magnitude_colours, posteriors, tex
from src import plotting as style
from src.plotting import hpd_thresholds, weighted_hist

# the 13 retrieval parameters (Table G2)
PARAMS = ["log_H2O", "log_CO", "log_CO2", "log_CH4", "log_SO2", "log_NH3", "log_C2H2", "log_C2H4", "log_H2S", "log_HCN",
          "log_clouds_pressure", "T", "planet_radius"]
FIGURES = {"I1": ("naive", 4.041), "I2": ("naive", 4.293), "I3": ("gaussian", 4.041), "I4": ("gaussian", 4.293)}


def corner_plot(label, kernel, wl):
    hist_bins, grid_res, levels = 100, 200, [1, 2]  # marginal bins, KDE grid per axis, contour levels in sigma
    panel_in = 1.6  # inches per row / column
    wl_actual, by_xi = posteriors(kernel, wl)
    n = len(PARAMS)

    # axis range of each parameter, pooled over the magnitudes
    ranges, data_1d = {}, {}
    for p in PARAMS:
        for mag in MAGNITUDES:
            data_1d[(p, mag)] = finite_samples(by_xi[mag], p)
        vals = np.concatenate([data_1d[(p, mag)][0] for mag in MAGNITUDES])
        pad = 0.05 * (vals.max() - vals.min())
        lo, hi = vals.min() - pad, vals.max() + pad
        ranges[p] = (lo, hi, np.linspace(lo, hi, grid_res), np.linspace(lo, hi, hist_bins + 1))

    def axis_fmt(p):  # two decimals for parameters within (-1, 1), e.g. planet_radius
        lo, hi, _, _ = ranges[p]
        return '%.2f' if max(abs(lo), abs(hi)) < 1 else '%d'

    cmap, norm = magnitude_colours()
    fig_size = max(n * panel_in, 6)
    fig = plt.figure(figsize=(fig_size, fig_size))
    gs = fig.add_gridspec(n, n, wspace=0.06, hspace=0.06)
    col_ref_x, row_ref_y = [None] * n, [None] * n  # sharex per column (its diagonal), sharey per row (first off-diagonal)
    for i in range(n):  # row: y parameter
        for j in range(i + 1):  # column: x parameter; upper triangle left empty
            ax = fig.add_subplot(gs[i, j], sharex=col_ref_x[j], sharey=row_ref_y[i] if j < i else None)
            if i == j:
                col_ref_x[j] = ax
            elif row_ref_y[i] is None:
                row_ref_y[i] = ax
            p_col, p_row = PARAMS[j], PARAMS[i]

            if i == j:  # weighted marginal
                for mag in MAGNITUDES:
                    v, w = data_1d[(p_col, mag)]
                    if len(v) < 10:
                        continue
                    xc, xh = weighted_hist(v, w, ranges[p_col][3])
                    ax.plot(xc, xh, color=cmap(norm(mag)), linewidth=1.0)
                ax.set_yticks([])
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.xaxis.set_major_formatter(FormatStrFormatter(axis_fmt(p_col)))
            else:  # weighted 2D KDE contours
                lo_x, hi_x, xgrid, _ = ranges[p_col]
                lo_y, hi_y, ygrid, _ = ranges[p_row]
                XX, YY = np.meshgrid(xgrid, ygrid)
                grid_pts = np.vstack([XX.ravel(), YY.ravel()])
                for mag in MAGNITUDES:
                    x, y, w = finite_samples(by_xi[mag], p_col, p_row)
                    if len(x) < 10:
                        print(f"skipping mag={mag}, {p_row} vs {p_col}: only {len(x)} valid samples")
                        continue
                    density = gaussian_kde(np.vstack([x, y]), weights=w)(grid_pts).reshape(XX.shape)
                    ax.contour(XX, YY, density, levels=hpd_thresholds(density.ravel(), levels),
                               colors=[cmap(norm(mag))], linewidths=0.9)
                ax.set_xlim(lo_x, hi_x)
                ax.set_ylim(lo_y, hi_y)
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.yaxis.set_major_locator(MaxNLocator(3))
                ax.xaxis.set_major_formatter(FormatStrFormatter(axis_fmt(p_col)))
                ax.yaxis.set_major_formatter(FormatStrFormatter(axis_fmt(p_row)))

            ax.tick_params(labelsize=6)
            if i < n - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(f"${{\\mathbf{{{tex(p_col)}}}}}$", fontsize=7)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            if j == 0 and i > 0:
                ax.set_ylabel(f"${{\\mathbf{{{tex(p_row)}}}}}$", fontsize=7)
            elif j > 0:
                ax.tick_params(labelleft=False)

    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.08, top=0.93)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=fig.add_axes([0.92, 0.15, 0.015, 0.7]), ticks=MAGNITUDES)
    cbar.ax.set_yticklabels([f"{m:g}" for m in MAGNITUDES])
    cbar.set_label("perturbation magnitude (σ)", fontsize=8)
    fig.suptitle(f"$\\mathbf{{{wl_actual:.3f}\\ \\mu m}}$", fontsize=12)
    print("saved", style.save(fig, f"{Path(__file__).stem}_{label}_{kernel}_{str(wl).replace('.', 'p')}", dpi=150))
    plt.close(fig)


if __name__ == "__main__":
    style.use(minor_ticks=True)
    for label, (kernel, wl) in FIGURES.items():
        corner_plot(label, kernel, wl)
