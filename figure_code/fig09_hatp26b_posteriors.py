"""Fig 9: HAT-P-26 b, log CO2 against log SO2 at xi = -3, 0, +3 sigma, for perturbations centred on the SO2 band
(4.041 um, left) and the CO2 band (4.293 um, right); (a) naive, (b) Gaussian (Sec. 7.3.1). Weighted KDE contours
and weighted marginals: MultiNest samples are not equally weighted.

Reads: E2_Hat-P-26b/outputs/runs.csv, posteriors/*.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.stats import gaussian_kde

import _CONFIG as config
from src import plotting as style
from src.plotting import hpd_thresholds, weighted_hist

E2 = Path(__file__).resolve().parents[1] / "E2_Hat-P-26b" / "outputs"
MAGNITUDES = np.array([-3.0, 0.0, 3.0])  # the xi grid of the HAT-P-26 b retrievals (Eq. 10)
ROW_LABELS = {"naive": ("a.", "Naive"), "gaussian": ("b.", "Gaussian")}
# mathtext names of the retrieval parameters (Figs 9, H1-H4)
TEX = {"SO2": "SO_2", "CO2": "CO_2", "CH4": "CH_4", "NH3": "NH_3", "C2H2": "C_2H_2", "C2H4": "C_2H_4", "H2S": "H_2S",
       "HCN": "HCN", "clouds_pressure": "Cloud Pressure", "T": "T_{isothermal}", "planet_radius": "R_{planet}"}


def tex(param):
    name = param.removeprefix("log_")
    return TEX.get(name, name)


def posteriors(kernel, centre_um):
    """The perturbation centre nearest centre_um, and {xi: posterior DataFrame} for that kernel and centre; the
    unperturbed retrieval is xi = 0 (Sec. 5.2: X'_m at xi = 0 is the same for every m)."""
    runs = pd.read_csv(E2 / "runs.csv").query("kernel == @kernel")
    centre = runs["centre_um"].iloc[int(np.argmin(np.abs(runs["centre_um"] - centre_um)))]
    out = {0.0: pd.read_csv(E2 / "posteriors" / "baseline.csv")}
    for r in runs[runs["centre_um"] == centre].itertuples():
        out[r.xi] = pd.read_csv(E2 / "posteriors" / f"{r.name}.csv")
    return centre, out


def finite_samples(posterior, *params):
    """The given columns of a posterior and its weights, without non-finite or zero-weight samples."""
    cols = [posterior[p].values for p in params] + [posterior["weight"].values]
    keep = np.all([np.isfinite(c) for c in cols], axis=0) & (cols[-1] > 0)
    return [c[keep] for c in cols]


def magnitude_colours():
    """One viridis colour per magnitude, and the norm that maps each xi on to its colour."""
    cmap = mcolors.ListedColormap(plt.cm.viridis(np.linspace(0, 1, len(MAGNITUDES))))
    mids = (MAGNITUDES[:-1] + MAGNITUDES[1:]) / 2
    lo = MAGNITUDES[0] - (mids[0] - MAGNITUDES[0])
    hi = MAGNITUDES[-1] + (MAGNITUDES[-1] - mids[-1])
    return cmap, mcolors.BoundaryNorm(np.concatenate([[lo], mids, [hi]]), cmap.N)


def figure(kernel):
    hist_bins, levels = 100, [1, 2]  # marginal histogram bins; contour levels in sigma
    mol_x, mol_y = "log_SO2", "log_CO2"

    wl_actuals, samples = [], {}
    for wl in config.TARGET_BANDS.values():
        wl_actual, by_xi = posteriors(kernel, wl)
        wl_actuals.append(wl_actual)
        for mag in MAGNITUDES:
            samples[(wl_actual, mag)] = finite_samples(by_xi[mag], mol_x, mol_y)
    n_wl = len(wl_actuals)

    # y (log CO2) is shared by both panels; each panel gets its own x range
    all_y = np.concatenate([s[1] for s in samples.values()])
    pad_y = 0.05 * (all_y.max() - all_y.min())
    ygrid = np.linspace(all_y.min() - pad_y, all_y.max() + pad_y, 200)
    y_edges = np.linspace(ygrid.min(), ygrid.max(), hist_bins + 1)
    xgrids, x_edges_by_wl = {}, {}
    for wl_actual in wl_actuals:
        xs = np.concatenate([samples[(wl_actual, mag)][0] for mag in MAGNITUDES])
        pad_x = 0.05 * (xs.max() - xs.min())
        xgrids[wl_actual] = np.linspace(xs.min() - pad_x, xs.max() + pad_x, 200)
        x_edges_by_wl[wl_actual] = np.linspace(xgrids[wl_actual].min(), xgrids[wl_actual].max(), hist_bins + 1)

    cmap, norm = magnitude_colours()
    fig = plt.figure(figsize=(7.1, 4.15))  # RASTI two-column width
    outer = fig.add_gridspec(1, n_wl + 1, width_ratios=[4] * n_wl + [0.18], left=0.08, right=0.93, bottom=0.15,
                             top=0.86, wspace=0.05)
    main_axes = []
    for i, wl_actual in enumerate(wl_actuals):
        block = outer[0, i].subgridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.06, hspace=0.06)
        ax = fig.add_subplot(block[1, 0], sharey=main_axes[0] if main_axes else None)
        ax_histx = fig.add_subplot(block[0, 0], sharex=ax)
        ax_histy = fig.add_subplot(block[1, 1], sharey=ax)
        XX, YY = np.meshgrid(xgrids[wl_actual], ygrid)
        grid_pts = np.vstack([XX.ravel(), YY.ravel()])

        for mag in MAGNITUDES:
            x, y, w = samples[(wl_actual, mag)]
            if len(x) < 10:
                print(f"skipping wl={wl_actual:.3f}, mag={mag}: only {len(x)} valid samples")
                continue
            color = cmap(norm(mag))
            density = gaussian_kde(np.vstack([x, y]), weights=w)(grid_pts).reshape(XX.shape)
            ax.contour(XX, YY, density, levels=hpd_thresholds(density.ravel(), levels), colors=[color], linewidths=1.1)
            xc, xh = weighted_hist(x, w, x_edges_by_wl[wl_actual])
            ax_histx.plot(xc, xh, color=color, linewidth=1.1)
            yc, yh = weighted_hist(y, w, y_edges)
            ax_histy.plot(yh, yc, color=color, linewidth=1.1)  # rotated: density on x

        ax.set_xlim(-12.5, -2.5)
        ax.set_ylim(-6.5, -0.5)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(integer=True))
            axis.set_major_formatter(FormatStrFormatter('%d'))
        ax.set_xlabel(f"$ \\log {{ \\mathbf{{ {tex(mol_x)}}} }} $")
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histx.set_ylim(0, 1.3)
        ax_histx.set_yticks([0, 0.4, 0.8, 1.2])
        ax.text(0.03, 0.03, f"$\\mathbf{{ {wl_actual:.3f} \\mu m }} $", transform=ax.transAxes, fontsize=9,
                verticalalignment="bottom")
        ax_histy.tick_params(axis="y", labelleft=False)
        ax_histy.set_xticks([0, 0.4, 0.8, 1.2])
        ax_histy.set_xlim(0, 1.3)
        ax_histy.set_xlabel("density", fontsize=9)
        if i == 0:
            ax.set_ylabel(f"$ \\log {{ \\mathbf{{ {tex(mol_y)}}} }} $")
            ax_histx.set_ylabel("density", fontsize=9)
        else:
            ax.tick_params(axis="y", labelleft=False)  # shared y axis: only the first panel labels it
            ax_histx.tick_params(axis="y", labelleft=False)
        main_axes.append(ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=fig.add_subplot(outer[0, -1]), label="perturbation magnitude (σ)", ticks=MAGNITUDES)
    cbar.ax.set_yticklabels([f"{m:g}" for m in MAGNITUDES])
    letter, name = ROW_LABELS[kernel]
    fig.text(0.0347, 0.8706, rf"$\mathbf{{{letter}}}$ {name} Perturbation", fontsize=9, ha="left", va="bottom")
    print("saved", style.save(fig, f"{Path(__file__).stem}_{kernel}", dpi=300))
    plt.close(fig)


if __name__ == "__main__":
    style.use(minor_ticks=True)
    for kernel in ("naive", "gaussian"):
        figure(kernel)
