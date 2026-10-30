"""Fig 5: HD 209458 b, log H2O. Response gradient under naive (top) and Gaussian (bottom) perturbation, each on its own
colour scale, with the local metric of H2O between them (Sec. 7.2.5).

Reads: E1_HD-209458-b/outputs/response_gradient.csv, metrics.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from fig04_hd209_h2o_response import maps
from src import plotting as style
from src.plotting import gradient_map, wavelength_axis

E1 = Path(__file__).resolve().parents[1] / "E1_HD-209458-b" / "outputs"


def metrics_panel(ax, centres):
    """Local metric of H2O (posterior forward models, raw, weighted), dashed in the H2O colour as Fig D1."""
    m = pd.read_csv(E1 / "metrics.csv")
    assert np.allclose(m["wl_um"].values, centres, atol=2e-3)
    ax.axhline(0, color="0.6", lw=0.5)
    ax.plot(centres, m["local_H2O"].values, "--", color=style.COLOURS["H2O"], lw=1.2)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.set_ylabel(r"$\rho_{\mathrm{post}}(\mathcal{D}_{m}, \log\mathbf{H_2O})$")


def figure():
    fig = plt.figure(figsize=(3.4, 4))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 0.04], height_ratios=[1, 0.7, 1], hspace=0.1, wspace=0.02, figure=fig)
    axes, vmax = [], {}
    for row, name in enumerate(["naive", "metric", "gaussian"]):
        ax = fig.add_subplot(gs[row, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        if name == "metric":
            metrics_panel(ax, centres)
            continue
        _, gradient, centres, xis = maps(name)
        vmax[name] = np.nanmax(np.abs(gradient))
        gradient_map(ax, fig.add_subplot(gs[row, 1]), gradient, centres, xis)
        ax.text(0.015, 0.96, name.capitalize(), transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    wavelength_axis(axes[-1], centres)
    fig.align_ylabels(axes)
    name = Path(__file__).stem
    print(name, {n: round(float(v), 3) for n, v in vmax.items()}, "saved", style.save(fig, name, dpi=300))
    plt.close(fig)


if __name__ == "__main__":
    style.use()
    figure()
