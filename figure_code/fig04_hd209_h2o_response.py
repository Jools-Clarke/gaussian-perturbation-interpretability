"""Fig 4: HD 209458 b, Gaussian perturbation (Sec. 7.1). Top: the response of the posterior mean of log H2O (Eq. 12)
to each centre bin and magnitude. Bottom: its slope between neighbouring magnitudes; the mean down each column is the
importance (Eq. 13).

Reads: E1_HD-209458-b/outputs/response.csv, response_gradient.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import plotting as style
from src.plotting import colorbar, edges, gradient_map, wavelength_axis, xi_labels

E1 = Path(__file__).resolve().parents[1] / "E1_HD-209458-b" / "outputs"


def maps(kernel="gaussian", mol="H2O"):
    """Response (n_centre, K) and its gradient between neighbouring magnitudes (n_centre, K - 1), centres, xi."""
    r = pd.read_csv(E1 / "response.csv").query("kernel == @kernel and gas == @mol")
    g = pd.read_csv(E1 / "response_gradient.csv").query("kernel == @kernel and gas == @mol")
    response = r.pivot(index="centre_um", columns="xi", values="response")
    gradient = g.pivot(index="centre_um", columns="xi_low", values="gradient")
    return response.values, gradient.values, response.index.values, response.columns.values


def main():
    response, gradient, centres, xis = maps()
    fig = plt.figure(figsize=(3.4, 4.6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.045], hspace=0.05, wspace=0.06, figure=fig)
    top, cax_top = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    bot, cax_bot = fig.add_subplot(gs[1, 0], sharex=top), fig.add_subplot(gs[1, 1])
    v = np.nanmax(np.abs(response))
    mesh = top.pcolormesh(edges(centres), edges(xis), response.T, cmap="viridis", shading="flat", vmin=-v, vmax=v,
                         rasterized=True)
    colorbar(mesh, cax_top, r"$\Delta\log\mathbf{H_2O}$ (dex)")
    top.set_yticks(xis); top.set_yticklabels(xi_labels(xis))
    top.set_ylabel(r"$\xi$ ($\sigma$)")
    top.tick_params(labelbottom=False)
    gradient_map(bot, cax_bot, gradient, centres, xis)
    wavelength_axis(bot, centres)
    print("Fig 4: response range", np.round([response.min(), response.max()], 2), "dex; saved",
          style.save(fig, Path(__file__).stem, dpi=300))
    plt.close(fig)


if __name__ == "__main__":
    style.use()
    main()
