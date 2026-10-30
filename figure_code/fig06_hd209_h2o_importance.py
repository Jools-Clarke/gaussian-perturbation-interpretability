"""Fig 6 (and the shared panel of Fig E1): HD 209458 b, the importance of each bin for H2O under Gaussian perturbation
against its local metric, with the linear fit and its Pearson r, and the retrieved log H2O (Sec. 7.2.4).

Reads: E1_HD-209458-b/outputs/importance.csv, metrics.csv, posterior_summary.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from scipy import stats

from src import plotting as style

E1 = Path(__file__).resolve().parents[1] / "E1_HD-209458-b" / "outputs"
MOLS = ["H2O", "CH4", "NH3", "CO", "CO2"]
XLABEL = r"importance (dex $\sigma^{-1}$)"  # Eq. 13; units and wording as Fig L1
YLABEL = r"$\rho_{\mathrm{post}}\left(\mathcal{D}_{m}, \log \mathbf{Y}\right)$"


def data():
    """Importance (Gaussian kernel) and local metric of each gas per bin, the retrieved value labels, wavelengths."""
    imp = pd.read_csv(E1 / "importance.csv").query("kernel == 'gaussian'").pivot(index="centre_um", columns="gas",
                                                                                     values="importance")
    m = pd.read_csv(E1 / "metrics.csv")
    assert np.allclose(imp.index.values, m["wl_um"].values, atol=2e-3)
    summary = pd.read_csv(E1 / "posterior_summary.csv").set_index("param")
    retrieved = {}
    for g in MOLS:
        med, lo, hi = summary.loc[f"log_{g}", ["median", "p16", "p84"]]
        retrieved[g] = rf"${med:.2f}^{{+{hi - med:.2f}}}_{{-{med - lo:.2f}}}$"
    return {g: imp[g].values for g in MOLS}, {g: m[f"local_{g}"].values for g in MOLS}, retrieved, m["wl_um"].values


def panel(ax, mol, x, y, retrieved, colour="k", marker="o", fit_colour="firebrick"):
    ax.axhline(0, color="k", lw=0.7, ls=":", zorder=2)
    ax.axvline(0, color="k", lw=0.7, ls=":", zorder=2)
    ax.scatter(x, y, s=16, marker=marker, color=colour, linewidths=0, zorder=3)
    fit = stats.linregress(x, y)
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, fit.slope * xx + fit.intercept, color=fit_colour, ls="--", lw=1, zorder=3)
    ax.annotate(rf"$\mathbf{{{style.TEX[mol]}}}$", xy=(0.05, 0.95), xycoords="axes fraction", ha="left", va="top", fontsize=9)
    ax.annotate(retrieved, xy=(0.05, 0.88), xycoords="axes fraction", ha="left", va="top", fontsize=9)
    ax.annotate(rf"$r$ = {fit.rvalue:.2f}", xy=(0.95, 0.05), xycoords="axes fraction", ha="right", va="bottom", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_minor_locator(AutoMinorLocator()); ax.yaxis.set_minor_locator(AutoMinorLocator())
    return fit.rvalue


def fig7(imp, rho, retrieved):
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    r = panel(ax, "H2O", imp["H2O"], rho["H2O"], retrieved["H2O"], colour=style.COLOURS["H2O"])
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4]); ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_xlabel(r"importance, $\mathrm{H_2O}$ (dex $\sigma^{-1}$)"); ax.set_ylabel(YLABEL)
    print(f"Fig 6: H2O r = {r:.2f}; saved", style.save(fig, Path(__file__).stem))
    plt.close(fig)


if __name__ == "__main__":
    style.use()
    imp, rho, retrieved, _ = data()
    fig7(imp, rho, retrieved)
