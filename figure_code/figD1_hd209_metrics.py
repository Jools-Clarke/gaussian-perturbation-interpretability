"""Fig D1 (and, via plot(), Fig H1): for each gas, the global (solid, Eq. 14) and local (dashed, Eq. 15) metric against
wavelength, and below it the gas's contribution (in colour) over the other gases' (grey) (Sec. 7.2).

Reads: E1_HD-209458-b/outputs/metrics.csv, contributions.csv
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from src import plotting as style

E1 = Path(__file__).resolve().parents[1] / "E1_HD-209458-b" / "outputs"


def plot(outputs, mols, ncols, width, per_gas, name):
    """One metric panel and one contribution panel per gas, in ncols columns of the given width and height per gas (in)."""
    m = pd.read_csv(outputs / "metrics.csv")
    c = pd.read_csv(outputs / "contributions.csv")
    wl = m["wl_um"].values
    glob = np.array([m[f"global_{g}"].values for g in mols])
    loc = np.array([m[f"local_{g}"].values for g in mols])
    colours = [style.COLOURS[g] for g in mols]

    style.use(minor_ticks=True)
    nrows = math.ceil(len(mols) / ncols)
    fig = plt.figure(figsize=(width, nrows * per_gas))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.05, wspace=0.45)
    last_row = [max(i // ncols for i in range(len(mols)) if i % ncols == col) for col in range(ncols)]
    ref, columns = None, [[] for _ in range(ncols)]
    for i, (mol, colour) in enumerate(zip(mols, colours)):
        row, col = divmod(i, ncols)
        inner = outer[row, col].subgridspec(2, 1, height_ratios=[1, 0.6], hspace=0)
        top = fig.add_subplot(inner[0], sharex=ref)
        bot = fig.add_subplot(inner[1], sharex=top)
        ref = ref or top
        columns[col] += [top, bot]
        label = style.TEX.get(mol, mol)
        top.plot(wl, glob[i], "-", color=colour, lw=1.2)
        top.plot(wl, loc[i], "--", color=colour, lw=1.2)
        top.axhline(0, color="0.6", lw=0.5)
        lo, hi = top.get_ylim()  # each panel scales to its own curves
        top.set_ylim(lo, hi + 0.3 * (hi - lo))  # headroom for the gas name
        top.yaxis.set_major_locator(MaxNLocator(nbins=4))
        top.set_ylabel(style.RHO % label)
        top.text(0.02, 0.90, rf"$\mathbf{{{label}}}$", transform=top.transAxes, ha="left", va="top", fontsize=8)
        for other in mols:
            if other != mol:
                bot.plot(c["wl_um"], c[other], "-", color="k", lw=0.4, alpha=0.2)
        bot.plot(c["wl_um"], c[mol], "-", color=colour, lw=0.4, alpha=0.9)
        band = c.loc[c["wl_um"].between(wl.min(), wl.max()), mol]
        bot.set_ylim(float(band.min()), float(band.max()) * 1.01)
        bot.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)  # shape only; caption says what it is
        plt.setp(top.get_xticklabels(), visible=False)
        if row != last_row[col]:
            plt.setp(bot.get_xticklabels(), visible=False)
        else:
            bot.set_xlabel(r"Wavelength ($\mu$m)")
    ref.set_xlim(wl.min(), wl.max())
    for column in columns:
        fig.align_ylabels(column)
    fig.legend(handles=[Line2D([0], [0], color="0.3", lw=1.2, ls="-", label="global: prior forward models"),
                        Line2D([0], [0], color="0.3", lw=1.2, ls="--", label="local: posterior forward models")],
               loc="lower center", bbox_to_anchor=(0.5, 0.885 if ncols == 1 else 0.89), ncol=1 if ncols == 1 else 2,
               fontsize=7, handlelength=1.4, handletextpad=0.5, columnspacing=1.5)
    out = style.save(fig, name)
    plt.close(fig)
    for mol, a, b in zip(mols, glob, loc):
        print(f"{mol:5s} r(global, local) = {np.corrcoef(a, b)[0, 1]:+.2f}")
    print("saved", out)


if __name__ == "__main__":
    plot(E1, ["H2O", "CH4", "CO", "CO2", "NH3"], ncols=1, width=3.5, per_gas=1.7, name=Path(__file__).stem)
