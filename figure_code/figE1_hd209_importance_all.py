"""Fig E1: Fig 6 for each of the five HD 209458 b gases.

Reads: E1_HD-209458-b/outputs/importance.csv, metrics.csv, posterior_summary.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fig06_hd209_h2o_importance import MOLS, XLABEL, YLABEL, data, panel
from src import plotting as style


def main(imp, rho, retrieved):
    colours = style.COLOURS
    markers = {"H2O": "o", "CH4": "s", "CO": "^", "CO2": "D", "NH3": "v"}
    ncols, nrows = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(6, nrows * 2.8), sharex=True, sharey=True, squeeze=False,
                             gridspec_kw={"wspace": 0, "hspace": 0})
    axes = axes.flatten()
    for ax, m in zip(axes, MOLS):
        r = panel(ax, m, imp[m], rho[m], retrieved[m], colour=colours[m], marker=markers[m], fit_colour="k")
        print(f"E1: {m:4s} r = {r:.2f}")
    for ax in axes[len(MOLS):]:
        ax.axis("off")
    axes[len(MOLS) - ncols].tick_params(labelbottom=True)  # the panel above the blank one carries its column's ticks
    axes[0].set_xlim(-1, 1); axes[0].set_ylim(-0.5, 0.5)
    axes[0].set_xticks([-0.8, -0.4, 0, 0.4, 0.8]); axes[0].set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    fig.supxlabel(XLABEL, fontsize=9, x=0.55)
    fig.supylabel(YLABEL, fontsize=9, y=0.53)
    fig.tight_layout()
    print("saved", style.save(fig, Path(__file__).stem))
    plt.close(fig)


if __name__ == "__main__":
    style.use()
    imp, rho, retrieved, _ = data()
    main(imp, rho, retrieved)
