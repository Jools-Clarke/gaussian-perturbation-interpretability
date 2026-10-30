"""Fig 7: HD 209458 b, H2O. Top: global (solid, Eq. 14) and local (dashed, Eq. 15) metric against wavelength. Bottom:
the contribution of H2O, native (faint) and averaged in each WFC3 bin. Shaded: the 1.27-1.32 um window between the
water bands, where the local metric peaks, and the 1.4 um band, where the global metric peaks (Sec. 7.2).

Reads: E1_HD-209458-b/outputs/metrics.csv, contributions.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import plotting as style

E1 = Path(__file__).resolve().parents[1] / "E1_HD-209458-b" / "outputs"
WINDOW_UM, BAND_UM = (1.27, 1.32), (1.33, 1.50)  # caption of Fig 7

if __name__ == "__main__":
    m = pd.read_csv(E1 / "metrics.csv")
    c = pd.read_csv(E1 / "contributions.csv")
    wl = m["wl_um"].values
    colour = style.COLOURS["H2O"]
    bins = style.edges(wl)
    binned = [c.loc[c["wl_um"].between(lo, hi), "H2O"].mean() for lo, hi in zip(bins[:-1], bins[1:])]

    style.use(minor_ticks=True)
    fig, (top, bot) = plt.subplots(2, 1, figsize=(3.5, 3.4), sharex=True, gridspec_kw={"height_ratios": [1, 0.75], "hspace": 0})
    for ax in (top, bot):
        ax.axvspan(*WINDOW_UM, color="0.6", alpha=0.35, lw=0)
        ax.axvspan(*BAND_UM, color=colour, alpha=0.12, lw=0)
    top.plot(wl, m["global_H2O"], "-", color=colour, label="global")
    top.plot(wl, m["local_H2O"], "--", color=colour, label="local")
    top.axhline(0, color="0.6", lw=0.5)
    for (lo, hi), text in ((WINDOW_UM, "1.3 $\\mu$m\nwindow"), (BAND_UM, "1.4 $\\mu$m\nH$_2$O band")):
        top.text(hi - 0.005, 0.97, text, transform=top.get_xaxis_transform(), rotation=90, ha="right", va="top", fontsize=7)
    top.set_ylabel(style.RHO % style.TEX["H2O"])
    top.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)

    bot.plot(c["wl_um"], c["H2O"], "-", color=colour, lw=0.4, alpha=0.2)
    bot.stairs(binned, bins, color=colour, lw=1.5)
    band = c.loc[c["wl_um"].between(bins[0], bins[-1]), "H2O"]
    bot.set_ylim(float(band.min()), float(band.max()))
    bot.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)  # shape only, as Fig D1
    bot.set_ylabel(r"H$_2$O contribution")
    bot.set_xlabel(r"Wavelength ($\mu$m)")
    bot.set_xlim(bins[0], bins[-1])
    fig.align_ylabels([top, bot])
    print("saved", style.save(fig, Path(__file__).stem))
