"""Fig 10: correlation between the depth in the 4.293 um bin (the CO2 band core) and log CO2, within sliding 1-dex
windows of log CO2 across the CNN's training set; it turns over near the retrieved log CO2 (dashed) (Sec. 7.3.3).

Reads: E3_CNN/outputs/co2_core_correlation.csv, E2_Hat-P-26b/outputs/posterior_summary.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from E3_CNN.step5_analysis import WINDOW_DEX
from src import plotting as style

ROOT = Path(__file__).resolve().parents[1]


def window(windows, baseline_log_co2):
    fig, ax = plt.subplots(figsize=(3.4, 3.3))
    ax.plot(windows["log_co2"].values, windows["rho"].values, color="k", lw=1.3)
    ax.axhline(0, color="0.5", lw=0.5)
    ax.axvline(baseline_log_co2, color="0.3", lw=0.9, ls="--", label=r"retrieved log CO$_2$")
    ax.set_xlabel(r"log CO$_2$ (centre of %g dex window)" % WINDOW_DEX)
    ax.set_ylabel(rf"$\rho$(transit depth at {windows['core_um'].iloc[0]:.3f} $\mu$m, log CO$_2$)")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_box_aspect(1)
    ax.minorticks_on()
    return fig


if __name__ == "__main__":
    style.use()
    windows = pd.read_csv(ROOT / "E3_CNN" / "outputs" / "co2_core_correlation.csv")
    summary = pd.read_csv(ROOT / "E2_Hat-P-26b" / "outputs" / "posterior_summary.csv")
    baseline = summary.query("run == 'baseline' and param == 'log_CO2'")["median"].item()
    fig = window(windows, baseline)
    print("saved", style.save(fig, Path(__file__).stem, dpi=300))
    plt.close(fig)
