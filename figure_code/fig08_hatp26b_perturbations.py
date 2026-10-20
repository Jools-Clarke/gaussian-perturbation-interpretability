"""Fig 8: HAT-P-26 b (Sec. 7.3). Top: the observation and its +-3 sigma Gaussian perturbations at the SO2 (4.04 um) and
CO2 (4.29 um) bands. Middle: the two windows (Eq. 9). Bottom: global (solid) and local (dashed) metric of SO2 and CO2,
whose signs predict the direction of the response.

Reads: DATA/HAT-P-26b/HAT-P-26b_NIRSpec_G395H.dat, E2_Hat-P-26b/outputs/perturbed_spectra.csv, kernels.csv, metrics.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import _CONFIG as config
from src import plotting as style

E2 = Path(__file__).resolve().parents[1] / "E2_Hat-P-26b" / "outputs"
XI = 3.0
TARGETS = config.TARGET_BANDS


def main():
    wl, x, err = np.loadtxt(config.HATP26B_OBS).T
    metrics = pd.read_csv(E2 / "metrics.csv")
    perturbed = pd.read_csv(E2 / "perturbed_spectra.csv")
    kernels = pd.read_csv(E2 / "kernels.csv")
    fine, on_bin = kernels[~kernels["on_bin"]], kernels[kernels["on_bin"]]
    assert np.allclose(metrics["wl_um"].values, wl) and np.allclose(on_bin["wl_um"].values, wl)
    mols = list(TARGETS)
    rho = {m: metrics[f"global_{m}"].values for m in mols}
    local = {m: metrics[f"local_{m}"].values for m in mols}

    fig, (top, mid, bot) = plt.subplots(3, 1, figsize=(7, 5.2), sharex=True,
                                        gridspec_kw={"height_ratios": [2, 0.6, 1.8], "hspace": 0.05})
    top.errorbar(wl, x, yerr=err, fmt="o", color="k", ecolor="gray", capsize=0, markersize=3, elinewidth=0.6,
                 label=r"$\mathbf{x}_\star$ Gressier et al. 2025", zorder=4)
    for mol, centre in TARGETS.items():
        c = int(np.argmin(np.abs(wl - centre)))
        colour = style.COLOURS[mol]
        at = perturbed[np.isclose(perturbed["centre_um"], wl[c])]
        up, down = (at[at["xi"] == xi]["depth"].values for xi in (XI, -XI))
        near = np.flatnonzero(on_bin[mol].values > 0.01)  # bins the perturbation visibly moves
        span = slice(max(near[0] - 1, 0), near[-1] + 2)
        top.plot(wl[near], up[near], "^", color=colour, markersize=3)
        top.plot(wl[near], down[near], "v", color=colour, markersize=3)
        top.fill_between(wl[span], down[span], up[span], color=colour, alpha=0.25, lw=0,
                         label=rf"$\mathbf{{x}}^\prime$ ($\lambda={wl[c]:.2f}\,\mu$m)")
        mid.plot(fine["wl_um"], fine[mol], color=colour)
        for ax in (top, mid, bot):
            ax.axvline(wl[c], color=colour, ls="--", lw=0.8, alpha=0.7)
    top.set_ylabel(r"$(R_p/R_s)^2$")
    top.legend(loc="upper right")
    mid.set_ylabel(r"$w_{\mathrm{gauss.}}$")

    for m in mols:
        bot.plot(wl, rho[m], "-", color=style.COLOURS[m], lw=1.2)
        bot.plot(wl, local[m], "--", color=style.COLOURS[m], lw=1.2)
        for name, centre in TARGETS.items():
            c = int(np.argmin(np.abs(wl - centre)))
            print(f"{m:4s} at {wl[c]:.3f} um: global {rho[m][c]:+.2f}, local {local[m][c]:+.2f}")
    bot.axhline(0, color="0.6", lw=0.5)
    bot.set_ylabel(r"$\rho\left(\mathcal{D}_{m}, \log \mathbf{Y}\right)$")
    bot.set_xlabel(r"Wavelength ($\mu$m)")
    lo, hi = bot.get_ylim()
    bot.set_ylim(lo - 0.15 * (hi - lo), hi)
    handles = [Line2D([], [], color=style.COLOURS[m], lw=1.2, label=rf"$\mathrm{{{style.TEX[m]}}}$") for m in mols]
    handles += [Line2D([], [], color="0.3", lw=1.2, label="global: prior forward models"),
                Line2D([], [], color="0.3", ls="--", lw=1.2, label="local: posterior forward models")]
    leg = bot.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=True, handletextpad=0.8, columnspacing=1.2,
                     framealpha=1.0, edgecolor="none", facecolor="white", borderpad=0.3)
    leg.set_zorder(5)

    for ax in (top, mid, bot):
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    fig.align_ylabels([top, mid, bot])
    print("saved", style.save(fig, Path(__file__).stem))
    plt.close(fig)


if __name__ == "__main__":
    style.use(minor_ticks=True)
    main()
