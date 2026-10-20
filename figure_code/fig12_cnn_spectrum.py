"""Fig 12: the CNN retrieval of the simulated HAT-P-26 b observation (Sec. 8.1): x* +- sigma, the TauREx spectrum
rebuilt from the predictions (median, 1 and 2 sigma at R = 200), residuals in sigma, and an inset of the log SO2 /
log CO2 posterior with the input values in red.

Reads: E2_Hat-P-26b/outputs/simulated_observation.csv, E3_CNN/outputs/reconstructed_spectrum.csv,
       reconstructed_residuals.csv, posterior_draws.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb

import _CONFIG as config
from E3_CNN.step3_posterior import RESOLUTION
from figJ1_cnn_corner import ACCENT, LABELS, TEXTWIDTH_IN, contour_style, plot_ranges
from src import plotting as style

ROOT = Path(__file__).resolve().parents[1]
E2, E3 = ROOT / "E2_Hat-P-26b" / "outputs", ROOT / "E3_CNN" / "outputs"
PAIR = ["CO2", "SO2"]
SPECTRUM_SIGMAS = (1, 2)  # bands of the spectrum plots and contours of their CO2 / SO2 inset
BAND_TINT = {1: 0.375, 2: 0.225, 3: 0.125}  # how much of ACCENT (vs white) fills the spectrum's 1, 2, 3 sigma bands
# CO2 / SO2 contour inset in the spectrum panel: left, bottom, width, height of its axes (axes fraction).
# The y axis is raised so that no data lie under the inset or its labels.
INSET_RECT = (0.81, 0.55, 0.175, 0.42)


def tint(color, amount):
    """color mixed with white (amount 1 = the colour itself); opaque, so nested bands don't add up."""
    return tuple(1 - amount * (1 - np.array(to_rgb(color))))


def draw_contour_inset(ax, samples, inputs, labels):
    """The SPECTRUM_SIGMAS contours of two retrieved log VMRs, input values marked, inset at INSET_RECT in ax."""
    ranges = plot_ranges(samples, inputs)
    ax_in = ax.inset_axes(INSET_RECT)
    corner.hist2d(samples[:, 0], samples[:, 1], ax=ax_in, range=ranges, **contour_style(SPECTRUM_SIGMAS))
    ax_in.axvline(inputs[0], color=ACCENT, lw=1.0)
    ax_in.axhline(inputs[1], color=ACCENT, lw=1.0)
    ax_in.plot(*inputs, "s", ms=3, color=ACCENT)
    ax_in.set(xlim=ranges[0], ylim=ranges[1])
    ax_in.set_xlabel(labels[0], fontsize=7, labelpad=1)
    ax_in.set_ylabel(labels[1], fontsize=7, labelpad=1)
    ax_in.tick_params(labelsize=6)
    ax_in.locator_params(nbins=3)
    return ax_in


def draw_spectrum(ax, ax_res, obs, spectrum, residuals):
    """Recovered spectrum at uniform resolution (median and SPECTRUM_SIGMAS bands) against x_star +- sigma, and the
    residuals against it binned like the data, in sigma.

    Returns the wavelengths and tops of everything in the spectrum panel, to keep an inset clear of it.
    """
    wl, x_star, sigma = obs["wl_um"].values, obs["depth"].values, obs["sigma"].values
    wl_uniform = spectrum["wl_um"].values
    for k in SPECTRUM_SIGMAS[::-1]:  # widest first, so the narrower bands sit on top
        ax.fill_between(wl_uniform, spectrum[f"low_{k}sigma"], spectrum[f"high_{k}sigma"],
                        color=tint(ACCENT, BAND_TINT[k]), lw=0, label=rf"{k}$\sigma$")
    ax.plot(wl_uniform, spectrum["median"], color=ACCENT, lw=1.0, label=f"recovered spectrum (median, $R$ = {RESOLUTION:g})")
    ax.errorbar(wl, x_star, yerr=sigma, fmt="o", ms=2.5, color="k", elinewidth=0.7, capsize=0, zorder=4,
                label=r"input spectrum $x_\star \pm \sigma$")
    ax.set_ylabel(r"$(R_p/R_s)^2$")
    ax.locator_params(axis="y", nbins=5)

    # residuals (x* - model) / sigma: the median as points, the model's bands about the median model as before
    residual = residuals["median"].values
    for k in SPECTRUM_SIGMAS[::-1]:
        ax_res.fill_between(wl, residual - residuals[f"high_{k}sigma"], residual - residuals[f"low_{k}sigma"],
                            color=tint(ACCENT, BAND_TINT[k]), lw=0)
    ax_res.axhline(0, color=ACCENT, lw=1.0)
    ax_res.errorbar(wl, residual, yerr=1, fmt="o", ms=2.5, color="k", elinewidth=0.7, capsize=0, zorder=4)
    ax_res.set_ylim(np.array([-1, 1]) * np.max(np.abs(ax_res.get_ylim())))  # symmetric about zero
    ax_res.set_ylabel(r"residual ($\sigma$)")
    ax_res.set_xlabel(r"Wavelength ($\mu$m)")

    for mol, band in config.TARGET_BANDS.items():
        for a in (ax, ax_res):
            a.axvline(band, color="0.35", lw=0.7, ls=":", zorder=0)
        ax.text(band + 0.012, 0.03, LABELS[mol], transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=8,
                color=style.COLOURS[mol])
    ax.tick_params(labelbottom=False)
    ax.set_xlim(wl[0] - 0.03, wl[-1] + 0.03)
    top = max(SPECTRUM_SIGMAS)
    return np.concatenate([wl_uniform, wl]), np.concatenate([spectrum[f"high_{top}sigma"].values, x_star + sigma])


def spectrum_figure(obs, spectrum, residuals, inset):
    """The spectrum and residual panels; inset = (samples, inputs, labels) of two gases adds their contours."""
    fig = plt.figure(figsize=(TEXTWIDTH_IN, 4.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06, left=0.1, right=0.98, bottom=0.12, top=0.9)
    ax = fig.add_subplot(grid[0])
    ax_res = fig.add_subplot(grid[1], sharex=ax)
    x_top, y_top = draw_spectrum(ax, ax_res, obs, spectrum, residuals)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7,
              frameon=False)  # reversed: input spectrum first
    fig.align_ylabels([ax, ax_res])

    # raise the top of the y axis until the data under the inset and its labels lie below them
    ax_in = draw_contour_inset(ax, *inset)
    box = ax_in.get_tightbbox(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())  # axes fraction
    x_lo, x_hi = ax.get_xlim()
    under = (x_top > x_lo + box.x0 * (x_hi - x_lo)) & (x_top < x_lo + box.x1 * (x_hi - x_lo))
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, max(y_hi, y_lo + (y_top[under].max() - y_lo) / (box.y0 - 0.03)))
    return fig


if __name__ == "__main__":
    style.use()
    obs = pd.read_csv(E2 / "simulated_observation.csv")
    spectrum, residuals = pd.read_csv(E3 / "reconstructed_spectrum.csv"), pd.read_csv(E3 / "reconstructed_residuals.csv")
    assert np.allclose(residuals["wl_um"].values, obs["wl_um"].values)
    draws = pd.read_csv(E3 / "posterior_draws.csv")
    inset = (draws[[f"log_{m}" for m in PAIR]].values, np.array([config.HATP26B_LOG_VMR[m] for m in PAIR]),
             [f"log {LABELS[m]}" for m in PAIR])
    fig = spectrum_figure(obs, spectrum, residuals, inset)
    print("saved", style.save(fig, Path(__file__).stem, dpi=300, pad_inches=0.1))  # the original savefig default pad
    plt.close(fig)
