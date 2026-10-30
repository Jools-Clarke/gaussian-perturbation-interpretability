"""Experiment 2, step 5: the numbers the HAT-P-26 b figures plot (Sec. 7.3). Posterior statistics are weighted.

    metrics.csv              Figs 8, H1, 13: global (Eq. 14, prior grid) and local (Eq. 15, posterior draws) metric
                             of the nine gases of the grid
    perturbed_spectra.csv    Fig 8 top: the +-3 sigma Gaussian perturbations of the observation at 4.04 and 4.29 um
    kernels.csv              Fig 8 middle: the two windows (Eq. 9), on a fine grid and on the bins (on_bin)
    posterior_summary.csv    Table G3, Fig 10 (retrieved log CO2), Sec. 7.3.3 shifts: weighted median, p16, p84 and
                             mean of every parameter of all nine retrievals
    co2_band_changes.csv     Fig 11: changes relative to TauREx's best fit to x*, in sigma of each bin: the +3 sigma
                             perturbation at 4.29 um, CO2 +-0.5 dex, H2O -0.5 dex, and TauREx's best fit to the
                             perturbed spectrum
"""
from pathlib import Path

import numpy as np
import pandas as pd

import _CONFIG as config
from src.forward_model import wl_columns
from src.metrics import metric, summary
from src.perturbation import gaussian_window, perturbed_spectra

OUT = Path(__file__).resolve().parent / "outputs"
XI = 3.0  # Fig 8 perturbations, in sigma of the centre bin
# the parameters of the scale-height degeneracy (Sec. 7.3.3), printed as shifts from the baseline
SHIFT_COLUMNS = ["log_CO2", "log_SO2", "log_H2O", "log_H2S", "T", "planet_radius", "log_clouds_pressure", "mu_derived"]


if __name__ == "__main__":
    out = {}
    wl, x, err = np.loadtxt(config.HATP26B_OBS).T

    # Figs 8, H1, 13: step 3's posterior draws are already equally weighted, so the local metric needs no weights
    grid, post = pd.read_csv(OUT / "grid_spectra.csv"), pd.read_csv(OUT / "posterior_spectra.csv")
    cols = wl_columns(grid)
    assert np.allclose(np.array(cols, float), wl)
    gases = [c.removeprefix("log_") for c in grid.columns if c.startswith("log_")]
    ok = np.all(np.isfinite(post[cols].values), axis=1)
    out["metrics"] = pd.DataFrame({"wl_um": wl,
        **{f"global_{g}": metric(grid[cols].values, grid[f"log_{g}"].values, normalise=True) for g in gases},
        **{f"local_{g}": metric(post[cols].values[ok], post[f"log_{g}"].values[ok], normalise=False) for g in gases}})

    # Fig 8 top and middle
    fine = np.linspace(wl.min(), wl.max(), 1000)
    perturbed, kernels = [], {"wl_um": np.concatenate([fine, wl]), "on_bin": [False] * len(fine) + [True] * len(wl)}
    for gas, band_um in config.TARGET_BANDS.items():
        c = int(np.argmin(np.abs(wl - band_um)))
        for xi, depth in zip([-XI, XI], perturbed_spectra(x, err, wl, c, [-XI, XI], fwhm=config.GAUSSIAN_FWHM_UM)):
            perturbed += [dict(centre_um=wl[c], xi=xi, wl_um=w, depth=d) for w, d in zip(wl, depth)]
        kernels[gas] = gaussian_window(kernels["wl_um"], wl[c], config.GAUSSIAN_FWHM_UM)
    out["perturbed_spectra"], out["kernels"] = perturbed, kernels

    # Table G3, Fig 10, Sec. 7.3.3
    runs = pd.read_csv(OUT / "runs.csv")
    out["posterior_summary"] = []
    for name in runs["name"]:
        posterior = pd.read_csv(OUT / "posteriors" / f"{name}.csv")
        out["posterior_summary"] += [dict(run=name, param=p, **{s: summary(posterior, p, s) for s in ("median", "p16", "p84", "mean")})
                                     for p in posterior.columns if p != "weight"]
    medians = pd.DataFrame(out["posterior_summary"]).pivot(index="run", columns="param", values="median")
    print("weighted posterior medians, baseline:\n", medians.loc["baseline", SHIFT_COLUMNS].round(3).to_string())
    print("\nshift of the median from the baseline:\n",
          (medians.drop(index="baseline") - medians.loc["baseline"])[SHIFT_COLUMNS].round(3).to_string())

    # Fig 11
    models = pd.read_csv(OUT / "co2_band_models.csv")
    core = int(np.argmin(np.abs(wl - config.TARGET_BANDS["CO2"])))
    plus = runs.query("kernel == 'gaussian' and xi == @XI")
    plus = plus.loc[(plus["centre_um"] - wl[core]).abs().idxmin(), "name"]
    taurex = pd.read_csv(OUT / "spectra" / f"{plus}.csv")["best_fit"].values
    best = pd.read_csv(OUT / "spectra" / "baseline.csv")["best_fit"].values
    out["co2_band_changes"] = pd.DataFrame({
        "wl_um": wl, "sigma": err,
        "perturbation": XI * err[core] * gaussian_window(wl, wl[core], config.GAUSSIAN_FWHM_UM) / err,
        **{k: (models[k].values - models["best_fit"].values) / err for k in ("co2_plus", "co2_minus", "h2o_minus")},
        "taurex_plus3": (taurex - best) / err})

    for name, table in out.items():
        pd.DataFrame(table).to_csv(OUT / f"{name}.csv", index=False)
        print("wrote", OUT / f"{name}.csv")
