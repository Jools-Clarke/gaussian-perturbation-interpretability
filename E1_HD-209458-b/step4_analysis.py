"""Experiment 1, step 4: the numbers the HD 209458 b figures plot (Secs 5.1, 7.1, 7.2). Every posterior statistic is
weighted by the MultiNest weights.

    correlation_matrix.csv       Fig C1: Pearson correlation of every pair of bins across the min-max normalised grid
    correlation_separation.csv   Fig 2: that correlation against bin separation, in 0.05 um bins (Sec. 5.1)
    gaussian_fit.csv             Fig 2: FWHM of the Gaussian fitted to it, the kernel width (0.101 um in the paper),
                                 and the binning it was fitted with
    metrics.csv                  Figs 5, 6, 7, D1: global (Eq. 14) and local (Eq. 15) metric of each gas per bin
    response.csv                 Fig 4: response (Eq. 12), the shift of the posterior mean of each gas from the baseline
    response_gradient.csv        Figs 4, 6: its slope between neighbouring magnitudes (Sec. 7.1)
    importance.csv               Figs 7, D1: the mean of those slopes (Eq. 13)
    posterior_summary.csv        Figs 7, D1 labels, Table B3: weighted median, p16, p84 and mean of the baseline
    fraction_fitted.csv          Fig E1 (c): fraction of each perturbation the best fit reproduces (Eq. 16)
    fraction_fitted_example.csv  Fig E1 (a, b): the changes behind it for the 1.46 um, +2 sigma runs
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.optimize import curve_fit

from src.forward_model import wl_columns
from src.metrics import gaussian_fwhm, metric, summary
from src.perturbation import importance

OUT = Path(__file__).resolve().parent / "outputs"
GASES = ["H2O", "CH4", "CO", "CO2", "NH3"]
BIN_WIDTH = 0.05  # um, Fig 2 separation bins, centred on zero (caption)
MIN_BIN_COUNT = 5  # bins with this many pairs or fewer are left out of the fit (Fig 2)
EXAMPLE_UM, EXAMPLE_XI = 1.457, 2.0  # Fig E1 (a, b)


# ---------------------------------------------------------------- data access
def load_retrievals(kernel, sub="posteriors"):
    """Every retrieval of one kernel ('gaussian' | 'naive') as {(centre_um, xi): its posterior (or spectra) CSV}."""
    runs = pd.read_csv(OUT / "runs.csv")
    return {(round(r.centre_um, 6), r.xi): pd.read_csv(OUT / sub / f"{r.name}.csv")
            for r in runs[runs["kernel"] == kernel].itertuples()}


def load_baseline(sub="posteriors"):
    return pd.read_csv(OUT / sub / "baseline.csv")


def posterior_metric(mols):
    """Local metric (Eq. 15) of each gas, (len(mols), n_wl), over step 3's forward models at the posterior samples:
    raw spectra, weighted by the posterior. Returns it and the wavelengths."""
    p = pd.read_csv(OUT / "posterior_spectra.csv")
    cols = wl_columns(p)
    rho = np.array([metric(p[cols].values, p[f"log_{m}"].values, normalise=False, weight=p["weight"].values)
                    for m in mols])
    return rho, np.array(cols, float)


def hd209_response(mol, runs, base):
    """Response r_{m,p} (Eq. 12) of one gas to every centre m: the shift of its weighted posterior mean (f, Sec. 2.5)
    from the baseline's, (n_centre, K). Returns it with the centres (um) and the grid xi (K, including 0)."""
    xis = sorted({k[1] for k in runs} | {0.0})
    centres = sorted({k[0] for k in runs})
    p = f"log_{mol}"
    b = summary(base, p, "mean")
    response = np.array([[summary(runs[(c, xi)], p, "mean") - b if xi else 0.0 for xi in xis] for c in centres])
    return response, np.array(centres), np.array(xis)


# ---------------------------------------------------------------- the kernel width (Sec. 5.1; Figs 2, C1)
def correlation(grid):
    """Pearson correlation of every pair of bins across the min-max normalised grid spectra, (wl, wl2), and its mean
    and standard error in BIN_WIDTH bins of separation, centred on zero."""
    cols = wl_columns(grid)
    results = xr.DataArray(grid[cols].values, dims=("sample_id", "wl"), coords={"wl": np.array(cols, float)})

    norm = (results - results.min(dim = 'wl')) / (results.max(dim = 'wl') - results.min(dim = 'wl'))

    wl_vals = results['wl'].values
    rprs2_norm = norm.rename({'wl': 'wl2'})

    corr_da = xr.corr(norm, rprs2_norm, dim='sample_id')  # dims: (wl, wl2)
    corr_da = corr_da.assign_coords(wl=wl_vals, wl2=wl_vals)

    rows = []
    for i in range(corr_da.sizes['wl']):
        wl_val = corr_da['wl'].values[i]
        row = corr_da.isel(wl=i)
        dist = row['wl2'].values - wl_val          # distance from target wl
        row = row.assign_coords(wl2=dist).rename(wl2='distance')
        row = row.expand_dims(wl=[wl_val])
        rows.append(row)

    shifted = xr.concat(rows, dim='wl', join='outer')

    # Every pair as one point. The self-pairs (rho = 1) fill the zero-separation bin, where the Gaussian is 1 for any
    # FWHM, so they are drawn but do not move the fit.
    stacked = shifted.stack(pair=('wl', 'distance'))

    k = np.ceil(float(np.abs(stacked['distance']).max()) / BIN_WIDTH)
    bin_centers = np.arange(-k, k + 1) * BIN_WIDTH
    bin_edges = np.append(bin_centers - BIN_WIDTH / 2, bin_centers[-1] + BIN_WIDTH / 2)

    grouped = stacked.groupby_bins('distance', bins=bin_edges, labels=bin_centers)
    binned_n = grouped.count()
    assert int(binned_n.sum()) == len(wl_vals) ** 2  # every pair in exactly one bin
    binned = xr.Dataset({
        'corr': grouped.mean(skipna=True),
        # standard error, not standard deviation: the number of pairs falls with separation (Fig 2 caption)
        'corr_sem': (grouped.std(skipna=True, ddof=1) / np.sqrt(binned_n)).where(binned_n > MIN_BIN_COUNT, np.nan),
        'n': binned_n,
    })
    binned = binned.rename({'distance_bins': 'distance'})
    binned['distance'] = binned['distance'].astype(float)
    binned = binned.dropna(dim='distance', how='all', subset=['corr'])
    return corr_da, binned


# ---------------------------------------------------------------- how much of a perturbation the retrieval fits (Sec. 7.2.5)
def changes(base, ds):
    """Change in the spectrum and in TauREx's best fit from the baseline, in sigma of each bin, and the fraction of
    the perturbation fitted (Eq. 16): 1 if the best fit follows it, 0 if the retrieval treats it as noise."""
    s = base["error"].values
    dx = (ds["observed"].values - base["observed"].values) / s
    dbf = (ds["best_fit"].values - base["best_fit"].values) / s
    return dx, dbf, float(dbf @ dx / (dx @ dx))


if __name__ == "__main__":
    grid = pd.read_csv(OUT / "grid_spectra.csv")
    cols = wl_columns(grid)
    out = {}

    # Figs C1, 2
    corr_da, binned = correlation(grid)
    out["correlation_matrix"] = corr_da.to_pandas().rename_axis(index="wl_um", columns=None).reset_index()
    sep = pd.DataFrame({"separation_um": binned["distance"].values, "rho": binned["corr"].values,
                        "rho_err": binned["corr_sem"].values, "n": binned["n"].values})
    sep["excluded"] = ~(np.isfinite(sep["rho_err"]) & (sep["rho_err"] > 0))
    out["correlation_separation"] = sep
    fit = ~sep["excluded"]
    popt, pcov = curve_fit(gaussian_fwhm, sep["separation_um"][fit], sep["rho"][fit], p0=[0.1],
                           sigma=sep["rho_err"][fit], absolute_sigma=True)
    out["gaussian_fit"] = pd.DataFrame({"fwhm_um": popt, "fwhm_err_um": np.sqrt(np.diag(pcov)),
                                        "bin_width_um": BIN_WIDTH, "min_bin_count": MIN_BIN_COUNT})
    z = (popt[0] - 0.1) / np.sqrt(pcov[0, 0])  # is the fit consistent with the 0.1 um kernel?
    print(f"Fig 2: FWHM = {popt[0]:.4f} +/- {np.sqrt(pcov[0, 0]):.4f} um; z = {z:.2f} from 0.1 um, "
          f"two-sided p = {2 * (1 - stats.norm.cdf(abs(z))):.3f}")

    # Figs 5, 6, 7, D1: global metric over the prior grid (Eq. 14), local metric over the posterior (Eq. 15)
    glob = [metric(grid[cols].values, grid[f"log_{g}"].values, normalise=True) for g in GASES]
    loc, wl = posterior_metric(GASES)
    assert np.allclose(wl, np.array(cols, float))
    out["metrics"] = pd.DataFrame({"wl_um": wl, **{f"global_{g}": r for g, r in zip(GASES, glob)},
                                   **{f"local_{g}": r for g, r in zip(GASES, loc)}})

    # Figs 4, 6, 7, D1, E1: response, its gradient and the importance (Eqs. 12, 13), and the fraction fitted (Eq. 16)
    base, base_spectra = load_baseline(), load_baseline("spectra")
    for name in ("response", "response_gradient", "importance", "fraction_fitted", "fraction_fitted_example"):
        out[name] = []
    for kernel in ("gaussian", "naive"):
        runs, spectra = load_retrievals(kernel), load_retrievals(kernel, "spectra")
        for gas in GASES:
            response, centres, xis = hd209_response(gas, runs, base)
            gradient = np.diff(response, axis=1) / np.diff(xis)  # between neighbouring magnitudes
            for c, r, g, imp in zip(centres, response, gradient, importance(response, xis)):
                out["response"] += [dict(kernel=kernel, centre_um=c, xi=xi, gas=gas, response=v) for xi, v in zip(xis, r)]
                out["response_gradient"] += [dict(kernel=kernel, centre_um=c, xi_low=a, xi_high=b, gas=gas, gradient=v)
                                             for a, b, v in zip(xis[:-1], xis[1:], g)]
                out["importance"].append(dict(kernel=kernel, centre_um=c, gas=gas, importance=imp))
        out["fraction_fitted"] += [dict(kernel=kernel, centre_um=c, xi=xi, fraction=changes(base_spectra, s)[2])
                                   for (c, xi), s in spectra.items()]
        key = min(spectra, key=lambda k: abs(k[0] - EXAMPLE_UM) + abs(k[1] - EXAMPLE_XI) * 10)
        dx, dbf, _ = changes(base_spectra, spectra[key])
        out["fraction_fitted_example"] += [dict(kernel=kernel, centre_um=key[0], xi=key[1], wl_um=w,
                                                delta_spectrum_sigma=a, delta_best_fit_sigma=b)
                                           for w, a, b in zip(base_spectra["wl_um"], dx, dbf)]

    # Figs 7, D1 labels, Table B3
    out["posterior_summary"] = pd.DataFrame([dict(param=p, **{s: summary(base, p, s) for s in ("median", "p16", "p84", "mean")})
                                             for p in base.columns if p != "weight"])

    for name, table in out.items():
        pd.DataFrame(table).to_csv(OUT / f"{name}.csv", index=False)
        print("wrote", OUT / f"{name}.csv")
