"""Experiment 2, step 3: the HAT-P-26 b forward models that need the baseline retrieval.

1. Local metric (Eq. 15, Fig G1): forward models at 4 000 draws from the baseline posterior, resampled by weight so
   that they are equally weighted, with all 13 parameters set.
       -> outputs/posterior_spectra.csv      one row per draw: the 13 parameters, then one column per bin
2. Simulated observation (Secs 6 and 8.1): the forward model at the posterior means (Table F3), with the observed
   sigma. It stands in for the real spectrum, whose instrumental and reduction artefacts no forward model contains,
   in Fig 3 and as the CNN's input.
       -> outputs/simulated_observation.csv  wl_um, depth, sigma
3. Fig 11: TauREx's best fit (MAP sample) to the unperturbed spectrum, and the same with CO2 +-0.5 dex or H2O -0.5 dex.
       -> outputs/co2_band_models.csv        wl_um, best_fit, co2_plus, co2_minus, h2o_minus
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import _CONFIG as config
from src import forward_model

OUT = Path(__file__).resolve().parent / "outputs"
N = 4000  # equal-weight posterior draws for the local metric
PRIOR = [f"log_{g}" for g in config.MOLECULES] + ["T", "planet_radius", "log_clouds_pressure"]  # the 13 retrieval parameters
SINGLE = {"co2_plus": ("log_CO2", 0.5), "co2_minus": ("log_CO2", -0.5), "h2o_minus": ("log_H2O", -0.5)}
_model = _binner = _wngrid = None


def _init():
    global _model, _binner, _wngrid
    _model, _binner, _wngrid = forward_model.load_model(config.HATP26B_PAR, config.HATP26B_OBS)


def spectrum(theta):
    """theta: {retrieval parameter: value} -> binned spectrum, ascending wavelength."""
    return forward_model.spectrum(_model, _binner, _wngrid, forward_model.model_params(theta))


def _spectrum(theta):
    try:
        return spectrum(theta)
    except Exception:  # e.g. a draw whose gases sum past a VMR of 1; left out of the metric
        return np.full(len(_wngrid), np.nan)


def _work(rows):
    return np.array([_spectrum(dict(zip(PRIOR, r))) for r in rows])


if __name__ == "__main__":
    posterior = pd.read_csv(OUT / "posteriors" / "baseline.csv")
    best = pd.read_csv(OUT / "spectra" / "baseline.csv")
    run = pd.read_csv(OUT / "runs.csv").set_index("name").loc["baseline"]
    _init()
    wl = np.sort(10000 / _wngrid)
    columns = [f"{w:.6f}" for w in wl]

    # 1. local metric
    rng = np.random.default_rng(3)
    idx = rng.choice(len(posterior), N, p=posterior["weight"].values)
    theta = posterior[PRIOR].values[idx]
    with Pool(config.N_WORKERS, initializer=_init) as pool:
        spectra = np.concatenate(pool.map(_work, np.array_split(theta, 80)))
    assert np.isfinite(spectra[:, 0]).mean() > 0.5, "most forward models failed: a setup error, not bad draws"
    pd.DataFrame(np.hstack([theta, spectra]), columns=PRIOR + columns).to_csv(OUT / "posterior_spectra.csv", index=False)
    print("wrote", OUT / "posterior_spectra.csv", "failed:", int(np.isnan(spectra[:, 0]).sum()))

    # 2. simulated observation
    means = {f"log_{m}": v for m, v in config.HATP26B_LOG_VMR.items()} | config.FIXED_PARAMS
    depth = spectrum(means)
    wl_obs, observed, sigma = np.loadtxt(config.HATP26B_OBS).T
    assert np.allclose(wl_obs, wl)
    pd.DataFrame({"wl_um": wl, "depth": depth, "sigma": sigma}).to_csv(OUT / "simulated_observation.csv", index=False)
    print(f"wrote {OUT / 'simulated_observation.csv'} (chi^2/N against the real observation: "
          f"{np.mean(((depth - observed) / sigma) ** 2):.2f})")

    # 3. Fig 11: the check makes sure the MAP sample rebuilds TauREx's own best fit
    base = {p: run[f"map_{p}"] for p in PRIOR}
    f0 = spectrum(base)
    assert np.allclose(f0, best["best_fit"].values, rtol=1e-5), "forward model does not reproduce TauREx's stored best fit"
    single = {name: spectrum({**base, key: base[key] + step}) for name, (key, step) in SINGLE.items()}
    pd.DataFrame({"wl_um": wl, "best_fit": f0, **single}).to_csv(OUT / "co2_band_models.csv", index=False)
    print("wrote", OUT / "co2_band_models.csv")
