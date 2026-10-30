"""Experiment 3, step 3: the CNN's posterior for the simulated HAT-P-26 b observation (Sec. 8.1; Figs 12, K1).

1. 20 000 noisy realisations x* + sigma_m N(0, 1), each passed through the network once with its own fresh dropout
   mask, so the posterior combines the network's uncertainty with the observational noise.
       -> outputs/posterior_draws.csv            one row per draw: log_<gas> x10
2. TauREx spectra of the first 5 000 draws (training-set T, Rp, cloud top), at R = 200 and on the 60 bins: median
   and 1, 2 sigma percentiles over the draws whose forward model ran.
       -> outputs/reconstructed_spectrum.csv     wl_um (R = 200), median, low/high_1sigma, low/high_2sigma
       -> outputs/reconstructed_residuals.csv    wl_um (60 bins), the same percentiles of (x* - model) / sigma
Prints the numbers of Sec. 8.1: median and 1 sigma range of each gas against the input value.
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import _CONFIG as config
from src import forward_model
from src.cnn import BlackBox, fixed_noisy_copies

OUT = Path(__file__).resolve().parent / "outputs"
OBSERVATION = Path(__file__).resolve().parents[1] / "E2_Hat-P-26b" / "outputs" / "simulated_observation.csv"
N_NOISE = 20_000  # noisy copies of x*, one network pass each
N_FM = 5_000  # draws re-generated with TauREx (the 2 sigma band needs a few thousand)
RESOLUTION = 200  # resolving power of the recovered spectrum
QUANTILES = {"median": 0.5, **{f"{side}_{k}sigma": norm.cdf(s * k) for k in (1, 2) for side, s in (("low", -1), ("high", 1))}}
_model = _binners = _wngrid = None


def _init(wave_range):
    global _model, _binners, _wngrid
    _model, binner, _wngrid = forward_model.load_model(config.HATP26B_PAR, config.HATP26B_OBS)
    _binners = [binner, forward_model.uniform_binner(*wave_range, RESOLUTION)[0]]


def _spectra(log_vmr):
    """TauREx (Rp/Rs)^2 for one draw on the observation's bins and at R = 200, concatenated; None if it fails."""
    params = config.FIXED_PARAMS | {m: 10.0 ** v for m, v in zip(config.MOLECULES, log_vmr)}
    try:
        return np.concatenate([forward_model.spectrum(_model, b, _wngrid, params) for b in _binners])
    except Exception:  # e.g. a non-finite spectrum; left out of the percentiles
        return None


def percentiles(samples):
    return {k: np.quantile(samples, q, axis=0) for k, q in QUANTILES.items()}


if __name__ == "__main__":
    obs = pd.read_csv(OBSERVATION)
    wl, x_star, sigma = obs["wl_um"].values, obs["depth"].values, obs["sigma"].values
    black_box = BlackBox(OUT)
    assert np.allclose(np.array(black_box.bins, float), wl)

    # 1. posterior
    noisy = fixed_noisy_copies(x_star[None], sigma, 1.0, N_NOISE, config.SEED)
    log_vmr = black_box.draw(noisy, config.SEED)
    pd.DataFrame(log_vmr, columns=black_box.targets).to_csv(OUT / "posterior_draws.csv", index=False)

    # 2. reconstructed spectrum
    wave_range = (wl[0] - 0.05, wl[-1] + 0.05)  # a little past the data, to span the whole plot
    wl_uniform = forward_model.uniform_binner(*wave_range, RESOLUTION)[1]
    with Pool(config.N_WORKERS, initializer=_init, initargs=(wave_range,)) as pool:
        spectra = pool.map(_spectra, log_vmr[:N_FM], chunksize=4)
    ran = np.array([s is not None and np.all(np.isfinite(s) & (s > 0)) for s in spectra])
    assert ran.mean() > 0.5, f"{(~ran).sum()} of {len(ran)} forward models failed: a setup error, not bad draws"
    spectra = np.array([s for s, ok in zip(spectra, ran) if ok])
    binned, uniform = spectra[:, :len(wl)], spectra[:, len(wl):]
    pd.DataFrame({"wl_um": wl_uniform, **percentiles(uniform)}).to_csv(OUT / "reconstructed_spectrum.csv", index=False)
    pd.DataFrame({"wl_um": wl, **percentiles((x_star - binned) / sigma)}).to_csv(OUT / "reconstructed_residuals.csv",
                                                                                index=False)
    print(f"{ran.sum()} of {len(ran)} forward models ran")

    # Sec. 8.1
    low, mid, high = np.quantile(log_vmr, norm.cdf([-1, 0, 1]), axis=0)
    table = pd.DataFrame({
        "input": [config.HATP26B_LOG_VMR[m] for m in black_box.molecules],
        "noiseless_prediction": black_box(x_star[None])[0],
        "median": mid, "minus_1sigma": mid - low, "plus_1sigma": high - mid,
        "mc_dropout_std_noiseless": black_box.samples(x_star[None])[:, 0].std(axis=0),
    }, index=pd.Index(black_box.molecules, name="molecule"))
    table["offset"] = table["median"] - table["input"]
    table["offset_sigma"] = table["offset"] / np.where(table["offset"] > 0, table["minus_1sigma"], table["plus_1sigma"])
    print(f"CNN retrieval of x* ({N_NOISE} copies with 1 sigma noise, one fresh dropout mask each), log10 VMR:")
    print(table.round(2).to_string())
    print("wrote", OUT / "posterior_draws.csv", OUT / "reconstructed_spectrum.csv", OUT / "reconstructed_residuals.csv")
