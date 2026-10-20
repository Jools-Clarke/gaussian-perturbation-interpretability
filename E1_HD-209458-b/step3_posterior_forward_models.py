"""Experiment 1, step 3: forward models at the baseline posterior samples, for the local metric (Eq. 15, Sec. 7.2.3).

Keeps the samples that hold 99.999 per cent of the posterior weight (8 862 in the paper) and computes the forward
model at each, with all eight fitted parameters set; their weights go with them, as the metric is weighted.
    -> outputs/posterior_spectra.csv    one row per sample: the 8 parameters, weight, then one column per bin
"""
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import _CONFIG as config
from src import forward_model

OUT = Path(__file__).resolve().parent / "outputs"
WEIGHT_KEPT = 0.99999

_model = _binner = _names = _wngrid = None


def _init_pool(names):
    global _model, _binner, _wngrid, _names
    _model, _binner, _wngrid = forward_model.load_model(config.HD209_PAR, config.HD209_OBS)
    _names = names


def spectrum(theta):
    """theta: {retrieval parameter: value} -> binned spectrum, ascending wavelength."""
    return forward_model.spectrum(_model, _binner, _wngrid, forward_model.model_params(theta))


def _work(rows):
    return np.array([spectrum(dict(zip(_names, r))) for r in rows])


if __name__ == "__main__":
    posterior = pd.read_csv(OUT / "posteriors" / "baseline.csv")
    best = pd.read_csv(OUT / "spectra" / "baseline.csv")
    run = pd.read_csv(OUT / "runs.csv").set_index("name").loc["baseline"]
    names = [p for p in posterior.columns if np.isfinite(run.get(f"map_{p}", np.nan))]  # fitted parameters have a MAP
    trace, w = posterior[names].values, posterior["weight"].values
    x, sig = best["observed"].values, best["error"].values

    # check: the forward model at the MAP sample reproduces TauREx's own best fit, so the models are the retrieval's
    _init_pool(names)
    t0 = time.time(); fit = spectrum({p: run[f"map_{p}"] for p in names}); dt = time.time() - t0
    diff = np.max(np.abs(fit - best["best_fit"].values) / sig)
    print(f"forward model at MAP vs TauREx stored best fit: max |diff| = {diff:.2e} sigma  ({dt*1e3:.0f} ms per model)")
    assert diff < 1e-2, "forward model does not reproduce the stored best fit"

    order = np.argsort(w)[::-1]
    keep = order[: np.searchsorted(np.cumsum(w[order]), WEIGHT_KEPT) + 1]
    print(f"forward-modelling {len(keep)} of {len(w)} samples ({WEIGHT_KEPT:.5f} of the weight)...", flush=True)
    with Pool(config.N_WORKERS, initializer=_init_pool, initargs=(names,)) as pool:
        m = np.concatenate(pool.map(_work, np.array_split(trace[keep], min(64, len(keep)))))
    pd.DataFrame(np.column_stack([trace[keep], w[keep] / w[keep].sum(), m]),
                 columns=names + ["weight"] + [f"{v:.6f}" for v in best["wl_um"]]) \
        .to_csv(OUT / "posterior_spectra.csv", index=False)
    print("wrote", OUT / "posterior_spectra.csv")
