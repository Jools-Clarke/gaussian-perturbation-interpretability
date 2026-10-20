"""Experiment 3, step 1: the CNN's training set, 100 000 noiseless HAT-P-26 b forward models (Appendix I2).

The ten log VMRs are drawn from the retrieval prior U(-12, -0.3) (Table F2), redrawing any draw whose gases sum past
a VMR of 0.85 so that the H2/He fill stays positive. T, Rp and the cloud top are fixed at the Table F3 posterior
means, so the network needs no auxiliary inputs. Split 8:1:1 into train / validation / test.
    -> outputs/training_set.csv    one row per model: sample_id, split, log_<gas> x10, then one column per bin (um)

~0.18 s per model on _CONFIG.N_WORKERS processes (about 30 min on 10).
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import _CONFIG as config
from src import forward_model

OUT = Path(__file__).resolve().parent / "outputs"
N_DRAWS = 100_000
LOG_VMR_BOUNDS = (-12.0, -0.3)  # log VMR prior of every gas, as in the retrieval
MAX_TOTAL_VMR = 0.85
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
_model = _binner = _wngrid = None


def sample_prior(n, rng):
    """n draws of log10 VMR ~ U(bounds), rejecting draws whose total VMR exceeds MAX_TOTAL_VMR."""
    lo, hi = LOG_VMR_BOUNDS
    accepted = []
    n_accepted = 0
    while n_accepted < n:
        batch = rng.uniform(lo, hi, size=(2 * (n - n_accepted), len(config.MOLECULES)))
        batch = batch[(10.0**batch).sum(axis=1) <= MAX_TOTAL_VMR]
        accepted.append(batch)
        n_accepted += len(batch)
    return np.concatenate(accepted)[:n]


def assign_split(sample_id):
    """Each model's split depends only on its own id, so it does not change with the size of the set."""
    u = np.random.default_rng([config.SEED, 2, int(sample_id)]).random()
    edges = np.cumsum(list(SPLIT_FRACTIONS.values()))
    return list(SPLIT_FRACTIONS)[int(np.searchsorted(edges, u, side="right"))]


def _init():
    global _model, _binner, _wngrid
    _model, _binner, _wngrid = forward_model.load_model(config.HATP26B_PAR, config.HATP26B_OBS)


def _spectrum(log_vmr):
    """Binned (Rp/Rs)^2 for one draw; NaN if TauREx fails, and the draw is left out."""
    try:
        return forward_model.spectrum(_model, _binner, _wngrid,
                                      config.FIXED_PARAMS | {m: 10.0 ** v for m, v in zip(config.MOLECULES, log_vmr)})
    except Exception:
        return np.full(len(_wngrid), np.nan)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    draws = sample_prior(N_DRAWS, np.random.default_rng([config.SEED, 0]))
    wl_obs = np.loadtxt(config.HATP26B_OBS)[:, 0]

    spectra = []
    with Pool(config.N_WORKERS, initializer=_init) as pool:
        for i, s in enumerate(pool.imap(_spectrum, draws, chunksize=50), start=1):
            spectra.append(s)
            if i % 10_000 == 0:
                print(f"  {i}/{len(draws)} forward models", flush=True)
    spectra = np.array(spectra)

    ok = np.all(np.isfinite(spectra) & (spectra > 0), axis=1)
    assert ok.mean() > 0.5, f"{(~ok).sum()} of {len(ok)} forward models failed: a setup error, not bad draws"
    ids = np.arange(len(draws))
    table = pd.DataFrame(np.hstack([draws, spectra]), columns=[f"log_{m}" for m in config.MOLECULES]
                         + [f"{w:.6f}" for w in wl_obs])
    table.insert(0, "split", [assign_split(i) for i in ids])
    table.insert(0, "sample_id", ids)
    table[ok].to_csv(OUT / "training_set.csv", index=False)
    print(f"wrote {OUT / 'training_set.csv'}: {ok.sum()} models ({(~ok).sum()} failed and left out), "
          + ", ".join(f"{k} {v}" for k, v in table[ok]["split"].value_counts().items()))
