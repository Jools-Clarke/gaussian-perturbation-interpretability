"""Experiment 1, step 2: every HD 209458 b TauREx-III retrieval (Sec. 7), one after another (~36 min each).

Observation: Tsiaras et al. (2018), 18 WFC3 bins (Sec. 4.1). Settings: the .par next to it (Tables B1, B2).
Runs (217):
    baseline                         x*, the unperturbed spectrum; it is xi = 0 for every centre (Sec. 5.2)
    gaussian, 18 centres x 6 xi      Eq. 11 with FWHM 0.1 um and xi = +-0.5, +-1, +-2 sigma (Sec. 7)
    naive,    18 centres x 6 xi      the same grid, one bin at a time (Eqs. 1-3)
    -> outputs/posteriors/<name>.csv   MultiNest samples and their weights
    -> outputs/spectra/<name>.csv      wl_um, observed (perturbed) spectrum, error, best fit
    -> outputs/runs.csv                name, kernel, centre_um, xi, wall_time_s, map_<param>
"""
from pathlib import Path

import numpy as np
import pandas as pd

import _CONFIG as config
from src.perturbation import perturbed_spectra
from src.retrieval import retrieve

OUT = Path(__file__).resolve().parent / "outputs"
XI = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]  # the non-zero magnitudes of Eq. 10, in sigma of the centre bin

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    wl, x, sigma = np.loadtxt(config.HD209_OBS).T
    jobs = [("baseline", "baseline", np.nan, 0.0, x)]
    for kernel in ("gaussian", "naive"):
        for m, centre in enumerate(wl):
            for xi, depth in zip(XI, perturbed_spectra(x, sigma, wl, m, XI, fwhm=config.GAUSSIAN_FWHM_UM, kernel=kernel)):
                jobs.append((f"{kernel}_{centre:.3f}um_{xi:+g}", kernel, centre, xi, depth))

    runs = []
    for i, (name, kernel, centre, xi, depth) in enumerate(jobs, start=1):
        print(f"[{i}/{len(jobs)}] {name}", flush=True)
        result = retrieve(name, wl, depth, sigma, config.HD209_PAR, OUT)
        runs.append({"name": name, "kernel": kernel, "centre_um": centre, "xi": xi, **result})
        pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False)  # after every run: a crash keeps what finished
