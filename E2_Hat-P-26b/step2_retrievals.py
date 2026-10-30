"""Experiment 2, step 2: every HAT-P-26 b TauREx-III retrieval (Sec. 7.3), one after another (~6.5 h each).

Observation: Gressier et al. (2025), NIRSpec G395H, transitspectroscopy reduction at R ~ 100, 60 bins (Sec. 4.2).
Settings: the .par next to it (Tables G1, G2).
Runs (9):
    baseline                           x*, xi = 0 for every centre
    gaussian, 2 centres x 2 xi         the bins nearest the SO2 (4.04 um) and CO2 (4.29 um) bands, xi = +-3 sigma
    naive,    2 centres x 2 xi         the same, one bin at a time
    -> outputs/posteriors/<name>.csv, outputs/spectra/<name>.csv, outputs/runs.csv   as in experiment 1
"""
from pathlib import Path

import numpy as np
import pandas as pd

import _CONFIG as config
from src.perturbation import perturbed_spectra
from src.retrieval import retrieve

OUT = Path(__file__).resolve().parent / "outputs"
XI = [-3.0, 3.0]  # the non-zero magnitudes of Eq. 10, in sigma of the centre bin

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    wl, x, sigma = np.loadtxt(config.HATP26B_OBS).T
    jobs = [("baseline", "baseline", np.nan, 0.0, x)]
    for kernel in ("gaussian", "naive"):
        for band_um in config.TARGET_BANDS.values():
            m = int(np.argmin(np.abs(wl - band_um)))
            for xi, depth in zip(XI, perturbed_spectra(x, sigma, wl, m, XI, fwhm=config.GAUSSIAN_FWHM_UM, kernel=kernel)):
                jobs.append((f"{kernel}_{wl[m]:.3f}um_{xi:+g}", kernel, wl[m], xi, depth))

    runs = []
    for i, (name, kernel, centre, xi, depth) in enumerate(jobs, start=1):
        print(f"[{i}/{len(jobs)}] {name}", flush=True)
        result = retrieve(name, wl, depth, sigma, config.HATP26B_PAR, OUT)
        runs.append({"name": name, "kernel": kernel, "centre_um": centre, "xi": xi, **result})
        pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False)  # after every run: a crash keeps what finished
