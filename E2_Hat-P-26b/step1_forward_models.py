"""Experiment 2, step 1: the HAT-P-26 b forward models that need no retrieval (Sec. 7.3, Fig H1).

1. Prior grid for the global metric (Eq. 14) and Fig 3's classifier: 486 models, nine of the ten gases drawn from
   their LogUniform(-12, -0.3) priors by a seeded Latin hypercube (C2H4 stays at its .par value); T, Rp and the cloud
   top fixed at the .par starting values (701 K, 0.54 RJ, 10^2.55 Pa).
       -> outputs/grid_spectra.csv     one row per model: log_<gas> x9, then one column per bin (um)
2. Contributions (Sec. 7.2.1, Fig H1 lower panels): one native-resolution model per gas.
       -> outputs/contributions.csv    wl_um, then one column per gas
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

import _CONFIG as config
from src.forward_model import contribution, load_model, spectrum

OUT = Path(__file__).resolve().parent / "outputs"
GRID_GASES = ["H2O", "CH4", "C2H2", "CO", "CO2", "NH3", "SO2", "H2S", "HCN"]
BAND_UM = (2.6, 5.4)  # contributions are kept over the G395H band

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    model, binner, wngrid = load_model(config.HATP26B_PAR, config.HATP26B_OBS)
    wl = np.sort(10000 / wngrid)

    d = len(GRID_GASES)
    # the paper's 486 models are the first 486 of 500 draws: the original run stopped at draw 486, whose gases sum
    # past a VMR of 1
    grid = qmc.scale(qmc.LatinHypercube(d=d, seed=1729).random(n=500), [-12] * d, [-0.3] * d)[:486]
    spectra = np.array([spectrum(model, binner, wngrid, {g: 10**v for g, v in zip(GRID_GASES, row)}) for row in grid])
    pd.DataFrame(np.hstack([grid, spectra]), columns=[f"log_{g}" for g in GRID_GASES] + [f"{w:.6f}" for w in wl]) \
        .to_csv(OUT / "grid_spectra.csv", index=False)

    contributions = {}
    for gas in config.MOLECULES:
        wl_native, contributions[gas] = contribution(model, gas, config.MOLECULES, BAND_UM)
    pd.DataFrame({"wl_um": wl_native, **contributions}).to_csv(OUT / "contributions.csv", index=False)
    print(f"wrote {OUT / 'grid_spectra.csv'} ({len(grid)} models), {OUT / 'contributions.csv'}")
