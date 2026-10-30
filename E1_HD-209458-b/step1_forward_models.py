"""Experiment 1, step 1: the HD 209458 b forward models that need no retrieval (Sec. 7.2).

1. Prior grid for the global metric (Eq. 14) and Figs 2, C1: the five gases on a regular grid, log VMR -10 to -2 in
   steps of 2 dex (3 125 models); T, Rp and the cloud top fixed at the .par starting values (902.4 K, 1.39 RJ, 10^3 Pa).
       -> outputs/grid_spectra.csv     one row per model: log_<gas> x5, then one column per bin (um)
2. Contributions (Sec. 7.2.1, Figs 7, D1 lower panels): one native-resolution model per gas.
       -> outputs/contributions.csv    wl_um, then one column per gas
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import _CONFIG as config
from src.forward_model import contribution, load_model, spectrum

OUT = Path(__file__).resolve().parent / "outputs"
GASES = ["H2O", "CH4", "CO", "CO2", "NH3"]  # the retrieved gases (Table B2)
BAND_UM = (1.0, 1.8)  # contributions are kept over the WFC3 band

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    model, binner, wngrid = load_model(config.HD209_PAR, config.HD209_OBS)
    wl = np.sort(10000 / wngrid)

    grid = np.array(list(itertools.product(np.linspace(-10, -2, 5), repeat=len(GASES))))
    spectra = np.array([spectrum(model, binner, wngrid, {g: 10**v for g, v in zip(GASES, row)}) for row in grid])
    pd.DataFrame(np.hstack([grid, spectra]), columns=[f"log_{g}" for g in GASES] + [f"{w:.6f}" for w in wl]) \
        .to_csv(OUT / "grid_spectra.csv", index=False)

    contributions = {}
    for gas in GASES:
        wl_native, contributions[gas] = contribution(model, gas, GASES, BAND_UM)
    pd.DataFrame({"wl_um": wl_native, **contributions}).to_csv(OUT / "contributions.csv", index=False)
    print(f"wrote {OUT / 'grid_spectra.csv'} ({len(grid)} models), {OUT / 'contributions.csv'}")
