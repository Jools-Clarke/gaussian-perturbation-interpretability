"""Experiment 3, step 4: Gaussian perturbation of the CNN (Sec. 8.2; Figs 13, K1).

Every one of the 60 bins as centre, 101 magnitudes from -3 to +3 sigma (steps of 0.06 sigma), FWHM 0.1 um (Eq. 11).
The response (Eq. 12) of f, evaluated three ways:
    fixed1000   the mean over the same 1 000 dropout masks for every input: f of the paper (Fig 13; K1 a, e)
    fresh1000   the mean over 1 000 fresh masks per evaluation (K1 b, f)
    fresh1      one fresh mask per evaluation (K1 c, g)
    -> outputs/response_<mode>.csv    centre_um, xi, gas, response (change in the predicted log VMR)

About 15 min on CPU, almost all of it fresh1000; fixed1000 alone takes seconds (Sec. 8.3).
"""
from pathlib import Path

import numpy as np
import pandas as pd

import _CONFIG as config
from src.cnn import BlackBox, FreshMaskBlackBox
from src.perturbation import perturbed_spectra

OUT = Path(__file__).resolve().parent / "outputs"
OBSERVATION = Path(__file__).resolve().parents[1] / "E2_Hat-P-26b" / "outputs" / "simulated_observation.csv"
# xi in sigma of the centre bin (Eq. 10): an odd number, so xi = 0 is on the grid
XI = np.round(np.linspace(-3.0, 3.0, 101), 6)
MODES = {"fixed1000": lambda: BlackBox(OUT),
         "fresh1000": lambda: FreshMaskBlackBox(OUT, 1000, seed=1000 + 1000),
         "fresh1": lambda: FreshMaskBlackBox(OUT, 1, seed=1000 + 1)}

if __name__ == "__main__":
    obs = pd.read_csv(OBSERVATION)
    wl, x_star, sigma = obs["wl_um"].values, obs["depth"].values, obs["sigma"].values
    spectra = np.stack([perturbed_spectra(x_star, sigma, wl, m, XI, fwhm=config.GAUSSIAN_FWHM_UM)
                        for m in range(len(wl))])  # (centre, xi, wl), Eq. 11

    for mode, make in MODES.items():
        f = make()
        baseline = f(x_star[None])[0]  # first: the fresh masks are seeded by call order
        response = f(spectra) - baseline  # (centre, xi, gas), Eq. 12
        centre, xi, gas = np.meshgrid(wl, XI, f.molecules, indexing="ij")
        pd.DataFrame({"centre_um": centre.ravel(), "xi": xi.ravel(), "gas": gas.ravel(), "response": response.ravel()}) \
            .to_csv(OUT / f"response_{mode}.csv", index=False)
        print("wrote", OUT / f"response_{mode}.csv", flush=True)
