"""Experiment 3, step 5: the numbers the CNN figures plot (Secs 7.3.3 and 8.2).

    global_metric.csv               Fig 13 middle: global metric (Eq. 14) over the training split, which the
                                    network learned from (almost the same as Fig G1's for SO2 and CO2)
    response_gradient_<mode>.csv    Figs 13, K1: slope of the response between neighbouring magnitudes
    importance.csv                  Fig K1 (d, h): the mean of those slopes (Eq. 13), for each mode of step 4
    co2_core_correlation.csv        Fig 10: correlation between the raw depth in the 4.293 um bin and log CO2 within
                                    sliding 1-dex windows of log CO2, across every training-set model
"""
from pathlib import Path

import numpy as np
import pandas as pd

import _CONFIG as config
from src.forward_model import wl_columns
from src.metrics import metric
from src.perturbation import importance

OUT = Path(__file__).resolve().parent / "outputs"
MODES = ["fixed1000", "fresh1000", "fresh1"]
WINDOW_DEX = 1.0  # width of the log CO2 windows of Fig 10
WINDOW_CENTRES = np.round(np.arange(-11.5, -0.75, 0.25), 2)  # windows stay inside the U(-12, -0.3) prior

if __name__ == "__main__":
    out = {}
    data = pd.read_csv(OUT / "training_set.csv")
    bins = wl_columns(data)
    wl = np.array(bins, float)

    # Fig 13 middle
    train = data[data["split"] == "train"]
    out["global_metric"] = pd.DataFrame({"wl_um": wl, **{f"global_{m}": metric(train[bins].values, train[f"log_{m}"].values)
                                                         for m in config.MOLECULES}})

    # Figs 13, K1
    out["importance"] = []
    for mode in MODES:
        response = pd.read_csv(OUT / f"response_{mode}.csv").pivot(index=["gas", "centre_um"], columns="xi",
                                                                   values="response")
        xi = response.columns.values
        gradient = np.diff(response.values, axis=1) / np.diff(xi)
        out[f"response_gradient_{mode}"] = [dict(centre_um=c, xi_low=a, xi_high=b, gas=g, gradient=v)
                                            for (g, c), row in zip(response.index, gradient)
                                            for a, b, v in zip(xi[:-1], xi[1:], row)]
        out["importance"] += [dict(mode=mode, centre_um=c, gas=g, importance=v)
                              for (g, c), v in zip(response.index, importance(response.values, xi))]

    # Fig 10: on the raw spectra, within windows of log CO2
    core = int(np.argmin(np.abs(wl - config.TARGET_BANDS["CO2"])))
    depth, co2 = data[bins[core]].values, data["log_CO2"].values
    rho = [np.corrcoef(depth[np.abs(co2 - c) < WINDOW_DEX / 2], co2[np.abs(co2 - c) < WINDOW_DEX / 2])[0, 1]
           for c in WINDOW_CENTRES]
    out["co2_core_correlation"] = pd.DataFrame({"log_co2": WINDOW_CENTRES, "core_um": wl[core], "rho": rho})
    print(f"Fig 10: rho(depth at {wl[core]:.3f} um, log CO2) over all {len(co2)} models: "
          f"{np.corrcoef(depth, co2)[0, 1]:+.3f}")

    for name, table in out.items():
        pd.DataFrame(table).to_csv(OUT / f"{name}.csv", index=False)
        print("wrote", OUT / f"{name}.csv")
