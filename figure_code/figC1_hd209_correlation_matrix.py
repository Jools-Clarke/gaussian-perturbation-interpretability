"""Fig C1: Pearson correlation of every pair of bins across the min-max normalised HD 209458 b grid: neighbouring
bins correlate most (Sec. 5).

Reads: E1_HD-209458-b/outputs/correlation_matrix.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from src import plotting as style

E1 = Path(__file__).resolve().parents[1] / "E1_HD-209458-b" / "outputs"

style.use(minor_ticks=True)
m = pd.read_csv(E1 / "correlation_matrix.csv", index_col="wl_um")
corr_da = xr.DataArray(m.values, dims=("wl", "wl2"), coords={"wl": m.index.values, "wl2": m.columns.astype(float)})


# RASTI single-column width is ~3.32 in (244 pt); use ~2/3 that height for a square-ish corr plot
fig, ax = plt.subplots(figsize=(3.32, 3.0), constrained_layout=True)

im = corr_da.plot(
    x='wl', y='wl2',
    cmap='viridis',
    ax=ax,
    add_colorbar=False,   # add manually for full control over size/label
)

ax.set_xlabel(r'Wavelength $\lambda$ $(\mu m)$')
ax.set_ylabel(r'Wavelength $\lambda^{\dagger}$ $(\mu m)$')
ax.set_aspect('equal')   # correlation matrices should be square

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label(r'Pearson $\rho(\lambda,$ $\lambda^{\dagger})$', fontsize=9)
cbar.ax.tick_params(labelsize=8)

print("saved", style.save(fig, Path(__file__).stem))
