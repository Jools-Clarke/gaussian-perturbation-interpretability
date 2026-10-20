"""Fig 3: the one-dimensional reduction of Sec. 6: naive (red) and Gaussian (black) perturbed spectra of the simulated
HAT-P-26 b observation (dashed line), on a scale from random noise to forward models from the prior.

Reads: E2_Hat-P-26b/outputs/dimension_reduction.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import plotting as style

E2 = Path(__file__).resolve().parents[1] / "E2_Hat-P-26b" / "outputs"

style.use(minor_ticks=True)
scores = pd.read_csv(E2 / "dimension_reduction.csv")
p_gauss, p_naive, p_target = (scores.loc[scores["kernel"] == k, "score"].values for k in ("gaussian", "naive", "target"))


combined = np.concatenate([p_gauss, p_naive])
global_min = combined.min()
global_max = combined.max()

# Fig 3's own font sizes, on top of the RASTI style
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "serif",
})

BLACK = "black"
BRICK = "firebrick"

fig, ax = plt.subplots(figsize=(3.35, 2.6), dpi=300)  # ~ single-column width

bins = np.linspace(global_min, global_max, 45)

ax.hist(p_naive, bins=bins, alpha=0.5, density=True,
        label="Naive Perturbation", histtype='stepfilled',
        lw=1.2, edgecolor=BRICK, facecolor=BRICK)

ax.hist(p_gauss, bins=bins, alpha=0.5, density=True,
        label="Gaussian Perturbation", histtype='stepfilled',
        lw=1.2, edgecolor=BLACK, facecolor=BLACK)

ax.axvline(p_target[0], color=BLACK, linestyle="--", linewidth=1.5,
           label="HAT P 26 b Forward Model")

ax.set_xlabel("Reduced Dimension 1", labelpad=18)
ax.set_ylabel("Density")

ax.set_xlim([global_min - 1e-5, global_max + 1e-5])

# the score has no units: label the two ends of the scale instead of ticks
ax.set_xticks([])

ax.text(0.0, -0.02, r"$\leftarrow$random noise",
        transform=ax.transAxes, ha="left", va="top", fontsize=9)
ax.text(1.0, -0.02, r"$\mathcal{D}\rightarrow$",
        transform=ax.transAxes, ha="right", va="top", fontsize=9)
ax.legend(frameon=False, loc="best")


plt.tight_layout(pad=0.5)
print("saved", style.save(fig, Path(__file__).stem))
