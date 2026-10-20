"""Fig G1: Fig 5 for HAT-P-26 b, the nine gases of the prior grid (C2H4 is not drawn in it).

Reads: E2_Hat-P-26b/outputs/metrics.csv, contributions.csv
"""
from pathlib import Path

import pandas as pd

from fig05_hd209_metrics import plot

E2 = Path(__file__).resolve().parents[1] / "E2_Hat-P-26b" / "outputs"

if __name__ == "__main__":
    mols = [c.removeprefix("global_") for c in pd.read_csv(E2 / "metrics.csv").columns if c.startswith("global_")]
    plot(E2, mols, ncols=2, width=7.0, per_gas=2.0, name=Path(__file__).stem)
