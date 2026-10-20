"""Experiment 2, step 4: Fig 3, naive and Gaussian perturbed spectra reduced to one dimension (Sec. 6).

A small 1D CNN classifier (footnote 8 of the paper) learns to tell forward models drawn from the prior (step 1's grid,
label 1, "in distribution") from uniform random noise (label 0). Its score of a spectrum is then a one-dimensional
scale from noise (0) to in-distribution (1). We score the simulated observation (step 3) and its naive and Gaussian
perturbations by xi = +-1 sigma centred on each of the 60 bins (120 spectra each).
    -> outputs/dimension_reduction.csv   kernel ("gaussian" | "naive" | "target"), centre_um, xi, score

The scores depend on the seed (an unseeded run moves the histograms visibly); the separation of the two kernels does
not.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import _CONFIG as config
from src.forward_model import wl_columns
from src.perturbation import perturbed_spectra

OUT = Path(__file__).resolve().parent / "outputs"
MAGNITUDES = [-1, 1]
SEED = 0
EPOCHS, BATCH_SIZE, LR = 8, 64, 1e-3


def to_tensor(X, y=None):
    """(N, n_wl) -> (N, 1, n_wl): Conv1d wants a channel axis."""
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    if y is None:
        return X_t
    return X_t, torch.tensor(y, dtype=torch.float32).unsqueeze(1)


class CNNClassifier(nn.Module):
    """Three Conv1d layers (32, 32, 64 filters, kernel 3, max pooling after the second), global average pooling, and
    dense layers of 32 (dropout 0.2) and 1 unit: raw logits."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        return self.fc(self.pool(self.conv(x)).squeeze(-1))


if __name__ == "__main__":
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    grid = pd.read_csv(OUT / "grid_spectra.csv")
    prior_spectra = grid[wl_columns(grid)].values
    target = pd.read_csv(OUT / "simulated_observation.csv")
    wl, x_star, sigma = target["wl_um"].values, target["depth"].values, target["sigma"].values

    # the simulated observation perturbed at every centre, Gaussian and naive (Eq. 11)
    rows, perturbed = [], []
    for kernel in ("gaussian", "naive"):
        for m in range(len(wl)):
            for xi, spectrum in zip(MAGNITUDES, perturbed_spectra(x_star, sigma, wl, m, MAGNITUDES,
                                                                  fwhm=config.GAUSSIAN_FWHM_UM, kernel=kernel)):
                rows.append({"kernel": kernel, "centre_um": wl[m], "xi": xi})
                perturbed.append(spectrum)

    # each bin scaled to [0, 1] by its minimum and maximum across the prior grid
    lo, hi = prior_spectra.min(axis=0), prior_spectra.max(axis=0)
    prior_norm = (prior_spectra - lo) / (hi - lo)
    perturbed_norm = (np.array(perturbed) - lo) / (hi - lo)
    target_norm = np.array([(x_star - lo) / (hi - lo)])

    # training set: the prior grid (1) and as many spectra of uniform noise (0); 80 / 20 train / held-out split
    noise = rng.random(prior_norm.shape)
    rng.shuffle(prior_norm)
    X = np.vstack([prior_norm, noise]).astype("float32")
    y = np.concatenate([np.ones(len(prior_norm)), np.zeros(len(noise))]).astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    X_tr_t, y_tr_t = to_tensor(X_tr, y_tr)
    X_te_t, y_te_t = to_tensor(X_te, y_te)
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clf = CNNClassifier().to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(EPOCHS):
        clf.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(clf(xb), yb)
            loss.backward()
            optimizer.step()
    clf.eval()
    with torch.no_grad():
        held_out = ((torch.sigmoid(clf(X_te_t.to(device))) > 0.5).float().cpu() == y_te_t).float().mean().item()
    print("held-out accuracy:", held_out)

    def score(X):
        """P(in distribution) for each normalised spectrum: the reduced dimension of Fig 3."""
        with torch.no_grad():
            return torch.sigmoid(clf(to_tensor(np.asarray(X, dtype="float32")).to(device))).cpu().numpy().ravel()

    p_target = score(target_norm)
    print("target score:", p_target)
    pd.concat([pd.DataFrame(rows).assign(score=score(perturbed_norm)),
               pd.DataFrame([{"kernel": "target", "centre_um": np.nan, "xi": 0, "score": p_target[0]}])]) \
        .to_csv(OUT / "dimension_reduction.csv", index=False)
    print("wrote", OUT / "dimension_reduction.csv")
