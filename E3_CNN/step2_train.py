"""Experiment 3, step 2: train the CNN (src.cnn, Table I1) on outputs/training_set.csv.

Adam (lr 1e-3), batch 128, MSE on the normalised targets. Noise of sigma_m is added to every training spectrum and
redrawn every epoch: perturbations of order sigma_m only stay in distribution for a network trained on noisy spectra
(Appendix I2). Validation on 5 fixed noisy copies; the learning rate is halved after 10 epochs without improvement,
training stops after 30, and the best validation weights are kept (epoch 246 of 276 in the paper).
    -> outputs/normalisation.csv      0.5th and 99.5th training percentiles, one column per bin and per gas
    -> outputs/cnn_weights.pt         the trained weights
    -> outputs/training_history.csv   epoch, train loss, val loss, learning rate

About 15 min on an M4 Pro GPU (mps). The device is picked automatically, so another machine trains a slightly
different network.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import _CONFIG as config
from src.cnn import NoisyDataset, RetrievalCNN, fixed_noisy_copies, normalise, predict, save_checkpoint
from src.forward_model import wl_columns

OUT = Path(__file__).resolve().parent / "outputs"
NORM_PERCENTILES = (0.5, 99.5)
EPOCHS = 300
BATCH_SIZE = 128
LR = 1e-3
PATIENCE = 30  # early stopping: epochs without a better validation loss
LR_PATIENCE = 10  # LR halved after this many
VAL_COPIES = 5  # fixed noisy copies of each validation spectrum, 1 sigma
SEED = 0  # torch: weights, shuffle, training noise
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def run_epoch(model, loader, optimiser):
    """One training pass over loader. Returns the sample-weighted mean loss."""
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.mse_loss(model(x), y)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        total += loss.item() * len(x)
        n += len(x)
    return total / n


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    data = pd.read_csv(OUT / "training_set.csv")
    bins = wl_columns(data)
    targets = [f"log_{m}" for m in config.MOLECULES]
    train, val = data[data["split"] == "train"], data[data["split"] == "val"]
    wl_obs, _, sigma = np.loadtxt(config.HATP26B_OBS).T
    assert np.allclose(np.array(bins, float), wl_obs)

    # percentiles of the training split only, per bin and per gas
    norm = pd.DataFrame(np.percentile(train[bins + targets].values, NORM_PERCENTILES, axis=0),
                        index=[f"p{p:g}" for p in NORM_PERCENTILES], columns=bins + targets)
    norm.to_csv(OUT / "normalisation.csv")
    sigma_norm = sigma / np.diff(norm[bins].values, axis=0)[0]  # observational error in normalised units

    x_train, y_train = normalise(train[bins].values, norm, bins), normalise(train[targets].values, norm, targets)
    train_loader = torch.utils.data.DataLoader(NoisyDataset(x_train, y_train, sigma_norm, fresh=True, seed=SEED),
                                               batch_size=BATCH_SIZE, shuffle=True)
    # validation on noisy copies too: the perturbation feeds the network noisy-looking inputs
    val_x = fixed_noisy_copies(normalise(val[bins].values, norm, bins), sigma_norm, 1.0, VAL_COPIES, config.SEED)
    val_y = torch.tensor(np.tile(normalise(val[targets].values, norm, targets), (VAL_COPIES, 1)), dtype=torch.float32)

    model = RetrievalCNN(n_wavelengths=len(bins), n_targets=len(targets)).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.5, patience=LR_PATIENCE)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters, {len(train)} train / {len(val)} val spectra, "
          f"on {DEVICE}")

    history, best_val, best_epoch = [], np.inf, 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimiser)
        val_loss = F.mse_loss(torch.as_tensor(predict(model, val_x, device=DEVICE)), val_y).item()
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            save_checkpoint(model, OUT / "cnn_weights.pt", epoch=epoch, val_loss=val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "lr": optimiser.param_groups[0]["lr"]})
        print(f"epoch {epoch:4d}  train {train_loss:.5f}  val {val_loss:.5f}  {'*' if epoch == best_epoch else ''}",
              flush=True)
        scheduler.step(val_loss)
        if epoch - best_epoch >= PATIENCE:
            break

    assert best_epoch > 0, "validation loss never improved (NaN?): cnn_weights.pt is not from this run"
    pd.DataFrame(history).to_csv(OUT / "training_history.csv", index=False)
    print(f"best epoch {best_epoch} of {len(history)} (val loss {best_val:.5f}); wrote", OUT / "normalisation.csv",
          OUT / "cnn_weights.pt", OUT / "training_history.csv")
