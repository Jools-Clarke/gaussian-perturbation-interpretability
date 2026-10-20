"""The CNN retrieval of Sec. 8 (Appendix I, Table I1) and the black box f it gives (Secs 2.5 and 8.2). PyTorch.

f(x) is the mean prediction over a FIXED set of 1 000 Monte Carlo dropout masks, the same for every input, so the
dropout noise cancels when two predictions are compared and a perturbation changes f only through the spectrum.
FreshMaskBlackBox redraws the masks at every evaluation instead (Fig K1).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import _CONFIG as config


# ---------------------------------------------------------------- the network (Appendix I1)
class RetrievalCNN(nn.Module):
    """60 normalised bins -> 10 normalised log10 VMRs: the ADC 2022 Monte Carlo dropout baseline (Yip et al. 2022)
    without its stellar and planetary radius inputs, which are fixed in this training set. Three blocks of two Conv1d
    (kernel 3, no padding, ReLU) + MaxPool(2) with 32 / 64 / 128 filters, dense 500 and 100 with dropout 0.1.

    forward() takes explicit dropout masks (dropout_masks()), so one fixed set can be reused for every input;
    without masks it is the usual nn.Dropout, off in eval mode.
    """

    def __init__(
        self,
        n_wavelengths=60,
        n_targets=10,
        filters=(32, 64, 128),
        kernel_size=3,
        dense=(500, 100),
        dropout=0.1,
    ):
        super().__init__()
        self.kwargs = dict(
            n_wavelengths=n_wavelengths, n_targets=n_targets, filters=list(filters),
            kernel_size=kernel_size, dense=list(dense), dropout=dropout,
        )

        layers, in_channels = [], 1
        for n_filters in filters:
            layers += [
                nn.Conv1d(in_channels, n_filters, kernel_size), nn.ReLU(),
                nn.Conv1d(n_filters, n_filters, kernel_size), nn.ReLU(),
                nn.MaxPool1d(2),
            ]
            in_channels = n_filters
        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            n_flat = self.features(torch.zeros(1, 1, n_wavelengths)).numel()

        head, in_features = [nn.Flatten()], n_flat
        for n_units in dense:
            head += [nn.Linear(in_features, n_units), nn.ReLU(), nn.Dropout(dropout)]
            in_features = n_units
        head.append(nn.Linear(in_features, n_targets))
        self.head = nn.Sequential(*head)

    def forward(self, x, dropout_masks=None):
        """x: (batch, n_wavelengths) normalised spectra -> (batch, n_targets).

        dropout_masks: Monte Carlo dropout, one 0/1 mask per Dropout layer from dropout_masks(); kept units are
        scaled by 1 / (1 - p) as in training. Masks of shape (n_samples, 1, n_units) give n_samples passes of
        every spectrum, (n_samples, batch, n_targets); masks of shape (batch, n_units) give one pass per spectrum.
        """
        h = self.features(x.unsqueeze(1))
        if dropout_masks is None:
            return self.head(h)
        masks = iter(dropout_masks)
        for layer in self.head:
            h = next(masks) * h / (1 - layer.p) if isinstance(layer, nn.Dropout) else layer(h)
        return h

    def dropout_masks(self, shape, seed):
        """Random masks for forward(), one per Dropout layer, each (*shape, n_units): 1 keeps a unit, 0 drops it."""
        gen = torch.Generator().manual_seed(seed)
        p = self.kwargs["dropout"]
        return [(torch.rand(*shape, n_units, generator=gen) >= p).float() for n_units in self.kwargs["dense"]]


def save_checkpoint(model, path, **extra):
    torch.save({"model_kwargs": model.kwargs, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path, device="cpu"):
    """Rebuild a model from a checkpoint; returns (model in eval mode, checkpoint dict)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)  # no pickle code runs from a downloaded file
    model = RetrievalCNN(**checkpoint["model_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


# ---------------------------------------------------------------- normalisation (Table I1)
def normalise(values, norm, columns):
    """values (..., len(columns)) -> the linear map sending norm's two rows (the training set's 0.5th and 99.5th
    percentiles) to 0 and 1. Never clipped: perturbed spectra must pass through the same map."""
    lo, hi = norm[columns].to_numpy()
    return (np.asarray(values) - lo) / (hi - lo)


def denormalise(values, norm, columns):
    lo, hi = norm[columns].to_numpy()
    return np.asarray(values) * (hi - lo) + lo


# ---------------------------------------------------------------- noise (Appendix I2: sigma_m per bin)
def fixed_noisy_copies(x, sigma, level, n_copies, seed):
    """n_copies noisy versions x + level sigma N(0, 1) of the spectra x (n, n_wl), stacked copy by copy:
    (n_copies * n, n_wl). sigma is in the units of x; the same seed always gives the same noise."""
    x = np.asarray(x)
    rng = np.random.default_rng([seed, 4])
    return np.concatenate([x + level * np.asarray(sigma) * rng.standard_normal(x.shape) for _ in range(n_copies)])


class NoisyDataset(torch.utils.data.Dataset):
    """Noisy (x, y) pairs for training: x (n, n_wl) noiseless spectra plus level * sigma (n_wl,) Gaussian noise.
    fresh=True draws new noise every time a sample is read, i.e. every epoch (Appendix I2); otherwise n_copies
    fixed copies."""

    def __init__(self, x, y, sigma, n_copies=1, level=1.0, fresh=False, seed=0):
        self.x, self.y = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
        self.std = level * torch.as_tensor(sigma, dtype=torch.float32)
        self.n_copies, self.fresh = n_copies, fresh
        if not fresh:
            gen = torch.Generator().manual_seed(seed)
            self.noise = self.std * torch.randn((n_copies, *self.x.shape), generator=gen)

    def __len__(self):
        return self.n_copies * len(self.x)

    def __getitem__(self, i):
        copy, j = divmod(i, len(self.x))
        noise = self.std * torch.randn(self.x.shape[1]) if self.fresh else self.noise[copy, j]
        return self.x[j] + noise, self.y[j]


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict(model, x, batch_size=4096, device="cpu"):
    """Deterministic predictions (eval mode) for an array of normalised spectra."""
    model.eval()
    x = torch.as_tensor(x, dtype=torch.float32)
    out = [model(x[i:i + batch_size].to(device)).cpu() for i in range(0, len(x), batch_size)]
    return torch.cat(out).numpy()


@torch.no_grad()
def predict_mc(model, x, masks, device="cpu", max_rows=2**18):
    """Monte Carlo dropout predictions for normalised spectra x (n, n_wl), with masks from model.dropout_masks():
    shape (n_samples, 1, n_units) applies every mask to every spectrum -> (n_samples, n, n_targets);
    shape (n, n_units) applies mask i to spectrum i -> (n, n_targets). Batched to bound the dense layers' memory."""
    model.eval()
    x = torch.as_tensor(x, dtype=torch.float32)
    shared = masks[0].dim() == 3
    batch_size = max(1, max_rows // len(masks[0])) if shared else max_rows
    out = []
    for i in range(0, len(x), batch_size):
        batch_masks = [m.to(device) if shared else m[i:i + batch_size].to(device) for m in masks]
        out.append(model(x[i:i + batch_size].to(device), dropout_masks=batch_masks).cpu())
    return torch.cat(out, dim=-2).numpy()


# ---------------------------------------------------------------- the black box f
class BlackBox:
    """The trained CNN in out_dir (cnn_weights.pt, normalisation.csv) as f: raw (Rp/Rs)^2 -> log10 VMRs (dex).

    f is the mean over n_masks dropout masks drawn once with config.SEED and shared by every spectrum (Sec. 8.2).
    samples() returns the individual passes; draw() makes one pass per spectrum with its own fresh mask, the
    posterior draws of Sec. 8.1.
    """

    def __init__(self, out_dir, n_masks=1000, device="cpu"):
        self.model, _ = load_checkpoint(Path(out_dir) / "cnn_weights.pt", device)
        self.norm = pd.read_csv(Path(out_dir) / "normalisation.csv", index_col=0)
        self.targets = [c for c in self.norm.columns if c.startswith("log_")]
        self.bins = [c for c in self.norm.columns if c not in self.targets]
        self.molecules = [t.removeprefix("log_") for t in self.targets]
        self.device = device
        self.masks = self.model.dropout_masks((n_masks, 1), config.SEED)

    def __call__(self, rprs, chunk=4096):
        """rprs: (..., n_wl) raw spectra -> (..., n_mol) predicted log10 VMR, the mean over the fixed masks."""
        flat = np.asarray(rprs, dtype=float).reshape(-1, np.shape(rprs)[-1])
        mean = np.concatenate([self.samples(flat[i:i + chunk]).mean(axis=0) for i in range(0, len(flat), chunk)])
        return mean.reshape(*np.shape(rprs)[:-1], len(self.molecules))

    def samples(self, rprs):
        """rprs: (..., n_wl) -> (n_masks, ..., n_mol): one prediction per fixed dropout mask."""
        return self._predict(rprs, lambda x: predict_mc(self.model, x, self.masks, device=self.device),
                             len(self.masks[0]))

    def draw(self, rprs, seed):
        """rprs: (..., n_wl) -> (..., n_mol): one pass per spectrum, each with a fresh random dropout mask."""
        n = int(np.prod(np.shape(rprs)[:-1]))
        masks = self.model.dropout_masks((n,), seed)
        return self._predict(rprs, lambda x: predict_mc(self.model, x, masks, device=self.device))

    def _predict(self, rprs, predict_normalised, n_samples=None):
        rprs = np.asarray(rprs, dtype=float)
        flat = rprs.reshape(-1, rprs.shape[-1])
        pred = denormalise(predict_normalised(normalise(flat, self.norm, self.bins)), self.norm, self.targets)
        leading = () if n_samples is None else (n_samples,)
        return pred.reshape(*leading, *rprs.shape[:-1], len(self.molecules))


class FreshMaskBlackBox:
    """f as the mean of n passes, each with a FRESH dropout mask, so f is no longer deterministic (Fig K1):
    n = 1000 recovers the importance but not the response-gradient maps, n = 1 is a single noisy pass."""

    def __init__(self, out_dir, n, seed, chunk=2000):
        self.black_box, self.n, self.seed, self.chunk = BlackBox(out_dir), n, seed, chunk
        self.molecules = self.black_box.molecules
        self._calls = 0

    def __call__(self, rprs):
        rprs = np.asarray(rprs, dtype=float)
        flat = rprs.reshape(-1, rprs.shape[-1])
        rows = max(1, self.chunk // self.n) if self.n > 1 else 50000
        out = []
        for i in range(0, len(flat), rows):
            x = flat[i:i + rows]
            self._calls += 1
            out.append(self.black_box.draw(np.broadcast_to(x, (self.n, *x.shape)), self.seed + self._calls).mean(axis=0))
        return np.concatenate(out).reshape(*rprs.shape[:-1], len(self.molecules))
