"""Posterior statistics, and the two reference metrics the importance is compared against (Sec. 7.2). Pure numpy.

TauREx-III (MultiNest) posteriors are weighted samples, so every statistic here takes the weights: an unweighted
average would mix in the low-likelihood samples drawn on the way in from the prior.
"""
import numpy as np


def weighted_quantile(x, weight, q):
    """Weighted quantile of the samples x, with the convention TauREx-III uses for its medians."""
    i = np.argsort(x)
    x, w = np.asarray(x)[i], np.asarray(weight)[i]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(q, c, x)


def summary(posterior, param, stat):
    """'mean' (f of Sec. 2.5), 'median', 'p16' or 'p84' of one parameter of a posterior DataFrame, weighted."""
    x = posterior[param].values
    w = posterior["weight"].values
    if stat == "mean":
        return float(np.average(x, weights=w))
    q = {"median": 0.5, "p16": 0.16, "p84": 0.84}[stat]
    return float(weighted_quantile(x, w, q))


def gaussian_fwhm(x, fwhm):
    """Peak-1 Gaussian in wavelength separation x with the given FWHM: the fit of Fig 2 (Sec. 5.1)."""
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-0.5 * (x / sigma) ** 2)


def minmax(spectra):
    """Min-max normalisation of each spectrum over wavelength."""
    lo, hi = spectra.min(axis=-1, keepdims=True), spectra.max(axis=-1, keepdims=True)
    return (spectra - lo) / (hi - lo)


def metric(spectra, log_y, normalise=True, weight=None):
    """rho(D_m, log Y_p): Pearson correlation, across forward models, between each bin m and one log abundance.

    Global metric (Eq. 14): forward models across the prior, normalise=True, unweighted. Each spectrum is min-max
    normalised because its overall depth, set by the scale height, varies ~50x more than its features (Sec. 7.2.2);
    spectra flat across the band cannot be normalised and are left out.
    Local metric (Eq. 15): forward models at posterior samples, raw spectra (normalise=False), weighted by the
    posterior weights.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        s = minmax(spectra) if normalise else np.asarray(spectra, float)
    keep = np.all(np.isfinite(s), axis=1)
    s, log_y = s[keep], np.asarray(log_y, float)[keep]
    w = np.full(len(s), 1.0 / len(s)) if weight is None else np.asarray(weight, float)[keep] / np.sum(np.asarray(weight)[keep])
    ds = s - w @ s
    dy = log_y - w @ log_y
    sd = np.sqrt(w @ ds ** 2)
    a = np.divide(ds, sd, out=np.zeros_like(ds), where=sd > 0)  # a bin that never varies carries no information
    return w @ (a * (dy / np.sqrt(w @ dy ** 2))[:, None])
