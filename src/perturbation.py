"""The method itself: naive and Gaussian perturbation of a spectrum, and the importance (Secs 3, 5.2 and 7.1).

Pure numpy, so it applies to any black-box retrieval f. The response (Eq. 12) is one subtraction,
R_m = f(X'_m) - f(x*), done where f is called (E1 step4, E3 step4).
"""
import numpy as np


def gaussian_window(wl, centre, fwhm):
    """w_gauss(eps, m), Eq. 9: a Gaussian in wavelength (not bin index) with FWHM eps, 1 at the centre bin."""
    return np.exp(-4 * np.log(2) * (np.asarray(wl) - centre) ** 2 / fwhm ** 2)


def perturbed_spectra(x, sigma, wl, centre_index, xi, fwhm=0.1, kernel="gaussian"):
    """X'_m = 1_K x* + xi sigma_m w, Eq. 11: one perturbed spectrum per magnitude xi, shape (K, M).

    Every bin is moved by the noise sigma_m of the CENTRE bin m, so the perturbation keeps the smooth shape of the
    window (Sec. 5.2 discusses the alternative, each bin's own noise). kernel="naive" replaces the window with a
    one-hot vector: only bin m moves, the ceteris paribus profile of Eqs. 1-3.
    """
    xi = np.atleast_1d(xi)
    w = gaussian_window(wl, wl[centre_index], fwhm) if kernel == "gaussian" else (np.arange(len(wl)) == centre_index) * 1.0
    return x[None, :] + sigma[centre_index] * xi[:, None] * w[None, :]


def importance(response, xi):
    """Importance_{m,p}, Eq. 13: the mean of the K - 1 slopes of the response between neighbouring magnitudes.

    response: (..., K), the response at each xi (xi = 0 included, where it is 0). Returns dex per sigma.
    """
    xi = np.asarray(xi, float)
    order = np.argsort(xi)
    r = np.take(response, order, axis=-1)
    return (np.diff(r, axis=-1) / np.diff(xi[order])).mean(axis=-1)
