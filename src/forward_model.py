"""TauREx-III forward models g (Sec. 2.5), built from the same .par file as the retrieval, so the two cannot drift apart.

They give the prior grids of the global metric, the forward models at posterior samples of the local metric,
the contributions (Sec. 7.2), the simulated observation (Sec. 6) and the CNN's training set (Appendix J2).
"""
import numpy as np

import _CONFIG as config


def load_model(par_file, observation_dat):
    """TauREx model from a retrieval .par file in DATA/, every parameter at its .par starting value.

    Returns (model, binner, wngrid): binner bins on to the observation's wavelength grid, and wngrid is that grid in
    wavenumber, which the retrieval also passes to model.model().
    """
    import taurex.log
    from taurex.cache import CIACache, OpacityCache
    from taurex.data.spectrum.observed import ObservedSpectrum
    from taurex.parameter import ParameterParser

    taurex.log.disableLogging()
    pp = ParameterParser()
    pp.read(str(par_file))
    # the .par files leave out the machine's data paths ([Global]); they come from _CONFIG.py
    OpacityCache().set_opacity_path(str(config.CROSS_SECTION_PATH))
    CIACache().set_cia_path(str(config.CIA_PATH))
    model = pp.generate_appropriate_model()
    model.build()
    observation = ObservedSpectrum(str(observation_dat))
    return model, observation.create_binner(), observation.wavenumberGrid


def spectrum(model, binner, wngrid, params):
    """Set model[name] = value for each of params (e.g. {"H2O": 1e-4, "T": 900.0}) and return the binned (Rp/Rs)^2
    in ascending wavelength. Clipping to wngrid, as the retrieval does, gives the same binned spectrum ~15x faster."""
    for name, value in params.items():
        model[name] = value
    wn, binned = binner.bin_model(model.model(wngrid=wngrid))[:2]
    return np.asarray(binned)[np.argsort(10000 / np.asarray(wn))]


def contribution(model, gas, gases, band_um):
    """Contribution of one gas (Sec. 7.2.1): the native spectrum with that gas at VMR 1e-2 and every other gas at
    1e-12, within band_um, in ascending wavelength. Returns (wl_um, contribution)."""
    for g in gases:
        model[g] = 1e-2 if g == gas else 1e-12
    model.model()  # model_full_contrib() reuses this run
    wn, contribs = model.model_full_contrib()
    native = dict((name, absorp) for name, absorp, _, _ in contribs["Absorption"])[gas]
    wl = 10000 / wn
    keep = (wl >= band_um[0]) & (wl <= band_um[1])
    order = np.argsort(wl[keep])
    return wl[keep][order], native[keep][order]


def uniform_binner(wave_min, wave_max, resolution):
    """Binner of constant resolving power R between two wavelengths (um), for spectrum(): the R = 200 recovered
    spectrum of Fig 12. Returns (binner, bin centres in um, ascending)."""
    from taurex.binning import FluxBinner
    from taurex.util.util import create_grid_res

    wavelength, width = create_grid_res(resolution, wave_min, wave_max).T
    wavenumber = 10000 / wavelength
    return FluxBinner(wngrid=wavenumber, wngrid_width=wavenumber * width / wavelength), wavelength


def model_params(theta):
    """Retrieval parameters -> model parameters: log_<name> becomes <name> = 10**value, the rest pass through."""
    return {name.removeprefix("log_"): 10 ** value if name.startswith("log_") else value for name, value in theta.items()}


def wl_columns(df):
    """The spectrum columns of the experiments' CSVs, which are named by their wavelength in um."""
    return [c for c in df.columns if c[0].isdigit()]
