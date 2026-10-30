"""Shared plot style and helpers for figure_code/: RASTI style, one colour per gas, heatmaps, weighted contours."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RASTI = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# One viridis colour per gas in every figure, placed to separate the gases that share a panel
VIRIDIS_POSITION = {"H2O": 0.000, "H2S": 0.102, "CO": 0.204, "SO2": 0.307, "NH3": 0.409, "C2H4": 0.511,
                    "C2H2": 0.613, "CH4": 0.716, "HCN": 0.818, "CO2": 0.920}
COLOURS = {m: plt.cm.viridis(p) for m, p in VIRIDIS_POSITION.items()}

TEX = {"H2O": "H_2O", "CO2": "CO_2", "CH4": "CH_4", "CO": "CO", "NH3": "NH_3", "C2H2": "C_2H_2", "SO2": "SO_2",
       "H2S": "H_2S", "HCN": "HCN", "C2H4": "C_2H_4"}

# rho(D_m, log Y_p), the metric of Eqs. 14 and 15
RHO = r"$\rho\left(\mathcal{D}_{m}, \log \mathbf{%s}\right)$"


def use(minor_ticks=False):
    plt.rcParams.update(RASTI)
    plt.rcParams.update({"xtick.minor.visible": minor_ticks, "ytick.minor.visible": minor_ticks})


def save(fig, name, **kw):
    """figure_code/outputs/<name>.pdf, and a .png preview."""
    out = Path(__file__).resolve().parents[1] / "figure_code" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight", "pad_inches": 0.02, **kw}
    fig.savefig(out / f"{name}.pdf", **kw)
    fig.savefig(out / f"{name}.png", **kw)
    return out / f"{name}.pdf"


def bin_edges_with_gaps(wavelength):
    """Cell edges for a binned spectrum; a detector gap (spacing > 2x median) gets its own empty cell."""
    spacing = np.diff(wavelength)
    median = np.median(spacing)
    edges, columns = [wavelength[0] - spacing[0] / 2], []
    for i, wl in enumerate(wavelength):
        right = wl + (spacing[i] / 2 if i < len(spacing) else spacing[-1] / 2)
        if i < len(spacing) and spacing[i] > 2 * median:
            right = wl + median / 2
            edges += [right, wavelength[i + 1] - median / 2]
            columns += [i, None]
        else:
            edges.append(right)
            columns.append(i)
    return np.array(edges), columns


def expand_columns(values, columns):
    """values (..., n_wl) -> (..., n_cells) with NaN in the gap cells of bin_edges_with_gaps."""
    out = np.full(values.shape[:-1] + (len(columns),), np.nan)
    for j, i in enumerate(columns):
        if i is not None:
            out[..., j] = values[..., i]
    return out


# ---------------------------------------------------------------- response-gradient heatmaps (Sec. 7.1; Figs 4, 5)
GRADIENT = r"$\partial\Delta\log\mathbf{H_2O}/\partial\xi$" + "\n" + r"(dex $\sigma^{-1}$)"


def xi_labels(xis):
    return [f"{x:g}".replace("-", "\u2212") for x in xis]


def edges(centres):
    centres = np.asarray(centres, float)
    mid = (centres[:-1] + centres[1:]) / 2
    return np.concatenate([[centres[0] - (mid[0] - centres[0])], mid, [centres[-1] + (centres[-1] - mid[-1])]])


def colorbar(mesh, cax, label):
    """Colour bar with rasterized solids: a vector mesh shows hairline seams between cells in PDF viewers."""
    cbar = plt.colorbar(mesh, cax=cax, label=label)
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")
    return cbar


def gradient_map(ax, cax, gradient, centres, xis):
    """Response-gradient heatmap: each cell spans the interval between two neighbouring magnitudes, whose slope it
    shows, so the mean down a column is the importance (Eq. 13)."""
    v = np.nanmax(np.abs(gradient))
    mesh = ax.pcolormesh(edges(centres), xis, gradient.T, cmap="PuOr", shading="flat", vmin=-v, vmax=v, rasterized=True)
    colorbar(mesh, cax, GRADIENT)
    ax.set_yticks(xis); ax.set_yticklabels(xi_labels(xis))
    ax.set_ylabel(r"$\xi$ ($\sigma$)")
    return mesh


def wavelength_axis(ax, centres):
    ax.set_xticks(centres)
    ax.set_xticklabels([f"{w:.2f}" for w in centres], rotation=45, fontsize=7)
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_xlim(edges(centres)[0], edges(centres)[-1])


# ---------------------------------------------------------------- weighted posterior contours (Figs 9, I1-I4)
SIGMA_TO_FRAC = {1: 0.6827, 2: 0.9545}


def hpd_thresholds(density_flat, sigmas):
    """HPD-style sigma thresholds from a KDE (same convention as corner.py's hist2d)."""
    order = np.argsort(density_flat)[::-1]
    d_sorted = density_flat[order]
    cumfrac = np.cumsum(d_sorted)
    cumfrac /= cumfrac[-1]
    out = []
    for s in sorted(sigmas, reverse=True):          # largest sigma = lowest density first
        i = min(np.searchsorted(cumfrac, SIGMA_TO_FRAC[s]), len(d_sorted) - 1)
        out.append(d_sorted[i])
    return sorted(out)                                # ascending, as ax.contour() needs


def weighted_hist(vals, weights, edges):
    """Weighted 1D histogram: bin centres and density-normalised heights."""
    h, _ = np.histogram(vals, bins=edges, weights=weights, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, h
