"""Fig 2: Pearson correlation between every pair of bins of the HD 209458 b grid against their separation, in 0.05 um
bins (standard error; bins of five pairs or fewer excluded), and the Gaussian fit that sets the kernel FWHM (Sec. 5.1).

Reads: E1_HD-209458-b/outputs/correlation_separation.csv, gaussian_fit.csv
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator

from src import plotting as style
from src.metrics import gaussian_fwhm

E1 = Path(__file__).resolve().parents[1] / "E1_HD-209458-b" / "outputs"


def gaussian_band(x, popt, pcov):
    """1 sigma band of the fit, propagated linearly from the FWHM uncertainty (numerical derivative)."""
    eps = 1e-6
    p_up = popt + eps
    p_down = popt - eps
    J = ((gaussian_fwhm(x, *p_up) - gaussian_fwhm(x, *p_down)) / (2 * eps)).reshape(-1, 1)
    var = np.einsum('ij,jk,ik->i', J, pcov, J)
    return np.sqrt(var)


style.use(minor_ticks=True)
binned = pd.read_csv(E1 / "correlation_separation.csv")
fit = pd.read_csv(E1 / "gaussian_fit.csv").iloc[0]
BIN_WIDTH, MIN_BIN_COUNT = fit["bin_width_um"], int(fit["min_bin_count"])

x_data = binned['separation_um'].values
y_data = binned['rho'].values
yerr   = binned['rho_err'].values

finite_mask = ~binned['excluded'].values
x_fit, yerr_fit = x_data[finite_mask], yerr[finite_mask]
x_excluded = x_data[~finite_mask]  # bins of MIN_BIN_COUNT pairs or fewer, left out of the fit

popt, pcov = np.array([fit['fwhm_um']]), np.array([[fit['fwhm_err_um'] ** 2]])
fwhm_fit = popt[0]

high_res_grid = np.linspace(x_data.min(), x_data.max(), 200)
fit_curve = gaussian_fwhm(high_res_grid, fwhm_fit)
fit_std = gaussian_band(high_res_grid, popt, pcov)

fit_curve_at_data = gaussian_fwhm(x_data, fwhm_fit)
residuals = y_data - fit_curve_at_data

fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True,
    figsize=(3.5, 4.2),
    gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.06}
)

data_color = '0.15'      # near-black
fit_color = '#B22222'    # firebrick, prints well in greyscale too
c3sig = '#B22222'
excl_color = '0.7'       # light grey for excluded/undefined-uncertainty points

ax1.plot(high_res_grid, fit_curve, '-', color=fit_color, lw=1.3,
          label=f'Fit (FWHM={fwhm_fit:.3f} $\\mu$m)', zorder=3)
ax1.fill_between(high_res_grid, fit_curve - 3*fit_std, fit_curve + 3*fit_std,
                  color=c3sig, alpha=0.25, lw=0, label=r'$\pm$3$\sigma$', zorder=2)

# excluded bins: marker only, no error bar
if x_excluded.size > 0:
    ax1.plot(x_excluded, y_data[~finite_mask], 'x', color=excl_color, mew=1.0, zorder=4)

ax1.errorbar(x_fit, y_data[finite_mask], yerr=yerr_fit, fmt='.', color=data_color,  # dots, no caps: the paper's version
             mew=1.0, elinewidth=0.8, capsize=0, ecolor=data_color,
             label=r"$\mu \pm \frac{\sigma}{\sqrt{n}}$", zorder=5)
ax1.set_ylabel(r'Pearson $\rho(\lambda,$ $\lambda^{\dagger})$')

handles, labels = ax1.get_legend_handles_labels()
if x_excluded.size > 0:
    excl_marker_proxy = Line2D([0], [0], marker='x', color=excl_color, mew=1.0, linestyle='None')
    handles.append((excl_marker_proxy))
    labels.append(f'$n\leq{MIN_BIN_COUNT}$ (excl.)')

ax1.legend(handles, labels, loc='upper center', handlelength=1.6, labelspacing=0.4,
           borderaxespad=1, ncols=2,
           handler_map={tuple: HandlerTuple(ndivide=None)})
ax1.tick_params(labelbottom=False)


ax2.fill_between(high_res_grid, -3*fit_std, 3*fit_std,
                  color=c3sig, alpha=0.25, lw=0, zorder=2)
ax2.axhline(0, color=fit_color, lw=1.0, alpha=0.9, zorder=3)

if x_excluded.size > 0:
    ax2.plot(x_excluded, residuals[~finite_mask], 'x', color=excl_color, mew=1.0, zorder=4)

ax2.errorbar(x_fit, residuals[finite_mask], yerr=yerr_fit, fmt='.', color=data_color,
             mew=1.0, elinewidth=0.8, capsize=0, ecolor=data_color, zorder=5)

ax2.set_xlabel(f'$\\lambda - \\lambda^{{\\dagger}} $ ({BIN_WIDTH} $\\mu$m bins)')
ax2.set_ylabel('Residuals')

ax1.set_ylim(None, 1.3)

for ax in (ax1, ax2):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())


fig.align_ylabels([ax1, ax2])
plt.tight_layout()
print("saved", style.save(fig, Path(__file__).stem))
