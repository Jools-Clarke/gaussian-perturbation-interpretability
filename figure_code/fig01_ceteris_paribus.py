"""Fig 1: ceteris paribus profiles (Sec. 3) of a toy model with two inputs: each profile varies one input with the
other fixed at the sample X. A schematic; reads no data."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import plotting as style


def surface(x1, x2):
    """The toy model f(x1, x2)."""
    return (np.sin(x1)*np.cos(x2) + 0.5*np.sin(2*x1 + 0.7*x2)
            + 0.4*np.cos(0.5*x1*x2)*np.exp(-((x1-1)**2 + (x2+0.5)**2)/4))


x = np.linspace(-6, 6, 200)
y = np.linspace(-6, 6, 200)
X, Y = np.meshgrid(x, y)
Z = surface(X, Y)
x0, y0 = 1.5, -2.0  # the sample X
Zx, Zy, Z_cross = surface(x0, y), surface(x, y0), surface(x0, y0)  # its two profiles, and f(X)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d',computed_zorder=False)

# the planes x1 = x1* and x2 = x2*, cutting the surface along the two profiles
z_plane = np.linspace(-10, 10, 50)
ZP, YP = np.meshgrid(z_plane, y)
XP, ZP2 = np.meshgrid(x, z_plane)
ax.plot_surface(np.full_like(YP, x0), YP, ZP, alpha=0.2, color='red', zorder=2)
ax.plot_surface(XP, np.full_like(XP, y0), ZP2, alpha=0.2, color='blue', zorder=2)
ax.plot_surface(X, Y, Z, cmap='Greys', alpha=0.7, linewidth=0)
ax.plot([x0]*len(y), y, Zx, color='red', linewidth=2.5, label=r'$x_1 = x_1^*$ plane intersection', zorder=4)
ax.plot(x, [y0]*len(x), Zy, color='blue', linewidth=2.5, label=r'$x_2 = x_2^*$ plane intersection', zorder=4)
ax.scatter(x0, y0, Z_cross, color='black', s=80, label=r'$X$ (current sample)', zorder=5)
ax.plot([x0, x0], [y0, y0], [-10, 10], color='purple', linewidth=2.0, linestyle='--', zorder=5)
ax.set_zlim(-10,10)

ax.text(x0-1, y.min(), Z.min(), r'$x_1$ profile', color='red', fontsize=12)
ax.text(x.min()+0.5, y0-2, Z.min(), r'$x_2$ profile', color='blue', fontsize=12)
ax.text(x0+0.2, y0+0.6, Z_cross+0.8, r'$X$ sample', color='black', fontsize=12, zorder=6)
ax.text(-2, 5, 0, r'$f(X)$ model', color='black', fontsize=12, zorder=6)

# no axes: a schematic
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_axis_off()

ax.view_init(elev=35, azim=45)
plt.tight_layout()
print("saved", style.save(plt.gcf(), Path(__file__).stem, dpi=600, transparent=False))
