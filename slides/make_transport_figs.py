"""Figures for Part I of the talk, all from the closed-form example N(0, 1) -> two points at +-a (no training).

    python slides/make_transport_figs.py slides/figs

Outputs
    ode_euler.png       velocity field in the (t, x) plane, exact ODE paths and Euler with K = 4
    continuity_1d.png   density p_t(x) and velocity u_t(x) at five times
    many_paths.png      three different density paths with the same endpoints
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "slides" / "figs")
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

NAVY, BG, GRID, GRAY, ORANGE, BLUE = "#14213D", "#FAF8F3", "#D9DDE3", "#8A94A3", "#D9822B", "#2F6DB5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "text.color": NAVY, "axes.labelcolor": NAVY, "axes.edgecolor": GRID,
                     "xtick.color": NAVY, "ytick.color": NAVY})
A = 2.0
DPI = 200


def velocity(x, t):
    return (A * np.tanh(A * t * x / (1 - t) ** 2) - x) / (1 - t)


def density(x, t, a_t=None, b_t=None):
    a_t = t if a_t is None else a_t
    b_t = 1 - t if b_t is None else b_t
    gauss = lambda m: np.exp(-((x - m) ** 2) / (2 * b_t**2)) / (b_t * np.sqrt(2 * np.pi))
    return 0.5 * gauss(a_t * A) + 0.5 * gauss(-a_t * A)


def integrate(x0, ts):
    xs = [x0]
    for t0, t1 in zip(ts[:-1], ts[1:]):
        xs.append(xs[-1] + (t1 - t0) * velocity(xs[-1], t0))
    return np.array(xs)


# ---- ODE and Euler in the (t, x) plane ---------------------------------------
fig, ax = plt.subplots(figsize=(7.8, 5.6), dpi=DPI)
tg, xg = np.meshgrid(np.linspace(0.04, 0.94, 12), np.linspace(-3.2, 3.2, 13))
dt = 0.055
dx = np.clip(velocity(xg, tg) * dt, -0.42, 0.42)
ax.quiver(tg, xg, np.full_like(tg, dt), dx, angles="xy", scale_units="xy", scale=1, width=0.0032, color=GRAY)
fine = np.linspace(0, 1, 801)
x0 = np.linspace(-2.4, 2.4, 14)
ax.plot(fine, integrate(x0, fine), c=ORANGE, lw=1.2, alpha=0.55)
starts = np.array([-1.3, 0.7])
ax.plot(fine, integrate(starts, fine), c=ORANGE, lw=3)
ax.plot([], [], c=ORANGE, lw=3, label="exact ODE path")
coarse = np.linspace(0, 1, 5)
ax.plot(coarse, integrate(starts, coarse), c=NAVY, lw=2.2, marker="o", ms=8)
ax.plot([], [], c=NAVY, lw=2.2, marker="o", ms=8, label="Euler, K = 4 (4 network calls)")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-3.4, 3.4)
ax.set_xlabel("time t  (noise at 0, data at 1)"); ax.set_ylabel("position x")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=16)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "ode_euler.png")
plt.close(fig)

# ---- density and velocity at five times --------------------------------------
TIMES = [0.0, 0.25, 0.5, 0.75, 0.9]
fig, axes = plt.subplots(1, 5, figsize=(16.64, 3.5), dpi=DPI, sharey=True)
xs = np.linspace(-3.6, 3.6, 800)
xa = np.linspace(-3.2, 3.2, 9)
for ax, t in zip(axes, TIMES):
    p = density(xs, t)
    ax.fill_between(xs, 0, p, color=ORANGE, alpha=0.35, lw=0)
    ax.plot(xs, p, c=ORANGE, lw=2)
    u = np.clip(velocity(xa, t) * 0.4, -0.7, 0.7)
    ax.quiver(xa - u / 2, np.full_like(xa, -0.28), u, np.zeros_like(u), angles="xy", scale_units="xy", scale=1,
              width=0.009, headwidth=4, headlength=4, headaxislength=3.5, color=NAVY)
    ax.set_title(f"t = {t:g}")
    ax.set_xlim(-3.6, 3.6); ax.set_ylim(-0.55, 2.2)
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_xlabel("x")
fig.tight_layout(w_pad=1.2)
fig.savefig(OUT / "continuity_1d.png")
plt.close(fig)

# ---- many paths, same endpoints ----------------------------------------------
PATHS = [("Path A", lambda t: (t, 1 - t)),
         ("Path B", lambda t: (np.sin(np.pi * t / 2), np.cos(np.pi * t / 2))),
         ("Path C", lambda t: (t**3, 1 - t**0.5))]
cmap = LinearSegmentedColormap.from_list("path", [BG, "#F2C79A", ORANGE, "#8A4A10"])
fig, axes = plt.subplots(1, 3, figsize=(8.16, 3.3), dpi=DPI, sharey=True)
ts = np.linspace(0.0, 0.985, 300)
xs = np.linspace(-3.4, 3.4, 300)
for ax, (name, path) in zip(axes, PATHS):
    img = np.stack([density(xs, t, *path(t)) for t in ts], axis=1)
    img = img / img.max(axis=0, keepdims=True)
    ax.imshow(img, origin="lower", aspect="auto", extent=(0, 1, -3.4, 3.4), cmap=cmap, vmin=0, vmax=1)
    ax.set_title(name); ax.set_xlabel("t"); ax.set_xticks([0, 1]); ax.set_yticks([-2, 0, 2])
axes[0].set_ylabel("x")
fig.tight_layout(w_pad=0.8)
fig.savefig(OUT / "many_paths.png")
plt.close(fig)
print("saved", OUT / "ode_euler.png", OUT / "continuity_1d.png", OUT / "many_paths.png")
