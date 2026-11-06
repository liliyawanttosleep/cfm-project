"""Figures for Part IV of the talk: diffusion as the same regression on another path.

The running example continues: x0 ~ N(0, 1), x1 = +-2, exact field, no network.
The numbers the slides quote come from slides/make_link_numbers.py.

    python slides/make_link_figs.py slides/figs

Outputs
    vp_paths.png    coefficients of both paths; marginal paths on each, from the same starting points
    vp_overlay.png  VP paths after rescaling fall on the straight-path ones; where each sampler spends its steps
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

NAVY, BG, GRID, GRAY, ORANGE, BLUE = "#14213D", "#FAF8F3", "#D9DDE3", "#8A94A3", "#D9822B", "#2F6DB5"
DARK_ORANGE, MID = "#B35C12", "#5B6B82"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "text.color": NAVY, "axes.labelcolor": NAVY, "axes.edgecolor": GRID,
                     "xtick.color": NAVY, "ytick.color": NAVY})
A, DPI, T = 2.0, 200, 1000
abar = np.cumprod(1 - np.linspace(1e-4, 0.02, T))
START = 1.0


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def x1_hat(x, a, b):
    return A * np.tanh(A * a * x / b ** 2)


def vp_grid():
    """Every DDPM step, in our direction of time; ends at pure data."""
    idx = np.arange(T - 1, -1, -1)
    a = np.append(np.sqrt(abar[idx]), 1.0)
    b = np.append(np.sqrt(1 - abar[idx]), 0.0)
    return a, b, np.append(1 - idx / T, 1.0)


def vp_paths(x0):
    a, b, t = vp_grid()
    xs = [np.asarray(x0, dtype=float)]
    for k in range(T):
        x = xs[-1]
        x1 = x1_hat(x, a[k], b[k])
        e = (x - a[k] * x1) / b[k]
        xs.append(a[k + 1] * x1 + b[k + 1] * e)
    return a, b, t, np.stack(xs)


def straight_paths(x0, steps=2000, stop=0.999):
    ts = np.linspace(0, stop, steps + 1)
    xs = [np.asarray(x0, dtype=float)]
    for k in range(steps):
        t = ts[k]
        xs.append(xs[-1] + (ts[k + 1] - t) * (x1_hat(xs[-1], t, 1 - t) - xs[-1]) / (1 - t))
    return ts, np.stack(xs)


def make_paths():
    starts = np.linspace(-2.4, 2.4, 25)
    fig, axes = plt.subplots(1, 3, figsize=(16.64, 3.7), dpi=DPI)
    # coefficients
    ax = axes[0]
    clean(ax)
    a, b, t, _ = vp_paths(np.array([0.0]))
    tt = np.linspace(0, 1, 200)
    ax.plot(tt, tt, c=NAVY, lw=2.2, ls=(0, (4, 3)))
    ax.plot(tt, 1 - tt, c=GRAY, lw=2.2, ls=(0, (4, 3)))
    ax.plot(t, a, c=NAVY, lw=3)
    ax.plot(t, b, c=GRAY, lw=3)
    ax.text(1.04, 1.0, "aₜ: data", color=NAVY, fontsize=18, va="center", clip_on=False)
    ax.text(1.04, 0.0, "bₜ: noise", color=MID, fontsize=18, va="center", clip_on=False)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.08)
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_xlabel("t")
    ax.set_title("Weights: dashed straight, solid VP", fontsize=19, loc="left")
    # paths
    ts, xs = straight_paths(starts)
    ts1, x1 = straight_paths(np.array([START]))
    _, _, tv, xv = vp_paths(starts)
    _, _, _, xv1 = vp_paths(np.array([START]))
    for ax, (tg, xg, tg1, xg1, title) in zip(axes[1:], [(ts, xs, ts1, x1, "Straight path (I-CFM)"), (tv, xv, tv, xv1, "VP path (DDPM, DDIM)")]):
        clean(ax)
        ax.plot(tg, xg, c=GRAY, lw=0.8, alpha=0.5)
        ax.plot(tg1, xg1[:, 0], c=DARK_ORANGE, lw=3)
        ax.scatter([0], [START], s=70, c=DARK_ORANGE, zorder=6, clip_on=False)
        ax.scatter([1, 1], [-A, A], s=70, c=NAVY, zorder=5, clip_on=False)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-3.2, 3.2)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([-2, 0, 2]); ax.set_xlabel("t"); ax.set_ylabel("x")
        ax.set_title(title, fontsize=19, loc="left")
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT / "vp_paths.png")
    plt.close(fig)
    print("saved vp_paths.png")


def make_overlay():
    starts = np.linspace(-2.4, 2.4, 13)
    a, b, _, xv = vp_paths(starts)
    tau = a / (a + b)
    y = xv / (a + b)[:, None]
    ts, xs = straight_paths(starts)
    fig, (ax, gr) = plt.subplots(2, 1, figsize=(7.8, 5.0), dpi=DPI, height_ratios=[3.2, 1.5], sharex=True)
    clean(ax)
    ax.plot(ts, xs, c=NAVY, lw=1.6, alpha=0.8)
    keep = np.searchsorted(tau, np.linspace(0.03, 0.95, 16))
    for j in range(len(starts)):
        ax.scatter(tau[keep], y[keep, j], s=26, c=ORANGE, zorder=4, linewidths=0)
    ax.plot([], [], c=NAVY, lw=1.6, label="straight path")
    ax.scatter([], [], s=50, c=ORANGE, label="VP path, rescaled")
    ax.legend(loc="upper left", frameon=False, fontsize=18, ncol=2, bbox_to_anchor=(-0.02, 1.2), handletextpad=0.3, columnspacing=1.0)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-2.8, 2.8)
    ax.set_yticks([-2, 0, 2]); ax.set_ylabel("x / (a + b)")
    ax.tick_params(axis="x", length=0)
    # where the steps go, for 5 network calls
    gr.axis("off")
    gr.set_ylim(-0.75, 1.75)
    equal = np.linspace(0, 1, 6)
    ddim_idx = np.round(np.linspace(T - 1, 0, 5)).astype(int)
    aa, bb = np.append(np.sqrt(abar[ddim_idx]), 1.0), np.append(np.sqrt(1 - abar[ddim_idx]), 0.0)
    ddim = aa / (aa + bb)
    print("DDIM grid, 5 calls:", np.round(ddim, 3))
    for yy, g, col, marker, name in [(1.1, equal, NAVY, "o", "Equal steps"), (0.1, ddim, DARK_ORANGE, "s", "DDIM")]:
        gr.plot([0, 1], [yy, yy], c=GRID, lw=2)
        gr.scatter(g, [yy] * len(g), s=60, c=col, marker=marker, zorder=3, clip_on=False)
    gr.text(0.03, 1.3, "Equal steps: 5 calls", fontsize=18, color=NAVY, va="bottom")
    gr.text(0.03, 0.3, "DDIM: 5 calls", fontsize=18, color=DARK_ORANGE, va="bottom")
    gr.text(0.75, 0.3, "one call: 0.51 → 0.99", fontsize=18, color=DARK_ORANGE, va="bottom", ha="center")
    for v in [0, 0.5, 1]:
        gr.text(v, -0.2, f"{v:g}", fontsize=18, ha="center", va="top")
    gr.text(0.25, -0.2, "τ = a / (a + b)", fontsize=18, ha="center", va="top")
    fig.subplots_adjust(left=0.1, right=0.985, top=0.9, bottom=0.04, hspace=0.08)
    fig.savefig(OUT / "vp_overlay.png")
    plt.close(fig)
    err = max(np.abs(np.interp(ts[:-1], tau, y[:, j]) - xs[:-1, j]).max() for j in range(len(starts)))
    print(f"saved vp_overlay.png; largest gap between the rescaled VP paths and the straight-path ones: {err:.3f}")


if __name__ == "__main__":
    make_paths()
    make_overlay()
