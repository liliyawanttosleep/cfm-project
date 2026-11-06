"""Figures for Part II of the talk. Running example: x0 ~ N(0, 1), x1 = +-2 with equal probability (no training).

The point (t, x) = (0.5, 0.1) is followed through the slides: exactly two pairs pass through it.

    python slides/make_cfm_figs.py slides/figs

Outputs
    one_pair.png        one straight conditional path and its constant velocity
    all_pairs.png       6 pairs, 200 pairs, and the resulting density path p_t
    posterior_mean.png  the two lines through (0.5, 0.1), their weights and the average velocity
    closed_form.png     p_t, the exact marginal velocity and integrated paths
    loss_parabola.png   expected loss at (0.5, 0.1) as a function of the network output
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
DARK_ORANGE = "#B35C12"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "text.color": NAVY, "axes.labelcolor": NAVY, "axes.edgecolor": GRID,
                     "xtick.color": NAVY, "ytick.color": NAVY})
A, DPI = 2.0, 200
T0, X0 = 0.5, 0.1
XP, XM = (X0 - T0 * A) / (1 - T0), (X0 + T0 * A) / (1 - T0)   # starts of the two lines through the point
VP, VM = A - XP, -A - XM
W_P = 1 / (1 + np.exp(-2 * A * T0 * X0 / (1 - T0) ** 2))
U0 = W_P * VP + (1 - W_P) * VM
FLOOR = W_P * (1 - W_P) * (VP - VM) ** 2
print(f"x0 = {XP:.2f}, {XM:.2f}; v = {VP:.2f}, {VM:.2f}; weight = {W_P:.3f}; u = {U0:.3f}; floor = {FLOOR:.3f}")
cmap = LinearSegmentedColormap.from_list("path", [BG, "#F6DDC2", "#EDB27A", ORANGE])


def velocity(x, t):
    return (A * np.tanh(A * t * x / (1 - t) ** 2) - x) / (1 - t)


def density(x, t):
    gauss = lambda m: np.exp(-((x - m) ** 2) / (2 * (1 - t) ** 2)) / ((1 - t) * np.sqrt(2 * np.pi))
    return 0.5 * gauss(t * A) + 0.5 * gauss(-t * A)


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def noise_and_data(ax, width=0.17):
    """Draw p0 sideways to the left of t = 0 and the two data points at t = 1."""
    xs = np.linspace(-3.2, 3.2, 300)
    p = np.exp(-(xs**2) / 2) / np.sqrt(2 * np.pi)
    ax.fill_betweenx(xs, -p / p.max() * width, 0, color=GRAY, alpha=0.3, lw=0)
    ax.plot(-p / p.max() * width, xs, c=GRAY, lw=1.5)
    ax.scatter([1, 1], [A, -A], s=110, c=NAVY, zorder=5)
    ax.axvline(0, c=GRID, lw=1); ax.axvline(1, c=GRID, lw=1)


# ---- one pair ----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.8, 5.6), dpi=DPI)
noise_and_data(ax)
ax.plot([0, 1], [XP, A], c=ORANGE, lw=3.5)
ax.scatter([0], [XP], s=110, c=DARK_ORANGE, zorder=5)
for t in [0.25, 0.5, 0.75]:
    x = (1 - t) * XP + t * A
    ax.annotate("", xy=(t + 0.13, x + 0.13 * VP), xytext=(t, x), zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=3, mutation_scale=24))
    ax.scatter([t], [x], s=60, c=NAVY, zorder=6)
ax.text(0.02, XP - 0.5, r"$x_0=-1.8$", color=DARK_ORANGE)
ax.text(0.97, A + 0.3, r"$x_1=2$", color=NAVY, ha="right")
ax.text(0.53, -0.7, r"velocity $x_1-x_0=3.8$" + "\nat every t", color=NAVY)
ax.text(-0.19, 2.85, r"$p_0$", color=GRAY)
ax.text(0.93, -2.75, "data", color=NAVY)
ax.set_xlim(-0.2, 1.08); ax.set_ylim(-3.3, 3.3)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xlabel("time t  (noise at 0, data at 1)"); ax.set_ylabel("position x")
clean(ax)
fig.tight_layout()
fig.savefig(OUT / "one_pair.png")
plt.close(fig)

# ---- all pairs ---------------------------------------------------------------
rng = np.random.default_rng(4)
fig, axes = plt.subplots(1, 3, figsize=(16.64, 4.0), dpi=DPI, sharey=True)
for ax, n, title in zip(axes[:2], (6, 200), ("6 pairs", "200 pairs")):
    x0, x1 = rng.standard_normal(n), rng.choice([-A, A], n)
    ax.plot([0, 1], np.stack([x0, x1]), c=ORANGE, lw=2 if n < 10 else 0.6, alpha=1 if n < 10 else 0.5)
    ax.scatter(np.zeros(n), x0, s=40 if n < 10 else 6, c=DARK_ORANGE, zorder=5)
    ax.scatter([1, 1], [A, -A], s=70, c=NAVY, zorder=5)
    ax.set_title(title)
ts, xs = np.linspace(0, 0.985, 300), np.linspace(-3.4, 3.4, 300)
img = np.stack([density(xs, t) for t in ts], axis=1)
axes[2].imshow(img / img.max(axis=0, keepdims=True), origin="lower", aspect="auto", extent=(0, 1, -3.4, 3.4), cmap=cmap)
axes[2].set_title(r"all pairs: density $p_t(x)$")
for ax in axes:
    ax.set_xlim(-0.03, 1.03); ax.set_ylim(-3.4, 3.4); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([-2, 0, 2]); ax.set_xlabel("t")
    clean(ax)
axes[0].set_ylabel("x")
fig.tight_layout(w_pad=2)
fig.savefig(OUT / "all_pairs.png")
plt.close(fig)

# ---- posterior mean ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.8, 5.9), dpi=DPI)
noise_and_data(ax)
ax.plot([0, 1], [XP, A], c=ORANGE, lw=1.5 + 5 * W_P, alpha=0.9)
ax.plot([0, 1], [XM, -A], c=BLUE, lw=1.5 + 5 * (1 - W_P), alpha=0.9)
ax.scatter([0], [XP], s=110, c=DARK_ORANGE, zorder=5)
ax.scatter([0], [XM], s=110, c=BLUE, zorder=5)
ax.annotate("", xy=(T0 + 0.13, X0 + 0.13 * U0), xytext=(T0, X0), zorder=7,
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=3.5, mutation_scale=28))
ax.scatter([T0], [X0], s=110, c=NAVY, zorder=8)
ax.text(0.02, XP - 0.55, r"$x_0=-1.8$", color=DARK_ORANGE)
ax.text(0.02, XM + 0.28, r"$x_0=2.2$", color=BLUE)
ax.text(0.45, 2.85, "velocity +3.8\nweight 0.69", color=DARK_ORANGE, ha="left", va="top", fontsize=17)
ax.text(0.45, -2.85, "velocity −4.2\nweight 0.31", color=BLUE, ha="left", va="bottom", fontsize=17)
ax.text(0.65, 0.04, r"$u_t(x)=1.32$", color=NAVY, ha="left", va="center", fontsize=20)
ax.text(-0.19, 2.85, r"$p_0$", color=GRAY)
ax.set_xlim(-0.2, 1.08); ax.set_ylim(-3.3, 3.3)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xlabel("time t"); ax.set_ylabel("position x")
clean(ax)
fig.tight_layout()
fig.savefig(OUT / "posterior_mean.png")
plt.close(fig)

# ---- closed form -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.8, 5.6), dpi=DPI)
ts, xs = np.linspace(0, 0.985, 300), np.linspace(-3.4, 3.4, 300)
img = np.stack([density(xs, t) for t in ts], axis=1)
ax.imshow(img / img.max(axis=0, keepdims=True), origin="lower", aspect="auto", extent=(0, 1, -3.4, 3.4), cmap=cmap)
fine = np.linspace(0, 1, 801)
x = np.linspace(-2.6, 2.6, 16)
paths = [x]
for a, b in zip(fine[:-1], fine[1:]):
    paths.append(paths[-1] + (b - a) * velocity(paths[-1], a))
ax.plot(fine, np.array(paths), c=NAVY, lw=1.6)
ax.scatter([T0], [X0], s=120, c=DARK_ORANGE, zorder=6, edgecolors=BG, linewidths=1.5)
ax.annotate("", xy=(T0 + 0.13, X0 + 0.13 * U0), xytext=(T0, X0), zorder=7,
            arrowprops=dict(arrowstyle="-|>", color=DARK_ORANGE, lw=3, mutation_scale=24))
ax.text(T0 + 0.02, X0 - 0.62, r"$u=1.32$", color=DARK_ORANGE, fontweight="bold")
ax.set_xlim(0, 1); ax.set_ylim(-3.4, 3.4)
ax.set_xlabel("time t"); ax.set_ylabel("position x")
clean(ax)
fig.tight_layout()
fig.savefig(OUT / "closed_form.png")
plt.close(fig)

# ---- loss parabola -----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.8, 4.0), dpi=DPI)
v = np.linspace(-5.5, 5.5, 300)
ax.plot(v, (v - U0) ** 2 + FLOOR, c=NAVY, lw=3)
ax.axhline(FLOOR, c=GRAY, ls="--", lw=1.5)
ax.text(-5.3, 6, "floor = conditional variance = 13.69", color="#5B6B82", ha="left", va="center", fontsize=16, zorder=6,
        bbox=dict(facecolor=BG, edgecolor="none", pad=2))
ax.scatter([U0], [FLOOR], s=130, c=DARK_ORANGE, zorder=5)
ax.annotate("minimum at the\nposterior mean, v = 1.32", xy=(U0, FLOOR), xytext=(U0, 34), ha="center", color=DARK_ORANGE, fontsize=16,
            arrowprops=dict(arrowstyle="-", color=DARK_ORANGE, lw=1.2))
for tgt, c, lab in ((VP, ORANGE, "target +3.8"), (VM, BLUE, "target −4.2")):
    ax.axvline(tgt, c=c, lw=2, alpha=0.8)
    ax.text(tgt, 62, lab, color=c, ha="center", va="bottom", fontsize=16)
ax.set_xlim(-5.5, 5.5); ax.set_ylim(0, 62)
ax.set_xlabel(r"network output $v$ at $(t,x)=(0.5,\,0.1)$"); ax.set_ylabel("expected loss")
clean(ax)
fig.tight_layout()
fig.savefig(OUT / "loss_parabola.png")
plt.close(fig)
print("saved Part II figures to", OUT)
