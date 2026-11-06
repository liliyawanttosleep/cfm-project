"""Animated figures for the slides (moons, seed 0 checkpoints from results/e1/ckpt).

    python slides/make_slide_gifs.py slides/figs

Outputs
    sampling.gif         noise -> data for OT-CFM, I-CFM and DDIM, same starting noise (online deck)
    sampling_strip.png   static snapshots of the same process (used in the PDF instead of the GIF)
    paths.gif            probability paths p_t: flow-matching interpolation vs diffusion noising (online deck)
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "e1_2d"))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "slides" / "figs")
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common.data_2d import sample
from common.models import MLP
from train_fm import euler_sample
from train_ddpm import Schedule, ddim_sample

NAVY, BG, GRID, GRAY = "#14213D", "#FAF8F3", "#D9DDE3", "#8A94A3"
COLORS = {"OT-CFM": "#D9822B", "I-CFM": "#B8651B", "DDIM": "#2F6DB5"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "figure.facecolor": BG,
                     "axes.facecolor": BG, "savefig.facecolor": BG})
dev = "cuda" if torch.cuda.is_available() else "cpu"
LIM, STEPS = 3.0, 60


def load(name):
    m = MLP(width=256).to(dev)
    m.load_state_dict(torch.load(ROOT / "results/e1/ckpt" / f"{name}.pt", map_location=dev))
    return m.eval()


def style(ax):
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


def frame(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def save_gif(frames, path, hold=25):
    frames = frames + [frames[-1]] * hold
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=70, loop=0, optimize=True)
    print("saved", path, f"{path.stat().st_size / 1e6:.1f} MB")


# ---- sampling trajectories -------------------------------------------------
models = {"OT-CFM": load("otcfm_moons_s0"), "I-CFM": load("icfm_moons_s0"), "DDIM": load("ddim_moons_s0")}
sch = Schedule(1000, "linear", dev)
x0 = torch.randn(1500, 2, generator=torch.Generator().manual_seed(7)).to(dev)
with torch.no_grad():
    trajs = {n: (ddim_sample(m, sch, x0, STEPS) if n == "DDIM" else euler_sample(m, x0, STEPS)).cpu().numpy()
             for n, m in models.items()}

TRAIL = 120  # number of particles drawn with their trail
frames = []
for k in range(STEPS + 1):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for ax, (name, tr) in zip(axes, trajs.items()):
        ax.plot(tr[: k + 1, :TRAIL, 0], tr[: k + 1, :TRAIL, 1], c=COLORS[name], alpha=0.35, lw=0.8)
        ax.scatter(tr[k, :, 0], tr[k, :, 1], s=3, c=COLORS[name], alpha=0.7)
        ax.set_title(name, color=NAVY, fontsize=20, pad=8)
        style(ax)
    fig.suptitle(f"step {k} / {STEPS}   (t = {k / STEPS:.2f})", color=NAVY, fontsize=17, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    frames.append(frame(fig))
save_gif(frames, OUT / "sampling.gif")

# static strip for the PDF
ks = [0, 15, 30, 45, 60]
fig, axes = plt.subplots(3, len(ks), figsize=(15, 9.4))
for r, (name, tr) in enumerate(trajs.items()):
    for c, k in enumerate(ks):
        ax = axes[r, c]
        ax.plot(tr[: k + 1, :TRAIL, 0], tr[: k + 1, :TRAIL, 1], c=COLORS[name], alpha=0.3, lw=0.7)
        ax.scatter(tr[k, :, 0], tr[k, :, 1], s=2, c=COLORS[name], alpha=0.7)
        style(ax)
        if r == 0:
            ax.set_title(f"t = {k / STEPS:.2f}", color=NAVY, fontsize=19)
        if c == 0:
            ax.set_ylabel(name, color=NAVY, fontsize=19)
fig.tight_layout()
fig.savefig(OUT / "sampling_strip.png", dpi=100)
plt.close(fig)
print("saved", OUT / "sampling_strip.png")

# ---- probability paths ------------------------------------------------------
x1 = sample("moons", 1500, np.random.default_rng(3)).numpy()
z = torch.randn(1500, 2, generator=torch.Generator().manual_seed(11)).numpy()
frames = []
for t in np.linspace(0, 1, 41):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.4))
    xt = (1 - t) * z + t * x1
    ab = float(sch.abar[int(round((1 - t) * 999))]) if t < 1 else 1.0
    xd = np.sqrt(ab) * x1 + np.sqrt(1 - ab) * z
    axes[0].scatter(xt[:, 0], xt[:, 1], s=3, c=COLORS["OT-CFM"], alpha=0.7)
    axes[1].scatter(xd[:, 0], xd[:, 1], s=3, c=COLORS["DDIM"], alpha=0.7)
    axes[0].set_title("Flow matching: (1−t)·x₀ + t·x₁", color=NAVY, fontsize=16)
    axes[1].set_title("Diffusion: √ᾱ·x₁ + √(1−ᾱ)·ε", color=NAVY, fontsize=16)
    for ax in axes:
        style(ax)
    fig.suptitle(f"t = {t:.2f}   (0 = noise, 1 = data)", color=NAVY, fontsize=17, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    frames.append(frame(fig))
save_gif(frames, OUT / "paths.gif")
