"""Slides 用图：从 results/e1/ckpt 读已训练模型，画轨迹对比、样本随 NFE 网格、概率路径示意、W2 vs NFE。"""
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Wentao\Desktop\cfm_project")
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "e1_2d"))
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.data_2d import sample
from common.models import MLP
from train_fm import euler_sample
from train_ddpm import Schedule, ddim_sample

NAVY, BG, GRID = "#14213D", "#FAF8F3", "#D9DDE3"
ORANGE, BLUE, GRAY = "#D9822B", "#2F6DB5", "#8A94A3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.edgecolor": GRID,
                     "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG})

dev = "cuda" if torch.cuda.is_available() else "cpu"
def load(name):
    m = MLP(width=256).to(dev)
    m.load_state_dict(torch.load(ROOT / "results/e1/ckpt" / f"{name}.pt", map_location=dev))
    return m.eval()

models = {"OT-CFM": load("otcfm_moons_s0"), "I-CFM": load("icfm_moons_s0"), "DDIM": load("ddim_moons_s0")}
sch = Schedule(1000, "linear", dev)
def sampler(name, x0, nfe):
    if name == "DDIM":
        return ddim_sample(models[name], sch, x0, nfe).cpu().numpy()
    return euler_sample(models[name], x0, nfe).cpu().numpy()
color = {"OT-CFM": ORANGE, "I-CFM": "#B8651B", "DDIM": BLUE}

g = torch.Generator().manual_seed(7)
x0 = torch.randn(2000, 2, generator=g).to(dev)
real = sample("moons", 2000, np.random.default_rng(3)).numpy()
LIM = 3.0
def style(ax):
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color(GRID)

# 1) 轨迹对比
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
for ax, name in zip(axes, models):
    tr = sampler(name, x0[:250], 100)
    ax.plot(tr[:, :, 0], tr[:, :, 1], c=color[name], alpha=0.35, lw=0.8)
    ax.scatter(tr[0, :, 0], tr[0, :, 1], s=6, c=GRAY, zorder=3)
    ax.scatter(tr[-1, :, 0], tr[-1, :, 1], s=6, c=NAVY, zorder=3)
    ax.set_title(name, color=NAVY, fontsize=20, pad=10)
    style(ax)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "traj_compare.png", dpi=110)
plt.close(fig)

# 2) 样本随 NFE
nfes = [1, 2, 5, 20]
fig, axes = plt.subplots(3, len(nfes), figsize=(13, 9.6))
for r, name in enumerate(models):
    for c, n in enumerate(nfes):
        s = sampler(name, x0, n)[-1]
        ax = axes[r, c]
        ax.scatter(s[:, 0], s[:, 1], s=2, c=color[name], alpha=0.6)
        style(ax)
        if r == 0: ax.set_title(f"NFE = {n}", color=NAVY, fontsize=20, pad=8)
        if c == 0: ax.set_ylabel(name, color=NAVY, fontsize=20)
fig.tight_layout()
fig.savefig(OUT / "samples_grid.png", dpi=110)
plt.close(fig)

# 3) 概率路径示意：FM 线性插值 vs Diffusion 前向加噪（都按“噪声 → 数据”方向画）
ts = [0.0, 0.25, 0.5, 0.75, 1.0]
x0c, x1 = torch.randn(2000, 2, generator=g).numpy(), real
fig, axes = plt.subplots(2, 5, figsize=(16, 6.8))
for c, t in enumerate(ts):
    xt = (1 - t) * x0c + t * x1
    axes[0, c].scatter(xt[:, 0], xt[:, 1], s=2, c=ORANGE, alpha=0.6)
    axes[0, c].set_title(f"t = {t}", color=NAVY, fontsize=19)
    ab = float(sch.abar[int(round((1 - t) * 999))]) if t < 1 else 1.0  # t=1 数据端，t=0 噪声端
    xd = np.sqrt(ab) * x1 + np.sqrt(1 - ab) * x0c
    axes[1, c].scatter(xd[:, 0], xd[:, 1], s=2, c=BLUE, alpha=0.6)
    style(axes[0, c]); style(axes[1, c])
axes[0, 0].set_ylabel("Flow Matching\n(1-t)x0 + t x1", color=NAVY, fontsize=16)
axes[1, 0].set_ylabel("Diffusion (VP)\n√ᾱ x1 + √(1-ᾱ) ε", color=NAVY, fontsize=16)
fig.tight_layout()
fig.savefig(OUT / "prob_paths.png", dpi=110)
plt.close(fig)

# 4) W2 vs NFE
df = pd.read_csv(ROOT / "results/e1/summary.csv")
d = df[df.data == "moons"]
fig, ax = plt.subplots(figsize=(9, 6))
lab = {"otcfm": ("OT-CFM", ORANGE, "o"), "icfm": ("I-CFM", "#B8651B", "s"), "ddim": ("DDIM", BLUE, "^"), "ddpm": ("DDPM (1000 steps)", "#1D4A80", "D")}
for m, (name, col, mk) in lab.items():
    r = d[d.method == m]
    ax.plot(r.nfe, r.w2_mean, marker=mk, ms=8, lw=2.5, c=col, label=name)
ax.axhline(d.ref_w2.mean(), ls="--", c=GRAY, lw=1.5, label="real vs real (floor)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("NFE (network evaluations per sample)", color=NAVY)
ax.set_ylabel("W2 distance to test set", color=NAVY)
ax.tick_params(colors=NAVY)
ax.grid(True, which="major", color=GRID, lw=0.8)
ax.legend(frameon=False, fontsize=14)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "w2_vs_nfe.png", dpi=110)
plt.close(fig)
print("ok", sorted(p.name for p in OUT.glob("*.png")))
