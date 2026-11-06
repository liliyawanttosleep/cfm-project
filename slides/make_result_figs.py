"""E1 result figures across all four 2D datasets (seed 0 checkpoints in results/e1/ckpt).

    python slides/make_result_figs.py slides/figs

Outputs
    datasets_grid.png   rows = datasets, columns = real data and each method at a small sampling budget
    w2_all.png          W2 vs NFE, one panel per dataset (from results/e1/summary.csv)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "code" / "e1_2d"))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "slides" / "figs")

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.data_2d import sample
from common.models import MLP
from train_fm import euler_sample
from train_ddpm import Schedule, ddim_sample, ddpm_sample

NAVY, BG, GRID, GRAY = "#14213D", "#FAF8F3", "#D9DDE3", "#8A94A3"
STYLE = {"otcfm": ("OT-CFM", "#D9822B", "o"), "icfm": ("I-CFM", "#B8651B", "s"),
         "ddim": ("DDIM", "#2F6DB5", "^"), "ddpm": ("DDPM, 1000 steps", "#1D4A80", "D")}
DATASETS = ["moons", "8gaussians", "checkerboard", "swissroll"]
NFE = 5
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "figure.facecolor": BG,
                     "axes.facecolor": BG, "savefig.facecolor": BG})
dev = "cuda" if torch.cuda.is_available() else "cpu"
sch = Schedule(1000, "linear", dev)


def load(name):
    m = MLP(width=256).to(dev)
    m.load_state_dict(torch.load(ROOT / "results/e1/ckpt" / f"{name}.pt", map_location=dev))
    return m.eval()


def style(ax):
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)


# ---- sample grid -----------------------------------------------------------
x0 = torch.randn(2000, 2, generator=torch.Generator().manual_seed(7)).to(dev)
cols = ["real", "otcfm", "icfm", "ddim", "ddpm"]
fig, axes = plt.subplots(len(DATASETS), len(cols), figsize=(15, 12.4))
with torch.no_grad():
    for r, d in enumerate(DATASETS):
        for c, m in enumerate(cols):
            ax = axes[r, c]
            if m == "real":
                s, col = sample(d, 2000, np.random.default_rng(3)).numpy(), NAVY
            else:
                net = load(f"{m}_{d}_s0")
                if m in ("otcfm", "icfm"):
                    s = euler_sample(net, x0, NFE)[-1]
                elif m == "ddim":
                    s = ddim_sample(net, sch, x0, NFE)[-1]
                else:
                    torch.manual_seed(0)
                    s = ddpm_sample(net, sch, x0)[-1]
                s, col = s.cpu().numpy(), STYLE[m][1]
            ax.scatter(s[:, 0], s[:, 1], s=1.5, c=col, alpha=0.6)
            style(ax)
            if r == 0:
                title = "Real data" if m == "real" else (STYLE[m][0] if m == "ddpm" else f"{STYLE[m][0]}, {NFE} steps")
                ax.set_title(title, color=NAVY, fontsize=17, pad=8)
            if c == 0:
                ax.set_ylabel(d, color=NAVY, fontsize=18)
fig.tight_layout()
fig.savefig(OUT / "datasets_grid.png", dpi=100)
plt.close(fig)
print("saved", OUT / "datasets_grid.png")

# ---- W2 vs NFE, one panel per dataset ---------------------------------------
df = pd.read_csv(ROOT / "results/e1/summary.csv")
fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
for ax, d in zip(axes, DATASETS):
    sub = df[df.data == d]
    for m, (name, col, mk) in STYLE.items():
        r = sub[sub.method == m]
        ax.plot(r.nfe, r.w2_mean, marker=mk, ms=7, lw=2.2, c=col, label=name)
    ax.axhline(sub.ref_w2.mean(), ls="--", c=GRAY, lw=1.4, label="real vs real (floor)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(d, color=NAVY, fontsize=18)
    ax.set_xlabel("NFE", color=NAVY)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("W2 to test set", color=NAVY)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=15)
fig.tight_layout(rect=(0, 0.1, 1, 1))
fig.savefig(OUT / "w2_all.png", dpi=100)
plt.close(fig)
print("saved", OUT / "w2_all.png")
