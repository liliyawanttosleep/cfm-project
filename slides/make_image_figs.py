"""Figures for Part V of the talk: from 2D to images.

    python slides/make_image_figs.py slides/figs          # everything that has its inputs
    python slides/make_image_figs.py slides/figs f8       # only one figure

Outputs
    image_xt.png      F8: one MNIST digit and one CIFAR-10 image along the straight path, and the regression target
    mnist_grid.png    F9: the same noise at NFE 1, 2, 5, 10, 100 for I-CFM, OT-CFM and DDIM
    mnist_traj.png    F9: generation from t = 0 to t = 1
    cifar_grid.png    F10: the same noise at NFE 2, 5, 10, 20, 100 and dopri5 for I-CFM and OT-CFM
    fid_vs_nfe.png    F11: FID vs NFE on MNIST and CIFAR-10
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "slides" / "figs")
ONLY = sys.argv[2] if len(sys.argv) > 2 else None
E4 = ROOT / "results" / "e4"

NAVY, BG, GRID, GRAY, ORANGE, BLUE = "#14213D", "#FAF8F3", "#D9DDE3", "#8A94A3", "#D9822B", "#2F6DB5"
DARK_ORANGE, MID = "#B35C12", "#5B6B82"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "text.color": NAVY, "axes.labelcolor": NAVY, "axes.edgecolor": GRID,
                     "xtick.color": NAVY, "ytick.color": NAVY})
DPI = 200


def show(ax, img, cmap=None, vmin=None, vmax=None):
    """img is CHW or HW, already in display range."""
    if img.ndim == 3:
        img = img[0] if img.shape[0] == 1 else img.transpose(1, 2, 0)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def make_xt():
    from torchvision import datasets
    mnist = datasets.MNIST(ROOT / "data", train=True, download=False).data[7].numpy()[None] / 127.5 - 1
    cifar = datasets.CIFAR10(ROOT / "conditional-flow-matching-main/examples/images/cifar10/data", train=True, download=False).data[7].transpose(2, 0, 1) / 127.5 - 1
    rng = np.random.default_rng(0)
    ts = [0, 0.25, 0.5, 0.75, 1]
    titles = ["t = 0: x₀", "t = 0.25", "t = 0.5", "t = 0.75", "t = 1: x₁", "target x₁ − x₀"]
    fig, axes = plt.subplots(2, 6, figsize=(10.4, 3.9), dpi=DPI)
    for r, x1 in enumerate([mnist, cifar]):
        x0 = rng.standard_normal(x1.shape)
        for c, t in enumerate(ts):
            xt = (1 - t) * x0 + t * x1
            img = np.clip((xt + 1) / 2, 0, 1)
            show(axes[r, c], img, cmap="gray" if x1.shape[0] == 1 else None, vmin=0, vmax=1)
        target = x1 - x0
        m = np.abs(target).max()
        img = (target + m) / (2 * m)
        show(axes[r, 5], img, cmap="gray" if x1.shape[0] == 1 else None, vmin=0, vmax=1)
    for c, title in enumerate(titles):
        axes[0, c].set_title(title, fontsize=17, pad=8)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.01, wspace=0.08, hspace=0.06)
    fig.savefig(OUT / "image_xt.png")
    plt.close(fig)
    print("saved image_xt.png")


def samples(data):
    return np.load(E4 / f"{data}_samples.npz")


def tile(batch, cols):
    """batch: N C H W uint8 -> one image with small gaps."""
    n, c, h, w = batch.shape
    rows = int(np.ceil(n / cols))
    gap = 1
    out = np.full((c, rows * h + (rows - 1) * gap, cols * w + (cols - 1) * gap), 255, dtype=np.uint8)
    for i in range(n):
        y, x = (i // cols) * (h + gap), (i % cols) * (w + gap)
        out[:, y:y + h, x:x + w] = batch[i]
    return out / 255


def grid_fig(data, rows, cols_keys, labels, n, tile_cols, name, figsize, cmap=None):
    s = samples(data)
    fig, axes = plt.subplots(len(rows), len(cols_keys), figsize=figsize, dpi=DPI, squeeze=False)
    for r, (m, rname) in enumerate(rows):
        for c, k in enumerate(cols_keys):
            key = f"{m}_{k}"
            ax = axes[r, c]
            if key in s:
                show(ax, tile(s[key][:n], tile_cols), cmap=cmap, vmin=0, vmax=1)
            else:
                ax.axis("off")
            if r == 0:
                ax.set_title(labels[c], fontsize=18, pad=6)
        axes[r, 0].set_ylabel(rname, fontsize=18, labelpad=8)
    fig.subplots_adjust(left=0.045, right=0.998, top=0.91, bottom=0.005, wspace=0.04, hspace=0.05)
    fig.savefig(OUT / name)
    plt.close(fig)
    print("saved", name)


def make_mnist():
    grid_fig("mnist", [("icfm", "I-CFM"), ("otcfm", "OT-CFM"), ("ddpm", "DDIM")], [1, 2, 5, 10, 100],
             ["NFE 1", "NFE 2", "NFE 5", "NFE 10", "NFE 100"], 16, 4, "mnist_grid.png", (10.4, 6.3), cmap="gray")


def make_cifar():
    grid_fig("cifar", [("icfm", "I-CFM"), ("otcfm", "OT-CFM")], [2, 5, 10, 20, 100, "dopri5"],
             ["NFE 2", "NFE 5", "NFE 10", "NFE 20", "NFE 100", "dopri5"], 9, 3, "cifar_grid.png", (10.4, 3.7))


def make_traj():
    s = samples("mnist")
    traj = s["otcfm_traj"]  # frames, N, C, H, W
    fig, axes = plt.subplots(1, traj.shape[0], figsize=(10.4, 2.0), dpi=DPI)
    for c, (ax, t) in enumerate(zip(axes, ["t = 0", "t = 0.25", "t = 0.5", "t = 0.75", "t = 1"])):
        show(ax, tile(traj[c][:8], 4), cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=17, pad=4)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.82, bottom=0.01, wspace=0.05)
    fig.savefig(OUT / "mnist_traj.png")
    plt.close(fig)
    print("saved mnist_traj.png")


def make_fid():
    style = {"otcfm": (ORANGE, "o", "OT-CFM"), "icfm": (DARK_ORANGE, "s", "I-CFM"), "ddpm": (BLUE, "^", "DDIM")}
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), dpi=DPI)
    for ax, data, title in zip(axes, ["mnist", "cifar"], ["MNIST", "CIFAR-10"]):
        r = json.loads((E4 / f"{data}_eval.json").read_text())
        for m, rec in r["by_method"].items():
            col, mk, lab = style[m]
            ks = [k for k in rec["fid"] if k != "dopri5"]
            ax.plot([int(k) for k in ks], [rec["fid"][k] for k in ks], c=col, marker=mk, lw=2, ms=8, label=lab)
        ax.axhline(r["fid_real_vs_real"], c=GRAY, lw=1.5, ls=(0, (4, 3)))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
        ax.minorticks_off()
        ax.set_xlabel("NFE")
        ax.set_title(title, fontsize=19, loc="left")
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.8)
    axes[0].set_ylabel("FID (log scale)")
    axes[0].legend(frameon=False, fontsize=17)
    fig.tight_layout(w_pad=2)
    fig.savefig(OUT / "fid_vs_nfe.png")
    plt.close(fig)
    print("saved fid_vs_nfe.png")


FIGS = {"f8": make_xt, "f9": make_mnist, "f9t": make_traj, "f10": make_cifar, "f11": make_fid}
if __name__ == "__main__":
    for key, fn in FIGS.items():
        if ONLY is None or ONLY == key:
            fn()
