"""Part V 的图：从 2D 到 MNIST。

要完成的任务：把 code/e4_cifar/eval_images.py 的结果画成第 3、24–27 页要用的图。每张图的样子和用途：
    f8    image_xt.png    两个数字沿直线路径从噪声变清晰，最右一列是回归目标 x1 - x0（第 24 页）
    f9    mnist_grid.png  行是 I-CFM、OT-CFM、DDIM，列是 NFE 1, 2, 5, 10, 100，噪声相同（第 25 页）
    f9t   mnist_traj.png  OT-CFM 从 t = 0 到 t = 1，数字逐渐出现（第 25 页）
    real  mnist_real.png  真实数字和模糊的数据均值，作为对照（第 25 页）
    f11   fid_vs_nfe.png  FID 随 NFE 的曲线，三种方法、两个训练种子（实线种子 0，虚线种子 1），加 real-vs-real 虚线，对数坐标（第 26 页）
    e3    e3_kmeans.png   E3：左边是落在各簇的比例与真实比例的差距 TV 随 NFE 变化，两个种子；右边是 DDIM 两个种子在 NFE 100 时各数字的比例（第 28 页）
    loss  mnist_loss.png  I-CFM 与 OT-CFM 的训练 loss（第 27 页）
    flow  mnist_flow.gif  OT-CFM 从噪声生成数字的动图和末帧 png（第 3 页）

Figures for Part V of the talk: from 2D to MNIST.

    python slides/make_image_figs.py slides/figs          # everything that has its inputs
    python slides/make_image_figs.py slides/figs f8       # only one figure

Outputs
    image_xt.png        F8: two MNIST digits along the straight path, and the regression target
    mnist_grid.png      F9: the same noise at NFE 1, 2, 5, 10, 100 for I-CFM, OT-CFM and DDIM
    mnist_traj.png      F9: generation from t = 0 to t = 1 with OT-CFM
    mnist_real.png      real digits, and the mean digit, next to the samples
    fid_vs_nfe.png      F11: FID vs NFE on MNIST, two training seeds
    e3_kmeans.png       E3: cluster-share distance vs NFE, and DDIM's digit shares for both seeds
    mnist_loss.png      training loss of I-CFM and OT-CFM
    mnist_flow.gif/.png the opening animation: OT-CFM digits from noise (needs a GPU and results/e2/ckpt)

Inputs: results/e4/mnist_eval.json and mnist_samples.npz from code/e4_cifar/eval_images.py --data mnist,
results/e2/{icfm,otcfm}_s0_train.json from code/e2_mnist/train_mnist.py,
results/e3/fid_s1.json from code/e3_kmeans/fid_seed.py and results/e3/kmeans.json from code/e3_kmeans/cluster_coverage.py.
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
E2, E4 = ROOT / "results" / "e2", ROOT / "results" / "e4"

NAVY, BG, GRID, GRAY, ORANGE, BLUE = "#14213D", "#FAF8F3", "#D9DDE3", "#8A94A3", "#D9822B", "#2F6DB5"
DARK_ORANGE, MID = "#B35C12", "#5B6B82"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "text.color": NAVY, "axes.labelcolor": NAVY, "axes.edgecolor": GRID,
                     "xtick.color": NAVY, "ytick.color": NAVY})
DPI = 200
STYLE = {"otcfm": (ORANGE, "o", "OT-CFM"), "icfm": (DARK_ORANGE, "s", "I-CFM"), "ddpm": (BLUE, "^", "DDIM")}


def show(ax, img, cmap="gray", vmin=0, vmax=1):
    """img is CHW or HW, already in display range."""
    if img.ndim == 3:
        img = img[0]
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def make_xt():
    from torchvision import datasets
    data = datasets.MNIST(ROOT / "data", train=True, download=False).data
    rng = np.random.default_rng(0)
    ts = [0, 0.25, 0.5, 0.75, 1]
    titles = ["t = 0: x₀", "t = 0.25", "t = 0.5", "t = 0.75", "t = 1: x₁", "target x₁ − x₀"]
    fig, axes = plt.subplots(2, 6, figsize=(10.4, 3.9), dpi=DPI)
    for r, i in enumerate([7, 0]):
        x1 = data[i].numpy()[None] / 127.5 - 1
        x0 = rng.standard_normal(x1.shape)
        for c, t in enumerate(ts):
            show(axes[r, c], np.clip(((1 - t) * x0 + t * x1 + 1) / 2, 0, 1))
        target = x1 - x0
        m = np.abs(target).max()
        show(axes[r, 5], (target + m) / (2 * m))
    for c, title in enumerate(titles):
        axes[0, c].set_title(title, fontsize=17, pad=8)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.01, wspace=0.08, hspace=0.06)
    fig.savefig(OUT / "image_xt.png")
    plt.close(fig)
    print("saved image_xt.png")


def samples():
    return np.load(E4 / "mnist_samples.npz")


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


def make_grid():
    s = samples()
    rows = [(m, STYLE[m][2]) for m in ["icfm", "otcfm", "ddpm"] if f"{m}_100" in s]
    ks = [1, 2, 5, 10, 100]
    fig, axes = plt.subplots(len(rows), len(ks), figsize=(10.4, 2.1 * len(rows)), dpi=DPI, squeeze=False)
    for r, (m, name) in enumerate(rows):
        for c, k in enumerate(ks):
            show(axes[r, c], tile(s[f"{m}_{k}"][:16], 4))
            if r == 0:
                axes[r, c].set_title(f"NFE {k}", fontsize=18, pad=6)
        axes[r, 0].set_ylabel(name, fontsize=18, labelpad=8)
    fig.subplots_adjust(left=0.05, right=0.998, top=0.93, bottom=0.005, wspace=0.04, hspace=0.05)
    fig.savefig(OUT / "mnist_grid.png")
    plt.close(fig)
    print("saved mnist_grid.png")


def make_traj():
    traj = samples()["otcfm_traj"]  # frames, N, C, H, W
    fig, axes = plt.subplots(1, traj.shape[0], figsize=(10.4, 1.35), dpi=DPI)
    for c, (ax, t) in enumerate(zip(axes, ["t = 0", "t = 0.25", "t = 0.5", "t = 0.75", "t = 1"])):
        show(ax, tile(traj[c][:8], 4))
        ax.set_title(t, fontsize=17, pad=4)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.76, bottom=0.01, wspace=0.05)
    fig.savefig(OUT / "mnist_traj.png")
    plt.close(fig)
    print("saved mnist_traj.png")


def make_real():
    s = samples()
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.2), dpi=DPI, gridspec_kw={"width_ratios": [1, 1]})
    show(axes[0], tile(s["real"][:16], 4))
    axes[0].set_title("Real digits", fontsize=16, pad=4)
    show(axes[1], s["mean"][0] / 255)
    axes[1].set_title("Data mean", fontsize=16, pad=4)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.01, wspace=0.06)
    fig.savefig(OUT / "mnist_real.png")
    plt.close(fig)
    print("saved mnist_real.png")


SEED_LS = {0: "-", 1: (0, (5, 3))}  # 实线种子 0，虚线种子 1；颜色只表示方法


def seed_legend(ax, extra=(), loc="best", ncol=2):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], c=STYLE[m][0], marker=STYLE[m][1], lw=2, ms=8, label=STYLE[m][2]) for m in ["otcfm", "icfm", "ddpm"]]
    h += [Line2D([], [], c=MID, lw=2, ls=SEED_LS[0], label="seed 0"),
          Line2D([], [], c=MID, lw=2, ls=SEED_LS[1], marker="o", mfc=BG, ms=7, label="seed 1")]
    ax.legend(handles=h + list(extra), frameon=False, fontsize=14, ncol=ncol, columnspacing=1.0, handlelength=2.2, loc=loc)


def make_fid():
    from matplotlib.lines import Line2D
    r = json.loads((E4 / "mnist_eval.json").read_text())
    s1 = json.loads((ROOT / "results" / "e3" / "fid_s1.json").read_text())["fid"]
    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=DPI)
    for m, rec in r["by_method"].items():
        col, mk, lab = STYLE[m]
        for seed, fid in [(0, rec["fid"]), (1, s1[m])]:
            ks = [k for k in fid if k != "dopri5"]
            ax.plot([int(k) for k in ks], [fid[k] for k in ks], c=col, marker=mk, lw=2, ms=8, ls=SEED_LS[seed],
                    mfc=col if seed == 0 else BG, mew=2)
    ax.axhline(r["fid_real_vs_real"], c=GRAY, lw=1.5, ls=(0, (1, 2)))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel("NFE")
    ax.set_ylabel("FID (log scale)")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    seed_legend(ax, [Line2D([], [], c=GRAY, lw=1.5, ls=(0, (1, 2)), label="real vs real")])
    fig.tight_layout()
    fig.savefig(OUT / "fid_vs_nfe.png")
    plt.close(fig)
    print("saved fid_vs_nfe.png")


def make_loss():
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=DPI)
    for m in ["icfm", "otcfm"]:
        col, _, lab = STYLE[m]
        loss = json.loads((E2 / f"{m}_s0_train.json").read_text())["loss"]
        ax.plot(range(1, len(loss) + 1), loss, c=col, lw=2, label=lab)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss per pixel")
    ax.set_ylim(bottom=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, fontsize=16)
    fig.tight_layout()
    fig.savefig(OUT / "mnist_loss.png")
    plt.close(fig)
    print("saved mnist_loss.png")


def make_e3():
    """E3. 左：TV（k = 10）随 NFE，两个种子，NFE 100 是三次换噪声的平均和范围；灰色虚线是真实测试集。
    右：DDIM 两个种子在 NFE 100 时各数字的比例（SVM 打的标签，三次平均），灰色是真实比例。NFE 1、2 不画：
    模糊的图靠近簇中心，TV 会显得太好。"""
    r = json.loads((ROOT / "results" / "e3" / "kmeans.json").read_text())
    R = r["runs"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=DPI, gridspec_kw={"width_ratios": [1, 1.15]})
    ks = [5, 20, 100]
    for m in ["otcfm", "icfm", "ddpm"]:
        col, mk, _ = STYLE[m]
        for seed in (0, 1):
            d100 = [R[f"{m}_s{seed}_nfe100_noise{n}"]["k10"]["tv"] for n in (7, 11, 12)]
            ys = [R[f"{m}_s{seed}_nfe{k}_noise7"]["k10"]["tv"] for k in ks[:-1]] + [np.mean(d100)]
            a.plot(ks, ys, c=col, marker=mk, lw=2, ms=8, ls=SEED_LS[seed], mfc=col if seed == 0 else BG, mew=2)
            a.plot([100, 100], [min(d100), max(d100)], c=col, lw=2)
    base = R["real_test"]["k10"]["tv"]
    from matplotlib.lines import Line2D
    a.axhline(base, c=GRAY, lw=1.5, ls=(0, (1, 2)))
    a.set_xscale("log"); a.set_xticks(ks); a.set_xticklabels([str(k) for k in ks]); a.minorticks_off()
    a.set_xlim(4, 125); a.set_ylim(0, 0.46)
    a.set_xlabel("NFE")
    a.set_ylabel("TV distance to real shares")
    a.set_title("Cluster shares vs. real (k-means, k = 10)", fontsize=17, loc="left", color=NAVY)
    seed_legend(a, [Line2D([], [], c=GRAY, lw=1.5, ls=(0, (1, 2)), label="real test set")], loc="upper right", ncol=1)

    ref = np.array(r["ref_digit_share"]) * 100
    sh = {s: np.mean([R[f"ddpm_s{s}_nfe100_noise{n}"]["digit_share"] for n in (7, 11, 12)], 0) * 100 for s in (0, 1)}
    x = np.arange(10)
    wbar = 0.27
    b.bar(x - wbar, ref, wbar - 0.03, color=GRAY, label="real", zorder=2)
    b.bar(x, sh[0], wbar - 0.03, color=BLUE, label="DDIM, seed 0", zorder=2)
    b.bar(x + wbar, sh[1], wbar - 0.03, color=BG, edgecolor=BLUE, lw=1.6, hatch="///", label="DDIM, seed 1", zorder=2)
    b.set_xticks(x); b.set_xticklabels([str(i) for i in x])
    b.set_xlabel("digit (SVM label)")
    b.set_ylabel("share at NFE 100 (%)")
    b.set_title("Which digits: DDIM, two training runs", fontsize=17, loc="left", color=NAVY)
    b.legend(frameon=False, fontsize=14, ncol=3, loc="upper left")
    b.set_ylim(0, 24)
    for ax in (a, b):
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    fig.tight_layout(w_pad=2.5)
    fig.savefig(OUT / "e3_kmeans.png")
    plt.close(fig)
    print("saved e3_kmeans.png")


def make_flow():
    """OT-CFM digits from noise, Euler with 100 steps, for the opening slide. Saves the last frame as .png for the PDF."""
    import torch
    from PIL import Image
    from torchcfm.models.unet import UNetModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(device)
    net.load_state_dict(torch.load(E2 / "ckpt" / "otcfm_s0.pt", map_location=device, weights_only=True))
    net.eval()
    x = torch.randn(36, 1, 28, 28, generator=torch.Generator().manual_seed(3)).to(device)

    def frame(x):
        img = (x * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu().numpy()
        arr = (tile(img, 6)[0] * 255).astype(np.uint8)
        return Image.fromarray(arr).resize((arr.shape[1] * 3, arr.shape[0] * 3), Image.NEAREST)

    frames = []
    with torch.no_grad():
        for i in range(100):
            if i % 2 == 0:
                frames.append(frame(x))
            x = x + net(torch.full((36,), i / 100, device=device), x) / 100
    last = frame(x)
    frames += [last] * 25  # rest on the result before the loop restarts
    frames[0].save(OUT / "mnist_flow.gif", save_all=True, append_images=frames[1:], duration=60, loop=0, optimize=True)
    last.save(OUT / "mnist_flow.png")
    print("saved mnist_flow.gif, mnist_flow.png")


FIGS = {"f8": make_xt, "f9": make_grid, "f9t": make_traj, "real": make_real, "f11": make_fid, "loss": make_loss, "e3": make_e3, "flow": make_flow}
if __name__ == "__main__":
    for key, fn in FIGS.items():
        if ONLY is None or ONLY == key:
            fn()
