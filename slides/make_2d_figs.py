"""Figures for Part III of the talk: why independent pairs give curved paths, and what OT pairs change.

The models are the ones trained in Flow_matching_tutorial.ipynb (8 Gaussians -> two moons).
The running example from Part II continues: x0 ~ N(0, 1), x1 = +-2, followed at the point (0.5, 0.1).

    python slides/make_2d_figs.py slides/figs

Outputs
    icfm_demo.gif / .png  I-CFM: 8 Gaussians -> moons, paths drawn as they grow (the PNG is the last frame, for the PDF)
    bend_1d.png           running example: lines cross, marginal paths bend, one Euler step lands on the mean
    pairs_swap.png        running example before / after the swap; a 2D batch with random and OT pairs
    traj_ot_vs_i.png      I-CFM and OT-CFM trajectories from the same starting points
    w2_four.png           W2 vs NFE on four datasets (from results/e1/summary.csv)
    moons_5steps.png      top row of datasets_grid.png, without its labels
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "slides" / "figs")
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchcfm.models.models import MLP
from torchcfm.optimal_transport import OTPlanSampler
from torchcfm.utils import sample_8gaussians, sample_moons

NAVY, BG, GRID, GRAY, ORANGE, BLUE = "#14213D", "#FAF8F3", "#D9DDE3", "#8A94A3", "#D9822B", "#2F6DB5"
DARK_ORANGE, MID = "#B35C12", "#5B6B82"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18, "figure.facecolor": BG, "axes.facecolor": BG,
                     "savefig.facecolor": BG, "text.color": NAVY, "axes.labelcolor": NAVY, "axes.edgecolor": GRID,
                     "xtick.color": NAVY, "ytick.color": NAVY})
A, DPI = 2.0, 200
T0, X0 = 0.5, 0.1
MODELS = ROOT / "conditional-flow-matching-main/examples/2D_tutorials"


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def velocity(x, t):
    return (A * np.tanh(A * t * x / (1 - t) ** 2) - x) / (1 - t)


def integrate(x0, steps=2000, stop=0.999):
    """Exact marginal velocity of the running example, integrated with Euler."""
    ts = np.linspace(0, stop, steps + 1)
    xs = [np.asarray(x0, dtype=float)]
    for k in range(steps):
        xs.append(xs[-1] + (ts[k + 1] - ts[k]) * velocity(xs[-1], ts[k]))
    return ts, np.stack(xs)


def trajectories(name, n, steps, seed=0):
    net = MLP(dim=2, time_varying=True)
    net.load_state_dict(torch.load(MODELS / f"{name}.pt", map_location="cpu"))
    net.eval()
    torch.manual_seed(seed)
    x = sample_8gaussians(n)
    traj = [x.numpy()]
    with torch.no_grad():
        for k in range(steps):
            t = torch.full((n, 1), k / steps)
            x = x + net(torch.cat([x, t], dim=-1)) / steps
            traj.append(x.numpy())
    return np.stack(traj)


def path_stats(traj):
    """Mean path length / mean straight-line distance, and the straightness used in code/common/metrics_2d.py."""
    length = np.linalg.norm(traj[1:] - traj[:-1], axis=-1).sum(0)
    chord = np.linalg.norm(traj[-1] - traj[0], axis=-1)
    vel = (traj[1:] - traj[:-1]) * (len(traj) - 1)
    return length.sum() / chord.sum(), ((vel - (traj[-1] - traj[0])[None]) ** 2).sum(-1).mean()


def draw_2d(ax, traj, k, n_paths=300, highlight=None):
    ax.plot(traj[: k + 1, :n_paths, 0], traj[: k + 1, :n_paths, 1], c=ORANGE, lw=0.5, alpha=0.18)
    if highlight is not None:
        ax.plot(traj[: k + 1, highlight, 0], traj[: k + 1, highlight, 1], c=NAVY, lw=1.6)
    ax.scatter(traj[0, :, 0], traj[0, :, 1], s=5, c=GRAY, alpha=0.6)
    ax.scatter(traj[k, :, 0], traj[k, :, 1], s=7, c=ORANGE)
    ax.set_xlim(-8, 8); ax.set_ylim(-7, 7); ax.set_aspect("equal"); ax.axis("off")


# ---- slide 15: I-CFM demo ----------------------------------------------------
def make_demo():
    steps, n = 60, 1024
    traj = trajectories("moon_classical", n, steps)
    highlight = np.arange(0, 96, 8)

    def frame(k):
        fig, ax = plt.subplots(figsize=(8.6, 7.0), dpi=100)
        draw_2d(ax, traj, k, highlight=highlight)
        ax.text(0.02, 0.97, f"t = {k / steps:.2f}", transform=ax.transAxes, va="top", fontsize=18)
        ax.scatter([], [], s=40, c=GRAY, label="source: 8 Gaussians")
        ax.scatter([], [], s=40, c=ORANGE, label="current samples")
        ax.plot([], [], c=NAVY, lw=1.6, label="12 of the paths")
        ax.legend(loc="lower center", ncol=3, frameon=False, fontsize=15, columnspacing=1.2, handletextpad=0.3)
        fig.tight_layout(pad=0.2)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

    frames = [frame(k) for k in range(steps + 1)]
    frames[-1].save(OUT / "icfm_demo.png")
    frames = frames + [frames[-1]] * 25
    pal = frames[-1].quantize(colors=64, method=Image.MEDIANCUT)
    frames[0].quantize(palette=pal).save(OUT / "icfm_demo.gif", save_all=True, duration=80, loop=0, optimize=True,
                                         append_images=[f.quantize(palette=pal) for f in frames[1:]])


# ---- slide 16: running example -----------------------------------------------
def make_bend():
    fig, axes = plt.subplots(1, 3, figsize=(16.64, 4.0), dpi=DPI, sharey=True)
    for ax in axes:
        clean(ax)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-3.4, 3.4); ax.set_xlabel("t")
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([-2, 0, 2])
        ax.scatter([1, 1], [-A, A], s=70, c=NAVY, zorder=5, clip_on=False)
    axes[0].set_ylabel("x")
    # a) the two lines through the point
    ax = axes[0]
    rng = np.random.default_rng(3)
    x0, x1 = rng.standard_normal(16), rng.choice([-A, A], 16)
    for a, b in zip(x0, x1):
        ax.plot([0, 1], [a, b], c=GRAY, lw=0.8, alpha=0.5)
    xp, xm = (X0 - T0 * A) / (1 - T0), (X0 + T0 * A) / (1 - T0)
    ax.plot([0, 1], [xp, A], c=BLUE, lw=2.6)
    ax.plot([0, 1], [xm, -A], c=DARK_ORANGE, lw=2.6)
    ax.scatter([T0], [X0], s=90, c=NAVY, zorder=6)
    ax.annotate("velocity +3.8", (0.76, 1.05), (0.33, 2.75), color=BLUE, fontsize=17, ha="left")
    ax.annotate("velocity −4.2", (0.76, -1.05), (0.33, -3.1), color=DARK_ORANGE, fontsize=17, ha="left")
    ax.set_title("Lines cross, so the network averages", fontsize=19, loc="left")
    # b) the marginal paths bend
    ax = axes[1]
    ts, xs = integrate(np.linspace(-2.4, 2.4, 25))
    ax.plot(ts, xs, c=GRAY, lw=0.8, alpha=0.5)
    ts, x = integrate(np.array([START]))
    ax.plot([0, 1], [START, A], c=NAVY, lw=1.8, ls=(0, (4, 3)))
    ax.plot(ts, x[:, 0], c=DARK_ORANGE, lw=3)
    ax.scatter([0], [START], s=70, c=DARK_ORANGE, zorder=6, clip_on=False)
    ax.text(0.03, 0.05 + START - 1.35, f"x₀ = {START:g}: starts with\nvelocity −{START:g}, away from +2", fontsize=15, color=DARK_ORANGE, va="top")
    ax.annotate("straight line", (0.32, START + 0.32 * (A - START)), (0.16, 2.72), fontsize=15, color=NAVY,
                arrowprops=dict(arrowstyle="-", color=NAVY, lw=1))
    ax.set_title("So the marginal paths bend", fontsize=19, loc="left")
    # c) one Euler step
    ax = axes[2]
    for a in np.linspace(-2.4, 2.4, 13):
        ax.plot([0, 1], [a, 0], c=GRAY, lw=1.1, alpha=0.8)
    ax.scatter([1], [0], s=110, c=DARK_ORANGE, zorder=6, clip_on=False)
    ax.annotate("all land on E[x₁] = 0,\nwhere there is no data", (1, 0), (0.30, -3.25), color=DARK_ORANGE, fontsize=16, ha="left")
    ax.text(0.955, 2, "data", fontsize=15, ha="right", va="center")
    ax.text(0.955, -2, "data", fontsize=15, ha="right", va="center")
    ax.set_title("One Euler step from t = 0", fontsize=19, loc="left")
    fig.tight_layout(pad=0.4, w_pad=2.0)
    fig.savefig(OUT / "bend_1d.png")
    plt.close(fig)


# ---- slide 17: swap and OT pairs ---------------------------------------------
def make_pairs():
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.5), dpi=DPI)
    xp, xm = (X0 - T0 * A) / (1 - T0), (X0 + T0 * A) / (1 - T0)
    for ax, title, lines in [(axes[0, 0], "Crossing pair: cost 32.08", [(xp, A, BLUE), (xm, -A, DARK_ORANGE)]),
                             (axes[0, 1], "After the swap: cost 0.08", [(xp, -A, BLUE), (xm, A, DARK_ORANGE)])]:
        clean(ax)
        for a, b, c in lines:
            ax.plot([0, 1], [a, b], c=c, lw=2.6)
            ax.scatter([0, 1], [a, b], s=45, c=c, zorder=5, clip_on=False)
        ax.set_xlim(0, 1); ax.set_ylim(-3, 3); ax.set_xticks([0, 1]); ax.set_yticks([-2, 0, 2])
        ax.set_xlabel("t", labelpad=-12)
        ax.set_title(title, fontsize=17, loc="left")
    axes[0, 0].scatter([T0], [X0], s=70, c=NAVY, zorder=6)
    axes[0, 1].set_yticklabels([])

    torch.manual_seed(2); np.random.seed(2)
    x0, x1 = sample_8gaussians(16).numpy(), sample_moons(16).numpy()
    plan = OTPlanSampler(method="exact").get_map(torch.tensor(x0), torch.tensor(x1))
    match = plan.argmax(1)
    assert len(set(match)) == 16
    rand_cost = ((x0 - x1) ** 2).sum(1).mean()
    ot_cost = ((x0 - x1[match]) ** 2).sum(1).mean()
    print(f"2D batch: mean squared length random {rand_cost:.1f}, OT {ot_cost:.1f}")
    for ax, y, title in [(axes[1, 0], x1, f"Random: mean ‖x₁−x₀‖² = {rand_cost:.0f}"),
                         (axes[1, 1], x1[match], f"OT: mean ‖x₁−x₀‖² = {ot_cost:.0f}")]:
        for a, b in zip(x0, y):
            ax.plot([a[0], b[0]], [a[1], b[1]], c=GRAY, lw=1.3)
        ax.scatter(x0[:, 0], x0[:, 1], s=42, c=GRAY, zorder=5)
        ax.scatter(x1[:, 0], x1[:, 1], s=42, c=ORANGE, zorder=5)
        ax.set_xlim(-7, 7); ax.set_ylim(-6, 6); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=16, loc="left")
    fig.tight_layout(pad=0.4, h_pad=1.0, w_pad=1.2)
    fig.savefig(OUT / "pairs_swap.png")
    plt.close(fig)


# ---- slide 18: trajectories side by side -------------------------------------
def make_traj():
    steps, n = 200, 1024
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.7), dpi=DPI)
    highlight = np.arange(0, 96, 8)
    for ax, name, label in [(axes[0], "moon_classical", "I-CFM"), (axes[1], "moon_optimal", "OT-CFM")]:
        traj = trajectories(name, n, steps)
        draw_2d(ax, traj, steps, highlight=highlight)
        ax.text(0.01, 0.98, label, transform=ax.transAxes, va="top", fontsize=20)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=7, c=GRAY), plt.Line2D([], [], ls="", marker="o", ms=7, c=ORANGE),
               plt.Line2D([], [], c=NAVY, lw=1.6)]
    fig.legend(handles, ["source: 8 Gaussians", "generated", "the same 12 starting points"], loc="lower center", ncol=3,
               frameon=False, fontsize=15, columnspacing=1.5, handletextpad=0.4)
    fig.tight_layout(pad=0.2, w_pad=0.5, rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "traj_ot_vs_i.png")
    plt.close(fig)
    for name in ["moon_classical", "moon_optimal", "moon_sch"]:
        ratio, straight = path_stats(trajectories(name, 2000, steps))
        print(f"{name}: length / straight line = {ratio:.3f}, straightness = {straight:.3f}")


# ---- slide 19: W2 vs NFE -----------------------------------------------------
def make_w2():
    df = pd.read_csv(ROOT / "results/e1/summary.csv")
    series = [("otcfm", "OT-CFM", ORANGE, "o"), ("icfm", "I-CFM", DARK_ORANGE, "s"), ("ddim", "DDIM", BLUE, "^"),
              ("ddpm", "DDPM, 1000 steps", NAVY, "D")]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.5), dpi=DPI, sharex=True, sharey=True)
    for ax, data in zip(axes.flat, ["moons", "8gaussians", "checkerboard", "swissroll"]):
        clean(ax)
        sub = df[df.data == data]
        ax.axhline(sub.ref_w2.mean(), ls=(0, (4, 3)), c=GRAY, lw=1.3)
        for key, label, color, marker in series:
            r = sub[sub.method == key].sort_values("nfe")
            ax.plot(r.nfe, r.w2_mean, c=color, lw=2, marker=marker, ms=8, mec=BG, mew=1, label=label)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.035, 20)
        ax.set_xticks([1, 10, 100, 1000]); ax.set_xticklabels(["1", "10", "100", "1000"])
        ax.set_yticks([0.1, 1, 10]); ax.set_yticklabels(["0.1", "1", "10"])
        ax.minorticks_off(); ax.grid(True, axis="y", c=GRID, lw=0.8)
        ax.set_title({"8gaussians": "8 Gaussians", "moons": "Moons", "checkerboard": "Checkerboard",
                      "swissroll": "Swiss roll"}[data], fontsize=18, loc="left")
    for ax in axes[1]:
        ax.set_xlabel("NFE (solver steps)")
    for ax in axes[:, 0]:
        ax.set_ylabel("W2 to test set")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], ls=(0, (4, 3)), c=GRAY, lw=1.3)); labels.append("real vs. real")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=15, columnspacing=1.2)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "w2_four.png")
    plt.close(fig)


def make_moons_crop():
    Image.open(OUT / "datasets_grid.png").convert("RGB").crop((46, 50, 1484, 330)).save(OUT / "moons_5steps.png")


def toy_numbers():
    """Exact W2 of the running example under Euler with K steps, from a symmetric quantile grid."""
    from scipy.stats import norm
    q = (np.arange(200000) + 0.5) / 200000
    z = norm.ppf(q)
    for K in [1, 2, 5, 10, 20]:
        x = z.copy()
        for k in range(K):
            x = x + velocity(x, k / K) / K
        x = np.sort(x)
        y = np.where(q < 0.5, -A, A)
        print(f"running example, I-CFM field, K = {K}: W2 = {np.sqrt(np.mean((x - y) ** 2)):.3f}")
    for start in [0.5, 1.0, 1.5]:
        ts, xs = integrate(np.array([start]), steps=20000)
        i = xs[:, 0].argmin()
        print(f"x0 = {start}: minimum {xs[i, 0]:.3f} at t = {ts[i]:.3f}, x(0.5) = {xs[10000, 0]:.3f}, end {xs[-1, 0]:.3f}")


START = 1.0
if __name__ == "__main__":
    toy_numbers()
    make_demo()
    make_bend()
    make_pairs()
    make_traj()
    make_w2()
    make_moons_crop()
