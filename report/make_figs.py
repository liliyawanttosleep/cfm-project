"""Figures for the final report (vector PDF, CPU only). Reads results/ and writes report/figs/.

    python report/make_figs.py

Figures
    fig_c1_reliability.pdf  FID and TV against NFE for the three training seeds of I-CFM and OT-CFM (C1).
    fig_c2_solvers.pdf      (a) curvature kappa(t) and the sqrt-kappa grid; (b) FID against NFE for Euler / Heun / RK4
                            on the uniform and sqrt-kappa grids, and with the last step kept off t = 1 (C2, OT-CFM seed 0).
    fig_samples.pdf         MNIST samples from the same 8 noise vectors at 1 / 2 / 5 / 100 Euler steps for I-CFM, OT-CFM,
                            DDIM and (when trained) the reflow models; the reflow grids are generated here on the CPU.
    fig_2d_paths.pdf        2D moons: 100-step trajectories of I-CFM and OT-CFM and where one Euler step lands (appendix).
    fig_frontier.pdf        FID against NFE for every model and sampler that has results, plus the lower envelope (C6).
                            Models without results yet are skipped, so the script runs at every stage of the project.
Colours follow the talk: OT-CFM orange, I-CFM dark orange, DDIM blue; solvers use navy / teal / purple.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report" / "figs"
ORANGE, DARK_ORANGE, BLUE, NAVY, GRAY = "#D9822B", "#B35C12", "#2F6DB5", "#14213D", "#8A94A3"
TEAL, PURPLE, GREEN = "#2A9D8F", "#7B4FA0", "#3D9A5B"
MCOL = {"icfm": DARK_ORANGE, "otcfm": ORANGE, "ddpm": BLUE}
NAMES = {"icfm": "I-CFM", "otcfm": "OT-CFM", "ddpm": "DDIM"}
SCOL = {"euler": NAVY, "heun": TEAL, "rk4": PURPLE}
MMARK = {"icfm": "s", "otcfm": "o", "ddpm": "^"}  # the two oranges are close, so markers also separate the methods
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "lines.linewidth": 1.2})


def load(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None


def seed_fids():
    e4 = load("e4/mnist_eval.json")["by_method"]
    out = {m: {0: e4[m]["fid"]} for m in ("icfm", "otcfm", "ddpm")}
    for s in (1, 2):
        r = load(f"e3/fid_s{s}.json")
        if r:
            for m in out:
                if m in r["fid"]:
                    out[m][s] = r["fid"][m]
    return out


def log_axes(ax, xt, yt=None):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(xt, [str(v) for v in xt])
    ax.minorticks_off()
    if yt:
        ax.set_yticks(yt, [f"{v:g}" for v in yt])
    ax.grid(alpha=0.25, lw=0.5)


def fig_c1():
    fid = seed_fids()
    km = load("c1_reliability/kmeans_c1.json")
    ks = [1, 2, 5, 10, 20, 50, 100]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.0))
    ls = {0: "-", 1: "--", 2: ":"}
    for m in ("icfm", "otcfm"):
        for s, f in fid[m].items():
            axes[0].plot(ks, [f[str(k)] for k in ks], ls[s], marker=MMARK[m], ms=2.5, color=MCOL[m], label=f"{NAMES[m]}, seed {s}")
    log_axes(axes[0], ks, [10, 20, 50, 100, 200, 300])
    axes[0].set(xlabel="NFE (Euler steps)", ylabel="FID (10k samples), lower is better", title="(a) FID across training seeds")
    if km:
        kt = [1, 2, 5, 20, 100]
        for m in ("icfm", "otcfm"):
            for s in (0, 1, 2):
                v = [km["runs"].get(f"{m}_s{s}_nfe{k}_noise7", {}).get("k10", {}).get("tv") for k in kt]
                if None not in v:
                    axes[1].plot(kt, v, ls[s], marker=MMARK[m], ms=2.5, color=MCOL[m])
        real = km["runs"]["real_test"]["k10"]["tv"]
        axes[1].axhline(real, color=GRAY, lw=0.8, ls="-.", label=f"real test set ({real:.3f})")
        log_axes(axes[1], kt, [0.02, 0.05, 0.1, 0.2, 0.5])
        axes[1].set(xlabel="NFE (Euler steps)", ylabel="TV to real cluster shares", title="(b) Class coverage (k = 10)")
    h, l = axes[0].get_legend_handles_labels()
    if km:
        h2, l2 = axes[1].get_legend_handles_labels()
        h, l = h + h2, l + l2
    fig.legend(h, l, frameon=False, ncol=3, loc="lower center", handlelength=2.6, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.24, 1, 1))
    fig.savefig(OUT / "fig_c1_reliability.pdf")
    plt.close(fig)


def fig_c2():
    d = load("c2_solvers/otcfm_s0.json")
    di = load("c2_solvers/icfm_s0.json")
    t1 = load("c2_solvers/t1_otcfm_s0.json")
    e4 = load("e4/mnist_eval.json")["by_method"]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 3.6), gridspec_kw={"width_ratios": [1, 1]})
    ax = axes[0]
    for m, dd in (("otcfm", d), ("icfm", di)):
        t, k = np.array(dd["kappa"]["t"]), np.array(dd["kappa"]["kappa"])
        ax.plot(t, k, color=MCOL[m], ls="-" if m == "otcfm" else "--", label=f"{NAMES[m]} (G = {dd['kappa']['G']:.2f})")
    # sqrt-kappa grid for 8 steps, OT-CFM
    t, k = np.array(d["kappa"]["t"]), np.array(d["kappa"]["kappa"])
    F = np.concatenate([[0], np.cumsum(0.5 * (np.sqrt(k[1:]) + np.sqrt(k[:-1])) * np.diff(t))])
    grid = np.interp(np.linspace(0, 1, 9), F / F[-1], t)
    for g in grid:
        ax.axvline(g, color=GRAY, lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.set(xlabel="t (0 = noise, 1 = data)", ylabel=r"$\kappa(t)=\mathbb{E}\,\|dv/dt\|$", title=r"(a) Curvature and 8-step $\sqrt{\kappa}$ grid")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    for s in ("euler", "heun", "rk4"):
        for g, lsty in (("uniform", "--"), ("sqrt_kappa", "-")):
            pts = sorted((int(n), r[g]) for n, r in d["fid"][s].items() if g in r)
            ax.plot([p[0] for p in pts], [p[1] for p in pts], lsty, marker="o", ms=2, color=SCOL[s],
                    label=f"{ {'euler': 'Euler', 'heun': 'Heun', 'rk4': 'RK4'}[s] }, {'uniform' if g == 'uniform' else '√κ grid'}")
    if t1:
        for s, g, c in (("heun_e", "uniform", TEAL), ("rk4_e", "uniform", PURPLE)):
            pts = sorted((int(k.split("|")[1]), v) for k, v in t1["fid"].items() if k.startswith(s + "|") and k.endswith("|" + g))
            ax.plot([p[0] for p in pts], [p[1] for p in pts], ":", marker="x", ms=3.5, color=c,
                    label=f"{'Heun' if s == 'heun_e' else 'RK4'}, uniform, last step off t = 1")
    ax.axhline(e4["otcfm"]["fid"]["dopri5"], color=GRAY, lw=0.8, ls="-.", label=f"dopri5 (~{e4['otcfm']['dopri5_mean_nfe']:.0f} NFE)")
    log_axes(ax, [2, 4, 8, 16, 32], [10, 20, 50, 100, 200, 400])
    ax.set(xlabel="NFE (network calls)", ylabel="FID (10k), lower is better", title="(b) OT-CFM seed 0: solver × grid")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=2, loc="lower center", fontsize=7.5, handlelength=2.6, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.33, 1, 1))
    fig.savefig(OUT / "fig_c2_solvers.pdf")
    plt.close(fig)


def fig_frontier():
    """Every sampler with results, seed 0. Lower envelope = best FID reachable at each NFE."""
    e4 = load("e4/mnist_eval.json")["by_method"]
    curves = {}
    for m in ("icfm", "otcfm", "ddpm"):
        ks = [1, 2, 5, 10, 20, 50, 100]
        curves[f"{NAMES[m]}, Euler" if m != "ddpm" else "DDIM"] = (ks, [e4[m]["fid"][str(k)] for k in ks], MCOL[m], "-")
    for m in ("icfm", "otcfm"):
        d = load(f"c2_solvers/{m}_s0.json")
        pts = sorted((int(n), r["sqrt_kappa"]) for n, r in d["fid"]["euler"].items() if "sqrt_kappa" in r)
        t1 = load(f"c2_solvers/t1_{m}_s0.json")
        if t1:
            pts = sorted((int(k.split("|")[1]), v) for k, v in t1["fid"].items() if k.startswith("heun_e|") and k.endswith("|sqrt_kappa"))
            curves[f"{NAMES[m]}, Heun*, √κ grid"] = ([p[0] for p in pts], [p[1] for p in pts], MCOL[m], ":")
    # Training variants without a detectable gain (OT batch 512 / 2048, κ-weighted t) are left to Table 2 to keep the figure readable.
    for tag, lab, col in (("_reflow", "reflow", GREEN), ("_tlogitnormal", "logit-normal t", PURPLE)):
        for m in ("icfm", "otcfm"):
            r = load(f"e3/fid_s0{tag}.json")
            if r and m in r["fid"]:
                ks = [int(k) for k in r["fid"][m] if k.isdigit()]
                curves[f"{NAMES[m]} {lab}, Euler"] = (sorted(ks), [r["fid"][m][str(k)] for k in sorted(ks)], col, "-." if m == "otcfm" else "-")
    for m in ("icfm", "otcfm"):  # C6: the √κ grid on the reflow models
        d = load(f"c2_solvers/{m}_s0_reflow.json")
        if d:
            pts = sorted((int(n), r["sqrt_kappa"]) for n, r in d["fid"]["euler"].items() if "sqrt_kappa" in r)
            curves[f"{NAMES[m]} reflow, Euler, √κ grid"] = ([p[0] for p in pts], [p[1] for p in pts], GREEN, ":")
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    allpts = []
    for lab, (x, y, c, lsty) in curves.items():
        m = "icfm" if lab.startswith("I-CFM") else "otcfm" if lab.startswith("OT-CFM") else "ddpm"
        ax.plot(x, y, lsty, marker=MMARK[m], ms=2.2, color=c, label=lab, lw=1)
        allpts += list(zip(x, y))
    xs = sorted({p[0] for p in allpts})
    env, best = [], np.inf
    for x in xs:
        best = min(best, min(y for xx, y in allpts if xx == x))
        env.append(best)
    ax.plot(xs, env, color="k", lw=2.2, alpha=0.18, label="lower envelope")
    log_axes(ax, [1, 2, 4, 8, 16, 32, 100], [10, 20, 50, 100, 200, 300])
    ax.set(xlabel="NFE (network calls)", ylabel="FID (10k), lower is better", title="Quality–efficiency frontier on MNIST (seed 0)")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=7.5, ncol=1, loc="center right", handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 0.64, 1))
    fig.savefig(OUT / "fig_frontier.pdf")
    plt.close(fig)


def reflow_grids(m, ks, n=8):
    """CPU sampling of a reflow model from the fixed noise used by eval_images.py (seed 2), so the grids match E4."""
    import sys
    import torch
    from torchcfm.models.unet import UNetModel
    ck = R / f"e2/ckpt/{m}_s0_reflow.pt"
    if not ck.exists():
        return None
    net = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1)
    net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
    net.eval()
    x0 = torch.randn(32, 1, 28, 28, generator=torch.Generator().manual_seed(2))[:n]
    out = {}
    with torch.no_grad():
        for k in ks:
            x = x0.clone()
            for i in range(k):
                x = x + net(torch.full((n,), i / k), x) / k
            out[k] = ((x * 127.5 + 128).clamp(0, 255)).to(torch.uint8).numpy()
    return out


def fig_samples():
    g = np.load(R / "e4/mnist_samples.npz")
    ks, n = [1, 2, 5, 100], 8
    rows = [("Real data", None, g["real"][:n])]
    for m in ("icfm", "otcfm", "ddpm"):
        for k in ks:
            rows.append((f"{NAMES[m]}, {k} step{'s' if k > 1 else ''}", m, g[f"{m}_{k}"][:n]))
    for m in ("icfm", "otcfm"):
        rg = reflow_grids(m, ks, n)
        if rg:
            for k in ks:
                rows.append((f"{NAMES[m]} + reflow, {k} step{'s' if k > 1 else ''}", m, rg[k]))
    # group rows by model into columns of panels: each panel = one model, rows = NFE
    groups = [("Real data", [rows[0]])]
    for i in range(1, len(rows), len(ks)):
        groups.append((rows[i][0].split(",")[0], rows[i:i + len(ks)]))
    ncol = 2
    nrow = int(np.ceil((len(groups) - 1) / ncol)) + 1
    fig = plt.figure(figsize=(5.5, 1.05 * (nrow - 1) * len(ks) / 4 * 4 * 0.33 + 0.55))
    gs = fig.add_gridspec(nrow, ncol, height_ratios=[1] + [len(ks)] * (nrow - 1), hspace=0.35, wspace=0.25)
    def strip(ax, imgs_rows, labels, title):
        big = np.concatenate([np.concatenate([im[0] for im in r], axis=1) for r in imgs_rows], axis=0)
        ax.imshow(big, cmap="gray", vmin=0, vmax=255, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([14 + 28 * i for i in range(len(labels))], labels, fontsize=8)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=7.5, loc="left")
    ax = fig.add_subplot(gs[0, :])
    strip(ax, [rows[0][2]], [""], "Real MNIST")
    for j, (title, rs) in enumerate(groups[1:]):
        ax = fig.add_subplot(gs[1 + j // ncol, j % ncol])
        strip(ax, [r[2] for r in rs], [f"{k}" for k in ks], title + " (rows: NFE)")
    fig.savefig(OUT / "fig_samples.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_2d():
    import sys
    import torch
    sys.path.insert(0, str(ROOT / "code"))
    from common.data_2d import sample
    from common.models import MLP
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4), sharex=True, sharey=True)
    data = sample("moons", 2000, np.random.default_rng(1)).numpy()
    for ax, m in zip(axes, ("icfm", "otcfm")):
        net = MLP(width=256)
        net.load_state_dict(torch.load(R / f"e1/ckpt/{m}_moons_s0.pt", map_location="cpu", weights_only=True))
        net.eval()
        x = torch.randn(300, 2, generator=torch.Generator().manual_seed(3))
        x0 = x.clone()
        traj = [x.numpy().copy()]
        with torch.no_grad():
            one = (x0 + net(x0, torch.zeros(300))).numpy()
            for i in range(100):
                x = x + net(x, torch.full((300,), i / 100)) / 100
                traj.append(x.numpy().copy())
        traj = np.stack(traj)
        ax.scatter(data[:, 0], data[:, 1], s=1.5, color=GRAY, alpha=0.5, lw=0, label="data")
        for j in range(0, 300, 3):
            ax.plot(traj[:, j, 0], traj[:, j, 1], color=MCOL[m], lw=0.6, alpha=0.85)
        ax.scatter(traj[0, :, 0], traj[0, :, 1], s=2, color=NAVY, lw=0, label="noise $x_0$")
        ax.scatter(one[:, 0], one[:, 1], s=4, marker="x", color="k", lw=0.5, label="one Euler step")
        ax.set_title(f"{NAMES[m]}: 100-step paths", fontsize=8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower left", markerscale=2)
    fig.tight_layout()
    fig.savefig(OUT / "fig_2d_paths.pdf")
    plt.close(fig)


def fig_kmeans():
    """Appendix B: generated / real share of each k-means cluster (k = 10), seed 0, 1 and 100 Euler steps."""
    km = load("c1_reliability/kmeans_c1.json")
    ref = np.array(km["ref_counts"]["k10"], float)
    ref /= ref.sum()
    order = np.argsort(-ref)
    fig, ax = plt.subplots(figsize=(5.5, 2.4))
    runs = [("real_test", "real test set", GRAY, None), ("icfm_s0_nfe1_noise7", "I-CFM, 1 step", DARK_ORANGE, "///"),
            ("otcfm_s0_nfe1_noise7", "OT-CFM, 1 step", ORANGE, "///"), ("icfm_s0_nfe100_noise7", "I-CFM, 100 steps", DARK_ORANGE, None),
            ("otcfm_s0_nfe100_noise7", "OT-CFM, 100 steps", ORANGE, None)]
    w = 0.16
    for j, (key, lab, c, hatch) in enumerate(runs):
        cnt = np.array(km["runs"][key]["k10"]["counts"], float)
        ratio = (cnt / cnt.sum())[order] / ref[order]
        ax.bar(np.arange(10) + (j - 2) * w, np.minimum(ratio, 3), w, color="white" if hatch else c, edgecolor=c, hatch=hatch, lw=0.6,
               label=f"{lab} (TV {km['runs'][key]['k10']['tv']:.3f})")
    ax.axhline(1, color="k", lw=0.6)
    ax.set_xticks(range(10), [f"C{i + 1}" for i in range(10)])
    ax.set(ylabel="generated / real share", xlabel="k-means cluster (fitted on real images only), ordered by real share", ylim=(0, 3))
    ax.set_title("Cluster shares relative to real data (k = 10, seed 0, 10,000 samples; bars capped at 3)", fontsize=7.5, loc="left")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=7.5, ncol=3, loc="lower center")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(OUT / "fig_kmeans.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_kmeans()
    fig_samples()
    fig_2d()
    fig_c1()
    fig_c2()
    fig_frontier()
    print("figures in", OUT)


if __name__ == "__main__":
    main()
