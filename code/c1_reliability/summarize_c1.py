"""C1 summary: error bars for every later comparison (CPU only, reads the json files).

    python code/c1_reliability/summarize_c1.py

Tasks
    Task 1  FID by training seed: seed 0 from results/e4/mnist_eval.json, seeds 1 and 2 from results/e3/fid_s{1,2}.json.
            Mean, sample sd (ddof = 1) and the improvement threshold 2 x sd for each model and NFE.
    Task 2  Sampling-noise sd (results/c1_reliability/fid_noise.json) next to the training-seed sd, and
            sigma_seed^2 ~ s_between^2 - sigma_noise^2 (derivations D8).
    Task 3  Paired OT-CFM - I-CFM differences per seed (same seed index, same noise); do the three share a sign?
    Task 4  Sample size: nested FID at 2k-20k, the 1/n fit FID_inf, and the spread of disjoint 2k / 5k blocks.
    Task 5  Straightness: path-length ratio for the three seeds.
    Task 6  Class coverage, if results/c1_reliability/kmeans_c1.json exists: TV with a 1,000-draw bootstrap 95% interval,
            the two-sample chi-square, and the TV noise floor from D8.
    Task 7  Figure figs/c1_fid_seeds.png: FID vs NFE for the three seeds of each model, log axes.

Output results/c1_reliability/summary.md and results/c1_reliability/figs/.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results" / "c1_reliability"
FIG = D / "figs"
NAMES = {"icfm": "I-CFM", "otcfm": "OT-CFM"}
COLORS = {"icfm": "#2f6db3", "otcfm": "#d1772b"}
KEYS = ["1", "2", "5", "10", "20", "50", "100", "dopri5"]


def load_seeds():
    """{method: {seed: {nfe: fid}}}, {method: {seed: path ratio}}, {method: {seed: dopri5 mean NFE}}."""
    e4 = json.loads((ROOT / "results/e4/mnist_eval.json").read_text())["by_method"]
    fid = {m: {0: dict(e4[m]["fid"])} for m in NAMES}
    pr = {m: {0: e4[m]["path_ratio"]} for m in NAMES}
    nfe = {m: {0: e4[m]["dopri5_mean_nfe"]} for m in NAMES}
    for s in (1, 2):
        f = ROOT / f"results/e3/fid_s{s}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        for m in NAMES:
            if m in r["fid"]:
                fid[m][s] = dict(r["fid"][m])
            if m in r.get("path_ratio", {}):
                pr[m][s] = r["path_ratio"][m]
            if m in r.get("dopri5_mean_nfe", {}):
                nfe[m][s] = r["dopri5_mean_nfe"][m]
    return fid, pr, nfe


def fmt(x, d=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def tv_floor(p, n, m):
    """D8: E[TV] when the generator is exactly right, n generated and m reference images."""
    return 0.5 * np.sum(np.sqrt(2 / np.pi * p * (1 - p) * (1 / n + 1 / m)))


def tv_boot(counts, ref, draws=1000, seed=0):
    """Bootstrap 95% interval for TV: redraw both the generated and the reference counts (both are estimates)."""
    rng = np.random.default_rng(seed)
    c, r = np.asarray(counts), np.asarray(ref)
    pc = rng.multinomial(c.sum(), c / c.sum(), draws) / c.sum()
    pr = rng.multinomial(r.sum(), r / r.sum(), draws) / r.sum()
    tv = 0.5 * np.abs(pc - pr).sum(1)
    return np.percentile(tv, [2.5, 97.5]).tolist()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fid, pr, dnfe = load_seeds()
    lines = ["# C1 summary: error bars (MNIST, FID on 10k samples unless stated)", ""]

    # Task 1
    sd = {}
    lines += ["## FID by training seed", "",
              "| model | NFE | seed 0 | seed 1 | seed 2 | mean | sd | threshold 2·sd |", "|---|---|---|---|---|---|---|---|"]
    for m in NAMES:
        sd[m] = {}
        for k in KEYS:
            v = [fid[m].get(s, {}).get(k) for s in (0, 1, 2)]
            got = np.array([x for x in v if x is not None])
            s_ = float(got.std(ddof=1)) if len(got) >= 2 else None
            sd[m][k] = s_
            lines.append(f"| {NAMES[m]} | {k} | " + " | ".join(fmt(x) for x in v) +
                         f" | {fmt(got.mean() if len(got) else None)} | {fmt(s_)} | {fmt(2 * s_ if s_ else None)} |")
    lines.append("")
    lines += ["dopri5 mean NFE by seed: " + "; ".join(f"{NAMES[m]} " + ", ".join(f"s{s} {v:.0f}" for s, v in sorted(dnfe[m].items())) for m in NAMES), ""]

    # Task 2
    fn = D / "fid_noise.json"
    if fn.exists():
        noise = json.loads(fn.read_text())
        lines += ["## Sampling noise vs training seed", "",
                  "| model | NFE | noise draws | noise mean | noise sd | seed sd | seed sd after removing noise (D8) |", "|---|---|---|---|---|---|---|"]
        for m in NAMES:
            for k, cell in noise["noise"].get(m, {}).items():
                v = np.array(list(cell.values()))
                ns = float(v.std(ddof=1)) if len(v) > 1 else None
                ss = sd[m].get(k)
                adj = np.sqrt(max(ss ** 2 - ns ** 2, 0)) if ss is not None and ns is not None else None
                lines.append(f"| {NAMES[m]} | {k} | {len(v)} | {fmt(v.mean())} | {fmt(ns, 3)} | {fmt(ss)} | {fmt(adj)} |")
        lines.append("")
        # Task 4
        lines += ["## Sample size (NFE 100, noise seed 100)", "",
                  "| model | 2k | 5k | 10k | 20k | FID_inf (fit c/n) | 2k blocks: mean ± sd | 5k blocks: mean ± sd | 10k check vs E4 |", "|---|---|---|---|---|---|---|---|---|"]
        for m, r in noise.get("sample_size", {}).items():
            nest = r["nested"]
            b2, b5 = np.array(r["blocks"]["2000"]), np.array(r["blocks"]["5000"])
            chk = r["check_10k_vs_eval"]
            lines.append(f"| {NAMES[m]} | " + " | ".join(fmt(nest.get(n)) for n in ("2000", "5000", "10000", "20000")) +
                         f" | {fmt(r['fid_inf'])} | {b2.mean():.2f} ± {b2.std(ddof=1):.2f} | {b5.mean():.2f} ± {b5.std(ddof=1):.2f} | {fmt(chk[0])} vs {fmt(chk[1])} |")
        lines.append("")

    # Task 3
    lines += ["## Paired difference OT-CFM − I-CFM (same seed index, same noise)", "",
              "| NFE | seed 0 | seed 1 | seed 2 | same sign? |", "|---|---|---|---|---|"]
    for k in KEYS:
        d = [fid["otcfm"].get(s, {}).get(k, np.nan) - fid["icfm"].get(s, {}).get(k, np.nan) for s in (0, 1, 2)]
        got = [x for x in d if not np.isnan(x)]
        same = "yes" if len(got) == 3 and (all(x < 0 for x in got) or all(x > 0 for x in got)) else ("no" if len(got) == 3 else "–")
        lines.append(f"| {k} | " + " | ".join(fmt(x) for x in d) + f" | {same} |")
    lines.append("")

    # Task 5
    lines += ["## Straightness (path-length ratio, 100 Euler steps, 256 paths; 1 = straight)", "",
              "| model | seed 0 | seed 1 | seed 2 |", "|---|---|---|---|"]
    for m in NAMES:
        lines.append(f"| {NAMES[m]} | " + " | ".join(fmt(pr[m].get(s), 4) for s in (0, 1, 2)) + " |")
    lines.append("")

    # Task 6
    fk = D / "kmeans_c1.json"
    if fk.exists():
        km = json.loads(fk.read_text())
        lines += ["## TV (k = 10) by training seed, noise 7; redraw sd from noise 7 / 11 / 12 at NFE 100", "",
                  "| model | NFE | seed 0 | seed 1 | seed 2 | mean | seed sd | threshold 2·sd | redraw sd (mean over seeds) |", "|---|---|---|---|---|---|---|---|---|"]
        for m in NAMES:
            for k in ("1", "2", "5", "20", "100"):
                v = [km["runs"].get(f"{m}_s{s}_nfe{k}_noise7", {}).get("k10", {}).get("tv") for s in (0, 1, 2)]
                got = np.array([x for x in v if x is not None])
                ssd = float(got.std(ddof=1)) if len(got) > 1 else None
                rd = None
                if k == "100":
                    rds = [np.std([km["runs"][f"{m}_s{s}_nfe100_noise{r}"]["k10"]["tv"] for r in (7, 11, 12)], ddof=1)
                           for s in (0, 1, 2) if all(f"{m}_s{s}_nfe100_noise{r}" in km["runs"] for r in (7, 11, 12))]
                    rd = float(np.mean(rds)) if rds else None
                lines.append(f"| {NAMES[m]} | {k} | " + " | ".join(fmt(x, 3) for x in v) +
                             f" | {fmt(got.mean() if len(got) else None, 3)} | {fmt(ssd, 3)} | {fmt(2 * ssd if ssd else None, 3)} | {fmt(rd, 3)} |")
        lines.append("")
        lines += ["## Class coverage (k-means fitted on real images only)", "",
                  "| run | k | TV | bootstrap 95% | noise floor (D8) | χ² two-sample p | w |", "|---|---|---|---|---|---|---|"]
        for key, r in km["runs"].items():
            for kk in ("k10", "k20"):
                if kk not in r or "counts" not in r[kk]:
                    continue
                c, ref = np.array(r[kk]["counts"]), np.array(km["ref_counts"][kk])
                lo, hi = tv_boot(c, ref)
                floor = tv_floor(ref / ref.sum(), c.sum(), ref.sum())
                lines.append(f"| {key} | {kk[1:]} | {r[kk]['tv']:.3f} | {lo:.3f}–{hi:.3f} | {floor:.4f} | {r[kk].get('p_2s', float('nan')):.2g} | {r[kk]['w']:.3f} |")
        lines.append("")

    (D / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # Task 7
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ks = [int(k) for k in KEYS if k != "dopri5"]
    for m in NAMES:
        for s, ls in zip((0, 1, 2), ("-", "--", ":")):
            if s in fid[m]:
                ax.plot(ks, [fid[m][s][str(k)] for k in ks], ls, marker="o", ms=3, color=COLORS[m], label=f"{NAMES[m]}, seed {s}")
    ax.set(xscale="log", yscale="log", xlabel="NFE (Euler steps)", ylabel="FID (10k)", title="FID across training seeds")
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "c1_fid_seeds.png", dpi=160)
    print("figure in", FIG)


if __name__ == "__main__":
    main()
