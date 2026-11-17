"""LaTeX tables for the final report, generated from results/ so every number is traceable (CPU only).

    python report/make_tables.py

Tables (report/tables/*.tex, each a complete table environment)
    tab_reliability.tex  C1: FID over three training seeds, sampling-noise sd, sample-size effect.
    tab_pairs.tex        C3 step 1 (+ C5 step 5): pairing cost saved by minibatch OT at each OT batch size, and reflow pairs.
    tab_training.tex     C3-C5: every trained variant at seed 0 with Euler FID, dopri5, straightness, NFE@20 and training time.
    tab_sampler.tex      C2: NFE@20 and FID at 8 / 16 / 32 calls for each sampler on the seed-0 base models.
    tab_seeds.tex        C2 on training seeds 0-2: each inference-side conclusion as a paired before -> after.
    tab_literature.tex   Published CIFAR-10 numbers for reference (not comparable with our MNIST numbers).
Runs that have not finished appear as "running"; rerun the script when they finish.
Also writes report/materials/numbers.json with every value used, keyed by table.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "report" / "tables"
NAMES = {"icfm": "I-CFM", "otcfm": "OT-CFM", "ddpm": "DDIM"}
NUMBERS = {}


def load(p):
    p = R / p
    return json.loads(p.read_text()) if p.exists() else None


def f3(x):
    """Three significant figures, as required by report/README.md."""
    if x is None:
        return "--"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    if abs(x) >= 1:
        return f"{x:.2f}"
    return f"{x:.3g}"


def first_reach(nfes, fids, target=20.0):
    """NFE where FID first reaches target, linear in log NFE; None if never."""
    for i, (n, f) in enumerate(zip(nfes, fids)):
        if f <= target:
            if i == 0:
                return float(n)
            n0, f0 = nfes[i - 1], fids[i - 1]
            r = (f0 - target) / (f0 - f)
            return float(np.exp(np.log(n0) + r * (np.log(n) - np.log(n0))))
    return None


def write(name, body):
    (OUT / name).write_text(body, encoding="utf-8")


def seed_fids():
    e4 = load("e4/mnist_eval.json")["by_method"]
    fid = {m: {0: e4[m]["fid"]} for m in ("icfm", "otcfm", "ddpm")}
    for s in (1, 2):
        r = load(f"e3/fid_s{s}.json")
        if r:
            for m in fid:
                if m in r["fid"]:
                    fid[m][s] = r["fid"][m]
    return fid


def tab_reliability():
    fid = seed_fids()
    noise = load("c1_reliability/fid_noise.json")
    rows, nums = [], {}
    for m in ("icfm", "otcfm"):
        cells = []
        for k in ("1", "5", "100", "dopri5"):
            v = np.array([fid[m][s][k] for s in (0, 1, 2) if k in fid[m].get(s, {})])
            cells.append(f"${f3(v.mean())}\\pm{f3(v.std(ddof=1))}$")
            nums[f"{m}_{k}"] = {"mean": v.mean(), "sd": v.std(ddof=1), "values": v.tolist()}
        rows.append(f"{NAMES[m]} (3 seeds) & " + " & ".join(cells) + r" \\")
    d = fid["ddpm"]
    rows.append("DDIM (2 seeds) & " + " & ".join(", ".join(f3(d[s][k]) for s in (0, 1)) for k in ("1", "5", "100")) + r" & -- \\")
    nsd = {m: {k: np.std(list(noise["noise"][m][k].values()), ddof=1) for k in ("2", "10", "100")} for m in ("icfm", "otcfm")}
    ss = noise["sample_size"]
    nums["noise_sd"] = nsd
    nums["sample_size"] = ss
    body = r"""\begin{table}[t]
\centering
\caption{Evaluation error on MNIST (FID, lower is better). Top: base models, mean $\pm$ sd over 3 training seeds (DDIM: the 2 seeds' values), 10,000 samples. Middle: sd over 6 sampling-noise draws of the seed-0 model. Bottom: seed-0 model at NFE~100 against the number of generated samples, and the $c/n$ extrapolation FID$_\infty$ \citep{chong2020fid}.}
\label{tab:c1}
\small
\begin{tabular}{lcccc}
\toprule
 & NFE 1 & NFE 5 & NFE 100 & dopri5 \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
sampling-noise sd & NFE 2 & NFE 10 & NFE 100 & \\
\midrule
I-CFM / OT-CFM & """ + " & ".join(f"{nsd['icfm'][k]:.2f} / {nsd['otcfm'][k]:.2f}" for k in ("2", "10", "100")) + r""" & \\
\midrule
samples ($n$) & 2k / 5k & 10k & 20k & FID$_\infty$ \\
\midrule
""" + "\n".join(f"{NAMES[m]}, NFE 100 & {f3(ss[m]['nested']['2000'])} / {f3(ss[m]['nested']['5000'])} & {f3(ss[m]['nested']['10000'])} & {f3(ss[m]['nested']['20000'])} & {f3(ss[m]['fid_inf'])}" + r" \\" for m in ("icfm", "otcfm")) + r"""
\bottomrule
\end{tabular}
\end{table}
% sources: results/e4/mnist_eval.json, results/e3/fid_s{1,2}.json, results/c1_reliability/fid_noise.json
"""
    write("tab_reliability.tex", body)
    NUMBERS["reliability"] = nums


def tab_pairs():
    p = load("c3_otbatch/pairs.json")
    bs = sorted(p["pairs"], key=int)
    total = np.mean([p["pairs"][b]["random"] for b in bs])
    extra, nums = [], {"ot": p["pairs"], "bound": p["two_tr_sqrt_cov"], "total": total}
    for m in ("icfm", "otcfm"):
        r = load(f"c5_reflow/pairs_{m}_s0.json")
        if r and "summary" in r:
            c = r["summary"]["pair_cost"]
            extra.append(f"\\multicolumn{{9}}{{l}}{{Reflow pairs from {NAMES[m]} seed 0 (60,000 pairs): cost {f3(c)}, "
                         f"saved {100 * (total - c) / total:.1f}\\% of the random cost, {100 * (total - c) / p['two_tr_sqrt_cov']:.0f}\\% of the bound.}} \\\\")
            nums[f"reflow_{m}"] = r["summary"]
    body = r"""\begin{table}[t]
\centering
\caption{Pairing cost $\E\|x_0-x_1\|^2$ saved relative to random pairs on MNIST (random-pair cost """ + f"{total:,.0f}" + r"""), as a share of that cost and of the Gelbrich bound $2\operatorname{tr}\Sigma^{1/2}=""" + f"{p['two_tr_sqrt_cov']:.0f}" + r"""$, and the CPU time of one exact OT solve (POT, 4 threads).}
\label{tab:pairs}
\small
\setlength{\tabcolsep}{4.5pt}
\adjustbox{max width=\linewidth}{%
\begin{tabular}{lcccccccc}
\toprule
OT batch & """ + " & ".join(f"{int(b):,}" for b in bs) + r""" \\
\midrule
saved / random cost & """ + " & ".join(f"{100 * p['pairs'][b]['share_of_total']:.1f}\\%" for b in bs) + r""" \\
saved / bound & """ + " & ".join(f"{100 * p['pairs'][b]['share_of_bound']:.0f}\\%" for b in bs) + r""" \\
OT solve (s) & """ + " & ".join(f"{p['pairs'][b]['ot_seconds']:.2g}" for b in bs) + r""" \\
""" + ("\\midrule\n" + "\n".join(extra) + "\n" if extra else "") + r"""\bottomrule
\end{tabular}}
\end{table}
% sources: results/c3_otbatch/pairs.json, results/c5_reflow/pairs_{icfm,otcfm}_s0.json
"""
    write("tab_pairs.tex", body)
    NUMBERS["pairs"] = nums


def pre_resume_seconds(m, tag):
    """A run resumed after a crash stores only the wall time after the resume in "seconds"
    (ot_seconds is carried over in the checkpoint). Add the elapsed time of the last epoch
    that was kept before the resume, read from the training log."""
    import re
    logf = ROOT / "results" / "c3_otbatch" / "logs" / f"train_{m}_s0{tag}.log"
    if not logf.exists():
        return 0.0
    txt = logf.read_text(errors="ignore")
    if "resumed after epoch" not in txt:
        return 0.0
    before, after = txt.split("resumed after epoch", 1)
    kept = int(after.split()[0])
    hit = re.search(rf"epoch {kept}/\d+ .*?elapsed (\d+)s", before)
    return float(hit.group(1)) if hit else 0.0


def tab_training():
    e4 = load("e4/mnist_eval.json")["by_method"]
    ks = ["1", "2", "5", "10", "100"]
    fid = seed_fids()
    sd = {m: {k: np.std([fid[m][s][k] for s in (0, 1, 2)], ddof=1) for k in ks + ["dopri5"]} for m in ("icfm", "otcfm")}
    base_time = {m: load(f"e2/{m}_s2_train.json")["seconds"] for m in ("icfm", "otcfm")}  # throttled base runs
    variants = [("icfm", "", "I-CFM (base)", "e2/icfm_s2_train.json"),
                ("otcfm", "", "OT-CFM (base)", "e2/otcfm_s2_train.json"),
                ("otcfm", "_otb512", "OT-CFM, OT batch 512", "e2/otcfm_s0_otb512_train.json"),
                ("otcfm", "_otb2048", "OT-CFM, OT batch 2,048", "e2/otcfm_s0_otb2048_train.json"),
                ("icfm", "_tlogitnormal", "I-CFM, logit-normal $t$", "e2/icfm_s0_tlogitnormal_train.json"),
                ("icfm", "_tkappa", "I-CFM, $\\kappa$-weighted $t$", "e2/icfm_s0_tkappa_train.json"),
                ("icfm", "_reflow", "I-CFM + reflow", "e2/icfm_s0_reflow_train.json"),
                ("otcfm", "_reflow", "OT-CFM + reflow", "e2/otcfm_s0_reflow_train.json")]
    rows, nums = [], {}
    for m, tag, lab, tj in variants:
        if tag == "":
            f, pr, dop = e4[m]["fid"], e4[m]["path_ratio"], e4[m]["fid"]["dopri5"]
        else:
            r = load(f"e3/fid_s0{tag}.json")
            if not r or m not in r["fid"] or not all(k in r["fid"][m] for k in ks):
                rows.append(f"{lab} & \\multicolumn{{9}}{{c}}{{\\textit{{running}}}} \\\\")
                continue
            f, pr, dop = r["fid"][m], r.get("path_ratio", {}).get(m), r["fid"][m].get("dopri5")
        nf = first_reach([int(k) for k in ["1", "2", "5", "10", "20", "50", "100"]], [f[k] for k in ["1", "2", "5", "10", "20", "50", "100"]])
        t = load(tj)
        cost = None
        if t:
            sec = t["seconds"] + pre_resume_seconds(m, tag)
            if tag == "_reflow":
                pj = load(f"c5_reflow/pairs_{m}_s0.json")
                sec += pj["summary"]["gpu_seconds"] if pj and "summary" in pj else 0
            cost = sec / base_time[m]
        # mark differences beyond 2 x training-seed sd of the same base method
        cells = []
        for k in ks:
            v = f3(f[k])
            if tag and abs(f[k] - e4[m]["fid"][k]) > 2 * sd[m][k]:
                v = f"\\textbf{{{v}}}"
            cells.append(v)
        dv = f3(dop)
        if tag and dop is not None and abs(dop - e4[m]["fid"]["dopri5"]) > 2 * sd[m]["dopri5"]:
            dv = f"\\textbf{{{dv}}}"
        rows.append(f"{lab} & " + " & ".join(cells) + f" & {dv} & {pr:.3f} & {f3(nf) if nf else '$>100$'} & {f3(cost) if cost else '--'}" + r" \\")
        nums[m + tag] = {"fid": f, "path_ratio": pr, "nfe20": nf, "cost_rel": cost}
    thr = " & ".join(f"{f3(2 * sd['icfm'][k])} / {f3(2 * sd['otcfm'][k])}" for k in ks + ["dopri5"])
    body = r"""\begin{table}[t]
\centering
\caption{Training-side variants, training seed 0, Euler sampler on the uniform grid, FID on 10,000 samples (lower is better). $R$: path-length ratio (1 = straight). NFE@20: steps to reach FID 20. Time: training wall time (reflow: plus generating the pairs) relative to the base model, both throttled to the same duty cycle (OT batch 2,048 was resumed after a power loss at epoch 20; its time adds the 20 epochs before the resume). Bold: differs from the base model of the same method by more than the improvement threshold $2\,\mathrm{sd}$ over training seeds (last row).}
\label{tab:training}
\small
\setlength{\tabcolsep}{2.9pt}
\begin{tabular}{lccccccccc}
\toprule
 & \multicolumn{5}{c}{FID at NFE} & & & & \\
\cmidrule(lr){2-6}
Model & 1 & 2 & 5 & 10 & 100 & dopri5 & $R$ & NFE@20 & Time \\
\midrule
""" + "\n".join(rows) + r"""
\midrule
\multicolumn{10}{l}{Threshold $2\,\mathrm{sd}$ over three training seeds, I-CFM / OT-CFM:} \\
 & """ + thr.replace(" / ", "/") + r""" & & & \\
\bottomrule
\end{tabular}
\end{table}
% sources: results/e4/mnist_eval.json, results/e3/fid_s0_*.json, results/e2/*_train.json, results/c5_reflow/pairs_*.json
"""
    write("tab_training.tex", body)
    NUMBERS["training"] = nums


def tab_sampler():
    rows, nums = [], {}
    for m in ("icfm", "otcfm"):
        d = load(f"c2_solvers/{m}_s0.json")
        t1 = load(f"c2_solvers/t1_{m}_s0.json")
        e4 = load("e4/mnist_eval.json")["by_method"][m]["fid"]
        ks = ["1", "2", "5", "10", "20", "50", "100"]
        samplers = []
        for s, lab in (("euler", "Euler"), ("heun", "Heun"), ("rk4", "RK4")):
            for g, gl in (("uniform", "uniform"), ("sqrt_kappa", "$\\sqrt\\kappa$")):
                if s != "euler" and g != "uniform":
                    continue  # Heun/RK4 only on the uniform grid here; their sqrt-kappa results are in results/c2_solvers/summary.md
                pts = sorted((int(n), r[g]) for n, r in d["fid"][s].items() if g in r)
                samplers.append((f"{lab}, {gl}", [p[0] for p in pts], [p[1] for p in pts]))
        if t1:
            for s, g, lab in (("heun_e", "uniform", "Heun, last step Euler, uniform"), ("heun_e", "sqrt_kappa", "Heun, last step Euler, $\\sqrt\\kappa$"),
                              ("rk4_e", "uniform", "RK4, last step Heun, uniform")):
                pts = sorted((int(k.split("|")[1]), v) for k, v in t1["fid"].items() if k.split("|")[0] == s and k.split("|")[2] == g)
                samplers.append((lab, [p[0] for p in pts], [p[1] for p in pts]))
        for lab, x, y in samplers:
            # Euler on the uniform grid: NFE@20 from the finer E4 grid (1, 2, 5, ..., 100), which also covers NFE > 32
            nf = first_reach([int(k) for k in ks], [e4[k] for k in ks]) if lab == "Euler, uniform" else first_reach(x, y)
            near = {}
            for target in (8, 16, 32):
                j = int(np.argmin([abs(np.log(v / target)) for v in x]))
                near[target] = (x[j], y[j]) if abs(np.log(x[j] / target)) < 0.2 else None
            cells = " & ".join(f"{f3(near[t][1])}" + (f"$^{{({near[t][0]})}}$" if near[t] and near[t][0] != t else "") if near[t] else "--" for t in (8, 16, 32))
            rows.append(f"{NAMES[m] if lab.startswith('Euler, uniform') else ''} & {lab} & {cells} & {f3(nf) if nf else '$>32$'}" + r" \\")
            nums[f"{m}|{lab}"] = {"nfe": x, "fid": y, "nfe20": nf}
        rows.append(r"\midrule")
    rows = rows[:-1]
    body = r"""\begin{table}[t]
\centering
\caption{Inference-side samplers on the seed-0 base models: FID (10,000 samples, lower is better) at about 8, 16 and 32 network calls (exact count in parentheses when it differs), and NFE@20. The last-step variants never evaluate the network at $t=1$. Paired comparisons on one model: the sampling-noise sd is 0.05--0.6 (Table~\ref{tab:c1}).}
\label{tab:sampler}
\small
\setlength{\tabcolsep}{4pt}
\adjustbox{max width=\linewidth}{%
\begin{tabular}{llcccc}
\toprule
Model & Sampler & NFE $\approx$ 8 & 16 & 32 & NFE@20 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
% sources: results/e4/mnist_eval.json, results/c2_solvers/{icfm,otcfm}_s0.json, results/c2_solvers/t1_{icfm,otcfm}_s0.json
"""
    write("tab_sampler.tex", body)
    NUMBERS["sampler"] = nums


def tab_seeds():
    """C2 on all three training seeds: does each inference-side conclusion keep its direction?"""
    rows, nums = [], {}
    arrow = lambda a, b: "--" if a is None else f"{f3(a)} $\\to$ {'running' if b is None else f3(b)}"
    for m in ("icfm", "otcfm"):
        for s in (0, 1, 2):
            d, t = load(f"c2_solvers/{m}_s{s}.json"), load(f"c2_solvers/t1_{m}_s{s}.json") or {}
            if d is None:
                continue
            g = lambda solver, n, grid: d["fid"].get(solver, {}).get(str(n), {}).get(grid)
            tf, diag = t.get("fid", {}), t.get("t1_diagnostic", {})
            rel = lambda k: None if k not in diag else 100 * diag[k] / diag["rms_x"]
            r = {"euler4": (g("euler", 4, "uniform"), g("euler", 4, "sqrt_kappa")),
                 "euler16": (g("euler", 16, "uniform"), g("euler", 16, "sqrt_kappa")),
                 "euler32": (g("euler", 32, "uniform"), g("euler", 32, "sqrt_kappa")),
                 "heun32_to_heun_e31": (g("heun", 32, "uniform"), tf.get("heun_e|31|uniform")),
                 "heun_e31_sqrtk": tf.get("heun_e|31|sqrt_kappa"),
                 "t1_rel_err_real_gen": (rel("real_rms_v1_minus_x"), rel("gen_rms_v1_minus_x"))}
            nums[f"{m}_s{s}"] = r
            pct = "--" if r["t1_rel_err_real_gen"][0] is None else f"{r['t1_rel_err_real_gen'][0]:.0f} / {r['t1_rel_err_real_gen'][1]:.0f}"
            rows.append(f"{NAMES[m] if s == 0 else ''} & {s} & {arrow(*r['euler4'])} & {arrow(*r['euler16'])} & {arrow(*r['euler32'])} & "
                        f"{arrow(*r['heun32_to_heun_e31'])} & {f3(r['heun_e31_sqrtk']) if r['heun_e31_sqrtk'] else 'running'} & {pct} \\\\")
        rows.append("\\midrule")
    NUMBERS["tab_seeds"] = nums
    body = r"""\begin{table}[h]
\centering
\caption{The inference-side conclusions of Section~\ref{sec:c2} on all three training seeds (FID, 10,000 samples, lower is better; ``a $\to$ b'' is a paired comparison on one model). Columns 3--5: Euler, uniform $\to$ $\sqrt\kappa$ grid at 4, 16 and 32 steps. Column 6: Heun on the uniform grid at 32 NFE $\to$ Heun whose last step is Euler (31 NFE). Column 7: the latter on the $\sqrt\kappa$ grid. Last column: RMS error of $v(x,1)$ against its ideal value $x$, in \% of $\mathrm{RMS}(x)$, on real / generated images.}
\label{tab:seeds}
\small
\setlength{\tabcolsep}{3pt}
\adjustbox{max width=\linewidth}{%
\begin{tabular}{llcccccc}
\toprule
Model & Seed & Euler 4 & Euler 16 & Euler 32 & Heun 32 $\to$ Heun* 31 & Heun* 31, $\sqrt\kappa$ & $t=1$ error (\%) \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}}
\end{table}
% source: results/c2_solvers/{icfm,otcfm}_s{0,1,2}.json and t1_*.json (code/c2_solvers/sample_grid.py --core, t1_check.py)
"""
    write("tab_seeds.tex", body)


def tab_literature():
    body = r"""\begin{table}[t]
\centering
\caption{Published unconditional CIFAR-10 results for context (FID on 50,000 samples, numbers as reported). They use far larger networks and training budgets and a different dataset, so they are not comparable with our MNIST numbers; they show the same pattern of NFE versus quality.}
\label{tab:literature}
\small
\begin{tabular}{lcc}
\toprule
Method & NFE & FID \\
\midrule
DDPM \citep{ho2020ddpm} & 1,000 & 3.17 \\
DDIM \citep{song2021ddim} & 10 / 100 & 13.36 / 4.16 \\
EDM, Heun with tuned grid \citep{karras2022edm} & 35 & 1.97 \\
I-CFM / OT-CFM, dopri5 \citep{tong2024otcfm} & 146 / 134 & 3.66 / 3.58 \\
1-rectified flow, one Euler step \citep{liu2023rf} & 1 & 378 \\
2-rectified flow (reflow), one Euler step \citep{liu2023rf} & 1 & 12.21 \\
Consistency distillation \citep{song2023cm} & 1 / 2 & 3.55 / 2.93 \\
\bottomrule
\end{tabular}
\end{table}
% sources: papers/10 (p.1), 11 (Table 1, eta = 0), 16 (Table 2), 23 (Table 5), 21 (Table 1), 31 (Table 1)
"""
    write("tab_literature.tex", body)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tab_reliability()
    tab_pairs()
    tab_training()
    tab_sampler()
    tab_seeds()
    tab_literature()
    (ROOT / "report/materials").mkdir(exist_ok=True)
    (ROOT / "report/materials/numbers.json").write_text(json.dumps(NUMBERS, indent=1, default=float), encoding="utf-8")
    print("tables in", OUT)


if __name__ == "__main__":
    main()
