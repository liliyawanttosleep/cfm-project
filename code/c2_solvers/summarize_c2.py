"""C2 的汇总：表格和图（只用 CPU，读 results/c2_solvers/*.json）。

    python code/c2_solvers/summarize_c2.py

要完成的任务
    任务 1  表：每个模型、求解器、NFE 下，均匀网格、按曲率分配、终点误差校准、FID 最好的固定网格的 FID
    任务 2  达标步数：NFE@20 和 NFE@1.1×dopri5（在对数 NFE 上线性插值，评估协议见 FINAL_REPORT_PLAN §4）
    任务 3  图 c2_fid_vs_nfe.png：FID vs NFE，三种求解器 × {均匀, 按曲率分配}，加 dopri5 的水平线
    任务 4  图 c2_error_vs_fid.png：终点误差和 FID 的关系，按网格类型着色（检验“终点误差能否预测 FID”）
    任务 5  图 c2_kappa.png：κ(t)，对数纵轴
    任务 6  如果 t1_*.json 存在，汇总 heun_e / rk4_e 的结果

输出 results/c2_solvers/summary.md 和 results/c2_solvers/figs/。
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results" / "c2_solvers"
FIG = D / "figs"
NAMES = {"otcfm": "OT-CFM", "icfm": "I-CFM"}
SOLVERS = ["euler", "heun", "rk4"]
FIXED = ["uniform", "sqrt_kappa", "shift_0.25", "shift_0.5", "shift_2", "shift_4"]
COLORS = {"euler": "#2f6db3", "heun": "#d1772b", "rk4": "#3d9a5b"}


def first_reach(nfes, fids, target):
    """FID 第一次 ≤ target 时的 NFE，在 log NFE 上线性插值；达不到返回 None。"""
    for i, (n, f) in enumerate(zip(nfes, fids)):
        if f <= target:
            if i == 0:
                return float(n)
            n0, f0 = nfes[i - 1], fids[i - 1]
            r = (f0 - target) / (f0 - f)
            return float(np.exp(np.log(n0) + r * (np.log(n) - np.log(n0))))
    return None


def curve(d, solver, grid):
    """(NFE, FID) 按 NFE 排序；只有一步时网格没有区别，用均匀网格的值。"""
    pts = []
    for n, r in d["fid"].get(solver, {}).items():
        f = r.get(grid, r.get("uniform") if len(r) == 1 else None)
        if f is not None:
            pts.append((int(n), f))
    return sorted(pts)


def t1_curve(m, solver, grid):
    """任务 6：heun_e / rk4_e（最后一步不在 t = 1 调用网络）的 (NFE, FID)；没有 t1 文件时为空。"""
    f = D / f"t1_{m}_s0.json"
    if not f.exists():
        return []
    fid = json.loads(f.read_text()).get("fid", {})
    pts = [(int(k.split("|")[1]), v) for k, v in fid.items() if k.split("|")[0] == solver and k.split("|")[2] == grid]
    return sorted(pts)


def fmt(x, d=1):
    return "–" if x is None else f"{x:.{d}f}"


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    e2 = json.loads((ROOT / "results/e4/mnist_eval.json").read_text())["by_method"]
    data = {m: json.loads((D / f"{m}_s0.json").read_text()) for m in NAMES if (D / f"{m}_s0.json").exists()}
    lines = ["# C2 summary (seed 0, FID on 10k samples, noise seed 100)", ""]

    # 任务 1
    for m, d in data.items():
        dop = e2[m]["fid"]["dopri5"]
        lines += [f"## {NAMES[m]}  (dopri5 FID {dop:.1f} at ~{e2[m]['dopri5_mean_nfe']:.0f} NFE; "
                  f"κ peak t = {d['kappa']['t_peak']:.2f}, share of ∫κ in t > 0.9 = {d['kappa']['share_last_10pct']:.2f}, G = {d['kappa']['G']:.2f})", "",
                  "| solver | NFE | uniform | √κ | calibrated (endpoint) | best fixed grid | endpoint err: uniform → √κ → calibrated |",
                  "|---|---|---|---|---|---|---|"]
        for s in SOLVERS:
            for n, r in d["fid"].get(s, {}).items():
                cal = d["calibrated"][s][n]["grid"]
                fixed = {k: v for k, v in r.items() if k in FIXED}
                best = min(fixed, key=fixed.get)
                e = d["endpoint_error"][s][n]
                lines.append(f"| {s} | {n} | {fmt(r.get('uniform'))} | {fmt(r.get('sqrt_kappa'))} | {fmt(r.get(cal))} ({cal}) | "
                             f"{fmt(fixed[best])} ({best}) | {e['uniform']['B']:.3f} → {fmt(e.get('sqrt_kappa', {}).get('B'), 3)} → {e[cal]['B']:.3f} |")
        lines.append("")

    # 任务 2
    lines += ["## Steps to reach a target FID (interpolated in log NFE)", "",
              "| model | sampler | NFE@20 | NFE@1.1×dopri5 |", "|---|---|---|---|"]
    for m, d in data.items():
        dop = e2[m]["fid"]["dopri5"]
        ns = [k for k in ["1", "2", "5", "10", "20", "50", "100"]]
        curves = {"Euler, uniform (E2 grid 1–100)": ([int(k) for k in ns], [e2[m]["fid"][k] for k in ns])}
        for s in SOLVERS:
            for g in ("uniform", "sqrt_kappa"):
                pts = curve(d, s, g)
                if pts:
                    curves[f"{s}, {g} (C2 grid 2–32)"] = ([p[0] for p in pts], [p[1] for p in pts])
        for s, g in (("heun_e", "uniform"), ("heun_e", "sqrt_kappa"), ("rk4_e", "uniform")):
            pts = t1_curve(m, s, g)
            if pts:
                curves[f"{s} (last step avoids t = 1), {g}"] = ([p[0] for p in pts], [p[1] for p in pts])
        for name, (x, y) in curves.items():
            lines.append(f"| {NAMES[m]} | {name} | {fmt(first_reach(x, y, 20))} | {fmt(first_reach(x, y, 1.1 * dop))} |")
    lines.append("")

    # 任务 6
    for m in NAMES:
        f = D / f"t1_{m}_s0.json"
        if not f.exists():
            continue
        t = json.loads(f.read_text())
        lines += [f"## t = 1 check, {NAMES[m]}", "", "```", json.dumps(t.get("t1_diagnostic", {}), indent=1), "```", "",
                  "| setting (solver|NFE|grid) | endpoint err | FID |", "|---|---|---|"]
        for k, e in t.get("endpoint_error", {}).items():
            lines.append(f"| {k} | {e:.3f} | {fmt(t.get('fid', {}).get(k))} |")
        lines.append("")
    (D / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # 任务 3
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 4), squeeze=False)
    for ax, (m, d) in zip(axes[0], data.items()):
        for s in SOLVERS:
            for g, ls in (("uniform", "--"), ("sqrt_kappa", "-")):
                pts = curve(d, s, g)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ls, marker="o", ms=4, color=COLORS[s],
                        label=f"{s.capitalize() if s != 'rk4' else 'RK4'}, {'uniform' if g == 'uniform' else '√κ grid'}")
        for s, g, c in (("heun_e", "sqrt_kappa", COLORS["heun"]), ("rk4_e", "uniform", COLORS["rk4"])):
            pts = t1_curve(m, s, g)
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], ":", marker="x", ms=5, color=c,
                        label=f"{'Heun' if s == 'heun_e' else 'RK4'}, last step avoids t = 1, {'uniform' if g == 'uniform' else '√κ grid'}")
        ax.axhline(e2[m]["fid"]["dopri5"], color="0.4", lw=1, ls=":", label=f"dopri5 (~{e2[m]['dopri5_mean_nfe']:.0f} NFE)")
        ax.set(xscale="log", yscale="log", xlabel="NFE (network calls)", ylabel="FID (10k)", title=f"{NAMES[m]}, seed 0")
        ax.set_xticks([2, 4, 8, 16, 32], ["2", "4", "8", "16", "32"])
        ax.grid(alpha=0.3, which="both")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "c2_fid_vs_nfe.png", dpi=160)

    # 任务 4
    fig, axes = plt.subplots(1, len(data), figsize=(5.2 * len(data), 4), squeeze=False)
    fam = lambda k: "uniform" if k == "uniform" else "√κ" if k == "sqrt_kappa" else ("shift s<1 (dense near noise)" if float(k.split("_")[1]) < 1 else "shift s>1 (dense near data)")
    fc = {"uniform": "k", "√κ": "#c0392b", "shift s<1 (dense near noise)": "#2f6db3", "shift s>1 (dense near data)": "#3d9a5b"}
    mk = {"euler": "o", "heun": "s", "rk4": "^"}
    for ax, (m, d) in zip(axes[0], data.items()):
        seen = set()
        for s in SOLVERS:
            for n, r in d["fid"].get(s, {}).items():
                for k, f in r.items():
                    e = d["endpoint_error"][s][n][k]["B"]
                    lab = fam(k)
                    ax.scatter(e, f, c=fc[lab], marker=mk[s], s=18, alpha=0.8, label=lab if lab not in seen else None)
                    seen.add(lab)
        ax.set(xscale="log", yscale="log", xlabel="endpoint error vs dopri5 (RMS per pixel)", ylabel="FID (10k)", title=f"{NAMES[m]}: o Euler, □ Heun, △ RK4")
        ax.grid(alpha=0.3, which="both")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "c2_error_vs_fid.png", dpi=160)

    # 任务 5
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for m, d in data.items():
        ax.plot(d["kappa"]["t"], d["kappa"]["kappa"], label=NAMES[m])
    ax.set(yscale="log", xlabel="t (0 = noise, 1 = data)", ylabel="κ(t) = E‖dv/dt‖ (RMS per pixel)", title="Velocity change along dopri5 paths")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "c2_kappa.png", dpi=160)
    print("figures in", FIG)


if __name__ == "__main__":
    main()
