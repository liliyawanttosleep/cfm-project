"""C2 · 推理端优化：同一个训练好的 CFM 模型，只换求解器和时间网格，固定 NFE 下比较终点误差和 FID。

    python -u code/c2_solvers/sample_grid.py --method otcfm --seed 0
    python -u code/c2_solvers/sample_grid.py --method icfm  --seed 0
    python -u code/c2_solvers/sample_grid.py --method otcfm --quick      # 冒烟测试：2,000 张、只跑少数设置

要完成的任务（对应 docs/FINAL_REPORT_PLAN.md 的 C2）
    任务 1  参照解：两组各 256 个噪声（校准组 A、测试组 B），用 dopri5（tol 1e-5）解到 t = 1，记录平均 NFE
    任务 2  曲率：在 A 组的 dopri5 轨迹上取 401 个时刻，算速度沿路径的变化率 κ(t) = E‖dv/dt‖（每像素 RMS），
            以及 D5 的预测误差比 G = (∫√κ)² / ∫κ
    任务 3  候选网格：均匀；shift t' = s t / (1 + (s − 1) t)，s = 2^e，e ∈ {−3, −2.75, …, 3}（含 s = 0.25, 0.5, 2, 4）；
            按曲率分配（每步分到相同的 ∫√κ dt）
    任务 4  终点误差：Euler / Heun / RK4 × NFE {2, 4, 8, 16, 32} × 每个候选网格，在 A、B 两组上算和 dopri5 终点的
            每像素 RMS 距离
    任务 5  校准网格：每个（求解器, NFE）在 A 组上选终点误差最小的候选网格，在 B 组上报告它的误差（避免“在同一批上选又在
            同一批上评”）
    任务 6  FID：每个设置 1 万张，噪声 torch.manual_seed(100)，和 E2 / E3 的口径相同。先算均匀网格和校准网格，
            再算按曲率分配和 4 个固定 shift；其余 shift 只有终点误差，不算 FID

NFE 的记法：Euler 每步调用网络 1 次，Heun 2 次，RK4 4 次。只有一步时（Euler 1、Heun 2、RK4 4）网格没有区别，只算均匀网格。
记号：t = 0 是噪声，t = 1 是数据；s > 1 让步子集中在数据端，s < 1 集中在噪声端。

限速：code/common/throttle.py，每生成一批就调用一次 th.step()，限速不改变随机数。
中断后再跑：结果每算完一项就写入 results/c2_solvers/{method}_s{seed}.json，再跑会跳过已有的项。
"""
import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):  # 在 import numpy 之前限制线程
    os.environ.setdefault(_k, "4")
import numpy as np
import torch
from torchdiffeq import odeint

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from common.throttle import Throttle  # noqa: E402

spec = importlib.util.spec_from_file_location("eval_images", ROOT / "code/e4_cifar/eval_images.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
device = ev.device

SOLVERS = {"euler": 1, "heun": 2, "rk4": 4}  # 每步的 NFE
NFES = [2, 4, 8, 16, 32]
SHIFT_EXPONENTS = [e / 4 for e in range(-12, 13)]  # s = 2^e，e = −3 … 3，步长 0.25
FIXED_SHIFTS = [0.25, 0.5, 2.0, 4.0]


# ---------- 网格 ----------
def shift_key(s):
    return f"shift_{s:.4g}"


def shift_grid(n, s):
    t = torch.linspace(0, 1, n + 1, dtype=torch.float64)
    return s * t / (1 + (s - 1) * t)


def kappa_grid(n, t, kappa):
    """D5：让每一步分到相同的 ∫√κ dt。t、kappa 是 401 个时刻上的值。"""
    w = np.diff(t)
    mid = np.sqrt(0.5 * (kappa[1:] + kappa[:-1]))
    F = np.concatenate([[0.0], np.cumsum(mid * w)])
    F /= F[-1]
    return torch.tensor(np.interp(np.linspace(0, 1, n + 1), F, t), dtype=torch.float64)


def candidate_grids(n, t, kappa):
    g = {"uniform": shift_grid(n, 1.0), "sqrt_kappa": kappa_grid(n, t, kappa)}
    for e in SHIFT_EXPONENTS:
        if e != 0:
            g[shift_key(2.0 ** e)] = shift_grid(n, 2.0 ** e)
    return g


# ---------- 求解器：grid 是 t_0 = 0 < … < t_N = 1 ----------
def v(net, x, t):
    return net(torch.full((x.shape[0],), float(t), device=device), x)


@torch.no_grad()
def solve(net, x, grid, solver):
    g = grid.tolist()
    for a, b in zip(g[:-1], g[1:]):
        h = b - a
        if solver == "euler":
            x = x + h * v(net, x, a)
        elif solver == "heun":
            k1 = v(net, x, a)
            k2 = v(net, x + h * k1, b)
            x = x + h / 2 * (k1 + k2)
        else:
            k1 = v(net, x, a)
            k2 = v(net, x + h / 2 * k1, a + h / 2)
            k3 = v(net, x + h / 2 * k2, a + h / 2)
            k4 = v(net, x + h * k3, b)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def rms(a, b):
    return float((a - b).pow(2).mean().sqrt())


# ---------- 任务 1、2 ----------
@torch.no_grad()
def reference(net, x0, tol, n_t=401):
    """dopri5 解到 t = 1；同时在 n_t 个等距时刻取轨迹点，用来算 κ(t)。"""
    c = ev.Counted(net)
    t = torch.linspace(0, 1, n_t, device=device)
    traj = odeint(c, x0, t, rtol=tol, atol=tol, method="dopri5")
    nfe = c.nfe
    vs = torch.stack([v(net, traj[j], t[j]) for j in range(n_t)])  # n_t × B × …
    dv = (vs[1:] - vs[:-1]).flatten(2).pow(2).mean(2).sqrt() / (t[1] - t[0])  # 每像素 RMS，(n_t − 1) × B
    k = dv.mean(1).cpu().double().numpy()
    tm = t.cpu().double().numpy()
    kappa = np.concatenate([[k[0]], 0.5 * (k[1:] + k[:-1]), [k[-1]]])  # 放回 n_t 个时刻
    kappa = np.maximum(kappa, 1e-3 * kappa.max())
    return traj[-1], nfe, tm, kappa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["icfm", "otcfm"], required=True)
    p.add_argument("--seed", type=int, default=0, help="训练种子")
    p.add_argument("--n", type=int, default=10000, help="FID 的样本数")
    p.add_argument("--tol", type=float, default=1e-5)
    p.add_argument("--n-ref", type=int, default=256, help="每组参照路径数")
    p.add_argument("--no-fid", action="store_true", help="只算终点误差和校准")
    p.add_argument("--quick", action="store_true", help="冒烟测试：2,000 张，NFE 4 和 16，写到 *_quick.json")
    p.add_argument("--duty", type=float, default=0.5, help="显卡忙碌时间的比例，1 = 不限速")
    p.add_argument("--max-temp", type=int, default=72)
    p.add_argument("--resume-temp", type=int, default=65)
    p.add_argument("--min-free-gb", type=float, default=1.5)
    args = p.parse_args()
    nfes = [4, 16] if args.quick else NFES
    n_fid = 2000 if args.quick else args.n

    th = Throttle(args.duty, args.max_temp, args.resume_temp, args.min_free_gb, threads=int(os.environ["OMP_NUM_THREADS"]))
    out = ROOT / "results" / "c2_solvers" / f"{args.method}_s{args.seed}{'_quick' if args.quick else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {}
    if res and (res.get("n_fid"), res.get("tol"), res.get("n_ref")) != (n_fid, args.tol, args.n_ref):
        sys.exit(f"{out} was made with other settings; delete it or change the arguments")
    res.update({"method": args.method, "seed": args.seed, "n_fid": n_fid, "tol": args.tol, "n_ref": args.n_ref})

    def save():
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=2))
        os.replace(tmp, out)

    net = ev.load_mnist(args.method, args.seed)
    shape = (1, 28, 28)

    # 任务 1、2：参照解和曲率（每次重算，约 1 分钟；A、B 两组噪声固定）
    t0 = time.time()
    th.reset()
    xa = torch.randn(args.n_ref, *shape, generator=torch.Generator().manual_seed(300)).to(device)
    xb = torch.randn(args.n_ref, *shape, generator=torch.Generator().manual_seed(301)).to(device)
    ya, nfe_a, tk, kappa = reference(net, xa, args.tol)
    th.step()
    yb, nfe_b, _, _ = reference(net, xb, args.tol)
    th.step()
    w = np.diff(tk)
    mid = 0.5 * (kappa[1:] + kappa[:-1])
    G = float(np.sum(np.sqrt(mid) * w) ** 2 / np.sum(mid * w))
    res["reference"] = {"dopri5_nfe": [nfe_a, nfe_b], "seconds": time.time() - t0}
    res["kappa"] = {"t": tk[::4].tolist(), "kappa": kappa[::4].tolist(), "G": G,
                    "t_peak": float(tk[np.argmax(kappa)]), "share_last_10pct": float(np.sum(mid[tk[1:] > 0.9] * w[tk[1:] > 0.9]) / np.sum(mid * w))}
    save()
    print(f"reference dopri5 NFE {nfe_a}/{nfe_b}; G = {G:.3f}; kappa peak at t = {res['kappa']['t_peak']:.3f}; "
          f"share of int kappa in t > 0.9: {res['kappa']['share_last_10pct']:.2f}", flush=True)

    # 任务 3–5：终点误差和校准
    err, cal = res.setdefault("endpoint_error", {}), res.setdefault("calibrated", {})
    for solver, per in SOLVERS.items():
        for nfe in nfes:
            if nfe < per:
                continue
            steps = nfe // per
            grids = candidate_grids(steps, tk, kappa) if steps > 1 else {"uniform": shift_grid(1, 1.0)}
            rec = err.setdefault(solver, {}).setdefault(str(nfe), {})
            th.reset()
            for name, g in grids.items():
                if name not in rec:
                    rec[name] = {"A": rms(solve(net, xa, g, solver), ya), "B": rms(solve(net, xb, g, solver), yb)}
                    th.step()
            best = min(rec, key=lambda k: rec[k]["A"])
            cal.setdefault(solver, {})[str(nfe)] = {"grid": best, "B": rec[best]["B"], "uniform_B": rec["uniform"]["B"],
                                                     "grid_t": grids[best].tolist()}
            save()
            print(f"{solver:5s} NFE {nfe:2d}: uniform {rec['uniform']['B']:.4f}  sqrt_kappa {rec.get('sqrt_kappa', {}).get('B', float('nan')):.4f}"
                  f"  calibrated {best} {rec[best]['B']:.4f}", flush=True)

    if args.no_fid:
        print("saved", out, flush=True)
        return

    # 任务 6：FID。按优先级排队：均匀和校准 → 按曲率分配 → 固定 shift
    jobs = []
    for prio in range(3):
        for solver, per in SOLVERS.items():
            for nfe in nfes:
                if nfe < per:
                    continue
                steps = nfe // per
                if steps == 1:
                    names = ["uniform"] if prio == 0 else []
                elif prio == 0:
                    names = ["uniform", cal[solver][str(nfe)]["grid"]]
                elif prio == 1:
                    names = ["sqrt_kappa"]
                else:
                    names = [shift_key(s) for s in FIXED_SHIFTS]
                for name in names:
                    jobs.append((solver, nfe, name))
    real = ev.real_images("mnist")
    meter = ev.FID("mnist", real)
    fids = res.setdefault("fid", {})
    for solver, nfe, name in jobs:
        rec = fids.setdefault(solver, {}).setdefault(str(nfe), {})
        if name in rec:
            continue
        steps = nfe // SOLVERS[solver]
        g = candidate_grids(steps, tk, kappa)[name] if steps > 1 else shift_grid(1, 1.0)
        t0 = time.time()
        torch.manual_seed(100)
        th.reset()

        def gen(i, b):
            x = solve(net, torch.randn(b, *shape, device=device), g, solver)
            th.step()
            return x

        rec[name] = meter.score(gen, n_fid)
        save()
        print(f"FID {solver:5s} NFE {nfe:2d} {name:12s} {rec[name]:7.2f}  ({time.time() - t0:.0f}s)", flush=True)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
