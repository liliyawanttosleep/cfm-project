"""C2 的补充检验：高阶求解器少步时 FID 很差，是不是因为它们在 t = 1 处调用网络？

    python -u code/c2_solvers/t1_check.py --method otcfm --seed 0

背景：C2 中 16 次调用、均匀网格时，Heun / RK4 的终点误差和 Euler 相当，FID 却差 3–4 倍。Heun 的校正步和 RK4 的
最后一次调用都在 t = 1 处算网络，而 Euler 从不在 t = 1 处调用；κ(t) 的峰也在 t = 1。

要完成的任务
    任务 1  t = 1 处网络的误差：x0 与 x1 独立时理想速度是 v*(1, x) = E[x1 − x0 | x1 = x] = x。对 2,000 张真实图算
            每像素 RMS ‖v(1, x) − x‖，并和 dopri5 生成图上的同一量比较；再看 t = 0.98、0.99 处 v 的变化有多快
    任务 2  变体 heun_e：Heun，但最后一步改用 Euler（EDM 的做法），N 步共 2N − 1 次调用，不在 t = 1 处调用网络。
            N = 2, 4, 8, 16（NFE 3, 7, 15, 31），均匀网格和按曲率分配的网格（κ 用和 C2 相同的 A 组重新测），算终点误差和 FID
    任务 3  变体 rk4_e：RK4，但最后一步改用 Heun 的预测–校正、终点处的调用改在 t = 1 − h/2（中点）；N 步共 4N − 2 次调用。
            只用均匀网格，N = 2, 4, 8（NFE 6, 14, 30）

判定：如果 heun_e 在同样调用次数下 FID 接近或好于 Euler，而终点误差和 Heun 相近，就说明问题出在 t = 1 处的那一次调用，
而不是高阶求解器本身。

输出 results/c2_solvers/t1_{method}_s{seed}.json，可中断后续跑。
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "4")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from common.throttle import Throttle  # noqa: E402

spec = importlib.util.spec_from_file_location("sample_grid", ROOT / "code/c2_solvers/sample_grid.py")
sg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sg)
ev, device, v = sg.ev, sg.device, sg.v


@torch.no_grad()
def solve_e(net, x, grid, solver):
    """heun_e：前 N − 1 步 Heun，最后一步 Euler。rk4_e：前 N − 1 步 RK4，最后一步用中点法（两次调用，都在 t < 1）。"""
    g = grid.tolist()
    for j, (a, b) in enumerate(zip(g[:-1], g[1:])):
        h, last = b - a, j == len(g) - 2
        if solver == "heun_e":
            k1 = v(net, x, a)
            if last:
                x = x + h * k1
            else:
                x = x + h / 2 * (k1 + v(net, x + h * k1, b))
        else:
            k1 = v(net, x, a)
            if last:
                x = x + h * v(net, x + h / 2 * k1, a + h / 2)
            else:
                k2 = v(net, x + h / 2 * k1, a + h / 2)
                k3 = v(net, x + h / 2 * k2, a + h / 2)
                k4 = v(net, x + h * k3, b)
                x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["icfm", "otcfm"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--duty", type=float, default=0.5)
    p.add_argument("--min-free-gb", type=float, default=1.5)
    args = p.parse_args()
    th = Throttle(args.duty, min_free_gb=args.min_free_gb, threads=int(os.environ["OMP_NUM_THREADS"]))
    out = ROOT / f"results/c2_solvers/t1_{args.method}_s{args.seed}.json"
    res = json.loads(out.read_text()) if out.exists() else {"method": args.method, "seed": args.seed, "n_fid": args.n}

    def save():
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=2))
        os.replace(tmp, out)

    net = ev.load_mnist(args.method, args.seed)
    real = ev.real_images("mnist")
    shape = (1, 28, 28)

    # 任务 1
    th.reset()
    g = torch.Generator().manual_seed(5)
    x1 = real[torch.randperm(len(real), generator=g)[:2000]].to(device)
    x0 = torch.randn(x1.shape, generator=g).to(device)
    def rms_chunked(fn, n, b=250):
        """分批算每像素 RMS，避免一次送 2,000 张图占满显存。"""
        s = 0.0
        for i in range(0, n, b):
            s += float(fn(slice(i, i + b)).pow(2).sum())
            th.step()
        return (s / (n * 784)) ** 0.5

    with torch.no_grad():
        diag = {"rms_x": float(x1.pow(2).mean().sqrt())}
        diag["real_rms_v1_minus_x"] = rms_chunked(lambda s: v(net, x1[s], 1.0) - x1[s], len(x1))
        for t in (0.9, 0.98, 0.99, 0.999):
            diag[f"real_rms_v{t}_minus_target"] = rms_chunked(
                lambda s: v(net, t * x1[s] + (1 - t) * x0[s], t) - (x1[s] - x0[s]), len(x1))
        xa = torch.randn(256, *shape, generator=torch.Generator().manual_seed(300)).to(device)
        ya, _, tk, kappa = sg.reference(net, xa, 1e-5)  # 和 C2 相同的 A 组：401 个时刻的 κ，按曲率分配的网格与 C2 完全一致
        diag["gen_rms_v1_minus_x"] = float((v(net, ya, 1.0) - ya).pow(2).mean().sqrt())
        # 生成图的终点附近：v(t, y) 在 t = 0.99 → 1 之间变了多少（每像素 RMS / Δt）
        diag["gen_dv_dt_0.99_to_1"] = float((v(net, ya, 1.0) - v(net, ya, 0.99)).pow(2).mean().sqrt() / 0.01)
        diag["gen_dv_dt_0.9_to_0.91"] = float((v(net, ya, 0.91) - v(net, ya, 0.9)).pow(2).mean().sqrt() / 0.01)
    th.step()
    res["t1_diagnostic"] = diag
    save()
    print("diagnostic", {k: round(x, 4) for k, x in diag.items()}, flush=True)

    # 任务 2、3：终点误差（B 组，和 C2 相同的噪声和参照解）
    xb = torch.randn(256, *shape, generator=torch.Generator().manual_seed(301)).to(device)
    yb, _, _, _ = sg.reference(net, xb, 1e-5)
    settings = [("heun_e", n, name) for n in (2, 4, 8, 16) for name in ("uniform", "sqrt_kappa")]
    settings += [("rk4_e", n, "uniform") for n in (2, 4, 8)]
    err = res.setdefault("endpoint_error", {})
    for solver, n, name in settings:
        nfe = 2 * n - 1 if solver == "heun_e" else 4 * n - 2
        key = f"{solver}|{nfe}|{name}"
        grid = sg.shift_grid(n, 1.0) if name == "uniform" else sg.kappa_grid(n, tk, kappa)
        if key not in err:
            err[key] = sg.rms(solve_e(net, xb, grid, solver), yb)
            save()
        print(f"endpoint {key:28s} {err[key]:.4f}", flush=True)

    meter = ev.FID("mnist", real)
    fids = res.setdefault("fid", {})
    for solver, n, name in settings:
        nfe = 2 * n - 1 if solver == "heun_e" else 4 * n - 2
        key = f"{solver}|{nfe}|{name}"
        if key in fids:
            continue
        grid = sg.shift_grid(n, 1.0) if name == "uniform" else sg.kappa_grid(n, tk, kappa)
        t0 = time.time()
        torch.manual_seed(100)
        th.reset()

        def gen(i, b):
            x = solve_e(net, torch.randn(b, *shape, device=device), grid, solver)
            th.step()
            return x

        fids[key] = meter.score(gen, args.n)
        save()
        print(f"FID {key:28s} {fids[key]:7.2f} ({time.time() - t0:.0f}s)", flush=True)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
