"""E3 的配套检查：换一个训练种子后，FID 的结论是否还成立。

    python -u code/e3_kmeans/fid_seed.py --seed 1

要完成的任务
    任务 1  用种子 1 的三个模型，按和种子 0 完全相同的做法算 FID：同样的 NFE、同样的 1 万张、同样的噪声种子、同样的参考集
    任务 2  和 results/e4/mnist_eval.json（种子 0）对比，检查 slide 上的说法：
            OT-CFM 在 2–5 步比 I-CFM 低约 40%，100 步时差 4%；100 步时 DDIM 的 FID 最低
    任务 3  限速（code/common/throttle.py），和 cluster_coverage.py 相同

理论目标：之前每个方法只训练过一次，无法区分方法差异和训练之间的波动。两个种子的差距可以粗略估计这个波动，
但两次还算不出可靠的标准差，所以报告时给出两个值，不写 mean ± std。

结果每算完一项就写入 results/e3/fid_s{seed}.json，中断后再跑会跳过已有的项。
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):  # 任务 3：在 import numpy 之前限制线程
    os.environ.setdefault(_k, "4")
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from common.throttle import Throttle  # noqa: E402

spec = importlib.util.spec_from_file_location("eval_images", ROOT / "code/e4_cifar/eval_images.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1, help="训练种子")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--methods", nargs="+", default=["ddpm", "icfm", "otcfm"])
    p.add_argument("--duty", type=float, default=0.5, help="显卡忙碌时间的比例，1 = 不限速")
    p.add_argument("--max-temp", type=int, default=72)
    p.add_argument("--resume-temp", type=int, default=65)
    p.add_argument("--min-free-gb", type=float, default=3.0)
    p.add_argument("--tag", default="", help="C3/C4 变体模型的文件名后缀，例如 _otb2048（读 {method}_s{seed}{tag}.pt，写 fid_s{seed}{tag}.json）")
    p.add_argument("--dopri", action="store_true", help="C1：流模型再加 dopri5 的 FID、平均调用次数和路径长度比")
    args = p.parse_args()
    th = Throttle(args.duty, args.max_temp, args.resume_temp, args.min_free_gb, threads=int(os.environ["OMP_NUM_THREADS"]))
    out = ROOT / f"results/e3/fid_s{args.seed}{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out.read_text()) if out.exists() else {"seed": args.seed, "n": args.n, "fid": {}}

    real = ev.real_images("mnist")
    meter = ev.FID("mnist", real)
    for m in args.methods:
        net = ev.load_mnist(m, f"{args.seed}{args.tag}")  # 文件名是 {method}_s{seed}{tag}.pt
        step = ev.ddim if m == "ddpm" else ev.euler
        rec = res["fid"].setdefault(m, {})
        for k in ev.NFES:
            if str(k) in rec:
                continue
            t0 = time.time()
            torch.manual_seed(100)  # 和 eval_images.py 的任务 3 相同
            th.reset()
            def gen(i, b):
                x = step(net, torch.randn(b, *real.shape[1:], device=ev.device), k)
                th.step()  # 限速不改变随机数
                return x

            rec[str(k)] = meter.score(gen, args.n)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(res, indent=2))
            os.replace(tmp, out)
            print(m, k, round(rec[str(k)], 2), f"{time.time() - t0:.0f}s", flush=True)
        if args.dopri and m != "ddpm":
            # C1 第 1、5 步：dopri5 参照和直线度，做法和 eval_images.py 的任务 5、7 相同
            if "dopri5" not in rec:
                nfes = []
                t0 = time.time()
                torch.manual_seed(100)
                th.reset()

                def gen_d(i, b):
                    y, c = ev.dopri(net, torch.randn(b, *real.shape[1:], device=ev.device), 1e-5)
                    nfes.append(c)
                    th.step()
                    return y

                rec["dopri5"] = meter.score(gen_d, args.n)
                res.setdefault("dopri5_mean_nfe", {})[m] = sum(nfes) / len(nfes)
                tmp = out.with_suffix(".tmp")
                tmp.write_text(json.dumps(res, indent=2))
                os.replace(tmp, out)
                print(m, "dopri5", round(rec["dopri5"], 2), "mean NFE", round(res["dopri5_mean_nfe"][m], 1), f"{time.time() - t0:.0f}s", flush=True)
            if m not in res.get("path_ratio", {}):
                torch.manual_seed(201)
                res.setdefault("path_ratio", {})[m] = ev.path_ratio(net, real.shape[1:])
                tmp = out.with_suffix(".tmp")
                tmp.write_text(json.dumps(res, indent=2))
                os.replace(tmp, out)
                print(m, "path ratio", res["path_ratio"][m], flush=True)
        del net
        torch.cuda.empty_cache()
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
