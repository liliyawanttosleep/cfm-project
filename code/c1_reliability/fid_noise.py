"""C1 steps 2 and 3: how much does FID move when only the sampling noise or the sample count changes?

    python -u code/c1_reliability/fid_noise.py --duty 0.5

Tasks
    Task 1  Sampling-noise error: seed-0 I-CFM and OT-CFM, Euler NFE 2 / 10 / 100, five more noise seeds (101-105),
            10k samples each. Together with noise seed 100 (results/e4/mnist_eval.json) this gives 6 values per cell.
    Task 2  Sample-size effect: NFE 100, noise seed 100, 20k samples. FID on the first 2k / 5k / 10k / 20k
            (the first 10k are the same images as the standard 10k number, which checks the setup), and on the
            disjoint 2k and 5k blocks to show the spread at small n.
    Task 3  FID is biased upward by roughly c / n. Fit FID(n) = FID_inf + c / n on the nested values and report FID_inf.
    Task 4  Throttle (code/common/throttle.py), same as the other scripts.

Why: every later "improvement" is compared with an error bar. Training-seed spread is measured by fid_seed.py;
this script measures the two cheaper sources of error, so the report can say which one dominates.

Results are written to results/c1_reliability/fid_noise.json after each item; a rerun skips items already there.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):  # Task 4: limit threads before importing numpy
    os.environ.setdefault(_k, "4")
import numpy as np
import torch
from cleanfid import fid as cfid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from common.throttle import Throttle  # noqa: E402

spec = importlib.util.spec_from_file_location("eval_images", ROOT / "code/e4_cifar/eval_images.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

OUT = ROOT / "results/c1_reliability/fid_noise.json"
NOISES = [101, 102, 103, 104, 105]
NFES = [2, 10, 100]


def frechet(f, meter):
    return float(cfid.frechet_distance(f.mean(0), np.cov(f, rowvar=False), meter.mu, meter.sigma))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["icfm", "otcfm"])
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--n-big", type=int, default=20000)
    p.add_argument("--duty", type=float, default=0.5)
    p.add_argument("--max-temp", type=int, default=72)
    p.add_argument("--resume-temp", type=int, default=65)
    p.add_argument("--min-free-gb", type=float, default=3.0)
    args = p.parse_args()
    th = Throttle(args.duty, args.max_temp, args.resume_temp, args.min_free_gb, threads=int(os.environ["OMP_NUM_THREADS"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(OUT.read_text()) if OUT.exists() else {"n": args.n, "noise_seeds": NOISES, "noise": {}, "sample_size": {}}
    base = json.loads((ROOT / "results/e4/mnist_eval.json").read_text())["by_method"]

    def save():
        tmp = OUT.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=2))
        os.replace(tmp, OUT)

    real = ev.real_images("mnist")
    meter = ev.FID("mnist", real)
    shape = real.shape[1:]
    for m in args.methods:
        net = ev.load_mnist(m, 0)

        def gen(k):
            def g(i, b):
                x = ev.euler(net, torch.randn(b, *shape, device=ev.device), k)
                th.step()  # throttling does not touch the random stream
                return x
            return g

        # Task 1
        rec = res["noise"].setdefault(m, {})
        for k in NFES:
            cell = rec.setdefault(str(k), {"100": base[m]["fid"][str(k)]})
            for s in NOISES:
                if str(s) in cell:
                    continue
                t0 = time.time()
                torch.manual_seed(s)
                th.reset()
                cell[str(s)] = meter.score(gen(k), args.n)
                save()
                print(m, "nfe", k, "noise", s, round(cell[str(s)], 2), f"{time.time() - t0:.0f}s", flush=True)
            v = np.array(list(cell.values()))
            print(m, "nfe", k, "mean", round(v.mean(), 2), "sd", round(v.std(ddof=1), 3), flush=True)

        # Tasks 2 and 3
        if m not in res["sample_size"]:
            t0 = time.time()
            torch.manual_seed(100)  # same noise as the standard 10k number
            th.reset()
            f = meter.features(gen(100), args.n_big)
            nested = {str(n): frechet(f[:n], meter) for n in (2000, 5000, 10000, 20000) if n <= len(f)}
            blocks = {str(n): [frechet(f[i:i + n], meter) for i in range(0, len(f) - n + 1, n)] for n in (2000, 5000)}
            ns = np.array([int(n) for n in nested], float)
            A = np.stack([np.ones_like(ns), 1 / ns], 1)
            fid_inf, c = np.linalg.lstsq(A, np.array(list(nested.values())), rcond=None)[0]
            res["sample_size"][m] = {"nested": nested, "blocks": blocks, "fid_inf": float(fid_inf), "c": float(c),
                                     "check_10k_vs_eval": [nested.get("10000"), base[m]["fid"]["100"]]}
            save()
            print(m, "sample size", {k: round(v, 2) for k, v in nested.items()}, "FID_inf", round(fid_inf, 2),
                  f"{time.time() - t0:.0f}s", flush=True)
        del net
        torch.cuda.empty_cache()
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
