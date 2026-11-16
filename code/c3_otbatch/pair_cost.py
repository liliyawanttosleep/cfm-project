"""C3 step 1: how much does minibatch OT save on MNIST as the OT batch grows? CPU only.

    CUDA_VISIBLE_DEVICES="" python -u code/c3_otbatch/pair_cost.py

Tasks
    Task 1  Mean pair cost E||x0 - x1||^2 under random pairing and under exact OT within a batch,
            for OT batch 16, 64, 128, 256, 512, 1024, 2048, 4096 (same estimator as slides/make_ot_numbers.py).
    Task 2  Compare the saving with the most any coupling can save, 2 tr Sigma^(1/2) (Gelbrich bound, derivations D4),
            and with the total cost. Fit saved(B) = a + b log B to see how slowly it grows, and read off where it
            would reach 10% of the total cost.
    Task 3  Time one exact OT solve at each batch size (the extra training cost of C3 step 2).
    Task 4  Figure figs/c3_saving_vs_batch.png: saving as a share of the total cost and of the bound, log batch axis.

Why: the talk concluded that the OT advantage fades in 784 dimensions. This separates "batch 128 is too small"
from "there is little to save in the first place".

Output results/c3_otbatch/pairs.json (written after every batch size; a rerun skips sizes already there).
"""
import json
import os
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):  # leave cores for the GPU job and the user
    os.environ.setdefault(_k, "4")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import ot  # noqa: E402
import torch  # noqa: E402
from torchvision import datasets  # noqa: E402

torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "c3_otbatch"
SIZES = [(16, 200), (64, 50), (128, 20), (256, 10), (512, 5), (1024, 3), (2048, 2), (4096, 2)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figs").mkdir(exist_ok=True)
    f = OUT / "pairs.json"
    res = json.loads(f.read_text()) if f.exists() else {"pairs": {}}
    x = datasets.MNIST(ROOT / "data", train=True, download=False).data.float().flatten(1) / 127.5 - 1
    if "two_tr_sqrt_cov" not in res:
        eig = np.linalg.eigvalsh(np.cov(x.numpy().astype(np.float64), rowvar=False)).clip(min=0)
        res.update({"tr_cov": float(eig.sum()), "two_tr_sqrt_cov": float(2 * np.sqrt(eig).sum())})
    bound = res["two_tr_sqrt_cov"]

    # Tasks 1 and 3
    for batch, rounds in SIZES:
        if str(batch) in res["pairs"]:
            continue
        g = torch.Generator().manual_seed(batch)
        rnd, opt, secs = [], [], []
        for _ in range(rounds):
            x1 = x[torch.randperm(len(x), generator=g)[:batch]]
            x0 = torch.randn(x1.shape, generator=g)
            M = torch.cdist(x0.double(), x1.double()).pow(2).numpy()
            rnd.append(float(np.trace(M) / batch))
            t0 = time.time()
            opt.append(float(ot.emd2(ot.unif(batch), ot.unif(batch), M, numItermax=10_000_000)))
            secs.append(time.time() - t0)
        r, o = float(np.mean(rnd)), float(np.mean(opt))
        res["pairs"][str(batch)] = {"rounds": rounds, "random": r, "ot": o, "saved": r - o, "share_of_total": (r - o) / r,
                                    "share_of_bound": (r - o) / bound, "ot_seconds": float(np.mean(secs))}
        f.write_text(json.dumps(res, indent=2))
        print(batch, {k: round(v, 4) for k, v in res["pairs"][str(batch)].items()}, flush=True)

    # Task 2
    B = np.array(sorted(int(b) for b in res["pairs"]), float)
    saved = np.array([res["pairs"][str(int(b))]["saved"] for b in B])
    total = np.mean([res["pairs"][str(int(b))]["random"] for b in B])
    b, a = np.polyfit(np.log(B), saved, 1)
    res["fit_log"] = {"a": float(a), "b_per_log": float(b), "saved_per_doubling": float(b * np.log(2)),
                      "batch_for_10pct_of_total": float(np.exp((0.1 * total - a) / b)),
                      "batch_for_bound": float(np.exp((bound - a) / b)), "total_random_cost": float(total)}
    f.write_text(json.dumps(res, indent=2))
    print("fit", res["fit_log"], flush=True)

    # Task 4
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(B, 100 * saved / total, "o-", color="#2f6db3", label="saved / random-pair cost")
    ax.plot(B, 100 * saved / bound, "s--", color="#d1772b", label="saved / most any coupling can save")
    ax.axhline(100 * bound / total, color="0.4", lw=1, ls=":", label=f"bound / random-pair cost ({100 * bound / total:.0f}%)")
    ax.set(xscale="log", xlabel="OT batch size", ylabel="%", title="MNIST: minibatch OT saving vs batch size")
    ax.set_xticks(B, [str(int(v)) for v in B])
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "figs" / "c3_saving_vs_batch.png", dpi=160)
    print("saved", f, flush=True)


if __name__ == "__main__":
    main()
