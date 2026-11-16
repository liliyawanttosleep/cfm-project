"""E3 · 用课程方法（k-means 聚类）检验生成的 MNIST 是否覆盖均衡。

    python -u code/e3_kmeans/cluster_coverage.py --seeds 0 1

要完成的任务
    任务 1  只用真实图拟合 k-means，这样簇的定义不受任何生成模型影响
    任务 2  三个模型、两个训练种子、各种 NFE 下生成 n 张图，统计落在各簇的比例
    任务 3  和真实图的比例比较：总变差距离 TV、χ² 检验、效应量 w = sqrt(χ²/n)。n 上万时 χ² 几乎总是显著，所以主要看 TV 和 w
    任务 4  真实测试集走同样的流程，作为基线：一个完美生成器大约能到多少
    任务 5  同一模型换噪声重新生成，看抽样波动；再和换训练种子的差别对比
    任务 6  用在真实数据上训练的 SVM 给生成图打标签，看具体是哪个数字偏多或偏少。这只用来解释，主结果不依赖它
    任务 7  稳健性：换 k-means 的初始化，以及 k = 20
    任务 8  限速（code/common/throttle.py）：显卡只忙一部分时间，过热或内存不足时暂停，跑实验时电脑仍可正常使用

理论目标：FID 只给一个分数，而且特征来自 ImageNet。这里不用 ImageNet 特征，也不用标签，
回答一个 FID 回答不了的问题：生成的数字是否比例均衡，有没有某些写法缺失或过多。局限：簇不等于数字，也看不出笔画是否清晰。

结果每算完一项就写入 results/e3/kmeans.json，中断后再跑会跳过已有的项。输出供 slides/make_image_figs.py 作图（e3）。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):  # 任务 8：在 import numpy 之前限制线程
    os.environ.setdefault(_k, "4")
import numpy as np
import torch
from scipy.stats import chi2_contingency, chisquare
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from torchcfm.models.unet import UNetModel
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from common.throttle import Throttle  # noqa: E402

OUT = ROOT / "results" / "e3"
NFES = [1, 2, 5, 20, 100]
REDRAWS = [11, 12]  # 额外的噪声种子，只用于 NFE 100
T = 1000
METHODS = {"otcfm": "OT-CFM", "icfm": "I-CFM", "ddpm": "DDIM"}


@torch.no_grad()
def euler(net, x, k, device):
    for i in range(k):
        x = x + net(torch.full((x.shape[0],), i / k, device=device), x) / k
    return x


@torch.no_grad()
def ddim(net, x, k, device):
    """和 code/e4_cifar/eval_images.py 相同的网格。"""
    abar = torch.cumprod(1 - torch.linspace(1e-4, 0.02, T, dtype=torch.float64), 0).float().to(device)
    idx = torch.linspace(T - 1, 0, k).round().long().tolist()
    for j, i in enumerate(idx):
        ab = abar[i]
        e = net(torch.full((x.shape[0],), i / (T - 1), device=device), x)
        x1 = (x - (1 - ab).sqrt() * e) / ab.sqrt()
        ab2 = abar[idx[j + 1]] if j + 1 < len(idx) else torch.tensor(1.0, device=device)
        x = ab2.sqrt() * x1 + (1 - ab2).sqrt() * e
    return x


def generate(net, step, k, n, seed, device, batch=250, th=None):
    torch.manual_seed(seed)
    out = []
    if th is not None:
        th.reset()
    for _ in range(int(np.ceil(n / batch))):
        x = step(net, torch.randn(batch, 1, 28, 28, device=device), k, device)
        out.append((x * 127.5 + 128).clip(0, 255).to(torch.uint8).cpu().numpy())  # 和存图、算 FID 时相同的截断
        if th is not None:
            th.step()  # 限速不改变随机数，所以结果和不限速时相同
    return np.concatenate(out)[:n].reshape(n, -1) / 255.0


def compare(km, X, ref_counts):
    """任务 3：X 落在各簇的比例，和真实比例比较。

    chi2 / p：拟合优度检验，把真实比例当作已知（E3 报告用的口径）。
    chi2_2s / p_2s：两样本齐性检验（C1），真实比例本身也是 3 万张图上的估计值。
    counts：每簇计数，供 C1 做 bootstrap 区间和噪声下限的核对。"""
    k = km.n_clusters
    counts = np.bincount(km.predict(X), minlength=k)
    ref_share = ref_counts / ref_counts.sum()
    share = counts / counts.sum()
    chi2, p = chisquare(counts, ref_share * counts.sum())
    keep = (counts + ref_counts) > 0
    chi2_2s, p_2s, _, _ = chi2_contingency(np.stack([counts[keep], ref_counts[keep]]), correction=False)
    return {"tv": float(0.5 * np.abs(share - ref_share).sum()), "chi2": float(chi2), "p": float(p), "w": float(np.sqrt(chi2 / counts.sum())),
            "chi2_2s": float(chi2_2s), "p_2s": float(p_2s),
            "max_ratio": float((share / ref_share).max()), "min_ratio": float((share / ref_share).min()),
            "counts": counts.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="训练种子")
    p.add_argument("--tag", default="", help="C3–C5 变体模型的文件名后缀，例如 _reflow（读 {method}_s{seed}{tag}.pt）")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=OUT / "kmeans.json")
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS), help="按这个顺序跑")
    p.add_argument("--duty", type=float, default=0.5, help="显卡忙碌时间的比例，1 = 不限速")
    p.add_argument("--max-temp", type=int, default=72, help="显卡超过这个温度就暂停")
    p.add_argument("--resume-temp", type=int, default=65)
    p.add_argument("--min-free-gb", type=float, default=3.0, help="系统可用内存低于这个值就暂停")
    args = p.parse_args()
    th = Throttle(args.duty, args.max_temp, args.resume_temp, args.min_free_gb, threads=int(os.environ["OMP_NUM_THREADS"]))
    print(f"throttle: duty {args.duty}, pause above {args.max_temp} C until {args.resume_temp} C, min free RAM {args.min_free_gb} GB", flush=True)
    device = torch.device(args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tr = datasets.MNIST(ROOT / "data", train=True, download=False)
    te = datasets.MNIST(ROOT / "data", train=False, download=False)
    X, Y = tr.data.numpy().reshape(-1, 784) / 255.0, tr.targets.numpy()
    Xte, Yte = te.data.numpy().reshape(-1, 784) / 255.0, te.targets.numpy()
    perm = np.random.default_rng(0).permutation(len(X))
    fit, ref = X[perm[:30000]], X[perm[30000:]]  # 一半拟合 k-means，另一半给出真实比例

    # 任务 1、7
    t0 = time.time()
    kms = {"k10": KMeans(10, n_init=10, random_state=0).fit(fit), "k10_init1": KMeans(10, n_init=10, random_state=1).fit(fit), "k20": KMeans(20, n_init=10, random_state=0).fit(fit)}
    refs = {name: np.bincount(km.predict(ref), minlength=km.n_clusters) for name, km in kms.items()}  # 真实图（3 万张）的每簇计数
    print("k-means fitted", f"{time.time() - t0:.0f}s", flush=True)
    # 任务 6
    labeller = SVC(kernel="rbf", C=10, gamma="scale").fit(X[perm[30000:40000]], Y[perm[30000:40000]])
    lab_acc = float(labeller.score(Xte, Yte))
    print("labeller test accuracy", round(lab_acc, 4), flush=True)

    res = json.loads(args.out.read_text()) if args.out.exists() else {}
    res.update({"n": args.n, "fit_images": 30000, "labeller_test_accuracy": lab_acc, "ref_digit_share": (np.bincount(labeller.predict(ref), minlength=10) / len(ref)).tolist(),
                "ref_counts": {name: c.tolist() for name, c in refs.items()}, "runs": res.get("runs", {})})

    def save():
        tmp = args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(res, indent=2))
        os.replace(tmp, args.out)

    def measure(G):
        r = {name: compare(km, G, refs[name]) for name, km in kms.items()}
        r["digit_share"] = (np.bincount(labeller.predict(G), minlength=10) / len(G)).tolist()
        return r

    # 任务 4
    if "real_test" not in res["runs"]:
        res["runs"]["real_test"] = measure(Xte[:args.n])
        print("real_test", res["runs"]["real_test"]["k10"], flush=True)
        save()
    # 任务 2、5
    for m in args.methods:
        step = ddim if m == "ddpm" else euler
        for seed in args.seeds:
            net = UNetModel(dim=(1, 28, 28), num_channels=32, num_res_blocks=1).to(device)
            net.load_state_dict(torch.load(ROOT / f"results/e2/ckpt/{m}_s{seed}{args.tag}.pt", map_location=device, weights_only=True))
            net.eval()
            for k, noise in [(k, 7) for k in NFES] + [(100, s) for s in REDRAWS]:
                key = f"{m}_s{seed}{args.tag}_nfe{k}_noise{noise}"
                if key in res["runs"]:
                    continue
                t0 = time.time()
                G = generate(net, step, k, args.n, noise, device, th=th)
                tg = time.time() - t0
                res["runs"][key] = measure(G)
                save()
                print(key, "tv", round(res["runs"][key]["k10"]["tv"], 3), "w", round(res["runs"][key]["k10"]["w"], 3),
                      f"{time.time() - t0:.0f}s (generate {tg:.0f}s)", flush=True)
            del net
            torch.cuda.empty_cache()
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
