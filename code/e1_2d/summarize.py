"""E1 汇总。

要完成的任务：读 results/e1/*.json，输出“方法 × 数据集 × NFE”的指标表（多个种子取均值 ± 标准差），
并画 W2 vs NFE 曲线。第 19 页的数字来自这里的 results/e1/summary.csv。

用法：python code/e1_2d/summarize.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from common.experiment import RESULTS
from common.plotting import plt

rows = []
for f in sorted(RESULTS.glob("*.json")):
    r = json.loads(f.read_text())
    for row in r["by_nfe"]:
        rows.append({"method": r["method"], "data": r["data"], "seed": r["seed"], **row,
                     "ref_w2": r["reference"]["w2"], "train_s": r["train_seconds"]})
if not rows:
    sys.exit(f"no results in {RESULTS}")

df = pd.DataFrame(rows)
agg = df.groupby(["data", "method", "nfe"]).agg(
    w2_mean=("w2", "mean"), w2_std=("w2", "std"), mmd_mean=("mmd", "mean"), mmd_std=("mmd", "std"),
    straight=("straightness", "mean"), ref_w2=("ref_w2", "mean"), n_seeds=("seed", "nunique"),
).reset_index()
agg.to_csv(RESULTS / "summary.csv", index=False)
with pd.option_context("display.width", 200, "display.max_rows", 500, "display.float_format", "{:.4f}".format):
    print(agg)

for data, d in agg.groupby("data"):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for method, m in d.groupby("method"):
        ax.errorbar(m["nfe"], m["w2_mean"], yerr=m["w2_std"].fillna(0), marker="o", ms=3, capsize=2, label=method)
    ax.axhline(d["ref_w2"].mean(), ls="--", c="gray", lw=1, label="real vs real")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("NFE (number of function evaluations)")
    ax.set_ylabel("W2 to test set")
    ax.set_title(data)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / f"w2_vs_nfe_{data}.png", dpi=150)
    plt.close(fig)
print(f"saved {RESULTS / 'summary.csv'} and w2_vs_nfe_*.png")
