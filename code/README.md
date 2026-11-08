# 代码说明

所有命令都在 `cfm_project` 根目录下、`conda activate flow` 之后运行。

```
code/
├── common/                 两人共用（改动前跟对方说一声）
│   ├── data_2d.py          2D 数据集：moons / 8gaussians / swissroll / checkerboard（已标准化）
│   ├── models.py           FM 与 DDPM 共用的 MLP（时间 t 统一缩放到 [0,1]）
│   ├── metrics_2d.py       W2（POT 精确 EMD）、MMD（多带宽 RBF）、轨迹直线度
│   ├── plotting.py         样本 / 轨迹图
│   └── experiment.py       参数、种子、按 NFE 评估、保存 json/png/ckpt
└── e1_2d/
    ├── train_fm.py         成员 A：icfm / otcfm / fm_lipman，Euler ODE 采样
    ├── train_ddpm.py       成员 B：DDPM 训练；ddpm（祖先采样，1000 步）/ ddim（确定性，少步）
    └── summarize.py        汇总所有结果 → results/e1/summary.csv + w2_vs_nfe_*.png
```

## 图像实验（Part V）

```bash
python code/e2_mnist/train_mnist.py --method icfm --seed 0    # 也跑 otcfm、ddpm，每个约 33 分钟
python -u code/e4_cifar/eval_images.py --data mnist --n 10000   # FID、样本网格等，可中断后续跑
python slides/make_ot_numbers.py                                # 第 27 页的配对数字，只用 CPU
python slides/make_image_figs.py slides/figs                     # 第 3、24–28 页的图
```

第二个训练种子和 E3（课程方法：k-means + χ²）：

```bash
python code/e2_mnist/train_mnist.py --method ddpm --seed 1    # 三个模型都用 --seed 1 再训练一次
python -u code/e3_kmeans/cluster_coverage.py --seeds 0 1        # E3：生成图在真实图 k-means 簇上的比例，第 28 页
python -u code/e3_kmeans/fid_seed.py --seed 1                   # 种子 1 的 FID，第 26、27 页
```

两个 E3 脚本默认限速（`--duty 0.5`，显卡超过 72°C 或可用内存不足时暂停，见 `code/common/throttle.py`），
跑实验时电脑仍可使用；`--duty 1` 不限速。限速不改变结果。

每个脚本开头都写了要完成的任务。结果在 `results/e2/`、`results/e3/` 和 `results/e4/`。

## 运行

```bash
python code/e1_2d/train_fm.py --method otcfm --data moons --seed 0
python code/e1_2d/train_ddpm.py --method ddim --data moons --seed 0
python code/e1_2d/summarize.py
```

常用参数：`--iters 20000 --batch 256 --lr 1e-3 --width 256 --n-eval 2000 --nfes 1 2 5 10 20 50 100`；
FM 另有 `--sigma`（默认 0）；DDPM 另有 `--T 1000 --schedule linear|cosine`。

在 RTX 5070 Ti 上，每个 run 约几分钟。W2 的完整网格（4 数据集 × 5 方法 × 3 种子）可以用循环批量跑，例如：

```bash
for d in moons 8gaussians swissroll checkerboard; do for s in 0 1 2; do
  for m in icfm otcfm fm_lipman; do python code/e1_2d/train_fm.py --method $m --data $d --seed $s; done
  for m in ddim ddpm; do python code/e1_2d/train_ddpm.py --method $m --data $d --seed $s; done
done; done
```

## 输出（`results/e1/`）

- `<method>_<data>_s<seed>.json`：配置、训练时间、每个 NFE 下的 W2 / MMD / 直线度 / 采样时间，以及 `reference`（两份独立真实样本之间的 W2/MMD，即有限样本下指标的下限，判断“够不够好”时用它作参照）
- `<method>_<data>_s<seed>.png`：真实数据、各 NFE 生成样本、轨迹
- `ckpt/*.pt`：模型权重

## 公平对比的约定

- FM 与 DDPM **用同一个 MLP、同样的迭代数 / batch / lr**，评估时用**同一份噪声 x0 和同一份测试集**。
- DDPM 与 DDIM 训练过程完全一样（同一个 ε-预测目标），只是采样器不同。
- 直线度只对确定性采样器（FM 的 ODE、DDIM）有意义；DDIM 的时间网格是按离散步编号均匀切分的，写 report 时要说明。

## W1 要读懂的点

- **A**：`train_fm.py` 中 `sample_location_and_conditional_flow` 返回的 `(t, xt, ut)` 分别是什么？去 `conditional-flow-matching-main/torchcfm/conditional_flow_matching.py` 对照论文公式；OT-CFM 只多了一步 minibatch OT 重新配对。
- **B**：`train_ddpm.py` 中 DDPM 后验均值和方差、DDIM 更新式分别对应论文哪个公式？为什么 DDIM 用 1–2 步时会失败（提示：t=T−1 时 abar 很小，x0_pred 要除以 sqrt(abar)）？
