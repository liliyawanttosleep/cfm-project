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

## Final report 的实验（C1–C7，见 `docs/FINAL_REPORT_PLAN.md`）

```bash
python -u code/c2_solvers/sample_grid.py --method otcfm --seed 0        # C2：求解器 × 时间网格，终点误差 + FID
python -u code/c2_solvers/sample_grid.py --method icfm --seed 0
python -u code/c2_solvers/sample_grid.py --method otcfm --quick         # 冒烟测试，2,000 张，写到 *_quick.json
python -u code/c2_solvers/sample_grid.py --method otcfm --seed 1 --core   # C2 复核：其他种子只算均匀和 √κ 网格的 FID
python -u code/c2_solvers/t1_check.py --method otcfm                    # C2 补充：高阶求解器最后一步不在 t = 1 调用网络
python code/c2_solvers/summarize_c2.py                                  # C2 汇总表和图，只用 CPU
python code/c2_solvers/check_derivations.py                             # 推导笔记 D4/D5/D8/D9 的数值核对，只用 CPU

python code/e2_mnist/train_mnist.py --method icfm --seed 2              # C1：第三个种子（设定和种子 0、1 相同）
python -u code/c3_otbatch/pair_cost.py                                 # C3 第 1 步：OT batch 16–4,096 能省多少配对代价，只用 CPU
python code/e2_mnist/train_mnist.py --method otcfm --seed 0 --ot-batch 2048   # C3：在 2,048 个样本上求 OT 计划
python -u code/e3_kmeans/fid_seed.py --seed 0 --tag _otb2048 --methods otcfm --dopri   # C3/C4：评估变体模型（文件名后缀 --tag）
python code/e2_mnist/train_mnist.py --method icfm --seed 0 --t-dist logitnormal   # C4（kappa 需要先跑 C2）
python -u code/e3_kmeans/fid_seed.py --seed 2 --methods icfm otcfm --dopri   # C1：种子 2 的 FID，加 dopri5 和路径长度比
python -u code/c1_reliability/fid_noise.py                              # C1：换采样噪声（5 份）和换样本数（2k–20k）时 FID 的波动
python -u code/e3_kmeans/cluster_coverage.py --seeds 0 1 2 --methods icfm otcfm --out results/c1_reliability/kmeans_c1.json   # C1：带每簇计数和两样本 χ²
python code/c1_reliability/summarize_c1.py                              # C1 汇总：种子标准差、改进门槛、TV 的 bootstrap 区间，只用 CPU
```

- 训练脚本现在也默认限速（`--duty 0.5`）；默认参数下的随机数顺序和以前完全相同，所以种子 2 与种子 0、1 可比。
- 变体模型的文件名带后缀（`otcfm_s0_otb2048.pt`、`icfm_s0_tlogitnormal.pt`、`icfm_s0_tkappa.pt`），不会覆盖原模型。
- C2 结果在 `results/c2_solvers/{method}_s{seed}.json`：`kappa`（κ(t) 和 D5 的 G）、`endpoint_error`（A 组校准 / B 组测试）、`calibrated`（每个求解器和 NFE 选中的网格）、`fid`。

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
