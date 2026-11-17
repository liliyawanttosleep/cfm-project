# 交接文档（2026-11-07）

> 新对话先读本文件，再按需读：`TEAM_PLAN.md`（开头的“项目全流程”）、`docs/FINAL_REPORT_PLAN.md`（实验 C1–C7）、`report/README.md`（report 格式）、`docs/notes/derivations.md`（公式推导 D1–D9）。
> 取代 10/29 第一次会话的交接文档（旧内容见 git 历史）。

## 0. 工作规则

- 成员：Wentao Yang（A，flow matching 线）、Kaiming Zhang（B，diffusion / 评估线）。
- 和用户用**中文**交流；slide、讲稿、report、图中文字一律**英文**；代码注释可以用中文。
- 只在 `C:\Users\Wentao\Desktop\cfm_project` 工作。实验用 `D:\Anaconda\envs\flow\python.exe`（torch CUDA，RTX 5070 Ti 12 GB）。系统 python 是 CPU 版，不要用它跑实验。
- Git：**只有用户要求时才 commit / push**，走“分支 → PR”流程；仓库 https://github.com/liliyawanttosleep/cfm-project （私有）。
- 不改 TDR 等系统设置。
- GPU 任务一律限速运行（`code/common/throttle.py`，默认 `--duty 0.5`，显卡超过 72°C 暂停、65°C 恢复，可用内存低于阈值时暂停；本机内存占用高，阈值用 1.5 GB）。同一时间只跑一个 GPU 任务。
- 报告不排练，不要再讨论报告时长。

## 1. 研究目标和问题

**课题**：Conditional Flow Matching（CFM）把生成变成回归，采样就是解 ODE，代价是网络调用次数（NFE）。**Final report 的主线：怎样让 CFM 用更少的 NFE 生成好样本，以及怎样可靠地检测生成质量。** Diffusion（DDPM 网络 + DDIM 采样）只作参照。

| # | 问题 |
|---|---|
| RQ1 | 质量检测：怎样可靠地判断一个 CFM 模型更好？FID 的误差有多大？FID 漏掉了什么（类别覆盖）？ |
| RQ2 | 推理端：不重新训练，只换求解器和时间网格，固定 NFE 下能提升多少？ |
| RQ3 | 训练端：OT batch 大小、训练时间分布、reflow 各能改善多少少步质量？训练代价多大？ |
| RQ4 | 诊断：路径直线度能否预测需要多少步？两端的改进能否叠加？ |

## 2. 已确定的结论和决定

**时间节点**：课堂报告 **11/10（周二）**，30 分钟，两人讲；final report **12/16 23:59**，英文，正文约 5 页。

**决定**
- 课程方法只要求**一种**（11/07 再次确认）：用聚类，即 k-means + χ²。
- 报告 PPT 已定稿，不再改结构。
- Final 实验收窄到 CFM 的效率和质量检测（C1–C7），不做新的 diffusion 实验。
- Report 用 NeurIPS 风格 LaTeX，结构为 Introduction / Related Work / Methods / Experiments / Conclusion / References，另加 Individual Contributions（课程硬性要求）。

**实验结论（均为我们自己跑出的结果）**
- 2D（种子 0）：OT-CFM 从 1 步起就接近最好；I-CFM 走 1 步会落在数据均值上；DDIM 在 1–2 步时失败。
- MNIST FID，Euler 步数 NFE = 1 / 2 / 5 / 10 / 20 / 50 / 100，每格是“种子 0 / 种子 1”：

  | 模型 | NFE 1 | 2 | 5 | 10 | 20 | 50 | 100 |
  |---|---|---|---|---|---|---|---|
  | OT-CFM | 223 / 207 | 76.0 / 83.2 | 26.7 / 36.9 | 19.0 / 25.1 | 17.0 / 20.1 | 15.6 / 18.2 | 13.9 / 16.8 |
  | I-CFM | 332 / 318 | 132 / 136 | 45.3 / 37.9 | 27.9 / 22.8 | 20.7 / 17.4 | 16.9 / 15.0 | 14.5 / 12.9 |
  | DDIM | 293 / 356 | 291 / 356 | 99.3 / 106 | 41.0 / 53.0 | 28.4 / 40.1 | 23.1 / 34.4 | 12.2 / 32.8 |

  - 种子 0 用 dopri5 时，OT-CFM 11.5、I-CFM 12.1（约 275 NFE）。
  - 真实 vs 真实的 FID 是 0.54。
- OT-CFM 只在 1–2 步时两个种子都更好：FID 低 33–42%。5 步时两个种子分别低 41% 和 3%；从 10 步起两个种子不一致。
- 在 784 维中，minibatch OT（batch 128）的配对代价只比随机配对低 4%，2D 中低 85–94%。可节省的上限是 2 tr Σ^½ = 386，总代价约 1,509。
- E3（k-means，k = 10，100 步，10,000 张）：
  - DDIM 种子 0：FID 最好（12.2），但类别最不均衡（TV 0.19；数字 1 占 18.7%，3 只占 4.5%）。
  - DDIM 种子 1：类别均衡（TV 0.06），FID 却是 32.8。
  - 结论：只看 FID、只看一个种子都会误判。
  - 真实测试集基线 TV 0.017。所有模型的 TV 是基线的 2.6–11 倍。换采样噪声，TV 最多变 0.012；效应量 w 在 100 步时为 0.11–0.46。
- DDIM 两个种子的训练损失几乎相同（0.0215 / 0.0213），FID 却差很多：训练损失不能用来比较模型质量。

## 3. 当前版本的关键内容

**方法**
- I-CFM：独立配对 x₀ ~ N(0, I)、x₁ ~ data；x_t = (1 − t)x₀ + t·x₁；回归目标是 x₁ − x₀；σ = 0。
- OT-CFM：在每个 batch 内用精确 OT 重新配对。
- DDPM：线性 β，T = 1000，ε 预测；采样用 DDIM（η = 0）。
- 记号：t = 0 是噪声，t = 1 是数据。

**MNIST 设定（三个模型完全相同）**
- 网络：torchcfm U-Net，32 通道，1.08M 参数。
- 训练：50 epoch（23,400 步），batch 128，Adam 默认学习率，无 EMA；每个模型约 33 分钟（不限速）。

**评估协议**
- FID：clean-fid，10,000 张，参考集是 MNIST 训练集，采样噪声 `torch.manual_seed(100)`。
- 类别覆盖：k-means 只在 30,000 张真实图上拟合，生成图分到最近的中心；报告 TV、χ²、w = √(χ²/n)。
- 数字标签：用 RBF-SVM（准确率 97%）给生成图打标签，只用来说明“哪个数字偏多或偏少”。
- 直线度：路径长度比（100 步 Euler 走过的路径长度 / 首尾直线距离）。

**代码**
- `code/e1_2d/`：2D 训练（`train_fm.py`、`train_ddpm.py`）和汇总。
- `code/e2_mnist/train_mnist.py`：`--method icfm|otcfm|ddpm --seed N`。
- `code/e4_cifar/eval_images.py`：MNIST / CIFAR 的 FID、一步行为、路径长度比。
- `code/e3_kmeans/cluster_coverage.py`：E3 的 k-means 实验。
- `code/e3_kmeans/fid_seed.py`：其他种子的 FID。
- `code/common/throttle.py`：限速。
- 结果：`results/e1/`、`results/e2/`、`results/e3/`、`results/e4/`。权重、日志、FID 统计量不进 git。

**报告材料**
- `slides/FM_vs_Diffusion_Talk.pptx`：37 页（正文 31 + B1–B6），带讲者备注，已用 PowerPoint 逐页检查。
- `slides/FM_vs_Diffusion_Talk.pdf`、讲稿 `slides/FM_vs_Diffusion_Speech_Script.docx`。
- slide 源文件：`slides/deck_v2/project/`。
- 生成方法：
  - PDF：`slides/build_pdf.py`；
  - PPT：`slides/build_pptx.py`，用 `D:\Anaconda\python.exe`（3.13）运行，因为 flow 环境缺 lxml；
  - 讲稿：`slides/speech/`；
  - 图：`slides/make_image_figs.py`。
- 在线 slide：https://claude.ai/artifact/7Lpjq76Dw2ywYrmNCLgUDR
- 在线讲稿：https://claude.ai/artifact/SsCxbkUcGE1DdD8HQvhD8r

## 4. 被否定的方案和原因

| 方案 | 原因 |
|---|---|
| TSTR（在生成图上训练、在真实图上测试，逻辑回归） | 预实验中所有模型的准确率都是 0.89–0.90，和真实数据（0.90）一样，区分不了模型 |
| real-vs-fake 分类器（随机森林） | 所有模型都被识别出 89–95%，主要靠背景像素不是纯黑，不反映生成质量 |
| PCA + Fréchet 距离 | 能用，但又是一个和 FID 类似的单一分数，看不出缺了哪类数字 |
| 四类课程方法都做 | 课程只要求一种 |
| 第一版实验计划 R1–R7（包括 DDPM 1000 步、DDIM 网格消融、Fashion-MNIST、补齐 2D 三个种子、GMM / KDE 基线） | 偏离主线；收窄为 CFM 的效率和质量检测（C1–C7） |
| EMA、cosine 调度、classifier-free guidance、MNIST-FID | 和少步效率无关，或需要额外模型 |
| Consistency model、MeanFlow | 训练目标不同，超出 CFM 框架和时间预算，只在 Related Work 中提及 |
| VAE / GAN 四模型对比、表格数据合成 | 选题阶段否定：范围太宽 / 网络部分太少 |

## 5. 下一步待办

**Git**
- PR #3（报告定稿、C1–C7 计划、项目全流程、report 格式、推导笔记 D1–D8）已于 11/08 合并进 main。
- 推导笔记 D9 和本文件的更新走新分支 `update/derivations-d9` 的 PR。
- `slides/FM_vs_Diffusion_Talk.pptx` 本地被 PowerPoint 重新保存过，没有提交；需要时由用户决定是否提交。

**报告前（11/08–11/10）**
- [ ] 用 PowerPoint 放映（F5），确认第 3、15 页的动图在动
- [ ] 回复在线讲稿上的评论 `1ef6e105-1ea5`
- [ ] 确认 slide 3 的措辞 “Only the data and the network change”
- [ ] Kaiming 过一遍自己负责的页和讲稿

**Final 实验（11/11–12/02，详见 `docs/FINAL_REPORT_PLAN.md`）**
- **先做（必做）**
  - C1：I-CFM、OT-CFM 训练第 3 个种子；测采样噪声和样本数带来的 FID 误差；TV 的 bootstrap 区间。
  - C2：Euler / Heun / RK4 × 均匀 / shift / 按曲率分配 / 校准网格，不需要训练。1D 核对显示按曲率分配的网格在目标平滑时略好、目标集中时更差，所以实际方法是校准网格（推导和核对见 `docs/notes/derivations.md` D5）。
- **再做**
  - C5 reflow（必做）。
  - C3 OT batch 128–4,096（应做）。
  - C4 训练时间分布：logit-normal 和按曲率加权（应做）。
  - C6 综合分析（必做）。
  - C7 CIFAR-10 验证（选做）。
- **代码（11/09 已完成）**
  - `train_mnist.py` 加了 `--ot-batch`（C3）、`--t-dist logitnormal|kappa`（C4）和限速；默认参数下的随机数顺序不变。
  - 新建 `code/c2_solvers/sample_grid.py`（C2）。
  - `cluster_coverage.py` 额外保存每簇计数、真实图计数和两样本 χ²（C1）。
- **C2 结果（11/10）**：OT-CFM 已全部跑完；I-CFM 主要格子已完成，其余的固定 shift 正在补跑。详见 FINAL_REPORT_PLAN 的 C2“结果”和 `results/c2_solvers/summary.md`。
  - 按曲率分配的网格（√κ）在 8 次调用以上一致地降低 FID。OT-CFM 用 Euler 32 次调用时 FID 12.5，dopri5 要约 270 次调用才到 11.5。
  - 高阶求解器离不开 √κ 网格：均匀网格上，Heun、RK4 比 Euler 差得多。
  - 终点误差预测不了 FID：按终点误差校准选出的网格，FID 反而更差。所以实际采用的方法改为 √κ，而不是校准网格。
- **11/10 晚的后台任务链**（依次运行，限速）：
  1. `t1_check.py`（OT-CFM、I-CFM）：检验 t = 1 假设；
  2. 补完 I-CFM 的 C2；
  3. C1 第三个种子的训练（I-CFM、OT-CFM 的 `--seed 2`）。
  - 日志在 `results/c2_solvers/logs/` 和 `results/c1_reliability/logs/`。
  - 用户的 MuMu 模拟器在用显卡时温度 80°C，限速模块会让任务暂停，进度会慢。
- **11/11 重新排队**：11/10 晚的任务链在 MuMu 占用显卡时暂停，之后两个会话的后台进程都已结束，没有新结果（I-CFM 的 C2 仍停在 11/10 16:38）。11/11 按同样顺序重新开始：`t1_check.py`（OT-CFM、I-CFM）→ 补完 I-CFM 的 C2 → C1 第三个种子的训练。
- **11/11 结果**（任务链 08:24–13:05 跑完，中间没有中断）：
  - t = 1 假设成立：网络在 t = 1 处的相对误差在真实图上约 43%，在生成图上约 85%。Heun / RK4 的最后一步不在 t = 1 调用网络后，均匀网格 32 次调用的 FID 从 50.4 / 40.7 降到约 16（OT-CFM）。NFE@20：OT-CFM 9.2 → 6.6，I-CFM 23.7 → 9.9。详见 FINAL_REPORT_PLAN 的 C2。
  - I-CFM 的 C2 已补完，`summarize_c2.py` 已重跑，表和图包含 heun_e / rk4_e。
  - 种子 2 的 I-CFM、OT-CFM 已训练完（各 50 个 epoch，约 70 分钟）。
  - 11/11 13:07 起的 C1 任务链（依次运行，限速）：
    1. `fid_seed.py --seed 2 --dopri`：Euler 1–100 步、dopri5、路径长度比；再给种子 1 补 dopri5 和路径长度比。结果在 `results/e3/fid_s{1,2}.json`。
    2. `code/c1_reliability/fid_noise.py`：种子 0 在 NFE 2 / 10 / 100 换 5 份采样噪声；NFE 100 时 2k–20k 张的 FID 和 1/n 外推。
    3. `cluster_coverage.py --seeds 0 1 2 --methods icfm otcfm`，写到 `results/c1_reliability/kmeans_c1.json`，带每簇计数。
    - 日志都在 `results/c1_reliability/logs/`。跑完后运行 `summarize_c1.py`，生成 `results/c1_reliability/summary.md` 和改进门槛。
  - 第 1 步 14:58 跑完：I-CFM 种子 2 明显更差（100 步 FID 23.7，种子 0、1 是 14.5 / 12.9），所以 I-CFM 的门槛很大（11.6）；OT-CFM 在 1–5 步时三个种子都优于 I-CFM；直线度在种子之间很稳定。详见 FINAL_REPORT_PLAN 的 C1“结果”。
  - 第 2–4 步 19:20 跑完：采样噪声的 FID 标准差只有 0.05–0.6，比训练种子小 10–100 倍；10k 张的 FID 比外推值高约 0.9；两样本 χ² 下真实测试集不再被拒绝；TV 门槛约 0.05。C1 全部完成，详见 FINAL_REPORT_PLAN 的 C1“结果”。
  - 之后排队的是 C2 复核：种子 1、2 的 `sample_grid.py --core` 和 `t1_check.py`（OT-CFM、I-CFM），日志在 `results/c2_solvers/logs/{method}_s{1,2}.log` 和 `t1_{method}_s{1,2}.log`。
  - C3 第 1 步（CPU）19:22 跑完：OT batch 从 16 到 4,096，省下的代价只从 2.9% 升到 6.1%，每翻一倍多省约 0.4%。详见 FINAL_REPORT_PLAN 的 C3“结果”。
  - C2 复核之后排队的是 C3 第 2、3 步：OT-CFM `--ot-batch 512`、`2048` 的训练和评估，日志在 `results/c3_otbatch/logs/`。

- **11/14 重新开始**：11/11 20:02 电脑意外关机（Kernel-Power 41），当时的任务链只完成了 OT-CFM 种子 1 的 C2 复核。11/14 用 `code/run_queue.sh` 重新排队，依次是：
  1. C5 reflow（I-CFM 种子 0）：`code/c5_reflow/make_pairs.py` 生成 60,000 对配对 → `train_mnist.py --reflow`（从 I-CFM 权重微调 50 epoch）→ `fid_seed.py --tag _reflow --dopri`；
  2. C2 种子 1、2 复核（`--core` 和 `t1_check.py`）；
  3. C3：OT batch 512、2,048 的训练和评估；
  4. C4：logit-normal、按 κ 加权的 t 分布（I-CFM 种子 0）的训练和评估；
  5. C5 第 4 步：从 OT-CFM 出发的 reflow。
  - 每一步都能中断后续跑：脚本内部会跳过已完成的部分，训练完成的模型会被整步跳过。电脑再关机时，重新运行 `bash code/run_queue.sh` 即可。
  - 冒烟测试中 100 对 reflow 配对的平均代价是 1,316，随机配对约 1,510，minibatch OT 在 batch 4,096 时是 1,418。
- **Report 初稿（11/14）**：`report/` 已整理成完整的写作文件夹（结构见 `report/README.md` 开头）。正文分节放在 `report/sections/`；附录 A 是完整推导（D1–D9），附录 B 是课程方法（k-means + 两样本 χ²），附录 C 是补充结果，附录 D 是设定和算力。图由 `make_figs.py` 生成，共 6 张，包括样本展示图和 2D 路径图；表由 `make_tables.py` 直接从 `results/` 生成。每张图表的数据来源、文献数字的出处和课程要求的对照都写在 `report/materials/README.md`。编译用 `bash report/build.sh`。还在跑的结果用红色 `\pending{}` 标出，目前剩 11 处；正文约 9 页，结果齐了以后压缩到约 5 页。Individual Contributions 要两人按实际分工填写。

- **11/15 凌晨的进度和 report 更新**
  - 已跑完：C5 reflow（I-CFM）；C2 种子 1、2 的复核（只差 I-CFM 种子 2 的 `t1_check`，02:23 开始运行）；reflow 模型的类别覆盖（CPU，只差最后一份噪声）。
  - 排队中：C3 OT batch 512 / 2,048 → C4 t 分布 → OT-CFM 的 reflow → reflow 模型上的 C2（`code/run_queue.sh`）。
  - Report：
    - 新增附录表 `tab_seeds`（由 `make_tables.py` 生成）：三个种子上，C2 的每个结论方向都一致。正文 4.3 节和附录 C 已改写。
    - reflow 的类别覆盖已写进 4.4 节：TV 在 1–100 步都是 0.066–0.078，多步时和原模型的差别在门槛之内。
    - 红色 `\pending{}` 从 11 处减到 7 处，剩下的是：C3、C4、4.5 节综合、结论、Individual Contributions（要两人填写）、I-CFM 种子 2 的最后一步变体、算力总计。
    - 现在是 21 页（正文约 9 页），结果齐了以后压缩到约 5 页。

- **11/15 13:35 电脑第二次意外断电**（Kernel-Power 41，13:57 重启；第一次是 11/11），两次都发生在 GPU 任务运行期间。
  - 断电前跑完的：OT batch 512 的训练和 FID（05:10）；I-CFM 种子 2 的 `t1_check`。
  - 断电时 OT batch 2,048 训练到第 21 个 epoch，存档是第 20 个 epoch。
  - 13:59 重新运行 `bash code/run_queue.sh`，日志在 `results/run_queue_1115.log`。已完成的步骤会自动跳过，OT batch 2,048 从第 20 个 epoch 接着训练。
  - OT batch 512 的结果：1 步 FID 195.5（原模型 223.1），5 步以上每个 NFE 都略差（+2.6 到 +5.2），只有 dopri5 超过门槛，路径长度比 1.0098。结论是在种子波动之内，没有改进。
  - `report/` 由“CFM project: finish Pre slides”会话负责修改，这些数字已发给它写入。

- **11/16 05:23 GPU 队列全部跑完**（各步都正常退出）。门槛 = 三个训练种子 FID 标准差的 2 倍。
  - **C3 OT batch 2,048**：没有改进。1 步 FID 208.1，落在原模型三个种子的范围 205–223 之内；2 步以上比种子 0 差 3–11，只有 dopri5 超过门槛。路径最直（R = 1.0085），loss 最低（0.100 对 0.118），但都没有反映到 FID 上。训练中途因断电续跑，总共 2.37 h（其中 OT 求解占 20%）；`*_train.json` 里的 `seconds` 只记了续跑后的 30 个 epoch。
  - **C4 logit-normal t**：每个预算下都更好。2 步 FID 从 131.9 降到 102.5（超过门槛）；20 步从 20.7 降到 14.1；从 2 步到 100 步，每个 NFE 都好于原模型的全部三个种子；NFE@20 从 23.7 降到 9.1。但只有一个训练种子，按严格规则只算 2 步的改进。dopri5 只要 111 NFE，FID 14.2，反而比它自己 50 步的 12.0 差。
  - **C4 按 κ 加权的 t**：没有超过门槛的差别；R = 1.024，dopri5 要 345 NFE。结论：采样时应把步数放在 t≈1 附近，训练时却不应把样本集中到那里。
  - **C5 从 OT-CFM 出发的 reflow**：1 步和 2 步 FID 是 61.2 / 28.9（原模型 223 / 76），20 步以上变差且超过门槛（dopri5 21.1 对 11.5），和 I-CFM 的 reflow 几乎一样。生成配对 1.84 h，重新训练 1.56 h，共约 2.8 倍原模型的训练时间。这个模型没有算类别覆盖。
  - **C6 reflow 模型上的 C2**：√κ 网格从 4–8 步起改善 0.7–2.4。均匀网格上的 Heun 和 RK4 在几乎是直线的 reflow 模型上仍然失败（32 NFE：46.2 / 34.6，Euler 是 21.7 / 21.9）。综合下限：1–4 NFE 时 reflow 最好，约 5–8 NFE 以上原模型加 Heun* 最好。
  - **R 和 1 步 FID**：12 个模型的 Spearman ρ = 0.92，主要来自组间差别。
  - 全部数字和建议写法在共享笔记 `queue_results_1116.md`，已发给负责 report 的会话。今后不再排 GPU 任务。

- **11/17 Report 扩写**：用户要求“正文约 12 页、全文约 15 页”，推导要严谨，准备工作和实验过程要写具体，引用的工作要附解释，个人贡献要写上 Kaiming。
  - **新结构**：
    - §1 引言
    - §2 相关工作：每篇引用都说明它做了什么、我们怎么用
    - §3 理论：原附录 A 的推导移进正文，每个命题都带假设和证明；新增 3.8 节，讲 t = 1 为什么让 Heun 和 RK4 失败
    - §4 准备工作和实验设计：文献阅读和推导核对、2D 预实验（W2 数字）、从预实验到研究问题、模型和采样器的设定、评估协议（课程方法、TV 下限、判定规则）、算力和续跑
    - §5 实验 C1–C6：每节先写“跑了什么”，再写结果；新增 DDIM 两个种子 FID 和类别覆盖相互矛盾的结果
    - §6 结论
    - 个人贡献
    - 附录 B：两个较长的证明
  - **删除和改动**：
    - 删掉旧的 03_methods、04_experiments、05_conclusion、06_contributions 和附录 A–D。
    - 删掉 k-means 柱状图、三个种子的复核表和配对代价表；这些数字仍写在正文里，表文件仍由 `make_tables.py` 生成。
    - 采样器表只保留 Heun 和 RK4 在均匀网格上的结果。
    - 表格从“拉伸到整页宽”改成“只在过宽时缩小”（adjustbox）。
  - **个人贡献**：按用户说明，Kaiming 的部分对应他讲的 slide，即 diffusion 基线（2D 和 MNIST 的 DDPM / DDIM）、diffusion 和 CFM 的对应关系、k-means 类别覆盖。**需要两人确认措辞。**
  - **页数**：现在正文约 13.5 页（第 1–14 页），全文 17 页，没有红色待补标记。要压到正文 12 页、全文 15 页，还得删内容，比如 2D 图、文献表或 C1 的表。

- **11/17 定稿校对**：逐节核对正文数字与 results/ 结果文件，改了 5 处：
  - 摘要：种子 sd 最大 7.8 限定为 2–100 步（1 步时为 9.3–9.9）。
  - C1：2,000 个样本使 FID 偏高 3.6–4.0（原写 3.6）。
  - C2：32 步时 OT-CFM 与 dopri5 相差 10% 以内，I-CFM 相差 25% 以内（原写两者都在 10% 以内）。
  - 结论：种子 sd 是采样噪声 sd 的 7–55 倍（原写 10–100 倍）。
  - 附录：去掉对 Villani 书的具体章节号。
  - 另外，个人贡献一节改为不用人称代词的写法。
  - 编译：17 页，无未定义引用，无 overfull。
  - 最终 PDF `report/FM_vs_Diffusion_Final_Report.pdf` 纳入 git（.gitignore 例外）。

**Report（12/03–12/16，格式见 `report/README.md`）**
- [ ] 向老师确认附录是否允许、是否计入 5 页
- [ ] 12/07 初稿 → 12/09 交叉审阅 → 12/15 定稿 → 12/16 提交
