# Final report：格式、提纲和提交检查

> 写于 2026-11-07。这是 12/16 要交的 final report 的规格说明。实验内容见 `docs/FINAL_REPORT_PLAN.md`，全流程见 `TEAM_PLAN.md` 开头的“项目全流程”。
> 规划用中文；**report 全文用英文**。

---

## 1. 交什么

| 项目 | 规定 |
|---|---|
| 文件 | 一个 PDF：`FM_vs_Diffusion_Final_Report.pdf` |
| 截止 | **2026-12-16 23:59**（计划 12/15 定稿） |
| 篇幅 | 正文约 **5 页**（课程要求“约 5 页”），参考文献不计入 |
| 附录 | 放证明和次要结果，**是否允许、是否计入页数要向老师确认**；确认前按“正文 5 页自成一体，附录删掉也不影响阅读”来写 |
| 语言 | 英文 |
| 格式 | 正式技术论文：Introduction / Related Work / Methods / Experiments / Conclusion / References（课程建议的结构），外加 **Individual Contributions**（课程硬性要求） |
| 代码 | 文末给出 GitHub 仓库链接，`README.md` 写明每张图表的复现命令 |

---

## 2. 模板和排版

- **模板**：NeurIPS 风格的 LaTeX 模板（单栏，10pt，正文宽度固定），用 final 模式显示作者名。可以在 Overleaf 上两人协作，源文件同时放进本目录（`report/`）。
- **作者**：Wentao Yang, Kaiming Zhang；Department of Statistics。
- **图**：用 matplotlib 导出 PDF（矢量），字号不小于正文的 8pt；同一种方法在所有图里用同一种颜色（和 PPT 一致：OT-CFM 橙色、I-CFM 深橙、DDIM 蓝色）。
- **表**：用 `booktabs`（三线表）；数字统一保留 3 位有效数字；均值 ± 标准差，注明种子数。
- **公式**：记号和 PPT 一致：t = 0 是噪声，t = 1 是数据；x₀ ~ N(0, I)，x₁ ~ p_data；v_θ(x, t) 是网络，u_t(x) 是边缘速度场。
- **引用**：BibTeX，作者–年份格式；所有文献数字注明出处和设定（NFE、样本数），不和我们的数字直接比大小。

---

## 3. 提纲和页数分配（正文 5 页）

**题目（暂定）**：*Fewer Steps, Same Regression: Measuring and Improving the Sampling Efficiency of Conditional Flow Matching*

| 章节 | 页数 | 内容 | 图表 | 数据来源 |
|---|---|---|---|---|
| **Abstract** | 0.15 | 4–5 句：问题（CFM 采样要多少步）→ 做法（同一个网络上比较训练端和推理端的优化，并检查评估本身的误差）→ 主要发现（用数字）→ 意义 | — | C6 |
| **1 Introduction** | 0.5 | 生成 = 从样本学分布再采样；CFM 把它变成回归，采样代价 = NFE；高维中 OT 配对的优势会消失（我们在 MNIST 上看到的现象）；研究问题 RQ1–RQ4；3 条贡献 | — | — |
| **2 Related Work** | 0.4 | Score matching → DDPM / DDIM / score SDE；Flow Matching、Rectified Flow、Stochastic Interpolants；OT-CFM；少步生成：reflow、蒸馏、consistency model、MeanFlow；ODE 求解器和时间网格（DDIM、EDM）；生成模型评估（FID 及其局限） | — | papers/ |
| **3 Methods** | 1.3 | 3.1 CFM：条件路径、CFM 损失、后验均值定理（证明放附录）<br>3.2 为什么路径会弯、NFE 意味着什么（一步 Euler 落在 E[x₁] 上）<br>3.3 四种优化办法：配对与 OT batch；训练时间分布；reflow；求解器与时间网格（包括**校准网格**，我们的实用方法；以及按曲率分配的网格，推导见 `docs/notes/derivations.md` D5）<br>3.4 评估：FID、类别覆盖（k-means + χ²，课程方法）、直线度、效率指标（NFE@20、训练代价） | 图 1：2D 中 I-CFM 与 OT-CFM 的路径（一张小图说明“路径弯 → 少步差”） | E1、slides 9–19 |
| **4 Experiments** | 2.2 | 4.1 设定：MNIST、同一个 U-Net、训练预算、种子、FID 协议（一张表）<br>4.2 评估本身的误差：训练种子 vs 采样噪声 vs 样本数；类别覆盖的区间<br>4.3 推理端：求解器 × 网格；曲率分布 κ(t)<br>4.4 训练端：OT batch、训练时间分布、reflow，以及各自的训练代价<br>4.5 综合：质量–效率前沿；直线度能否预测少步质量；DDIM 作参照 | 表 1：设定和误差<br>图 2：κ(t) 和不同网格<br>图 3：FID vs NFE（前沿）<br>表 2：各办法的 1 步 / 5 步 FID、NFE@20、训练代价<br>图 4：直线度 vs 少步 FID | C1–C6 |
| **5 Conclusion** | 0.3 | 哪种办法最划算；局限（50 epoch、无 EMA；3 个种子；FID 用 ImageNet 特征；diffusion 参照只有 2 个种子；只有 MNIST，CIFAR 只做了推理端验证）；未来工作 | — | C7 |
| **6 Individual Contributions** | 0.15 | 两人各自做了什么（模板见第 4 节） | — | — |
| **References** | 不计 | 约 15–20 篇 | — | `docs/03_参考资料.md` |
| **Appendix** | 不计（待确认） | A 证明（CFM 与 FM 同梯度、连续性方程，即 backup B1、B2）；B 课程方法的预实验（TSTR、real-vs-fake、PCA）；C 2D 完整结果；D CIFAR-10；E 超参数和计算资源 | — | — |

**正文最多 4 张图、2 张表**。放不下的移到附录，正文用一句话加引用。

**3 条贡献（写 Introduction 时以实验结果为准再改）**
1. 在同一个网络和训练预算下，系统比较 CFM 少步生成的训练端和推理端优化办法，并给出各自的训练代价。
2. 从局部误差推导出按曲率分配的时间网格及其预测收益，并说明它何时失效（最后一大步落在还没收敛的后验均值上）；在此基础上给出不需要真实数据的校准网格。
3. 量化评估本身的误差（训练种子、采样噪声、样本数），并用聚类方法检查 FID 看不到的类别覆盖问题。

---

## 4. Individual Contributions 的写法

课程硬性要求。按实际情况如实写，每人 2–3 句，用第三人称。示例结构（内容在实验完成后按实际分工改写）：

> **Wentao Yang** implemented the flow-matching models and the training pipeline, ran the solver, time-grid and reflow experiments (C2, C3, C5, C7), and wrote Sections 1, 3 and 4.3–4.4.
> **Kaiming Zhang** implemented the diffusion baseline and the evaluation code, ran the reliability, coverage and time-distribution experiments (C1, C4), and wrote Sections 2, 4.2 and 5.
> Both authors designed the experiments, analysed the results (C6), prepared the presentation and revised the full paper.

---

## 5. 写作规则

1. **每个数字都能追到结果文件**：正文和表里的每个数字，在 `results/` 下都有对应的 json / csv。写初稿时在 LaTeX 注释里标出来源，例如 `% results/c5_reflow/eval.json`。
2. **只报告真实跑出来的结果**；没做完的实验写进 Limitations，不写成结论。文献数字只作参照，注明出处。
3. **误差**：基础模型报告 3 个种子的均值 ± 标准差；单种子的结果标明 “single seed”。差距小于改进门槛（2 × 种子标准差）时写“no detectable difference”。
4. **结论措辞**：只有 3 个种子，不写 “significant”；写 “consistent across all three seeds”。
5. **图注自成一体**：读者只看图和图注就能明白画了什么、样本数多少、越低越好还是越高越好。
6. **和 PPT 的关系**：PPT 的图可以重画后使用，但 report 的重点是 ⑥ 的新实验；PPT 里已讲过的推导在 report 中压缩到 Methods 的一节，证明放附录。

---

## 6. 提交前检查清单

- [ ] 正文约 5 页，参考文献和附录另计（附录是否允许已向老师确认）
- [ ] 结构包含 Introduction / Related Work / Methods / Experiments / Conclusion / References
- [ ] 有 **Individual Contributions** 一节，两人都写到
- [ ] 课程方法的要求已满足：只要求一种（k-means + χ²，在 3.4 和 4.2 节），另外三类的预实验放在附录 B
- [ ] 有和 state-of-the-art 的比较（文献表 + C7）
- [ ] 每张图表都在正文中被引用；编号连续；坐标轴有标签；图例清楚
- [ ] 每个数字都能追到 `results/` 下的文件
- [ ] 参考文献完整，格式统一
- [ ] 作者名、题目、GitHub 链接正确
- [ ] 两人交叉读过全文（12/09 前）
- [ ] 导出 PDF 后检查字体嵌入、图片清晰度
- [ ] **12/16 23:59 前提交**

---

## 7. 时间表

| 日期 | 事项 |
|---|---|
| 11/11–12/02 | 做实验（`docs/FINAL_REPORT_PLAN.md`）；同时把 Methods 和 Related Work 先写出来（不依赖结果） |
| 12/03 | 搭好 LaTeX 框架，放入所有图表 |
| 12/03–12/07 | 写 Experiments、Introduction、Conclusion；初稿完成 |
| 12/08–12/09 | 交叉审阅：A 审 B 写的部分，B 审 A 写的部分 |
| 12/10–12/14 | 修改；压缩到 5 页 |
| 12/15 | 定稿 |
| 12/16 23:59 前 | 提交 |
