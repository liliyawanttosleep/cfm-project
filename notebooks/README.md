# Notebooks

放我们自己的探索和测试 notebook。命名规则：`<成员>_<日期>_<主题>.ipynb`，例如 `A_1105_otcfm_8gaussians.ipynb`。

- torchcfm 的官方例子在 `conditional-flow-matching-main/examples/`（上游仓库，不进我们的 git）。想改它们时，先**复制到这里**再改。
- **提交前清空输出**（Jupyter 菜单：Kernel → Restart & Clear Output），否则 diff 很大、容易冲突。也可以安装 `pip install nbstripout`，然后在仓库根目录执行一次 `nbstripout --install`，之后提交时会自动清空输出。
- 稳定下来的代码要移到 `code/` 里写成脚本，notebook 只用来探索和画图。
- 两个人不要同时改同一个 notebook（notebook 的合并冲突很难解决）。
