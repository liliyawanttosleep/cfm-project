# Slides

| 内容 | 位置 |
|---|---|
| **在线幻灯片（演示用，以它为准）** | https://claude.ai/artifact/GRZZAyn8ir3RXQE1Hc8P5e （可导出 PPTX/PDF） |
| 幻灯片源文件（版本记录） | `deck/project/deck.json` + `deck/project/slides/<id>.html`（每页一个文件） |
| 幻灯片用图 | `figs/`，用 `python slides/make_slide_figs.py slides/figs` 重新生成（需先跑完 E1 的 moons 实验） |

## 怎么改幻灯片

**方式 1：直接在线改（推荐，适合改文字和排版）**
在线幻灯片的拥有者是 Wentao，队友需要先在页面的 Share 菜单中被授予**编辑权限**。之后打开链接就能直接编辑，改动实时保存。

**方式 2：让 Claude 改源文件（适合大段内容或批量修改）**
1. 修改 `deck/project/slides/<id>.html`。格式规则：每页只有一个 `<section>`，样式全部内联；讲稿写在最后的 `<aside>` 里。
2. 让 Claude 把改过的文件发布到上面的链接（Artifact publish，`url` 设为上面的链接，`root` 设为 `slides/deck`）。只有拥有编辑权限的人才能发布。
3. 提交到 git。

**同步规则**：在线版本是演示时的最终版本。每次在线改完一轮，请 Claude 用 Artifact `read` 把在线的 `project/` 文件拉回 `slides/deck/project/` 并提交，这样 git 里始终有一份最新快照，也能看到每次改了什么。

## 页面编号与讲者

页面顺序见 `deck/project/deck.json` 中的 `order`；每页讲稿的开头标有【A 讲】或【B 讲】以及时长。完整大纲见 `TEAM_PLAN.md` §4 P6。
