# PaperPulse Skill

把 CS / AI / LLM 学术 PDF 变成图文并茂的中文解读页，可直接分享或部署到 GitHub Pages。

```
PDF → source_text.md + images/ + captions.json → report.md → report.html
```

## Quick Start

**1. 安装依赖**

```bash
pip install pymupdf pymupdf4llm
```

可选（用于更好的 Markdown 渲染和 HTML 模板）：

```bash
pip install markdown jinja2
```

**2. 添加 Skill**

将本仓库克隆到 Codex 的 skill 目录，或在 Codex 配置中指向 `openai.yaml`：

```bash
git clone https://github.com/w1ndz321/paperpulse-skill ~/.codex/skills/paperpulse-skill
```

**3. 使用**

在 Codex 中直接把 PDF 路径发给 PaperPulse Skill，例如：

```
用 PaperPulse 读这篇论文：/path/to/paper.pdf
```

**4. 输出**

```
outputs/<paper-slug>/
├── report.md        # 中文解读正文
├── report.html      # 可直接打开或部署的 HTML 页面
├── source_text.md   # 提取的论文正文
├── captions.json    # 图表元数据
└── images/          # 截取的图表截图
```

---

## 效果展示

以下是用 PaperPulse 生成的论文解读页面，点击标题直接阅读：

| 论文解读 |
|---|
| [别再把 Agent 的记忆当数据库了：A-MEM 让它学会整理、联想和更新经验](https://w1ndz321.github.io/paperpulse-skill/a-mem-agentic-memory/report.html) |
| [别再把 Prompt 越改越短了：ACE 让大模型把经验攒成一本会进化的攻略书](https://w1ndz321.github.io/paperpulse-skill/ace-agentic-context-engineering/report.html) |
| [别再把 Agent 关在玩具环境里：Agent-World 想给它造一座会进化的训练城市](https://w1ndz321.github.io/paperpulse-skill/agent-world-scaling-real-world-environment/report.html) |
| [别只训练模型了：让 Coding Agent 的"操作系统"自己进化起来](https://w1ndz321.github.io/paperpulse-skill/agentic-harness-engineering/report.html) |
| [别再只盯参数了：DeepSeek-V4 真正想回答的是百万 token 怎么跑得动](https://w1ndz321.github.io/paperpulse-skill/deepseek-v4-towards-highly-efficient-million-token/report.html) |
| [别再把奖励平均撒给每个 token：FIPO 想让大模型学会"哪一步推理真有用"](https://w1ndz321.github.io/paperpulse-skill/fipo-future-kl-policy-optimization/report.html) |
| [8K 窗口硬闯 350 万 tokens：MemAgent 让模型学会边读边记](https://w1ndz321.github.io/paperpulse-skill/memagent-reshaping-long-context/report.html) |
| [让表格特征工程不再靠拍脑袋：MALMAS 给 LLM Agent 装上记忆后，开始会复盘了](https://w1ndz321.github.io/paperpulse-skill/memory-augmented-llm-based-multi-agent-system-automated/report.html) |
| [让 AI 团队不再"群聊"：RecursiveMAS 把多智能体协作搬进隐空间循环](https://w1ndz321.github.io/paperpulse-skill/recursive-multi-agent-systems/report.html) |
| [会写代码不等于会修仓库：SWE-bench 把大模型拉进真实 GitHub 现场](https://w1ndz321.github.io/paperpulse-skill/swe-bench-can-language-models-resolve/report.html) |
| [别只让多模态模型"看清楚"：DeepSeek 让它边想边指](https://w1ndz321.github.io/paperpulse-skill/thinking-visual-primitives/report.html) |
