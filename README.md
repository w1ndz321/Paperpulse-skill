# 📄🐣 PaperPulse

> 丢进去一篇论文的pdf，自动生成一份「图文简报」。

产出不是摘要翻译，也不是大纲提炼，而是类似公众号精心排版的图文稿：有叙事结构、有关键图表、有实验结论、有作者观点，不光能够迅速阅读学术文章，还能够方便的分享出去。

---

## 📖 效果展示

点击标题，直接阅读生成的解读页：

| &nbsp; | 解读标题 |
|:---:|---|
| 🧠 | [别再把 Agent 的记忆当数据库了：A-MEM 让它学会整理、联想和更新经验](https://w1ndz321.github.io/paperpulse-skill/a-mem-agentic-memory/report.html) |
| 📚 | [别再把 Prompt 越改越短了：ACE 让大模型把经验攒成一本会进化的攻略书](https://w1ndz321.github.io/paperpulse-skill/ace-agentic-context-engineering/report.html) |
| 🌍 | [别再把 Agent 关在玩具环境里：Agent-World 想给它造一座会进化的训练城市](https://w1ndz321.github.io/paperpulse-skill/agent-world-scaling-real-world-environment/report.html) |
| ⚙️ | [别只训练模型了：让 Coding Agent 的"操作系统"自己进化起来](https://w1ndz321.github.io/paperpulse-skill/agentic-harness-engineering/report.html) |
| ⚡ | [别再只盯参数了：DeepSeek-V4 真正想回答的是百万 token 怎么跑得动](https://w1ndz321.github.io/paperpulse-skill/deepseek-v4-towards-highly-efficient-million-token/report.html) |
| 🎯 | [别再把奖励平均撒给每个 token：FIPO 想让大模型学会"哪一步推理真有用"](https://w1ndz321.github.io/paperpulse-skill/fipo-future-kl-policy-optimization/report.html) |
| 💾 | [8K 窗口硬闯 350 万 tokens：MemAgent 让模型学会边读边记](https://w1ndz321.github.io/paperpulse-skill/memagent-reshaping-long-context/report.html) |
| 📊 | [让表格特征工程不再靠拍脑袋：MALMAS 给 LLM Agent 装上记忆后，开始会复盘了](https://w1ndz321.github.io/paperpulse-skill/memory-augmented-llm-based-multi-agent-system-automated/report.html) |
| 🔄 | [让 AI 团队不再"群聊"：RecursiveMAS 把多智能体协作搬进隐空间循环](https://w1ndz321.github.io/paperpulse-skill/recursive-multi-agent-systems/report.html) |
| 🔧 | [会写代码不等于会修仓库：SWE-bench 把大模型拉进真实 GitHub 现场](https://w1ndz321.github.io/paperpulse-skill/swe-bench-can-language-models-resolve/report.html) |
| 👁️ | [别只让多模态模型"看清楚"：DeepSeek 让它边想边指](https://w1ndz321.github.io/paperpulse-skill/thinking-visual-primitives/report.html) |

---

## ✨ 它能做什么

- 📑 &nbsp;**自动提取论文正文**，剔除参考文献噪音
- 🖼️ &nbsp;**智能截图图表**，自动识别 Figure / Table 并裁剪
- ✍️ &nbsp;**生成图文简报**，有叙事结构、有实验结论、有作者观点，不是机器翻译腔
- 🌐 &nbsp;**输出 HTML 页面**，本地打开或部署到 GitHub Pages 即可分享

---

## 🚀 Quick Start

### 1. 安装 Python 依赖

```bash
pip install pymupdf pymupdf4llm
pip install markdown jinja2   # 可选，用于更精美的渲染
```

### 2. 在 Codex 中安装 Skill

**方式一：让 Codex 帮你安装**

在 Codex 中发送：

```
install skill from https://github.com/w1ndz321/paperpulse-skill
```

**方式二：手动安装**

```bash
git clone https://github.com/w1ndz321/paperpulse-skill ~/.codex/skills/paperpulse-skill
```

---

### 使用

在 Codex 中发送：

```
阅读这篇文章 /path/to/paper.pdf
```

或指定 Skill：

```
使用 paperpulse-skill 阅读这篇文章 /path/to/paper.pdf
```

**输出结构**

```
outputs/<paper-slug>/
├── report.html      ← 精美 HTML，可直接分享
├── report.md        ← 中文解读正文
└── images/          ← 截取的图表截图
```

---

## 🤔 为什么不用别的？

**直接丢给 AI 网页端？**

最方便，但只能得到纯文字总结——方法框架图、实验结果表格全都看不到，AI 复述的数据也无从核实，幻觉风险难以排查。

**用其他 paper reading 项目？**

市面上同类工具即使能产出图文，部署往往繁琐，需要启动额外的服务或配置环境，门槛不低。

**PaperPulse 的定位**

- 🔍  **截图取证**：直接从 PDF 裁出方法框架图、实验表格，所见即原文，无幻觉风险
- 💬  **有态度的文风**：稿件效仿公众号精品文章，有观点、有判断，不是干巴巴的摘要
- 🚀  **零额外部署**：装两个 Python 包，接入 Codex 即用，无需启动任何额外服务
- 📤  **即产即发**：精美 HTML 本地可读、可直接托管分享，促进学术传播

---

本项目由 [Codex](https://openai.com/codex) 与 [Claude](https://claude.ai)（Anthropic）辅助开发。
