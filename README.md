# 📄🐣 PaperPulse

> 给它一篇论文 PDF，输出一份可以直接分享的「图文简报」。

产出不是摘要翻译，也不是大纲提炼——而是有叙事结构、有关键图表、有实验结论、有作者观点的图文稿，类似公众号精品文章的质感，拿到即可阅读和传播。

---

## 📖 效果展示

点击标题查看输出效果：

|   | 解读标题 |
|   | 解读标题 |
|:---:|---|
| 🧠 | [别再把 Agent 的记忆当数据库了：A-MEM 让它学会整理、联想和更新经验](https://w1ndz321.github.io/paperpulse-skill/a-mem-agentic-memory/report.html) |
| 💾 | [8K 窗口硬闯 350 万 tokens：MemAgent 让模型学会边读边记](https://w1ndz321.github.io/paperpulse-skill/memagent-reshaping-long-context/report.html) |
| 📚 | [别再把 Prompt 越改越短了：ACE 让大模型把经验攒成一本会进化的攻略书](https://w1ndz321.github.io/paperpulse-skill/ace-agentic-context-engineering/report.html) |
| 🔧 | [会写代码不等于会修仓库：SWE-bench 把大模型拉进真实 GitHub 现场](https://w1ndz321.github.io/paperpulse-skill/swe-bench-can-language-models-resolve/report.html) |
| 🌍 | [别再把 Agent 关在玩具环境里：Agent-World 想给它造一座会进化的训练城市](https://w1ndz321.github.io/paperpulse-skill/agent-world-scaling-real-world-environment/report.html) |
| 📊 | [让表格特征工程不再靠拍脑袋：MALMAS 给 LLM Agent 装上记忆后，开始会复盘了](https://w1ndz321.github.io/paperpulse-skill/memory-augmented-llm-based-multi-agent-system-automated/report.html) |
| ⚙️ | [别只训练模型了：让 Coding Agent 的操作系统自己进化起来](https://w1ndz321.github.io/paperpulse-skill/agentic-harness-engineering/report.html) |
| 🔄 | [让 AI 团队不再群聊：RecursiveMAS 把多智能体协作搬进隐空间循环](https://w1ndz321.github.io/paperpulse-skill/recursive-multi-agent-systems/report.html) |
| ⚡ | [别再只盯参数了：DeepSeek-V4 真正想回答的是百万 token 怎么跑得动](https://w1ndz321.github.io/paperpulse-skill/deepseek-v4-towards-highly-efficient-million-token/report.html) |
| 👁️ | [别只让多模态模型看清楚：DeepSeek 让它边想边指](https://w1ndz321.github.io/paperpulse-skill/thinking-visual-primitives/report.html) |
| 🎯 | [别再把奖励平均撒给每个 token：FIPO 想让大模型学会哪一步推理真有用](https://w1ndz321.github.io/paperpulse-skill/fipo-future-kl-policy-optimization/report.html) |
---

## ✨ 它能做什么

- 📑  **自动提取论文正文**，剔除参考文献噪音
- 🖼️  **智能截图图表**，自动识别 Figure / Table 并裁剪
- ✍️  **生成图文简报**，有叙事结构、有实验结论、有作者观点，不是机器翻译腔
- 🌐  **输出 HTML 页面**，本地可读，可一键部署到 GitHub Pages 分享

---

## 🚀 Quick Start

**1. 安装依赖**

```bash
pip install pymupdf pymupdf4llm
pip install markdown jinja2   # 可选
```

**2. 安装 Skill**

让 Codex 安装：

或手动克隆：

```bash
git clone https://github.com/w1ndz321/paperpulse-skill ~/.codex/skills/paperpulse-skill
```

**3. 使用**

```
阅读这篇文章 /path/to/paper.pdf
# 或
使用 paperpulse-skill 阅读这篇文章 /path/to/paper.pdf
```

输出至 ，包含 、 和 。

---

## 🤔 为什么用 Codex Skill 来做这件事？

**直接用 AI 网页端不行吗？**

可以，但只能得到纯文字总结——方法框架图、实验结果表格全都看不到，AI 复述的数据无从核实，幻觉风险难以排查。

**用其他 paper reading 工具呢？**

市面上同类工具即使能产出图文，部署往往繁琐，需要启动额外服务或配置环境，门槛不低。

**PaperPulse 的定位**

- 🔍  **截图取证**：直接从 PDF 裁出方法框架图、实验表格，所见即原文，无幻觉风险
- 💬  **有态度的文风**：效仿公众号精品文章，有观点、有判断，不是干巴巴的摘要
- 🚀  **零额外部署**：装两个 Python 包，接入 Codex 即用，无需启动任何额外服务
- 📤  **即产即发**：精美 HTML 本地可读、可直接托管分享，促进学术传播

**为什么不直接用脚本？**

纯脚本截图有一系列难以彻底避免的细节问题：链接识别错误、HTML 排版错乱、单双栏图片错位、表格识别不全、图表截歪……这些以前只能靠人工逐一修补。

PaperPulse 让 Codex 扮演「人工审校」的角色，在生成过程中自动发现瑕疵并修正，最终成品经过完整检查，最大程度保障可分享的质量。Codex 的记忆功能还会在遇到报错时自动记录并修复 Skill，共性问题持续沉淀——越用越顺手。

实测数据：用不同排版、不同类型（新方法论文、综述）、不同体量的文章反复验证，处理单篇耗时从最初约 **15 分钟**降至现在平均 **5 分钟**。

---

本项目借助 [Codex](https://openai.com/codex) 与 [Claude](https://claude.ai)（Anthropic）辅助开发。
