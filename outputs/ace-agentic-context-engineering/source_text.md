Published as a conference paper at ICLR 2026 

# - AGENTIC CONTEXT ENGINEERING: EVOLVING CON TEXTS FOR SELF-IMPROVING LANGUAGE MODELS 

**Qizheng Zhang**[1] _[∗]_ **, Changran Hu**[2] _[∗]_ **, Shubhangi Upasani**[2] **, Boyuan Ma**[2] **, Fenglu Hong**[2] **, Vamsidhar Kamanuru**[2] **, Jay Rainton**[2] **, Chen Wu**[2] **, Mengmeng Ji**[2] **, Hanchen Li**[3] **, Urmish Thakker**[2] **, James Zou**[1] **, Kunle Olukotun**[1] 

1Stanford University 2SambaNova Systems, Inc. 3UC Berkeley 

_{_ qizhengz,kunle _}_ @stanford.edu changran_hu@berkeley.edu 

� ace-agent/ace � ace-agent.github.io _∗_ Equal contribution. 

## ABSTRACT 

Large language model (LLM) applications such as agents and domain-specific reasoning increasingly rely on _context adaptation_ : modifying inputs with instructions, strategies, or evidence, rather than weight updates. Prior approaches improve usability but often suffer from brevity bias, which drops domain insights for concise summaries, and from context collapse, where iterative rewriting erodes details over time. We introduce ACE ( **A** gentic **C** ontext **E** ngineering), a framework that treats contexts as evolving playbooks that accumulate, refine, and organize strategies through a modular process of generation, reflection, and curation. ACE prevents collapse with structured, incremental updates that preserve detailed knowledge and scale with long-context models. Across agent and domain-specific benchmarks, ACE optimizes contexts both offline ( _e.g.,_ system prompts) and online ( _e.g.,_ agent memory), consistently outperforming strong baselines: +10.6% on agents and +8.6% on finance, while significantly reducing adaptation latency and rollout cost. Notably, ACE could adapt effectively without labeled supervision and instead by leveraging natural execution feedback. On the AppWorld leaderboard, ACE matches the top-ranked production-level agent on the overall average and surpasses it on the harder test-challenge split, despite using a smaller opensource model. These results show that comprehensive, evolving contexts enable scalable, efficient, and self-improving LLM systems with low overhead. 

## 1 INTRODUCTION 

**==> picture [378 x 116] intentionally omitted <==**

**----- Start of picture text -----**<br>
Agent: AppWorld Domain Knowledge: FiNER Numerical Reasoning: Formula<br>59.5% 82.5 80<br>60<br>80.0 78.3% 76.5%<br>55 51.9% 77.5 75<br>50 75.0 73.5% [74.2%] 71.5%<br>46.0% [46.4%] 72.5 72.3% 70 69.5%<br>45 42.4% 70.0 70.7% 67.5%67.0%<br>40 65<br>Base LLM ICL GEPA DC ACE Base LLM ICL GEPA DC ACE Base LLM ICL GEPA DC ACE<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 1: **Overall Performance Results.** Our proposed framework, ACE, consistently outperforms strong baselines across agent and domain-specific tasks. 

Modern AI applications based on large language models (LLMs), such as LLM agents (Yao et al., 2023; Yang et al., 2024) and compound AI systems (Zaharia et al., 2024), increasingly depend on _context adaptation_ . Instead of modifying model weights, context adaptation improves performance after model training by incorporating clarified instructions, structured reasoning steps, or domain- 

1 

Published as a conference paper at ICLR 2026 

specific input formats directly into the model’s inputs. Contexts underpin many AI system components, including system prompts that guide downstream tasks (Opsahl-Ong et al., 2024; Agrawal et al., 2025), memory that carries past facts and experiences (Suzgun et al., 2025; Xu et al., 2025), and factual evidence that reduces hallucination and supplements knowledge (Asai et al., 2024). 

Adapting through _contexts_ rather than _weights_ offers several key advantages. Contexts are interpretable and explainable for users and developers (Wei et al., 2022; Wang et al., 2023), allow rapid integration of new knowledge at runtime (Lewis et al., 2020; Borgeaud et al., 2022), and can be shared across models or modules in a compound system (Khot et al., 2023). Meanwhile, advances in long-context LLMs (Peng et al., 2024) and context-efficient inference such as KV cache reuse (Gim et al., 2024; Yao et al., 2025) are making context-based approaches increasingly practical for deployment. As a result, context adaptation is emerging as a central paradigm for building capable, scalable, and self-improving AI systems. 

Despite this progress, existing approaches to context adaptation face two limitations. First, _brevity bias_ : many prompt optimizers prioritize concise applicable instructions over comprehensive accumulation. For example, GEPA (Agrawal et al., 2025) highlights brevity as a strength, but such abstraction can omit domain-specific heuristics, tool-use guidelines, or common failure modes that matter in practice (Gao et al., 2025). This objective aligns with validation metrics in some settings, but often fails to capture the detailed strategies required by agents and knowledge-intensive applications. Second, _context collapse_ : methods that rely on monolithic rewriting by an LLM often degrade into shorter, less informative summaries over time, causing sharp performance declines (Figure 2). In domains such as interactive agents (Trivedi et al., 2024; Patil et al., 2024; Zhang et al., 2024), domain-specific programming (Ye et al., 2023; Zhang et al., 2025a;b; Mang et al., 2025), and financial or legal analysis (Loukas et al., 2022; Guha et al., 2023; Wang et al., 2025a), strong performance depends on retaining detailed, task-specific knowledge rather than compressing it away. 

As applications like agents and knowledge-intensive reasoning demand greater reliability, recent work has shifted toward saturating contexts with abundant, potentially useful information (Jiang et al., 2025; Chung et al., 2025; Chen et al., 2025), enabled by advances in long-context LLMs (Peng et al., 2024; Mao et al., 2024). **We argue that contexts should function not as concise summaries, but as comprehensive, structured playbooks that are detailed, inclusive, and rich with domain insights.** Unlike humans, who often benefit from concise generalization, LLMs are more effective when provided with long, detailed contexts and can distill relevance autonomously (Jiang et al., 2025; Liu et al., 2025b; Suzgun et al., 2025). Thus, instead of compressing away domain-specific heuristics and tactics, contexts should preserve them, allowing the model to decide what matters during inference time. 

To address these limitations, we introduce ACE ( **A** gentic **C** ontext **E** ngineering), a framework for comprehensive context adaptation in both offline settings ( _e.g.,_ system prompt optimization) and online settings ( _e.g.,_ test-time memory adaptation). Rather than compressing contexts into distilled summaries, ACE treats them as evolving playbooks that accumulate and organize strategies over time. By design, ACE incorporates a modular workflow of generation, reflection, and curation, while adding structured, incremental updates guided by a grow-and-refine principle. This design preserves detailed, domain-specific knowledge, prevents context collapse, and yields contexts that remain comprehensive and scalable throughout adaptation. 

We evaluate ACE on two categories of LLM applications that most benefit from comprehensive, evolving contexts: (1) _agents_ (Trivedi et al., 2024), which require multi-turn reasoning, tool use, and environment interaction, where accumulated strategies can be reused across episodes; and (2) _domain-specific benchmarks_ , which demand specialized tactics and knowledge, like financial analysis (Loukas et al., 2022; Wang et al., 2025a). Our key findings are: 

- ACE consistently outperforms strong baselines, yielding average gains of 10.6% on _agents_ and 8.6% on _domain-specific benchmarks_ , across both offline and online adaptation settings. 

- ACE is able to construct effective contexts _without_ labeled supervision, instead leveraging execution feedback and environment signals, key ingredients for self-improving LLMs and agents. 

- On the AppWorld benchmark leaderboard (AppWorld), ACE surpasses the top-1-ranked production-level agent IBM-CUGA (Marreed et al., 2025) (powered by GPT-4.1) while using an open-source model (DeepSeek-V3.1). 

2 

Published as a conference paper at ICLR 2026 

- ACE requires significantly fewer rollouts and achieves lower adaptation latency than existing adaptive methods, demonstrating that scalable self-improvement can be achieved with both higher accuracy and lower cost. 

## 2 BACKGROUND AND MOTIVATION 

## 2.1 CONTEXT ADAPTATION 

Context adaptation (or context engineering) refers to methods that improve model behavior by constructing or modifying inputs to an LLM, rather than altering its weights. The current state of the art leverages _natural language feedback_ (Shinn et al., 2023; Yuksekgonul et al., 2025; Agrawal et al., 2025). In this paradigm, a language model inspects the current context along with signals such as execution traces, reasoning steps, or validation results, and generates natural language feedback on how the context should be revised. This feedback is then incorporated into the context, enabling iterative adaptation. Representative methods include Reflexion (Shinn et al., 2023), which reflects on failures to improve agent planning; TextGrad (Yuksekgonul et al., 2025), which optimizes prompts via gradient-like textual feedback; GEPA (Agrawal et al., 2025), which refines prompts iteratively based on execution traces and achieves strong performance, even surpassing reinforcement learning approaches in some settings; and Dynamic Cheatsheet (Krause et al., 2019), which constructs an external memory that accumulates strategies and lessons from past successes and failures during inference. These natural language feedback methods represent a major advance, offering flexible and interpretable signals for improving LLM systems beyond weight updates. 

## 2.2 LIMITATIONS OF EXISTING CONTEXT ADAPTATION METHODS 

**Brevity Bias** A recurring limitation of context adaptation methods is _brevity bias_ : the tendency of optimization to collapse toward short, generic prompts. Gao et al. (Gao et al., 2025) document this effect in prompt optimization for test generation, where iterative methods repeatedly produced nearidentical instructions ( _e.g.,_ , “Create unit tests to ensure methods behave as expected”), sacrificing diversity and omitting domain-specific detail. This convergence not only narrows the search space but also propagates recurring errors across iterations, since optimized prompts often inherit the same faults as their seeds. More broadly, such bias undermines performance in domains that demand detailed, context-rich guidance—such as multi-step agents, program synthesis, or knowledge-intensive reasoning—where success hinges on accumulating rather than compressing task-specific insights. 

**==> picture [318 x 126] intentionally omitted <==**

**----- Start of picture text -----**<br>
# Tokens: 18,282<br>Accuracy: 66.7<br># Tokens: 122<br>Accuracy w/o context: 63.7 Accuracy: 57.1<br>**----- End of picture text -----**<br>


Figure 2: **Context Collapse.** Monolithic rewriting of context by an LLM can collapse it into shorter, less informative summaries, leading to sharp performance drops. 

**Context Collapse** In a case study on the AppWorld benchmark (Trivedi et al., 2024), we observe a phenomenon we call _context collapse_ , which arises when an LLM is tasked with fully rewriting the accumulated context at each adaptation step. As the context grows large, the model tends to compress it into much shorter, less informative summaries, causing a dramatic loss of information. For instance, at step 60 the context contained 18,282 tokens and achieved an accuracy of 66.7, but at the very next step it collapsed to just 122 tokens, with accuracy dropping to 57.1—worse than the baseline accuracy of 63.7 without adaptation. While we highlight this through Dynamic Cheat- 

3 

Published as a conference paper at ICLR 2026 

sheet (Suzgun et al., 2025), the issue is not specific to that method; rather, it reflects a fundamental risk of end-to-end context rewriting with LLMs, where accumulated knowledge can be abruptly erased instead of preserved. 

**==> picture [358 x 238] intentionally omitted <==**

Figure 3: **Example ACE-Generated Context on the AppWorld Benchmark** (partially shown). ACE-generated contexts contain detailed, domain-specific insights along with tools and code that are readily usable, serving as a comprehensive playbook for LLM applications. 

## 3 AGENTIC CONTEXT ENGINEERING (ACE) 

We present ACE ( **A** gentic **C** ontext **E** ngineering), a framework for scalable and efficient context adaptation in both offline ( _e.g.,_ system prompt optimization) and online ( _e.g.,_ test-time memory adaptation) scenarios. Instead of condensing knowledge into terse summaries or static instructions, ACE treats contexts as evolving playbooks that continuously accumulate, refine, and organize strategies over time. Inspired by the agentic design of Dynamic Cheatsheet (Suzgun et al., 2025), ACE introduces a structured division of labor across three roles (Figure 4): the _Generator_ , which produces reasoning trajectories; the _Reflector_ , which distills concrete insights from successes and errors; and the _Curator_ , which integrates these insights into structured context updates. This mirrors how humans learn: experimenting, reflecting, and consolidating, while avoiding the bottleneck of overloading a single model with all responsibilities. 

To address the limitations of prior methods discussed in §2.2 (notably _brevity bias_ and _context collapse_ ) ACE introduces three key innovations: (1) a dedicated _Reflector_ that separates evaluation and insight extraction from curation, improving context quality and downstream performance (§4.6); (2) incremental _delta updates_ (§3.1) that replace costly monolithic rewrites with localized edits, reducing both latency and compute cost (§4.7); and (3) a _grow-and-refine_ mechanism (§3.2) that balances steady context expansion with redundancy control. 

As shown in Figure 4, the workflow begins with the Generator producing reasoning trajectories for new queries, which surface both effective strategies and recurring pitfalls. The Reflector critiques these traces to extract lessons, optionally refining them across multiple iterations. The Curator then synthesizes these lessons into compact _delta entries_ , which are merged deterministically into the existing context by lightweight, non-LLM logic. Because updates are itemized and localized, multiple deltas can be merged in parallel, enabling batched adaptation at scale. ACE further supports multi-epoch adaptation, where the same queries are revisited to progressively strengthen the context. 

4 

Published as a conference paper at ICLR 2026 

**==> picture [378 x 125] intentionally omitted <==**

**----- Start of picture text -----**<br>
Iterative Refinement<br>Query<br>Trajectory Insights<br>Generator Reflector Curator<br>Context<br>Playbook Update<br>Delta Context Items<br>**----- End of picture text -----**<br>


Figure 4: **The ACE Framework.** Inspired by Dynamic Cheatsheet, ACE adopts an agentic architecture with three specialized components: a Generator, a Reflector, and a Curator. 

## 3.1 INCREMENTAL DELTA UPDATES 

A core design principle of ACE is to represent context as a collection of _structured, itemized bullets_ , rather than a single monolithic prompt. The concept of a bullet is similar to the concept of a memory entry in LLM memory frameworks like Dynamic Cheatsheet (Suzgun et al., 2025) and A-MEM (Xu et al., 2025), but builds on top of that and consists of (1) **metadata** , including a unique identifier and counters tracking how often it was marked helpful or harmful; and (2) **content** , capturing a small unit such as a reusable strategy, domain concept, or common failure mode. When solving new problems, the Generator highlights which bullets were useful or misleading, providing feedback that guides the Reflector in proposing corrective updates. 

This itemized design enables three properties: (1) _localization_ , so only the relevant bullets are updated; (2) _fine-grained retrieval_ , so the Generator can focus on the most pertinent knowledge; and (3) _incremental adaptation_ , allowing efficient merging, pruning, and de-duplication during inference. 

Rather than regenerating contexts in full, ACE incrementally produces compact _delta contexts_ : small sets of candidate bullets distilled by the Reflector and integrated by the Curator. This avoids the computational cost and latency of full rewrites, while ensuring that past knowledge is preserved and new insights are steadily appended. As contexts grow, this approach provides the scalability needed for long-horizon or domain-intensive applications. 

## 3.2 GROW-AND-REFINE 

Beyond incremental growth, ACE ensures that contexts remain compact and relevant through periodic or lazy refinement. In grow-and-refine, bullets with new identifiers are appended, while existing bullets are updated in place ( _e.g.,_ incrementing counters). A de-duplication step then prunes redundancy by comparing bullets via semantic embeddings. This refinement can be performed proactively (after each delta) or lazily (only when the context window is exceeded), depending on application requirements for latency and accuracy. 

Together, incremental updates and grow-and-refine maintain contexts that expand adaptively, remain interpretable, and avoid the potential variance introduced by monolithic context rewriting. 

## 4 RESULTS 

Our evaluation of ACE shows that: 

- **Enabling High-Performance, Self-Improving Agents.** ACE enables agents to self-improve by dynamically refining their input context, both in offline and online settings. It boosts accuracy on the AppWorld benchmark by up to 17.1% by learning to engineer better contexts from execution feedback alone, without needing ground-truth labels. (§4.3) 

- **Large Gains on Domain-Specific Benchmarks.** On complex financial reasoning benchmarks, ACE delivers an average performance gain of 8.6% over strong baselines by constructing comprehensive playbooks with domain-specific concepts and insights. (§4.4) 

5 

Published as a conference paper at ICLR 2026 

- **Effective by Design.** Ablation studies confirm our design choices are key to success, with components like the Reflector, multi-epoch refinement, and incremental delta update each contributing substantial performance gains. (§4.6) 

- **Lower Cost and Adaptation Latency.** ACE achieves these gains efficiently, reducing adaptation latency by 86.9% on average, while requiring fewer rollouts and lower token dollar costs. (§4.7) 

## 4.1 TASKS AND DATASETS 

We evaluate ACE on two categories of LLM applications that benefit most from evolving contexts: (1) _LLM agent_ , which require multi-turn reasoning, tool use, and environment interaction; with ACE, agents can accumulate and reuse strategies across episodes and environments; and (2) _domainspecific reasoning_ , which demand mastery of specialized concepts and tactics; we focus on financial analysis as a main case study, and show additional results on medical reasoning and text-to-SQL. 

- **LLM Agent: AppWorld (Trivedi et al., 2024)** is a suite of autonomous agent tasks involving API understanding, code generation, and environment interaction. It provides a realistic execution environment with common applications and APIs ( _e.g.,_ email, file system) and tasks of two difficulty levels (normal and challenge). A public leaderboard (AppWorld) tracks performance, where, at the time of submission, the best system achieved only 60.3% average accuracy, highlighting the benchmark’s difficulty and realism. 

- **Domain-Specific Reasoning: Financial, Medical, and Text-to-SQL Benchmarks** We use finance as our main case study in §4.4. For financial analysis, we focus on FiNER (Loukas et al., 2022) and Formula (Wang et al., 2025a), which test LLMs on financial reasoning tasks that rely on the eXtensible Business Reporting Language (XBRL). _FiNER_ requires labeling tokens in XBRL financial documents with one of 139 fine-grained entity types, a key step for financial information extraction in regulated domains. _Formula_ focuses on applying financial concepts and performing computations to answer queries, _i.e.,_ numerical reasoning. Beyond finance, we evaluate on two additional domain tasks from StreamBench (Wu et al., 2024): DDXPlus (Fansi Tchango et al., 2022) (medical reasoning) and BIRD-SQL (Li et al., 2023) (text-to-SQL). 

**Evaluation Metrics** For AppWorld, we follow the official benchmark protocol and report _Task Goal Completion_ (TGC) and _Scenario Goal Completion_ (SGC) on both the test-normal and testchallenge splits. For FiNER, Formula and DDXPlus, we follow the original setup and report accuracy, measured as the proportion of predicted answers that exactly match the ground truth. For BIRD-SQL, we use GPT-4o-mini (OpenAI, 2024) under LLM-as-a-judge (Zheng et al., 2023). 

All datasets follow the original train/validation/test splits. For _offline_ context adaptation, methods are optimized on the training split and evaluated on the test split with pass@1 accuracy. For _online_ context adaptation, methods are evaluated sequentially on the test split: for each sample, the model first predicts with the current context, then updates its context based on that sample. The same shuffled test split is used across all methods. 

## 4.2 BASELINES AND METHODS 

**Base LLM** The base model is evaluated directly on each benchmark without any context engineering, using the default prompts provided by dataset authors. For AppWorld, we follow the official ReAct (Yao et al., 2023) implementation released by the benchmark authors, and build all other baselines and methods on top of this framework. 

**In-Context Learning (ICL) (Agarwal et al., 2024)** ICL provides the model with task demonstrations in the input prompt (few-shot or many-shot). This allows the model to infer the task format and desired output without weight updates. We supply all training samples when they fit within the model’s context window; otherwise, we fill the window with as many demonstrations as possible. 

**MIPROv2 (Opsahl-Ong et al., 2024)** MIPROv2 is a popular prompt optimizer for LLM applications that works by jointly optimizing system instructions and in-context demonstrations via bayesian optimization. We use the official DSPy implementation (DSPy, b), setting auto="heavy" to maximize optimization performance. 

6 

Published as a conference paper at ICLR 2026 

**GEPA (Agrawal et al., 2025)** GEPA (Genetic-Pareto) is a sample-efficient prompt optimizer based on reflective prompt evolution. It collects execution traces (reasoning, tool calls, intermediate outputs) and applies natural-language reflection to diagnose errors, assign credit, and propose prompt updates. A genetic Pareto search maintains a frontier of high-performing prompts, mitigating local optima. Empirically, GEPA outperforms reinforcement learning methods such as GRPO and prompt optimizers like MIPROv2, achieving up to 10–20% higher accuracy with as much as 35× fewer rollouts. We use the official DSPy implementation (DSPy, a), setting auto="heavy" to maximize optimization performance. 

**Dynamic Cheatsheet (DC) (Suzgun et al., 2025)** DC is a test-time learning approach that introduces an adaptive external memory of reusable strategies and code snippets. By continuously updating this memory with newly encountered inputs and outputs, DC enables models to accumulate knowledge and reuse it across tasks, often leading to substantial improvements over static prompting methods. A key advantage of DC is that it does not require ground-truth labels: the model can curate its own memory from its generations, making the method highly flexible and broadly applicable. We use the official implementation released by the authors (Suzgun et al.) and set it to use the cumulative mode (DC-CU). 

**ACE (ours)** ACE optimizes LLM contexts for both offline and online adaptation through an agentic context engineering framework. To ensure fairness, we use the same LLM for the Generator, Reflector, and Curator (non-thinking mode of DeepSeek-V3.1 (Liu et al., 2024a)), preventing knowledge transfer from a stronger Reflector or Curator to a weaker Generator. This isolates the benefit of context construction itself. We additionally evaluate ACE with other backbone LLMs in the appendix, where we observe consistent gains. We adopt a batch size of 1 (constructing a delta context from each sample). We set the maximum number of Reflector refinement rounds and the maximum number of epoch in offline adaptation to 5. 

4.3 RESULTS ON AGENT BENCHMARK 

|**Method**<br>**GT Labels**|**Test-Normal**<br>**TGC**_↑_<br>**SGC**_↑_|**Test-Challenge**<br>**Average**<br>**TGC**_↑_<br>**SGC**_↑_|**Test-Challenge**<br>**Average**<br>**TGC**_↑_<br>**SGC**_↑_|
|---|---|---|---|
|DeepSeek-V3.1-671B as Base LLM||||
|ReAct|63.7<br>42.9|41.5<br>21.6|42.4|
|Offline Adaptation||||
|ReAct+ ICL<br>✓<br>ReAct+ GEPA<br>✓<br>ReAct+ ACE<br>✓<br>ReAct+ ACE<br>✗|64_._3+0_._6<br>46_._4+3_._5<br>64_._9+1_._2<br>44_._6+1_._7<br>**76**_._**2**+**12**_._**5**<br>**64**_._**3**+**21**_._**4**<br>75_._0+11_._3<br>**64**_._**3**+**21**_._**4**|46_._0+4_._5<br>27_._3+5_._7<br>46_._0+4_._5<br>30_._2+8_._6<br>**57**_._**3**+**15**_._**8**<br>**39**_._**6**+**18**_._**0**<br>54_._4+12_._9<br>35_._2+13_._6|46_._0+3_._6<br>46_._4+4_._0<br>**59**_._**4**+**17**_._**0**<br>57_._2+14_._8|
|Online Adaptation||||
|ReAct+ DC (CU)<br>✗<br>ReAct+ ACE<br>✗|65_._5+1_._8<br>**58**_._**9**+**16**_._**0**<br>**69**_._**6**+**5**_._**9**<br>53_._6+10_._7|52_._3+10_._8<br>30_._8+9_._2<br>**66**_._**0**+**24**_._**5**<br>**48**_._**9**+**27**_._**3**|51_._9+9_._5<br>**59**_._**5**+**17**_._**1**|



Table 1: **Results on the AppWorld Agent Benchmark (DeepSeek-V3.1-671B as the Base LLM).** “GT labels” indicates whether ground-truth labels are available to the Reflector during adaptation. We evaluate the ACE framework against multiple baselines on top of the official ReAct implementation, both for offline and online context adaptation. ReAct + ACE outperforms selected baselines by an average of 10.6%, and could achieve good performance even without access to GT labels. 

**Analysis: AppWorld** As shown in Table 1, ACE consistently improves over strong baselines on AppWorld. In the offline setting, ReAct + ACE outperforms both ReAct + ICL and ReAct + GEPA by significant margins (12.3% and 11.9%, respectively), demonstrating that structured, evolving, and detailed contexts enable more effective agent learning than fixed demonstrations or single optimized instruction prompts. These gains extend to the online setting, where ACE continues to outperform prior adaptive methods such as Dynamic Cheatsheet by an average of 7.6%. 

In the agent use case, ACE remains effective even _without_ access to ground-truth labels during adaptation: ReAct + ACE achieves an average improvement of 14.8% over the ReAct baseline 

7 

Published as a conference paper at ICLR 2026 

in this setting. This robustness arises because ACE leverages signals naturally available during execution ( _e.g.,_ code execution success or failure) to guide the Reflector and Curator in forming structured lessons of successes and failures. Together, these results establish ACE as a strong and versatile framework for building self-improving agents that adapt reliably both with and without labeled supervision. 

Notably, on the latest AppWorld leaderboard (as of September 20, 2025; Figure 5), ReAct + ACE (59.4% average) matches the top-1-ranked IBM CUGA (60.3%)[1] , a production-level GPT-4.1–based agent (Marreed et al., 2025), despite using the much smaller open-source model DeepSeek-V3.1. With online adaptation, ReAct + ACE even surpasses IBM CUGA by 8.4% in TGC and 0.7% in SGC on test-challenge, underscoring the effectiveness of ACE in building comprehensive and self-evolving contexts for agents. 

- 4.4 RESULTS ON DOMAIN-SPECIFIC BENCHMARK 

|**Method**<br>**GT Labels**<br>**FiNER (Acc**_↑_**)**<br>**Formula (Acc**_↑_**)**<br>**Average**|**Method**<br>**GT Labels**<br>**FiNER (Acc**_↑_**)**<br>**Formula (Acc**_↑_**)**<br>**Average**|**Method**<br>**GT Labels**<br>**FiNER (Acc**_↑_**)**<br>**Formula (Acc**_↑_**)**<br>**Average**|**Method**<br>**GT Labels**<br>**FiNER (Acc**_↑_**)**<br>**Formula (Acc**_↑_**)**<br>**Average**|
|---|---|---|---|
|DeepSeek-V3.1 as Base LLM||||
|Base LLM|70.7|67.5|69.1|
|Offline Adaptation||||
|ICL<br>✓<br>MIPROv2<br>✓<br>GEPA<br>✓<br>ACE<br>✓<br>ACE<br>✗|72_._3+1_._6<br>72_._4+1_._7<br>73_._5+2_._8<br>**78**_._**3**+**7**_._**6**<br>71_._1+0_._4|67_._0_−_0_._5<br>69_._5+2_._0<br>71_._5+4_._0<br>**85**_._**5**+**18**_._**0**<br>83_._0+15_._5|69_._6+0_._5<br>70_._9+1_._8<br>72_._5+3_._4<br>**81**_._**9**+**12**_._**8**<br>77_._1+8_._0|
|Online Adaptation||||
|DC (CU)<br>✓<br>DC (CU)<br>✗<br>ACE<br>✓<br>ACE<br>✗|74_._2+3_._5<br>68_._3_−_2_._4<br>**76**_._**7**+**6**_._**0**<br>67_._3_−_3_._4|69_._5+2_._0<br>62_._5_−_5_._0<br>76_._5+9_._0<br>**78**_._**5**+**11**_._**0**|71_._8+2_._7<br>65_._4_−_3_._7<br>**76**_._**6**+**7**_._**5**<br>72_._9+3_._8|



Table 2: **Results on Financial Analysis Benchmark (DeepSeek-V3.1-671B as the Base LLM).** “GT labels” indicates whether ground-truth labels are available to the Reflector during adaptation. With GT labels, ACE achieves consistent improvements in both offline and online settings, highlighting the advantage of structured and evolving contexts for domain-specific reasoning. However, we also observe that in the absence of reliable feedback signals ( _e.g.,_ ground-truth labels or execution outcomes), both ACE and other adaptive methods such as Dynamic Cheatsheet may degrade, suggesting that context adaptation depends critically on feedback quality. 

**Analysis: Finance Benchmark** As shown in Table 2, ACE delivers strong improvements on financial analysis benchmarks. In the offline setting, when provided with ground-truth answers from the training split, ACE surpasses ICL, MIPROv2, and GEPA by clear margins (an average of 10.9%), showing that structured and evolving contexts are particularly effective when tasks require precise domain knowledge ( _e.g.,_ financial concepts, XBRL rules) that goes beyond fixed demonstrations or monolithic optimized prompts. In the online setting, ACE continues to exceed prior adaptive methods such as DC by an average of 6.2%, further confirming the benefit of agentic context engineering for accumulating reusable insights across specialized domains. 

Moreover, we also observe that when ground-truth supervision or reliable execution signals are absent, both ACE and DC may degrade in performance. In such cases, the constructed context can be polluted by spurious or misleading signals, highlighting a potential limitation of inference-time adaptation without reliable feedback. This suggests that while ACE is robust under rich feedback ( _e.g.,_ code execution results or formula correctness in agent tasks), its effectiveness depends on the 

> 1We mention IBM CUGA as a rough contextual reference to show that ACE operates in a similar performance range on the AppWorld leaderboard. It is not used as a methodological baseline, and we do not make direct comparisons. CUGA’s internal design differs from ACE’s context-adaptation focus, and all baselines are evaluated under identical setups to isolate methodological effects rather than agent-engineering choices. 

8 

Published as a conference paper at ICLR 2026 

availability of signals that allow the Reflector and Curator to make sound judgments. We return to this limitation in §5. 

**Analysis: Medical and Text-to-SQL Benchmark** While this subsection focuses on finance as a detailed case study, ACE is not finance-specific: we also see consistent gains on other domainspecific tasks, including medical reasoning and text-to-SQL, suggesting that the same playbookstyle context adaptation transfers across domains. Full results are reported in Appendix §A.2. 

## 4.5 GENERALIZATION ACROSS LLMS 

Table 1 and Table 2 use our default backbone (DeepSeek-V3.1), but ACE is not specific to this model. We can swap in other LLMs without changing the algorithm or prompts, and still see consistent gains on AppWorld and Finance benchmarks. Appendix §A.1 reports full results on GPTOSS-120B, GPT-5.1, and Llama-3.3-70B-Instruct, where ACE improves over the corresponding base agents or models. These results suggest ACE is a generalizable method for test-time context evolution across LLM families. 

## 4.6 ABLATION STUDY AND SENSITIVITY ANALYSIS 

**Ablation Study** Table 3 reports ablation studies on AppWorld, analyzing how individual design choices of ACE contribute to effective context adaptation. We examine three factors: (1) _the Reflector with iterative refinement_ , our addition to the agentic framework beyond Dynamic Cheatsheet, (2) _multi-epoch adaptation_ , which refines contexts over training samples multiple times, and (3) _offline warmup_ , which initializes the context through offline adaptation before online adaptation begins. Additionally, we study the effect of _incremental context update_ and why it is a key enabler for ACE’s performance gain in Appendix §A.5. 

|**Method**<br>**GT Labels**|**Test-Normal**<br>**TGC**_↑_<br>**SGC**_↑_|**Test-Challenge**<br>**Average**<br>**TGC**_↑_<br>**SGC**_↑_|**Test-Challenge**<br>**Average**<br>**TGC**_↑_<br>**SGC**_↑_|
|---|---|---|---|
|DeepSeek-V3.1 as Base LLM||||
|ReAct|63.7<br>42.9|41.5<br>21.6|42.4|
|Offline Adaptation||||
|ReAct+ ACE w/o Refector or multi-epoch<br>✓<br>ReAct+ ACE w/o multi-epoch<br>✓<br>ReAct+ ACE<br>✓|70_._8+7_._1<br>55_._4+12_._5<br>72_._0+8_._3<br>60_._7+17_._8<br>76_._2+12_._5<br>64_._3+21_._4|55_._9+14_._4<br>38_._1+17_._5<br>54_._9+13_._4<br>39_._6+18_._0<br>57_._3+15_._8<br>39_._6+18_._0|55_._1+12_._7<br>56_._8+14_._4<br>59_._4+17_._0|
|Online Adaptation||||
|ReAct+ ACE<br>✗<br>ReAct+ ACE + offine warmup<br>✗|67_._9+4_._2<br>51_._8+8_._9<br>69_._6+5_._9<br>53_._6+10_._7|61_._4+19_._9<br>43_._2+21_._6<br>66_._0+24_._5<br>48_._9+27_._3|56_._1+13_._7<br>59_._5+17_._1|



Table 3: **Ablation Studies on AppWorld.** We study how particular design choices of ACE (iterative refinement, multi-epoch adaptation, and offline warmup) could help high-quality context adaptation. 

**Robustness to Reflection Quality** ACE is robust to reflection quality: it remains effective with a much weaker Reflector and shows only modest additional gains from stronger reflectors, and it degrades gracefully under noisy/harmful reflections, staying above the base model except under fully adversarial updates every iteration. Full experiment results are in Appendix §A.4. 

**Sensitivity to Hyperparameter Choice** ACE’s gains are stable across a wide range of reasonable hyperparameter settings ( _e.g.,_ Reflector refinement rounds, number of adaptation epochs, and growand-refine thresholds): performance changes are modest, and ACE consistently remains above the corresponding baselines. Full discussion and detailed results are reported in Appendix §A.6. 

## 4.7 COST AND SPEED ANALYSIS 

Due to its support for incremental, “delta” context updates and non-LLM-based context merging and de-duplication, ACE demonstrates particular advantages in reducing the cost (in terms of the number of rollouts or the amount of dollar cost for token ingestion/generation) and latency of adaptation. 

As examples, on the offline adaptation of AppWorld, ACE achieves 82.3% reduction in adaptation latency and 75.1% reduction in the number of rollouts as compared to GEPA (Table 4(a)). On 

9 

Published as a conference paper at ICLR 2026 

the online adaptation of FiNER, ACE achieves 91.5% reduction in adaptation latency and 83.6% reduction in token dollar cost for token ingestion/generation as compared to DC (Table 4(b)). 

|**Method**<br>**Latency (s)**_↓_<br>**# Rollouts**_↓_<br>ReAct + GEPA<br>53898<br>1434<br>ReAct + ACE<br>9517(-82.3%)<br>357(-75.1%)|**Method**<br>**Latency (s)**_↓_<br>**Token Cost ($)**_↓_|
|---|---|
||DC (CU)<br>65104<br>17.7<br>ACE<br>5503(-91.5%)<br>2.9(-83.6%)|



(a) **Offline** (AppWorld). (b) **Online** (FiNER). 

Table 4: **Cost and Speed Analysis.** We measure the context adaptation latency, number of rollouts, and dollar costs of ACE against GEPA (offline) and DC (online). 

**Fine-Grained Cost Analysis** We conduct a fine-grained cost analysis of ACE and GEPA (as a representative baseline). On AppWorld, ACE is substantially cheaper during offline adaptation, reducing input/output token usage by **80.8%/83.6%** vs. GEPA: ACE avoids GEPA’s prompt-validation loop and replaces repeated full rewrites with localized delta updates. At evaluation time, while ACE may use more _raw_ input tokens due to a richer playbook, this does not necessarily translate to higher _billed_ serving cost because a large fraction of the context is reused by KV caching; we quantify this effect in the next paragraph. Full results, including a component-wise token breakdown for both methods, are in Appendix §A.3. 

**KV Cache Reuse: Longer Context** = **Higher Serving Cost** Although ACE produces longer contexts than methods such as GEPA, this does not translate to linearly higher inference cost or GPU memory usage. Modern serving infrastructures are increasingly optimized for long-context workloads through techniques such as the reuse (Gim et al., 2024; Yao et al., 2025), compression (Liu et al., 2024c;b), and offload (Lee et al., 2024; Li et al., 2025) of KV cache. These mechanisms allow frequently reused context segments to be cached locally or remotely, avoiding repetitive and expensive prefill operations. Ongoing advances in ML systems suggest that the amortized cost of handling long contexts is likely to decrease, making context-rich approaches like ACE increasingly practical in deployment. In our prompt-caching study with the OpenAI API (GPT-5.1), we find that ACE achieves _high cache reuse_ : **91.8%** of input tokens are served from cache during evaluation stage, which reduces billed input-token cost by **82.6%** relative to counting raw context tokens. 

## 5 DISCUSSION 

**Implications for Online and Continual Learning** Online and continual learning are key research directions in machine learning for addressing issues like distribution shifts (Koh et al., 2021; Gulrajani & Lopez-Paz, 2021) and limited training data (Pan & Yang, 2010; Hutchinson et al., 2017; Zhuang et al., 2019). ACE offers a flexible and efficient alternative to conventional model finetuning, as adapting contexts is generally cheaper than updating model weights (Brown et al., 2020; Lester et al., 2021; Li & Liang, 2021; Hu et al., 2022). Moreover, because contexts are humaninterpretable, ACE enables _selective unlearning_ (Cao & Yang, 2015; Bourtoule et al., 2021; Liu et al., 2025a), whether due to privacy or legal constraints (gdp, 2016; ccp, 2018), or when outdated or incorrect information is identified by domain experts. These are promising directions for future work, where ACE could play a central role in advancing continuous and responsible learning. 

**Limitations and Challenges** A limitation of ACE is its reliance on a reasonably strong Reflector: if the Reflector fails to extract meaningful insights from generated traces or outcomes, the constructed context may become noisy or even harmful. In domain-specific tasks where no model can extract useful insights, the resulting context will naturally lack them. This dependency is similar to Dynamic Cheatsheet (Suzgun et al., 2025), where the quality of adaptation hinges on the underlying model’s ability to curate memory. We also note that not all applications require rich or detailed contexts. Tasks like HotPotQA (Yang et al., 2018) often benefit more from concise, high-level instructions ( _e.g.,_ how to retrieve and synthesize evidence) than from long contexts. Similarly, games with fixed strategies such as Game of 24 (Suzgun et al., 2025) may only need a single reusable rule, rendering additional context redundant. Overall, ACE is most beneficial in settings that demand detailed domain knowledge, complex tool use, or environment-specific strategies that go beyond what is already embedded in model weights or simple system instructions. 

10 

Published as a conference paper at ICLR 2026 

## ACKNOWLEDGEMENT 

We thank the anonymous reviewers and area chair for their constructive feedback, which improved this paper. Qizheng Zhang is supported by NSF award CNS-2211384 and DARPA award TFAWIHR00112520038. We also thank Lakshya A Agrawal, Xuekai Zhu, Yuhan Liu, Junchen Jiang, and Azalia Mirhoseini for helpful discussions. 

## ETHICS STATEMENT 

This work does not raise specific ethical concerns. Our contributions focus on developing algorithms and system frameworks for effective context adaptation in large language models (LLMs). All experiments are conducted on publicly available benchmarks with open-source models, without involving human subjects, sensitive data, or privacy-related information. No potential conflicts of interest are present. 

## REPRODUCIBILITY STATEMENT 

Our code is available at github.com/ace-agent/ace. We provide detailed descriptions of our experimental setup, including datasets, benchmarks, evaluation metrics, baselines, and hyperparameter choices. Additional details, such as prompts for large language models and extended experimental settings, are included in the appendix. With this information, readers with reasonable computational resources should be able to reproduce our results.