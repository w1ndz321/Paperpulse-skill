# **Agentic Harness Engineering: Observability-Driven Automatic Evolution of Coding-Agent Harnesses** 

**Jiahang Lin[1]** _[∗‡]_ **, Shichun Liu[1]** _[∗‡]_ **, Chengjun Pan[2]** _[∗‡]_ **, Lizhi Lin[3] , Shihan Dou[1]** , **Xuanjing Huang[1]** , **Hang Yan[3]** , **Zhenhua Han[3]** _[†]_ , **Tao Gui[1]** _[†]_ 1Fudan University 2Peking University 3Shanghai Qiji Zhifeng Co., Ltd 

## **Abstract** 

Harnesses are now central to coding-agent performance, mediating how models interact with tools and execution environments. Yet harness engineering remains a manual craft, because automating it faces a heterogeneous action space across editable components, voluminous trajectories that bury actionable signal, and edits whose effect is hard to attribute. We introduce **A** gentic **H** arness **E** ngineering ( **AHE** ), a closed loop that addresses these challenges through three matched observability pillars: ❶ _component observability_ gives every editable harness component a file-level representation so the action space is explicit and revertible; ❷ _experience observability_ distills millions of raw trajectory tokens into a layered, drill-down evidence corpus that an evolving agent can actually consume; and ❸ _decision observability_ pairs every edit with a self-declared prediction, later verified against the next round’s task-level outcomes. Together, these pillars turn every edit into a falsifiable contract, so harness evolution proceeds autonomously without collapsing into trial-and-error. Empirically, ten AHE iterations lift pass@1 on Terminal-Bench 2 from 69.7% to 77.0%, surpassing the human-designed harness Codex-CLI (71.9%) and the self-evolving baselines ACE and TF-GRPO. The frozen harness transfers without re-evolution: on SWE-bench-verified it tops aggregate success at 12% fewer tokens than the seed, and on Terminal-Bench 2 it yields +5 _._ 1 to +10 _._ 1 pp cross-family gains across three alternate model families, indicating the evolved components encode general engineering experience rather than benchmark-specific tuning. Ablations localize the gain to tools, middleware, and long-term memory rather than the system prompt, suggesting factual harness structure transfers while prose-level strategy does not. These results position observability-driven evolution as a practical pathway to keep coding-agent harnesses continually improving alongside their base models. 

## **1 Introduction** 

Coding agents are increasingly deployed on long-horizon software-engineering tasks, with measurable progress on issue resolution over real-world code repositories [14, 46, 7] and multi-step terminal workflows [21]. In practice, such progress relies not only on the underlying language model, but equally on the surrounding engineering components: the system prompt that shapes work style, the tools that expose the file system and shell, and the middleware that controls context, execution, and recovery. This collection of model-external, editable components is collectively referred to as the agent’s _harness_ [29, 18, 42, 45, 33, 31]. 

> _∗_ Equal contributions. _†_ Corresponding authors. _‡_ Work done during an internship at Shanghai Qiji Zhifeng Co., Ltd. Code: `https://github.com/china-qijizhifeng/agentic-harness-engineering` 

Preprint. 

**==> picture [377 x 183] intentionally omitted <==**

**----- Start of picture text -----**<br>
80 AHE best-so-far (3) Cross-step risk monitor: (4) Post-success hard-block +<br>AHE pass@1 per iteration observe command sequences pre-turn risk salience<br>[middleware] [tool + middleware]<br>TFGRPO (72.3)<br>78 Codex (71.9)<br>ACE (68.9)<br>76<br>74<br>72<br>(1) Contract-first workflow (2) Publish-state guard: protect<br>+ tunable shell timeout verified state post-success<br>70 [prompt + tool] [prompt + tool]<br>68<br>1 2 3 4 5 6 7 8 9 10<br>Automatic evolution iteration<br>Score on Terminal-Bench (%)<br>**----- End of picture text -----**<br>


Figure 1: **AHE evolves a bash-only seed past every human-designed and self-evolving baseline on Terminal-Bench 2.** All three role agents share one base model, isolating the gain to harness edits rather than analyzer or editor capability. 

Harness design materially shifts task completion on long-horizon coding benchmarks, even with the base model held fixed [40, 42], making harness engineering a first-class lever for improving coding agents. Moreover, the optimal harness is model-specific: a harness tuned for one base model often underperforms on another and must be re-adapted as the base model changes. In current practice, this adaptation is performed manually—developers inspect trajectories, identify recurring failure patterns, and hand-craft edits across prompts, tools, middleware, and skills. Yet as base models advance rapidly [39, 38, 44, 6, 35, 36], this manual loop struggles to keep pace, creating a widening gap between model capability and the harness needed to realize it [33]. 

An intuitive direction is to automate this loop with an _evolution agent_ that optimizes harness components based on experience [1, 49, 4]. However, few existing approaches jointly evolve the full set of editable components [16]; most focus on a single component, typically the prompt [32, 50, 20], skills [19, 43], or an in-context playbook [49]. Jointly evolving multiple components end-to-end faces two structural obstacles: long, unstructured trajectories yield little actionable signal, and tightly coupled harness frameworks make edits beyond the prompt error-prone. This leaves the central question of agent-driven harness evolution open: _How can an evolution agent_ _**jointly** and_ _**stably** evolve all editable components of a coding agent’s harness?_ 

Our central insight is that this question is bottlenecked by _observability_ , not by agent capability: once the evolution agent receives structured context over a clear action space, it can reliably converge on better harness designs [34, 53]. We implement this in **A** gentic **H** arness **E** ngineering ( **AHE** , Figure 2), a closed loop driven by three observability pillars: ❶ _component observability_ via a decoupled harness that exposes seven editable component types as files, so each failure pattern maps cleanly to a single component class; ❷ _experience observability_ via a layered, drill-down evidence corpus distilled from millions of raw trajectory tokens, so the evolver consumes structured root causes rather than raw logs; and ❸ _decision observability_ via a change manifest that pairs every edit with a self-declared prediction, later verified against the next round’s task-level outcomes, so each edit becomes a falsifiable contract and ineffective ones are reverted at file granularity. 

We empirically validate AHE on Terminal-Bench 2[21]: ten iterations lift pass@1 from 69.7% to 77.0%, surpassing the human-designed Codex CLI [25] and the self-evolving baselines ACE [49] and TF-GRPO [4]. Without further evolution, the frozen harness transfers to SWE-bench-verified [14], and across three alternate base-model families it yields consistent pass@1 gains of +5 _._ 1 to +10 _._ 1 pp, with the largest on bases further from saturation, suggesting that AHE encodes coordination patterns that less-saturated models lean on more heavily. A component ablation pinpoints where this gain lives: tools, middleware, and long-term memory each carry the improvement on their own, while the system prompt alone regresses, indicating that factual harness structure transfers across tasks and models whereas prose-level strategy does not. 

2 

This paper makes three contributions: 

- We formulate _agent-driven harness evolution_ for coding agents and propose **AHE** , which identifies _observability across components, trajectories, and decisions_ as the design pivot and turns every harness edit into a falsifiable, file-level contract through three observability pillars: a decoupled component substrate, a layered trajectory-distillation pipeline, and a change manifest whose self-declared predictions are verified by next-round task deltas. 

- We empirically show that AHE lifts pass@1 on Terminal-Bench 2 from 69.7% to 77.0%, surpasses human-designed and automated baselines, and produces a frozen harness that transfers across benchmarks and base-model families. 

- Our analysis reveals two limits of agent-driven evolution: harness components interact nonadditively, so stacking effective edits caps the aggregate gain; and the loop’s self-attribution is reliable for fixes but blind to regressions, pinpointing regression foresight as the clearest direction for future self-evolution loops. 

## **2 Related Work** 

## **2.1 Harness Engineering and Evaluation for Coding Agents** 

Harness engineering refers to the practice of designing the system surrounding the model, including its tools, interfaces, memory, execution constraints, and feedback loops, which together shape what an agent can do on long-horizon tasks [29, 18, 40, 3, 33, 31]. Concretely, the harness mediates how the model perceives and acts on its environment: it exposes the action and observation interfaces over which tool-augmented reasoning unfolds [3], custom agent-computer interfaces for repository navigation, file editing, and command execution [45], as well as sandboxed execution and orchestration support that keep long-horizon runs reproducible [42]. 

Verifying that such systems actually help has driven the parallel maturation of coding-agent evaluation along two axes: task horizon and environmental realism. Coverage extends from shorthorizon function-level benchmarks focused on contamination and freshness control [52, 12], through repository-scale executable patch resolution [14, 46, 7], to multi-hour, terminal-driven workflows that exercise long-horizon, realistic execution [22, 5, 21]. A parallel infrastructure track packages executable runtimes and verifiers around these benchmarks [28, 13, 47], whose attention to reproducible, traceable, and verifiable execution directly motivates the observation system AHE builds on. 

## **2.2 Automated Optimization of LLM Agents** 

Approaches to automated agent optimization differ in what evidence the optimizer observes and what it can edit. Some revise the agent’s own outputs through episodic critique and reflection [20, 32, 9]. Others target prompts and instructions [15]: structured playbooks [49], semantic-advantage priors [4], jointly optimized instruction-demonstration pipelines for multi-stage programs [27], and reflective updates driven by Pareto-frontier traces [1]. A separate line edits program structure itself, in the form of skill libraries [41], scored program and agent archives evolved through mutation [24, 11], and graph-structured workflows searched or learned from rollouts [48, 51]. 

AHE tunes the full harness as a combinatorial whole rather than a single editable surface, so crosscomponent trade-offs become legible to the optimizer. It also keeps the human prior minimal, leaving methodology for the optimizer to discover from rollouts rather than fixing it by hand. We describe the substrate, trajectory analysis, and iteration that realize these choices in Section 3. 

## **3 Method** 

AHE turns harness optimization into a closed loop driven by another agent, with the base model held fixed and only the explicit harness edited. Our design principle is that every phase of this loop must be _observable_ : AHE faithfully records the artifacts each phase produces (the harness components an iteration writes, the rollout trajectories it generates, the edit decisions it commits) and represents them in structured, layered forms that another agent can read and act on. 

3 

**==> picture [318 x 203] intentionally omitted <==**

**----- Start of picture text -----**<br>
I. Component<br>Observability<br>NexAU Harness<br>System<br>Prompt<br>~10M tokens<br>Coding Agent Raw trace Evolve Agent<br>Skills Tools Middleware Agent<br>Debugger<br>Sub-agent Memory ~10K tokens<br>Environment Overview II. Experience<br>Observability<br>History<br>III. Decision<br>Evidence Modify Component<br>Observability<br>**----- End of picture text -----**<br>


Figure 2: **The AHE pipeline links three observable surfaces into one closed loop.** Components, rollout experience, and edit decisions each surface as structured artifacts another agent reads, and every edit becomes a falsifiable prediction the next round verifies. 

Three observability layers implement this principle. **Component observability** (§3.1) is realized by a decoupled, file-level harness substrate that maps each failure pattern to a single component class. **Experience observability** (§3.2) is realized by a layered evidence corpus distilled from raw rollouts and indexed for drill-down access. **Decision observability** (§3.3) is realized by a change manifest that pairs every edit with a self-declared prediction the next round verifies. The three layers compose into the iteration of Algorithm 1, which runs unattended round after round. 

## **3.1 NexAU: an editable, decoupled harness substrate** 

We instantiate the harness _H_ on the NexAU framework [23, 37], which exposes seven orthogonal component types as explicit files at fixed mount points in a single workspace: system prompt, tool description, tool implementation, middleware, skill, sub-agent configuration, and long-term memory. The component types are loosely coupled, so adding a middleware does not require editing the system prompt, and adding a skill does not require touching any tool. 

This decoupling is what realizes **component observability** : each failure pattern maps to a single component class, giving the evolve agent a clean action space and localizing every pass-rate change to one file rather than scattering it across hundreds of lines of unstructured prompt prose. Each logical edit becomes one commit on the workspace’s git history, which yields file-level diffs and rollback granularity for free. 

Our seed harness _H_ 0 is deliberately minimal: a single shell-execution tool, no middleware, no skills, no sub-agents. A seed already fitted to the target benchmark would contaminate every subsequent edit’s attribution, since we could not tell whether a gain came from the loop or from the seed. The minimal seed forces every component AHE adds to earn its place against measured rollouts. 

## **3.2 Agent Debugger: layered trajectory evidence** 

We generate _k_ traces for each task in a benchmark using a harness _H_ , which may contain errors resulting from the deficiencies of the harness that can be acted on, but scattered across millions of tokens of raw messages. To extract insights from agent trajectories and enable **experience observability** , we apply Agent Debugger [17] framework to use an agent to explore trajectories framed as a navigable, file-based environment where each trajectory message lives in its own file and is reached through generic shell and scripting tools. Traces with the same query are placed in one environment, and the debugger is required to analyze the root cause of the failure or the success pattern, which is stored in _per-task analysis_ report for each task. The analysis also includes pass/fail 

4 

status of the task to ground the Evolve Agent. Finally, a _benchmark-level overview_ is aggregated from every report into a single document as an entry point for every iteration. 

In addition to these reports, we also provide _original_ traces in case the agents need to verify the claims in the reports. The traces are provided both in raw form and lightly processed to remove unnecessary content. All of these content is provided as files allowing _progressive disclosure_ [30] which saves on tokens and enable better agent decisions. 

## **3.3 Evolve Agent: evidence-driven, auditable edits** 

The Evolve Agent closes the AHE loop. In each round it reads the layered evidence corpus produced by the Agent Debugger, decides which harness components to add, modify, or remove, applies those edits to the workspace, and records the reasoning behind every edit. Two constraints govern these edits, and together they realize **decision observability** : every edit becomes a falsifiable, file-level claim recorded in a versioned manifest, and the next round’s verdict either confirms or reverts it. 

The first constraint is _controllability_ : the Evolve Agent writes only inside the harness workspace, while the runs directory, tracer, verifier, and LLM configuration are read-only, and the seed system prompt (Appendix B.1) is marked non-deletable. These restrictions block the shortcuts an unconstrained self-modifier would take, such as disabling the verifier, swapping the model, or raising the reasoning budget, and keep every recorded gain attributable to harness edits. 

The second constraint is that every change is _evidence-driven_ and ships with a recorded prediction. Each edit attaches a manifest entry that names the failure evidence, the inferred root cause, the targeted fix, and a predicted impact comprising both expected fixes and at-risk regressions; this manifest is the loop’s evidence ledger (see Appendix B.2). In the next round, the loop intersects the predicted-fix and predicted-regression sets with the observed task-level deltas to produce a per-edit verdict. Each edit thereby becomes falsifiable by the next evaluation, which replaces rationale-driven self-justification with a measurable contract between rounds. 

**Algorithm 1** AHE outer loop. 

**Require:** seed harness _H_ 0, base model _M_ , benchmark _D_ , rollouts per task _k_ , max iterations _N_ 1: _H_ best _← H_ 0 

2: **for** _t_ = 1 to _N_ **do** 4:3: _TT_ � _tt ←←_ CROLLOUTLEAN( _Tt_ () _M, Ht−_ 1 _, D, k_ ) _▷_ phase 2: drop base64, dedup tool output _▷_ phase 1: _k_ rollouts per task 5: **if** _t ≥_ 2 **then** _▷_ phase 3: attribute prior manifest, then rollback 6: _Vt ←_ ATTRIBUTE( _Ct−_ 1 _, Tt−_ 1 _, Tt_ ) 7: _Ht−_ 1 _←_ ROLLBACK( _Ht−_ 1 _, Vt_ ) 8: **else** 9: _Vt ←∅_ 10: **end if** 11: _Rt ←_ AGENTDEBUGGER( _T_[�] _t_ ) _▷_ phase 4: layered distillation 12: ( _Ht, Ct_ ) _←_ EVOLVE( _Ht−_ 1 _, Rt, Vt_ ) _▷_ phase 5: workspace edits + new manifest 13: COMMIT( _Ht, Ct, t_ ) _▷_ phase 6: tag iteration in git 14: **if** PASS@1( _Tt_ ) _>_ PASS@1( _H_ best) **then** _H_ best _← Ht_ 15: **end if** 16: **end for** 17: **return** _H_ best 

Algorithm 1 composes the three substrates into one iteration: rollout, clean, attribute the prior manifest and revert rejected edits, distill, edit, commit. We run _k ≥_ 2 rollouts per task so each task carries a pass-rate signal, which stabilizes pass@1 and lets partial-pass tasks anchor comparative diagnosis. Attribution runs _before_ distillation, so its verdict lands inside the evidence corpus and binds each prior manifest entry as a contract rather than a rationale. A one-shot explore agent (Appendix B.3) runs in parallel with iteration 1 to seed a small number of reusable skills from the NexAU source and public coding-agent references. These skills receive no special protection: from iteration 2 onward the Evolve Agent may keep, refine, or remove them based on observed rollouts. 

5 

## **4 Experiments** 

We organize our empirical study around three questions: where AHE sits on the map of existing approaches to harness design, whether what it produces is portable beyond its optimization target, and what inside the loop drives the gain. 

## **Research Questions** 

1. **RQ1** (§4.2) **: Why agentic harness engineering, rather than human-engineered harnesses or other automated methods?** 

2. **RQ2** (§4.3) **: Does agentic harness engineering overfit to its optimization target?** 

3. **RQ3** (§4.4) **: What inside AHE drives its gains, and how reliable is the loop’s self-attribution?** 

## **4.1 Setup** 

**Evaluation.** We drive evolution on the full 89 tasks of Terminal-Bench 2 [21], split as 4 easy, 55 medium, and 30 hard, with per-task timeout extended to 1 hour. For cross-benchmark transfer we evaluate the AHE harness on SWE-bench-verified [14], 500 tasks across seven repositories. We report two metrics per configuration: _pass@1_ , the mean binary success rate over _k_ rollouts per task; and _tokens/trial_ , the mean per-trial total of prompt plus completion tokens across all LLM calls, in thousands. Infrastructure-aborted or timed-out trials count as failures under pass@1 (matching the official terminal-bench leaderboard) and are excluded from token means to avoid truncated figures. Runtime infrastructure (framework, dispatcher, sandbox, tracer, and concurrency) is detailed in Appendix A. 

**Models.** For both the evolution loop and the main experiment of §4.2, all three role agents (the Code Agent, the Agent Debugger, and the Evolve Agent) share one base model, GPT-5.4 [26] at the high reasoning setting. For cross-model transfer (§4.3), we re-evaluate the Code Agent on five alternate bases: GPT-5.4 at medium and xhigh reasoning, qwen-3.6-plus [38, 44], gemini-3.1-flashlite-preview [8], and deepseek-v4-flash [6]. 

## **4.2 RQ1: Main Results** 

We run a single AHE campaign of ten iterations from the bash-only **NexAU0** seed (§3.1), with _k_ =2 rollouts per task per iteration on TerminalBench 2, finishing in roughly 32 hours; the best resulting configuration is reported as **AHE** . The two self-evolve baselines ACE [49] and TFGRPO [4] start from the same NexAU0 seed. 

**AHE outperforms both human-designed and self-evolve baselines.** AHE outperforms every baseline on our panel: three human-designed harnesses, opencode [2], terminus-2 [10], and Codex-CLI [25], and the two self-evolve baselines ACE and TF-GRPO. Figure 1 shows the gain accumulates across iterations, with continued evolution pushing pass@1 further above the NexAU0 seed. By difficulty, the only exception is the Hard tier, where AHE marginally trails Codex-CLI. We trace this gap to interference be- 

Table 1: Pass@1 on Terminal-Bench 2 across 89 tasks, by official difficulty. **NexAU0** is the shared seed; ACE, TF-GRPO, and **AHE** are three selfevolution loops layered on top of it. Bold marks the best per column; ties are all bold. 

||Method|All<br>Easy|Med.|Hard|
|---|---|---|---|---|
|||89<br>4|55|30|
||**Human-designed harness**<br>opencode<br>47.2%<br>75.0%<br>terminus-2<br>62.9%<br>75.0%<br>Codex<br>71.9%<br>75.0%<br>**Self-evolved from NexAU0**<br>NexAU0<br>69.7%<br>87.5%<br>ACE<br>68.9%<br>91.7%<br>TF-GRPO<br>72.3%<br>**100.0%**<br>**AHE**<br>**77.0%**<br>**100.0%**||52.7%<br>74.5%<br>80.0%<br>78.2%<br>78.2%<br>79.4%<br>**88.2%**|33.3%<br>40.0%<br>**56.7%**<br>51.7%<br>48.9%<br>55.6%<br>53.3%|



tween AHE’s components on long-horizon tasks rather than to a missing capability: swapping AHE’s long-term memory alone into the NexAU0 seed, without the other AHE components, already surpasses Codex-CLI on Hard (§4.4.1). 

6 

Table 2: Cross-benchmark transfer on SWE-bench-verified. ACE, TF-GRPO, and **AHE** share the **NexAU0** seed and differ only in their self-evolution loop; all four columns run on GPT-5.4. AHE and the two self-evolve baselines are evolved on Terminal-Bench 2 and evaluated without in-domain re-evolution. Per-column bold marks the best; ties are all bold. 

|Repo<br>_N_|Success rate_↑_<br>Tokens k_↓_<br>ACE<br>TF-GRPO<br>NexAU0<br>**AHE**<br>ACE<br>TF-GRPO<br>NexAU0<br>**AHE**|
|---|---|
|All<br>500<br>74.6%<br>74.2%<br>75.2%<br>**75.6%**<br>679<br>582<br>526<br>**461**||
|django<br>231<br>79.2%<br>78.8%<br>79.2%<br>**81.0%**<br>707<br>583<br>527<br>**484**<br>sympy<br>75<br>69.3%<br>68.0%<br>**70.7%**<br>**70.7%**<br>602<br>572<br>494<br>**479**<br>sphinx-doc<br>44<br>61.4%<br>65.9%<br>68.2%<br>**70.5%**<br>990<br>848<br>731<br>**656**<br>matplotlib<br>34<br>70.6%<br>70.6%<br>**73.5%**<br>**73.5%**<br>622<br>530<br>486<br>**391**<br>scikit-learn<br>32<br>**93.8%**<br>**93.8%**<br>**93.8%**<br>87.5%<br>451<br>378<br>307<br>**257**<br>pydata<br>22<br>**77.3%**<br>**77.3%**<br>**77.3%**<br>72.7%<br>563<br>516<br>386<br>**338**<br>astropy<br>22<br>**59.1%**<br>**59.1%**<br>54.5%<br>50.0%<br>546<br>470<br>667<br>**277**||



**Prompt-only self-evolution misses the components that carry AHE’s gain.** The gaps to ACE and TF-GRPO trace to a layer mismatch. ACE distills natural-language playbooks the agent reads in-context, and TF-GRPO is a trajectory-feedback variant of GRPO that reinforces successful tool sequences; starting from the same NexAU0 seed as AHE, neither method opens the surrounding scaffolding to edits. AHE jointly evolves system prompt, tools, middleware, and long-term memory across iterations, and §4.4.1 quantifies which of these layers carries the improvement: swapping in AHE’s tools, middleware, or long-term memory alone yields +3 _._ 3, +2 _._ 2, and +5 _._ 6 pp, while the system prompt alone is _−_ 2 _._ 3 pp. The harness components ACE and TF-GRPO never edit are exactly where the gain lives. 

## **4.3 RQ2: Transfer to Unseen Tasks and Base Models** 

AHE’s harness is evolved on Terminal-Bench 2 with GPT-5.4 high. We probe whether it encodes general coding-agent experience or overfits to that target by re-using the workspace as-is, without further evolution, in two off-target settings: a different task surface (SWE-bench-verified) and four alternate base models. 

**Cross-benchmark transfer.** We re-point the AHE harness at SWE-bench-verified against the seed and the two self-evolve baselines (NexAU0, ACE, TF-GRPO) under identical infrastructure (Table 2). 

ACE and TF-GRPO both regress below the untouched NexAU0 seed in aggregate success while spending 11% to 29% more tokens than the seed: the playbook ACE injects and the trajectory distribution TF-GRPO reinforces were distilled on terminal-bench traces and ride the prompt at every model call, so on a different task surface that text adds cost without reshaping the underlying policy. AHE instead achieves the highest aggregate, with the seed-relative gain concentrating on django and sphinx-doc, the two largest and most token-expensive repositories whose multi-step edit-and-verify loop matches the structure AHE’s tools, middleware, and long-term memory compress on TerminalBench 2. Marginal regressions appear only on the three smallest repositories, consistent with pass@1 variance on small repos exceeding the per-repo gain. AHE also cuts aggregate tokens by 32% against ACE, 21% against TF-GRPO, and 12% against the seed: encoding behavior in tools, middleware, and memory rather than in the prompt avoids the per-call re-derivation cost that prompt-only baselines pay. 

**Cross-model transfer.** We re-evaluate both the NexAU0 seed and AHE on the five alternate bases listed in §4.1. Figure 3 reports five positive pass@1 gains from +2 _._ 3 to +10 _._ 1 pp. 

Cross-family gains dominate within-family ones: deepseek-v4-flash moves +10 _._ 1 pp from 51 _._ 7% to 61 _._ 8%, qwen-3.6-plus +6 _._ 3 pp from 56 _._ 2% to 62 _._ 5%, and gemini-3.1-flash-lite-preview +5 _._ 1 pp from 36 _._ 5% to 41 _._ 6%, all above the +2 _._ 3 pp on GPT-5.4 medium and xhigh. We read this as bases further from saturation leaning more on the coordination patterns AHE has fixed inside tools, middleware, and long-term memory, while a stronger base re-derives the same coordination from its prompt at low marginal cost. 

7 

**==> picture [357 x 136] intentionally omitted <==**

**----- Start of picture text -----**<br>
+7.3<br>80 77.0 +2.3 NexAU0 seed<br>74.7 AHE (evolved on GPT-5.4 high)<br>+2.3 72.5<br>69.7<br>70 65.7 68.0 +10.1 +6.3<br>61.8 62.5<br>60 56.2<br>51.7<br>50<br>+5.1<br>41.6<br>40 36.5<br>GPT-5.4 med GPT-5.4 high GPT-5.4 xhigh gemini-3.1-flash-litedeepseek-v4-flash qwen-3.6-plus<br>pass@1<br>**----- End of picture text -----**<br>


Figure 3: Cross-model transfer on Terminal-Bench 2, 89 tasks. The AHE workspace evolved on GPT-5.4 high is re-evaluated on each base without further evolution, paired against the NexAU0 seed on the same base. 

Table 3: Component-level ablations on Terminal-Bench 2. Each “+ X only” row swaps a single AHE component into the NexAU0 seed: long-term memory, tool set, middleware, or system prompt. Per-column best is bolded. 

|Variant|All|Easy|Medium|Hard|
|---|---|---|---|---|
||89 tasks|4 tasks|55 tasks|30 tasks|
|NexAU0|69.7%|87.5%|78.2%|51.7%|
|+ memory only|75.3%|50.0%|83.6%|**63.3%**|
|+ tool only|73.0%|75.0%|87.3%|46.7%|
|+ middleware only|71.9%|**100.0%**|81.8%|50.0%|
|+ system_prompt only|67.4%|75.0%|78.2%|46.7%|
|**AHE**full|**77.0%**|**100.0%**|**88.2%**|53.3%|



Within one family the profile is non-monotone: +2 _._ 3 pp on medium, +7 _._ 3 pp on high from §4.2, and +2 _._ 3 pp on xhigh. AHE’s step budget and per-task timeout were fitted to GPT-5.4 high during evolution; medium has more time-per-step slack but loses a reasoning tier of raw capability, while xhigh pushes more trials past the per-task timeout, which our pass@1 convention (§4.1) counts as failures. Either direction discounts the gain. 

The load-bearing finding is that all five gains land positive: the AHE workspace is not specific to one provider’s idioms or one reasoning depth. Their magnitude tracks the evolution operating point rather than raw base capability, so we treat the timeout-budget coupling as a generalization hazard discussed in our Limitations section. 

## **4.4 RQ3: Analysis** 

We analyze the loop along two architectural choices that §3 places weight on: decomposed components (§4.4.1) and self-declared attribution (§4.4.2). 

## **4.4.1 RQ3a: where value accumulates across components** 

Table 3 decomposes the AHE gain at the component level. Each “+ X only” row takes the NexAU0 seed and swaps in one component from the fully evolved AHE configuration, namely long-term memory, tools, middleware, or system prompt, leaving the other three at their seed defaults. Three of the four single-component variants outperform the seed; the system-prompt swap is the only regression. 

**Each component owns a different failure surface.** Memory adds 12 boundary-case lessons (performance margin, queued-over-limit cancellation, evaluator-style closure, source-packaging layout); on Hard the lessons lift it above full AHE, while on Easy they reduce to superfluous reverification. Tools become a 1364-line shell that auto-surfaces contract hints from files near each 

8 

**==> picture [360 x 150] intentionally omitted <==**

**----- Start of picture text -----**<br>
Cross-iteration mean Random baseline Cross-iteration mean Random baseline<br>51.4%<br>50 11.8%<br>11.1%<br>40 10<br>33.7%<br>30<br>5.6% 5.4%<br>20 5<br>10.6%<br>10 6.5%<br>0 0<br>Fix Fix Regression Regression<br>precision recall precision recall<br>Value (%) Value (%)<br>**----- End of picture text -----**<br>


Figure 4: Cross-iteration mean precision and recall of the evolve model’s self-predictions across 9 evaluation rounds of the GPT-5.4 AHE loop on Terminal-Bench 2, alongside the random-prediction baseline. Left: fix predictions. Right: regression predictions. 

command; on Medium it lands within 0 _._ 9 pp of full AHE, while on Hard a built-in publish guard closes the loop too early. Middleware adds a finish-hook that forces one evaluator-isomorphic closure check; on Easy it clears every task, while on Hard it inflates turn count. The system prompt encodes 79 lines of universal discipline whose executability depends on the other three; inserted alone it scores _−_ 2 _._ 3 pp aggregate. 

**Components interact non-additively, capping the aggregate gain.** The three positive singlecomponent gains sum to +11 _._ 1 pp against full AHE’s +7 _._ 3 pp, and on Hard the memory-only variant exceeds full AHE: memory, middleware, and the system prompt all push toward the same closurestyle verification, so stacking them spends turns on redundant re-checks within the long-horizon budget. Since the evolve agent optimises an aggregate dominated by 55 Medium tasks, it converges to a Medium-heavy trade-off that returns part of the Hard memory effect, and we leave interaction-aware evolution to future work. 

## **4.4.2 RQ3b: how reliably the loop’s self-attribution tracks reality** 

Each evolution round, our evolve model produces a change manifest naming which Terminal-Bench 2 tasks it expects to **fix** in the next round and which it flags **at risk of regression** . We compare the round- _N −_ 1 prediction against the round- _N_ ground truth, computing standard precision and recall over the 89 tasks separately for fixes and regressions. 

**Evidence-driven targeting.** The fix panel of Figure 4 shows the evolve model’s targeting is evidence-driven rather than guesswork. Cross-iteration fix-precision of **33.7%** and fix-recall of **51.4%** sit roughly 5x above the random-prediction baselines of 6.5% and 10.6%, so each harness edit lands on a real, agent-anticipated target rather than on an arbitrary subset of the panel. 

**Regression blindness.** The regression panel tells the opposite story: cross-iteration regressionprecision of **11.8%** and regression-recall of **11.1%** sit only about 2x above their random baselines of 5.6% and 5.4%, so most upcoming regressions go unforeseen. The agent can justify why an edit should help, but it cannot reliably name the tasks the same edit is about to break, which is what produces the non-monotone steps in the evolution curve of §4.2. Closing this gap is the clearest direction for future self-evolution loops. Section D gives the per-round breakdown. 

## **5 Conclusion** 

We introduced **Agentic Harness Engineering (AHE)** , an observability-driven loop that turns a coding agent’s harness into a learnable adaptation surface while the base model remains fixed. AHE exposes components as files, distills rollouts into a layered evidence corpus, and binds each edit to a falsifiable next-round prediction; ten iterations lift pass@1 on Terminal-Bench 2 from 69.7% to 

9 

77.0%, and the frozen harness transfers to SWE-bench-verified and three alternate model families. We see harness-level evolution as a complementary axis to model-side training: an externalized, auditable surface where coding-agent experience can accumulate. 

## **Limitations** 

This work studies a promising but high-variance setting, and the scope of our claims should be interpreted accordingly. 

**Benchmark scope.** Our evaluation drives evolution on Terminal-Bench 2 and probes transfer on SWE-bench-verified. Even though the frozen harness transfers to a second task surface and to three alternate base-model families, broader programming languages, repository-scale deployments, and human-in-the-loop workflows remain untested. 

**Evolution operating point.** AHE’s step budget and per-task timeout were fitted to GPT-5.4 high during evolution, so cross-model transfer numbers conflate harness portability with operating-point coupling—within one family the gain is non-monotone across reasoning tiers (§4.3). Untangling these factors will require re-running the loop under multiple operating points. 

**Self-modification governance.** AHE bounds edits to a workspace, attributes every change in a versioned manifest, and rolls back ineffective edits at file granularity, but it does not provide a complete guardrail stack. Long-horizon harness cleanup and stronger misuse prevention remain incomplete, and AHE should be viewed as a controlled research prototype rather than a fully mature autonomous self-improvement system.