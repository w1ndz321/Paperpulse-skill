**==> picture [120 x 21] intentionally omitted <==**

## **Recursive Multi-Agent Systems** 

**Xiyuan Yang**[1,*] **, Jiaru Zou**[1,2,*][†] **, Rui Pan**[1] **, Ruizhong Qiu**[1] **, Pan Lu**[2] **, Shizhe Diao**[3] **, Jindong Jiang**[3] **, Hanghang Tong**[1] **, Tong Zhang**[1] **, Markus J. Buehler**[4] **, Jingrui He**[1][ �] **, James Zou**[2][ �] 

1UIUC 2Stanford University 3NVIDIA 4MIT 

*Equal Contribution, Alphabetical Order †Project Lead �Corresponding Authors 

**==> picture [11 x 11] intentionally omitted <==**

**Project Page: https://recursivemas.github.io** 

**Recursive or looped language models have recently emerged as a new scaling axis by iteratively refining the same model computation over latent states to deepen reasoning. We extend such scaling principle from a single model to multi-agent systems, and ask:** _**Can agent collaboration itself be scaled through recursion?**_ **To this end, we introduce RecursiveMAS, a recursive multi-agent framework that casts the entire system as a unified latent-space recursive computation. RecursiveMAS connects heterogeneous agents as a collaboration loop through the lightweight RecursiveLink module, enabling in-distribution latent thoughts generation and cross-agent latent state transfer. To optimize our framework, we develop an inner-outer loop learning algorithm for iterative whole-system co-optimization through shared gradient-based credit assignment across recursion rounds. Theoretical analyses of runtime complexity and learning dynamics establish that RecursiveMAS is more efficient than standard text-based MAS and maintains stable gradients during recursive training. Empirically, we instantiate RecursiveMAS under 4 representative agent collaboration patterns and evaluate across 9 benchmarks spanning mathematics, science, medicine, search, and code generation. In comparison with advanced single/multi-agent and recursive computation baselines, RecursiveMAS consistently delivers an average accuracy improvement of 8.3%, together with 1.2** × **–2.4** × **end-to-end inference speedup, and 34.6%–75.6% token usage reduction.** 

**==> picture [15 x 14] intentionally omitted <==**

## RecursiveMAS Scaling Law 

## **Sequential-Style (Light)** 

**==> picture [470 x 139] intentionally omitted <==**

**----- Start of picture text -----**<br>
Training Recursion Round Training Recursion Round Training Recursion Round Training Recursion Round<br>Collaboration Patterns<br>Accuracy (%)<br>Inference Recursion Round<br>**----- End of picture text -----**<br>


**==> picture [152 x 131] intentionally omitted <==**

**----- Start of picture text -----**<br>
Speed-up<br>1.6× 1.5×<br>1.5× 1.5×<br>1.4×<br>**----- End of picture text -----**<br>


**==> picture [168 x 125] intentionally omitted <==**

**==> picture [153 x 126] intentionally omitted <==**

Figure 1 | **Performance Landscape of RecursiveMAS across Training/Inference Recursion Depths (Top):** The _lightweight_ RecursiveMAS with sub-1.5B agents shows a clean scaling trend as recursion deepens. **Generalization across Common Collaboration Patterns (Bottom):** The _Scaled_ RecursiveMAS with stronger LLM agents (5-10B) seamlessly adapts to diverse multi-agent system structures. 

_Contact: jiaru@stanford.edu, xiyuany4@illinois.edu_ 

Recursive Multi-Agent Systems 

## **1. Introduction** 

To tackle complex tasks, a single language model often falls short due to limited capacity, myopic generation, or inefficient exploration of the solution space (Li et al., 2025b; Shojaee et al., 2025; Song et al., 2026). Once intelligence reaches a threshold, a natural direction is to treat individual models as specialized agents and organize them as a collaborative system (Tran et al., 2025; Xu et al., 2025). A multi-agent system (MAS) (Wang et al., 2025b; Wu et al., 2024) can scale performance by enabling individuals to work together and contribute complementary strengths. Consider a set of heterogeneous agents, each assigned a distinct role and expertise. The system can either arrange agents into a sequential pipeline (Gu et al., 2025; Qian et al., 2024) to progressively decompose and solve a problem, or engage and integrate multiple domain-specialized agents (Babu et al., 2025; Qian et al., 2025; Ye et al., 2025b) for the task. 

While MAS establishes a structural foundation, the next question is how to enable the system to evolve over time and adapt to different scenarios. Prior work has explored prompt-based adaptation (Shen et al., 2025; Zhang et al., 2025b; Zhou et al., 2025), where model interactions are improved through the iterative refinement of shared context. Although these updated prompts can help agents generate more aligned responses to the question, each agent itself cannot improve. A more principled line of work is to optimize agents through learning (Motwani et al., 2024; Subramaniam et al., 2025; Zhao et al., 2025). However, training entire agents inside the system is hard, as updating all model parameters is non-trivial (Hu et al., 2025), and the sequential dependency in text-based interactions introduces substantial latency when agents must wait for others to complete generation. 

Instead of improving each agent’s capabilities as a standalone component, we adopt a higher-level learning perspective and aim to co-evolve and scale the entire system as an integrated whole. We recast agent collaboration through the lens of recursive language models (RLMs) (Jolicoeur-Martineau, 2025; Zhang et al., 2025a; Zhu et al., 2025), where a shared set of layers is iteratively applied and optimized within a continuous latent space. In this view, the entire multi-agent system can be treated as a recursive computation, where each agent acts like an RLM layer, iteratively passing latent representations to the next and forming a looped interaction process. 

We call this new system-level agentic recursion framework **RecursiveMAS** . Without updating all model parameters, agents are connected and iteratively optimized solely via the lightweight **RecursiveLink** , a two-layer residual projection module for latent states transmission and refinement. An _inner RecursiveLink_ within each agent first consolidates the model’s ongoing latent thoughts between input and output spaces during auto-regressive generation. An _outer RecursiveLink_ then bridges hidden representations across heterogeneous agents built on different model types and sizes, enabling seamless cross-agent interaction. Together, all agents are chained in a unified loop to perform iterative latent collaboration, with only the last agent producing the textual output in the final recursion round. 

Correspondingly, we pair RecursiveMAS with an **Inner-Outer Loop** training paradigm for progressive co-optimization. The _inner loop_ provides a preliminary model-level warm start for each agent, by training its inner RecursiveLink to better align with latent thoughts generation. The _outer loop_ then trains the outer RecursiveLink across agents at the system-level, with gradients recursively backpropagated through the full computation traces over recursion rounds. By exposing each agent to the feedback of itself and others from previous rounds, RecursiveMAS learns to leverage RecursiveLink for iterative refinement of collaboration, thus enabling the entire system to optimize in a unified manner. 

To justify why recursion should occur in latent space rather than text-mediated interaction, we provide two theoretical analyses on runtime complexity and learning dynamics. From an architectural standpoint, RecursiveLink enables direct transformation of latent-space information, avoiding repeated decoding of intermediate agents with more efficient runtime complexity. From the learning perspective, 

2 

Recursive Multi-Agent Systems 

latent-space connections in RecursiveMAS maintain stable gradient propagation flow across recursion rounds during training, avoiding the gradient vanishing induced by text-based interactions. 

Empirically, we evaluate RecursiveMAS on 9 benchmarks spanning mathematics, science, medicine, search, and code generation. We instantiate RecursiveMAS with diverse model families, including Qwen3/3.5, LLama-3, Gemma3, and Mistral, and adapt our framework to 4 representative MAS collaboration scenarios: step-by-step sequential reasoning, mixture-of-experts collaboration, expert-tolearner knowledge distillation, and tool-integrated deliberation. As illustrated in Figure 1, compared with advanced recursive language models and MAS baselines, RecursiveMAS achieves an average accuracy improvement of 8.3%, while delivering 1.2×–2.4× inference speedup and reducing token usage by 34.6%–75.6%. In addition, RecursiveMAS is structure-agnostic and can generalize to various agent collaboration patterns with effective performance. Our additional detailed analyses of scaling laws with deeper recursion, RecursiveLink architectures, semantic distributions across recursions, and training cost further validate the efficiency and performance scalability of the RecursiveMAS. 

## **2. Preliminary** 

**Auto-regressive Generation in Latent Space.** Let _𝑓𝜃_ (·) denote a standard Transformer model (Vaswani et al., 2017) parameterized by _𝜃_ . Given a question _𝑥_ with corresponding input embeddings _𝐸_ = [ _𝑒_ 1 _, . . . , 𝑒𝑡_ ] ∈ ℝ _[𝑡]_[×] _[𝑑][ℎ]_ , the model computes the last-layer hidden state _ℎ𝑡_ through the forward pass. In standard auto-regressive decoding, _ℎ𝑡_ is projected to the vocabulary space to predict the next token. In contrast, latent generation keeps the recurrence entirely in continuous representation space by directly feeding the previously generated latent embedding _ℎ𝑡_ back into the next forward pass. Formally, the next latent generation at step _𝑡_ + 1 is: 

**==> picture [283 x 11] intentionally omitted <==**

We refer to the newly generated latent state _ℎ𝑡_ +1 as the model’s ongoing _latent thought_ . 

**Recursive Computation.** A recursive language model (RLM) increases reasoning depth by reusing the same transformation across recurrent steps. Consider a Transformer _𝑓𝜃_ with _𝐿_ layer blocks, denoted as _𝑓𝜃_ = M _𝐿_ ◦· · · ◦M1. Instead of passing the input through the _𝐿_ -layer stack only once to obtain the last representation, a recursive model reuses the same stack for _𝑛_ times of forward iterations, i.e., 

**==> picture [339 x 14] intentionally omitted <==**

The last round of latent representation _𝐻_[(] _[𝑛]_[)] is obtained through recursive refinement over the same shared Transformer layers, and is subsequently used for the final prediction. 

**LLM-based Multi-Agent Evolution.** We define a multi-agent system S (Tran et al., 2025; Zou et al., 2025) composed of _𝑁_ agents denoted as A = { _𝐴_ 1 _, . . . , 𝐴𝑁_ }, where each LLM agent _𝐴𝑖_ corresponds to _𝑓𝜃𝑖_ with its own last-layer representations _𝐻𝑖_ . We then denote the collective latent state of the system by H = { _𝐻_ 1 _, . . . , 𝐻𝑁_ }. Given any input problem _𝑥_ with the ground-truth _𝑦_ , the system S orchestrates interactions among agents to collaboratively produce a final prediction. With this setup in place, we now formalize the evolution of agents under recursive computation. 

## **Definition 2.1: Recursive Multi-Agent Evolution** 

_A_ _**recursive evolution** is the progressive refinement of_ H _, where each agent adjusts its latent representation through iterative interaction with others and its own reasoning state, so that the updated 𝐻_[(][1][)] _𝐻_[(][2][)] _𝐻_[(] _[𝑛]_[)] _system is better aligned for the given problem, i.e._ S[(][0][)] −−−−→ _Evolve_[S][(][1][)] −−−−→ _Evolve_[· · ·] −−−−→ _Evolve_[S][(] _[𝑛]_[)] _[.]_ 

**Collaboration Pattern.** As MAS architectures are generally not fixed and can vary across tasks, we do not restrict the collaboration pattern to a single style. In this paper, we consider four commonly 

3 

Recursive Multi-Agent Systems 

**==> picture [471 x 162] intentionally omitted <==**

**----- Start of picture text -----**<br>
Latent Thoughts of A1  (Last-layer Embs) Latent Thoughts of A2<br>Decode for Outputs<br>ℎ% ℎ%&' ℎ%&( … … (Final Round n)<br>Agent A1<br>(Recursion Rounds  < n)<br>Agent A1 Agent A2 … Agent AN<br>#! #" … ## ##%! ##%" Input EmbsA1-aligned  Contexts… … … Input EmbsA2-aligned<br>Input Contexts Inner Link<br>(Instruction, Question, etc.) Outer Link Inner Link Looping…<br>ℎ% ℎ%&' … … …<br>Condition Generated on A1<br>**----- End of picture text -----**<br>


Figure 2 | **Overall Architecture of RecursiveMAS.** Each agent first leverages the inner RecursiveLink to perform latent thoughts generation, and then transfers the generated information to the next agent through the outer RecursiveLink. After the last agent finishes generation, its latent thoughts are fed back to the first agent, thereby forming a recursive loop within the multi-agent system. 

adopted collaboration patterns in multi-agent systems: (i) _Sequential Style_ , where we follow the chain-of-agents setting to assign three agents with complementary roles of `Planner` , `Critic` , and `Solver` and progressively decompose, judge, refine, and solve the problem; (ii) _Mixture Style_ , where a mixture of domain-specialized agents ( `Math` , `Code` , `Science` ) reasons over the input problem in parallel, and their outputs are aggregated by a `Summarizer` agent to form the final answer; (iii) _Distillation Style_ , where a larger, more capable `Expert` agent is paired with a smaller, faster `Learner` agent to distill expert knowledge while retaining higher generation efficiency; and (iv) _Deliberation Style_ , where an inner-thinking `Reflector` is paired with a `Tool-Caller` that can invoke external tools (e.g., Python or search APIs). The agents iteratively exchange, critique, and refine candidate solutions until reaching a shared consensus, after which the `Tool-Caller` produces the final answer. 

## **3. Building a Recursive Multi-Agent System** 

We introduce RecursiveMAS, an end-to-end recursive framework that links heterogeneous LLM agents together to scale the entire system through efficient and seamless latent collaboration. In the following, we will first elaborate the detailed architectural design of RecursiveMAS, and then present the corresponding recursive learning algorithm. We also interleave theoretical analyses throughout the method pipeline to support underlying design principles. 

## **3.1. A Lightweight RecursiveLink** 

A language model’s last-layer hidden states provide a natural representation of its generated semantics. The RecursiveLink R is designed to preserve and transmit this information from one embedding space to another. In RecursiveMAS, the transition arises in two cases: (i) _Denseto-Shallow Transition_ , where the previous step’s last-layer embeddings are fed back as the next-step input embeddings during latent thoughts generation; and (ii) _CrossModel Transition_ , where one model’s newly generated latent representations are passed as conditioning inputs to another model. As illustrated in Figure 3, we bridge these two transitions through the inner and outer links. 

**==> picture [189 x 135] intentionally omitted <==**

**----- Start of picture text -----**<br>
RecursiveLink<br>Last-layer Emb Agent Ai Emb<br>Linear Linear<br>GELU GELU<br>Linear Linear<br>Residual Residual<br>Connection Connection<br>+ +<br>Input-layer Emb (Same Agent) Inner Agent Aj Emb Outer<br>Linear<br>**----- End of picture text -----**<br>


Figure 3 | Illustration on the inner and outer RecursiveLink Design. 

4 

Recursive Multi-Agent Systems 

**Inner Link.** Each LLM agent _𝐴𝑖_ ∈A is paired with an inner RecursiveLink Rin during auto-regressive generation. Given any new last-layer embedding vector _ℎ_ , Rin transforms it as: 

**==> picture [292 x 11] intentionally omitted <==**

where _𝑊_ 1 and _𝑊_ 2 are two standard linear layers, _𝜎_ (·) is the GELU activation, and the residual connection preserves the original latent semantics. The transformed embedding is then used as input to the next forward pass of agent _𝐴𝑖_ . 

**Outer Link.** An outer RecursiveLink Rout connects heterogeneous agents with different hidden dimensions. To support this, an additional linear layer _𝑊_ 3 is introduced in the residual branch to map the source embedding from agent _𝐴𝑖_ into the target embedding space of agent _𝐴 𝑗_ , i.e., 

**==> picture [301 x 11] intentionally omitted <==**

## **Why Residual Connection?** 

The residual branch largely preserves the original semantics of the input, allowing the RecursiveLink network to focus on _aligning distributional differences_ rather than learning the full projection from scratch. This leads to more stable and efficient training. We also explore other alternatives and empirically validate our proposed design in Section 5. 

## **3.2. Chain All Agents Together as a Loop** 

In recursive language models (RLMs), Transformer layers are connected through hidden states, and the residual stream loops across these layers to increase reasoning depth. Under this view, we cast each agent in RecursiveMAS as an RLM layer, with information flowing and recurring within and across agents as the hidden stream of the system. As shown in Figure 2, each agent contributes by reasoning and interacting with others in the latent space, together forming a recursive loop. 

**Latent Thoughts Generation inside Agents.** We start by describing how each agent unfolds reasoning through the auto-regressive generation of latent thoughts. Specifically, given input contexts’ embeddings _𝐸 𝐴_ 1 = [ _𝑒_ 1 _, 𝑒_ 2 _, . . . , 𝑒𝑡_ ] for the question and the agent-specific instructions, the first agent _𝐴_ 1 passes _𝐸 𝐴_ 1 through the Transformer and computes the last-layer hidden representation _ℎ𝑡_ at step _𝑡_ . Then, we insert _ℎ𝑡_ into the inner link Rin to map the distribution back into the input embedding space for the next step, yielding _𝑒𝑡_ +1 = Rin( _ℎ𝑡_ ). Agent _𝐴_ 1 repeats this process auto-regressively for _𝑚_ forward steps, generating a new continuous sequence of latent thoughts _𝐻𝐴_ 1 = [ _ℎ𝑡, ℎ𝑡_ +1 _, . . . , ℎ𝑡_ + _𝑚_ ]. 

**Interaction across Heterogeneous Agents.** Once agent _𝐴_ 1 completes latent reasoning, its latent thoughts _𝐻𝐴_ 1 are sent to the next agent _𝐴_ 2 for cross-agent interaction. To achieve seamless information transmission across different types of agents, we first pass _𝐻𝐴_ 1 through the outer link Rout to transform it into input embeddings aligned with agent _𝐴_ 2. Next, agent _𝐴_ 2 starts latent thoughts generation conditioned on both its own input contexts and transferred information from _𝐴_ 1 (i.e., _𝐸 𝐴_ 2 ⊕Rout( _𝐻𝐴_ 1)). 

We continue this interaction process across all consecutive agents in RecursiveMAS. In particular, after the last agent _𝐴𝑁_ completes latent thoughts generation, its latent outputs (representing the system’s latent answer to the input question) are passed back to the first agent _𝐴_ 1 through the inner-outer RecursiveLink, thereby closing the recursive loop. This recurrent connection allows each new recursion round to condition on information produced in previous rounds, so that each agent can iteratively reflect on earlier system outputs and refine their current generation. Throughout intermediate recursion rounds, all agents collaborate entirely in the latent space. Only after the final recursion round, the agent _𝐴𝑁_ decodes the textual output as the system’s final answer to the question. 

5 

Recursive Multi-Agent Systems 

**==> picture [471 x 362] intentionally omitted <==**

**----- Start of picture text -----**<br>
Preliminary Inner-Loop Training (Each Agent)<br>Parallelly Inner-train A1, A2, …, AN<br>Agent A1 Inner-Training<br>Connect Train & Optimize After Inner-Training<br>Regression Support Latent Reasoning<br>Agent A1 Loss Agent A1 Refine Latent Thoughts<br>Inner Link of A1 Predicted Latent Thoughts Ground-Truth Distribution Ready to Start Recursion<br>x N<br>Recursive Outer-Loop Training (Entire MAS)<br>Forward Pass<br>Recursion Round 1 Outer-Training Backward Propagation<br>Connect<br>ContextsContextsInput Input  Agent A1 Agent A2 … Agent AN ! !<br>Outer Link of A1,A2 Outer Link of A2,A3 Outer Link of AN,A1<br>Latent Outputs<br>Recursion Round 2 Outer-Training<br>ContextsInput  + Agent A1 Agent A2 … Agent AN ! "<br>Outer Link of A1,A2 Outer Link of A2,A3 Outer Link of AN,A1<br>…<br>Latent Outputs<br>Recursion Round n Outer-Training<br>CE Loss<br>ContextsInput  + Agent A1 Agent A2 … Agent AN<br>Outer Link of A1,A2 Outer Link of A2,A3 LM Head of AN ! #<br>Final Text Outputs<br>**----- End of picture text -----**<br>


Figure 4 | **Two-Stage Training Pipeline of RecursiveMAS.** We first perform inner-loop training for each agent in parallel to warm up the inner RecursiveLink for latent thoughts generation, and then conduct outer-loop training to recursively optimize the outer RecursiveLink over the entire system. 

**End-to-End Complexity Analyses.** To characterize the architectural efficiency of the full RecursiveMAS pipeline, we next analyze its end-to-end runtime complexity with RecursiveLink integrated throughout the system. The following proposition compares RecursiveMAS with a text-based recursive MAS, in which agents follow the same multi-round recursive collaboration structure but communicate through an explicit text medium rather than RecursiveLink-enabled latent interaction. 

**Proposition 3.1** ( **RecursiveMAS Runtime Complexity** ) **.** _Without RecursiveLink, a text-based Recursive MAS with the same collaboration structure requires runtime complexity of_ Θ( _𝑁_ ( _𝑚_ | _𝑉_ | _𝑑ℎ_ + ( _𝑡_ + _𝑚_ ) _𝑑ℎ_[2][+ (] _[𝑡]_[+] _[ 𝑚]_[)][2] _[𝑑][ℎ]_[))] _[; In contrast, with RecursiveLink-enabled collaboration,][ RecursiveMAS][ achieves] an end-to-end runtime complexity of_ Θ( _𝑁_ ( _𝑚𝑑ℎ_[2][+ (] _[𝑡]_[+] _[ 𝑚]_[)] _[𝑑] ℎ_[2][+ (] _[𝑡]_[+] _[ 𝑚]_[)][2] _[𝑑][ℎ]_[))] _[.]_ 

_Remark_ 3.2 _._ Since _𝑑ℎ_ ≪| _𝑉_ | in practice, RecursiveMAS replaces the expensive per-step vocabularyspace decoding cost _𝑚_ | _𝑉_ | _𝑑ℎ_ with a much more efficient latent-space transformation _𝑚𝑑ℎ_[2][.] 

Proposition 3.1 shows the end-to-end runtime advantage of RecursiveMAS. The full proof is provided in Appendix A.1. We also empirically analyze the efficiency advantage of our method in Section 5. 

6 

Recursive Multi-Agent Systems 

## **4. Learning to Recur as a Whole** 

With the framework in place, we next present the recursive learning algorithm, which only needs to train on the RecursiveLink to enable co-optimization of the entire system loop. As illustrated in Figure 4, the learning procedure consists of two stages: (i) a preliminary _inner-loop_ to equip each agent with stronger latent thoughts generation capabilities; and (ii) an iterative _outer-loop_ to progressively optimize the system as one unified entity over recursion rounds. 

**Model-Level Inner-Loop Training.** For practical deployment of RecursiveMAS, we directly adopt off-the-shelf text-generation models as agents. To adapt these agents to the latent thoughts generation pattern, we first warm-start them through the inner RecursiveLink Rin. Specifically, given each agent _𝐴𝑖_ ∈A with parameters _𝜃𝑖_ and the training example ( _𝑥, 𝑦_ ) ∈Dtrain, we construct the target latent thoughts distribution by passing the ground-truth text _𝑦_ through the standard input embedding layer Emb _𝜃𝑖_ of agent _𝐴𝑖_ . The objective of training the inner link Rin corresponding to _𝐴𝑖_ then formulates as: 

**==> picture [314 x 13] intentionally omitted <==**

where _𝐻_ denotes the last-layer latent thoughts generated by agent _𝐴𝑖_ , and cos(· _,_ ·) denotes the standard cosine similarity. The regression objective here encourages each agent to leverage its inner link Rin to align latent thoughts with the semantic distribution from the input embedding layer, while eliminating the process of explicit decoding and re-encoding. 

**System-Level Outer-Loop Training.** Next, we iteratively co-optimize the entire system through the outer RecursiveLink Rout. Let S[(] _[𝑟]_[)] denote the system state at recursion round _𝑟_ = 1 _, . . . , 𝑛_ . During outer-loop training, the system is first unrolled along its looped structure for _𝑛_ forward rounds. After the final textual prediction is produced in the last recursion round, we jointly optimize all outer links that connect the system with the following cross-entropy (CE) objective: 

**==> picture [331 x 20] intentionally omitted <==**

Throughout training, the computation graph is preserved along the full recursive paths. Gradient backpropagation assigns each outer link a shared credit signal according to its global contribution to the final prediction, thereby enabling information flow to be iteratively optimized as a whole. 

**Learning Advantage of RecursiveMAS.** To better understand why latent collaboration of agents in the inner-outer loop training confers a stronger learning advantage, we provide a detailed theoretical analysis below of the gradient propagation process throughout recursive training of RecursiveMAS. 

**Theorem 4.1** ( **Gradient Stability** ) **.** _Under the Realistic Assumptions (stated in Appendix A.2), if tokens are confident with entropy_ ≤ _𝜖, where typically 𝜖_ ≪ 1 _: directly applying text-based SFT (denoted by_ R _text_ ( _ℎ_ ) _) during recursion suffers from gradient vanishing (i.e., gradient norm close to 0); while RecursiveMAS with the RecursiveLink_ R _maintains stable and near constant gradients (i.e., gradient norm close to 1) during looped backpropagation process. Formally, with probability_ ≥ 1 − _𝛿,_ 

**==> picture [375 x 39] intentionally omitted <==**

The full proof is provided in Appendix A.3. Theorem 4.1 demonstrates the learning advantage of RecursiveMAS, by allowing gradients to remain informative across recursion rounds. Together, theoretical justifications in Proposition 3.1 and Theorem 4.1 motivate our design of latent-based interaction among agents rather than text mediation, as it makes the whole-system co-optimization of RecursiveMAS easier and more effective. During inference, RecursiveMAS performs recursive generation by following the same _𝑛_ recursion rounds as in the outer-loop training. 

7 

Recursive Multi-Agent Systems 

Table 1 | **Agent configurations for different collaboration patterns in RecursiveMAS.** We select off-the-shelf models from diverse model families to form heterogeneous agent compositions with complementary strengths. Each assignment is chosen to match the role-specific needs of the corresponding collaboration pattern while preserving both practical efficiency and scalability. 

|**Collaboration Pattern**|**Role**|**Model Size & Version**|
|---|---|---|
||||
|Sequential Style (Light)|Planner<br>Critic<br>Solver|Qwen3-1.7B (Yang et al.,2025)<br>Llama3.2-1B-Instruct (Grattafori et al.,2024)<br>Qwen2.5-Math-1.5B-Instruct (Qwen et al.,2025)|
||||
|Sequential Style (Scaled)|Planner<br>Critic<br>Solver|Gemma3-4B-it (Team et al.,2025)<br>Llama3.2-3B-Instruct (Grattafori et al.,2024)<br>Qwen3.5-4B (Yang et al.,2025)|
||||
|Mixture Style|Code Specialist<br>Science Specialist<br>Math Specialist<br>Summarizer|Qwen2.5-Coder-3B-Instruct (Hui et al.,2024)<br>BioMistral-7B (Labrak et al.,2024)<br>DeepSeek-R1-Distill-Qwen-1.5B (Qwen et al.,2025)<br>Qwen3.5-2B (Yang et al.,2025)|
||||
|Distillation Style|Learner<br>Expert|Qwen3.5-4B (Yang et al.,2025)<br>Qwen3.5-9B (Yang et al.,2025)|
||||
|Deliberation Style|Refector<br>Tool-Caller|Qwen3.5-4B (Yang et al.,2025)<br>Qwen3.5-4B (with Tool-Integration) (Yang et al.,2025)|



## **5. Empirical Evaluations** 

**Tasks and Datasets.** We conduct comprehensive evaluations of RecursiveMAS on nine benchmarks across various domains: (i) _Mathematical Reasoning_ , including MATH500 (HuggingFaceH4, 2023), AIME2025 (math ai, 2025), and AIME2026 (MathArena, 2026); (ii) _Scientific and Medical Tasks_ , including GPQA-Diamond (Rein et al., 2023) and MedQA (Yang et al., 2024a); (iii) _Code Generation_ , including LiveCodeBench-v6 (Jain et al., 2025) and MBPP Plus (Liu et al., 2023); and (iv) _Search QA_ , including HotpotQA (Yang et al., 2018) and Bamboogle (Press et al., 2023). We adopt the standard evaluation metric for each dataset. For AIME2025/2026, we report Pass@10 accuracy for testing robustness. Additional benchmark and metrics details are in Appendix B.1. 

**Models and Baselines.** We instantiate RecursiveMAS with diverse agent collaboration patterns, including (i) _Sequential Style_ , (ii) _Mixture Style_ , (iii) _Distillation Style_ , and (iv) _Deliberation Style_ , following the setups described in Section 2. For each collaboration style, we use off-the-shelf LLMs from diverse model families, covering Qwen (Qwen et al., 2025; Yang et al., 2025), Llama (Grattafiori et al., 2024), Gemma (Team et al., 2025), and Mistral (Jiang et al., 2024), to construct heterogeneous agent compositions. Detailed model configurations and their assigned roles are provided in Table 1. 

For baseline comparisons, we evaluate RecursiveMAS against (i) _Single Advanced Agents_ , where individual LLM agents from each collaboration pattern are isolated as standalone models to solve problems, such as the final agent in Sequential Style and each domain specialist in Mixture Style. For fair comparison, we provide full supervised and LoRA fine-tuning (Schulman and Lab, 2025) for single models on the same training set. (ii) _Recursion-based Methods_ , including single recursive language models, LoopLM (Zhu et al., 2025), and Recursive-TextMAS, where agents collaborate in the same way as RecursiveMAS but interact through text instead of latent thoughts; and (iii) additional _Representative Multi-Agent Frameworks_ , including TextGrad (Yuksekgonul et al., 2025) and 

8 

Recursive Multi-Agent Systems 

Table 2 | **Main results of RecursiveMAS over Different Recursion Rounds.** We report the accuracy (%, “Acc.”), end-to-end runtime (s, “Time”), and overall token usage (“Token”) across domains. For Code Gen., we evaluate the Light and Scaled settings on MBPP+ and LiveCodeBench, respectively. The average standard deviation of RecursiveMAS across 5 runs is ±0 _._ 0041 for accuracy, ±26 for runtime, and ±33 for tokens. We compare with all methods under the same MAS framework structure and recursion budgets. The performance and efficiency advantages of RecursiveMAS become increasingly significant as the recursion round _𝑟_ increases, with improvements highlighted. 

|**Method**<br>**Metric**|**Method**<br>**Metric**|**Math500**<br>Light Scaled|**AIME2025**<br> Light Scaled|**AIME2026**<br> Light Scaled|**GPQA-D**<br>Light Scaled|**MedQA**<br> Light Scaled|**Code Gen.**<br>**Improve**<br>Light Scaled|**Code Gen.**<br>**Improve**<br>Light Scaled|
|---|---|---|---|---|---|---|---|---|
||||**_Recursive Round r=1_**||||||
|Recursive-TextMAS|Acc.|71.9<br>84.2|24.0<br>71.3<br>16.7<br>76.7||28.1<br>61.5|29.0<br>76.1|30.7<br>38.5|Base|
||Time|1368<br>2401|2380<br>8462<br>2216<br>9376||1056<br>2190|1555<br>1522|976<br>8867|Base|
||Token|1185<br>1471|2993<br>9397<br>2754<br>8854||2084<br>3693|2382<br>1427|1146<br>3154|Base|
||Acc.|75.8<br>86.3|30.7<br>80.0<br>17.3<br>82.7||30.3<br>63.1|30.3<br>78.2|35.1<br>40.1|↑**3.4**|
|**RecursiveMAS**|Time|825<br>1701|1829<br>7784<br>1788<br>8134||586<br>1965|1194<br>1348|449<br>7908|×**1.2**|
||Token|523<br>816|1622<br>6338<br>1576<br>7021||829<br>2675|1369<br>964|577<br>2198|↓**34.6%**|
||||**_Recursive Round r=2_**||||||
|Recursive-TextMAS|Acc.|72.5<br>84.4|23.3<br>70.7<br>10.0<br>77.3||28.7<br>59.1|28.3<br>76.1|30.0<br>38.0|Base|
||Time|2204<br>3958|4247 14380 3960 14110||1825<br>4207|3097<br>2745|1847 14792|Base|
||Token|2117<br>2794|5318 16372 4982 16213||3708<br>6128|4436<br>2609|1998<br>5369|Base|
||Acc.|76.6<br>87.1|33.3<br>86.0<br>18.7<br>84.0||32.3<br>64.6|31.2<br>78.3|36.9<br>41.3|↑**6.0**|
|**RecursiveMAS**|Time|1096<br>1974|2367<br>8178<br>2263<br>8965||752<br>2342|1427<br>1664|627<br>8329|×**1.9**|
||Token|495<br>953|1614<br>5314<br>1552<br>6657||813<br>2521|1383<br>1008|531<br>2020|↓**65.5%**|
||||**_Recursive Round r=3_**||||||
|Recursive-TextMAS|Acc.|69.1<br>85.8|18.0<br>73.3<br>16.7<br>74.7||28.7<br>58.6|28.5<br>77.1|29.3<br>36.5|Base|
||Time|2952<br>6010|6183 19304 5907 19678||3322<br>7537|4684<br>3922|2310 22036|Base|
||Token|3059<br>4100|8645 23651 7813 22915||5820<br>8091|6307<br>3731|2676<br>7078|Base|
||Acc.|77.8<br>88.2|34.0<br>86.7<br>20.0<br>86.0||32.6<br>66.2|31.7<br>79.3|37.4<br>42.8|↑**7.2**|
|**RecursiveMAS**|Time|1360<br>2320|2727<br>8981<br>2629<br>9623||861<br>2638|1704<br>1912|805<br>10186|×**2.4**|
||Token|519<br>893|1586<br>5342<br>1537<br>6860||786<br>2524|1378<br>1056|595<br>2247|↓**75.6%**|



Mixture-of-Agents (MoA) (Wang et al., 2025b) for more holistic structure-wide evaluations. Detailed baseline implementations are provided in Appendix B.2. 

**Training and Implementation Details.** For inner-outer loop training, we freeze all LLM agent parameters and update only the inner/outer RecursiveLink. We curate a diverse training set spanning multiple domains, sourced from s1K (Muennighoff et al., 2025) for mathematical problem solving, m1k (Huang et al., 2025) for medical and scientific tasks, OpenCodeReasoning (Ahmad et al., 2025) for code generation, and ARPO-SFT (Dong et al., 2025) for agentic tool-augmentation (Python Code/Search-API) settings. We use AdamW with a learning rate of 5e-4, a cosine learning rate scheduler, and a batch size of 4. During inference, we set top-p to 0.95 and use a temperature of 0.6 for most reasoning tasks and 0 _._ 2 for code generation, as suggested in each model’s official report. The maximum output length is adjusted for each task based on its relative difficulty. We perform hyperparameter tuning and report the mean performance over five independent runs. More training/inference details and hyperparameter setups are provided in Appendix B.3. 

## **5.1. Scaling Performance via Recursion** 

We begin by evaluating how RecursiveMAS performs across different recursion depths _𝑟_ = 1 _,_ 2 _,_ 3. As shown in Table 2, we analyze agent collaboration behavior from three complementary perspectives: (i) accuracy, (ii) end-to-end runtime, and (iii) overall system token throughput. We also include a text-based recursive baseline for reference. Across seven math, science, and code generation tasks, both light and scaled versions of RecursiveMAS exhibit a consistent upward trend as recursion depth 

9 

Recursive Multi-Agent Systems 

Table 3 | **Comparison of RecursiveMAS with Other Methods.** We evaluate RecursiveMAS at recursion round _𝑟_ = 3. Under the same training budget and model setups, RecursiveMAS consistently outperforms advanced single-agent methods, alternative MAS frameworks, and recursive computation baselines. 

|**Method**|**MATH500 **|**AIME2025 **|**AIME2026 **|**GPQA-D **|**LiveCodeBench **|**MedQA**|
|---|---|---|---|---|---|---|
|Single Agent (w/ LoRA)|83.1|70.0|73.3|62.0|37.4|76.1|
|Single Agent (w/ Full-SFT)|83.2|73.3|76.7|62.8|38.6|77.0|
|Mixture-of-Agents (MoA)|79.8|60.0|63.3|47.6|27.0|57.5|
|TextGrad|84.9|73.3|76.7|62.5|39.8|77.2|
|LoopLM|84.6|66.7|63.3|48.1|24.9|56.4|
|Recursive-TextMAS|85.8|73.3|73.3|61.6|38.7|77.0|
|**RecursiveMAS**|**88.0**|**86.7**|**86.7**|**66.2**|**42.9**|**79.3**|



increases. When compared with the text-based recursion, RecursiveMAS consistently improves over the baseline by an average of 8.1% at _𝑟_ = 1, 19.6% at _𝑟_ = 2, and 20.2% at _𝑟_ = 3, with performance advantage more pronounced as the recursion deepens. Additionally, under identical MAS architectures, RecursiveMAS delivers steadily increasing efficiency gains across recursion rounds, accelerating end-toend inference time from 1 _._ 2× to 2 _._ 4× while reducing output tokens from 34 _._ 6% to 75 _._ 6%. Additional case studies on the running pipeline of RecursiveMAS across domains are provided in Appendix G. 

**Scaling Law on RecursiveMAS (Training v.s. Inference).** We further examine the scaling behavior of recursion in RecursiveMAS by jointly varying the training-time and inference-time recursion rounds. Figure 1 (Up) illustrates the performance landscape of RecursiveMAS under different training and inference settings. Increasing inference depth continues to improve systems trained with fewer rounds, while deeper training shifts the entire performance frontier upward, with the strongest results consistently appearing in the upper-right region where both are large. This trend suggests a complementary training-inference scaling effect in RecursiveMAS: training recursion progressively teaches the system to form refinement-ready latent states, and subsequent inference recursion translates this learned recursive structure into additional test-time gains. 

## **5.2. Broader Comparison with Alternative Architectures and Training Frameworks** 

Table 3 compares RecursiveMAS at the whole-system level against a broader set of baselines, including single fine-tuned agents, representative multi-agent frameworks, and alternative recursive methods. To ensure fair comparison, all methods are instantiated with identical backbone models and comparable training budgets (e.g., matched trainable parameter counts, recursion depth, training set). 

Overall, RecursiveMAS delivers a consistent whole-system advantage, achieving an average performance improvement of 8.3% over the strongest baseline on each benchmark. With the same training data, fine-tuning individual agents strengthens performance relative to their off-the-shelf versions, while RecursiveMAS delivers further gains by optimizing cross-agent collaboration at the system level. In addition, RecursiveMAS remains the performance advantage compared to advanced architectures such as TextGrad and LoopLM, especially on reasoning-intensive tasks (e.g., accuracy gains of 18.1% on AIME2025, 13.0% on AIME2026, and 5.4% on GPQA-Diamond). 

## **5.3. Can RecursiveMAS Generalize across Diverse Collaboration Patterns?** 

Beyond the sequential setting, we further instantiate RecursiveMAS under three additional MAS collaboration patterns in Table 1 to assess whether our method is agnostic to any specific system architecture and generalizes across diverse usage scenarios. As shown in Figure 1 (Down), we compare 

10 

Recursive Multi-Agent Systems 

**==> picture [471 x 98] intentionally omitted <==**

**----- Start of picture text -----**<br>
Avg. 1.2x  Avg. 1.9x  Avg. 2.4x<br>Inference Speedup Inference Speedup Inference Speedup<br>**----- End of picture text -----**<br>


Figure 5 | **Inference Time Speedup of RecursiveMAS across Three Recursion Rounds.** RecursiveMAS exhibits increasing inference speedup as the recursion depth increases. 

**==> picture [471 x 102] intentionally omitted <==**

**----- Start of picture text -----**<br>
Avg. 34.6%  Avg. 65.5%  Avg. 75.6%<br>Fewer Token Usage Fewer Token Usage Fewer Token Usage<br>**----- End of picture text -----**<br>


Figure 6 | **Token Reduction of RecursiveMAS across Three Recursion Rounds.** As recursion deepens, RecursiveMAS reduces substantially more tokens than Recursive-TextMAS. 

the accuracy of RecursiveMAS against strong standalone agents within each collaboration pattern. 

In _Mixture-style_ , RecursiveMAS achieves an average improvement of 6.2% over the strongest domain specialist on each benchmark, suggesting that recursive interaction enables non-trivial cross-domain composition beyond what can be attained by selecting one individual specialist alone. In _Deliberationstyle_ , we evaluate tool use on both mathematical and search-intensive tasks. RecursiveMAS improves the original tool-calling agent by 4.8%, showing that recursive latent coordination remains effective in tool-calling settings through iterative interaction with the Reflector. Finally, in _Distillation-style_ , RecursiveMAS improves the learner by 8.0% while retaining 1.5× end-to-end speed advantage over the expert. In this way, RecursiveMAS distills much of the expert’s capability into a more efficient system. We leave detailed reports of Figure 1 (Down) in Appendix D.1. 

## **5.4. Efficiency Analyses on Latent-space Recursion** 

**Inference Time Speedup.** We analyze the efficiency of RecursiveMAS to empirically support our complexity advantage in Proposition 3.1. We first compare RecursiveMAS against Recursive-TextMAS to study how our advantage on end-to-end inference time scales with recursion depth. As shown in Figure 5, although deeper recursion rounds introduce cost, we find that RecursiveMAS consistently exhibits efficiency gain, and the advantage further increases as recursion deepens. For example, at recursion round _𝑟_ = 1, RecursiveMAS already achieves a 1 _._ 2× speedup on average, and this advantage grows to 1 _._ 9× and 2 _._ 4× at larger recursion rounds of _𝑟_ = 2/3. This trend aligns well with our method design, where RecursiveMAS achieves a favorable scaling behavior by conducting recursive collaboration directly in latent space and avoiding repeated intermediate text generation. 

**Overall Token Usage Reduction.** We next demonstrate the substantial token usage reduction of RecursiveMAS in Figure 6. Within the comparison, we find that the baseline method suffers from rapidly growing token overhead as recursion round increases, while RecursiveMAS reduces the token usage by 34 _._ 6% for the first recursion round, and the reduction scales to 75 _._ 6% at _𝑟_ = 3. This is because Recursive-TextMAS repeatedly decode the intermediate text at every recursion round, whereas 

11 

Recursive Multi-Agent Systems 

RecursiveMAS performs most recursive interaction directly in latent space. Overall, RecursiveMAS enables a much more efficient system-level scaling behavior, and the resulting efficiency gain is amplified as the number of recursion rounds increases. 

## **6. In-depth Analyses on RecursiveMAS** 

**RecursiveLink Design.** To validate the effectiveness of RecursiveLink, we compare our 2-layer residual design against three alternatives: (i) a 1-layer network, (ii) a 1-layer network with the residual connection, and (iii) a 2-layer network without the residual connection. We conduct experiments using the scaled sequential-style RecursiveMAS and adapt the same architecture for both Rin and Rout. 

As shown in Table 4, our 2-layer residual Table 4 | **Efficacy on RecursiveLink Design.** We comdesign performs best across all three benchpare accuracy across alternative architectural designs. marks, and the residual connection delivers additional improvements across differ- **RecursiveLink Design** Math500 GPQA-D LiveCodeBench ent backbone models. For example, on 1-Layer 84.4 63.2 40.1 GPQA-Diamond, equipping a single-layer Res+1-Layer 86.7 65.3 41.4 design with a residual branch improves the performance from 63.2% to 65.3%, which 2-Layer 85.6 64.5 40.5 is even higher than the plain 2-layer de- **Res+2-Layer (ours) 88.0 66.2 42.9** sign (64.5%). These results align with our design intuition in Section 3.1: by preserving latent semantics while learning only the distributional shift, RecursiveLink achieves stable training and stronger inference performance. 

Table 4 | **Efficacy on RecursiveLink Design.** We compare accuracy across alternative architectural designs. 

**Semantic Representations in Recursion.** We analyze how the semantic distribution of RecursiveMAS changes across different recursion rounds. Under the scaled sequential setting of RecursiveMAS, we randomly sample 500 question-answer pairs spanning all downstream domains. We then use the solver agent’s input embedding layer to map each ground-truth answer string into embedding representations, which serves as the reference semantic distribution. We run RecursiveMAS at recursion rounds _𝑟_ = 1 _,_ 2 _,_ 3 to generate final answers for all these 500 questions, map the generated answers into embeddings using the same input embedding layer, and visualize both the ground-truth reference ("purple") and newly generated distributions ("orange") via PCA projection. 

**==> picture [471 x 117] intentionally omitted <==**

Figure 7 | **Semantic Representations of RecursiveMAS across Differnt Recursion Rounds.** We visualize the semantic distribution of the final answers generated by RecursiveMAS and the corresponding ground-truth across 500 questions. Increasing recursion rounds progressively aligns the generated distribution of RecursiveMAS with the ground truth distribution. 

In Figure 7, the generated answers at _𝑟_ = 1 remain visibly shifted from the ground-truth distribution, but this discrepancy progressively narrows as depth increases, with the two distributions becoming largely aligned by _𝑟_ = 3. This aligning trend suggests that RecursiveMAS iteratively refines the latent embeddings and corresponding answers through recursion. We further take a closer look to examine 

12 

Recursive Multi-Agent Systems 

Table 5 | **Cost analysis on RecursiveMAS.** We report the peak GPU memory usage (GB), number of trainable parameters, estimated cost, and average accuracy (%) across all downstream tasks. 

|**Methods**|**GPU Mem. **|**Trainable Param.**|**Cost**|**Avg. Acc.**|
|---|---|---|---|---|
|LoRA Training|21.67|15.92M (0.37%)|$6.64|66.9|
|Full-SFT|41.40|4.21B (100%)|$9.67|68.6|
|**RecursiveMAS**|**15.29**|**13.12M (0.31%) **|**$4.27**|**74.9**|



individual test instances and provide detailed case studies in Appendix F. Our case studies reveal a common pattern in which RecursiveMAS may produce an incorrect answer at an early stage, while deeper recursion successfully corrects it through iterative refinement. Together, these analyses provide further evidence that latent thoughts capture semantically meaningful representations, and that deeper recursion improves alignment toward correct final outputs. 

**Optimal Length of Latent Thoughts Generation.** We next study and ablate the latent thoughts length _𝑚_ to examine how much of each agent’s internal reasoning is sufficient to support effective collaboration. Under the scaled sequential-style of RecursiveMAS, we evaluate a broad range of _𝑚_ . As illustrated in Figure 8, increasing _𝑚_ improves performance in the early regime. Once _𝑚_ reaches a moderate scale (around _𝑚_ = 80), performance is stabilized across all benchmarks. The ablation suggests that RecursiveMAS enables effective agent reasoning and interaction with only a modest latent-thought budget, in sharp contrast to text-based collaboration that typically requires longer CoT and costly token generation. 

**==> picture [203 x 133] intentionally omitted <==**

Figure 8 | Effectiveness of RecursiveMAS’s latent thoughts with different step lengths. 

**Training Cost Analysis** We further analyze the training cost of RecursiveMAS under the scaled sequential-style MAS setting. We compare RecursiveMAS with direct training methods, including LoRA and full supervised fine-tuning with the same training data and backbone setup. For cost estimation, we follow prior methods (Liu et al., 2025; Lu et al., 2023) to measure the cost based on GPU usage. As shown in Table 5, RecursiveMAS utilizes the lowest per-agent GPU memory, trainable parameter count, and estimated cost among all compared training strategies. Meanwhile, RecursiveMAS achieves the highest accuracy across all downstream tasks, suggesting that optimizing the lightweight RecursiveLink provides a better cost-performance trade-off than other training methods. 

## **7. Related Works** 

**LLM-based Multi-Agent Systems.** Current LLMs achieve strong performance on general tasks, but they often exhibit bottlenecks when facing diverse reasoning patterns (Maheswaran et al., 2026; Mirzadeh et al., 2025; Valmeekam et al., 2023) or domain-specific challenges (Chen et al., 2025). To overcome these limitations, Multi-agent systems extend the single LLM paradigm to a collaborative setting (Su et al., 2025; Tran et al., 2025; Wu et al., 2024; Yang et al., 2024b) by organizing a set of agents with distinct roles that jointly address the problem. A standard multi-agent system topology involves a sequential configuration (Li et al., 2023; Qian et al., 2024), where agents are assigned in a linear pipeline to decompose and resolve problems in order. Beyond sequential settings, other 

13 

Recursive Multi-Agent Systems 

works also explore mixture-style settings (Wang et al., 2025b; Ye et al., 2025b; Yun et al., 2026), where multiple agents with domain expertise reason in parallel, and their outputs are aggregated into a final decision. Another line of work seeks to improve MAS through textual feedback signals. For example, related optimization methods (Shen et al., 2025; Yuksekgonul et al., 2025) leverage an LLM to generate natural language feedback to refine contextual inputs and instructions of each agent. Additionally, another study (Motwani et al., 2024) improves MAS by separately training each agent with role-specific responses. Rather than separate training each individual agent or only leveraging textual feedback, RecursiveMAS treats MAS as a unified whole, and scales the system performance via recursively refining the latent information flow. 

**Scaling Reasoning via Recursion.** Recent studies explore recursion as an alternative scaling axis for LLMs (Bae et al., 2025; Geiping et al., 2025; Li et al., 2026; Tang et al., 2026), where the same computation blocks are reused through multiple recurrent rounds (i.e., loops) to increase reasoning depth and iteratively refine hidden representations. One line of work studies recursive language models that apply shared layers to scale latent reasoning. For instance, LoopLM (Zhu et al., 2025) introduces pre-trained looped language models with iterative latent computation. Besides, other work explores other recursive architectures (Jolicoeur-Martineau, 2025; Wang et al., 2025a; Zhang et al., 2025a), including tiny recursive networks and recursive self-calling schemes for long-context inference. While existing methods in agentic AI primarily focus on recursion inside a single language model, RecursiveMAS exhibits the first attempt to extend the recursive scaling paradigm to system-level. Additional related works are provided in Appendix C. 

## **8. Conclusion** 

We introduce RecursiveMAS, a recursive multi-agent framework that scales agent collaboration through system-level recursion. RecursiveMAS first supports latent-thoughts generation within each agent through inner RecursiveLink, then connects heterogeneous agents through outer RecursiveLink, and optimizes the whole system with an inner-outer loop training paradigm. Theoretically, our framework leads to more stable training dynamics and improves efficiency compared to text-based baselines. Our empirical results across mathematical and scientific reasoning, code generation, and search benchmarks show that RecursiveMAS consistently improves accuracy while substantially reducing inference time and token usage. Overall, RecursiveMAS provides a scalable and efficient framework for multi-agent systems to recursively collaborate, refine, and evolve in latent space.