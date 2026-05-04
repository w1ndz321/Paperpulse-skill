**==> picture [152 x 18] intentionally omitted <==**

**==> picture [157 x 23] intentionally omitted <==**

# **- Agent World: Scaling Real-World Environment Synthesis for Evolving General Agent Intelligence** 

# **Renmin University of China** , **ByteDance Seed** 

See Contributions section for a full author list. 

## **Abstract** 

Large language models are increasingly expected to serve as general-purpose agents that interact with external, stateful tool environments. The Model Context Protocol (MCP) and broader agent skills offer a unified interface for connecting agents with scalable real-world services, but training robust agents remains limited by the lack of realistic environments and principled mechanisms for life-long learning. In this paper, we present **Agent-World** , a self-evolving training arena for advancing general agent intelligence through scalable environments. Agent-World has two main components: **(1) Agentic Environment-Task Discovery** , which autonomously explores topic-aligned databases and executable tool ecosystems from thousands of real-world environment themes and synthesizes verifiable tasks with controllable difficulty; and **(2) Continuous Self-Evolving Agent Training** , which combines multi-environment reinforcement learning with a self-evolving agent arena that automatically identifies capability gaps through dynamic task synthesis and drives targeted learning, enabling the co-evolution of agent policies and environments. Across 23 challenging agent benchmarks, Agent-World-8B and 14B consistently outperforms strong proprietary models and environment scaling baselines. Further analyses reveal scaling trends in relation to environment diversity and self-evolution rounds, offering insights for building general agent intelligence. 

**Date:** April 21, 2026 

**Correspondence:** Guanting Dong at dongguanting@ruc.edu.cn, Zhicheng Dou at dou@ruc.edu.cn **Project Page:** https://agent-tars-world.github.io/-/ 

## **1 Introduction** 

In recent years, large language models (LLMs) have delivered remarkable progress across a wide range of language understanding and decision-making tasks [18, 68, 82, 95, 114]. As their capability frontier continues to expand, expectations for LLMs are shifting from chat-oriented text generation toward general-purpose agent assistants [7, 16, 34, 47, 64, 65]. Ideally, such agents should seamlessly integrate real-world interaction with verbal reasoning, and continuously learn from experience to improve themselves, much like human intelligence [29, 72, 86, 132]. Realizing these agentic capabilities requires not only training LLMs in dynamic environments, but also equipping them with executable tools. On this basis, agents can take actions and observe timely feedback from the environment, forming a _“Generation–Execution–Feedback”_ interaction loop [61, 73, 110, 116, 117]. 

With the rise of agentic reinforcement learning (Agent RL), several agent systems built on static tool environments have demonstrated strong practical value, especially in deep information-seeking and software 

1 

**==> picture [472 x 208] intentionally omitted <==**

**Figure 1 Overview of Agent-World (left) and downstream general agent performance (right).** The environmentscaling analysis reports the average score across representative subdomains of MCP-Mark, BFCL V4, and _𝜏_[2] -Bench. 

engineering [21, 22, 42, 49, 92, 103, 115, 126]. However, open-world tool environments are inherently compositional and stateful. For instance, in a flight-booking workflow, an agent should follow a valid action order (check inventory → execute booking → update the calendar), while each action also modifies the underlying environment state. Consequently, agents must orchestrate multi-tool usage flows while tracking state transitions induced by their agent-environment interactions. Prior work centered on stateless or singletool settings is therefore insufficient for realistic applications [26, 42, 51]. This limitation has motivated growing interest in building general agents around standards such as the Model Context Protocol (MCP) [6, 44, 66] and broader agent skills [40, 56, 119], as well as harness engineering [59, 60, 71, 133]. In this setting, an ideal agent serves as a unified orchestrator that can invoke scalable real-world tools, track state changes in real time, and seamlessly integrate large-scale agentic services into automated workflows [37, 44, 62]. 

Importantly, a key requirement for such general-purpose agents is access to diverse and realistic interactive environments [3, 38]. However, manually crafting such environments is expensive and difficult to scale, which has driven research toward two main directions: **(i) Simulated environments** use LLMs as implicit textual world models to produce environment feedback for agent training [27, 32, 52, 55, 102, 109]. While highly scalable, such simulators are vulnerable to hallucinations and often deviate from real-world dynamics. In contrast, **(ii) Realistic environments** combine executable tools with real databases, providing stronger grounding for complex interactions [5, 9, 58, 75, 90, 94, 98, 105, 107, 111, 121, 124, 127, 129]. Benchmarks such as _𝜏_[2] -Bench and ClawEval have moved evaluation closer to frontier agent applications through stateful environments [12, 16, 73, 117]. More recently, several studies have taken initial steps toward synthesizing programmatic environments and tasks for agent training [11, 25, 88, 98, 108]. Unfortunately, their reliance on single-round training makes it difficult for agents to acquire robust, transferable interaction logic in broad environment spaces. 

Consequently, although these approaches improve the efficiency of environment construction, two key bottlenecks remain unresolved: 

- **Scalable realism and complex environment synthesis:** Existing environments are often purely LLMgenerated or derived from limited open-source toolchains, which often mismatch real-world interaction logic. Moreover, synthetic environments are often limited in complexity, restricting the training of agents on long-horizon, state-intensive tasks. 

- **Continuous self-evolving training mechanisms:** Although realistic environments can naturally serve as effective training arenas, existing work has primarily emphasized environment construction and scaling, 

2 

while lacking principled mechanisms that use such scalable environments to diagnose agent weaknesses and drive continual self-improvement. 

In this paper, we propose **Agent-World** , a general-purpose agent training arena that unifies scalable real-world environment synthesis with continuous self-evolving training. As shown in Figure 1, Agent-World follows a two-stage design that forms a closed-loop training process. 

**(1) Agentic Environment-Task Discovery.** We collect thousands of real-world environment themes and build a deep-research pipeline that autonomously mines topic-aligned databases and executable toolsets from the web, forming a scalable and realistic environment ecosystem ( **including 1978 environments and 19822 tools** ). On top of these environments, we synthesize high-quality agent tasks through both graph-based and programmatic generation, and further expand task difficulty with executable verification. 

**(2) Continuous Self-Evolving Agent Training.** We train agents via multi-environment reinforcement learning over “agent–tool–database” interaction rollouts, using executable rewards for state-aware supervision. Notably, our environment ecosystem naturally serves as a self-evolving arena for evolving agents. Built on these scalable environments, the arena can iteratively synthesize new tasks, automatically identify capability gaps in trained agents, and drive targeted learning, thereby forming a co-evolution loop between agent policies and environments. 

We conduct comprehensive evaluations on **23 benchmarks** covering _agentic tool-use_ , _advanced AI assistant_ , _software engineering_ , _deep research_ , and _general reasoning_ . As shown in Figure 1, Agent-World-8B and 14B consistently outperforms strong foundation models and competitive baselines. Our analysis further reveals clear scaling relationships among the number of synthesized environments, self-evolution rounds, and downstream agent performance, providing empirical insights into the development of more general agent intelligence. 

In summary, our main contributions are as follows: 

- We introduce **Agent-World** , a general-purpose agent training arena that unifies scalable real-world environment synthesis with a continuous self-evolving training mechanism, forming a co-evolution loop between agent policies and environments. 

- We propose **Agentic Environment-Task Discovery** , which mines realistic executable environments from real-world environment themes and synthesizes diverse verifiable tasks with controllable difficulty. 

- We propose **Continuous Self-Evolving Agent Training** , which integrates multi-environment agentic RL with a self-evolving arena to automatically diagnose agent weaknesses and drive targeted learning in a closed training loop. 

- Experiments across **23 challenging agent benchmarks** demonstrate the superior performance of Agent-World. Further analysis reveals scaling relationships among environment diversity, evolution rounds, and agent performance. 

## **2 Preliminary: Agentic Interaction with Multi-Environments** 

Following AgentSkiller [93], we model multi-turn agentic interaction with external environments as a Partially Observable Markov Decision Process (POMDP) [10], represented by the tuple ( _𝑈, 𝑆, 𝐴, 𝑂, 𝑃_ ). 

**Intent space (** _𝑈_ **).** Let _𝑞_ ∈ _𝑈_ denote the user’s latent intent. The assistant progressively infers _𝑞_ from the accumulated interaction history and environment feedback to choose appropriate actions. 

**State space (** _𝑆_ **).** We factor the global state into an environment state and a dialogue state: _𝑆_ = _𝑆𝐸_ × _𝑆𝐻_ . At turn _𝑡_ , the full state is _𝑠𝑡_ = ( _𝑠𝑡[𝐸][, 𝑠] 𝑡[𝐻]_[)][∈] _[𝑆]_[.][The][environment][state] _[𝑠][𝐸]_[∈] _[𝑆][𝐸]_[captures][the][external][world][the] assistant can query or modify (e.g., databases, files, services), while the dialogue state _𝑠[𝐻]_ ∈ _𝑆𝐻_ summarizes conversational context (e.g., dialogue history, constraints, user preferences). 

3 

**Databases and tools.** To connect the POMDP with multiple environments, we explicitly parameterize each environment by a pair _𝑒_ = (D _,_ F ), where D denotes an environment database and F denotes a toolset. Concretely, the database D is a primary carrier (storage) of the environment state _𝑠[𝐸]_ —it contains the structured records and/or files that constitute the mutable external world. The toolset F = { _𝑓𝑘_ } provides executable interfaces to interact with _𝑠[𝐸]_ : each tool _𝑓_ ∈F can be seen as a callable operator that reads and optionally writes the database, thereby inducing environment state transitions. 

**Action space (** _𝐴_ **).** The assistant chooses between tool-use actions and language-response actions: _𝐴_ = _𝐴_ tool ∪ _𝐴_ resp. For _𝑎𝑡_ ∈ _𝐴_ tool, the assistant invokes a tool with structured arguments (e.g., a function name with JSON parameters) to query/modify the environment; for _𝑎𝑡_ ∈ _𝐴_ resp, it emits a natural-language message (including intermediate responses or the final answer). 

**Observation space (** _𝑂_ **).** At each turn _𝑡_ , the assistant observes _𝑜𝑡_ ∈ _𝑂_ and then takes an action _𝑎𝑡_ . We define _𝑂_ = _𝑂 𝐸_ ∪ _𝑂 𝐻_ , where _𝑂 𝐸_ contains structured tool observations returned by tool execution (e.g., query results, logs, error codes), and _𝑂 𝐻_ contains dialogue-side observations (e.g., user utterances, system prompts, or an explicit termination signal in offline training). Importantly, the environment state _𝑠[𝐸]_ is not directly observed and must be inferred indirectly from tool observations in _𝑂 𝐸_ . 

**State dynamics (** _𝑃_ **).** The transition model _𝑃_ : _𝑆_ × _𝐴_ → Π( _𝑆_ × _𝑂_ ) specifies how the system evolves after an action. Given ( _𝑠𝑡 , 𝑎𝑡_ ), the process transitions to _𝑠𝑡_ +1 and emits the next observation _𝑜𝑡_ +1: 

- If _𝑎𝑡_ ∈ _𝐴_ tool, a tool _𝑓_ ∈F is executed against the database D. This execution may update the environment state _𝑠𝑡[𝐸]_ +1[via][reads/writes][on][D][and][produce][a][structured][observation] _[𝑜] 𝑡[𝐸]_ +1[∈] _[𝑂][𝐸]_[.][The][dialogue][state] _[𝑠] 𝑡[𝐻]_ +1 is updated by appending the new tool interaction. 

- If _𝑎𝑡_ ∈ _𝐴_ resp, the assistant updates the dialogue state (i.e., _𝑠𝑡[𝐻]_ +1[)][by][emitting][a][response.][In][interactive] settings this may lead to a new user observation _𝑜𝑡[𝐻]_ +1[∈] _[𝑂][𝐻]_[; in offline training it typically yields a termination] signal. The environment state remains unchanged for that turn: _𝑠𝑡[𝐸]_ +1[=] _[𝑠] 𝑡[𝐸]_[.] 

## **3 Methodology** 

We propose **Agent-World** , a general-purpose agent training arena that unifies scalable environment-task discovery with continuous self-evolving agent training. The method contains two tightly coupled components: 

(1) **Agentic Environment-Task Discovery.** We collect thousands of real-world environment themes and build a deep-research pipeline that autonomously mines topic-aligned databases and executable tool interfaces from the web, forming a scalable and realistic environment ecosystem. On top of these environments, we synthesize diverse verifiable tasks through both graph-based and programmatic generation, and further expand task difficulty with executable validation. 

(2) **Continuous Self-Evolving Agent Training.** We train agents via multi-environment reinforcement learning over “agent–tool–database” interaction rollouts, using executable rewards for state-aware supervision. The same environment ecosystem also serves as a dynamic diagnostic arena that refreshes evaluation tasks, identifies capability gaps, and drives targeted environment-task expansion, thereby enabling the co-evolution of agent policies and environments. 

These two components form a closed loop: scalable environments support agent training, while training-time diagnosis feeds back into the next round of environment-task construction. Below, we describe each component in detail. 

## **3.1 Agentic Environment-Task Discovery** 

**Environment Theme Collection:** Scalable environment synthesis begins with diverse and high-quality environment themes as anchors. We therefore systematically gather environment themes from three real-world sources: 

4 

**==> picture [472 x 158] intentionally omitted <==**

**----- Start of picture text -----**<br>
Environment Theme Collection Agentic Database Mining: Tool Generation & Verification  Environment Taxonomy  VerifiabaleTask Synthesis<br>DB-grounded Tools: Cluster Classify Graph-based Tasks: DAG graph + random walk<br>Web  DR Agent Output: Executable Python<br> MCP Servers: (~2.8K) functions, Schemas, Descriptions Level 1: 20 labels Programmatic Tasks:  solution code + verifier script<br> Tool Documentations (~0.5K) Cross-validation: Level 2: 50 labels<br> Industrial PRDs (~0.2K) Output: sandbox test , unit test Level 3: 1978 labels Difficulty Scaling:<br>long horizon + logical complexity<br>- Name: - Description:  Server primarily functions as a memory management system for AI assistants… - Name: - Description:  Server primarily functions as a memory management system for AI assistants… - Name: - Description:  Server primarily functions as a memory management system for AI assistants…MCP Server Memory Bank MCP Server Memory Bank MCP Server Memory Bank The MCP The MCP The MCP  MCP/ mcp-config.json.clinerules-codeprogress-log.json … File Structure … Memory Bank/ README.mdactive-context.mdprogress.md def        payload =         result = track_progressdef       result = track_progressread_product_contextreturnreturn result resultread_memory_bank_file"product-context.md"{ "action" "updateActiveContext" (action, description, updateActiveContext=True(): action, :( Tool Schema payload…"description"  }))({ "filename" : updateActiveContext : description,     : }): - Programmatic Synthesis- Graph-based SynthesisTask Synthesis  Our team uses a shared project workspace with standard … We decided to switch our audit trail to event-driven ingestion …  - Checklist- Verifier Script def """Rough check: dict root; contacts missing or a dict."""returnSingle JSON object outputExact key setAlternatives_countvalidate_environment_state(c := state.isinstance Task Verification get(("contacts"state, dict)))andis(state): None  ( Consequences_countReady_to_logor…isinstance(c, dict))<br>**----- End of picture text -----**<br>


**Figure 2 The Pipeline of Agentic Environment-Task Discovery.** We start from real-world environment themes, mine topic-aligned databases from the web, generate and verify executable tool interfaces, and synthesize verifiable tasks with controllable difficulty. 

**(1) MCP Servers:** We obtain real-world MCP server specifications from Smithery[1] . Each specification is accompanied by a structured JSON document that includes source-data descriptions and standardized tool definitions. We denote the corresponding topics as _𝑚_ ∈M1. 

**(2) Tool Documentations:** We broadly collect and filter open-source datasets covering real tool-use scenarios, extract tool-definition documents, and use an LLM to inversely map them to environment topics, denoted as _𝑚_ ∈M2. 

**(3) Industrial PRDs:** As product requirement documents for specific industries, PRDs naturally include background, domain workflows and system interfaces. We use them as theme anchors, denoted as _𝑚_ ∈M3. 

We finally merge these sources to form the seed topic set: M = M1 ∪M2 ∪M3 _._ 

**Agentic Database Mining:** Given the topic set M, our goal is to mine topic-aligned real-world environment databases. Unlike prior work that emphasizes LLM-synthesized databases [31, 88, 98], we argue that the World Wide Web already contains abundant, high-value structured data that can be updated in real time. 

Motivated by this, we design an agentic workflow to autonomously mine and process web data into environment databases. Concretely, we build a deep-research agent G centered on a policy model _𝜋 𝜃_ and an external toolset T including _search, browser, code compiler, and operating-system (OS) tools_ . For each topic _𝑚_ ∈M, the agent conducts iterative loops for in-depth information retrieval and data mining. After that process, the agent leverages OS tools for structuring and persistent storage, yielding the environment database as: 

**==> picture [133 x 10] intentionally omitted <==**

where G(·) denotes the topic-conditioned automated research pipeline. Empirically, a single autonomous mining flow often yields databases with limited scale and simple structure. To address this, we introduce a database complexification process _𝜙_ , which iteratively prompts a deep-research agent to expand and enrich topic-specific databases: 

D[(] _[𝑛]_[+][1][)] ( _𝑚_ ) = _𝜙_[�] D[(] _[𝑛]_[)] ( _𝑚_ ) _, 𝑚,_ T[�] _, 𝑛_ = 0 _, . . . , 𝑁_ − 1 _,_ 

where the final database denotes D[(] _[𝑁]_[)] ( _𝑚_ ). In practice, repeating this procedure for _𝑁_ rounds produces high-quality databases that better match realistic environment demands. 

**Tool Interface Generation and Verification.** To construct a database-grounded executable toolset, we introduce a coding agent _𝜓_ equipped with a _code compiler_ and _OS tools_ , denoted by T[ˆ] . Given ( _𝑚,_ D[(] _[𝑁]_[)] ( _𝑚_ )), the agent generates candidate tools together with their unit-test sets: 

�( _𝑓,_[ˆ] C[ˆ] ˆ _𝑓_ )� = _𝜓_ ( _𝑚,_ D[(] _[𝑁]_[)] ( _𝑚_ ); _𝜋 𝜃 ,_ T)[ˆ] _, 𝑚_ ∈M _,_ 

> 1https://smithery.ai/servers 

5 

**==> picture [472 x 187] intentionally omitted <==**

**Figure 3 Hierarchical environment taxonomy of Agent-World. Left:** distribution of the 20 first-tier categories with their server counts. **Right:** top-10 second-tier categories ranked by server count. 

where each tool _𝑓_[ˆ] is associated with a set of test cases C[ˆ] ˆ _𝑓_ (i.e., a one-to-many mapping). 

Motivated by a series of automated execution-based verification procedures [19, 123], we then perform cross-validation for quality control. For each candidate tool _𝑓_[ˆ] , its test accuracy is defined as 

**==> picture [166 x 31] intentionally omitted <==**

- A tool is retained only if it satisfies all of the following: 

- the function can be successfully compiled by the Python compiler; 

- Acc( _𝑓_[ˆ] ; C[ˆ] ˆ _𝑓_ ) _>_ 0 _._ 5 on its associated test set; 

- the corresponding environment contains at least one valid tool and one valid test case. 

After filtering, we obtain the quality-controlled tool set F ( _𝑚_ ). Finally, we define the scalable environment ecosystem as 

**==> picture [146 x 12] intentionally omitted <==**

**Environment Taxonomy Construction:** To systematically organize the synthesized environments, we build a hierarchical taxonomy. Based on thousands of environment themes, we apply hierarchical clustering [101] to obtain 50 cluster centers; we then trace back the sample set covered by each cluster and randomly select representative samples. Building on TOUCAN’s taxonomy [113], we use GPT-OSS-120B [70] as a supervised summarization model to identify the central environment theme of each cluster, yielding 50 second-tier labels. 

Since relying solely on LLM summarization may introduce templated text and bias, we invite three annotators to merge the second-tier labels and abstract them into 20 first-tier types; cross-validation and discussion yield the final hierarchical taxonomy of the environment ecosystem. As shown in Figure 3, the taxonomy contains 20 first-tier labels, 50 second-tier labels, and over 2K third-tier labels, providing a foundation for cross-environment task synthesis and stratified arena construction. We denote the set of first-tier categories by C. 

## **3.1.1 Verifiable Task Synthesis** 

After constructing the scalable environment ecosystem E, we synthesize high-quality agentic tasks that simulate diverse real-world tool-use scenarios. To generate complex, long-horizon tasks grounded in reliable execution, 

6 

we use two complementary synthesis strategies: **graph-based task synthesis** for modeling sequential tool dependencies, and **programmatic task synthesis** for modeling complex, non-linear reasoning and control flow. Both approaches rely on sandbox execution to collect execution traces, derive ground-truth answers, and preserve task verifiability [9, 98]. 

**(1) Graph-Based Task Synthesis.** In real-world scenarios, agents often need to invoke a sequence of tools in a specific logical order to accomplish a goal. Thus, a valid tool execution sequence and its returned results naturally define the underlying information requirements and data flows needed to answer a specific user query. Building on this insight, we adopt a reverse-engineering paradigm: we first synthesize a valid tool-call sequence and then generate the corresponding task description [113]. To ensure task rationality and diversity, we build connected tool graphs and walk on the graph to obtain the tool sequences. We detail the construction process as follows: 

**Tool Graph Construction.** For each environment (D[(] _[𝑁]_[)] ( _𝑚_ ) _,_ F ( _𝑚_ )) ∈E, we first construct a fully connected, weighted directed graph _𝐺_ = ( _𝑉, 𝐸_ ), where each node _𝑣_ ∈ _𝑉_ corresponds to a tool _𝑓_ ∈F ( _𝑚_ ) and each edge encodes call dependencies between tools. We define three types of edges, evaluated and assigned by an LLM: 

- **Strong dependency (** _𝑓𝑖_ → _𝑓 𝑗_ **,** _𝑤𝑖𝑗_ = 3 **):** The input of tool _𝑓 𝑗_ strictly relies on the output of tool _𝑓𝑖_ (e.g., calling create_order to obtain an order_id before calling get_order_details). This forms a strictly directed edge, ensuring the most logical data flow. 

- **Weak dependency (** _𝑓𝑖_ ↔ _𝑓 𝑗_ **,** _𝑤𝑖𝑗_ = 2 **):** The input of _𝑓 𝑗 can_ be derived from _𝑓𝑖_ ’s output, but can also be obtained via other means (e.g., querying a database directly or using a constant). This is modeled as a bidirectional edge, offering flexibility during the walk. 

- **Independent edge (** _𝑓𝑖_ ↔ _𝑓 𝑗_ **,** _𝑤𝑖𝑗_ = 1 **):** Tools with no parameter-level dependencies. These edges act as a fallback to guarantee that _𝐺_ is fully connected, preventing dead ends during random walks. 

**Random Walk on Tool Graph.** We generate a raw tool-call sequence _𝜏_ = [ _𝑓_ 1 _, 𝑓_ 2 _, . . . , 𝑓𝑘_ ] by performing a random walk on _𝐺_ . We prioritize starting nodes _𝑓_ 1 that return tool output but have no strong dependency precursors. At step _𝑡_ , the next tool _𝑓𝑡_ +1 is sampled from the successors of _𝑓𝑡_ with a probability distribution biased by the edge weights _𝑤_ , encouraging sequences with realistic reasoning. 

Once the tool sequence _𝜏_ is sampled, we instantiate its input parameters: (1) For strong/weak dependencies, we pass the output of the preceding tool; (2) For independent edges, we randomly sample valid values from the database D[(] _[𝑁]_[)] ( _𝑚_ ). Finally, an LLM reviews the populated chain to prune redundancies, verify logical consistency, and output a refined, executable tool sequence _𝜏_[∗] . 

**Task and Rubric Generation.** Given _𝜏_[∗] , an LLM drafts an initial task description _𝑞𝑖𝑛𝑖𝑡_ . To prevent data leakage, _𝑞𝑖𝑛𝑖𝑡_ is strictly prohibited from containing technical details such as tool names or database schema. Next, we execute _𝜏_[∗] step-by-step within a Python sandbox, recording the intermediate execution trace and the final return results. Observing the actual data fields and formats allows the LLM to refine _𝑞𝑖𝑛𝑖𝑡_ into a highly realistic and well-grounded final query _𝑞 𝑓𝑖𝑛𝑎𝑙_ . Simultaneously, the LLM generates a strictly formatted JSON ground-truth answer _𝑎_[∗] and structured evaluation rubrics _𝑅_ [39, 83, 85, 112]. The rubrics _𝑅_ enable automated evaluation across multiple dimensions, including field completeness, schema matching, and numerical tolerances. 

**Quality Consistency and Verification.** To ensure task stability, we evaluate the generated task ( _𝑞 𝑓𝑖𝑛𝑎𝑙, 𝑎_[∗] ) by deploying a ReAct agent to solve it 5 separate times within the sandbox. We retain the task only if the agent successfully reaches a consistent answer in at least two independent runs. 

**Difficulty Scaling.** To increase task difficulty while maintaining solvability, we complicate the reasoning path in each task. Specifically, we scale difficulty by increasing the maximum step count of the random walk to expand the tool chain, and by increasing the sampling probability of weak dependencies and independent edges to reduce reliance on obvious sequential outputs. We further rewrite the final task description to obscure explicit mentions of tool names and execution logic, forcing the agent to infer the required workflow purely from abstract task goals. The final task set synthesized by graph-based generation is denoted as Xgraph. 

7 

**==> picture [472 x 198] intentionally omitted <==**

**Figure 4 Comprehensive statistics of Agent-World environments and synthesized tasks** , including environment diversity, tool coverage, file-type distribution, and task difficulty characteristics. 

**(2) Programmatic Task Synthesis.** While graph-based synthesis effectively models sequential dependencies, real-world tasks often demand reasoning patterns that cannot be expressed linearly, such as conditional tool usage, multi-step loops, and result aggregation. To capture these behaviors, we introduce programmatic task synthesis. Unlike the graph-based method that simulates step-by-step sequences, this approach directly generates executable Python solutions capable of performing code-based reasoning over provided tools. 

**Task and Solution Code Generation.** We prompt an LLM with the environment’s tool schemas and database descriptions to generate a highly complex task query _𝑞 𝑝𝑟𝑜𝑔_ . The query must focus entirely on task scenarios and objectives without revealing details of tools or databases. Subsequently, the LLM acts as a solver to generate a comprehensive, end-to-end executable Python script _𝜋𝑐𝑜𝑑𝑒_ . This script must load the tool implementations and utilize complex control flows (e.g., for loops, if-else branches, statistical aggregations) to solve _𝑞 𝑝𝑟𝑜𝑔_ . To ensure _𝜋𝑐𝑜𝑑𝑒_ is executable, we wrap this step in a ReAct loop: if the sandbox throws syntax or runtime errors, the agent iteratively debugs and repairs the code. The successfully executed script yields the final ground-truth answer _𝑎_[∗] . 

**Verification Code Generation.** Traditional string-matching evaluation falls short for complex programmatic tasks. Therefore, we input ( _𝑞 𝑝𝑟𝑜𝑔, 𝜋𝑐𝑜𝑑𝑒, 𝑎_[∗] ) to an LLM to generate an executable verification script _𝑉𝑐𝑜𝑑𝑒_ ( _𝑎, 𝑎_[∗] ). The script includes multi-level assertions and custom logic to robustly determine whether the candidate answer _𝑎_ and the underlying database state _𝑠[𝐸]_ satisfy all task constraints. Similar to solution code generation, a ReAct agent debugs _𝑉𝑐𝑜𝑑𝑒_ in the sandbox to guarantee its reliability. 

**Quality Consistency and Verification.** Following the same rigorous filtering protocol as the graph-based method, we execute a ReAct agent 5 times against _𝑞 𝑝𝑟𝑜𝑔_ . The generated verification script _𝑉𝑐𝑜𝑑𝑒_ evaluates the agent’s output. Tasks are preserved only if the agent achieves a stable pass rate (at least 2 successful runs), ensuring the synthesized tasks are challenging yet solvable. 

**Difficulty Scaling.** Similar to graph-based synthesis, we also scale the difficulty of programmatic tasks. Specifically, we increase the number of unique tools and invocations through modifying LLM instructions. We also inject instructions of implementing intricate inter-tool logic such as conditional branches and mandate advanced data operations like cross-database aggregations, sorting, and filtering. Finally, similar to the graph-based approach, we rewrite the task description of any direct references to APIs or execution traces, ensuring the agent must plan complex programmatic logic entirely from high-level user intents. The final task set synthesized by programmatic generation is denoted as Xprog. 

**(3) Static Statistics of Environment--Task Data:** To more comprehensively demonstrate the quality of our 

8 

agentic environment scaling stage, Figure 4 provides a detailed analysis of Agent-World environments and tasks through six subfigures. 

**Environment Diversity:** We observe that **(a)** Agent-World covers a broad range of environment types, with over 2,000 environments in total (1,978 retained after filtering). **(b)** Each environment is equipped with a diverse toolset, averaging more than 10 tools, with some environments containing over 40 tools. **(c)** The overall ecosystem includes 19,822 distinct tools, each with rich parameters, ensuring both atomic functionality and tool diversity. Interestingly, **(d)** the underlying database file types are also highly diverse, including json, csv, sql, and html, as well as environment-specific formats such as tex and yaml. This further reflects the diversity of our databases and their alignment with real-world workspace file formats. 

**Task Difficulty:** As shown in **(e)** , all synthesized tasks contain at least 7 interaction turns, with an average of over 20 turns and a non-trivial portion exceeding 40 turns, already indicating substantial difficulty. To quantify difficulty more directly, in **(f)** we evaluate task execution under Pass@10 using the strong proprietary model Doubao-Seed-2.0-pro [81]. Only a small fraction of tasks are solved in all 10 attempts; most are solved only once out of 10, and some are not solved at all. This shows that our difficulty scaling strategy is effective at increasing task complexity. 

Beyond aggregate statistics, we provide reader-facing environment cards in Appendix B, where each card summarizes a seed domain’s on-disk layout and representative callable tool interfaces. In addition, Appendix C presents verifiable tasks, including the environment, tools, rubrics, and interaction trajectories. 

## **3.2 Continuous Self-Evolving Agent Training** 

In this section, we introduce continuous self-evolving agent training. Given a scalable environment ecosystem E = {(D[(] _[𝑁]_[)] ( _𝑚_ ) _,_ F ( _𝑚_ )) | _𝑚_ ∈M}, where each environment pairs a database with an executable toolset, we train general-purpose agents with multi-environment “agent–tool–database” interaction rollouts and executable rewards. Crucially, E also serves as a dynamic diagnostic arena: the current policy is evaluated on fresh tasks in held-out environments, its capability gaps are identified from executable evidence, and the resulting diagnosis guides targeted environment-task expansion. This creates a self-evolving loop in which agent policies and environments co-evolve over training rounds. 

## **3.2.1 Multi-Environment Agent Reinforcement Learning** 

After constructing a scalable environment ecosystem and synthesizing verifiable tasks, we perform multienvironment agent RL to improve state-aware reasoning, long-horizon tool use, and environment interaction robustness. 

**Multi-environment Rollout.** Unlike static tool-calling scenarios, we implement a closed-loop interaction among three components: 

- **An LLM policy** _𝜋 𝜃_ , which generates the next action conditioned on the dialogue history and tool feedback; 

- **A tool interface/runtime** , which executes the environment-specific tool set F ( _𝑚_ ) and maintains environmentside states (database connections, caches, etc.); 

- **A database state** D[(] _[𝑁]_[)] ( _𝑚_ ), which serves as the read/write substrate for tool execution and provides a verifiable, updatable structured data backbone. 

At each step, the model produces both natural-language reasoning and tool/action decisions. When a tool call is triggered, the interface executes the selected tool in a sandboxed environment to read or update the environment database state, and returns structured observations to the policy for subsequent decision making. 

Formally, given a task _𝑥_ and its training environment (D[(] _[𝑁]_[)] ( _𝑚_ ) _,_ F ( _𝑚_ )) ∈E, following Section 2, the policy _𝜋 𝜃_ samples an action _𝑎𝑡_ based on the instruction and history _ℎ𝑡_ = ( _𝑜_ 0 _, 𝑎_ 0 _, . . . , 𝑜𝑡_ ). If _𝑎𝑡_ ∈ _𝐴_ tool, it executes _𝑓_ ∈F ( _𝑚_ ) on D[(] _[𝑁]_[)] ( _𝑚_ ) and returns a structured observation _𝑜𝑡[𝐸]_ +1[∈] _[𝑂][𝐸]_[;][if] _[𝑎][𝑡]_[∈] _[𝐴]_[resp][,][it][outputs][a][natural-] language response (typically the final answer or completion marker) and terminates the trace. This yields a model output _𝑦_ = ( _𝜏, 𝑎_ final), where _𝜏_ = ( _𝑜_ 0 _, 𝑎_ 0 _, . . . , 𝑜𝑇 , 𝑎𝑇_ ) is the interaction trajectory and _𝑎_ final is the final answer. Following Group Relative Policy Optimization (GRPO) [84], we sample _𝑁_ outputs per task 

9 

**==> picture [472 x 227] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Multi-Environment Agent Reinforcement Learning<br>Multi-Environment Rollout Module  Output Reference Reward Types  Reward Advantages<br>Question ModelPolicy InterfaceTool  DatabaseState ���⋯�12⋯1����⋯��111+�+2+12 RewardModelModel Rubric-conditioned Code Execution RewardReward ���⋯��12⋯��1�⋯1��+�11+2+12 ComputationGroup ���⋯��12��⋯1�⋯��111+�+2+12<br>(b) Self-Evolving Agent Arena Dynamic Evaluation Tasks Synthesis<br>Environments Agent-Arena<br>Ecosystems Arena Construction Dynamic Sampling<br>Evaluation Dynamic Environment Sampling Round-by-round dynamic sampling based  Out-of-distribution task synthesis for  Verifiable Task Synthesis<br>Agent-World on an environment taxonomy sampled environments.<br>Evolved Scalable environment synthesis with  Vanilla<br>Agent continuous self-evolving training Agent Task Generation Guidelines Error Trajectory<br>Target Tasks EnvironmentsWeak      Weak Environments<br>Continue RL      Error Tool-use behaviors Diagnosis<br>Targeted Data Evolving Agentic Diagnosis      State Update Error Agent<br>**----- End of picture text -----**<br>


**Figure 5 The Overall Framework of Continuous Self-Evolving Agent Training.** The agent is trained with multi-environment RL under executable rewards (top), evaluated in a dynamic arena, diagnosed for capability gaps, and improved through targeted environment-task expansion (bottom). 

_𝑥_ , and tasks within each global batch are paired with independent and dynamic environments to realize multi-environment rollouts. 

**Structured Verifiable Reward.** Reward signals define the optimization objective and directly guide policy behavior. Distinct from prior static tool-RL settings [20, 26], automatic reward assignment for environment agents must account for multiple factors beyond answer correctness, including environment state, efficiency constraints, and format compliance. Accordingly, we instantiate two reward types. 

(i) **Graph-based tasks** (Xgraph) provide a structured rubric _𝑅_ = { _𝑟 𝑗_ } _[𝑛] 𝑗_ =1[(schema][matching,][fact][checking,][etc.).] We use a rubric-conditioned LLM-as-judge to evaluate each criterion _𝑟 𝑗_ from model output _𝑦_ under task _𝑥_ , and compute an overall pass rate by averaging criterion-level pass indicators. 

(ii) **Programmatic tasks** (Xprog) provide an executable validation script _𝑉_ code per task, which we run in the sandbox to verify either the predicted answer or the resulting database state. Therefore, the output-level reward is computed as 

**==> picture [284 x 33] intentionally omitted <==**

where I[·] is the indicator function. Judge( _𝑥, 𝑦, 𝑟 𝑗_ ) denotes a rubric-conditioned LLM judge that assesses whether model output _𝑦_ satisfies criterion _𝑟 𝑗_ under task _𝑥_ . Execute( _𝑉_ code( _𝑦, 𝑦_[∗] )) denotes running the taskspecific validation script _𝑉_ code in a sandbox over model output _𝑦_ to verify that answer/state are satisfied with the ground truth _𝑦_[∗] . 

**Policy Update.** To enable stable training with environment interaction, we adopt Group Relative Policy Optimization (GRPO) [84] to directly maximize the verifiable returns defined above. Concretely, for each input task _𝑥_ sampled from dataset _𝐷_ , we draw a group of _𝐺_ trajectories/outputs { _𝑦𝑖_ } _𝑖[𝐺]_ =1[from][the][behavior] policy _𝜋 𝜃_ old (· | _𝑥_ ), compute token-level advantages _𝐴_[ˆ] _𝑖,𝑡_ , and update _𝜋 𝜃_ by maximizing the GRPO objective 

10 

with a clipped importance ratio and a KL penalty to a reference policy _𝜋_ ref: 

**==> picture [416 x 65] intentionally omitted <==**

where _𝜖_ and _𝛽_ are hyperparameters, _𝑦𝑖_ denotes the model output (including interaction trajectory and final answer), and _𝐴_[ˆ] _𝑖,𝑡_ is the normalized advantage of the _𝑖_ -th rollout within the group. 

## **3.2.2 Self-Evolving Agent Arena** 

**Motivation.** Our scalable environment ecosystem E = {(D[(] _[𝑁]_[)] ( _𝑚_ ) _,_ F ( _𝑚_ )) | _𝑚_ ∈M} serves not only as a training source but also as an agentic diagnostic arena. Beyond synthesizing training data, we aim to continuously identify weaknesses of the current agent policy and then expand environments and tasks in a targeted manner to close those gaps. This yields a self-reinforcing loop in which evaluation, diagnosis, and data generation evolve together with the agent. 

**Arena Construction.** Based on the hierarchical environment taxonomy (Sec. 3.1), we construct an evaluation arena by stratified sampling. Specifically, for each first-tier category _𝑐_ ∈C, we randomly select _𝐾_ environments ( _𝐾_ = 5) and merge them into the arena set Earena = {(D[(] _[𝑁]_[)] ( _𝑚𝑖_ ) _,_ F ( _𝑚𝑖_ ))} _𝑖_[| E] =1[arena][|] . This design ensures broad coverage over different environment types while keeping evaluation cost controllable. 

**Dynamic Evaluation Task Synthesis.** For each arena environment (D[(] _[𝑁]_[)] ( _𝑚𝑖_ ) _,_ F ( _𝑚𝑖_ )) ∈Earena, we follow Section 3.1.1 and synthesize a fresh batch of verifiable tasks and validators at each iteration. Concretely, at iteration _𝑟_ we instantiate a task set Xarena[(] _[𝑟]_[)][(] _[𝑚] 𝑖_[)][consisting][of][both][graph-based][tasks][and][programmatic] tasks, each paired with an executable rubric _𝑅_ or verification code _𝑉_ code. The full evaluation set is defined as Xarena[(] _[𝑟]_[)][=][ �] _𝑖_[X] arena[(] _[𝑟]_[)][(] _[𝑚] 𝑖_[)][.][Importantly,][both][the][sampled][environments][and][the][synthesized][tasks][are][dynamic] across rounds, preventing overfitting to a static evaluation and enabling continual diagnosis. 

**Agentic Diagnosis.** Given a trained agent policy _𝜋 𝜃_ ( _𝑟_ ) , we evaluate it on synthesized tasks Xarena[(] _[𝑟]_[)][under] the agent-tool-database execution protocol, with task-level assessment performed by the corresponding executable rubric _𝑅_ or verification code _𝑉_ code. 

We then employ an auto-diagnosis agent _𝛿_ , equipped with a Python interpreter and search tools, to analyze failure patterns. The diagnosis agent takes as input: **(i)** per-task failure traces (tool logs, intermediate observations, and validator feedback), **(ii)** error distribution statistics by environment and taxonomy category, and **(iii)** environment metadata (tool schemas and database descriptions). 

The diagnosis agent outputs **(a)** a ranked set of weak environments W[(] _[𝑟]_[)] ⊆Earena and **(b)** environmentspecific task-generation guidelines Gguide[(] _[𝑟]_[)][(] _[𝑚]_[)][that][characterize][missing][capabilities][(e.g.,][erroneous][tool][use] or state-update mistakes). These outputs serve as anchors for subsequent environment and task expansion. Detailed prompts for agentic diagnosis are provided in Appendix A. 

**Agent-Environment Co-Evolution.** Conditioned on W[(] _[𝑟]_[)] and Gguide[(] _[𝑟]_[)][=][{G] guide[(] _[𝑟]_[)][(] _[𝑚][𝑖]_[)|(D][ (] _[𝑁]_[)][ (] _[𝑚][𝑖]_[)] _[,]_[ F (] _[𝑚][𝑖]_[))][∈W][ (] _[𝑟]_[)][}][,] we re-run the verifiable task synthesis pipeline (Sec. 3.1.1) to generate a targeted training set Xtarget[(] _[𝑟]_[)][, optionally] accompanied by environment expansion via database complexification when the weakness is due to insufficient state diversity. 

Starting from _𝜋 𝜃_ ( _𝑟_ ) , we then perform multi-environment agent RL (Sec. 3.2.1) on the augmented data to obtain an improved policy _𝜋 𝜃_ ( _𝑟_ +1) . Iterating the above steps yields a self-evolving agent-arena loop: 

**==> picture [263 x 17] intentionally omitted <==**

This arena-driven loop turns scalable environments into an automated curriculum engine, continuously driving targeted learning and enabling the co-evolution of agent policies and environments. 

11 

## **Algorithm 1: Self-Evolving Agent Arena Loop** 

**Input:** agent environment arena Earena ⊂E; initial policy _𝜋 𝜃_ (0) ; number of evolving rounds _𝑅_ **Output:** Evolved policy _𝜋 𝜃_ ( _𝑅_ ) 

- **1 for** _𝑟_ = 0 _, . . . , 𝑅_ − 1 **do** 

// **Phase 1: Dynamic Evaluation Task Synthesis 2 foreach** (D[(] _[𝑁]_[)] ( _𝑚𝑖_ ) _,_ F ( _𝑚𝑖_ )) ∈Earena **do 3** Synthesize fresh verifiable tasks Xarena[(] _[𝑟]_[)][(] _[𝑚] 𝑖_[)][with][executable][rubric] _[𝑅]_[or][verification][code] _[ 𝑉]_ code[;] **4** Define the full evaluation set Xarena[(] _[𝑟]_[)][=][ �] _𝑖_[X] arena[(] _[𝑟]_[)][(] _[𝑚] 𝑖_[)][;] **5** Evaluate _𝜋 𝜃_ ( _𝑟_ ) on Xarena[(] _[𝑟]_[)][under][agent-tool-database][execution][with][assessment][by] _[𝑅]_[or] _[ 𝑉]_ code[;] // **Phase 2: Agentic diagnosis 6** Input per-task failure traces, environment error statistics and metadata to diagnosis agent _𝛿_ ; **7** outputs weak environments W[(] _[𝑟]_[)] ⊆Earena and task-generation guidelines Gguide[(] _[𝑟]_[)][(] _[𝑚]_[)][;] // **Phase 3: Agent-Environment Co-Evolution. 8 foreach** (D[(] _[𝑁]_[)] ( _𝑚_ ) _,_ F ( _𝑚_ )) ∈ W[(] _[𝑟]_[)] **do 9** Complexify database: D[(] _[𝑁]_[)] ( _𝑚_ ) ← _𝜙_ (D[(] _[𝑁]_[)] ( _𝑚_ ) _,_ ·); **10** Generate targeted tasks Xtarget[(] _[𝑟]_[)][(] _[𝑚]_[)][conditioned][on][G] guide[(] _[𝑟]_[)][(] _[𝑚]_[)][;] **11** Define Xtarget[(] _[𝑟]_[)][=][ �] (D[(] _[𝑁]_[)] ( _𝑚_ ) _,_ F( _𝑚_ ))∈W[(] _[𝑟]_[)][X] target[(] _[𝑟]_[)][(] _[𝑚]_[)][;] **12** Continue RL on Xtarget[(] _[𝑟]_[)][obtain:] _[𝜋] 𝜃_[(] _[𝑟]_[+][1][)][←] _[𝜋] 𝜃_[(] _[𝑟]_[)][ ;] 

**13 return** _𝜋 𝜃_ ( _𝑅_ ) ; 

## **4 Experiment** 

In this section, we conduct experiments to evaluate the effectiveness of Agent-World and further analyze its key properties. First, we introduce the details of the experimental settings (Sec. 4.1). Next, we present the main results of Agent-Wolrd (Sec. 4.2). Finally, we present quantitative and qualitative analyses of our approach (Sec. 4.3). 

## **4.1 Experimental Settings** 

In this part, we introduce the datasets used for training and evaluation, the baseline approaches, and the implementation details. 

**Baselines.** We compare Agent-World against three baseline groups, consistent with Table 1: 

- **Frontier Proprietary Models:** GPT-5.2 High [69], Claude Sonnet-4.5 [4], Gemini-3 Pro [23], Seed2.0 [8]. 

- **Open-Source Foundation Models (8B–685B):** DeepSeek-V3.2-685B [58], GPT-OSS-120B [70], Qwen3235B-A22B [114], and Qwen3-8B, 14B, 32B [114]. 

- **Open-Source Environment Scaling Methods (7B-14B):** Simulator-8B [54], TOUCAN-7B [113], EnvScaler8B [89], AWM-8B, 14B [100], and ScaleEnv-8B [98]. 

**Evaluation Benchmarks.** We evaluate Agent-World on **23 benchmarks** spanning complementary capabilities: 

- **Core agentic tool-use suites:** MCP-Mark [106], BFCL V4 [73], and _𝜏_[2] -Bench [7]. 

- **Advanced AI assistant benchmarks:** SkillsBench [47], ARC-AGI-2 [15], and Claw-Eval [16]. 

- **General reasoning benchmarks:** MATH500 [57], GSM8K [17], MATH [36], AIME24 [1], AIME25 [2], KOR-Bench (Cipher) [63], and OlympiadBench ( _𝑂𝐸_  𝑇𝑂_  𝑚𝑎𝑡ℎ𝑠_  𝑒𝑛_  𝐶𝑂𝑀𝑃_ ) [33]. 

- **Agentic search & coding benchmarks:** WebWalkerQA [104], SWE-Bench Verified (SWE) [41], SWEbench Multilingual [122], Terminal-Bench 1.0, Terminal-Bench 2.0 [64], General AI Assistants (GAIA) [65], 

12 

and Humanity’s Last Exam (HLE) [74]. 

- **Knowledge and MCP benchmarks:** MMLU [35], SuperGPQA [96], MCP-Universe 5 sub-domains (Financial Analysis, Browser Automation, Web Searching, Location Navigation, and Repository Management) [62]. 

All baselines and benchmarks are evaluated using in-house evaluation framework, with results aligned to official scores. Following prior work [20, 21, 43, 48, 50], we use sampled subsets for some benchmarks (e.g., GAIA and HLE) to accelerate evaluation. 

**Implementation Details.** In Agentic Environment-Task Discovery, we use GPT-OSS-120B [70] as the policy model for environment mining. The same policy model is also used for task synthesis and for generating code and rubric artifacts across different toolsets. In Agentic Diagnosis, GPT-OSS-120B is likewise used to execute diagnosis trajectories and identify failure modes. For training initialization, we perform a cold-start supervised fine-tuning stage using the same data-synthesis strategy as Agentic Environment-Task Discovery, where 40K trajectories are generated by an in-house Doubao-Seed-1.8 policy version model [82]. 

After cold-start SFT, we initialize the Qwen3-8B/14B backbones [114], synthesize 5K RL samples, and apply GRPO [84] as the RLVR algorithm for subsequent training. To enhance training stability, we follow prior work [118] and set the clip ratio _𝜀_ low = 0 _._ 2 and _𝜀_ high = 0 _._ 28. Moreover, the maximum trajectory length is set to 80K tokens, and the maximum generation length per step is capped at 32k tokens. In each training step, we sample 32 tasks and perform 8 rollouts to collect RLVR experience, with temperature = 1 _._ 0 and top_p = 1 _._ 0. For evaluation, we also use temperature = 1 _._ 0 and top_p = 1 _._ 0 for decoding. To reduce random variance, we repeat each experiment eight times and report average accuracy (%). 

## **4.2 Main Results** 

The experimental results are shown in Table 1. Overall, Agent-World consistently outperforms existing environment-scaling baselines across diverse agentic tool-use benchmarks, demonstrating stronger robustness and more comprehensive generalization in long-horizon settings. We summarize the main findings as follows. 

(1) **Foundation models remain limited in complex agentic tool-use scenarios.** Even advanced proprietary models show clear limitations on challenging benchmarks. For instance, GPT-5.2 High achieves only 53.1% on MCP-Mark, while Gemini-3 Pro reaches 50.8%. Moreover, open-source foundation models are even more constrained, with GPT-OSS-120B and Qwen3-235B-A22B scoring only 4.7% and 5.8% on MCP-Mark. Since these benchmarks cover diverse stateful environments, the results suggest that current foundation models still struggle with long-horizon tool use requiring multi-step planning, tool orchestration, and state tracking. 

(2) **Existing environment-scaling methods still suffer from uneven capability gains.** Compared with the Qwen3 backbones, existing environment-scaling methods improve some benchmarks, but their gains remain uneven across environments. Simulator-based methods such as Simulator-8B achieve good results on _𝜏_[2] -Bench, yet still perform poorly on MCP-Mark and BFCL V4, suggesting that simulated environments are insufficient to capture complex real-world state transitions. programmatic environment-scaling methods such as EnvScaler-8B and AWM-8B/14B provide broader gains, but still show clear weaknesses on specific environments, including GitHub and Notion. This highlights that robust generalization depends not only on realistic feedback, but also on the diversity and quality of synthesized environments. 

(3) **Agent-World achieves more consistent cross-environment generalization.** Under the same training setting, Agent-World consistently outperforms prior environment-scaling baselines across all three benchmark suites. In detail, Agent-World-8B achieves 61.8% on _𝜏_[2] -Bench, 51.4% on BFCL V4, and 8.9% on MCP-Mark. These results clearly outperform EnvScaler-8B, ScaleEnv-8B and even Qwen3-235B-A22B. 

Moreover, Agent-World-14B achieves an additional improvement of about 5% over Agent-World-8B. It not only surpasses all prior environment-scaling baselines, but also delivers competitive performance against large open-source LLMs, particularly DeepSeek-V3.2-685B on BFCL-V4 (55.8% vs. 54.1%). These results indicate that Agent-World produces more consistent gains across diverse benchmarks and environments. We attribute 

13 

**Table 1 Main results on agentic tool-use benchmarks.** We report accuracy (%) across three benchmark suites: MCP-Mark, BFCL V4, and _𝜏_[2] -Bench. In the _Open-Source Environment Scaling Methods_ block, the best result in each column is marked in **bold** and the second best is underlined. 

|**Method**|**MCP-Mark**<br>File.<br>Github<br>Notion<br>Play.<br>Post. **Avg.**|**BFCL V4**<br> WebSearch Memory Multi-T. No live Live Relev. Irrelev. **Avg.**|_𝜏_2**-Bench**<br> Retail Telecom Airline **Avg.**|
|---|---|---|---|
||**Fronti**|**er Proprietary Models**||
|GPT-5.2 High<br>○Claude Sonnet-4.5<br>Gemini-3 Pro<br>Seed 2.0|60.0<br>47.8<br>42.9<br>40.0<br>66.7<br>53.1<br>32.5<br>29.4<br>25.0<br>27.0<br>50.0<br>33.3<br>56.7<br>45.7<br>43.8<br>40.0<br>70.2<br>50.8<br>60.0<br>39.1<br>53.6<br>40.0<br>81.0<br>54.7|75.5<br>45.8<br>48.5<br>81.9<br>70.4<br>75.0<br>88.7<br>62.9<br>81.0<br>65.0<br>61.4<br>88.7<br>81.1<br>68.8<br>86.6<br>73.2<br>80.0<br>61.7<br>60.8<br>90.7<br>83.1<br>68.8<br>85.6<br>72.5<br>92.0<br>57.8<br>62.3<br>89.0<br>82.2<br>76.6<br>75.0<br>73.4|81.6<br>95.8<br>62.5<br>80.2<br>86.2<br>98.0<br>70.1<br>84.7<br>85.3<br>98.0<br>72.7<br>85.4<br>90.4<br>94.2<br>64.4<br>83.0|
||**Open-Source**|**Foundation Models (8B–685B)**||
|DeepSeek-V3.2-685B<br>GPT-OSS-120B<br>Qwen3-8B<br>Qwen3-14B<br>Qwen3-32B<br>Qwen3-235B-A22B|36.7<br>20.7<br>45.5<br>17.0<br>66.6<br>36.7<br>5.8<br>4.4<br>3.6<br>3.0<br>7.1<br>4.7<br>3.3<br>0.0<br>0.0<br>**4.0**<br>4.8<br>2.4<br>3.3<br>**4.4**<br>0.0<br>0.0<br>9.5<br>3.4<br>10.0<br>0<br>3.6<br>0<br>23.8<br>7.5<br>13.3<br>0<br>10.7<br>0<br>4.8<br>5.8|69.5<br>54.2<br>37.4<br>34.9<br>53.7<br>37.5<br>93.2<br>54.1<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>7.0<br>17.6<br>35.4<br>**90.2**<br>80.9<br>81.3<br>77.2<br>40.4<br>4.0<br>19.8<br>36.9<br>90.0<br>**82.4**<br>81.3<br>79.4<br>41.0<br>26.0<br>15.7<br>43.3<br>90.3<br>82.0<br>81.3<br>82.4<br>46.7<br>54.0<br>23.9<br>45.4<br>37.4<br>68.9<br>87.5<br>81.7<br>47.9|–<br>–<br>–<br>80.3<br>67.8<br>49.2<br>48.0<br>55.0<br>34.0<br>18.0<br>26.5<br>26.2<br>55.3<br>14.9<br>27.0<br>32.4<br>59.5<br>27.2<br>48.0<br>44.9<br>71.9<br>58.0<br>45.6<br>58.5|
||**Open-Source Envir**|**onment Scaling Methods (7B-14B)**||
|Simulator-8B<br>TOUCAN-7B<br>EnvScaler-8B<br>AWM-8B<br>AWM-14B<br>ScaleEnv-8B|3.3<br>0.0<br>0.0<br>**4.0**<br>4.8<br>2.4<br>0.0<br>0.0<br>0.0<br>0.0<br>4.8<br>1.0<br>10.0<br>4.4<br>0.0<br>**4.0**<br>9.5<br>5.6<br>3.3<br>0.0<br>0.0<br>**4.0**<br>4.8<br>2.4<br>3.3<br>**8.7**<br>0.0<br>**4.0**<br>9.5<br>5.1<br>–<br>–<br>–<br>–<br>–<br>–|17.5<br>6.0<br>4.1<br>47.6<br>44.6<br>31.3<br>**87.3**<br>23.9<br>21.0<br>18.5<br>17.8<br>81.0<br>73.9<br>81.3<br>78.6<br>36.6<br>23.0<br>21.9<br>47.1<br>88.5<br>**82.2**<br>**93.8**<br>74.6<br>47.6<br>9.5<br>15.7<br>34.9<br>**90.2**<br>80.5<br>**93.8**<br>73.9<br>40.0<br>10.0<br>19.8<br>37.6<br>**90.2**<br>81.5<br>75.0<br>79.4<br>42.4<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–|32.2<br>29.2<br>34.0<br>31.8<br>22.8<br>10.5<br>20.0<br>17.7<br>49.6<br>32.7<br>31.5<br>37.9<br>41.2<br>38.5<br>23.5<br>34.4<br>63.6<br>17.8<br>31.5<br>39.0<br>50.9<br>27.2<br>37.5<br>38.5|
|**Agent-World-8B**|13.3<br>4.4<br>**3.6**<br>**4.0**<br>19.1<br>8.9|47.0<br>21.7<br>44.5<br>83.3<br>79.6<br>**93.8**<br>80.2<br>51.4|72.8<br>50.9<br>40.0<br>61.8|
|**Agent-World-14B**|**16.6**<br>4.4<br>**3.6**<br>**4.0**<br>**38.1**<br>**13.3**|**53.0**<br>**23.9**<br>**53.9**<br>82.3<br>79.3<br>**93.8**<br>81.0<br>**55.8**|**74.5**<br>**56.1**<br>**52.0**<br>**65.4**|



this advantage to its unified framework, which tightly integrates scalable environment-task discovery with continuous self-evolving agent training. 

## **4.3 Quantitative and Qualitative Analyses** 

## **4.3.1 Generalization on Long-horizon Agentic Reasoning Scenarios** 

To further assess long-horizon generalization in agentic tool-use scenarios, we compare Agent-World-8B against strong baselines on 17 benchmarks, organized into three complementary perspectives in Figure 6: **General Reasoning** , **Agentic Search & Coding** , and **Knowledge & MCP** . Overall, Agent-World demonstrates strong cross-domain generalization without benchmark-specific tuning, further validating the transferability of our environment-scaling paradigm. The key findings are summarized below. 

(1) **Agent-World strengthens agentic behavior while preserving strong general reasoning.** On the **General Reasoning** axis, Agent-World-8B achieves the best overall profile across seven widely-used reasoning benchmarks (MATH500, GSM8K, MATH, AIME24, AIME25, KOR-Bench, and OlympiadBench), with clear gains on most dimensions and no degradation on core math reasoning. This indicates that our Agent-World training pipeline improves difficult multi-step reasoning without sacrificing foundational reasoning capability. 

(2) **The largest gains are observed in long-horizon search and coding tasks.** On **Agentic Search & Coding** , Agent-World-8B consistently outperforms both baselines on WebWalkerQA, SWE-bench Verified, SWE-bench Multilingual, Terminal 1.0, Terminal 2.0, GAIA, and HLE. These benchmarks stress iterative planning, long-horizon software engineering, deep information retrieval, and multi-tool coordination. The consistent improvements indicate that Agent-World acquires transferable agentic strategies rather than benchmarkspecific heuristics. Notably, EnvScaler-8B underperforms its Qwen3-8B backbone on SWE and Terminal 1.0, possibly because its environment expansion is less effective at eliciting complex software-engineering reasoning patterns. 

14 

**==> picture [472 x 163] intentionally omitted <==**

**Figure 6 Generalization across long-horizon agentic reasoning scenarios.** Comparison of Qwen3-8B, EnvScaler8B, and Agent-World-8B from three capability groups: General Reasoning, Agentic Search & Coding, and Knowledge & MCP. 

**==> picture [472 x 146] intentionally omitted <==**

**Figure 7 Generalization on advanced agentic assistant benchmarks.** Comparison of Qwen3, EnvScaler, AWM, and Agent-World series on SkillsBench, ARC-AGI-2, and Claw-Eval. 

(3) **Agent-World shows stronger robustness in heterogeneous knowledge and MCP environments.** On **Knowledge & MCP** , Agent-World-8B also substantially outperforms baselines on five relatively orthogonal MCPUniverse capabilities: Browser Automation, Web Searching, Location Navigation, Repository Management, and Financial Analysis. In addition, Agent-World-8B maintains consistent improvements on knowledge-centric dimensions (e.g., MMLU and SuperGPQA), highlighting stronger compositional generalization and adaptation to structurally diverse external tools. 

## **4.3.2 Generalization on Agentic AI Assistant Scenarios** 

To further stress-test transfer in advanced assistant settings, we evaluate on three recent highly challenging AI Assistant benchmarks: **SkillsBench** , **ARC-AGI-2** , and **ClawEval** . These benchmarks emphasize long-horizon planning and execution in real-world assistant scenarios. We have the following observations: 

(1) **Existing open-source baselines struggle in real-world AI assistant settings.** Most baseline models obtain average scores below 20% across the three benchmarks and do not show consistent gains from 8B to 14B. For example, Qwen3 drops on ClawEval (25.6% → 24.7%), and AWM shows uneven improvements across tasks. This suggests that naive parameter scaling alone is insufficient for stable long-horizon agentic generalization. 

(2) **Agent-World generalizes strongly to unseen advanced assistant domains.** Without benchmark-specific training, Agent-World still outperforms strong open-source baselines on these challenging settings. At 

15 

**==> picture [472 x 125] intentionally omitted <==**

**Figure 8 Scaling relationship of training environments:** Downstream agent performance scales positively with the number of synthesized training environments. 

8B, Agent-World achieves 9.2%/6.5%/30.5% on SkillsBench/ARC-AGI-2/Claw-Eval, surpassing Qwen3-8B, EnvScaler-8B, and AWM-8B across all three tasks. 

(3) **Agent-World exhibits stable cross-scale gains.** Unlike the unstable scaling trends of several baselines, Agent-World improves consistently from 8B to 14B (SkillsBench: 9.2% → 12.6%, ARC-AGI-2: 6.5% → 8.5%, Claw-Eval: 30.5% → 31.5%). This supports that our method remains effective across parameter scales and transfers robustly to complex, integrated assistant scenarios. 

## **4.3.3 Scaling Analysis of Training Environments** 

To analyze how environment scaling affects downstream agentic tool-use performance, we progressively increase the number of training environments from 0 to 10, 100, 500, 1000, and 2000 (1,978), and evaluate the resulting models on four representative domains in Figure 1: MCPMark (Postgres), BFCL (WebSearch), BFCL (Multi-Turn), and _𝜏_[2] -Bench (Airline). 

Overall, performance improves consistently across all four domains as the environment scale grows, indicating a clear positive scaling relationship. Averaged over the four domains, the score rises from 18.4% to 38.5% (+20.1 points), more than doubling the initial level. 

A notable trend is the stage-wise gain pattern: performance jumps markedly from 10 to 100 environments and again from 100 to 500, suggesting that moderate-scale expansion rapidly improves coverage of critical interaction patterns. This effect is especially evident on BFCL-V4 and MCPMark: MCPMark (Postgres) improves from 4.8% to 19.9%, while BFCL (WebSearch) increases from 7.0% to 47.0%. BFCL (Multi-Turn) and _𝜏_[2] -Bench (Airline) also improve steadily, indicating broad transfer across task types. 

From 500 to 2000 environments, the trend remains upward but the marginal improvement gradually decreases, indicating diminishing-yet-positive returns at larger scales. This suggests that early expansion mainly captures missing high-impact environment diversity, while later expansion contributes finer-grained robustness gains. 

## **4.3.4 Analysis of Continuous Self-Evolution** 

To validate Continuous Self-Evolving Agent Training, we run the same tworound self-evolving arena loop (Sec. 3.2) from two different starting points: Agent-World-14B and the EnvScaler-8B base model. In each round, the current policy is first evaluated on newly synthesized verifiable tasks in held-out arena environments; a diagnosis agent then identifies weak environments and failure 

**Table 2 Effect of continuous self-evolution.** We run iterative selfevolving loops for Agent-World-14B and the EnvScaler-8B base model. 

||**Model / Round**<br>Agent-World-14B (base)<br>+1 round|_𝜏_2**-Bench**<br>60.2<br>63.5 (+3.3)|**BFCL-V4**<br>52.4<br>54.9 (+2.5)|**MCP-Mark (Post.)**<br>29.5<br>36.3 (+6.8)|
|---|---|---|---|---|
||+2 rounds<br>EnvScaler-8B (base)<br>+1 round<br>+2 rounds|**65.4** (+1.9) <br>37.9<br>40.2 (+2.3)<br>**41.6** (+1.4)|**55.8** (+0.9)<br>47.6<br>49.1 (+1.5)<br>**50.0** (+0.9)|**38.1** (+1.8)<br>9.5<br>13.9 (+4.4)<br>**15.1** (+1.2)|



16 

**==> picture [400 x 179] intentionally omitted <==**

**----- Start of picture text -----**<br>
0.40<br>0.7<br>0.35<br>0.6<br>0.30<br>0.5<br>0.25<br>0.4<br>0.20<br>0.3<br>0.15<br>0.2 Qwen3-14b Qwen3-8b Qwen3-14b Qwen3-8b<br>0.10<br>0 50 100 150 200 250 300 0 50 100 150 200 250 300<br>Step Step<br>(a) training score (b) training entropy<br>score<br>entropy<br>**----- End of picture text -----**<br>


**Figure 9 Training Dynamics of Agent-World.** (a) Training reward score and (b) actor entropy over training steps for Qwen3-8B and Qwen3-14B backbones using GRPO on synthesized environments. Curves are exponentially smoothed for clarity. 

modes from executable traces; finally, targeted synthesis and continual RL produce the next-round policy. 

Table 2 shows monotonic gains on all three evaluation suites for both models. For Agent-World-14B, performance on _𝜏_[2] -Bench/BFCL-V4/MCP-Mark improves from 45.3%/52.4%/29.5% to 50.5%/55.8%/38.1% after two rounds. Importantly, EnvScaler-8B also improves from 37.9%/47.6%/9.5% to 41.6%/50.0%/15.1%, indicating that the loop not only benefits our base model but also yields sustained gains for other environmentscaling baselines without relying on Agent-World initialization. 

Notably, the largest gains across two rounds appear on MCP-Mark: +8.6% for Agent-World and +5.6% for EnvScaler. This benchmark requires stronger state tracking and deeper interaction with realistic MCP server environments. This matches our self-evolving objective in Sec. 3.2: diagnosis continually localizes environment-specific weaknesses from closed-loop traces, while targeted synthesis generates harder instances around those failures, which is particularly beneficial for challenging agentic execution scenarios. BFCL-V4 and _𝜏_[2] -Bench also improve steadily, indicating concurrent gains in environment grounding and multi-turn tool coordination. 

Furthermore, second-round gains are smaller than first-round gains but remain positive, reflecting diminishing yet still effective returns. From the environment-diagnosis perspective, we find that early rounds mainly fix pattern-level errors in unfamiliar environment interactions, while later rounds focus on residual failures, especially in long-horizon complex interaction cases. 

Overall, Agent-World treats scalable environments as a persistent diagnostic arena and achieves continual policy improvement through agent-environment co-evolution, substantially outperforming one-pass static training. 

## **4.4 Training Dynamics of Agent-World.** 

As shown in Figure 9, we present the multi-environment reinforcement learning curves of Agent-World-8B and Agent-World-14B. We observe clear upward reward trends for both Qwen3-8B/14B backbones. These results indicate that policy performance improves steadily under GRPO supervision; this trend is also consistent across different environment complexities, further supporting the general effectiveness of multi-turn RL with executable rewards. Meanwhile, tool-use training shows relatively stable entropy growth over time (Figure 9b). This suggests that as the model gradually adapts to unseen APIs and heterogeneous state transitions, it maintains or even expands its exploration space, learning new interaction patterns instead of collapsing prematurely into narrow exploitation. This behavior indicates that Agent-World sustains exploration of agent execution patterns in structurally diverse and highly interactive real-world MCP environments. 

17 

## **5 Related Work** 

## **5.1 Scalable Environment Synthesis for Agent Training** 

As agent training shifts from imitation learning to self-exploration and evolution within interactive environments, training environments have become essential infrastructure [3, 24, 38]. However, real-world services and systems often have restricted access, while manually constructed sandboxes suffer from high costs and poor scalability. To automatically scale environments for training LLM agents, one line of research focuses on LLM-driven simulation [27, 52, 55, 80]. By leveraging the intrinsic world modeling and reasoning capabilities of LLMs, these systems use LLMs to simulate environmental feedback and state transitions. Another line of research focuses on programmatic environment synthesis, constructing deterministic sandboxes via programs, database backends, or finite-state machines [9, 24, 88, 91, 97, 98, 100]. Frameworks including EnvScaler [88], AWM [100], and AutoForge [9] leverage LLMs to plan and generate sandboxes and tasks comprising executable programs, databases, or tool interfaces, and provide rule-based reward signals for reinforcement learning. In addition, InfiniteWeb [129] expands synthesis to web-based and multimodal contexts. Meanwhile, ARE [3] incorporates asynchronous temporal dynamics to better align simulated environments with reality. Distinct from these works, Agent-World utilizes real MCP server metadata for intelligent environment discovery and modeling. By autonomously building theme-matched databases and executable tools from the web, it achieves deep anchoring within the real-world tool ecosystem. Moreover, by leveraging tool graphs and programmatic synthesis to generate verifiable tasks with progressive difficulty, it establishes a broad and challenging training foundation for agents. 

## **5.2 Agentic Reinforcement Learning** 

Recent agentic reinforcement learning has rapidly expanded from single-tool optimization to long-horizon web agents [125]. Early search-centric systems show that rule-based or verifiable RL can improve autonomous information-seeking behavior [13, 42, 87]. Follow-up work further strengthens tool-use training through improved reward design and policy optimization, including Tool-Star, ToolRL, OTC, and ARPO [20, 21, 76, 99]. In parallel, scalability-oriented research explores asynchronous pipelines and large-scale post-training for long-horizon agents [30, 45], while tree-structured rollouts improve exploration efficiency under high-entropy action spaces [28, 53]. Beyond single-agent settings, recent studies increasingly adopt multi-agent and interactive training paradigms, such as agentic RL with multi-agent distillation and multi-turn, user-interacting RL [14, 46, 120, 130], and this line of work is now extending into multimodal settings [77–79, 128, 131]. Despite this progress, most existing methods still emphasize policy optimization on relatively fixed training distributions. Recent environment-scaling efforts expose agents to broader and more diverse environments [67, 88, 98, 100], yet explicit coupling of diagnosis, targeted environment–task refresh, and continual RL remains limited. Agent-World is designed to fill this gap through a continuous agent RL loop. 

## **6 Conclusion** 

In this paper, we presented **Agent-World** , a self-evolving training arena for general-purpose agents in realistic tool environments. Agent-World unifies two tightly coupled components: **Agentic Environment-Task Discovery** , which mines topic-aligned real-world databases and executable toolsets from large-scale themes and synthesizes verifiable tasks with controllable difficulty; and **Continuous Self-Evolving Agent Training** , which combines multi-environment reinforcement learning with an agentic diagnostic arena to identify capability gaps and drive targeted iterative data expansion. Experiments across 23 challenging benchmarks demonstrate that Agent-World consistently improves performance over strong baselines. Further analyses reveal clear scaling trends with respect to environment diversity, evolution rounds, and task difficulty, suggesting that scalable realistic environments are not only useful data sources, but also critical infrastructure for advancing general agent capabilities. 

18 

## **Contributions** 

**Authors** Guanting Dong[1] _[,]_[∗] , Junting Lu[2] , Junjie Huang[2] _[,]_[∗] , Wanjun Zhong[2] _[,]_[†] , Longxiang Liu[2] , Shijue Huang[2] _[,]_[∗] , Zhenyu Li[2] , Yang Zhao[2] _[,]_[∗] , Xiaoshuai Song[1] , Xiaoxi Li[1] , Jiajie Jin[1] , Yutao Zhu[1] , Hanbin Wang[2] _[,]_[∗] , Fangyu Lei[2] _[,]_[∗] , Qinyu Luo[2] , Mingyang Chen[2] , Zehui Chen[2] , Jiazhan Feng[2] , Ji-Rong Wen[1] , Zhicheng Dou[1] _[,]_[†] 

**Affiliations**[1] Gaoling School of Artificial Intelligence, Renmin University of China,[2] ByteDance Seed 

**Acknowledgment** We greatly thank Yujia Qin[2] and Guang Shi[2] for supporting this work and providing valuable suggestions. We also thank Yifei Chen[1] for valuable discussions. 

- ∗ Work was done during their internship at ByteDance Seed 

- Corresponding Author 

19