# **Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data** 

**Fengxian Dong[1] , Zhi Zheng[1]**[*] **, Xiao Han[2] , Wei Chen[1] , Jingqing Ruan[3]** , **Tong Xu[1]**[*] , **Yong Chen[1]** , **Enhong Chen[1]** 

1University of Science and Technology of China 

2Zhejiang University of Technology, 3Meituan 

{fengxiandong, chenweicw, chenyong1997}@mail.ustc.edu.cn {zhengzhi97, tongxu, cheneh}@ustc.edu.cn hahahenha@gmail.com, ruanjingqing2019@ia.ac.cn 

## **Abstract** 

Automated feature generation extracts informative features from raw tabular data without manual intervention and is crucial for accurate, generalizable machine learning. Traditional methods rely on predefined operator libraries and cannot leverage task semantics, limiting their ability to produce diverse, high-value features for complex tasks. Recent Large Language Model (LLM)-based approaches introduce richer semantic signals, but still suffer from a restricted feature space due to fixed generation patterns and from the absence of feedback from the learning objective. To address these challenges, we propose a MemoryAugmented LLM-based Multi-Agent System ( **MALMAS** ) for automated feature generation. MALMAS decomposes the generation process into agents with distinct responsibilities, and a Router Agent activates an appropriate subset of agents per iteration, further broadening exploration of the feature space. We further integrate a memory module comprising procedural memory, feedback memory, and conceptual memory, enabling iterative refinement that adaptively guides subsequent feature generation and improves feature quality and diversity. Extensive experiments on multiple public datasets against state-of-the-art baselines demonstrate the effectiveness of our approach. The code is available at https://github.com/fxdong24/MALMAS 

## **1 Introduction** 

Recently, the advancement of Automated Machine Learning ( **AutoML** ) has greatly improved the efficiency of data modeling (Trirat et al., 2025; Guo et al., 2024; Xu et al., 2024; Jeong et al., 2025; Wei et al., 2024). Within this paradigm, _automated feature generation_ , which extracts informative features from raw data without manual intervention, has become a key enabler for building accurate and generalizable models. 

> *Corresponding authors. 

**==> picture [219 x 142] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Restricted feature space fusion (c)<br>feature<br>non-semantic<br>new  multi-agent<br>feature features feedback<br>fixed operation sets<br>expanded generation space<br>(b) Lack of task feedback feature1<br>thinkingrigid  featuresnew  feature2feature3<br>Z<br>feature featuresnew  summaryagent<br>single agent<br>no feedback abstract summary<br>**----- End of picture text -----**<br>


Figure 1: Comparison of traditional, LLM-based, and multi-agent feature generation approaches. 

However, traditional automated feature generation methods still suffer from several limitations, which hinder their ability to produce high-quality features effectively. As illustrated in Figure 1(a), these methods apply a predefined set of operators to original features to construct new feature sets (Horn et al., 2019; Kanter and Veeramachaneni, 2015; Zhang et al., 2023). They rely on limited operator sets and do not incorporate feature semantics, which confines the transformation search space to a narrow region. 

Recently, large language models (LLMs) have shown strong semantic understanding and generation capabilities (Matarazzo and Torlone, 2025), motivating LLM-based feature generation that leverages task descriptions to propose transformations (Hollmann et al., 2023; Nam et al., 2024). However, these methods typically rely on static, singular generation strategies rooted in rigid thinking, which still constrain the exploration of broader feature spaces. More critically, these methods lack mechanisms to adapt generation strategies based on historical experience or task-specific feedback. Without such adaptive signals, feature generation becomes disconnected from learning performance, leading to inefficient, trial-and-error exploration and limited ability to prioritize high-value transfor- 

1 

mations. This feedback-insensitive process undermines learning-goal alignment and hampers methods’ performance, as illustrated in Figure 1(b). 

To this end, we propose the Memory-Augmented LLM-based Multi-Agent System ( **MALMAS** ), an automated feature generation framework, as illustrated in Figure 1(c). 

Specifically, we decompose feature generation into multiple independent agents with roles grounded in a principled framework along three largely orthogonal dimensions from established feature engineering practice: transformation complexity, data scope, and data-type dependency. Inspired by the categorization of high-value “golden features” (Zhang et al., 2024), this design assigns each agent a clear, specialized responsibility. A Router Agent dynamically selects a subset of agents from a predefined pool based on task metadata and accumulated memory, enabling adaptive allocation of generation effort. Each agent then conducts multi-turn interactions with the LLM, constructing role-specific prompts conditioned on the current feature set and experiential feedback. By exploring complementary regions of the feature space, the agents mitigate feature homogenization in static single-agent strategies (Hollmann et al., 2023) and reduce functional redundancy, thereby broadening the overall search space. 

To address the lack of feedback-driven adaptability, we equip MALMAS with a multi-level memory that enables credit assignment and strategy updates across rounds. Procedural memory caches executed transformations to suppress redundant exploration, feedback memory attributes validation utility to generated features, and conceptual memory abstracts reusable heuristics from historical traces for longer-horizon adaptation. A Summary Agent aggregates cross-agent feedback and concepts into a global conceptual memory that conditions subsequent routing and prompting. This design turns perround evaluations into persistent learning signals, steering generation toward high-yield, task-relevant transformations. 

The main contributions of this paper are summarized as follows: 

- We propose the first multi-agent framework for automated feature generation, enabling collaborative exploration beyond predefined operators and improving feature diversity. 

- We develop a multi-level memory mechanism that integrates procedural traces, feedback, and 

conceptual abstractions, allowing agents to iteratively refine their strategies. 

- We evaluate MALMAS on 16 classification and 7 regression datasets, where it outperforms baselines, and we provide practical analyses demonstrating real-world applicability and efficiency. 

## **2 Related Work** 

## **2.1 Traditional and LLM-enhanced AutoML** 

Before the emergence of LLMs, end-to-end AutoML systems such as Auto-WEKA, Auto-sklearn, H2O AutoML, Google AutoML Tables, and FLAML (Thornton et al., 2013; Feurer et al., 2015; Olson and Moore, 2016; LeDell and Poirier, 2020; Wang et al., 2021; Feurer et al., 2022) mainly focused on pipeline search, hyperparameter optimization, and model selection. 

With the advent of LLMs, AutoML has increasingly adopted natural language as an interface for automation. Methods such as Text-to-ML, LLMSelect, DS-Agent, and GL-Agent (Trirat et al., 2025; Guo et al., 2024; Xu et al., 2024; Jeong et al., 2025; Wei et al., 2024) employ LLMs to generate or recommend ML pipelines, leveraging instruction following and agentic workflows to reduce manual pipeline design. 

However, most of these systems emphasize model and pipeline configuration, while feature construction remains limited to basic preprocessing operations (Gu et al., 2024). This indicates a gap between LLM-enhanced AutoML pipelines and domain-aware feature engineering. 

## **2.2 Automated Feature Generation** 

Feature generation is a long-standing and critical step for improving model performance. Traditional methods such as autofeat (Horn et al., 2019), Deep Feature Synthesis (DFS) (Kanter and Veeramachaneni, 2015), and OpenFE (Zhang et al., 2023) apply symbolic transformations over predefined operator sets. DFS, implemented in Featuretools, demonstrates the practicality of compositional operators by automatically generating features from relational data. While these methods are efficient and interpretable, they are constrained by fixed operator libraries and limited adaptation to task-specific semantics. More recently, LLMbased methods enable semantically driven generation (Hollmann et al., 2023; Nam et al., 2024; Abhyankar et al., 2025). CAAFE (Hollmann et al., 2023) uses task descriptions to better align gen- 

2 

erated features with downstream objectives, and OCTree combines LLMs with tree-based reasoning to support feature validation and interpretability. 

## **2.3 Multi-Agent Systems** 

LLM-based multi-agent systems have emerged as a promising paradigm for collaboration, specialization, and iterative reasoning (Wang et al., 2024; Li et al., 2024a). They have been applied to social simulation (e.g., Generative Agents and AgentSociety (Park et al., 2023; Piao et al., 2025)), software development (e.g., AutoGen and CodeAct (Wu et al., 2024; Hong et al., 2024)), and decisionmaking via multi-agent debate with sparse communication (Liu et al., 2024; Li et al., 2024b; Liang et al., 2024; Li et al., 2026). Recent methods such as ReAct and Reflexion further highlight the role of memory and feedback-driven reasoning-action loops for continual improvement (Yao et al., 2023; Shinn et al., 2023). Moreover, memory has been shown to be crucial for retaining useful experience and improving long-horizon decision-making (Liu et al., 2026; Xu et al., 2026). Despite these advances, multi-agent systems for automated feature generation remain underexplored. 

## **3 Problem Formulation** 

Given a labeled tabular dataset D = ( _X, y_ ), where _X ∈_ R _[m][×][n]_ is the feature matrix with _m_ instances and _n_ features, and _y ∈_ R _[m]_ is the corresponding label vector. The goal of feature generation is to find a transformation function _T_ : _X → X_[�] , where _X_[�] = _X ∪ T_ ( _X_ ), that improves the predictive performance of a model _f_ when trained on the enhanced feature space. Formally, the objective is to maximize the validation performance of _F_ : 

**==> picture [211 x 17] intentionally omitted <==**

where _E_ is the evaluation metric, and ( _X_ val _, Y_ val) is the validation set from cross-validation. Here, _T[∗]_ denotes the optimal transformation function that produces feature beneficial for model performance. 

## **4 Methodology** 

Feature engineering for tabular data requires diverse, context-aware transformations, yet most automated methods rely on a single strategy or weakly coupled modules, limiting broad, task-relevant exploration. We propose **MALMAS** , which coordinates specialized agents to generate and refine features. With role-specific agents and shared pro- 

cedural, feedback, and conceptual memories, MALMAS enables iterative exploration and underpins the pipeline in Figure 2. 

## **4.1 Multi-agent Structure** 

To address the limited ability of a single generator to deeply explore novel features, MALMAS maintains a pool of specialized agents and employs a Router Agent to activate an appropriate subset per iteration. This design increases the diversity and adaptability of generated features while avoiding unnecessary exploration by inapplicable strategies. 

## **4.1.1 Parallel Generation Architecture** 

MALMAS maintains an agent pool _A_ = _{Ai}[K] i_ =1[,] where each agent implements a distinct feature transformation strategy. At iteration _r_ , a Router Agent selects an active subset _A_[(] _[r]_[)] _⊆A_ , and only the selected agents run in parallel to explore complementary feature interactions, transformations, and compositions. Over multiple rounds, this design adapts to diverse feature types and modeling needs through heterogeneous strategies. The overall process is formulated as: 

**==> picture [191 x 30] intentionally omitted <==**

Here, _T_[(] _[r]_[)] denotes the aggregated set of features generated in the _r_ -th round, and _Ti_[(] _[r]_[)] represents the subset produced by agent _Ai_ when activated. 

In each round, each active agent _Ai ∈A_[(] _[r]_[)] independently generates a subset of new features _Ti_[(] _[r]_[)] from the current dataset _X_ by applying its designated strategy. Taking the union of the activated agents’ outputs yields an enriched and more comprehensive feature space for model training. 

## **4.1.2 Agent Responsibilities** 

To systematically explore the vast feature space, MALMAS adopts a principled multi-agent framework that decomposes feature generation along three largely orthogonal dimensions from feature engineering practice: transformation complexity, data scope, and data-type dependency. Inspired by the categorization of high-value “golden features” (Zhang et al., 2024), this design encourages agents to explore complementary aspects of the data, increasing feature diversity while reducing functional redundancy. 

Each agent _Ai_ applies a distinct transformation strategy _fi_ ( _·_ ) to the dataset _X_ , producing a feature 

3 

**==> picture [428 x 206] intentionally omitted <==**

Figure 2: Overview of the proposed MALMAS framework. (A) Agents construct prompts from metadata, selected features, and memory. (B) Agents generate and evaluate candidate features with a downstream model. (C) Agent memories are summarized into a global memory to guide subsequent rounds. 

## subset _Ti_ aligned with its objective: 

**==> picture [138 x 12] intentionally omitted <==**

We instantiate a fixed pool of strategy agents as follows, from which the Router Agent activates a subset at each iteration: 

- **Unary-Feature Agent** . Applies unary transformations _f_ unary( _X_ ) to individual features to generate basic but informative variants. 

- **Cross-Compositional Agent** . Combines multiple inputs _f_ compositional( _X_ ) to capture higherorder interactions. 

- **Temporal-Feature Agent** . Extracts temporal patterns _f_ temporal( _X_ ) for time-series data. 

- **Aggregation-Construct Agent** . Generates group-level summary features _f_ aggregation( _X_ ). 

- **Local-Transform Agent** . Applies regionspecific transformations _f_ local-transform( _X_ ) to capture locally informative patterns. 

- **Local-Pattern Agent** . Discovers latent patterns within feature subsets via clustering or local interaction modeling _f_ local-pattern( _X_ ). 

## **4.2 Memory Architecture and Management** 

Feature generation in MALMAS is formulated as an iterative search over transformations, where learning signals from downstream evaluation are _persisted_ and _reused_ to refine future generation. As illustrated in Figure 2(C), each agent maintains a structured memory state that supports cross-round _credit assignment_ and _strategy refinement_ , thereby 

turning expensive feedback into reusable guidance. 

Formally, at iteration _r_ , each agent _Ai_ maintains an explicit, structured memory state _M_[(] _i[r]_[)] = _{_ ProcMem[(] _i[r]_[)] _[,]_[ FeedMem][(] _i[r]_[)] _[,]_[ ConMem][(] _i[r]_[)] _[}]_[.] At the beginning of round _r_ , the agent retrieves its local memories together with the shared GlobalMem[(] _[r][−]_[1)] to condition prompt construction; after feature evaluation, it appends new traces and utilities to update _M_[(] _i[r]_[)][.][Intuitively,][procedural] memory captures _what was tried_ , feedback memory captures _what worked_ , and conceptual memory captures _why it worked_ in a compact form. 

## **4.2.1 Procedural Memory** 

Procedural memory serves as an _execution trace_ that records the concrete transformation actions performed by agent _Ai_ , enabling reproducibility and constraining redundant exploration. In iteration _r_ , after generating _n_[(] _i[r]_[)] features: 

ProcMem[(] _i[r]_[)] = _{_ ( _bj, tj, fj, dj, r_ ) _}nj_ =1[(] _i[r]_[)] _[,]_ (4) where _bj_ denotes the base columns, _tj_ the transformation type, _fj_ the generated feature name, _dj_ the transformation description, and _r_ the iteration index. During subsequent rounds, ProcMem _i_ is used to avoid duplicate transformations and to discourage patterns that repeatedly fail under evaluation. 

## **4.2.2 Feedback Memory** 

Feedback memory provides a _utility signal_ by associating each generated feature with its downstream 

4 

**Algorithm 1** Iterative Feature Generation 

|**Input**: _X_(1),_y_, metadata_M_, rounds_R_,|**Input**: _X_(1),_y_, metadata_M_, rounds_R_,||||||
|---|---|---|---|---|---|---|
|agent pool_A_=_{Ai}K_<br>_i_=1, metric_E_|||||||
|**Output**: _X_(_R_+1)|||||||
|1: Initialize memories for all agents_Ai ∈A_|||||||
|2: **for**_r_ = 1_,_2_, . . . , R_**do**|||||||
|3:|_A_(_r_) _←_Route(_M,_GlobalMem(_r−_1))||||||
|4:|**for**each active agent_Ai ∈A_(_r_) **do**||||||
|5:|_p_mem _←{_FeedMem(_r−_1)<br>_i_<br>_,_ConMem(_r−_1)<br>_i_|||_,_|||
|6:|GlobalMem(_r−_1)_}_||||||
|7:|_p ←_ConstructPrompt(_M, p_mem)||||||
|8:|_T_ (_r_)<br>_i_<br>_←πθ_(_p, X_(_r_)){Generate_f_ (_r_)<br>_i_||}||||
|9:|gains_←E_((_T_ (_r_)<br>_i_<br>_∪X_(_r_))_, y_)||||||
|10:|Update(ProcMem(_r_)<br>_i_<br>_,_FeedMem(_r_)<br>_i_|_,_ConMem(_r_)<br>_i_|||)||
|11:|**end for**||||||
|12:|Mem(_r_) _←_�<br>_Ai∈A_(_r_)<br>�<br>ConMem(_r_)<br>_i_<br>_∪_FeedMem(_r_)<br>_i_|||||�|
|13:|GlobalMem(_r_) _←_Summary(Mem(_r_))||||||
|14:|_T_ (_r_) _←_�<br>_Ai∈A_(_r_)_T_(_r_)<br>_i_||||||
|15:|_F_ (_r_) _←_�<br>_Ai∈A_(_r_)FeedMem(_r_)<br>_i_||||||
|16:|_S_(_r_) _←_TopN-Features(_T_ (_r_)_, F_ (_r_))||||||
|17:|_X_(_r_+1) _←X_(_r_) _∪S_(_r_)||||||
|18:|_M ←_UpdateMetadata(_M, S_(_r_))||||||
|19: **end for**|||||||



20: **return** _X_[(] _[R]_[+1)] 

validation outcome, enabling explicit credit assignment for feature transformations. For agent _Ai_ in iteration _r_ with _n_[(] _i[r]_[)] generated features: 

FeedMem[(] _i[r]_[)] = _{_ ( _fj, m, vj, ej, r_ ) _}nj_ =1[(] _i[r]_[)] _[,]_ (5) 

where _fj_ is the feature name, _m_ is the evaluation metric, _vj_ is the metric value, _ej_ indicates whether the feature is effective, and _r_ is the iteration index. This memory enables _utility attribution_ by linking each feature to validation gain, which biases later rounds toward high-yield transformations and away from noisy or low-impact candidates. 

## **4.2.3 Conceptual Memory** 

Conceptual memory stores a compact set of reusable heuristics distilled from an agent’s historical traces and utilities. After each round, the LLM summarizes ProcMem[(] _[r]_[)] and FeedMem[(] _[r]_[)] _i i_ into rules that guide subsequent generation: 

ConMem[(] _i[r]_[)] = LLM ProcMem[(] _i[r]_[)] _,_ FeedMem[(] _i[r]_[)] _._ (6) � � By compressing experience into high-level guidance, ConMem _i_ supports strategy adaptation across rounds keeping the prompt context concise. 

## **4.2.4 Global Conceptual Memory** 

To promote coordination and knowledge transfer across agents, after each iteration the SummaryAgent aggregates agents’ local conceptual and feed- 

back memories into a Global Conceptual Memory. This cross-agent consolidation forms a shared prior for the next round, propagating effective transformation heuristics across roles, reducing overlap among agents, and improving the efficiency of subsequent exploration and refinement. 

## **4.3 Iterative Feature Generation** 

This section describes the feature generation mechanism of our multi-agent system, as shown in Figure 2. Across iterative rounds, agents leverage local and global memories to refine their strategies. 

In each iteration _r_ , each active agent _Ai ∈A_[(] _[r]_[)] independently executes a fixed sequence of steps, as detailed in Algorithm 1: 

- **Prompt Construction** : Each agent constructs a prompt from statistics and metadata _M_ , effective features from FeedMem[(] _i[r][−]_[1)] , and distilled guidance from ConMem[(] _[r][−]_[1)] and GlobalMem[(] _[r][−]_[1)] . _i_ 

- **Feature Generation and Evaluation** : Conditioned on the prompt, the agent uses _πθ_ to propose a transformation, instantiates it as _fi_[(] _[r]_[)] , evaluates the resulting features under _E_ , and stores the feedback in FeedMem[(] _[r]_[)][[.]] 

   - _i_[[.]] 

- **Memory Update** : To guide the next round, the agent updates ProcMem[(] _i[r]_[)] and ConMem[(] _i[r]_[)] with attempted operations, effective transformations, and newly identified patterns. 

At the end of iteration _r_ , the system selects top-performing features generated by the activated agents and integrates them into the dataset. Specifically, each active agent _Ai ∈A_[(] _[r]_[)] applies TopN-Features to _Ti_[(] _[r]_[)] to retain the highestranked features under _E_ , and the selected features are aggregated to expand the global dataset. This iterative selection-and-aggregation procedure accumulates high-quality transformations and yields progressive improvements in model performance. 

## **5 Experiments** 

## **5.1 Experimental Setup** 

## **5.1.1 Datasets** 

Following prior work (Hollmann et al., 2023; Abhyankar et al., 2025), we evaluated our method on 16 classification and 7 regression datasets sourced from Kaggle and UCI. Following (Nam et al., 2024), we used a 6-4 train–test split and repeated each experiment three times with different seeds. 

5 

Table 1: Performance (AUC) of all methods on 16 classification datasets. Best results are in bold, second-best are underlined. Results are averaged across three random train-test splits using the XGBoost classifier. “N/A” indicates that the running time exceeded 12 hours. 

|Datasets|Base|Traditional Methods<br>DFS<br>AutoFeat<br>OpenFE|Traditional Methods<br>DFS<br>AutoFeat<br>OpenFE|LLM-based Methods<br>CAAFE<br>OCTree<br>LLMFE|LLM-based Methods<br>CAAFE<br>OCTree<br>LLMFE|MALMAS|
|---|---|---|---|---|---|---|
|||DFS<br>AutoFeat||CAAFE<br>OCTree|||
|Adult<br>Balance<br>Bank<br>Banknote<br>Breast_W<br>Car_Eval<br>Cdc<br>Credit_G<br>Heart<br>Jungle<br>Myocardial<br>Pima<br>Student<br>Churn<br>Titanic<br>Wine|0.849_±_0.009<br>0.908_±_0.009<br>0.869_±_0.007<br>0.995_±_0.002<br>0.989_±_0.002<br>0.982_±_0.004<br>0.863_±_0.001<br>0.756_±_0.004<br>0.915_±_0.001<br>0.970_±_0.000<br>0.802_±_0.003<br>0.809_±_0.007<br>0.978_±_0.001<br>0.829_±_0.002<br>0.843_±_0.007<br>0.878_±_0.001|0.857_±_0.001<br>0.849_±_0.009<br>0.989_±_0.009<br>0.908_±_0.009<br>0.891_±_0.014<br>0.869_±_0.007<br>0.998_±_0.001<br>0.994_±_0.003<br>0.988_±_0.003<br>0.989_±_0.002<br>0.978_±_0.002<br>0.982_±_0.004<br>0.863_±_0.002<br>N/A<br>0.758_±_0.011<br>0.756_±_0.004<br>0.916_±_0.008<br>0.914_±_0.001<br>0.975_±_0.000<br>0.970_±_0.000<br>0.800_±_0.010<br>N/A<br>0.810_±_0.003<br>0.805_±_0.006<br>0.977_±_0.000<br>0.978_±_0.001<br>0.833_±_0.003<br>0.829_±_0.002<br>0.839_±_0.005<br>0.843_±_0.007<br>0.885_±_0.005<br>0.879_±_0.002|0.849_±_0.009<br>0.908_±_0.009<br>**0.904**_±_**0.003**<br>0.998_±_0.001<br>0.988_±_0.003<br>0.994_±_0.004<br>0.864_±_0.001<br>0.755_±_0.011<br>0.920_±_0.008<br>0.980_±_0.000<br>0.803_±_0.002<br>0.815_±_0.008<br>0.983_±_0.000<br>0.828_±_0.001<br>0.816_±_0.005<br>**0.891**_±_**0.007**|0.868_±_0.005<br>0.845_±_0.011<br>**1.000**_±_**0.000**<br>0.933_±_0.031<br>0.874_±_0.022<br>0.873_±_0.010<br>0.993_±_0.002<br>0.993_±_0.004<br>**0.992**_±_**0.001**<br>0.986_±_0.003<br>0.988_±_0.006<br>0.975_±_0.007<br>0.866_±_0.002<br>0.865_±_0.007<br>0.751_±_0.008<br>0.754_±_0.011<br>0.909_±_0.002<br>0.912_±_0.007<br>0.983_±_0.005<br>0.972_±_0.003<br>0.805_±_0.003<br>0.803_±_0.003<br>0.810_±_0.007<br>0.810_±_0.005<br>0.979_±_0.001<br>0.978_±_0.001<br>0.827_±_0.003<br>0.825_±_0.002<br>0.843_±_0.006<br>0.847_±_0.004<br>0.878_±_0.001<br>0.869_±_0.006|0.853_±_0.011<br>0.994_±_0.008<br>0.875_±_0.012<br>0.991_±_0.002<br>**0.992**_±_**0.002**<br>0.986_±_0.004<br>0.864_±_0.001<br>0.748_±_0.018<br>0.911_±_0.004<br>0.981_±_0.006<br>0.805_±_0.002<br>0.810_±_0.008<br>0.978_±_0.001<br>0.829_±_0.001<br>0.849_±_0.004<br>0.879_±_0.003|**0.875**_±_**0.010**<br>**1.000**_±_**0.000**<br>0.895_±_0.002<br>**0.999**_±_**0.001**<br>**0.992**_±_**0.001**<br>**0.999**_±_**0.000**<br>**0.867**_±_**0.001**<br>**0.775**_±_**0.002**<br>**0.923**_±_**0.001**<br>**0.993**_±_**0.000**<br>**0.809**_±_**0.003**<br>**0.823**_±_**0.003**<br>**0.984**_±_**0.000**<br>**0.835**_±_**0.001**<br>**0.872**_±_**0.008**<br>0.886_±_0.003|
|**MeanRank**|4.37|3.69<br>4.75|3.12|3.57<br>4.81|3.75|**1.12**|



Table 2: Performance (NRMSE) of all methods on 7 regression datasets. Best results are in bold, second-best are underlined (Lower is better). Results are averaged across three random train-test splits using XGBoost regressor. 

|Datasets|Base|Traditional Methods<br>DFS<br>AutoFeat<br>OpenFE|Traditional Methods<br>DFS<br>AutoFeat<br>OpenFE|Traditional Methods<br>DFS<br>AutoFeat<br>OpenFE|LLM-based Method<br>LLMFE|MALMAS|
|---|---|---|---|---|---|---|
|||DFS|AutoFeat||||
|Airfoil<br>Bike<br>Crab<br>Insurance<br>House<br>Energy<br>Medical|0.015_±_0.001<br>0.230_±_0.001<br>0.220_±_0.003<br>0.367_±_0.006<br>0.173_±_0.005<br>0.060_±_0.002<br>0.368_±_0.003|0.016_±_0.001<br>0.225_±_0.003<br>0.217_±_0.002<br>0.365_±_0.010<br>0.179_±_0.019<br>**0.046**_±_**0.003**<br>0.373_±_0.002|0.015_±_0.001<br>0.230_±_0.001<br>0.220_±_0.003<br>0.367_±_0.006<br>0.173_±_0.005<br>0.060_±_0.002<br>0.368_±_0.003|0.014_±_0.000<br>**0.213**_±_**0.002**<br>0.214_±_0.002<br>0.381_±_0.006<br>0.160_±_0.001<br>0.054_±_0.003<br>0.377_±_0.001|0.015_±_0.001<br>0.225_±_0.003<br>0.218_±_0.002<br>0.358_±_0.007<br>0.165_±_0.005<br>0.058_±_0.005<br>0.370_±_0.006|**0.013**_±_**0.000**<br>0.215_±_0.001<br>**0.213**_±_**0.002**<br>**0.355**_±_**0.002**<br>**0.155**_±_**0.004**<br>0.050_±_0.007<br>**0.355**_±_**0.003**|
|**MeanRank**|3.86|3.29|3.86|2.86|3.14|**1.29**|



## **5.1.2 Baselines** 

We compared MALMAS against a range of automated feature engineering baselines, including traditional methods such as AutoFeat (Horn et al., 2019), OpenFE (Zhang et al., 2023), and DFS (Kanter and Veeramachaneni, 2015), and LLM-based approaches such as CAAFE (Hollmann et al., 2023), OCTree (Nam et al., 2024), and LLMFE (Abhyankar et al., 2025). The configurations of all baseline methods are detailed in Appendix B.1. 

## **5.1.3 Evaluation Metrics** 

For classification tasks, we adopted the area under the AUC as the primary evaluation metric, and additionally reported accuracy (ACC) as a complementary measure, as shown in Appendix C.1. For 

regression tasks, we used the normalized root mean squared error (NRMSE) as the evaluation metric. Following prior work (Abhyankar et al., 2025), we also adopted mean rank as a global indicator to compare the overall effectiveness. 

## **5.1.4 MALMAS Configuration** 

Across all experiments, MALMAS uses a fixed multi-agent configuration with _R_ =4 iterative rounds, where agents generate candidate features, evaluate them. All methods are evaluated with the same downstream model, XGBoost (Chen and Guestrin, 2016). LLM-based results in the main text use DeepSeekV3; additional details are deferred to Appendix B.2. 

6 

**==> picture [182 x 133] intentionally omitted <==**

**----- Start of picture text -----**<br>
6<br>5.11<br>5<br>4.50<br>4 3.69 3.88<br>3.40<br>3 2.69<br>2.30<br>2<br>1.12<br>1<br>Base +A1 +A2 +A3 +A4 +A5 +A6 Full<br>Mean Rank<br>**----- End of picture text -----**<br>


Figure 3: Mean rank (lower is better) across different ablation configurations of MALMAS. 

**==> picture [183 x 134] intentionally omitted <==**

**----- Start of picture text -----**<br>
With Memory<br>0.880 Without Memory<br>0.870<br>0.860<br>0.850<br>0.840<br>0 1 2 3 4<br>Rounds<br>AUC<br>**----- End of picture text -----**<br>


Figure 4: AUC performance on the Adult dataset across different rounds with and without memory. 

## **5.2 Overall Performance** 

Table 1 summarizes the AUC performance of all evaluated methods across 16 classification datasets. Overall, MALMAS achieves the highest average AUC, consistently outperforming both traditional feature engineering methods and recent LLMbased approaches. MALMAS consistently improves upon the base model, ranking first or second on most benchmark datasets and exhibiting strong generalization across diverse real-world domains. Although LLM-based methods such as OCTree and LLMFE benefit from semantic-aware transformations, they still underperform compared to MALMAS in terms of overall average AUC. These results clearly and collectively underscore the effectiveness of memory-enhanced multi-agent collaboration in facilitating high-quality feature discovery. 

As shown in Table 2, MALMAS also achieves the lowest mean NRMSE on almost regression tasks, indicating its strong and reliable feature generation capability beyond classification. Although some LLM-based baselines such as CAAFE and OCTree do not support regression, MALMAS still outperforms LLMFE by a large margin, confirming its advantage in continuous-value prediction. 

## **5.3 Ablation Study** 

To gain deeper insights into the contributions of the multi-agent and memory modules, we further analyzed the results in Figure 3. From “Base” to “+A6,” the mean rank decreases from 5.11 to 2.30, indicating that expanding the agent pool broadens the feature search space and improves diversity, which in turn enhances downstream performance. 

Beyond this, the “Full” configuration—which incorporates the memory module on top of all six agents—achieves a dramatic mean rank reduction to 1.12. This demonstrates the role of memory in 

accumulating cross-round information and refining feature-generation strategies. Specifically, procedural memory records attempted features to reduce redundancy, feedback memory stores downstream performance to guide the next round of exploration, and conceptual memory abstracts cross-round patterns summarized by the Summary Agent into a global conceptual memory shared across agents. 

We observe a slight non-monotonicity from +A2 to +A3. This can plausibly occur because adding agents expands the candidate pool but may introduce higher-variance transformations that, under a fixed top- _N_ budget, occasionally replace more robust features; mean-rank aggregation is also sensitive to small dataset-level fluctuations. 

## **5.4 Parameter Sensitivity** 

The number of generation rounds _R_ is a key parameter in MALMAS, governing iterative feature refinement. Memory facilitates this process by guiding feature reuse, evaluation, and abstraction. Figure 4 reports the AUC on the Adult dataset across rounds. With memory enabled, AUC increases from 0.85 to nearly 0.88 as _R_ grows, indicating that iterative generation can leverage accumulated feedback to uncover richer feature interactions. In contrast, without memory, performance plateaus after the first two rounds and slightly drops at round three, suggesting less directed exploration and limited gains in later rounds. Moreover, improvements diminish and plateau around rounds three to four, implying that MALMAS reaches a sufficiently rich feature set; dynamic scheduling could improve efficiency by using more agents/rounds early for exploration and focusing on high-value features later. 

7 

Table 3: Classification performance (AUC) of H2O and DS-Agent on benchmark tabular datasets. “w/o” indicates training with original features, while “w/” indicates training with our derived features. Results are reported as mean _±_ standard deviation over three runs. 

|Datasets|H2O<br>w/o<br>w/|DS-Agent<br>w/o<br>w/|
|---|---|---|
|Adult<br>Bank<br>Breast_W<br>Churn<br>Titanic|0.876_±_0.003<br>**0.881**_±_**0.001**<br>0.864_±_0.010<br>**0.899**_±_**0.029**<br>0.989_±_0.002<br>**0.992**_±_**0.003**<br>0.844_±_0.003<br>**0.846**_±_**0.001**<br>0.859_±_0.003<br>**0.869**_±_**0.001**|0.871_±_0.002<br>**0.880**_±_**0.001**<br>0.866_±_0.012<br>**0.892**_±_**0.021**<br>0.985_±_0.004<br>**0.990**_±_**0.002**<br>**0.846**_±_**0.003**<br>0.845_±_0.004<br>0.855_±_0.001<br>**0.866**_±_**0.003**|



## **5.5 Integration with Classical AutoML** 

To further validate the effectiveness of our derived features in an end-to-end setting, we integrate them into classical AutoML pipelines based on H2O AutoML and DS-Agent (LeDell and Poirier, 2020; Guo et al., 2024). Table 3 reports the AUC performance of both methods on multiple tabular benchmark datasets. For each method, “w/o” uses the original features only, whereas “w/” augments them with our derived features. 

In our experiments, H2O AutoML was run with a time budget of 2 hours for each run on each dataset, using five-fold cross-validation and “AUC” as the primary metric for model selection. For the DS-Agent pipeline, we used the same data splits and metric, and adopted DeepSeek-V3 as the LLM backbone to generate derived features. 

Across all datasets, incorporating our derived features consistently improves the performance of both H2O AutoML and DS-Agent. This demonstrates that our feature derivation method integrates well with established end-to-end AutoML frameworks, yielding robust and reproducible gains. 

## **5.6 Discussion on Feature Generalization** 

To assess whether MALMAS-generated features generalize beyond a single downstream model, we evaluated them across multiple classifiers, including XGBoost, LightGBM, Random Forest, and MLP (Chen and Guestrin, 2016; Ke et al., 2017; Liu et al., 2012). Table 10 shows that MALMAS consistently achieves the highest AUC, with a mean rank of 1.00 (vs. 2.40 for the next-best method), indicating stable performance across diverse learning architectures and suggesting that the generated features capture broadly informative patterns rather than being tuned to a specific model. Compared with conventional approaches that may favor a par- 

ticular algorithm, MALMAS remains effective under different decision boundaries and inductive biases; for example, both tree-based and neural models benefit from the enriched feature space. This cross-model consistency provides a reliable foundation in pipelines where model choice may vary across deployment settings. By mitigating dependence on any single learner, MALMAS can reduce repeated feature engineering and yield a more portable feature base for classification tasks. 

## **5.7 Computational and Token Cost** 

To assess the feasibility of our method, we measured the average runtime and token usage of MALMAS on 16 classification datasets using the DeepSeek-V3 API as the LLM backbone. On average, each dataset required 0.452 hours of computation and 147.57k tokens for feature generation, with an estimated cost of $0.17, as detailed in Appendix C.8. These results indicate that MALMAS incurs modest overhead and can be readily embedded within existing AutoML pipelines. 

## **6 Conclusion** 

We propose **MALMAS** , a memory-augmented multi-agent framework for automated feature generation, and validate its effectiveness through extensive experiments. By assigning distinct roles to specialized agents, MALMAS enables parallel and diverse exploration of the feature space, addressing the limitations of single-strategy approaches. Its memory module allows agents to retain useful signals and improve generation strategies across iterations. Together, these components provide a scalable and interpretable approach for producing high-quality, task-relevant features. We further present practical analyses demonstrating real-world applicability and efficiency. 

8 

## **7 Limitations** 

MALMAS is designed for labeled tabular datasets and relies on downstream evaluation signals; its effectiveness may degrade when labels are scarce or evaluation budgets are limited. While our framework targets tabular feature engineering, its applicability to other modalities or structured domains remains unexplored. Moreover, as the candidate feature pool grows, repeated downstream training and validation can become a computational bottleneck, making performance sensitive to the available evaluation budget. Finally, although MALMAS provides transformation descriptions and memory traces, the overall LLM-driven generation process does not guarantee full interpretability of every derived feature. 

## **8 Ethical Considerations** 

MALMAS is an automated feature generation framework for tabular data. Its outputs may inherit or amplify biases present in the input data, and the downstream evaluation signal may inadvertently favor transformations that correlate with sensitive attributes when such attributes are present or can be proxied. In addition, because MALMAS generates transformation programs, it may propose invalid or data-leaking features if the schema is ambiguous or the data pipeline is misconfigured. To mitigate these risks, we recommend applying strict schema constraints (e.g., explicitly marking protected attributes and leakage-prone fields), enforcing execution-time validation and leakage checks, and conducting fairness and privacy audits when deploying MALMAS in high-stakes settings. Finally, our experiments use publicly available datasets and do not involve human subjects.