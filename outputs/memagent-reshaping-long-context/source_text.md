**==> picture [152 x 18] intentionally omitted <==**

**==> picture [197 x 24] intentionally omitted <==**

# **MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent** 

**Hongli Yu**[1] _[,]_[2] _[,]_[3] , **Tinghong Chen**[2] , **Jiangtao Feng**[2] , **Jiangjie Chen**[1] _[,]_[3] , **Weinan Dai**[1] _[,]_[2] _[,]_[3] , **Qiying Yu**[1] _[,]_[2] _[,]_[3] , **Ya-Qin Zhang**[2] _[,]_[3] , **Wei-Ying Ma**[2] _[,]_[3] , **Jingjing Liu**[2] _[,]_[3] , **Mingxuan Wang**[1] _[,]_[3] , **Hao Zhou**[2] _[,]_[3] 

1ByteDance Seed 2Institute for AI Industry Research (AIR), Tsinghua University 3SIA-Lab of Tsinghua AIR and ByteDance Seed 

## **Abstract** 

Despite improvements by length extrapolation, efficient attention and memory modules, handling infinitely long documents with linear complexity without performance degradation during extrapolation remains the ultimate challenge in long-text processing. We directly optimize for long-text tasks in an end-to-end fashion and introduce a novel agent workflow, MemAgent, which reads text in segments and updates the memory using an overwrite strategy. We extend the DAPO algorithm to facilitate training via independent-context multi-conversation generation. MemAgent has demonstrated superb long-context capabilities, being able to extrapolate from an 8K context trained on 32K text to a 3.5M QA task with performance loss < 5% and achieves 95%+ in 512K RULER test. 

## **Date:** July 4, 2025 

**Correspondence:** `zhouhao@air.tsinghua.edu.cn` , `wangmingxuan.89@bytedance.com` **Project Page:** `https://memagent-sialab.github.io/` 

**==> picture [448 x 180] intentionally omitted <==**

**----- Start of picture text -----**<br>
80<br>60<br>40 RL-MemAgent-14B<br>RL-MemAgent-7B<br>QwenLong-L1-32B<br>Qwen2.5-Instruct-14B-1M<br>Qwen2.5-Instruct-7B-1M<br>20<br>DS-Distill-Qwen-32B<br>DS-Distill-Qwen-14B<br>DS-Distill-Qwen-7B<br>Truncation<br>0<br>7K 112K 224K 448K 896K 1.75M 3.5M<br>Context Length in Tokens<br>Score<br>**----- End of picture text -----**<br>


**Figure 1** Accuracy scores of RULER-HotpotQA [1, 2] . Even models that employ long-context continual pretraining and extrapolation techniques fail to maintain consistent performance. In contrast, MemAgent with RL demonstrates nearly lossless performance extrapolation. 

1 

## **1 Introduction** 

While having demonstrated impressive capabilities [3–7], industry-level Large Language Model (LLM) systems [8–10] still face a critical challenge: how to handle long contexts effectively - processing an entire book, executing a complex chain of reasoning over many steps, or managing the long-term memory of an agent system - all these complex tasks can generate overflowing text that quickly explodes the typical-size context window of current LLMs. 

Existing approaches to long-context tasks are three-pronged. The first involves length extrapolation methods by shifting the positional embeddings in order to extend the context window of the model [11–15], plus continued pre-training [16–18]. Despite promising potential, these methods often suffer from performance degradation and slow processing speed due to _O_ ( _n_[2] ) computational complexity when applied to extremely long text. The second school of methods leverages sparse attention [19–21] and linear attention mechanisms [22, 23] to reduce the complexity of attention for more efficient processing of longer sequences. However, this typically requires training from scratch, with inherent adversities such as linear attention facing difficulties in parallel training or sparse attention depending on human-defined patterns. The last line of inquiry investigates context compression [24–27], which aims to condense information in token-level or external-memory-plugin modules. Such approaches often struggle with extrapolation, and require the integration of additional modules or context operations, which ineluctably disrupts the standard generation process and hinders compatibility as well as parallelization. 

Hence, a successful LLM with strong long-context capabilities requires the trinity of: 1) processing infinite length of text; 2) scaling without performance drop; and 3) efficient decoding with linear complexity. To pursue this quest, we return to the basic intuition behind long-context modeling [28–31]. When humans process long-context information, we tend to abstract out the main revealing conceptions to capture the essence of the whole text, often by making notes of critical details or using short-handed stenograph to record the key points, while discarding redundant and irrelevant data. We do not attempt to memorize every single fact or each small piece of information; instead, we focus our intellectual energy on more important aspects of the task at hand. This selective attention not only simplifies the process but also aids in tackling complex problems more efficiently. 

Following this anthropocentric intuition, we propose a novel use of Reinforcement Learning (RL) to equip LLMs with a dynamically updated fixed-length ‘memory’, as illustrated in Figure 2. During inference, the LLM processes the input text segment-by-segment. As it reads each segment, the model proactively and selectively updates the memory, which then contributes to the generation of the final output after all relevant messages are aggregated and synergized in the memory. This clever mechanism allows the LLM to flexibly handle arbitrary text lengths while maintaining a linear time complexity during processing, since the length of the memory is fixed, which leads to a fixed context window size for the model. This segment-based approach generates multiple outputs from a single long-text input, requiring multiple rounds of memory updates and a final round for the generation of the final response. Training this type of agent workflow, which enables dialogues across multiple independent contexts, is still an unexplored territory in current LLM study. Existing systems typically handle workflow trajectories via alternating tool calls or environment feedback by either simply concatenating [32, 33] them or using a sliding window [34] approach, which lacks flexibility and scalability in practice. Our MemAgent approach, instead, proposes that treats each context-independent conversation as an optimization objective. Based on the DAPO[35] algorithm, we implement the Multi-Conv DAPO to optimize an arbitrary agent workflow by verifiable outcome reward. 

In our experiments, an RL-trained model with a modest 8K context window (with a 1024-token memory and a 5000-token document chunk) trained on 32K documents exhibits consistently superb capabilities for Question Answering (QA) tasks on documents of up to 4 million tokens, without performance drop and with linear computation cost. This demonstratively showcases the efficiency and scalability of our long-context memory approach. 

Our major contributions are threefold: 

- We introduce a novel approach that enables LLMs to process arbitrarily long inputs within limited context window under linear time complexity during inference, overcoming a significant bottleneck in 

2 

**==> picture [448 x 181] intentionally omitted <==**

**----- Start of picture text -----**<br>
Solving Long-Context Task with Long-Context LLM A<br>Long-Context LLM<br>1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 … N-2 N-1 N Q Text Chunk<br>Q Question<br>3  Memory<br>1 2 3 A<br>A Answer<br>LLM LLM LLM … LLM<br>1 Q 1 2 Q 2 3 Q K N Q<br>Solving Long-Context Task with Memory Agent via RL<br>**----- End of picture text -----**<br>


**Figure 2** MemAgent is inspired by the way humans process long documents. It divides the document into multiple chunks and allows LLMs to process them iteratively, recording relevant information in memory. Finally, LLMs generate answers based on the information stored in the memory. 

long-context processing. 

- We design an agent workflow to implement this mechanism and propose an end-to-end training approach using the multi-conversation DAPO algorithm. 

- We empirically demonstrate that our RL-trained method allows models to extrapolate to vastly long documents with minimal performance degradation, pushing the boundaries of what is currently achievable in long-context LLM systems. 

## **2 Related Work** 

**Long Context LLMs** Extrapolation methods for RoPE-based LLMs [11], such as NTK [12], PI [13], YaRN [14] and DCA [15], modify the base frequency, position index and other components of positional embeddings, enabling the model to capture long-range semantic dependencies. On the other hand, Linear attention mechanisms [22, 23], Recurrent Neural Networks (RNNs) and State Space Models (SSMs) [36–40] leverage different architectures to achieve _O_ ( _N_ ) computation complexity, aiming to process extremely long context. Sparse attention [19–21] shifts the attention mask matrix to apply patterns such as sliding windows, thereby eliminating irrelevant attention calculations. However, these patterns are typically based on predefined heuristics. The possibility of dynamic sparse attention [41, 42] has been explored recently. The Long ShortTerm Memory (LSTM) mechanism [29] achieved significant success in early NLP tasks, while Neural Turing Machines [30] and Memory Networks [31] demonstrated how to equip neural networks with memory. Existing memory mechanisms integrated to Transformer models are typically realized by adding external memory modules [26, 43–45] or external database [46–48]. In contrast, we use reinforcement learning (RL) to enable LLM itself the ability to memorize. 

**Reinforcement Learning for LLMs** In recent RL studies, the reward signals have gradually shifted from human preferences [49] or reward models distilled from them [50] to rule-based feedback, which has demonstrated great potential in enhancing model reasoning capabilities [3, 4, 51–53]. Key contributions include PPO [54] based on GAE [55], the Actor-Critic framework, as well as GRPO [56] that utilizes Group Normalization. Algorithmic enhancements [35, 57, 58] have mostly focused on improving training sustainability and sample efficiency of these algorithms. To further release the potential of RL, recent works such as Search-R1 [33], Agent-R1 [32] and RAGEN [59] have explored the training of tool-using agents based on multi-turn chat. However, these multi-turn chats are constructed by alternately concatenating tool responses and model replies, with the ultimate optimization goal being a single conversation with tool masking. GiGPO [34] further investigates the use of multiple independent contexts in agent training with environment feedback 

3 

with sliding window trajectories. However, these approaches are limited to optimizing interleaved trajectories of observation and generation, making them difficult to apply to more general agent workflows. 

## **3 The Proposed MemAgent** 

In this section, we describe the details of MemAgent approach for solving long-context tasks, including the overall workflow ( _§_ 3.1), Multi-conv RL algorithm for training MemAgent ( _§_ A), reward modeling design ( _§_ 3.3), and architecture implementation design ( _§_ 3.4). 

## **3.1 The MemAgent Workflow: RL-shaped Memory for Unbounded Contexts** 

As illustrated in Figure 2, MemAgent views an arbitrarily long document not as a monolithic block but as a controlled _stream_ of evidence. At every step, the model sees exactly two things: the next chunk of text and a compact, fixed-length _memory_ that summarizes everything deemed important so far. Crucially, the memory is just a sequence of ordinary tokens inside the context window, so the core generation process of the base LLM remains unchanged. 

After reading a new chunk, the model overwrites the previous memory with an updated one. This **overwrite** strategy seems almost too simple, yet it is precisely what enables the system to scale: because memory length never grows, the total compute per chunk stays _O_ (1) and end-to-end complexity is strictly linear to the number of chunks. We formulate the overwrite decision as a reinforcement learning problem: the agent is rewarded for retaining information that will later prove useful and for discarding distractors that would waste precious tokens. By optimizing this objective with our newly introduced multi-conversation DAPO algorithm (detailed in _§_ A), the model learns to compress aggressively while preserving answer-critical facts. 

The workflow naturally decomposes inference into two modules. Within the **Context-Processing** module the model iterates over chunks, updating memory with a prompt template (Table 1, top). Once the stream is exhausted, a final **Answer-Generation** module is invoked (Table 1, bottom) where the model consults only the problem statement and the memory to produce its boxed answer. Because positional embeddings are never re-scaled or patched, the same tokenization and attention layout apply in both modules, unlocking the model’s latent length-extrapolation capability without any architectural modifications. 

MemAgent therefore enjoys three benefits from this design: (1) **Unlimited length** : the document can be millions of tokens because it is processed as a stream; (2) **No performance cliff** : RL encourages the memory to retain exactly the information needed, yielding near-lossless extrapolation (Figure 1); (3) **Linear cost** : a constant window size implies decoding time and memory consumption grow linearly with input length ( _O_ ( _N_ )) (detailed in _§_ A.) This renders a practical recipe for turning any moderately context-sized LLM into an efficient long-context reasoner with minimal engineering overhead. 

## **3.2 Training MemAgent with Multi-conv RL** 

By viewing memory update in context processing for answer-generation tasks as part of the policy to be optimized by RL, we adopt the RLVR recipe [3, 51, 60] to train MemAgent. 

For the base algorithm, we adopt Group Relative Policy Optimization (GRPO) [56] for its simplicity and effectiveness in RLVR. In the rollout phase of GRPO, the policy model _πθ_ old samples a group of _G_ individual responses _{oi}[G] i_ =1[for][an][input] _[x]_[.][Let] _[{][R][i][}][G] i_ =1[refer][to][the][sequence-level][rewards,][then][the][group][normalizing] advantage of the _i_ -th response is calculated by the following function: 

**==> picture [297 x 26] intentionally omitted <==**

GRPO adopts a clipped objective with a KL penalty term: 

4 

You are presented with a **problem** , a **section** of an article that may contain the answer, and a **previous memory** . Please read the section carefully and update the memory with new information that helps to answer the problem, while retaining all relevant details from the previous memory. 

```
<problem>{prompt}</problem>
```

```
<memory>{memory}</memory>
```

```
<section>{chunk}</section>
```

## **Updated memory:** 

You are presented with a **problem** and a **previous memory** . Please answer the problem based on the previous memory and put the answer in `\boxed {}` . 

```
<problem>{prompt}</problem>
<memory>{memory}</memory>
```

## **Your answer:** 

**Table 1** Template of MemAgent for context processing (top part) and final answer generation (bottom). Curly-brace placeholders {} will be replaced with actual content. 

**==> picture [445 x 50] intentionally omitted <==**

where _ri,t_ ( _θ_ ) refers to the importance sampling weight: 

**==> picture [299 x 24] intentionally omitted <==**

However, due to the nature of the MemAgent approach, it generates multiple context-independent conversations for a single query, as illustrated in Figure 2. Therefore, policy optimization cannot be implemented by simply applying the attention mask as is done in multi-turn tool-calling optimization. 

To address this issue, we treat each conversation as an independent optimization target, as demonstrated in Figure 3. Let _ni_ denote the number of generated conversations ( _oi,_ 1 _, oi,_ 2 _, ..., oi,ni_ ) for a given sample ( _qi, ai_ ) in a group. _oi,j_ further decomposes into token-level outputs ( _oi,j,_ 1 _, oi,j,_ 2 _, ..., oi,j,|oi,j |_ ). We compute an outcome reward _Ri_ per sample by the final conversation that contains the final answer, and distribute group-normalized advantages across all associated conversations. 

Equations 4 and 5 illustrate how the advantage and loss are computed within our MemAgent algorithm for context-independent multi-conversation rollouts. The advantage value is derived from the conversation that contains the final answer, then uniformly applied across all conversations originating from the same sample. Our loss function is analogous to that used in DAPO [35], which incorporates a token-level averaging loss. Furthermore, we extend the dimensionality of the loss computation from the conventional `(group, token)` structure to `(group, conversation, token)` . Following DrGRPO [58], we do not normalize the advantage by the standard deviation of rewards. 

**==> picture [298 x 13] intentionally omitted <==**

5 

**==> picture [424 x 240] intentionally omitted <==**

**----- Start of picture text -----**<br>
KL 𝒥 clip<br>o 1 Reference Model  r 1 A 1<br>q Model Policy  o 2 Rule-Based Verifier  r 2 Normalization Group  A 2<br>… … …<br>GRPO oG rG AG Frozen Model<br>Trainable<br>Model<br>Group of Conversations KL 𝒥 clip Take part in<br>Adv. Compute<br>Context-Independent Conversations<br>o 1,1 o 1,2 … o 1, c 1 Reference Model  r 1 A 1<br>q Model Policy  o 2,1 o 2,2 … o 2, c 2 Rule-Based Verifier  r 2 Normalization Group  A 2<br>… …<br>…<br>Multi-conv  rG AG<br>DAPO oG ,1 oG ,2 … oG , cG<br>**----- End of picture text -----**<br>


**Figure 3** Comparison between vanilla GRPO and Multi-Conv DAPO. During the rollout phase of Multi-conv DAPO, each sample generates multiple conversations. The answer contained in the final conversation is used to compute the reward and advantage, which are then employed to optimize all preceding conversations. 

**==> picture [398 x 73] intentionally omitted <==**

## **3.3 Reward Modeling** 

Following the RLVR recipe [33, 35, 51], we train the model with a final outcome reward computed by a rule-based verifier. In RULER [1] and other datasets, questions may have multiple ground-truth answers. For some tasks, such as question answering, these ground truths are considered equivalent. Given a set of multiple ground-truth answers _Y_ = _{y_ 1 _, y_ 2 _, . . . , yn}_ , the reward score is defined as: 

**==> picture [311 x 18] intentionally omitted <==**

ˆ where _y_ is the predicted answer, and I( _·_ ) denotes the indicator function. 

For other tasks, all ground-truth answers are expected to be included in the final output. An example is the task of Multi-Value Needle in a Haystack, where the question might be: “What are all the special magic numbers for XXX?” In such cases, the reward function is defined as: 

**==> picture [299 x 24] intentionally omitted <==**

where _| · |_ denotes the cardinality of a set. 

## **3.4 Rethinking MemAgent from Autoregressive Modeling Perspectives** 

Finally, to get a deeper sense of the MemAgent design, we propose to re-think language-model factorization in the following fashion. A standard autoregressive LLM factorizes the joint likelihood of a sequence **x** 1: _N_ as 

6 

**==> picture [424 x 240] intentionally omitted <==**

**----- Start of picture text -----**<br>
External Input External Output External Input External Output<br>Controller<br>p ( c [k] ∣ m [k] [−1] ) p ( m [k] ∣ c [k] ,  m [k] [−1] ) c [1] c [2] … c [K] c [K] [+1]<br>Read Head Write Head<br>∅ m [1] m [2] … m [K]<br>Memory t  = 0 t  = 1 t  = 2 t  =  K t  =  K  + 1<br>Architecture of MemAgent Graphical Model of MemAgent<br>**----- End of picture text -----**<br>


**Figure 4** The architecture and graphic model of MemAgent. The memory is modeled as a latent memory variable, thereby enabling the decomposition of the autoregressive language model into multiple steps of reading from and writing to the memory. 

_p_ ( **x** 1: _N_ ) =[�] _[N] n_ =1 _[p]_[(] _[x][n][|]_ **[ x]**[1:] _[n][−]_[1][)] _[,]_[implicitly][assuming][that][every][past][token][(or][at][least][its][hidden][state)][must] stay in the active context. This is what turns quadratic attention into the long-context bottleneck. 

MemAgent replaces the unbounded history with a fixed-length _memory_ **m** _∈_ V _[M]_ , as shown in Figure 4. The input text is streamed through the model in _K_ contiguous chunks **c**[1] _, . . . ,_ **c** _[K]_ (each of length _≤ C_ ). After chunk _k_ is read, the model overwrites the panel with a new vector **m** _[k]_ that summarizes _all_ evidence seen so far. Because _|_ **m** _[k] |_ = _M_ is constant, both compute and memory per step are _O_ ( _C_ + _M_ ), yielding an overall linear complexity _O_ ( _N_ ). 

Introducing the latent sequence **m**[1:] _[K][−]_[1] decomposes the original likelihood as 

**==> picture [349 x 34] intentionally omitted <==**

with base case **m**[0] = ∅. Inside each chunk, we still run an ordinary transformer decoder, but conditioned on a _constant_ context window ( **c** _[k] ,_ **m** _[k][−]_[1] ). The read path factorizes token-by-token, _p_ ( **c** _[k] |_ **m** _[k][−]_[1] ) = � _kCi_ =( _k−_ 1) _C_ +1 _[p]_[(] _[x][i][|]_ **[ x]**[1:] _[i][−]_[1] _[,]_ **[ m]** _[k][−]_[1][)] _[,]_[while][the][write][path][generates][the][next][memory][in][the][same][autoregressive] fashion. 

MemAgent enjoys token-level compression of context, yet local-global or linear-attention models compress long context in the _feature_ space; their summaries are implicit and opaque. MemAgent’s summaries reside in token space, so every intermediate memory is human-readable and can be inspected or even edited — a property we exploit when designing the RL reward (§3.3). Conceptually, Equation 8 turns the transformer into a recurrent network whose state size is user-controllable. 

**Why is RL Essential?** Because memory tokens are latent and updated via a discrete overwrite rule, backpropagation alone cannot teach the model _what_ to keep and what to discard. Our multi-conversation GRPO algorithm (§A) treats each read–write–read loop as an RL transition, directly rewarding memories that lead to a correct final answer. This bridges the gap between explicit supervision (answers) and implicit structure (good memories), completing the training pipeline introduced earlier. 

7 

The resulting MemAgent architecture preserves the vanilla decoder’s training recipe, requires no exotic attention kernels, and satisfies the long-context trilemma of arbitrary length, lossless extrapolation, and linear cost. 

## **4 Experiments** 

For our training and primary evaluation, we utilize multi-hop long-text question answering (QA) tasks, and further conduct evaluations on other various long-text tasks. We select prior long-context methods as baselines to evaluate the long-text extrapolation capabilities of the models by comparing performance changes as the length of the test set data increases. 

## **4.1 Datasets** 

RULER [1] comprises various synthetic tasks with controllable context lengths, making it an ideal benchmark for investigating how model performance varies with increasing context length. 

The Question Answering subset of RULER adapts existing short-context QA datasets for long-context evaluation by embedding golden paragraphs (containing correct answers) within extensive distractor content sampled from the same dataset. This configuration represents a real-world adaptation of the Needle in a Haystack (NIAH) paradigm, where questions serve as queries, golden paragraphs function as needles, and distractor paragraphs constitute the haystack. This task bridges the gap between synthetic evaluation and practical long-context applications, well poised for assessing a model’s ability to locate and extract relevant information from realistic document collections. 

We synthesized training samples from the HotpotQA dataset using this methodology. Our synthetic data comprises a total of 200 articles, with an approximate token length of 28K. 

We thoroughly cleaned our dataset by filtering out questions where the Best-Of-2 score is 100% without requiring any context for Qwen2.5-7B-Base or Qwen2.5-7B-Instruct. These questions likely represent common knowledge already internalized within the models’ memories. Using this method, we processed 80,000 samples from the HotpotQA [2] training split. Approximately 50% of the data were filtered out, and from the remaining samples we selected the frist 32,768 samples for further use. 

We then applied a similar approach to synthesize 128 samples from the HotpotQA validation set. To further investigate how model performance varies with length, we synthesized test sets with different context lengths using the same questions. The number of articles ranges from 50, 100, up to 6400, corresponding to context lengths of approximately 7K, 14K, and up to 3.5M tokens, respectively. 

## **4.2 Experimental Setup** 

**Training Details** To maintain comparability with previous work, we choose Qwen2.5-7B-Instruct and Qwen2.514B-Instruct as base models for experiments. We implement the framework for multi-conversation with independent contexts based on verl [61]. During training, we intentionally limit the model to an 8K context window to highlight its extrapolation capabilities. This 8K-window was allocated as follows: 1024 tokens for the query, 5000 tokens for the context chunk, 1024 tokens for the memory, and 1024 tokens for the output, with remaining tokens reserved for the chat template. Consequently, the model typically requires 5 to 7 conversational turns to process the entire context. 

**Hyperparameters** We use the GRPO algorithm for training, applying a KL factor of 1e-3 and disabling the entropy loss. We employ the AdamW optimizer with a learning rate of 1e-6, scheduled with a constant learning rate with linear warm-up. We use a rollout batch size of 128 and 256 for 7B and 14B models, respectively, and a group size of 16. The ratio of the sample batch size to the backpropagation batch size is set to 16. 

**Model Configuration** We use DeepSeek-R1-Distill-Qwen [51], Qwen-2.5-Instruct-1M [62] and QwenLong-L1 [63] as baselines. We follow the official configurations of these baseline models to set context lengths. Specifically, for the Qwen2.5-Instruct-1M series, we further extrapolate the context length to 1M tokens using DCA. For the DeepSeek-R1-Distill-Qwen series and QwenLong, the context length is set to 128K tokens. For the model 

8 

with 128K context length, the input consists of 120,000 tokens, with an output of 10,000 tokens. For the model with 1M context length, the input is 990,000 tokens, with an output of 10,000 tokens. 

## **4.3 Main Results** 

The main experimental results are reported in Table 2. We conduct a comparative analysis of all model performances within the context length ranging from 7K to 896K. Specifically, for the MemAgent model, we extend our evaluation to explore its extrapolation capabilities on ultra-long contexts of 1.75M and 3.5M, assessing how the model generalizes beyond the standard context range. 

From these results, we observe that MemAgent exhibits remarkable length extrapolation capabilities with only marginal performance decay as the input context length increases. This demonstrates the effectiveness of the proposed memory mechanism combined with reinforcement learning for handling ultra-long context scenarios. 

In contrast, baseline models demonstrate distinct failure patterns even within the context window. The reasoning models (DS-Distill-Qwen series) show rapid performance degradation, while QwenLong-L1 maintains reasonable performance within its training length 60K but experiences substantial degradation afterward. The Qwen2.5-Instruct-1M series models maintains an acceptable performance within 112K tokens. However, their performances deteriorate to zero at 896K tokens, well before reaching their theoretical 1M token capacity. This suggests that despite extended context windows, these models struggle with effective information utilization in ultra-long contexts. 

**Table 2** Main experimental results comparing model performance across various context lengths. All values represent accuracy (%). 

|**Model**|**Length**<br>**7K**<br>**14K**<br>**28K**<br>**56K**<br>**112K**<br>**224K**<br>**448K**<br>**896K**<br>**1.75M**<br>**3.5M**|
|---|---|
|QwenLong-L1-32B<br>Qwen2.5-Instruct-14B-1M<br>Qwen2.5-Instruct-7B-1M|72.66<br>75.00<br>72.66<br>60.94<br>31.25<br>17.19<br>13.28<br>11.72<br>N/A<br>N/A<br>60.16<br>60.94<br>50.00<br>57.03<br>50.00<br>37.50<br>8.59<br>0.00<br>N/A<br>N/A<br>61.72<br>56.25<br>53.91<br>55.47<br>51.56<br>33.59<br>12.50<br>0.00<br>N/A<br>N/A|
|DS-Distill-Qwen-32B<br>DS-Distill-Qwen-14B<br>DS-Distill-Qwen-7B|70.31<br>66.41<br>65.62<br>46.88<br>23.44<br>13.28<br>7.81<br>7.03<br>N/A<br>N/A<br>64.06<br>64.84<br>57.03<br>40.62<br>14.84<br>8.59<br>3.12<br>6.25<br>N/A<br>N/A<br>30.47<br>12.50<br>3.12<br>0.00<br>0.00<br>0.78<br>0.00<br>0.00<br>N/A<br>N/A|
|RL-MemAgent-14B<br>RL-MemAgent-7B|**83.59**<br>**82.03**<br>**84.38**<br>**80.47**<br>76.56<br>**81.25**<br>**75.00**<br>**77.34**<br>**76.56**<br>**78.12**<br>82.03<br>79.69<br>78.91<br>77.34<br>**79.69**<br>72.66<br>74.22<br>76.56<br>75.78<br>71.09|



## **4.4 Ablation Study** 

**RL Training** To investigate the impact of reinforcement learning on the memory mechanism, we conduct further ablation experiments. Our baselines are Qwen2.5-Instruct [64] series and Qwen2.5-Instruct models which are equipped with memory mechanism without RL training. 

As shown in Figure 5, vanilla models exhibit severe performance degradation as context length increases, especially after 112K where the inputs are truncated because of the context window. While the model equipped with a memory, without RL training, demonstrates better performance and maintains reasonable performance on tasks exceeding the context length, it still experiences an overall decline in performance as the input length increases. 

In contrast, RL-trained models maintain consistently high performance across all context lengths with minimal degradation. This demonstrates that while the memory mechanism provides structural support for long contexts, reinforcement learning is essential for teaching models to properly leverage the memory. 

**Out-of-Distribution Tasks** To evaluate the generalization capabilities of our approach, we conduct comprehensive experiments on the OOD task in RULER benchmark, including **needle-in-a-haystack variants** , **variable tracking** , **frequent words extraction** , and **question answering** synthesized from SQuAD [65]. We synthesize 

9 

**==> picture [472 x 190] intentionally omitted <==**

**----- Start of picture text -----**<br>
80<br>70<br>60<br>50<br>40<br>RL-MemAgent-14B<br>30 RL-MemAgent-7B<br>MemAgent-32B w/o RL<br>MemAgent-14B w/o RL<br>20 MemAgent-7B w/o RL<br>Qwen2.5-Instruct-32B<br>Qwen2.5-Instruct-14B<br>10<br>Qwen2.5-Instruct-7B<br>Truncation<br>0<br>7K 28K 112K 224K 448K 896K<br>Context Length in Tokens<br>Score<br>**----- End of picture text -----**<br>


**Figure 5** Ablation study on RULER-HotpotQA comparing models with and without RL training across context lengths from 28K to 896K tokens. 

context lengths ranging from 8K to 512K tokens for these tasks, except that SQuAD extends only to 256K tokens due to limited document length. 

Figure 6 presents the performance comparison across different task categories. The results demonstrate that MemAgent maintains consistently superior performance across diverse task types. Particularly, MemAgent14B achieves over 95% accuracy on the average RULER tasks in context ranging from 8K to 512K, while MemAgent-7B achieves the best performance, surpassing 32B model without RL training and long-context post-trained models. MemAgent-7B/14B both maintain stable performance on the SQuAD-based QA task, indicating that memorizing ability can generalize beyond training data. In contrast, baseline models show significant degradation beyond 128K tokens across all task categories. 

The consistent performance strength across heterogeneous tasks validates that the memory mechanism effectively generalizes to various long-context scenarios rather than overfitting to specific formats. Complete results for all individual RULER tasks are provided in Appendix B. 

## **4.5 Case Study** 

To further illustrate the proposed memory mechanism in detail, we conduct a case study on a generation trajectory of MemAgent-14B. The input question is: _The director of the romantic comedy ‘Big Stone Gap’ is based in what New York city?_ This a 2-hop question with the following relevant Wikipedia entries: 

- 1) **Big Stone Gap** is a 2014 American drama romantic comedy film written and directed by Adriana Trigiani. 

2) **Adriana Trigiani** is an Italian American best-selling author of sixteen books, television writer, film director, and entrepreneur based in Greenwich Village, New York City. 

In the first round, the model is presented with the entry _Ghost_ , which refers to a production team also based in _New York_ . The model chooses to retain this potentially useful information for future use. In the second round, no relevant context is provided; nevertheless, the model maintains its agent state, demonstrating robustness against distraction. In the third round, both relevant entries are presented. The model correctly identifies critical information and updates its memory accordingly, leading to the correct answer: _Greenwich Village, New York City_ . At this point, the reasoning process is complete. In the remaining rounds, the model’s memory remains unchanged and is used to produce the final response. 

10 

**==> picture [472 x 196] intentionally omitted <==**

**----- Start of picture text -----**<br>
100 100<br>RL-MemAgent-14B 97.45 96.97 97.46 97.85 96.08 96.24 95.40 RL-MemAgent-14B 77.34 76.56 79.69 77.34 78.12 77.34<br>RL-MemAgent-7B 93.03 92.03 91.33 88.83 86.92 83.70 81.91 RL-MemAgent-7B 81.25 81.25 82.03 76.56 79.69 81.25<br>MemAgent-32B w/o RL 99.04 96.59 94.61 91.85 86.56 83.57 81.51 80 MemAgent-32B w/o RL 78.12 75.00 71.09 75.00 73.44 71.09 80<br>MemAgent-14B w/o RL 97.95 90.22 87.43 80.30 67.97 58.88 46.18 MemAgent-14B w/o RL 70.31 68.75 70.31 66.41 64.84 53.91<br>MemAgent-7B w/o RL 92.56 92.47 90.52 88.36 84.46 80.05 73.48 MemAgent-7B w/o RL 60.16 66.41 58.59 55.47 66.02 57.03<br>QwenLong-L1-32B 92.00 91.23 91.40 77.39 41.66 23.22 14.78 60 QwenLong-L1-32B 82.81 78.91 73.44 67.19 36.72 33.59 60<br>Qwen2.5-Instruct-14B-1M 98.34 97.31 93.47 90.50 89.95 83.91 62.34 Qwen2.5-Instruct-14B-1M 85.16 82.81 79.69 77.34 68.75 51.56<br>Qwen2.5-Instruct-7B-1M 90.28 89.57 88.56 87.37 85.11 78.14 39.21 Qwen2.5-Instruct-7B-1M 76.56 74.22 71.88 63.28 61.72 50.00<br>DS-Distill-Qwen-32B 97.28 97.54 95.21 76.63 40.11 24.09 15.73 40 DS-Distill-Qwen-32B 78.91 71.09 67.97 46.88 28.91 28.91 40<br>DS-Distill-Qwen-14B 95.33 95.06 89.89 64.50 28.65 17.59 12.40 DS-Distill-Qwen-14B 71.09 67.97 58.59 39.84 22.66 23.44<br>DS-Distill-Qwen-7B 53.96 14.77 1.45 0.03 0.00 0.00 0.00 DS-Distill-Qwen-7B 31.25 11.72 8.59 2.34 0.78 0.78<br>Qwen2.5-Instruct-32B 97.23 94.42 91.59 91.34 79.95 44.74 27.01 20 Qwen2.5-Instruct-32B 86.72 76.56 71.09 65.62 55.47 45.31 20<br>Qwen2.5-Instruct-14B 90.91 86.65 83.05 79.16 69.39 39.22 24.87 Qwen2.5-Instruct-14B 83.59 73.44 66.41 58.59 48.44 42.19<br>Qwen2.5-Instruct-7B 58.53 48.10 45.48 54.37 38.01 25.69 17.05 Qwen2.5-Instruct-7B 71.09 67.19 71.09 57.03 49.22 39.84<br>0 0<br>8K 16K 32K 64K 128K 256K 512K 8K 16K 32K 64K 128K 256K<br>RL-MemAgent Long Context Model Backbone RL-MemAgent Long Context Model Backbone<br>MemAgent w/o RL Reasoning Model MemAgent w/o RL Reasoning Model<br>**----- End of picture text -----**<br>


- **(a)** RULER average across 10 tasks 

- **(b)** RULER-QA task from SQuAD 

**Figure 6** Performance heatmaps on RULER benchmark tasks showing accuracy scores across different context lengths (greener indicates better performance). Models are grouped by type on the vertical axis. (a) Average performance across 10 synthetic tasks including needle-in-a-haystack variants, variable tracking, and word extraction. (b) Question answering task synthesized from SQuAD dataset with context lengths up to 256K tokens. 

## **Chunk1** 

... 

Document 20: Ghost (production team) 

Ghost is a Swedish record producing and songwriting team, composed of Ulf Lindström and Johan Ekhé, based in _New York City_ . 

... 

## **Memory1** 

The provided section does not contain any information about the director of the romantic comedy "Big Stone Gap" or their base in New York City. **However, it does mention a production team called Ghost which is based in New York City.** Since the problem is specifically asking about the director of "Big Stone Gap," and there is no direct information about the director in the given section, we cannot infer anything directly related to the problem from the provided text. Updated memory: 

- Ghost is a Swedish record producing and songwriting team, composed of Ulf Lindström and Johan Ekhé, based in New York City. They are perhaps best known for writing and producing Swedish singer Robyn’s three first studio albums. 

The problem remains unanswered based on the provided section. Additional information would be needed to determine the director of "Big Stone Gap" and their base in New York City. 

11 

## **Chunk2** 

...(No relevant document) 

## **Memory2** 

**The provided section does not contain any information about the director of the romantic comedy Big Stone Gap or their base in New York City. Therefore, the updated memory remains unchanged with respect to the problem at hand.** The memory still notes that the Ghost record producing and songwriting team is based in New York City, but it does not provide any information about the director of "Big Stone Gap." Updated memory: 

- Ghost is a Swedish record producing and songwriting team, composed of Ulf Lindström and Johan Ekhé, based in New York City. They are perhaps best known for writing and producing Swedish singer Robyn’s three first studio albums. 

- The problem remains unanswered based on the provided section. Additional information would be needed to determine the director of "Big Stone Gap" and their base in New York City. 

## **Chunk3** 

... Document 81: Big Stone Gap (film) 

**Big Stone Gap** is a 2014 American drama romantic comedy film written and directed by **Adriana Trigiani** and produced by Donna Gigliotti for Altar Identity Studios, a subsidiary of Media Society. ... 

Document 96: Adriana Trigiani 

**Adriana Trigiani** is an Italian American best-selling author of sixteen books, television writer, film director, and entrepreneur based in **Greenwich Village, New York City** . 

... 

## **Memory3** 

Updated memory: 

- Ghost is a Swedish record producing and songwriting team, composed of Ulf Lindström and Johan Ekhé, based in New York City. They are perhaps best known for writing and producing Swedish singer Robyn’s three first studio albums. 

- **Adriana Trigiani** is an Italian American best-selling author of sixteen books, television writer, film director, and entrepreneur based in Greenwich Village, New York City. Trigiani has published a novel a year since 2000. 

- **Big Stone Gap** is a 2014 American drama romantic comedy film written and directed by Adriana Trigiani and produced by Donna Gigliotti for Altar Identity Studios, a subsidiary of Media Society. Based on Trigiani’s 2000 best-selling novel of the same name, the story is set in the actual Virginia town of Big Stone Gap circa 1970s. The film had its world premiere at the Virginia Film Festival on November 6, 2014. 

- The director of the romantic comedy "Big Stone Gap" is Adriana Trigiani, who is based in **Greenwich Village, New York City** . 

This trajectory highlights several core capabilities learned by the memory agent: estimating potentially relevant content from query keywords and storing it preemptively, immediately updating memory upon encountering context that matches the query, and remaining unaffected by irrelevant information. Notably, these memory behaviors are not the result of architectural attention mechanisms, but emerge as text generation abilities reinforced through RL. 

## **5 Conclusion** 

We propose a novel approach to modeling long-context tasks by introducing a latent variable memory. This enables the decomposition of continuous autoregressive generation process into a series of steps that sequentially 

12 

generate context from memory. Our method can handle infinitely long input text with _O_ ( _N_ ) computational complexity, based on existing Dense-Attention Transformers, without altering the generation paradigm or introducing additional model architectures. We introduce MemAgent to implement this modeling approach, equipping LLMs with an RL-trained memory, allowing the model to learn the ability to record relevant information and ignore irrelevant details. Experiments show that when trained on 32K-length data with 8K context (including a 1024-token memory and processing 5000 tokens of input per step), the model can extrapolate to 3.5M with almost lossless performance during testing. Ablation studies demonstrate the effectiveness of using the memory itself as a long-context processing mechanism, as well as the benefits of further RL training on top of it. The results on both in-domain and out-of-domain tasks show that MemAgent surpasses long-context post-trained models, reasoning models and other baselines, achieving state-of-the-art performance on long-context tasks. 

13