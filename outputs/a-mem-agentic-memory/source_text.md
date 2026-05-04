## **A-Mem: Agentic Memory for LLM Agents** 

**Wujiang Xu**[1] **, Zujie Liang**[2] **, Kai Mei**[1] **, Hang Gao**[1] **, Juntao Tan**[1] **, Yongfeng Zhang**[1] _[,]_[3] 1 2 3 Rutgers University Independent Researcher AIOS Foundation `wujiang.xu@rutgers.edu` 

## **Abstract** 

While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems’ fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution – as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. 

**Code for Benchmark Evaluation** : 

https://github.com/WujiangXu/AgenticMemory 

**Code for Production-ready Agentic Memory** : https://github.com/WujiangXu/A-mem-sys 

## **1 Introduction** 

Large Language Model (LLM) agents have demonstrated remarkable capabilities in various tasks, with recent advances enabling them to interact with environments, execute tasks, and make decisions autonomously [23, 33, 7]. They integrate LLMs with external tools and delicate workflows to improve reasoning and planning abilities. Though LLM agent has strong reasoning performance, it still needs a memory system to provide long-term interaction ability with the external environment [35]. 

Existing memory systems [25, 39, 28, 21] for LLM agents provide basic memory storage functionality. These systems require agent developers to predefine memory storage structures, specify storage points within the workflow, and establish retrieval timing. Meanwhile, to improve structured memory organization, Mem0 [8], following the principles of RAG [9, 18, 30], incorporates graph databases for storage and retrieval processes. While graph databases provide structured organization for memory systems, their reliance on predefined schemas and relationships fundamentally limits their adaptability. This limitation manifests clearly in practical scenarios - when an agent learns a novel mathematical solution, current systems can only categorize and link this information within their preset framework, 

Preprint. 

**==> picture [360 x 61] intentionally omitted <==**

**----- Start of picture text -----**<br>
Write Write<br>Interaction Interaction<br>Read Read<br>Environment LLM Agents Memory Environment LLM Agents Agentic Memory<br>(a)  Traditional memory system. (b)  Our proposed agentic memory.<br>**----- End of picture text -----**<br>


**Figure 1:** Traditional memory systems require predefined memory access patterns specified in the workflow, limiting their adaptability to diverse scenarios. Contrastly, our A-MEM enhances the flexibility of LLM agents by enabling dynamic memory operations. 

unable to forge innovative connections or develop new organizational patterns as knowledge evolves. Such rigid structures, coupled with fixed agent workflows, severely restrict these systems’ ability to generalize across new environments and maintain effectiveness in long-term interactions. The challenge becomes increasingly critical as LLM agents tackle more complex, open-ended tasks, where flexible knowledge organization and continuous adaptation are essential. Therefore, _how to design a flexible and universal memory system that supports LLM agents’ long-term interactions_ remains a crucial challenge. 

In this paper, we introduce a novel agentic memory system, named as A-MEM, for LLM agents that enables dynamic memory structuring without relying on static, predetermined memory operations. Our approach draws inspiration from the Zettelkasten method [15, 1], a sophisticated knowledge management system that creates interconnected information networks through atomic notes and flexible linking mechanisms. Our system introduces an agentic memory architecture that enables autonomous and flexible memory management for LLM agents. For each new memory, we construct comprehensive notes, which integrates multiple representations: structured textual attributes including several attributes and embedding vectors for similarity matching. Then A-MEM analyzes the historical memory repository to establish meaningful connections based on semantic similarities and shared attributes. This integration process not only creates new links but also enables dynamic evolution when new memories are incorporated, they can trigger updates to the contextual representations of existing memories, allowing the entire memories to continuously refine and deepen its understanding over time. The contributions are summarized as: 

• We present A-MEM, an agentic memory system for LLM agents that enables autonomous generation of contextual descriptions, dynamic establishment of memory connections, and intelligent evolution of existing memories based on new experiences. This system equips LLM agents with long-term interaction capabilities without requiring predetermined memory operations. 

• We design an agentic memory update mechanism where new memories automatically trigger two key operations: link generation and memory evolution. Link generation automatically establishes connections between memories by identifying shared attributes and similar contextual descriptions. Memory evolution enables existing memories to dynamically adapt as new experiences are analyzed, leading to the emergence of higher-order patterns and attributes. 

• We conduct comprehensive evaluations of our system using a long-term conversational dataset, comparing performance across six foundation models using six distinct evaluation metrics, demonstrating significant improvements. Moreover, we provide T-SNE visualizations to illustrate the structured organization of our agentic memory system. 

## **2 Related Work** 

## **2.1 Memory for LLM Agents** 

Prior works on LLM agent memory systems have explored various mechanisms for memory management and utilization [23, 21, 8, 39]. Some approaches complete interaction storage, which maintains comprehensive historical records through dense retrieval models [39] or read-write memory structures [24]. Moreover, MemGPT [25] leverages cache-like architectures to prioritize recent information. Similarly, SCM [32] proposes a Self-Controlled Memory framework that enhances LLMs’ capability to maintain long-term memory through a memory stream and controller mechanism. However, these approaches face significant limitations in handling diverse real-world tasks. While they can provide basic memory functionality, their operations are typically constrained by predefined structures and fixed workflows. These constraints stem from their reliance on rigid operational 

2 

**==> picture [397 x 188] intentionally omitted <==**

**----- Start of picture text -----**<br>
Note Construction Link Generation Memory Evolution Memory Retrieval<br>Memory<br>Interaction Box 1 … … … Box  i … … Box  n+1 … … Retrieve Query  ModelText<br>Environment LLM Agents Embedding Query<br>… … … … … … … Top- k<br>Write Box  j Box  n Box  n+2<br>Can you help me implement a handle both memory and disk custom cache system for my web application? I need it to  Conversation 1 storage.  production. Can we modify great, but we're seeing high The cache system works it to implement an LRU memory usage in eviction policy? Conversation 2 mj Top- k MemoryRelative  1 st … …<br>LLM<br>LLM Note Attributes:Timestamp LLM Store Action<br>Content<br>Context … … … … LLM Agents<br>Keywords Box  n+1 Box  n+2<br>Tags Evolve<br>Note Embedding Note<br>… …<br>Retrieve<br>**----- End of picture text -----**<br>


**Figure 2:** Our A-MEM architecture comprises three integral parts in memory storage. During note construction, the system processes new interaction memories and stores them as notes with multiple attributes. The link generation process first retrieves the most relevant historical memories and then employs an LLM to determine whether connections should be established between them. The concept of a ’box’ describes that related memories become interconnected through their similar contextual descriptions, analogous to the Zettelkasten method. However, our approach allows individual memories to exist simultaneously within multiple different boxes. During the memory retrieval stage, we extract query embeddings using a text encoding model and search the memory database for relevant matches. When related memory is retrieved, similar memories that are linked within the same box are also automatically accessed. 

patterns, particularly in memory writing and retrieval processes. Such inflexibility leads to poor generalization in new environments and limited effectiveness in long-term interactions. Therefore, designing a flexible and universal memory system that supports agents’ long-term interactions remains a crucial challenge. 

## **2.2 Retrieval-Augmented Generation** 

Retrieval-Augmented Generation (RAG) has emerged as a powerful approach to enhance LLMs by incorporating external knowledge sources [18, 6, 10]. The standard RAG [37, 34] process involves indexing documents into chunks, retrieving relevant chunks based on semantic similarity, and augmenting the LLM’s prompt with this retrieved context for generation. Advanced RAG systems [20, 12] have evolved to include sophisticated pre-retrieval and post-retrieval optimizations. Building upon these foundations, recent researches has introduced agentic RAG systems that demonstrate more autonomous and adaptive behaviors in the retrieval process. These systems can dynamically determine when and what to retrieve [4, 14], generate hypothetical responses to guide retrieval, and iteratively refine their search strategies based on intermediate results [31, 29]. 

However, while agentic RAG approaches demonstrate agency in the retrieval phase by autonomously deciding when and what to retrieve [4, 14, 38], our agentic memory system exhibits agency at a more fundamental level through the autonomous evolution of its memory structure. Inspired by the Zettelkasten method, our system allows memories to actively generate their own contextual descriptions, form meaningful connections with related memories, and evolve both their content and relationships as new experiences emerge. This fundamental distinction in agency between retrieval versus storage and evolution distinguishes our approach from agentic RAG systems, which maintain static knowledge bases despite their sophisticated retrieval mechanisms. 

## **3 Methodolodgy** 

Our proposed agentic memory system draws inspiration from the Zettelkasten method, implementing a dynamic and self-evolving memory system that enables LLM agents to maintain long-term memory without predetermined operations. The system’s design emphasizes atomic note-taking, flexible linking mechanisms, and continuous evolution of knowledge structures. 

3 

## **3.1 Note Construction** 

Building upon the Zettelkasten method’s principles of atomic note-taking and flexible organization, we introduce an LLM-driven approach to memory note construction. When an agent interacts with its environment, we construct structured memory notes that capture both explicit information and LLMgenerated contextual understanding. Each memory note _mi_ in our collection _M_ = { _m_ 1 _, m_ 2 _, ..., mN_ } is represented as: 

**==> picture [263 x 11] intentionally omitted <==**

where _ci_ represents the original interaction content, _ti_ is the timestamp of the interaction, _Ki_ denotes LLM-generated keywords that capture key concepts, _Gi_ contains LLM-generated tags for categorization, _Xi_ represents the LLM-generated contextual description that provides rich semantic understanding, and _Li_ maintains the set of linked memories that share semantic relationships. To enrich each memory note with meaningful context beyond its basic content and timestamp, we leverage an LLM to analyze the interaction and generate these semantic components. The note construction process involves prompting the LLM with carefully designed templates _Ps_ 1: 

_Ki, Gi, Xi_ ← LLM( _ci_ ∥ _ti_ ∥ _Ps_ 1) (2) 

Following the Zettelkasten principle of atomicity, each note captures a single, self-contained unit of knowledge. To enable efficient retrieval and linking, we compute a dense vector representation via a text encoder [27] that encapsulates all textual components of the note: 

**==> picture [269 x 12] intentionally omitted <==**

By using LLMs to generate enriched components, we enable autonomous extraction of implicit knowledge from raw interactions. The multi-faceted note structure ( _Ki_ , _Gi_ , _Xi_ ) creates rich representations that capture different aspects of the memory, facilitating nuanced organization and retrieval. Additionally, the combination of LLM-generated semantic components with dense vector representations provides both context and computationally efficient similarity matching. 

## **3.2 Link Generation** 

Our system implements an autonomous link generation mechanism that enables new memory notes to form meaningful connections without predefined rules. When the constrctd memory note _mn_ is added to the system, we first leverage its semantic embedding for similarity-based retrieval. For each existing memory note _mj_ ∈ _M_ , we compute a similarity score: 

**==> picture [232 x 24] intentionally omitted <==**

The system then identifies the top- _k_ most relevant memories: 

**==> picture [283 x 13] intentionally omitted <==**

Based on these candidate nearest memories, we prompt the LLM to analyze potential connections based on their potential common attributes. Formally, the link set of memory _mn_ update like: 

**==> picture [262 x 12] intentionally omitted <==**

Each generated link _li_ is structured as: _Li_ = { _mi, ..., mk_ }. By using embedding-based retrieval as an initial filter, we enable efficient scalability while maintaining semantic relevance. A-MEM can quickly identify potential connections even in large memory collections without exhaustive comparison. More importantly, the LLM-driven analysis allows for nuanced understanding of relationships that goes beyond simple similarity metrics. The language model can identify subtle patterns, causal relationships, and conceptual connections that might not be apparent from embedding similarity alone. We implements the Zettelkasten principle of flexible linking while leveraging modern language models. The resulting network emerges organically from memory content and context, enabling natural knowledge organization. 

## **3.3 Memory Evolution** 

After creating links for the new memory, A-MEM evolves the retrieved memories based on their textual information and relationships with the new memory. For each memory _mj_ in the nearest 

4 

neighbor set _M[n]_ near[, the system determines whether to update its context, keywords, and tags.][This] evolution process can be formally expressed as: 

**==> picture [286 x 14] intentionally omitted <==**

The evolved memory _m_[∗] _j_[then][replaces][the][original][memory] _[m] j_[in][the][memory][set] _[M]_[.][This] evolutionary approach enables continuous updates and new connections, mimicking human learning processes. As the system processes more memories over time, it develops increasingly sophisticated knowledge structures, discovering higher-order patterns and concepts across multiple memories. This creates a foundation for autonomous memory learning where knowledge organization becomes progressively richer through the ongoing interaction between new experiences and existing memories. 

## **3.4 Retrieve Relative Memory** 

In each interaction, our A-MEM performs context-aware memory retrieval to provide the agent with relevant historical information. Given a query text _q_ from the current interaction, we first compute its dense vector representation using the same text encoder used for memory notes: 

**==> picture [224 x 12] intentionally omitted <==**

The system then computes similarity scores between the query embedding and all existing memory notes in _M_ using cosine similarity: 

**==> picture [287 x 24] intentionally omitted <==**

Then we retrieve the k most relevant memories from the historical memory storage to construct a contextually appropriate prompt. 

**==> picture [286 x 12] intentionally omitted <==**

These retrieved memories provide relevant historical context that helps the agent better understand and respond to the current interaction. The retrieved context enriches the agent’s reasoning process by connecting the current interaction with related past experiences stored in the memory system. 

## **4 Experiment** 

## **4.1 Dataset and Evaluation** 

To evaluate the effectiveness of instruction-aware recommendation in long-term conversations, we utilize the LoCoMo dataset [22], which contains significantly longer dialogues compared to existing conversational datasets [36, 13]. While previous datasets contain dialogues with around 1K tokens over 4-5 sessions, LoCoMo features much longer conversations averaging 9K tokens spanning up to 35 sessions, making it particularly suitable for evaluating models’ ability to handle long-range dependencies and maintain consistency over extended conversations. The LoCoMo dataset comprises diverse question types designed to comprehensively evaluate different aspects of model understanding: (1) single-hop questions answerable from a single session; (2) multihop questions requiring information synthesis across sessions; (3) temporal reasoning questions testing understanding of time-related information; (4) open-domain knowledge questions requiring integration of conversation context with external knowledge; and (5) adversarial questions assessing models’ ability to identify unanswerable queries. In total, LoCoMo contains 7,512 question-answer pairs across these categories. Besides, we use a new dataset, named DialSim [16], to evaluate the effectiveness of our memory system. It is question-answering dataset derived from long-term multi-party dialogues. The dataset is derived from popular TV shows (Friends, The Big Bang Theory, and The Office), covering 1,300 sessions spanning five years, containing approximately 350,000 tokens, and including more than 1,000 questions per session from refined fan quiz website questions and complex questions generated from temporal knowledge graphs. 

For comparison baselines, we compare to **LoCoMo** [22], **ReadAgent** [17], **MemoryBank** [39] and **MemGPT** [25]. The detailed introduction of baselines can be found in Appendix A.1 For evaluation, we employ two primary metrics: the F1 score to assess answer accuracy by balancing precision and recall, and BLEU-1 [26] to evaluate generated response quality by measuring word overlap 

5 

**Table 1:** Experimental results on LoCoMo dataset of QA tasks across five categories (Multi Hop, Temporal, Open Domain, Single Hop, and Adversial) using different methods. Results are reported in F1 and BLEU-1 (%) scores. The best performance is marked in bold, and our proposed method A-MEM (highlighted in gray) demonstrates competitive performance across six foundation language models. 

|**M**|**odel**<br>**Method**|**odel**<br>**Method**|**Category**|**Category**|**Category**|**Category**|**Category**|**Average**|**Average**|
|---|---|---|---|---|---|---|---|---|---|
||||**Multi Hop**<br>**F1**<br>**BLEU**|**Temporal**<br>**F1**<br>**BLEU**|**Open Domain**<br>**F1**<br>**BLEU**|**Single Hop**<br>**F1**<br>**BLEU**|**Adversial**<br>**F1**<br>**BLEU**|**Ranking**<br>**F1**<br>**BLEU**|**Token**<br>**Length**|
|**GPT**|**4o-mini**|LOCOMO<br>READAGENT<br>MEMORYBANK<br>MEMGPT|25.02<br>19.75<br>9.15<br>6.48<br>5.00<br>4.77<br>26.65<br>17.72|18.41<br>14.77<br>12.60<br>8.87<br>9.68<br>6.99<br>25.52<br>19.44|12.04<br>11.16<br>5.31<br>5.12<br>5.56<br>5.94<br>9.15<br>7.44|40.36<br>29.05<br>9.67<br>7.66<br>6.61<br>5.16<br>41.04<br>34.34|**69.23**<br>**68.75**<br>9.81<br>9.02<br>7.36<br>6.48<br>43.29<br>42.73|2.4<br>2.4<br>4.2<br>4.2<br>4.8<br>4.8<br>2.4<br>2.4|16,910<br>643<br>432<br>16,977|
|||**A-MEM**|**27.02**<br>**20.09**|**45.85**<br>**36.67**|**12.14**<br>**12.00**|**44.65**<br>**37.06**|50.03<br>49.47|**1.2**<br>**1.2**|2,520|
||**4o**|LOCOMO<br>READAGENT<br>MEMORYBANK<br>MEMGPT|28.00<br>18.47<br>14.61<br>9.95<br>6.49<br>4.69<br>30.36<br>22.83|9.09<br>5.78<br>4.16<br>3.19<br>2.47<br>2.43<br>17.29<br>13.18|16.47<br>14.80<br>8.84<br>8.37<br>6.43<br>5.30<br>12.24<br>11.87|**61.56**<br>**54.19**<br>12.46<br>10.29<br>8.28<br>7.10<br>60.16<br>53.35|**52.61**<br>**51.13**<br>6.81<br>6.13<br>4.42<br>3.67<br>34.96<br>34.25|2.0<br>2.0<br>4.0<br>4.0<br>5.0<br>5.0<br>2.4<br>2.4|16,910<br>805<br>569<br>16,987|
|||**A-MEM**|**32.86**<br>**23.76**|**39.41**<br>**31.23**|**17.10**<br>**15.84**|48.43<br>42.97|36.35<br>35.53|**1.6**<br>**1.6**|1,216|
|**Qwen2.5**|**1.5b**|LOCOMO<br>READAGENT<br>MEMORYBANK<br>MEMGPT|9.05<br>6.55<br>6.61<br>4.93<br>11.14<br>8.25<br>10.44<br>7.61|4.25<br>4.04<br>2.55<br>2.51<br>4.46<br>2.87<br>4.21<br>3.89|9.91<br>8.50<br>5.31<br>12.24<br>8.05<br>6.21<br>13.42<br>11.64|11.15<br>8.67<br>10.13<br>7.54<br>13.42<br>11.01<br>9.56<br>7.34|40.38<br>40.23<br>5.42<br>27.32<br>36.76<br>34.00<br>31.51<br>28.90|3.4<br>3.4<br>4.6<br>4.6<br>2.6<br>2.6<br>3.4<br>3.4|16,910<br>752<br>284<br>16,953|
|||**A-MEM**|**18.23**<br>**11.94**|**24.32**<br>**19.74**|**16.48**<br>**14.31**|**23.63**<br>**19.23**|**46.00**<br>**43.26**|**1.0**<br>**1.0**|1,300|
||**3b**|LOCOMO<br>READAGENT<br>MEMORYBANK<br>MEMGPT|4.61<br>4.29<br>2.47<br>1.78<br>3.60<br>3.39<br>5.07<br>4.31|3.11<br>2.71<br>3.01<br>3.01<br>1.72<br>1.97<br>2.94<br>2.95|4.55<br>5.97<br>5.57<br>5.22<br>6.63<br>6.58<br>7.04<br>7.10|7.03<br>5.69<br>3.25<br>2.51<br>4.11<br>3.32<br>7.26<br>5.52|16.95<br>14.81<br>15.78<br>14.01<br>13.07<br>10.30<br>14.47<br>12.39|3.2<br>3.2<br>4.2<br>4.2<br>4.2<br>4.2<br>2.4<br>2.4|16,910<br>776<br>298<br>16,961|
|||**A-MEM**|**12.57**<br>**9.01**|**27.59**<br>**25.07**|**7.12**<br>**7.28**|**17.23**<br>**13.12**|**27.91**<br>**25.15**|**1.0**<br>**1.0**|1,137|
|**Llama 3.2**|**1b**|LOCOMO<br>READAGENT<br>MEMORYBANK<br>MEMGPT|11.25<br>9.18<br>5.96<br>5.12<br>13.18<br>10.03<br>9.19<br>6.96|7.38<br>6.82<br>1.93<br>2.30<br>7.61<br>6.27<br>4.02<br>4.79|11.90<br>10.38<br>12.46<br>11.17<br>15.78<br>12.94<br>11.14<br>8.24|12.86<br>10.50<br>7.75<br>6.03<br>17.30<br>14.03<br>10.16<br>7.68|51.89<br>48.27<br>44.64<br>40.15<br>52.61<br>47.53<br>49.75<br>45.11|3.4<br>3.4<br>4.6<br>4.6<br>2.0<br>2.0<br>4.0<br>4.0|16,910<br>665<br>274<br>16,950|
|||**A-MEM**|**19.06**<br>**11.71**|**17.80**<br>**10.28**|**17.55**<br>**14.67**|**28.51**<br>**24.13**|**58.81**<br>**54.28**|**1.0**<br>**1.0**|1,376|
||**3b**|LOCOMO<br>READAGENT<br>MEMORYBANK<br>MEMGPT|6.88<br>5.77<br>2.47<br>1.78<br>6.19<br>4.47<br>5.32<br>3.99|4.37<br>4.40<br>3.01<br>3.01<br>3.49<br>3.13<br>2.68<br>2.72|10.65<br>9.29<br>5.57<br>5.22<br>4.07<br>4.57<br>5.64<br>5.54|8.37<br>6.93<br>3.25<br>2.51<br>7.61<br>6.03<br>4.32<br>3.51|30.25<br>28.46<br>15.78<br>14.01<br>18.65<br>17.05<br>21.45<br>19.37|2.8<br>2.8<br>4.2<br>4.2<br>3.2<br>3.2<br>3.8<br>3.8|16,910<br>461<br>263<br>16,956|
|||**A-MEM**|**17.44**<br>**11.74**|**26.38**<br>**19.50**|**12.53**<br>**11.83**|**28.14**<br>**23.87**|**42.04**<br>**40.60**|**1.0**<br>**1.0**|1,126|



with ground truth responses. Also, we report the average token length for answering one question. Besides reporting experiment results with four additional metrics (ROUGE-L, ROUGE-2, METEOR, and SBERT Similarity), we also present experimental outcomes using different foundation models including DeepSeek-R1-32B [11], Claude 3.0 Haiku [2], and Claude 3.5 Haiku [3] in Appendix A.3. 

## **4.2 Implementation Details** 

For all baselines and our proposed method, we maintain consistency by employing identical system prompts as detailed in Appendix B. The deployment of Qwen-1.5B/3B and Llama 3.2 1B/3B models is accomplished through local instantiation using Ollama[1] , with LiteLLM[2] managing structured output generation. For GPT models, we utilize the official structured output API. In our memory retrieval process, we primarily employ _k_ =10 for top- _k_ memory selection to maintain computational efficiency, while adjusting this parameter for specific categories to optimize performance. The detailed configurations of _k_ can be found in Appendix A.5. For text embedding, we implement the all-minilm-l6-v2 model across all experiments. 

## **4.3 Empricial Results** 

**Performance Analysis.** In our empirical evaluation, we compared A-MEM with four competitive baselines including LoCoMo [22], ReadAgent [17], MemoryBank [39], and MemGPT [25] on the LoCoMo dataset. For non-GPT foundation models, our A-MEM consistently outperforms all baselines across different categories, demonstrating the effectiveness of our agentic memory approach. For GPT-based models, while LoCoMo and MemGPT show strong performance in certain categories like Open Domain and Adversial tasks due to their robust pre-trained knowledge in simple fact retrieval, our A-MEM demonstrates superior performance in Multi-Hop tasks achieves at least two times better performance that require complex reasoning chains. In addition to experiments on the LoCoMo dataset, we also compare our method on the DialSim dataset against LoCoMo and MemGPT. A-MEM consistently outperforms all baselines across evaluation metrics, achieving an F1 

> 1 `https://github.com/ollama/ollama` 

> 2 `https://github.com/BerriAI/litellm` 

6 

**Table 2:** Comparison of different memory mechanisms across multiple evaluation metrics on DialSim [16]. Higher scores indicate better performance, with A-MEM showing superior results across all metrics. 

|**Method**|**F1**<br>**BLEU-1**<br>**ROUGE-L**<br>**ROUGE-2**<br>**METEOR**<br>**SBERT Similarity**|
|---|---|
|LoCoMo<br>MemGPT|2.55<br>3.13<br>2.75<br>0.90<br>1.64<br>15.76<br>1.18<br>1.07<br>0.96<br>0.42<br>0.95<br>8.54|
|**A-MEM**|**3.45**<br>**3.37**<br>**3.54**<br>**3.60**<br>**2.05**<br>**19.51**|



**Table 3:** An ablation study was conducted to evaluate our proposed method against the GPT-4o-mini base model. The notation ’w/o’ indicates experiments where specific modules were removed. The abbreviations LG and ME denote the link generation module and memory evolution module, respectively. 

|**Category**|**Category**|**Category**|**Category**|**Category**|**Category**|
|---|---|---|---|---|---|
|**Method**|**Multi Hop**<br>**F1**<br>**BLEU-1**|**Temporal**<br>**F1**<br>**BLEU-1**|**Open Domain**<br>**F1**<br>**BLEU-1**|**Single Hop**<br>**F1**<br>**BLEU-1**|**Adversial**<br>**F1**<br>**BLEU-1**|
|w/o LG & ME<br>w/o ME|9.65<br>7.09<br>21.35<br>15.13|24.55<br>19.48<br>31.24<br>27.31|7.77<br>6.70<br>10.13<br>10.85|13.28<br>10.30<br>39.17<br>34.70|15.32<br>18.02<br>44.16<br>45.33|
|**A-MEM**|**27.02**<br>**20.09**|**45.85**<br>**36.67**|**12.14**<br>**12.00**|**44.65**<br>**37.06**|**50.03**<br>**49.47**|



score of 3.45 (a 35% improvement over LoCoMo’s 2.55 and 192% higher than MemGPT’s 1.18). The effectiveness of A-MEM stems from its novel agentic memory architecture that enables dynamic and structured memory management. Unlike traditional approaches that use static memory operations, our system creates interconnected memory networks through atomic notes with rich contextual descriptions, enabling more effective multi-hop reasoning. The system’s ability to dynamically establish connections between memories based on shared attributes and continuously update existing memory descriptions with new contextual information allows it to better capture and utilize the relationships between different pieces of information. 

**Cost-Efficiency Analysis.** A-MEM demonstrates significant computational and cost efficiency alongside strong performance. The system requires approximately 1,200 tokens per memory operation, achieving an 85-93% reduction in token usage compared to baseline methods (LoCoMo and MemGPT with 16,900 tokens) through our selective top-k retrieval mechanism. This substantial token reduction directly translates to lower operational costs, with each memory operation costing less than $0.0003 when using commercial API services—making large-scale deployments economically viable. Processing times average 5.4 seconds using GPT-4o-mini and only 1.1 seconds with locally-hosted Llama 3.2 1B on a single GPU. Despite requiring multiple LLM calls during memory processing, A-MEM maintains this cost-effective resource utilization while consistently outperforming baseline approaches across all foundation models tested, particularly doubling performance on complex multi-hop reasoning tasks. This balance of low computational cost and superior reasoning capability highlights A-MEM’s practical advantage for deployment in the real world. 

## **4.4 Ablation Study** 

To evaluate the effectiveness of the Link Generation (LG) and Memory Evolution (ME) modules, we conduct the ablation study by systematically removing key components of our model. When both LG and ME modules are removed, the system exhibits substantial performance degradation, particularly in Multi Hop reasoning and Open Domain tasks. The system with only LG active (w/o ME) shows intermediate performance levels, maintaining significantly better results than the version without both modules, which demonstrates the fundamental importance of link generation in establishing memory connections. Our full model, A-MEM, consistently achieves the best performance across all evaluation categories, with particularly strong results in complex reasoning tasks. These results reveal that while the link generation module serves as a critical foundation for memory organization, the memory evolution module provides essential refinements to the memory structure. The ablation study validates our architectural design choices and highlights the complementary nature of these two modules in creating an effective memory system. 

## **4.5 Hyperparameter Analysis** 

We conducted extensive experiments to analyze the impact of the memory retrieval parameter k, which controls the number of relevant memories retrieved for each interaction. As shown in Figure 3, we evaluated performance across different k values (10, 20, 30, 40, 50) on five categories of tasks using GPT-4o-mini as our base model. The results reveal an interesting pattern: while increasing k generally leads to improved performance, this improvement gradually plateaus and sometimes slightly decreases at higher values. This trend is particularly evident in Multi Hop and Open Domain 

7 

**==> picture [334 x 165] intentionally omitted <==**

**----- Start of picture text -----**<br>
27.525.0 F1 BLEU-125.87 26.97 27.02 26.81 47.545.0 43.60 F1BLEU-145.08 45.22 45.85 45.60 1412 F1BLEU-1 12.24 12.1412.00<br>22.520.0 19.91 19.45 20.19 20.09 20.15 42.540.0 10 10.299.61 10.57 10.359.76<br>17.515.0 14.36 37.5 35.53 35.85 36.44 36.67 35.76 8 7.38 7.03<br>35.0 6<br>12.5<br>10 20 30 40 50 10 20 30 40 50 10 20 30 40 50<br>k values k values k values<br>(a)  Multi Hop (b)  Temporal (c)  Open Domain<br>45 F1 BLEU-1 41.55 44.55 50 F1 BLEU-1 50.03 49.47 47.7647.24<br>40 38.15 37.02 45 43.86 43.19<br>35 33.67 34.32 40 39.11 38.35<br>31.15 32.12<br>30 28.31 35<br>25 25.43 30 30.29 29.49<br>10 20 30 40 50 10 20 30 40 50<br>k values k values<br>(d)  Single Hop (e)  Adversarial<br>**----- End of picture text -----**<br>


**Figure 3:** Impact of memory retrieval parameter k across different task categories with GPT-4o-mini as the base model. While larger k values generally improve performance by providing richer historical context, the gains diminish beyond certain thresholds, suggesting a trade-off between context richness and effective information processing. This pattern is consistent across all evaluation categories, indicating the importance of balanced context retrieval for optimal performance. 

**Table 4:** Comparison of memory usage and retrieval time across different memory methods and scales. 

|**Memory Size**|**Method**|**Memory Usage (MB)**|**Retrieval Time (**_µ_**s)**|
|---|---|---|---|
|1,000|A-MEM<br>MemoryBank [39]<br>ReadAgent [17]|1.46<br>1.46<br>1.46|0.31±0.30<br>0.24±0.20<br>43.62±8.47|
|10,000|A-MEM<br>MemoryBank [39]<br>ReadAgent [17]|14.65<br>14.65<br>14.65|0.38±0.25<br>0.26±0.13<br>484.45±93.86|
|100,000|A-MEM<br>MemoryBank [39]<br>ReadAgent [17]|146.48<br>146.48<br>146.48|1.40±0.49<br>0.78±0.26<br>6,682.22±111.63|
|1,000,000|A-MEM<br>MemoryBank [39]<br>ReadAgent [17]|1464.84<br>1464.84<br>1464.84|3.70±0.74<br>1.91±0.31<br>120,069.68±1,673.39|



tasks. The observation suggests a delicate balance in memory retrieval - while larger k values provide richer historical context for reasoning, they may also introduce noise and challenge the model’s capacity to process longer sequences effectively. Our analysis indicates that moderate k values strike an optimal balance between context richness and information processing efficiency. 

## **4.6 Scaling Analysis** 

To evaluate storage costs with accumulating memory, we examined the relationship between storage size and retrieval time across our A-MEM system and two baseline approaches: MemoryBank [39] and ReadAgent [17]. We evaluated these three memory systems with identical memory content across four scale points, increasing the number of entries by a factor of 10 at each step (from 1,000 to 10,000, 100,000, and finally 1,000,000 entries). The experimental results reveal key insights about our A-MEM system’s scaling properties: In terms of space complexity, all three systems exhibit identical linear memory usage scaling ( _O_ ( _N_ )), as expected for vector-based retrieval systems. This confirms that A-MEM introduces no additional storage overhead compared to baseline approaches. For retrieval time, A-MEM demonstrates excellent efficiency with minimal increases as memory size grows. Even when scaling to 1 million memories, A-MEM’s retrieval time increases only from 0.31 _µ_ s to 3.70 _µ_ s, representing exceptional performance. While MemoryBank shows slightly faster retrieval times, A-MEM maintains comparable performance while providing richer memory representations and functionality. Based on our space complexity and retrieval time analysis, we conclude that A-MEM’s retrieval mechanisms maintain excellent efficiency even at large scales. The minimal growth in retrieval time across memory sizes addresses concerns about efficiency in large-scale memory systems, demonstrating that A-MEM provides a highly scalable solution for long-term conversation management. This unique combination of efficiency, scalability, and enhanced memory capabilities positions A-MEM as a significant advancement in building powerful and long-term memory mechanism for LLM Agents. 

8 

**==> picture [307 x 148] intentionally omitted <==**

**----- Start of picture text -----**<br>
A-mem 30 A-mem<br>Base Base<br>20<br>20<br>10<br>10<br>0<br>0<br>−10 −10<br>−20 −20<br>−20 −10 0 10 20 −20 −10 0 10 20<br>(a)  Dialogue 1 (b)  Dialogue 2<br>**----- End of picture text -----**<br>


**Figure 4:** T-SNE Visualization of Memory Embeddings Showing More Organized Distribution with A-MEM (blue) Compared to Base Memory (red) Across Different Dialogues. Base Memory represents A-MEM without link generation and memory evolution. 

## **4.7 Memory Analysis** 

We present the t-SNE visualization in Figure 4 of memory embeddings to demonstrate the structural advantages of our agentic memory system. Analyzing two dialogues sampled from long-term conversations in LoCoMo [22], we observe that A-MEM (shown in blue) consistently exhibits more coherent clustering patterns compared to the baseline system (shown in red). This structural organization is particularly evident in Dialogue 2, where well-defined clusters emerge in the central region, providing empirical evidence for the effectiveness of our memory evolution mechanism and contextual description generation. In contrast, the baseline memory embeddings display a more dispersed distribution, demonstrating that memories lack structural organization without our link generation and memory evolution components. These visualization results validate that A-MEM can autonomously maintain meaningful memory structures through dynamic evolution and linking mechanisms. More results can be seen in Appendix A.4. 

## **5 Conclusions** 

In this work, we introduced A-MEM, a novel agentic memory system that enables LLM agents to dynamically organize and evolve their memories without relying on predefined structures. Drawing inspiration from the Zettelkasten method, our system creates an interconnected knowledge network through dynamic indexing and linking mechanisms that adapt to diverse real-world tasks. The system’s core architecture features autonomous generation of contextual descriptions for new memories and intelligent establishment of connections with existing memories based on shared attributes. Furthermore, our approach enables continuous evolution of historical memories by incorporating new experiences and developing higher-order attributes through ongoing interactions. Through extensive empirical evaluation across six foundation models, we demonstrated that A-MEM achieves superior performance compared to existing state-of-the-art baselines in long-term conversational tasks. Visualization analysis further validates the effectiveness of our memory organization approach. These results suggest that agentic memory systems can significantly enhance LLM agents’ ability to utilize long-term knowledge in complex environments. 

## **6 Limitations** 

While our agentic memory system achieves promising results, we acknowledge several areas for potential future exploration. First, although our system dynamically organizes memories, the quality of these organizations may still be influenced by the inherent capabilities of the underlying language models. Different LLMs might generate slightly different contextual descriptions or establish varying connections between memories. Additionally, while our current implementation focuses on text-based interactions, future work could explore extending the system to handle multimodal information, such as images or audio, which could provide richer contextual representations. 

9