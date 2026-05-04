## **Thinking with Visual Primitives** 

> [‡] Ruijie Lu[1,2,][∗] , Yiyang Ma[1,][∗] , **Xiaokang Chen**[1,][∗][,] , Lingxiao Luo[1,3,][∗] , Zhiyu Wu[1,][∗] , Zizheng Pan[1,][∗] , Xingchao Liu[1,][∗] , Yutong Lin[1] , Hao Li[1] , Wen Liu[1] , Zhewen Hao[1] , Xi Gao[1] , Shaoheng Nie[1] , Yixuan Wei[1] , Zhenda Xie[1] , Ting Chen[3] , Gang Zeng[2] 

> 1 **DeepSeek-AI** 2 **Peking University** 3 **Tsinghua University** ∗ **: Core contributors** ‡ **: Project lead** 

## **Abstract** 

Despite the remarkable progress in Multimodal Large Language Models (MLLMs), the prevailing Chain-of-Thought (CoT) paradigms remain predominantly confined to the linguistic space. While recent advancements have focused on bridging the **Perception Gap** through highresolution cropping (e.g., Thinking with Images), they overlook a more fundamental bottleneck: the **Reference Gap** . The inherent ambiguity of natural language often fails to provide precise, unambiguous pointers to complex spatial layouts, leading to logical collapse in tasks requiring rigorous grounding. In this work, we introduce **Thinking with Visual Primitives** , a novel reasoning framework that elevates spatial markers—such as points and bounding boxes—to “minimal units of thought”. By interleaving these visual primitives directly into the thinking process, our model can “point” while it “reasons”, effectively grounding its cognitive trajectory in the physical coordinates of the image. Notably, our framework is built on a highly optimized architecture with extreme visual token efficiency. Despite its compact model scale and significantly lower image-token budget, our model achieves frontier-competitive performance on a focused suite of challenging visual QA tasks, matching or exceeding models such as GPT-5.4, Claude-Sonnet-4.6, and Gemini-3-Flash. This demonstrates a path toward more efficient and scalable System-2-like multimodal intelligence. 

## **1. Introduction** 

The convergence of Large Language Models (LLMs) and computer vision has ushered in an era of Multimodal Large Language Models (MLLMs) capable of sophisticated scene understanding. However, as we push these models toward complex reasoning, often conceptualized as Daniel Kahneman’s “System 2” thinking [23], a fundamental limitation of the current paradigm emerges. While the internal reasoning of these models, typically manifested as Chain-of-Thought (CoT), has become increasingly robust in the linguistic domain, it remains largely disjointed from the visual domain. 

Recent efforts to enhance multimodal reasoning, such as the visual-scaling strategies seen in frontier models [8, 12, 21, 24, 26, 34], have primarily addressed the Perception Gap. By employing high-resolution cropping and dynamic patching, these models ensure they “see” the fine-grained details of an image. Yet, “seeing” is not “reasoning”. Even with perfect perception, MLLMs frequently suffer from logical collapse in tasks involving complex spatial layouts or dense object interactions. We identify this failure as the Reference Gap: the inherent inability of 

**==> picture [456 x 214] intentionally omitted <==**

**----- Start of picture text -----**<br>
KV Cache Entries Tokens Before Compression<br>Gemma-4-31B ~289 69.7%<br>Ours-284B-A13B ~361 (~90 entries in KV Cache) 77.2%<br>Qwen3-VL-235B-A22B ~660 68.1%<br>GPT-5.4 ~740 71.1%<br>Claude-Sonnet-4.6 ~870 65.3%<br>Gemini-3-Flash ~1100 76.5%<br>0 200 400 600 800 1000 1200 1400 40 50 60 70 80<br>(a) Token Efficiency (b) Average Score on Selected Benchmarks<br>(  Lower is Better) (  Higher is Better)<br>**----- End of picture text -----**<br>


Figure 1 | **(a) Token consumption across various models for an** 800 × 800 **resolution image. (b) Average performance across** 7 **benchmarks—including counting and spatial reasoning—with in-house benchmarks excluded** . We note that the reported scores cover only a subset of evaluation dimensions that are directly relevant to the research focus of this paper, and are therefore not indicative of the models’ overall capabilities. For an 800 × 800 input, our model retains only approximately 90 entries in the KV cache, delivering competitive performance through a highly efficient compression strategy. Best viewed on screen. 

natural language to serve as a precise, unambiguous pointer within a continuous visual space. In scenarios like dense counting or multi-step spatial deduction, the model’s linguistic “thoughts” lose track of the visual entities they intend to reference, leading to cascading hallucinations. While recent works [6, 27, 32] have explored integrating bounding boxes into the chain-ofthought process, they primarily treat grounding as a post-hoc verification mechanism to enhance perception-heavy tasks. These approaches are often confined to high-resolution benchmarks where the challenge is “seeing” rather than “reasoning”, and their reliance on labor-intensive supervision further limits scalability. More importantly, they fail to address the Reference Gap in complex structural reasoning—such as topological navigation—where visual markers must function as the intrinsic medium of thought rather than merely verifiable evidence. 

In this work, we propose a paradigm shift: **Thinking with Visual Primitives** . We move beyond treating visual grounding as a secondary task or a final output. We elevate spatial markers—points and bounding boxes—to “minimal units of thought” that are interleaved directly into the model’s reasoning trajectory. This mechanism draws inspiration from human cognitive processes. When navigating a complex maze or counting a dense collection of objects, humans naturally employ deictic pointers—such as finger gestures—to reduce cognitive load and maintain logical consistency. By interleaving visual primitives into the thinking process, our model mimics this “point-to-reason” synergy, effectively anchoring abstract linguistic thoughts onto concrete spatial coordinates. 

Furthermore, our framework is built upon an architecturally efficient foundation [3] designed for high-throughput, long-context multimodal interactions. Unlike traditional approaches that rely on massive visual token sequences to compensate for visual deficiencies, our model leverages Compressed Sparse Attention [3] that compress the Key-Value (KV) cache of every _𝑚_ 

2 

**==> picture [455 x 173] intentionally omitted <==**

**----- Start of picture text -----**<br>
Language Response with Visual Primitives<br>Pretraining<br>De-Tokenizer<br>Specialized SFT<br>DeepSeek-V4-Flash Specialized RL<br>Unified RFT<br>DeepSeek-ViT Text Tokenizer<br>Image Language Instruct On-Policy Distillation<br>(a) Architecture (b) Training Pipeline<br>**----- End of picture text -----**<br>


Figure 2 | **Model architecture and training pipeline.** Developed upon the DeepSeek-V4-Flash [3], our model acquires foundational visual primitive generation capabilities during the pretraining phase. This is followed by a post-training stage employing an expert-wise specialization and consolidation paradigm. Best viewed on screen. 

visual token into one entry. This design allows the model to operate with only a fraction of the visual tokens used by other frontier systems, while maintaining comparable cognitive depth. 

Through extensive benchmarking, we demonstrate that Thinking with Visual Primitives delivers a significant leap in reasoning accuracy. Our model achieves competitive performance, standing on par with or surpassing the latest iterations of GPT, Claude, and Gemini across a wide spectrum of challenging spatial reasoning and visual QA tasks (seeing Fig. 1). Our findings suggest that the future of multimodal intelligence lies not just in seeing more pixels, but in developing more precise and less ambiguous referential mechanisms that bridges the gap between language and the visual world. 

## **2. Method** 

## **2.1. Overview** 

This section first introduces the model architecture. Next, we elaborate on the training pipeline, as illustrated in Fig. 2, and describe the corresponding data used across the pretraining and post-training phases. 

## **2.2. Architecture** 

Our model adopts a standard architecture similar to LLaVA [18, 19]. Specifically, input images are processed by a Vision Transformer (ViT) to extract visual features, which are then concatenated with language instructions to form an interleaved sequence of vision-language tokens. This sequence is subsequently fed into the Large Language Model (LLM) to generate responses. The language backbone is instantiated with DeepSeek-V4-Flash [3], a Mixture-of-Experts (MoE) model comprising 284B total parameters and 13B active parameters during inference. 

For visual encoding, we employ DeepSeek-ViT, an in-house ViT trained from scratch that supports arbitrary-resolution inputs. It first partitions the input image using a 14 × 14 patch size to generate patch tokens. Subsequently, at the ViT output, we apply a 3 × 3 spatial token 

3 

compression (compressing every 9 adjacent patch tokens into a single token along the channel dimension). Furthermore, leveraging the Compressed Sparse Attention (CSA) mechanism integrated within the base LLM, the visual tokens stored in the Key-Value (KV) cache are further compressed by a factor of 4. 

To illustrate this pipeline, consider an input image of 756 × 756 resolution comprising 571, 536 pixels. The patch embedding layer processes this into 2, 916 image patch tokens for the ViT. Following the 3 × 3 compression, only 324[1] visual tokens are fed into the LLM during the prefilling stage. Ultimately, the CSA mechanism reduces this to a mere 81 visual KV entries in the KV cache. Throughout this entire process, from raw pixels to the final KV cache entries, the ×. system achieves an overall compression ratio of 7, 056 

## **2.3. Pretraining** 

## _**2.3.1. Definition of Visual Primitives**_ 

During the pretraining phase, our objective is to equip the model with the fundamental capability to output “visual primitives”. We identify two standard output formats in computer vision as primitives: bounding boxes and points. Both representations fulfill the crucial role of spatial referencing. However, they exhibit distinct functional advantages: bounding boxes are adept at capturing the exact location and scale of specific objects, while points are more appropriate for abstract visual referencing, such as tracking motion trajectories or solving topological reasoning problems. 

## _**2.3.2. Motivation for Large-Scale Data Curation**_ 

While existing public datasets, such as COCO [17] and Pixmo-Points [4], provide relatively accurate box or point annotations, they suffer from insufficient scale and a notable lack of diversity. To ensure the generalizability of our “Thinking with Visual Primitives” paradigm, it is imperative to curate large-scale web data with rich semantics and high diversity. We prioritize the extensive scaling of bounding box data for the following reasons: 

- **Determinism of Annotations** : A bounding box tightly encloses an object, making its annotation relatively deterministic. Conversely, point annotations are highly ambiguous; any coordinate within the object’s boundaries can serve as a valid reference, leading to the absence of a strict ground truth. In extreme scenarios involving occlusion, a point intended for a background object might fall onto a foreground occluder, resulting in significant ambiguity. 

- **Task Generalizability** : A model trained to output bounding boxes can effortlessly generalize to point-based formats. Since a bounding box can be defined by two points (the top-left and bottom-right coordinates), it inherently encompasses the point representation. 

- **Information Richness** : Bounding boxes support a broader range of downstream tasks compared to points. While a point merely provides spatial localization, a bounding box encapsulates detailed geometric information (e.g., width and height). This additional context enables the model to perform more complex reasoning within the “Thinking with Visual Primitives” framework. 

> 1To balance performance and computational cost, we restrict the number of visual tokens output by the ViT to a range of 81 to 384. Images with resolutions falling outside this range are resized while preserving their original aspect ratio. 

4 

## _**2.3.3. Large-Scale Web Data Construction**_ 

**Raw Data Acquisition** . We acquire a massive volume of internet data related to box grounding by conducting large-scale web scraping across multiple websites. Taking Huggingface as an example, we utilize its official API to filter task data tagged with “Object Detection” or “Grounding”. We perform an initial screening based on popularity metrics (e.g., rankings by likes and downloads) and rigorously exclude all validation and test splits to prevent potential data contamination (i.e., data leakage) during model evaluation. Furthermore, we employ an LLM-based agent to parse the `README.md` files of these repositories, automatically converting the diverse dataset structures into our predefined, unified storage format. Following extensive crawling and deduplication across these website, we ultimately curate 97, 984 box-groundingrelated data sources. Manual inspection of the sampled data reveals a highly diverse range of object categories, spanning from common targets (e.g., people, faces) to domain-specific entities (e.g., lesion regions in CT scans or specific anime characters). However, these raw box annotations still suffer from various issues, such as semantic ambiguity and geometric inaccuracy, necessitating further rigorous filtering. We design a two-step filtering pipeline, as illustrated below. 

**Step I: Semantic-based Review** . Given that the directly crawled datasets are replete with noisy labels unsuitable for vision-language alignment training, we introduce an automated, MLLM-driven semantic review mechanism. While traditional data filtering primarily focuses on the geometric accuracy of bounding boxes, this stage aims to ensure the validity of the semantic label. Specifically, this review process focuses on eliminating three categories of fatal semantic defects: 

- **Meaningless Machine Codes and Gibberish** : Many raw datasets retain internal development codes (e.g., pure numeric classes like “0” or “1”). Since these labels lack humanreadable natural language semantics, forcing the model to learn such mappings would severely degrade its language generation capabilities. Thus, they are directly discarded. 

- **Ungeneralizable Private Entities** : Certain datasets employ specific personal pronouns (e.g., “MyRoommate”) or private identifiers (e.g., “ID_Card_1”). Because an MLLM cannot generalize the visual features of a non-public figure from isolated samples (i.e., the visual features of “someone” cannot generalize to a universal concept), such data are strictly filtered out. Conversely, widely recognized celebrities or public figures are retained. 

- **Ambiguous Abbreviations and Subjective Evaluations** : Labels common in specific domains (e.g., industrial inspection), such as “OK” or “NG” (Not Good), often lack concrete visual descriptiveness. For instance, a bare “OK” label introduces extreme semantic ambiguity, as an “intact apple” and an “intact circuit board” share absolutely no visual correlation. 

For each dataset, we sample three images and prompt the model to calculate a quality score (ranging from 0 to 10) based on the aforementioned criteria. The model then outputs a definitive “KEEP” or “DISCARD” decision, accompanied by a clear justification. This review stage retains 43, 141 out of the initial 97, 984 data sources, which are subsequently advanced to the next filtering phase. 

**Step II: Visual-Geometric Quality Review** . We further evaluate the geometric quality and annotation completeness of the bounding boxes to ensure the model learns precise region-text alignments. This process specifically targets three types of structural annotation defects: 

- **Severe Missing Annotations (Low Recall)** : This refers to scenarios where multiple in- 

5 

stances corresponding to the given label exist in the image, but only a few are annotated. If a large-scale missing annotation issue (e.g., a miss rate >50%) is detected during sampling, the dataset is immediately discarded. 

- **Severe Truncation and Offset** : This occurs when bounding boxes fail to reasonably enclose the target objects. In practice, we adopt a differential tolerance strategy: slightly loose boxes (incorporating minor background noise) are deemed acceptable; however, severe truncations that slice through critical visual features of the object (e.g., cutting off a head or wheels) are strictly unacceptable. 

- **Mega Boxes Issue** : If a bounding box meaninglessly covers over 90% of the image area, it is typically indicative of image classification data being forcibly converted into detection data. If this occurs only occasionally within a sampled batch, it is treated as acceptable noise. However, if such global boxes consistently appear across all three sampled images, the dataset is considered to lack meaningful localization information and is discarded. 

This review stage further retained 31, 701 out of the 43, 141 remaining data sources. To achieve dataset balance, we design a category-based sampling strategy. For each category within every dataset, we randomly sample _𝑁_ images associated with that class (if the total number of available images for a category is less than _𝑁_ , all are retained). Since a single image may simultaneously belong to multiple categories, we perform global deduplication on the aggregated set after the per-category selection. In practice, we set _𝑁_ = 1, 000, ultimately yielding over 40 million high-quality samples. 

## _**2.3.4. Unified Pretraining**_ 

For general multimodal data, we predominantly utilize large-scale web-crawled data rather than synthetic data generated via model distillation (e.g., synthetic image caption). The raw data undergoes careful curation, and we refrain from utilizing LLMs to rewrite the data content. Regarding the specialized data designed to equip the model with foundational capabilities to output visual primitives, in addition to the aforementioned web crawling and filtering, we also incorporate several high-quality public datasets, such as [4, 15, 17, 25, 29, 33]. We establish a unified formatting standard for both box grounding and point data. For box grounding tasks, we devise several prompt templates, such as “ _Locate_ _`TARGET` in this image and report its bounding box coordinates._ ”, where `TARGET` serves as a placeholder for the queried object. The corresponding response format is formulated as follows: `<|ref|>TARGET<|/ref|><|box|>[[x1,y1,x2,y2],[x3,y3, x4,y4]...]<|/box|>` , where `<|ref|>` , `<|/ref|>` , `<|box|>` , and `<|/box|>` are special tokens within the vocabulary. `x1,y1` and `x2,y2` denote the top-left and bottom-right coordinates of the bounding box. These coordinates are normalized to discrete integers ranging from 0 to 999. In scenarios with multiple instances, the bounding boxes are ordered from left to right. Similarly, for point tasks, we design prompt templates such as “ _Help me find_ ” _`TARGET` . Give me the center point for each instance._ The expected response format is defined as: `<|point|>[[x1,y1],[x2,y2]...]<|/point|>` , where `<|point|>` and `<|/point|>` are special tokens, and `x1,y1` denotes the point coordinates. Notably, in contrast to the box grounding format, the response paradigm for point tasks does not require outputting the object name. This design choice aims to extend point-based representations to more abstract concepts, such as utilizing a sequence of points to denote a trajectory. Ultimately, the whole pretraining phase consumes trillions of multimodal tokens. 

6 

## **2.4. Task Design & Cold-Start Data** 

**Cold-Start Data for Post-Training** . While pretraining equips the model with general multimodal priors and basic visual primitive capabilities, post-training (Specialized SFT/RL and the subsequent unified RFT) needs a small but high-precision cold-start dataset to bootstrap instruction following and reward learning under our visual primitive output interface. Concretely, we construct cold-start data with (i) explicit supervision targets derived from annotations (e.g., boxes/points) or programmatically generated, (ii) automatic verifiers (e.g., rule-based checkers) whenever possible to reduce label noise. We selected representative tasks that benefit from visual primitive-based reasoning (via boxes or points), and designed our cold-start data across four key dimensions: counting, spatial reasoning & general visual QA, maze navigation, and path tracing. 

## _**2.4.1. Counting**_ 

Multimodal Large Language Models consistently struggle with accurate counting, particularly in dense scenes. Unlike humans, who typically employ a systematic scanning-and-accumulation strategy, language-based models often fail to establish precise object correspondences when the object count is high. We address this fundamental bottleneck by employing bounding boxes as visual primitives to provide explicit referential anchors. 

**Task Decomposition.** We categorize counting tasks into two types: _Coarse-grained Counting_ and _Fine-grained Counting_ . The former focuses on counting general categories (e.g., “dogs”), while the latter requires distinguishing objects based on specific attributes or spatial constraints (e.g., “white dogs” or “the dog on the left”). 

**Coarse-Grained Counting.** We aggregate data from multiple dense detection datasets, including [2, 9, 14, 22, 28, 29, 35]. To ensure data quality, we implement a filtering process based on three primary criteria: avoiding excessive object density, ensuring bounding boxes are sufficiently large for clear identification, and maintaining a high recall rate for ground-truth box annotations. For the filtered samples, we prompt an MLLM to generate thinking content and concise final response based on the images and box annotations. The thinking content generation follows a structured three-step protocol: (1) _Intent Analysis_ , where the model identifies the target category; (2) _Batch Grounding_ , where the model utilizes visual primitives to locate all candidate objects simultaneously (we found that batch grounding is more efficient for coarse-grained tasks as it leverages the model’s inherent localization strengths while preventing repetitive enumeration); and (3) _Statistical Summation_ , based on the visual primitives. To eliminate noise during cold-start training, we implement a strict verification mechanism to ensure that all box visual primitives in the thinking content strictly align with the metadata coordinates, follow the predefined syntax, and match the final numerical count. 

**Fine-Grained Counting.** Due to the scarcity of publicly available datasets specifically for fine-grained counting, we developed a specialized data construction pipeline. (1) _Question Generation_ : Leveraging the images and scene-graph metadata from GQA [10], we prompt an MLLM to curate informative fine-grained counting questions. Samples that fail to yield meaningful questions are discarded. For each valid sample, we record the ground-truth object IDs, the IDs of excluded negative candidates, and the underlying rationale for the QA pair construction. (2) _Thinking Content Synthesis_ : Using the image, scene graph, and the previously generated question (along with its associated IDs and rationale) as input, we guide the MLLM to synthesize a reasoning chain integrated with visual primitives. While the overall thinking structure mirrors that of coarse-grained counting, the model is explicitly instructed to perform a 

7 

Coarse-grained Counting [Trigger_Placeholder] **Thinking with Visual Primitives** Count the number of men in this image. 1. **Deconstructing the query** The user wants me to count the total number of men in the image. This looks like a team photo. I need to count everyone present: the players in the team kit (striped jerseys) as well as the coaches or officials standing on the sides in regular clothes. 2. **Sweeping the photo for men** Looking at the group, I’ll identify all the men across the different rows. Here they are: <｜ref｜>the men<｜/ref｜><｜box ｜ >[[13,228,116,714],[106,226,202,707],[129,357,229,711],[154,519,255,747], [203,248,284,511],[222,377,310,609],[246,508,374,728],[277,239,363,506],[318,382,402,727],[359,219,438,499],[3 93,382,492,742],[429,230,505,499],[475,386,578,730],[500,221,581,517],[565,392,643,732],[580,221,653,532],[625 ,509,733,755],[634,392,728,613],[656,230,728,509],[721,248,799,509],[730,407,827,730],[733,511,851,755],[789,2 Original Image 15,892,524],[812,405,910,744],[885,234,977,747]]<｜/box｜> This covers the players in the front sitting down, the middle row seated or crouching, the back row standing, and the four non-players (two on the far left, two on the far right) dressed in suits/sweaters. 3. **Tallying the group** Counting everyone up, there are 4 guys sitting on the ground in the front row, 9 players seated in the middle row, 8 players standing in the back row, plus 2 coaches/officials on the left and 2 on the right. 4 + 9 + 8 + 2 + 2 = 25. The total number is 25. **Response** Image with Visual Primitives There are 25 men in this image. Fine-grained Counting **Thinking with Visual Primitives** 1. **What am I looking for** The question asks me to count the bears that are on the ground. I need to scan the entire scene, locate all [Trigger_Placeholder] How many bears are on the ground? the bears, and check their positions to ensure I only count those standing on the earth or rocks, excluding any that might be elevated or climbing. 2. **Evaluating each bear's position** Starting my scan near the top of the image, I spot a <｜ref｜>bear<｜/ref｜><｜box｜>[[452,23,804,411]] <｜/box｜> clinging vertically to a tall, dead tree trunk. Because it is climbing and clearly not on the ground, I will exclude it from my count. Looking lower down towards the left side of the scene, I see a large <｜ref｜>bear<｜/ref｜><｜box｜>[[50,447,647,771]]<｜/box｜> walking along a rocky edge. This one is standing firmly on the terrain, so it meets the criteria. Scanning just below and to the right of that ledge, I find another <｜ref｜>bear<｜/ref｜><｜box｜>[[380,645,868,961]]<｜/box｜> positioned among some broken pieces of wood and dirt. This bear is also on the ground surface, so it counts as well. 3. **Tally of bears on the ground** Out of the three bears visible in the image, one is high up in a tree and two are walking on the rocky ground Original Image Image with Visual Primitives below. The total count is 2. **Response** Based on the image, there are 2 bears on the ground. 

Figure 3 | **Illustrative examples of cold-start data for coarse-grained and fine-grained counting** . The model performs intent decomposition and utilizes visual primitives to anchor all pertinent entities, followed by a systematic counting procedure grounded in the visual domain. Best viewed on screen with zoom. 

_sequential scan_ —systematically identifying and verifying each possible object in the scene against the specified fine-grained constraints. We also applied this methodology to construct negative samples where the ground-truth count is zero, thereby enhancing the model’s robustness against hallucinations. 

In total, we have approximately 10, 000 cold-start samples for counting task. Examples could be seen in Fig. 3. 

## _**2.4.2. Spatial Reasoning and General Visual QA**_ 

We consolidate spatial reasoning and general VQA into a unified category. This integration effectively mitigates the referential ambiguity and semantic drift inherent in purely linguistic descriptions. In constructing our cold-start data, we prioritize spatial reasoning tasks, under the hypothesis that the capability to think with visual primitives developed here will naturally generalize to broader VQA scenarios. Our data curation covers both natural and synthetic environments. 

**Data Construction in Natural Scenes.** Utilizing the images and scene graphs from GQA [10], 

8 

**==> picture [442 x 248] intentionally omitted <==**

**----- Start of picture text -----**<br>
Spatial Reasoning<br>Thinking with Visual Primitives<br>1. **Analyzing the request**<br>The user is asking to verify if there is a "purple rubber object" that shares the same size as the "gray metallic<br>[Trigger_Placeholder] object". This requires a multi-step attribute verification and comparison: first, I must locate the gray<br>Is there a purple rubber object that has  metallic object to determine its relative size in the scene. Second, I need to scan the scene for any purple<br>the same size as the gray metallic object? rubber objects that match this specific size.<br>2. **Locating the gray metallic object**<br> Scanning the objects in the scene, I spot a gray, highly reflective spherical object in the background towards<br>the upper center-right. This is the <｜ref｜>small gray metal sphere<｜/ref｜><｜box｜>[[502,209,555,<br>288]] <｜/box｜>. Comparing it to the large cube, large sphere, and large cylinder, it is clearly categorized as<br>a small object.<br>3. **Searching for a matching purple rubber object**<br>Since the gray metallic object is small, I need to look for a small purple rubber object. I will systematically<br>check the other small objects in the scene:<br>- Right next to the gray sphere, there is a <｜ref｜>small brown metal cylinder<｜/ref｜><｜box｜<br>>[[462,255,519,367]]<｜/box｜>. It is brown and metallic.<br>- To the left, there is a <｜ref｜>small blue metal cube<｜/ref｜><｜box｜>[[301,300,386,429]]<｜/box｜>.<br>Original Image It is blue and metallic.<br>- Further right, in the mid-ground, there is a matte <｜ref｜>small blue rubber cube<｜/ref｜><｜box｜<br>>[[599,325,684,458]]<｜/box｜>. It is rubber, but its color is blue.<br>- Right beside that blue cube, there is a <｜ref｜>small yellow rubber cylinder<｜/ref｜><｜box｜<br>>[[647,281,711,397]]<｜/box｜>. It is yellow.<br>- On the far right edge of the cluster, there is a <｜ref｜>small blue metal sphere<｜/ref｜><｜box｜<br>>[[796,481,873,591]]<｜/box｜>. It is blue.<br>- Finally, in the lower left foreground, there is another matte <｜ref｜>small blue rubber cube<｜/ref｜<br>><｜box｜>[[321,600,413,773]]<｜/box｜>. This is also blue.<br>4. **Conclusion**<br>After examining all the small objects to match the size of the small gray metal sphere, I found blue, brown,<br>Image with Visual Primitives and yellow objects, but absolutely no purple ones. Therefore, no purple rubber object of the same size exists<br>in the scene.<br>**----- End of picture text -----**<br>


## **Response** 

**==> picture [241 x 8] intentionally omitted <==**

**----- Start of picture text -----**<br>
False, there is no purple rubber object that matches the size of the gray metallic object.<br>**----- End of picture text -----**<br>


Figure 4 | **Illustrative cold-start data for spatial reasoning** . The model performs intent decomposition and utilizes visual primitives to anchor all pertinent entities, facilitating sophisticated multi-hop logical inference. Best viewed on screen with zoom. 

we prompt an MLLM to design questions centered on spatial relations and object interactions, along with corresponding thinking content. The generated thinking content follows a structured process, including intent analysis, object grounding, and relational inference. To resolve potential ambiguities in crowded scenes, the model is instructed to select distinctive objects and apply multi-attribute constraints (e.g., combining actions and properties) to uniquely specify the target. However, due to the relatively simple relational structure in GQA, it remains challenging to generate complex, multi-hop reasoning samples at scale. To overcome this limitation and fully unlock the model’s potential, we further incorporate complex synthetic data. 

**Data Construction in Synthetic Scenes.** We leverage the CLEVR [13] toolchain to generate multihop reasoning data. This framework supports controllable scene generation with varying object densities, along with question generation and programmatic execution traces that map each reasoning step to object-level references (e.g., specific object IDs). To supervise the generation of visual primitives, we project 3D object coordinates onto 2D bounding boxes based on the official toolchain. Given the rendered images, scene graphs, questions, answers, and execution traces, we prompt the MLLM to synthesize “Thinking with Visual Primitives” chains, which include intent analysis, task decomposition, and multi-hop grounded reasoning. An example could be 4. seen in Fig. 

**Negative Sample Augmentation.** To enhance the model’s reliability, we construct negative training samples where the queried objects or relationships do not exist. In such cases, the model is trained to provide a “faithful refusal” based on the visual evidence rather than generating fabricated responses. 

In total, we generated 9, 000 cold-start samples for the spatial reasoning and general VQA 

9 

**==> picture [437 x 134] intentionally omitted <==**

**----- Start of picture text -----**<br>
Maze Navigation<br>[Trigger_Placeholder]<br>Is there a feasible<br>way to get from the<br>lime text label to the<br>tangerine circle?<br>Please draw the route<br>if any. Display<br>\boxed{True} at the<br>end if there is a path,<br>else display<br>\boxed{False}.<br>Original Image Image with Visual Primitives<br>Thinking with Visual Primitives<br>**----- End of picture text -----**<br>


I'll use a trial-and-error strategy to explore this maze. First locate the starting point: < ｜ point ｜ >[[309,512]]< ｜ /point ｜ >, and the destination: < ｜ point ｜ >[[652,644]]< ｜ /point ｜ >. 

**Start Exploring**: 

**Step1**: Reaching < ｜ point ｜ >[[357,369]]< ｜ /point ｜ > after going upper-right, I face 2 directions to choose from (lower-right, left).lower-right direction might also be correct, but I'll explore left direction first. **Step2**: Moving left: < ｜ point ｜ >[[357,369],[260,372]]< ｜ /point ｜ >...reaching < ｜ point ｜ >[[260,372]]< ｜ /point ｜ >, I see 2 forks here. Observing that upper-right direction is a straight dead end at < ｜ point ｜ >[[310,226]]< ｜ /point ｜ >, can see wall directly. Still need to explore left direction. **Step3**: Walking left: < ｜ point ｜ >[[260,372],[161,375]]< ｜ /point ｜ >...arriving at < ｜ point ｜ >[[161,375]]< ｜ /point ｜ >, this is another junction with 3 paths. Observing that upper-right direction is a straight dead end at < ｜ point ｜ >[[211,226]]< ｜ /point ｜ >, can see wall directly; upper-left direction to < ｜ point ｜ >[[113,226]]< ｜ /point ｜ > is obviously a dead end. Still need to explore lower-left direction. 

**Step4**: Moving lower-left: < ｜ point ｜ >[[161,375],[113,518],[211,515],[261,656],[162,658],[213,796]]< ｜ /point ｜ >...reaching < ｜ point ｜ >[[213,796]] < ｜ /point ｜ >, I see 2 forks here. Observing that right direction is a straight dead end at < ｜ point ｜ >[[310,791]]< ｜ /point ｜ >, can see wall directly; left direction is a straight dead end at < ｜ point ｜ >[[113,791]]< ｜ /point ｜ >, can see wall directly. All directions are obviously dead ends, need to backtrack. **Step5**: This path is blocked, backtracking to the previous junction < ｜ point ｜ >[[260,372]]< ｜ /point ｜ >, trying another direction. **Step6**: All paths in this branch are explored, all dead ends. Going back to main junction < ｜ point ｜ >[[357,369]]< ｜ /point ｜ >, taking lower-right direction. **Step7**: Reaching < ｜ point ｜ >[[357,369]]< ｜ /point ｜ > after going upper-right, I face 2 directions to choose from (lower-right, left).left direction was already explored, it's a dead end. **Step8**: Heading towards lower-right: < ｜ point ｜ >[[357,369],[408,510],[457,650]]< ｜ /point ｜ > 

**Step9**: The path is clear, going directly to the exit: < ｜ point ｜ >[[457,650],[554,648],[506,507],[603,507],[652,645]]< ｜ /point ｜ > **Step10**: Up ahead < ｜ point ｜ >[[652,645]]< ｜ /point ｜ > is the destination! Through systematic exploration, I find the correct path. 

- **Final Path**: After exploration, the correct route is: 

< ｜ point ｜ >[[310,513],[357,369],[408,510],[457,650],[554,648],[506,507],[603,507],[652,645]]< ｜ /point ｜ > 

Successfully reaching the destination: < ｜ point ｜ >[[652,644]] < ｜ /point ｜ >! 

## **Response** 

The maze is solvable. The verified path is: < ｜ point ｜ >[[310,513],[357,369],[408,510],[457,650],[554,648],[506,507],[603,507],[652,645]]< ｜ /point ｜ > [\boxed{True}] 

Figure 5 | **Example of cold-start data for the maze navigation task.** The model first identifies the start and end points, then explores possible paths in a DFS manner. Best viewed on screen with zoom. 

domain. 

## _**2.4.3. Maze Navigation**_ 

While MLLMs have shown proficiency in solving advanced scientific problems, a robust paradigm for topological reasoning remains elusive. Purely linguistic CoT struggles to accurately describe trajectories of irregular shapes. To address this gap, Thinking with Visual Primitives, which could employ points as cognitive units, is uniquely suited for such challenges. We first introduce a maze navigation task that requires the model to determine the solvability of a maze—a process that demands a fundamental understanding of spatial connectivity and reachability. We construct our cold-start data through synthetic data generation, the details of which are provided below. 

**Design Methodology.** We use Depth-First Search (DFS), Prim, and Kruskal algorithms to produce solvable and non-trivial mazes. All three algorithms generate challenging mazes where 

10 

only few paths exist between any two cells, ensuring solutions that cannot be trivially guessed. We design three maze topologies: rectangular grids, circular mazes composed of concentric rings with angular sectors, and hexagonal (honeycomb) lattices. To enhance model robustness, we additionally designed a series of unsolvable mazes. We first generate a solvable maze and obtain the solution paths, then deliberately place a few walls around the middle of that path—avoiding areas too close to the start or end. This breaks the connectivity in a less obvious way, making the maze appear solvable at first glance, but actually requiring a full search to confirm that no valid path exists. We apply diverse visual styles including gradient and extra-thick walls, varied background patterns, multiple marker types, and random small-angle rotations to prevent overfitting to specific visual patterns. Image resolutions are randomized, and aspect ratios are continuously sampled, with grid dimensions adjusted proportionally. 

**Difficulty Control.** The difficulty of maze navigation largely depends on how many visual reasoning steps the model needs to chain together. We control this by changing the grid size. As the grid becomes larger, the model has to parse more cells, track connectivity over longer distances, and deal with more dead ends that require backtracking. Each of these adds to the overall reasoning complexity. Concretely, easy mazes require the model to chain only a handful of local connectivity checks, while nightmare-level mazes demand sustained, longrange composition of hundreds of such primitive operations without losing track of previously explored regions. We enforce minimum resolution thresholds at each difficulty level to ensure that the visual primitives remain perceptible, even in the hardest configurations. This ensures that task difficulty stems from reasoning complexity rather than visual ambiguity. 

**Thinking Content Synthesis.** We design several natural language formats and templates to produce descriptions of the DFS-based exploration process, including forward exploration and backtracking. Each exploration step is grounded to the image via pointing coordinates, explicitly converting visual primitive operations—checking wall connectivity at a cell, advancing to an adjacent cell, or retreating from a dead end—into verbalized reasoning chains. This serves as the cold start supervision for teaching the model to think with visual primitives rather than merely perceive them. The final output indicates whether the maze is solvable and, if so, provides a verified solution path. 

In total, we generate 460, 000 cold-start samples with various difficulties for the task of Maze Navigation. An example is shown is Fig. 5. 

## _**2.4.4. Path Tracing**_ 

In addition to the maze navigation task, we further design a path tracing task to enhance the model’s ability to leverage visual primitives for reasoning across diverse scenarios. The task ask the model to follow a specified curve through a tangle of overlapping lines to identify the endpoint it reaches. We instantiate this task as line tracing through procedurally generated images of entangled curves, where each line connects a uniquely labeled start point to an endpoint. 

**Design Methodology.** We generate images which consist of multiple Bézier curves, each connecting a labeled start point to a labeled endpoint. The central challenge lies in intersection disambiguation: wherever two lines cross, the model must invoke a local geometric-continuity primitive to decide which branch continues the target curve. To ensure this primitive is genuinely tested, we carefully prevent any endpoint from overlapping with or being crossed by an unrelated line, discarding and regenerating configurations that violate these constraints. We further include a uniform-style mode in which every line shares the same color and stroke 

11 

**==> picture [455 x 208] intentionally omitted <==**

**----- Start of picture text -----**<br>
Path Tracing<br>Thinking with Visual Primitives<br>[Trigger_Placeholder]<br>I find the starting point you mentioned, it's located here:<br>Where does the crown icon connect to? Put the destination icon name in \boxed{}. < ｜ point ｜ >[[176,758]]< ｜ /point ｜ >.<br>Following this magenta line, the visual path I observe is:<br>< ｜ point ｜ ><br>[[176,759],[230,779],[294,799],[326,806],[361,806],[380,793],[386,780],[<br>387,753],[382,722],[375,699],[364,665],[356,643],[345,613],[337,587],[3<br>33,572],[330,555],[332,543],[336,539],[345,537],[359,539],[370,543],[38<br>8,551],[402,558],[424,569],[447,580],[463,587],[485,596],[506,604],[519,<br>607],[536,610],[546,609],[558,606],[568,600],[572,594],[577,582],[578,5<br>69],[578,559],[576,543],[573,527],[570,516],[565,502],[562,493],[558,48<br>2],[556,473],[555,470],[555,468],[557,469],[560,472],[566,480],[572,486]<br>,[581,499],[592,513],[600,523],[612,537],[624,551],[632,559],[641,568],[<br>649,573],[653,574],[655,570],[655,563],[652,548],[644,527],[638,509],[6<br>25,479],[611,444],[602,419],[588,381],[580,357],[573,325],[573,300],[57<br>9,289],[597,284],[632,295],[665,313],[732,356],[822,419]]< ｜ /point ｜ ><br>Following this path, it connects to:<br>< ｜ point ｜ > [[821,419]]< ｜ /point ｜ >.<br>Response<br>Original Image Image with Visual Primitives The line connects to the octopus icon.<br>[\boxed{octopus}]<br>**----- End of picture text -----**<br>


Figure 6 | **Example of cold-start data for the path tracing task.** The model identifies the start and end points, then traces the line using visual primitives. Best viewed on screen with zoom. 

width, stripping away color-based shortcuts and forcing the model to rely solely on curvature continuity at crossings—a direct test of whether the path-tracing primitive has been internalized rather than approximated by color matching. Difficulty scales naturally with the number of lines and their curvature amplitude: easy instances present a few gently curving lines with sparse crossings, while harder instances pack many tightly wound curves into the canvas, multiplying the intersection points at which the figure-ground primitive must be applied. Image resolutions, aspect ratios, and visual styles (color palettes, line styles, endpoint markers, backgrounds) are all randomized to prevent superficial pattern matching. 

**Thinking Content Synthesis.** We explicitly represent the path-tracing process as a sequence of coordinates sampled along the target curve, which reflects how the model attends to and follows the path across the image. The process starts by locating the queried start point, then follows the curve through a series of intermediate waypoints, and finally identifies the endpoint reached. Importantly, the density of these waypoints adapts to the local geometry of the curve. Straightforward segments are represented with fewer points, while highly curved regions or dense intersections are described with finer-grained coordinates, mirroring how a human would slow down and pay closer attention in visually complex regions. 

In total, we generate 125, 000 cold-start samples across different difficulty levels for the task of Path Tracing. An example is shown is Fig. 6. 

## **2.5. Post-Training Pipeline** 

To maximize the learning efficiency of the model for both box and point visual primitives, our post-training pipeline adopts a “train specialists—then—merge” strategy, which is detailed below. 

## _**2.5.1. Specialized SFT**_ 

In the Specialized SFT phase, the overall training data consists of 70% general multimodal and pure-text data, and 30% specialized “thinking with visual primitives” data. We conduct SFT 

12 

separately using the two types of cold-start data constructed in Section 2.3.4: box ( _thinking with grounding_ ) and point ( _thinking with pointing_ ). This separation prevents mode conflict when the volume of specialized data is relatively small. After this training phase, we obtain two specialized models, denoted as **F** TwG and **F** TwP. 

## _**2.5.2. Specialized RL**_ 

Subsequently, we apply Reinforcement Learning (RL) independently to both **F** TwG and **F** TwP. Following [3], we utilize the Group Relative Policy Optimization (GRPO) algorithm and follow their hyper-parameters. Given that the visual primitives (e.g., boxes and points) within the thinking content of our cold-start data have been rigorously verified, we do not explicitly supervise the visual primitives generated during the model’s thinking process in the RL phase. This design enhances the scalability of the RL training data. Consequently, we only require images, questions, and final answers when collecting RL data, which significantly broadens the scope of accessible data. 

During training, we design several Reward Models (RMs) to provide concurrent supervision for each task from three perspectives: format constraints, quality constraints, and accuracy constraints. The first two constraints are shared across different tasks, while the final accuracy constraint requires specific designs tailored to the task type. 

**Format RM** . This RM evaluates the output based on rules to generate a reward score ranging from 0 to 1. Specifically, it verifies whether the representation format of the visual primitives generated by the model is correct. For _thinking with grounding_ , this RM additionally checks for redundancy in the model’s output, such as generating duplicate bounding boxes; this effectively mitigates the issue of the SFT model falling into an infinite loop of box generation. 

**Quality RM** . This is an LLM-based Generative Reward Model (GRM). The Quality RM takes the thinking content and the final response generated by the model as inputs, and evaluates them from the following aspects: 

- Whether there is redundancy in the model’s response. 

- Whether the model’s thinking content is consistent with its final response. 

- Whether there are self-contradictions during the “thinking with visual primitives” process. 

- Whether the referred objects are meaningful entities when the model outputs visual primitives in the form of boxes. 

- Whether the model exhibits “reward hacking” behaviors, such as forcefully fabricating a fake ground truth identical to its own prediction in the response to deceive the reward model. 

Ultimately, the model outputs a score from three discrete tiers [0.0, 0.5, 1.0] and provides a rationale for the given score. Below, we will introduce the Accuracy RM specific to each task. 

**Accuracy RM for Counting** . To provide a smooth and informative learning signal, we design a rule-based counting reward model that captures the degree of deviation between the predicted and ground truth, instead of relying on binary exact-match supervision. Specifically, we apply a smooth exponential decay over the relative error, so that near-correct predictions are only lightly penalized, while larger mistakes receive significantly lower scores. The reward _𝑅_ is given as: 

**==> picture [301 x 28] intentionally omitted <==**

13 

where _𝑦_ ˆ and _𝑦_ denote the predicted and ground-truth counts, respectively. The normalization term | _𝑦_ | + 1 makes the reward depend on relative error, allowing small deviations to be more tolerable in scenes with larger object counts. The coefficients _𝛼_ and _𝛽_ control the overall reward scale and decay rate, respectively. In practice, we set _𝛼_ = 0.7 and _𝛽_ = 3, which are empirically chosen to provide stable and smooth learning signals. 

**Accuracy RM for Spatial Reasoning and General VQA** . For these tasks, we design an LLMbased GRM. We feed the model’s thinking content, its final response, the user query, and the ground-truth answer into the GRM to independently evaluate and score the thinking process and the response. The final reward is calculated as the average of the two scores. 

**Accuracy RM for Maze Navigation.** To encourage the model to explore the maze, we design a rule-based RM. The final reward is a weighted combination of the following components. 

- **Causal exploration progress.** We process the model’s step-by-step exploration sequentially. Upon encountering the first wall violation ( _i_ . _e_ ., where the model claims to move between two cells separated by a wall), we truncate all subsequent exploration, as it is causally invalidated. Then, we calculate the shortest distance between the explored regions and the endpoint. The score is 1 minus the fraction of the distance and the length of the ground-truth path. The score measures how much of the maze was legally explored before truncation relative to the optimal reachable region, rewarding thorough and correct exploration. This item is only applied for solvable mazes. For unsolvable mazes, this item is kept as 1. 

- **Exploration completeness.** For unsolvable mazes, the model must demonstrate that no path exists by exhaustively exploring the reachable region. The score is the fraction of the number of explored regions and the number of all the achievable regions. This component measures the fraction of truly reachable cells that appear in the model’s (causally valid) exploration trace. This item is only applied for unsolvable mazes. For solvable mazes, this item is kept as 1. 

- **Wall violation penalty.** Independent of the causal truncation above, we scan the entire exploration trace to count every wall-violating transition. This score is 1 minus the fraction of the number of wall-violating transitions and the number of all the legal transitions of the maze. This directly penalizes each incorrect connectivity judgment, ensuring that wall violations are never cost-free even if they occur late in the exploration. 

- **Final path validity.** When the model claims the maze is solvable, it must output a concrete solution path. We verify both that consecutive cells in this path are legally connected (no wall violations) and that the path forms a continuous route from start to end. This item is a binary score for solvable mazes. For unsolvable mazes, this item is kept as 1. 

- **Answer correctness.** A binary score for whether the model’s solvability judgment matches the ground truth. 

This decomposition ensures that the reward signal is dense and informative: the model receives credit for each correctly applied visual primitive rather than only for the final binary answer. 

**Accuracy RM for Path Tracing.** To enforce the model to trace follow the lines, we propose a rule-based RM to judge the generated sequences of points. The final reward is a weighted summation of the following items. 

- **Trajectory accuracy.** We evaluate the alignment between the predicted trajectory and the ground-truth curve from two complementary directions. In the forward direction, for each predicted point, we compute its minimum distance any segment of the groundtruth 

14 

poly-line, then average the distances across all the predicted points. This item penalizes points that deviate from the true path. In the reverse direction, for each ground-truth point, we compute its minimum distance to any segment of the predicted poly-line. This penalizes incomplete coverage where the model skips portions of the curve. The final trajectory score is the average of both directions. 

- **Endpoint accuracy.** We separately verify that the model correctly identifies the start and end locations. For each, we compute the distance between the model’s predicted coordinate and the ground-truth bounding box center. The score decays with distance, reaching zero beyond a tolerance threshold. 

- **Trajectory continuity penalty.** If the distance between the last point in the model’s trajectory and its predicted endpoint exceeds a threshold, a fixed penalty is applied. This discourages the model from outputting a partial trajectory and then “jumping” to a guessed endpoint without actually tracing the full path. 

- **Answer correctness.** A score for whether the endpoint label in the model’s final answer matches the ground truth. 

The bidirectional trajectory evaluation is critical. The forward direction alone would allow the model to output only a few safe points near the start, while the reverse direction alone would not penalize hallucinated detours. Together, they incentivize the model to produce a complete and accurate coordinate trace of the target curve. 

**RL Data** . We expand the data pool during the RL phase. Prior to RL training, we use the SFT cold-start model ( **F** TwG or **F** TwP) to perform rollouts over the data pool, generating _𝑁_ rollouts for each sample. Subsequently, based on the RM scores, we count the number of correct responses among the _𝑁_ rollouts for each sample and categorize the data pool into three difficulty levels: 

- Easy-Level: All _𝑁_ rollouts are correct. 

- Normal-Level: The number of correct rollouts _𝑘_ satisfies 1 ≤ _𝑘< 𝑁_ . 

- Hard-Level: All _𝑁_ rollouts are incorrect. 

We select samples from the “Normal-Level” category for RL, ensuring that the model receives valuable supervisory signals during the GRPO training process. Following the Specialized RL phase, we obtain two expert models, denoted as **E** TwG and **E** TwP. 

## _**2.5.3. Unified RFT**_ 

Equipped with the robust expert models **E** TwG and **E** TwP obtained above, we proceed to integrate the two visual-primitive-based reasoning paradigms— _thinking with grounding_ and _thinking with pointing_ —into a single unified model. We employ these expert models to perform rollouts over the data pool to generate RFT data. Applying the previously introduced difficulty categorization criteria, we retain all samples classified as “Normal-Level” and randomly sub-sample 5% of the “Easy-Level” data (to prevent catastrophic forgetting in excessively simple scenarios). Leveraging this larger and more diverse RFT dataset, we initialize from the base pretrained model to train an enhanced SFT model. Our RFT training configuration remains identical to that of the SFT cold-start phase (including the training hyper-parameters and the initial checkpoint), with the only difference being the updated training data mixture. Following this procedure, we obtain **F** . the unified model 

15 

## _**2.5.4. On-Policy Distillation**_ 

Although the RFT model **F** demonstrates substantial improvements over the cold-start models **F** TwG and **F** TwP in their respective domains, a noticeable performance gap remains when compared to the expert models **E** TwG and **E** TwP. To bridge this gap, we follow [3] and employ On-Policy Distillation (OPD) to effectively consolidate the capabilities of the expert models into a single unified model. This distillation process is achieved by enabling the student model to learn the output distribution of the teacher models based on its own generated trajectories. Formally, given a set of _𝑁_ expert models { _𝜋𝐸_ 1, _𝜋𝐸_ 2, _. . ._ , _𝜋𝐸𝑁_ }, the OPD objective function is defined as: 

**==> picture [308 x 32] intentionally omitted <==**

where _𝑤𝑖_ denotes the weight assigned to each expert model, DKL represents the reverse Kullback-Leibler (KL) divergence loss, and _𝜋𝜃_ denotes the student model. We adopt fullvocabulary logit distillation for our OPD implementation. In practice, we utilize two teacher models, including **E** TwG and **E** TwP. 

## **3. Experiments** 

## **3.1. Implementation Details** 

Our model is trained and evaluated using HAI-LLM [7], a lightweight and efficient distributed training framework built upon PyTorch. During the pre-training stage, we employ a sequence length of 64K and FP8 precision; in the post-training stage, the sequence length is extended to 256K. To maximize the performance of the domain experts, we utilize FP8 precision during the Specialized SFT and Specialized RL phases, and subsequently apply FP4 (MXFP4) quantization in the Unified RFT and OPD phases. 

## **3.2. Evaluation Setup** 

Our evaluation framework integrates widely adopted public benchmarks with a curated inhouse suite. While public benchmarks are vital for standardized comparisons, their constrained evaluation dimensions often fail to capture the full spectrum of a model’s capabilities, e.g., thinking with visual primitives. To bridge this gap, our in-house suite introduces more diverse and challenging axes, serving as a critical complement to the public datasets. 

**Public Benchmarks.** To evaluate counting capabilities, we use two widely used counting benchmarks, including CountQA [30] and Pixmo-Count [4]. We follow the standard evaluation protocols for each dataset, using the official test split for Pixmo-Count. To evaluate spatial reasoning and general VQA, we use benchmarks such as SpatialMQA [20], CV-Bench [31], EmbSpatial [5], OmniSpatial [11], and MIHBench [16]. 

**In-House Benchmarks.** To conduct a more granular evaluation of the model’s capacity to solve tasks by “Thinking with Visual Primitives”, we curate a tailored in-house benchmark suite spanning three critical dimensions: fine-grained counting, multi-hop spatial reasoning, and topological reasoning. 

16 

Table 1 | **Comparison with frontier models** . To ensure a fair comparison, we evaluated all models via their respective APIs using an identical set of prompts. The best results are highlighted in bold; the second-best results are underlined. 

|**Category**<br>**Benchmark (Metric)**|**Gemini-3-Flash**<br>**GPT-5.4**<br>**Claude-Sonnet-4.6 Gemma4-31B**<br>**Qwen3-VL**<br>**235B-A22B-Thinking**<br>**284**|**Ours**<br>**B-A13B-Thinking**|
|---|---|---|
|Counting<br>CountQA (EM / RA@10)<br>Pixmo-Count (EM)<br>DS_Finegrained_Counting (EM)|**66**.**1**/**75**.**1**<br>48.3 / 60.3<br>34.8 / 46.6<br>43.2 / 54.6<br>42.7 / 54.8<br>88.2<br>76.6<br>68.7<br>82.9<br>77.2<br>79.1<br>84.2<br>82.6<br>79.5<br>87.2|64.9/74.1<br>**89**.**2**<br>**88**.**7**|
|Spatial Reasoning<br>& General VQA<br>MIHBench (ACC)<br>SpatialMQA (ACC)<br>EmbSpatial (ACC)<br>CV-Bench (ACC)<br>OmniSpatial (ACC)<br>DS_Spatial_Reasoning (ACC)|83.2<br>83.5<br>81.7<br>82.2<br>75.1<br>67.0<br>61.9<br>58.2<br>60.6<br>54.5<br>82.6<br>80.9<br>75.1<br>82.1<br>**83**.**7**<br>**88**.**6**<br>87.5<br>85.1<br>87.5<br>88.1<br>**59**.**6**<br>58.8<br>53.2<br>49.4<br>55.3<br>93.2<br>81.1<br>97.2<br>77.2<br>96.8|**85**.**3**<br>**69**.**4**<br>**83**.**7**<br>88.4<br>59.5<br>**98**.**7**|
|Topological<br>Reasoning<br>DS_Maze_Navigation (ACC)<br>DS_Path_Tracing (ACC)|49.4<br>50.6<br>48.9<br>49.8<br>49.6<br>41.4<br>46.5<br>30.6<br>33.9<br>24.5|**66**.**9**<br>**56**.**7**|



- **Fine-grained Counting** : Existing fine-grained counting benchmarks, such as TallyQA [1], often suffer from annotation errors and ambiguities, rendering them suboptimal for rigorously evaluating a model’s fine-grained counting capabilities. To address this, we introduce the `DS_Finegrained_Counting` evaluation set. Specifically, we prompt an MLLM to generate counting queries constrained by specific attributes or spatial locations, deliberately ensuring the presence of hard negative samples (i.e., objects sharing the same category as the query target but different attributes). Following rigorous manual verification to ensure data quality, we retain a final set of 600 high-quality test cases. 

- **Multi-hop Spatial Reasoning** : We sample 1, 000 true/false questions and 1, 000 openended questions from the validation set of CLEVR [13]. To facilitate standardized and automated evaluation, we leverage an MLLM to generate plausible distractor options for the open-ended queries, thereby converting them into a multiple-choice format. This reorganized evaluation suite is denoted as `DS_Spatial_Reasoning` . 

- **Topological Reasoning** : Following methodologies in Section 2.4.3 and Section 2.4.4, we construct two distinct evaluation sets: `DS_Maze_Navigation` and `DS_Path_Tracing` , comprising 2, 000 instances for both of the two sets. 

## **3.3. Comparison with Frontier Models** 

For fair comparison, we adopt a unified evaluation protocol across all models. Given that some legacy public benchmarks contain low-resolution images, we apply a preprocessing step to ensure data quality. Specifically, any image with a total pixel count below 640, 000 is upscaled to reach this pixel threshold while strictly preserving its original aspect ratio. For frontier models that support configurable reasoning or thinking budgets (e.g., GPT and Gemini-3Flash), we uniformly set the thinking budget to low for all evaluations to ensure a fair and consistent comparison. For all other benchmarks, we follow the official evaluation protocols and metrics. Results are shown in Table 1. Benefiting from the ability to think with visual primitives, our model achieves competitive performance on these tasks with remarkable token efficiency. Notably, all frontier models exhibit suboptimal performance on topological reasoning tasks, suggesting that substantial room for improvement remains in the reasoning capabilities of multimodal large language models. 

17 

## **3.4. Qualitative Results** 

## _**3.4.1. Boxes as Visual Primitives**_ 

As shown in Figs. 7 to 9, our model demonstrates strong performance on coarse-grained and fine-grained counting tasks through thinking with grounding, while also exhibiting emergent capability synergies. For instance, the model is able to integrate world knowledge for visual question answering, perform counterfactual reasoning, and provide actionable suggestions with spatial coordinates tailored to users’ everyday needs. Although our post-training data about visual primitives does not include any Chinese corpus, the model is capable of thinking and responding in Chinese, benefiting from the multilingual capabilities inherited from the base model. 

## _**3.4.2. Points as Visual Primitives**_ 

As shown in Fig. 10, our model shows the ability of topological reasoning via thinking with pointing, producing step-by-step exploration traces for mazes and sequential tracking trajectories for path tracing. On in-domain instances, the model has the ability to identify and follow the paths, which is enforced via mitigating the cold start data and rewarding during the specialized RL process. 

## **4. Limitations** 

Despite these promising results, our current work has certain limitations. First, constrained by input resolution, the model’s performance in fine-grained scenarios remains sub-optimal, leading to occasionally imprecise outputs of visual primitives. This could potentially be addressed by integrating our framework with existing methods targeting the "Perception Gap" to achieve complementary benefits. Second, the current “thinking with visual primitives” capability relies on explicit trigger words for activation. In the future, we aim to enable the model to autonomously determine whether to invoke this mechanism based on the specific context. Third, utilizing points as visual primitives to solve complex topological reasoning problems remains a formidable challenge, and our current model exhibits limited cross-scenario generalization. Exploring ways to broaden the applicability and robustness of this technique constitutes an important direction for future research. 

## **5. Conclusion** 

To address the inherent “Reference Gap” in Multimodal Large Language Models (MLLMs) during complex reasoning, we introduce “Thinking with Visual Primitives”, a novel reasoning framework. Moving beyond the conventional reliance on simply increasing perceptual resolution, we elevate spatial markers—such as points and bounding boxes—to "minimal units of thought" and interleave them directly into the model’s thinking process. This mechanism endows the model with the ability to “point while it reasons”, precisely anchoring abstract linguistic concepts onto physical image coordinates. Furthermore, leveraging a highly efficient visual token compression architecture, our model achieves performance on par with frontier models across highly challenging tasks, including spatial reasoning, visual QA, and topological reasoning, while significantly reducing image token consumption. Our work demonstrates that the path to System-2 multimodal intelligence lies not merely in “seeing more pixels”, but in constructing a precise, unambiguous referential bridge between language and vision. 

18 

|[Trigger_Placeholder]<br>How many pokemons are there in this<br>photo?|**Thinking with Visual Primitives**<br>1. **Understanding the request**<br>The user wants me to count the number of Pokémon figures in the image. I need to look for|
|---|---|
||characters from the Pokémon franchise among the various toys displayed on the table.|
||2. **Scanning the table for Pokémon**|
||Looking at the collection of figures, I can identify the Pokémon characters:<｜ref｜>pokemons|
||<｜/ref｜><｜box｜>[[215,483,368,711],[330,180,422, 348],[465,438,582,638],[581,403,674,588],|
||[635,195,723,323],[640,244, 814,429]]<｜/box｜>. Breaking this down visually, I see Pikachu on the|
||left, Psyduck in the back row, Charmander in the center, Bulbasaur in the front row, Jigglypuff in|
||the back right, and Squirtle next to Jigglypuff, partially obscured by the Detective Pikachu figure.|
|Original Image|3. **Distinguishing Pokémon from other figures**|
||I am making sure not to include characters from other franchises. Specifically, I'm ignoring|
||Doraemon on the left, Luffy, Zoro, and the red-hatted figure from One Piece in the front left, the|
||Crayon Shin-chan figure in the back center, the female figure, the guy in the detective hat, and the|
||boxy character on the far right.|
||4. **Final count of Pokémon**|
||Counting the distinct Pokémon characters found, there are6of them.|
||**Response**|
|Image with Visual Primitives|Based on the image, there are6Pokémon figures.|
|[Trigger_Placeholder]<br>Which one is heavier?|**Thinking with Visual Primitives**<br>1. **Understanding the weight comparison task**|
||The user is asking to determine which of the two objects on the scale is heavier. To answer this, I|
||need to locate each object and evaluate the physical evidence provided by the scale's state (which|
||side is lower and which is higher).|
||2. **Locating the left object**|
||On the left side of the scale, I see a<｜ref｜>large gray metal locker<｜/ref｜><｜box｜|
||>[[26,41,396,699]]<｜/box｜>. It is constructed with solid, rigid materials and mounted securely|
||on the left<｜ref｜>metal tray<｜/ref｜><｜box｜>[[20,685,393,759]]<｜/box｜>of the balance.|
||The locker is rectangular and imposing, suggesting substantial structural mass.|
||3. **Locating the right object**|
||On the right side of the scale, resting on the corresponding<｜ref｜>metal tray<｜/ref｜><｜|
|Original Image|box｜>[[636,747,970,860]]<｜/box｜>, is a<｜ref｜>translucent rainbow bear<｜/ref｜><｜<br>box｜>[[691,407,916,808]]<｜/box｜>. It looks like a large, vibrant gummy bear made of a|
||translucent, jelly-like substance.|
||4. **Evaluating the scale's equilibrium**|
||The visual evidence from the balance scale is very clear. The<｜ref｜>metal tray<｜/ref｜><｜|
||box｜>[[636,747,970,860]]<｜/box｜>on the right is positioned lower to the ground than the<｜|
||ref｜>metal tray<｜/ref｜><｜box｜>[[20,685,393,759]]<｜/box｜>on the left. The arm holding|
||the right tray is also pointing downward. This means the right side of the scale is heavier.|
||5. **Conclusion**|
||Because the right side containing the gummy bear is lower than the left side containing the locker,|
||the bear must be heavier in this scenario.|
||**Response**|
|Image with Visual Primitives|Based on the scale's tilt, the rainbow bear is heavier.|
|[Trigger_Placeholder]<br>How many Chihuahuas are there in the<br>picture?||
||**Thinking with Visual Primitives**|
||1. **Defining the target animal**|
||The question asks me to count the number of Chihuahuas in the image. I need to focus on|
||differentiating the dogs from the blueberry muffins that also appear in the grid. Looking at the|
||overall layout, the grid seems to contain alternating photos of dogs and muffins.|
||2. **Scanning the grid for dogs**|
||I‘ll search the entire grid to locate every dog. After scanning the rows, I spot the following set of|
||distinct Chihuahuas:<｜ref｜>Chihuahuas<｜/ref｜><｜box｜>[[2,366,233,641],[247,0,498,341],|
|Original Image|[255,680,498,997],[499,347,749,670],[743,0,996,317],[743,658,992,997]]<｜/box｜>. These dogs|
||appear in the first row (second and fourth columns), the second row (first and third columns),|
||and the third row (second and fourth columns). They are mostly a light tan or brownish-white|
||color, with large dark eyes and large ears.|
||3. **Confirming the final count**|
||Looking at the identified dogs, I have counted6Chihuahuas in total.|
||**Response**|
||There are6Chihuahuas in the picture.|
|Image with Visual Primitives||



Figure 7 | **Showcases of thinking with grounding.** Examples include fine-grained counting and counter-commonsense visual question answering, Best viewed on screen with zoom. 

19 

||**Thinking with Visual Primitives**<br>1. **分析⽤户需求**<br>⽤户的问题是关于地理位置的：“这附近有NBA球队吗？”。要回答这个问题，我需要先确定图⽚展示的<br>具体地理位置，然后分析该地区是否拥有NBA球队。<br>2. **识别图中的世界地标**<br>我⾸先观察图⽚，寻找能够定位的地标。最显著的特征是这座横跨深蓝⾊⽔⾯、鲜艳红⾊的悬索桥。背<br>景中还可以看到城市天际线和丘陵。通过视觉识别，我确认这是<｜ref｜>彩⾊的⾦⻔⼤桥<｜/ref｜<br>><｜box｜>[[0,134,882,795]]<｜/box｜>，是旧⾦⼭（San Francisco）的标志性建筑。<br>3. **关联地理位置与NBA球队**<br>既然确认了地点是旧⾦⼭，接下来我将城市与NBA联系。旧⾦⼭市有⼀⽀NBA球队：⾦州勇⼠队<br>（Golden State Warriors）。虽然他们曾经在奥克兰⽐赛，但现在的球馆⼤通中⼼（Chase Center）就<br>位于旧⾦⼭。<br>4. **给出结论**<br>基于位置和球队的存在，这附近确实有NBA球队。<br>**Response**<br>是的，这附近有NBA球队。图中展示的是旧⾦⼭的⾦⻔⼤桥，⽽旧⾦⼭正是NBA球队⾦州勇⼠队<br>（Golden State Warriors）的主场所在地。<br>[Trigger_Placeholder]<br>这附近有NBA球队吗？<br>Original Image<br>Image with Visual Primitives|
|---|---|
||[Trigger_Placeholder]<br>观察桌⾯上的设备和原料，我应该如何制<br>作⼀杯美味的拿铁呢！<br>**Thinking with Visual Primitives**<br>1. **了解⽬标**<br>⽤户希望制作⼀杯美味的拿铁。我可以在桌⾯上看到⼀台全⾃动咖啡机和相关的制作原料。我需要根据这<br>些设备配置，来确定制作拿铁的具体步骤。<br>2. **定位咖啡机与原料**<br>⾸先，我注意到图中央的主体是⼀台<｜ref｜>⿊⾊⾃动意式咖啡机<｜/ref｜><｜box｜<br>>[[111,107,721,970]]<｜/box｜>。在它的前端，我看到<｜ref｜>蒸汽棒<｜/ref｜><｜box｜<br>>[[164,405,236,693]]<｜/box｜>，这通常⽤于加热和打发⽜奶。右侧放置着⼀个<｜ref｜>不锈钢⽜奶壶<br><｜/ref｜><｜box｜>[[670,638,853,905]]<｜/box｜>，这是制作奶泡的核⼼⼯具。旁边还有⼀袋<｜ref｜<br>>咖啡⾖包装袋<｜/ref｜><｜box｜>[[810,438,995, 885]]<｜/box｜>，⾥⾯装有整颗的<｜ref｜>咖啡⾖<br><｜/ref｜><｜box｜>[[385,108,603,150]]<｜/box｜>。<br>3. **选择饮品程序**<br>仔细查看咖啡机的操作⾯板，上⾯有多个饮品选项。为了制作⼀杯拿铁，我需要选择代表拿铁的图标并点<br>击它。因此，我点击<｜ref｜>拿铁按钮<｜/ref｜><｜box｜>[[408,219,444,261]]<｜/box｜>。<br>4. **准备奶泡**<br>选择程序后，下⼀步是利⽤<｜ref｜>蒸汽棒<｜/ref｜><｜box｜>[[164,405,236,693]]<｜/box｜>来处理<br>⽜奶。我拿起旁边的<｜ref｜>不锈钢⽜奶壶<｜/ref｜><｜box｜>[[670,638,853,905]]<｜/box｜>，将其<br>放置在蒸汽棒下⽅。接着，操作咖啡机，利⽤蒸汽棒将壶中的⽜奶打发成细腻的奶泡。<br>5. **准备咖啡杯**<br>当奶泡准备好后，我只需将其倒⼊⼀个合适的杯⼦中。在咖啡机右侧，我发现了⼀个空置的<｜ref｜>陶<br>瓷咖啡杯<｜/ref｜><｜box｜>[[535,779,770,988]]<｜/box｜>，倒奶泡时可以将其拿过来。同时，通过咖<br>啡机⾃动将热咖啡液注⼊杯中，与奶泡结合，最终完成美味的拿铁制作。<br>**Response**<br>要制作⼀杯美味的拿铁，你可以按照以下步骤使⽤这台⾃动咖啡机z：<br>1.在咖啡机控制⾯板上点击选择“拿铁”选项。<br>2.使⽤右侧的蒸汽棒将不锈钢奶壶中的⽜奶打发成细腻的奶泡。<br>3.将打发好的⽜奶和奶泡与机器制作的浓缩咖啡混合，倒⼊左侧准备好的陶瓷咖啡杯中即可。<br>Original Image<br>Image with Visual Primitives|
||[Trigger_Placeholder]<br>这是哪⼉？<br>**Thinking with Visual Primitives**<br>1. **初识⼩巷**<br>这是⼀张充满古朴与岁⽉感的照⽚，展现了⼀条幽静的巷弄。整个场景有⼀种令⼈放松的安静氛围，似乎<br>在某个历史悠久的古镇或传统村落⾥。<br>2. **观察建筑与街道**<br>⽬光⾸先被左侧的建筑吸引，那⾥有⾼挑的<｜ref｜>深⾊⽊制⻔框<｜/ref｜><｜box｜>[[2,249,176,<br>607]]<｜/box｜>，做⼯精致，上⽅还悬挂着<｜ref｜>传统的⽅形灯笼<｜/ref｜><｜box｜>[[56,219,118,<br>284]]<｜/box｜>，增添了⼀抹古⾊古⾹的韵味。⽊⻔旁边，⼀⾯<｜ref｜>泥⼟质感的墙壁<｜/ref｜<br>><｜box｜>[[123,254,341,609]]<｜/box｜>裸露着，看起来经过岁⽉的洗礼⽽显得斑驳，上⾯还贴着⼀些<br>颜⾊鲜艳的<｜ref｜>⼩型⼴告海报<｜/ref｜><｜box｜>[[182,412,259,510]]<｜/box｜>，给旧墙带⼊了<br>些许现代的⽣活⽓息。视线转向街道中央，是⼀条经典的<｜ref｜>鹅卵⽯铺成的步道<｜/ref｜><｜<br>box｜>[[0,554,675,995]]<｜/box｜>。步道由不规则的⽯块拼嵌⽽成，中间的平整⽯板和两侧圆润的砾⽯相<br>互交替，纹理⾮常清晰。迈步向巷⼦深处看去，可以看到更远处的<｜ref｜>经典⽼式建筑<｜/ref｜><｜<br>box｜>[[320,330,515,555]]<｜/box｜>，它们保持着传统的坡屋顶和⽊质结构，层层叠叠，延伸向远⽅。<br>3. **探索右侧庭院**<br>在巷⼦的右侧，我发现了⼀扇<｜ref｜>带有竖条栅栏的⽊⻔<｜/ref｜><｜box｜>[[481,433,536,563]]<｜<br>/box｜>，透着⼏分神秘。紧挨着这道⽊⻔，是⼀⾯<｜ref｜>⽼旧的⼟⻩⾊⽯墙<｜/ref｜><｜box｜<br>>[[525,231,981,726]]<｜/box｜>，墙⾯粗糙且颜⾊饱满。⽊⻔上嵌着⼀块<｜ref｜>⽊质牌匾<｜/ref｜<br>><｜box｜>[[556,386,616,472]]<｜/box｜>，上⾯刻着字。墙根下的条形⽯台上，错落有致地摆放着⼏盆<br>植物和扎染布艺装饰，显得⽣机盎然。<br>4. **感受整体氛围**<br>这条⼩巷完整地保留了传统建筑⻛貌与原始的⽯板路⾯，加上两侧的植物点缀，呈现出⼀种耐⼈寻味的古<br>韵与宁静感。从照⽚中的⽼建筑⻛格、尤其是⽊⻔与⽯墙的构建⽅式来看，这⾥极有可能是我曾⻅过⼀些<br>资料的中国云南古镇，例如丽江、沙溪或类似的⽩族、纳⻄族聚居区。<br>**Response**<br>从图⽚中的⽊构建筑、泥⽯墙体、鹅卵⽯街道和传统的扎染布艺元素来看，这应该是中国云南的⼀个古<br>镇（例如丽江古城、沙溪古镇或类似的拥有⽩族/纳⻄族传统建筑⻛貌的地⽅）。<br>Original Image<br>Image with Visual Primitives|



Figure 8 | **Showcases of thinking with grounding.** Examples include world-knowledge-based question answering and seeking actionable suggestions. Best viewed on screen with zoom. 

20 

**==> picture [455 x 631] intentionally omitted <==**

**----- Start of picture text -----**<br>
|||
|---|---|
|Thinking with Visual Primitives|
|[Trigger_Placeholder]Why is this funny?|1. **Initial impression of the image**|
|This is a side-by-side comparison image. The left side shows a chunk of fruit (maybe a pear or apple) with|
|a face-like indentation and dark marks. The right side shows a close-up of a cat's face. It looks like|
|someone carved the fruit to look exactly like the cat.|
|2. **Locating and explaining the resemblance**|
|I'm looking for the visual similarities. On the left, I see a <｜ref｜>fruit chunk<｜/ref｜><｜box｜|
|>[[75,303,414,758]]<｜/box｜>.|Within its pale flesh, there are natural dark spots that look like eyes.|
|Specifically, there is a <｜ref｜>circular black spot<｜/ref｜><｜box｜>[[185,463,244,552]]<｜/box｜>|
|that perfectly mimics a pupil. Additionally, twisted fiber or skin at the bottom of the fruit chunk looks like|
|a <｜ref｜>dark, textured nose<｜/ref｜><｜box｜>[[273,519,369,655]]<｜/box｜>. Now looking at the|
|right, there is a <｜ref｜>cat's white face<｜/ref｜><｜box｜>[[530,284,999,888]]<｜/box｜> filling the|
|Original Image|frame. I can see a pair of <｜ref｜>green eyes<｜/ref｜><｜box｜>[[594,547,690,617]]<｜/box｜>,|
|looking absolutely devastated or sad. Below the eyes is a <｜ref｜>small pink nose<｜/ref｜><｜box｜|
|>[[755,715,842,786]]<｜/box｜> located between the cheeks.|
|3. **Drawing the conclusion**|
|The humor comes from the 'devastated' expression on the cat's face being perfectly replicated by the|
|randomly shaped marks on the fruit. It looks like the fruit is a raw, unedited version of the cat's soul-|
|piercing sadness. The comparison is spot on, making it funny.|
|Response|
|This is funny because the piece of fruit on the left has natural dark spots that form a face looking|
|remarkably similar to the sad, green-eyed cat on the right. The similarities in the "eyes" and "nose" create a|
|Image with Visual Primitives|hilarious, uncanny resemblance between the two.|
|[Trigger_Placeholder]|
|该如何逃出⽣天呢？|
|Thinking with Visual Primitives|
|1. **初步观察**|
|我环顾四周，发现⾃⼰⾝处⼀个封闭的室内环境。墙⾯粗糙，左侧有⼀扇带有铁栅栏的窗户，右侧是⼀扇|
|关闭的⽊⻔。眼前最引⼈注⽬的是⾼处架⼦上的⼀把⾦⾊钥匙。我需要找到逃出去的⽅法。|
|2. **寻找潜在线索**|
|我注意到墙⾯上⾼悬着|<｜ref｜>⼀把发光的⾦⾊钥匙<｜/ref｜><｜box｜>[[566,110,649,165]]<｜/box｜|
|>，它似乎在暗示我这是解锁的关键。⽊架上并没有可视的楼梯或踏板。⽊⻔下⽅，我看到|<｜ref｜>⼀个|
|带有⾦属⻔把⼿和锁孔的深⾊⽊制⻔<｜/ref｜><｜box｜>[[665,231,991,910]]<｜/box｜>，这显然是需要|
|钥匙才能打开的出⼝。在房间中央的地板上，摆放在我⾯前的|<｜ref｜>⼀把简单的⽊制椅⼦<｜/ref｜|
|><｜box｜>[[399,554,603,936]]<｜/box｜>。它的位置似乎有些特殊。|
|3. **整合信息与推理**|
|Original Image|⽬前我⾯临的核⼼问题是如何拿到⾼处架⼦上的钥匙。我只有⼀把普通⾼度的⽊制椅⼦，直接伸⼿是够不|
|到钥匙的。观察⻔把⼿和椅⼦的位置，它们都在房间的较低⾼度，⽽光线和我的注意⼒都集中在上⽅。我|
|意识到，这很可能是⼀个简单的物理谜题。如果我移动这把椅⼦，把它推到⻔边或架⼦下⽅，我或许能够|
|踩在上⾯够到钥匙。|
|4. **得出结论**|
|综合来看，要逃出这个房间，最直接的策略是：先移动这把低矮的椅⼦，将其挪到悬挂着⾦⾊钥匙的墙壁|
|或架⼦下⽅，然后利⽤椅⼦作为垫脚⽯，伸⼿取下钥匙，最后⾛到右侧的⽊⻔处⽤钥匙开⻔。|
|Response|
|要逃出⽣天，你可以把⽊椅移动到挂着钥匙的架⼦下⽅，踩着椅⼦拿到钥匙，然后去开右侧的⻔。|
|Image with Visual Primitives|
|[Trigger_Placeholder]|
|How many people are there?|Thinking with Visual Primitives|
|1. **Understanding the task**|
|I need to count the number of people visible in the image. The picture is a large group portrait featuring|
|several rows of individuals, so I need to scan carefully from the back to the front to ensure no one is|
|missed.|
|2. **Scanning the group for individuals**|
|I'll look across the entire frame, identifying heads, faces, and bodies to pick out each individual. Sweeping|
|across the arrangement of people, I find all of them: <｜ref｜>people<｜/ref｜><｜box｜>[[4,459,80,|
|859],[32,506,147,900],[105,477,189,883],[107,335,202,638],[136,502,250,893],[202,362,276,616],[210,463,289,82|
|6],[248,499,363,894],[265,386,335,617],[307,455,419,835],[341,383,416,596],[355,510,478,898],[413,392,501,611|
|Original Image|],[425,447,529,835],[483,489,608,895],[487,374,584,576],[539,450,623,824],[582,374,664,594],[582,498,685,889],|
|[626,471,714,817],[658,371,750,591],[684,508,774,887],[720,469,812,844],[737,365,816,603],[768,510,879,895],[|
|788,320,882,601],[824,470,919,889],[828,370,930,625],[856,529,989,891]]<｜/box｜>. This includes people in|
|the very back standing tall, those standing in the middle slightly hunched forward to be seen, and the|
|individuals seated on chairs in the very front.|
|3. **Final count confirmation**|
|Counting the distinct individuals I just located across the entire group, I find 29 people in total.|
|Response|
|There are 29 people in the image.|
|Image with Visual Primitives|

**----- End of picture text -----**<br>


Figure 9 | **Showcases of thinking with grounding.** Examples include humor comprehension in images, escape room game guidance, and counting. Best viewed on screen with zoom. 

21 

**==> picture [420 x 97] intentionally omitted <==**

**----- Start of picture text -----**<br>
[Trigger_Placeholder]<br>Is it possible to navigate from the<br>green rhombus through the maze<br>to the red label? If so, explain the<br>path. Finally, output \boxed{True}<br>or \boxed{False} based on<br>solvability.<br>Original Image Image with Visual Primitives<br>**----- End of picture text -----**<br>


## **Thinking with Visual Primitives** 

Okay, let me analyze the structure of this maze step by step. First locate the starting point: < ｜ point ｜ >[[555,499]]< ｜ /point ｜ >, and the destination: < ｜ point ｜ >[[247,416]]< ｜ /point ｜ >. 

**Start Exploring**: 

- **Step1**: Arriving at < ｜ point ｜ >[[555,499]]< ｜ /point ｜ >, I see 2 paths ahead (counter-clockwise, outward). The counter-clockwise direction is also passable, but from the maze layout, outward direction's path looks wider. 

- **Step2**: At < ｜ point ｜ >[[597,472]]< ｜ /point ｜ >, I notice 2 directions to choose from (clockwise, counter-clockwise).Although counter-clockwise direction is also walkable, after observing path structure, I judge clockwise direction more likely leads to exit. 

- **Step3**: Reaching < ｜ point ｜ >[[600,594]]< ｜ /point ｜ >, I face 2 forks (clockwise, outward).Let me try outward direction first. 

**Step4**: Moving outward: < ｜ point ｜ >[[600,594],[660,667],[720,705]]< ｜ /point ｜ >...reaching < ｜ point ｜ >[[720,705]]< ｜ /point ｜ >, I see 2 forks here. Observing that up direction is a straight dead end at < ｜ point ｜ >[[740,620]]< ｜ /point ｜ >, can see wall directly. Still need to explore the continuing outward direction. **Step5**: Pushing clockwise: < ｜ point ｜ >[[720,705],[660,750]]< ｜ /point ｜ >...but hit a wall at < ｜ point ｜ >[660,750]]< ｜ /point ｜ >. It's a dead end, must backtrack to the previous fork < ｜ point ｜ >[[720,705]]< ｜ /point ｜ >. 

- **Step6**: All paths in this branch are explored, all dead ends. Going back to main junction < ｜ point ｜ >[[600,594]]< ｜ /point ｜ >, taking clockwise direction. **Step7**: Reaching < ｜ point ｜ >[[544,682]]< ｜ /point ｜ >, I face 3 forks (inward, outward, outward).Comparing to inward direction, outward direction's passage characteristics are more like a main path. **Step8**: Reaching < ｜ point ｜ >[[532,720]]< ｜ /point ｜ >, I face 2 forks (clockwise, outward).The outward direction is also passable, but from the maze layout, clockwise direction's path looks wider. 

- **Step9**: At < ｜ point ｜ >[[471,722]]< ｜ /point ｜ >, I notice 2 directions to choose from (clockwise, inward).Although inward direction is also walkable, after observing path structure, I judge clockwise direction more likely leads to exit. **Step10**: Reaching < ｜ point ｜ >[[415,683]]< ｜ /point ｜ >, I face 2 forks (clockwise, outward).Although outward direction is also walkable, after observing path structure, I judge clockwise direction more likely leads to exit. **Step11**: Reaching < ｜ point ｜ >[[371,637]]< ｜ /point ｜ >, I face 2 forks (inward, outward).Comparing to outward direction, inward direction's passage characteristics are more like a main path. 

- **Step12**: At < ｜ point ｜ >[[378,567]]< ｜ /point ｜ >, I notice 3 directions to choose from (clockwise, inward, outward).The inward, outward direction is also passable, but from the maze layout, clockwise direction's path looks wider. **Step13**: At < ｜ point ｜ >[[396,467]]< ｜ /point ｜ >, I notice 2 directions to choose from (outward, outward).After eliminating dead ends, outward is the best choice. **Step14**: Reaching < ｜ point ｜ >[[368,397]]< ｜ /point ｜ >, I face 2 forks (clockwise, outward).The outward direction is also passable, but from the maze layout, clockwise direction's path looks wider. **Step15**: At < ｜ point ｜ >[[330,355]]< ｜ /point ｜ >, I notice 2 directions to choose from (outward, outward).After eliminating dead ends, clockwise is the best choice. 

**Step16**: Continuing outward: < ｜ point ｜ >[[330,355],[300,450]]< ｜ /point ｜ >. Keeping this direction. **Step17**: Looking in outward direction, I can directly see the path to the exit, no need to linger at the junction: < ｜ point ｜ > [[300,450],[265,472],[247,416]]< ｜ /point ｜ > 

- **Step18**: Arriving at < ｜ point ｜ >[[247,416]]< ｜ /point ｜ >, I finally see the exit! 

**==> picture [443 x 229] intentionally omitted <==**

**----- Start of picture text -----**<br>
**Final Path**: After exploration, the correct route is:<br>< ｜ point ｜ >[[555,499],[597,472],[600,594],[544,682],[532,720],[471,722],[415,683],[371,637],[378,587],[396,467],[368,397],[330,355],[300,450],[265,472],[247,416]]<br>< ｜ /point ｜ ><br>Successfully reaching the destination: < ｜ point ｜ >[[247,416]]< ｜ /point ｜ >!<br>Response<br>The maze is solvable. The verified path is:<br>< ｜ point ｜ >[[555,499],[597,472],[600,594],[544,682],[532,720],[471,722],[415,683],[371,637],[378,587],[396,467],[368,397],[330,355],[300,450],[265,472],[247,416]]<br>< ｜ /point ｜ ><br>[\boxed{True}]<br>[Trigger_Placeholder]<br>What number does C connect to? Show the path and output<br>\boxed{endpoint number}.<br>Thinking with Visual Primitives<br>I find the starting point you mentioned, it's located here:<br>< ｜ point ｜ >[[195,594]]< ｜ /point ｜ >.<br>Following this beige line, the visual path I observe is:< ｜ point ｜ ><br>[[195,594],[319,617],[338,626],[342,588],[375,547],[438,522],[508,512],[565,5<br>05],[604,490],[609,464],[595,427],[579,391],[580,364],[606,358],[661,372],[72<br>0,408],[765,442],[803,458]]< ｜ /point ｜ ><br>Following this path, it connects to: < ｜ point ｜ >[[803,457]]< ｜ /point ｜ ><br>Response<br>Based on the visual trace, the endpoint is 5. [\boxed{5}]<br>Original Image Image with Visual Primitives<br>**----- End of picture text -----**<br>


Figure 10 | **Showcases of thinking with pointing.** Examples include maze navigation and path tracing. Best viewed on screen with zoom. 

22