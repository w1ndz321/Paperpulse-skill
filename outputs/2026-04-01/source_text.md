2026-04-01 

**==> picture [120 x 25] intentionally omitted <==**

# **FIPO: Eliciting Deep Reasoning with Future-KL Influenced Policy Optimization** 

**==> picture [13 x 13] intentionally omitted <==**

Qwen Pilot Team, Alibaba Group _[∗]_ 

**==> picture [14 x 12] intentionally omitted <==**

Project Page 

GitHub 

HuggingFace ModelScope 

## **Abstract** 

We present **Future-KL Influenced Policy Optimization (FIPO)** , a reinforcement learning algorithm designed to overcome reasoning bottlenecks in large language models. While GRPO style training scales effectively, it typically relies on outcome-based rewards (ORM) that distribute a global advantage uniformly across every token in a trajectory. We argue that this **coarse-grained credit assignment** imposes a performance ceiling by failing to distinguish critical logical pivots from trivial tokens. FIPO addresses this by incorporating **discounted future-KL divergence** into the policy update, creating a **dense advantage formulation** that re-weights tokens based on their influence on subsequent trajectory behavior. Empirically, FIPO enables models to break through the **length stagnation** seen in standard baselines. Evaluated on Qwen2.5-32B, FIPO extends the average chain-of-thought length from roughly 4,000 to over 10,000 tokens and increases AIME 2024 Pass@1 accuracy from 50.0% to a peak of **58.0%** (converging at approximately 56.0%). This outperforms both DeepSeek- _∼ ∼_ R1-Zero-Math-32B ( 47.0%) and o1-mini ( 56.0%). Our results suggest that establishing dense advantage formulations is a vital path for evolving ORMbased algorithms to unlock the full reasoning potential of base models. We open-source our training system, built on the **verl** framework. 

## **1 Introduction** 

Test-time scaling strategies such as those employed in OpenAI’s o-series (Jaech et al., 2024), Gemini series (Comanici et al., 2025), and DeepSeek’s R-series (Guo et al., 2025) mark a fundamental shift in how large language models carry out reasoning. By allocating greater computational resources at inference time, these approaches support longer chain-of-thought and more deliberate reasoning, leading to substantial gains on demanding tasks such as competitive mathematics and coding. Much of this progress stems from large-scale reinforcement learning with verifiable rewards (RLVR) (Guo et al., 2025; Team et al., 2025a; Yang et al., 2025; Team et al., 2025b; Zeng et al., 2025), which fine-tunes a model’s generation policy using feedback from task-specific verifiers, thereby eliciting and amplifying its reasoning capabilities. However, since the specific algorithms and training recipes remain largely undisclosed, it is still unclear how reinforcement learning serves as the primary catalyst to unlock potential reasoning depth, **effectively eliciting the emergence of long chain-of-thought behaviors from base models that initially exhibit no such tendencies.** 

In parallel, the open-source community has devoted substantial effort to reproducing and scaling similar algorithms in more transparent settings (Qin et al., 2024; Huang et al., 2024; Liu et al., 2025; Hu et al., 2025; Yu et al., 2025). Among these efforts, DAPO (Yu et al., 2025) provides a promising large-scale reproduction of GRPO-style training applied to clean base models. However, we argue that the inherent reliance on outcome-based rewards within the GRPO framework introduces a significant structural constraint. 

> _∗_ Full author list available in the Contributions section. 

1 

**==> picture [455 x 264] intentionally omitted <==**

**----- Start of picture text -----**<br>
70<br>60<br>50<br>40<br>30<br>20<br>FIPO avg@32<br>FIPO cons@32<br>DAPO avg@32<br>10<br>DAPO cons@32<br>DeepSeek-R1-Zero-Qwen-32B<br>o1-mini<br>0<br>0 100 200 300 400 500 600<br>Global Training Steps<br>Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 1: **FIPO vs. Baselines Performance Comparison on AIME2024.** FIPO demonstrates that **pure RL training alone** is sufficient to not only outperform other pure RL baselines (the reproduced DAPO and Deepseek-R1-Zero-32B), but also surpass o1-mini. This performance gain is accompanied by the generation of significantly longer responses on average. 

Because rewards are only binary-verifiable at the trajectory end, the standard formulation distributes a uniform advantage to every token. This results in a **completely coarse-grained credit assignment where the algorithm treats critical reasoning steps and trivial tokens with equal weight.** Specifically, we observe that reasoning trajectories produced by such baselines tend to plateau at intermediate lengths. We contend that this limitation imposes a lower performance ceiling on standard GRPO: because the uniform reward cannot highlight the specific tokens that drive correct logic, the model is unable to converge to the complex, extended reasoning paths needed for difficult tasks. While this limitation has led recent works (Hu et al., 2025; Yue et al., 2025; Fan et al., 2025) to revert to the PPO framework for granular advantage estimation, we contend that such density is achievable without the complexity of a critic model. 

We introduce **F** uture-KL **I** nfluenced **P** olicy **O** ptimization ( **FIPO** ). FIPO modifies the policy update by incorporating the **Future-KL divergence** , which re-weights the advantage of current tokens based on the cumulative behaviors of their subsequent trajectories. To maintain training stability, this objective is coupled with **influence weight clipping and filtering mechanism** . We evaluate this approach on **Qwen2.5-32B-Base** , a model with no prior exposure to long-CoT synthetic data, utilizing the publicly released training dataset from DAPO (Yu et al., 2025) to ensure a strictly controlled comparison. As shown in Figure 1, FIPO breaks the performance ceiling of standard baselines; while DAPO achieves 50.0% (Pass@1) on AIME 2024, FIPO enables a progressive lengthening of reasoning chains, where the model steadily scales from a baseline of 4,000 tokens to a deep-reasoning regime of over 10,000 tokens. This consistent expansion pushes accuracy to a peak of 58.0%, a result on par with recent PPO-based counterparts. **These findings demonstrate that establishing a dense advantage formulation effectively bridges the gap between GRPO efficiency and PPO performance, unlocking deep reasoning capabilities that otherwise remain untapped under uniform reward schemes.** 

Our implementation is built upon the **verl** framework (Sheng et al., 2025) and the DAPO codebase. By 

2 

fully releasing the complete training code and configuration recipes, we aim to reveal valuable insights into large-scale reinforcement learning for LLMs that benefit the broader research community. 

## **2 Related Work** 

**Reinforcement Learning for LLMs.** Reinforcement learning (RL) serves as a cornerstone of the posttraining pipeline for large language models. While foundational efforts primarily utilized Reinforcement Learning from Human Feedback (RLHF) to align model behavior with human preferences (Stiennon et al., 2020; Ouyang et al., 2022), recent advancements have shifted focus toward enhancing reasoning capabilities through RL. Notable examples include the OpenAI o-series (Jaech et al., 2024), which pioneered this reasoning-centric approach, and DeepSeek-R1 (Guo et al., 2025), which introduced a comprehensive RLVR (Lambert et al., 2024) framework for developing reasoning models via the GRPO algorithm (Shao et al., 2024). These breakthroughs have further inspired a wave of industry-leading subsequent works, such as Kimi (Team et al., 2025a), Qwen3 (Yang et al., 2025), and Gemini 2.5 (Comanici et al., 2025). 

**Large-scale open-source RL recipes.** Parallel to the proprietary advancements in reasoning models, the open-source community has made significant strides in democratizing large-scale RL training. These efforts aim to bridge the gap between high-level algorithmic concepts and practical, stable implementations that can scale efficiently, while providing continuous improvements to the training pipeline. Notably, GSPO (Zheng et al., 2025), BAPO (Xi et al., 2025), SAPO (Gao et al., 2025), and OR1 (He et al., 2025) primarily develop their RL algorithms on models that have already developed long-CoT capabilities. Other works devote significant effort to incentivizing complex reasoning abilities starting from a cleaner base model, specifically **Qwen2.5-32B-Base** . Among these efforts, Open-Reasoner-Zero (Hu et al., 2025), VC-PPO(Yuan et al., 2025), VAPO (Yue et al., 2025), and T-PPO (Fan et al., 2025) build their algorithms upon the PPO framework (Schulman et al., 2017), whereas DAPO (Yu et al., 2025) is developed as a modification of GRPO. 

To ensure a rigorous evaluation, we adopt Qwen2.5-32B-Base as our backbone and use DAPO as our primary baseline. While Open-Reasoner-Zero reverts to PPO to avoid the sparse advantage signals in vanilla GRPO, we address this challenge by refining the GRPO framework directly. Notably, since Open-Reasoner-Zero operates without auxiliary value models, its performance ultimately falls short of DAPO. In contrast, other methods like VC-PPO, VAPO and T-PPO rely heavily on value models that are pre-trained by models already supervised fine-tuned (SFT) with Long-CoT data. We contend that this methodology introduces an external knowledge prior through the value model, creating a potential confounding factor in the evaluation. This makes it difficult to discern whether the performance gains stem from the policy optimization algorithm itself or are simply inherited from the pre-trained value model. By eschewing the need for a value model and starting from a vanilla base model, FIPO achieves performance comparable to, and in some cases superior to, these pre-trained value-modelbased approaches. **This demonstrates that establishing a dense advantage formulation is a promising direction for evolving ORM-based GRPO algorithms to unlock the inherent reasoning potential of base models.** 

## **3 Preliminary** 

In this section, we review the policy optimization frameworks central to our work: PPO and its valuenetwork-free variants, GRPO and DAPO. Throughout this paper, let _T_ denote the total length of a trajectory and _t_ denote the index of the current step within that trajectory. In the GRPO setting, for each question prompt _q_ , we sample _G_ trajectories, yielding outputs denoted by _o_ . 

3 

## **3.1 Proximal Policy Optimization** 

**Proximal Policy Optimization (PPO) (Schulman et al., 2017)** introduces a clipped surrogate objective for policy optimization. By constraining policy updates to the proximity of the old policy through a clipping mechanism, PPO stabilizes training and improves sample efficiency. Specifically, PPO maximizes: 

**==> picture [321 x 16] intentionally omitted <==**

Here, _rt_ ( _θ_ ) = _ππθ_ old _θ_ ( _o_ ( _to|tq|_ , _qo_ , _<o<t_ ) _t_ )[denotes the token-level probability ratio at step] _[ t]_[,] _[A]_[ˆ] _[t]_[ is the advantage estimated] via a learned value function, and _ϵ_ is the clipping coefficient. Crucially, standard PPO implementations compute the advantage _A_[ˆ] _t_ using Generalized Advantage Estimation (GAE) (Schulman et al., 2015). This results in distinct, token-specific advantage signals, enabling the model to perform temporal credit assignment. This stands in contrast to simplified formulations that derive advantages solely from the final outcome, effectively broadcasting a uniform signal to all tokens within a trajectory. By leveraging GAE, PPO provides dense supervision at every step, allowing it to differentiate between critical and less influential actions along the generation process. 

## **3.2 Group Relative Policy Optimization** 

**Group Relative Policy Optimization (GRPO) (Shao et al., 2024)** circumvents the computational burden of a value network by estimating advantages through group-based sampling. For a given query _q_ (and ground truth _a_ ), a set of outputs _{oi}i[G]_ =1[is sampled from the old policy] _[ π][θ]_ old[.][The] _[ sequence-level]_[ advantage] for the _i_ -th sample is standardized as: 

**==> picture [320 x 21] intentionally omitted <==**

where _µ_ and _σ_ denote the empirical mean and standard deviation, respectively, of the rewards within the sampled group. Similar to PPO, GRPO adopts a clipped objective but adds a per-token KL penalty term directly to the loss: 

**==> picture [432 x 67] intentionally omitted <==**

Here, _ρi_ , _t_ ( _θ_ ) = _ππθ_ old _θ_ ( _o_ ( _io_ , _ti|_ , _tq|_ , _qo_ , _io_ , _<i_ , _<t_ ) _t_ )[represents][the][probability][ratio.][By][design,][the][computed][scalar] _[A]_[ˆ] _[i]_[is] broadcast across the entire sequence; specifically, for every token _t_ , the advantage is set identically as ˆ ˆ _Ai_ , _t_ = _Ai_ . Unlike PPO, where Generalized Advantage Estimation (GAE) provides a distinct signal for each token, GRPO assigns uniform credit to every step in the trajectory, regardless of its individual contribution to the final outcome. 

## **3.3 Decoupled Clip and Dynamic Sampling Policy Optimization** 

**Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO) (Yu et al., 2025)** extends the GRPO framework by eliminating the explicit KL penalty. Instead, it employs asymmetric clipping within the interval (1 _− ϵ_ low, 1 + _ϵ_ high) to amplify updates for advantageous actions, effectively mitigating the entropy collapse commonly observed with GRPO. Furthermore, DAPO implements a token-level policy gradient loss to sustain healthy optimization dynamics in the context of long Chain-of-Thought RL training. Furthermore, DAPO enforces a dynamic sampling mechanism that guarantees a mix of positive and negative samples within each group _{oi}i[G]_ =1[.][This mechanism ensures effective updates with] non-trivial gradients during optimization. We adopt DAPO as the primary baseline for this work. 

4 

## **3.4 Findings on Directions of Policy Update and Fine-grained Token Analysis** 

In our previous work, Meng et al. (2025) provides a systematic analysis on how RL rewrites the base model. We found that in over 98% of generation steps, the output distribution is identical. RL only intervenes at **highly sparse, critical** tokens to keep the model on track. Additionally, Huang et al. (2025) argue that Standard metrics (like KL divergence) fail to locate these sparse changes. By tracking the signed log-probability difference, we can precisely map the **“direction" of optimization** , and even boost inference accuracy just by amplifying these key tokens, with zero extra training. These insights lead to a clear conclusion: not all tokens contribute equally to the reasoning process. However, while the instantaneous log-probability difference indicates the direction of optimization, it serves merely as a primitive, localized signal. The key to eliciting more effective reasoning then lies in discovering how to leverage this raw ∆ log _p_ to formulate a much more accurate measurement of a token’s true downstream impact, thereby enabling us to automatically locate and reinforce these critical junctions during RL training. 

## **4 FIPO** 

In this section, we introduce the core framework of FutureKL-Induced Policy Optimization (FIPO). We begin by discussing the probability shift, the fundamental building block of our objective. Next, we detail the formulation of Future-KL. Finally, we illustrate how our method implements a “soft decay window” strategy by focusing on the local “future context”. This mechanism naturally prioritizes proximal signals over distant ones, limiting the effective horizon to the most relevant subsequent tokens. 

## **4.1 Probability Shift:** ∆ log _p_ 

Our method is grounded in our recent investigations into the dynamics of Large Language Models (LLMs) during reinforcement learning. Specifically, our previous work on RLVR updates (Huang et al., 2025) demonstrates that the magnitude and direction of the probability shift, ∆ log _p_ , serve as robust indicators of improved reasoning. Building upon this, our fine-grained analysis of distributional shifts (Meng et al., 2025) further reveals that this generation process is often driven by a few “sparse but critical” tokens that disproportionately influence the subsequent chain of thought. Inspired by these insights, we identify the token-level probability shift as the atomic unit for our credit assignment mechanism. Formally, we define the probability shift at time step _t_ as the log-space difference between the current policy and the old policy: 

∆ log _pt_ = log _πθ_ ( _ot | q_ , _o<t_ ) _−_ log _πθ_ old ( _ot | q_ , _o<t_ ). (3) 

This term serves as a differential signal capturing the instantaneous policy drift: 

- **Positive Shift (** ∆ log _pt >_ 0 **):** Indicates that the current policy has increased the likelihood of token _ot_ relative to the old policy. This typically suggests that the training objective is reinforcing this specific reasoning step. 

- **Negative Shift (** ∆ log _pt <_ 0 **):** Implies that the policy is suppressing the generation of _ot_ , signaling that the updated model is actively down-weighting this specific token relative to the reference policy. 

Unlike traditional KL penalties, which treat this drift primarily as a regularization cost to be minimized, we interpret ∆ log _pt_ as a directional signal of behavioral adjustment, thereby explicitly coupling the optimization objective to the generative dynamics. However, relying solely on this instantaneous shift is insufficient, as it fails to capture the long-term consequences of a decision. This limitation motivates our proposed **Future-KL** mechanism, which re-weights the current token by aggregating the distributional shifts of its _future_ trajectory. 

5 

## **4.2 Future-KL Estimation** 

While ∆ log _pt_ captures the local distributional shift, reasoning is inherently a sequential process where the true significance of this token depends on the trajectory it initiates. To capture this causal influence, we define **Future-KL** as the cumulative signed probability shift from the current step _t_ to the end of the sequence _T_ : 

**==> picture [284 x 28] intentionally omitted <==**

This summation is mathematically equivalent to the log-likelihood ratio of the joint probability distributions for the subsequent sequence _ot_ : _T_ . It can thus be interpreted as a sample-based estimate of the KL divergence restricted to the future horizon, measuring the cumulative deviation of the current policy from the reference policy for the remainder of the trajectory. We therefore term this metric _Future-KL_ . Functionally, FutureKL _t_ serves as a forward-looking metric that quantifies the cumulative shift in policy distribution regarding the future trajectory. A positive value (FutureKL _t >_ 0) indicates that the updated policy has overall **reinforced** the entire subsequent trajectory initiated by token _ot_ , suggesting that _ot_ acts as a stable anchor for the subsequent reasoning chain. In contrast, a negative value (FutureKL _t <_ 0) implies that the policy is collectively suppressing the future tokens following _ot_ , signaling that the trajectory stemming from this point is becoming less favored during the optimization process. 

However, in practice, such formulation tends to exacerbate the variance arising from distributional shifts. Since FutureKL _t_ acts as a weighting coefficient for the advantage function (as detailed in subsequent sections), excessive deviations in future logits (e.g., due to training-inference inconsistency) can disproportionately inflate the scale. This renders the optimization overly sensitive to noisy tokens rather than the intrinsic quality of the reasoning chain. Empirically, we observe that in the absence of safety mechanisms, training runs are prone to severe instability. As shown in Figure 2, this collapse is distinctively accompanied by a sharp spike in the “low-clip fraction” metric, which tracks the frequency of samples triggering the Dual-Clip threshold (a hard clip ratio on negative samples) (Ye et al., 2020). Such high importance ratios on negative samples signify a critical misalignment: the model assigns high probability to an action that is effectively harmful. In our experiments, this spike (at approximately Step 70) aligns with a surge in the gradient norm and Policy KL[1] , indicating a substantial shift in policy distribution, alongside an immediate drop in response length. This synchronization indicates that without regulation, the accumulated negative signals from FutureKL _t_ can reach some extreme values that destabilize the training process. 

Motivated by these observations, we refine the FutureKL computation by explicitly masking tokens that exceed the Dual-Clip threshold. Since these tokens represent ’harmful’ actions whose gradients are already clipped (via the clipped policy objective), allowing their excessively high importance ratios to propagate into the recursive sum introduces severe variance. By zeroing out the future accumulation for these specific outliers, we remove the primary source of instability. The refined objective is defined as: 

**==> picture [364 x 28] intentionally omitted <==**

Here, _Mk_ acts as a binary filter that evaluates to 1 only if the importance ratio remains within the Dual-Clip threshold _c_ (typically _c ≥_ 10), and 0 otherwise. This ensures that tokens triggering the hard constraints are effectively excluded from the FutureKL computation, preventing gradient explosion without altering the trajectory’s valid signals. 

> 1We compute the Policy KL divergence as the batch mean of the negative log-ratio: Policy KL = 1 _B·L_[∑] _i[B]_ =1[∑] _t[L]_ =1 �log _π_ old( _oi_ , _t|o<t_ ) _−_ log _π_ ( _oi_ , _t|o<t_ )�. It measures the KL divergence of the generated sequences between the current policy and the policy prior to the gradient update (the roll-out policy). 

6 

**==> picture [455 x 258] intentionally omitted <==**

**----- Start of picture text -----**<br>
1e 5 (a) Dual clip, low clip fraction (b) Policy KL<br>1.75<br>0.010<br>1.50<br>0.008<br>1.25<br>1.00 0.006<br>0.75<br>0.004<br>0.50<br>0.002<br>0.25<br>0.00 0.000<br>0 20 40 60 80 100 120 140 0 20 40 60 80 100 120 140<br>Global Training Steps Global Training Steps<br>(c) Gradient Norm (d) Mean Response Length<br>1600<br>0.40<br>1400<br>0.35<br>1200<br>0.30<br>1000<br>0.25<br>800<br>0.20<br>600<br>0.15<br>0 20 40 60 80 100 120 140 0 20 40 60 80 100 120 140<br>Global Training Steps Global Training Steps<br>**----- End of picture text -----**<br>


Figure 2: **Training instability with vanilla FutureKL.** Analysis of the unstable run observed around Step 70. (a) A sharp spike in the _low-clip fraction_ (indicating a drastic shift in the policy distribution driven by negative samples) triggers (b) a sudden divergence in Policy KL. (c) an immediate explosion in gradient norm and These internal instabilities collectively precipitate (d) a catastrophic collapse in response length, confirming that unregulated negative signals destabilize the optimization. 

## **4.2.1 Soft Decay Window** 

Beyond the stability constraints, we also address the inherent uncertainty of long-horizon generation. The causal dependency between the current action _ot_ and future tokens _ok_ naturally diminishes as the time horizon _k − t_ increases. Immediate successors are directly conditioned on the current choice, whereas distant tokens are subject to accumulating stochasticity and become less predictable. To model this diminishing influence, we introduce a discount factor _γ ∈_ (0, 1]. Incorporating this decay into the masked objective yields the final formulation used in our experiments: 

**==> picture [308 x 28] intentionally omitted <==**

We parameterize the decay rate as _γ_ = 2 _[−] τ_[1] , where _τ_ is a hyperparameter controlling the effective horizon (or “half-life”) of the future supervision. This formulation ensures that the credit assignment concentrates on the immediate reasoning chain, assigning lower weights to distant, highly uncertain tokens. Functionally, _τ_ defines the aperture of this _soft decay window_ . Unlike a hard truncation that abruptly discards information beyond a fixed step, this exponential formulation creates a continuous sliding window where _τ_ represents the distance at which the future signal’s influence attenuates by half. This mechanism allows the model to prioritize local coherence within the window _τ_ , while smoothly filtering out the noise from the distant future without introducing boundary artifacts. 

## **4.2.2 FutureKL Re-weighted Advantage with Clipping** 

Finally, we integrate the soft decay window and masking mechanisms into the policy optimization objective. We propose to modulate the standard advantage estimate _A_[ˆ] _t_ using a future influence weight _ft_ . The modified advantage _A_[˜] _t_ is defined as: 

7 

(7) 

**==> picture [276 x 19] intentionally omitted <==**

This formulation introduces two key operations: 

1. **Exponential Mapping:** We transform the accumulated scalar signal from log-space to a multiplicative domain. Mathematically, the unclipped term represents a decay-weighted product of likelihood ratios, which acts as an importance weight reflecting the policy’s effective preference for the generated future. 

2. **Influence Weight Clipping:** We constrain the multiplicative coefficient _ft_ to the interval [1 _− ϵ flow_ , 1 + _ϵ fhigh_ ]. This operation serves strictly to bound the magnitude of the advantage modulation, preventing the exponential term from introducing excessive variance into the gradient estimate. By capping the weight, we ensure that the future trajectory modulates the update signal within a controlled range, avoiding numerical instability caused by extreme accumulated log-probability shifts. 

Functionally, this modulation scales the magnitude of the policy update based on the reinforcement or suppression of the generated future. When the updated policy **reinforces** the subsequent trajectory (i.e., FutureKL _t >_ 0), the weighting term _ft >_ 1 magnifies the gradient signal. Consequently, positive advantages are boosted to encourage the current token as a stable anchor, while negative advantages incur harsher penalties to strictly correct errors initiating this path. Conversely, when the policy **suppresses** the future trajectory (i.e., FutureKL _t <_ 0), the term _ft <_ 1 attenuates the update. This attenuation effectively reduces the reward signal for locally harmful tokens that happen to be in a successful sequence and softens the penalty for good tokens trapped in a failing one. In practice, to ensure training stability and prevent over-penalization, we reset _ft_ = 1 for tokens associated with negative advantages ( _A_[ˆ] _t <_ 0) that exhibit excessively large importance ratios. 

## **4.3 Target Loss** 

Adopting the token-level formulation from DAPO (Yu et al., 2025), we maximize the following FIPO objective: 

**==> picture [431 x 31] intentionally omitted <==**

Here, _G_ represents the number of sampled outputs per query, and _ri_ , _t_ = _ππθ_ old _θ_ ( _a_ ( _ia_ , _ti|_ , _ts|is_ , _ti_ ), _t_ )[denotes the impor-] tance ratio between the current and old policies. The term _A_[ˆ] _i_ , _t_ refers to the group relative advantage, while _fi_ , _t_ serves as the Future-KL importance weight introduced previously. 

## **5 Experiment** 

## **5.1 Experiment Settings** 

In this work, we adopt the training settings of DAPO (Yu et al., 2025), specifically focusing on mathematical reasoning tasks to ensure a strictly controlled comparison. We utilize the VeRL framework (Sheng et al., 2025) for both training and baseline reproduction. We maintain optimization settings consistent with DAPO, and trained on the public-released DAPO-17K dataset. Each training batch consists of 512 prompts with 16 responses sampled per prompt, yielding a total of 8,192 training samples. In the standard DAPO configuration, updates are performed with a mini-batch size of 512 samples (32 prompts), resulting in 16 gradient updates per training iteration. **However, our empirical findings suggest that a larger mini-batch size improves training stability.** Consequently, we adopted a mini-batch size of 1,024 samples (64 prompts), resulting in 8 gradient updates per iteration. A more detailed discussion regarding 

8 

the impact of this increased minibatch size is provided in Appendix Sec. E. For the **Future-KL computation** , we set the effective horizon of the decay rate _τ_ to 32. Specific to the training of 32B model, the Future-KL weight is clipped within [1, 1.2]; **this effectively amplifies the reward for tokens associated with successful reasoning trajectories while imposing a more stringent penalty for those leading to incorrect outcomes.** Both FIPO and DAPO share a maximum response length of 20,480 tokens, with an overlong penalty applied to trajectories exceeding 16,384 tokens. Detailed hyperparameter configurations for both the baseline and FIPO are provided in Appendix Sec. A. 

For evaluation, we adopt AIME 2024 as our primary validation benchmark, supplemented by AIME 2025, to ensure a rigorous and comprehensive comparison with the DAPO baseline. To maintain results stability and account for variance in chain-of-thought generation, we follow the DAPO protocol by repeating the evaluation 32 times and reporting the Pass@1 (averaged over 32 samples). Inference hyperparameters are consistently set to a temperature of 1.0 and a top- _p_ of 0.7. 

## **5.2 Main Result** 

Table 1 presents the quantitative evaluation on the AIME 2024 and AIME 2025 benchmarks. FIPO achieves a systematic improvement of roughly 6.0% in Pass@1 (Avg@32) over the DAPO baseline across both datasets. We prioritize this metric as the most robust indicator of reasoning reliability. While we also observe gains in consistency, the improvement in coverage (Pass@32) is more modest, particularly on AIME 2025. We attribute this to the inherent challenge of expanding the absolute problem-solving scope of large models through reinforcement learning alone. Without external knowledge augmentation or tool integration, RL is primarily constrained to refining how the model navigates its existing internal knowledge. Consequently, while FIPO significantly enhances the model’s ability to reliably solve problems within its latent capacity (driving up Avg@32), shifting the boundary of solvable problems (Pass@32) remains non-trivial. 

Table 1: **Comparison of reasoning performance on AIME benchmarks.** All results are reported as percentages (%). We report the average Pass@1 across 32 samples (Avg@32), the majority vote (Cons@32), and the probability of at least one correct answer (Pass@32). To align with prior baseline reporting and reduce sensitivity to digit-level generation variance, final values are rounded to the nearest integer. 

|**Method**|**AIME 2024**<br>Avg@32<br>Cons@32<br>Pass@32|**AIME 2025**<br>Avg@32<br>Cons@32<br>Pass@32|
|---|---|---|
|DAPO (Baseline)<br>**FIPO (Ours)**|50.0%<br>60.0%<br>80.0%<br>**56.0%**<br>**73.0%**<br>**83.0%**|38.0%<br>47.0%<br>63.0%<br>**43.0%**<br>**50.0%**<br>**67.0%**|



## **6 Analysis** 

Beyond the aggregate metrics, we observe several distinct phenomena that we believe underpin these performance gains. By dissecting the training dynamics and inference behaviors, we identify three critical drivers of FIPO’s effectiveness: the **emergence of length-based scaling** in reasoning chains, the **distinct positive learning signal** captured by the response length weighted mean advantage formulation, and the significantly **improved stability** of the optimization process compared to standard baselines. 

## **6.1 The scaling of length and performance** 

**A central observation in FIPO’s training is that performance gains are deeply coupled with a continuous expansion of response length.** As training progresses, we observe a significant surge in token counts that scales alongside model accuracy. As illustrated in Figure 3, the response length of DAPO gradually enters a stagnation phase after an initial increase, plateauing at an average of approximately 

9 

**==> picture [455 x 250] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Min Length in Tokens (b) Q25 Length in Tokens (c) Mean Length in Tokens<br>12000<br>FIPO<br>1750 DAPO<br>8000 10000<br>1500<br>1250 6000 8000<br>1000<br>6000<br>750 4000<br>4000<br>500<br>2000<br>250 2000<br>0 0 0<br>0 100 200 300 400 500 0 100 200 300 400 500 0 100 200 300 400 500<br>(d) Median Length in Tokens (e) Q75 Length in Tokens (f) Accuracy Scaling (w in 10 5 units)<br>12000<br>14000<br>10000 12000 0.5<br>8000 10000 0.4<br>8000<br>6000<br>0.3<br>6000<br>4000<br>4000 0.2<br>2000 2000 Stage 1: Stage 2: RR [2][2] = 0.92,= 0.93, w w = 19.9 × 10 = 3.8 × 10 5 5<br>0.1 Stage 3: R [2] = 0.78, w = 1.5 × 10 5<br>0 0 Stage 4: R [2] = 0.63, w = 2.0 × 10 5<br>0 100 200 300 400 500 0 100 200 300 400 500 0 2000 4000 6000 8000 10000 12000<br>Global Training Steps Response Length (Tokens)<br>Accuracy (Mean@32)<br>**----- End of picture text -----**<br>


Figure 3: **Dynamics of response length and performance scaling during training.** Subplots (a)-(e) show the evolution of response length metrics (Min, Q25, Mean, Median, Q75) over global training steps. Compared to the DAPO baseline, FIPO significantly increases response length, effectively eliciting more extensive Chain-of-Thought reasoning. Subplot (f) demonstrates that this increased length correlates strongly with improved accuracy, suggesting that longer CoT is key to breaking performance barriers. 

4,000 tokens. In contrast, FIPO exhibits remarkable scaling resilience. This scaling process unfolds through distinct evolutionary phases (visualized by the colored regions in Figure 3), marking a transition from an initial rapid exploration to a sustained period of deep reasoning. Notably, although an overlong penalty is maintained to constrain redundancy, FIPO successfully guides the model to elicit extensive Chain-of-Thought (CoT) reasoning. Qualitative analysis provided in Appendix D reveals that this length expansion is driven by the gradual emergence of self-reflection behaviors; the model increasingly utilizes the expanded sequence length to re-evaluate its intermediate steps and explore multiple methodologies to verify its conclusions. Interestingly, this spontaneous emergence of systematic self-verification aligns with the inference-time scaling behaviors observed in advanced reasoning models (e.g., the OpenAI o-series and DeepSeek-R1). This suggests that FIPO effectively triggers **inference-time reasoning** , prioritizing **analytical depth** to unlock higher performance. 

**Further examination of the training dynamics reveals that this surge in length is not driven by isolated outliers but represents a comprehensive distributional migration.** As shown in Figure 3(a)–(e), all length-related percentiles, ranging from the Minimum and Q25 to the Median and Q75, exhibit a synchronized and stable upward shift under FIPO training. Specifically, across these training phases, the median token count climbs steadily from an initial 200 to over 10,000. Such a migration across the entire distribution demonstrates that FIPO facilitates a fundamental shift in the model’s underlying problem-solving strategy: the model transitions from direct response patterns to systematic, self-verifying reasoning processes. Crucially, we find that this collective shift toward longer reasoning chains is what unlocks the performance breakthroughs observed in our experiments. As illustrated in Figure 3(f), there is a strong positive correlation between model accuracy and response length across all identified stages. While the correlation slopes (denoted as _w_ ) vary slightly between phases, the trajectory remains consistently positive. While the DAPO baseline’s performance reaches a bottleneck as its length plateaus, FIPO’s ability to continuously unlock additional “thinking space” allows the model to navigate increasingly complex logical dependencies. This confirms that **FIPO successfully converts increased sequence length** 

10 

## **into genuine reasoning depth, enabling the model to surpass the performance ceilings of standard baselines on high-difficulty reasoning tasks.** 

## **6.2 The dynamics of advantage and sustained reasoning growth** 

We further investigate the training dynamics by comparing the evolution of rewards and advantages. As shown in Figure 4(a), the baseline (DAPO) consistently maintains a higher mean training reward than FIPO. However, we argue this disparity is a numerical artifact of the reward formulation rather than an indicator of superior performance. Because the reward function incorporates an overlong penalty, FIPO’s construction of elaborate reasoning chains inevitably leads to higher penalties, thus suppressing its average raw reward. Conversely, the baseline’s higher reward is driven by its tendency to generate shorter responses. While this strategy maximizes immediate reward by minimizing penalties, it suggests a convergence to a local optimum within a restricted search space. 

This hypothesis is further corroborated by the rapid escalation in the number of sampled batches for DAPO, as shown in Figure 4(b). This trend indicates that the model is overfitting the training set, increasingly generating non-discriminative samples (i.e., batches that are uniformly correct or incorrect) which yield negligible gradient information. Consequently, the algorithm is forced to sample more aggressively to harvest sufficient effective data for optimization. In contrast, FIPO actively traverses a more expansive search space, prioritizing the structural depth required for challenging reasoning tasks over the mere avoidance of penalties. 

This difference becomes even more pronounced when shifting from raw rewards to the dynamic incentives provided by advantages. As observed in Figure 4(c), DAPO exhibits a declining trend in response length weighted mean relative advantage[2] throughout training. This implies that the length of positive samples is increasingly dominated by that of negative samples, resulting in a diminishing incentive to extend derivations; since increased length no longer yields more positive relative advantages, the model eventually hits a plateau in reasoning growth. In stark contrast, FIPO demonstrates a consistent upward trajectory. This indicates that the positive samples are evolving to be significantly more substantive than their negative counterparts. **This dynamic fosters a sustained growth trajectory: as the generation of longer, valid reasoning chains yields increasingly positive advantages, facilitated by the steady rise in rewards, it preserves the model’s momentum to pursue even more extensive and rigorous reasoning paths.** 

## **6.3 Smooth Policy Drift, Exploration and Gradient Update** 

To further characterize the training process, we examine the evolution of policy behavior and optimization stability. As shown in Figure 5(a), FIPO exhibits a steady and structured increase in Policy KL divergence. This represents a progressive policy shift, where the model consistently moves away from its **previous policy state** to navigate toward a more specialized reasoning regime. This trend is qualitatively consistent with our rollout observations: the length of self-reflection segments increases incrementally rather than abruptly, reflecting a gradual expansion of the reasoning process (see Appendix D for examples). 

The optimization characteristics also differ significantly in terms of gradient scale. As shown in Figure 5(b), FIPO’s Gradient Norm remains low and consistent throughout training, characterizing an evolution built upon fine-grained updates. In contrast, the baseline (DAPO) displays highly volatile fluctuations, with frequent, violent spikes in its gradient norm. These fluctuated updates indicate that DAPO’s search process is at risk of abrupt shifts and potential instability. 

This contrast in stability is further reflected in the policy entropy (Figure 5c). While FIPO maintains a smooth and sustained rise in entropy, indicating a continuous and stable exploration of the reasoning 

> 2We define the response length weighted mean advantage as: _A_ ¯ =[∑] _i[B]_ =1[∑] _tLi_ =1 _[A][i]_[,] _[t]_ , where _B_ is the batch size, _Li_ is the ∑ _i[B]_ =1 _[L][i]_ response length of the _i_ -th sample, and _Ai_ , _t_ represents the token-level group relative advantage. 

11 

**==> picture [455 x 151] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Mean Rewards (b) Dynamic Sampling Batch Count (c) Response Length Weighted Advantages<br>FIPO 6<br>DAPO 0.00<br>0.6<br>0.4 5 0.05<br>0.10<br>0.2<br>4<br>0.15<br>0.0<br>3 0.20<br>0.2<br>0.25<br>0.4 2<br>0.30<br>0.6<br>1 0.35<br>0 100 200 300 400 500 600 0 100 200 300 400 500 600 0 100 200 300 400 500 600<br>Global Training Steps<br>**----- End of picture text -----**<br>


Figure 4: **Analysis of training reward and length-weighted advantages.** (a) **Mean training rewards.** DAPO achieves higher raw scores, which is expected as both methods incorporate an overlong penalty that suppresses the reward of longer responses. (b) **Number of Sampled Batches.** This metric indicates the sampling redundancy required to maintain a sufficient number of _effective_ batches. A higher sampling need suggests the model frequently generates non-informative trajectories on the training set, serving as a potential indicator of overfitting. (c) **Response length weighted mean advantages.** FIPO exhibits a sustained upward trend, establishing a positive reinforcement cycle where longer responses increasingly yield positive advantages. In contrast, DAPO shows a declining trend, suggesting a failure to convert length into effective reasoning gains, which ultimately limits its performance. 

**==> picture [455 x 150] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Policy KL (b) Gradient Norm (c) Entropy<br>FIPO 2.00<br>DAPO<br>0.0020 1.75 1.0<br>1.50<br>0.0015<br>1.25 0.8<br>1.00<br>0.0010 0.6<br>0.75<br>0.0005 0.50 0.4<br>0.25<br>0.0000 0.00 0.2<br>0 100 200 300 400 500 600 0 100 200 300 400 500 600 0 100 200 300 400 500 600<br>Global Training Steps<br>**----- End of picture text -----**<br>


Figure 5: **Policy evolution and optimization dynamics.** (a) Policy KL divergence. (b) Policy Entropy. (c) Gradient Norm. FIPO exhibits a more controlled policy drift and smoother update gradients than DAPO. Notably, FIPO’s rising entropy, paired with the weighted advantage trends in Fig. 4(b), indicates that the model is actively exploring a broader reasoning space where longer CoT paths increasingly correspond to correct solutions. 

space, DAPO’s entropy is marked by noisy oscillations throughout the training process. In contrast, DAPO’s entropy is marked by noisy oscillations as training progressed. **Together, these metrics depict FIPO as a model that achieves significant and purposeful policy evolution toward complex reasoning while ensuring the optimization process remains numerically well-behaved.** 

## **7 Conclusion** 

In this paper, we introduced **F** uture-KL **I** nfluenced **P** olicy **O** ptimization ( **FIPO** ), a reinforcement learning approach designed to resolve the coarse-grained credit assignment problem inherent in standard GRPO. By incorporating discounted _Future-KL divergence_ into policy updates, FIPO transforms sparse outcomebased rewards into dense, token-level supervision. Our empirical analysis identifies and addresses a critical “length-performance plateau” in existing baselines, demonstrating that standard uniform rewards fail to sustain long-chain reasoning. Validated on **Qwen2.5-32B-Base** , FIPO effectively breaks this ceiling: 

12 

it propels performance on AIME 2024 from a baseline of 50.0% to a **peak of 58.0% (converging at 56.0%)** and extends the average chain-of-thought length from 4,000 to over 10,000 tokens. Crucially, these findings challenge the prevailing assumption that complex critic models are necessary for granular credit assignment, proving that dense supervision can be effectively realized within the more efficient GRPO framework. To facilitate future research, we open-source our complete training code and recipes, providing the community with a scalable and accessible pathway to advance large-scale reasoning models. 

## **8 Limitations and Future Work** 

Despite its effectiveness, FIPO has certain limitations: 

**Cost and Efficiency.** A primary constraint is the increased computational cost associated with extending reasoning sequences. As FIPO successfully unlocks CoT lengths exceeding 10,000 tokens, the training and inference overhead grows significantly, posing challenges for resource-constrained deployments. We argue that the development of advanced reasoning should be a sequential process: first eliciting long, high-quality reasoning capabilities, and subsequently optimizing them for efficiency. While this paper focuses on the first stage, breaking through length stagnation to achieve superior performance, the task of transforming these long reasoning paths into more concise and efficient forms is a critical next step. We will leave this for future exploration. 

**Task Generalization.** Another limitation is that our evaluations are primarily conducted on mathematical reasoning benchmarks. However, we contend that mathematics serves as a rigorous and representative proxy for deep reasoning; its requirement for objective, verifiable ground truth and high-density logical consistency makes it the most demanding testbed for our algorithm. Having demonstrated that FIPO can overcome length stagnation in this challenging domain, we leave the exploration and validation of these elicited behaviors in other open-ended or less structured domains for future work. 

**Training Data Scope.** To ensure a rigorous and fair comparison with the baseline, we restricted our training exclusively to the dataset used in DAPO. Consequently, we have not yet explored the scalability of FIPO on larger-scale or higher-quality datasets. While this controlled setting serves to isolate the algorithmic contributions of our method, the potential of FIPO when trained on more extensive or diverse data distributions remains uncharted. Moreover, while FIPO achieves superior performance over o1-mini on mathematical benchmarks, this advantage is inherently domain-specific. Given that our training was strictly confined to the math dataset, we do not anticipate these gains to generalize across non-mathematical domains, such as coding or symbolic logic, where o1-mini benefits from massive-scale, multi-stage reinforcement learning. Consequently, we leave the exploration of FIPO’s generalization across broader data regimes and its fundamental scaling properties for future work. 

**Limited Model Scope.** A core objective of our study is to investigate RL-driven reasoning starting from a clean base model with no prior exposure to Long-CoT synthetic data. This strict requirement for experimental purity significantly limits the selection of suitable backbone models. Most contemporary open-source models optimized for reasoning have already undergone extensive supervised fine-tuning (SFT) or distillation from long-form reasoning traces. We contend that the underlying training dynamics of eliciting reasoning directly from a vanilla base model differ fundamentally from further optimizing a model that has already internalized distilled reasoning patterns. Consequently, our choice of models was restricted to a few high-quality vanilla base models, such as the Qwen2.5 series, to ensure that our findings specifically characterize the emergence of inherent reasoning potential rather than the refinement of pre-distilled CoT behaviors. In future work, we plan to investigate the efficacy and mechanistic behavior of our algorithm when applied to such pre-distilled Long-CoT models, exploring whether the dense advantage formulation can further refine or synergize with pre-existing distilled reasoning capabilities. 

13 

**Performance Gap vs. Distillation.** While RL-based self-evolution significantly enhances reasoning, it remains a "discovery-based" process that is inherently less efficient than direct distillation. Larger teacher models provide a much denser supervisory signal and superior heuristics (logits) that are difficult for a smaller model to self-derive through sparse rewards alone, resulting in a persistent performance gap between self-trained and distilled variants. 

14 

## **9 Contributions** 

## **Core Contributors** 

Chiyu Ma[1,5] , Shuo Yang[2,5] 

## **Contributors** 

Kexin Huang[5] , Jinda Lu[5] , Haoming Meng[3,5] , Shangshang Wang[4,5] 

## **Supervision** 

Bolin Ding[6] , Soroush Vosoughi[1] , Guoyin Wang[5] , Jingren Zhou[6] 

## **Affiliations** 

1 Dartmouth College 

> 2 Peking University 

- 3 University of Toronto 

> 4 University of Southern California 

- 5 Qwen Pilot Team 

- 6 Alibaba 

15