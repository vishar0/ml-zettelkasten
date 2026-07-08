# Generative Decision-Making

- **Created**: 2026-06-17
- **Last Updated**: 2026-07-08
- **Status**: `In Progress`
- **Related**:
  - [[compression]]
  - [[papers-aixi]]
  - [[papers-rl]]
  - [[papers-diffusion-models]]
  - [[papers-generalist-agents]]

---

- [ ] [2019] [Schmidhuber] Reinforcement Learning Upside Down: Don't Predict Rewards, Just Map Them to Actions - [paper](https://arxiv.org/abs/1912.02875), [training agents](https://arxiv.org/abs/1912.02877)
- [ ] [2021] [SergeyLevine] Trajectory Transformer: Offline RL as One Big Sequence Modeling Problem - [paper](https://arxiv.org/abs/2106.02039)
- [x] [2021] [PieterAbbeel] Decision Transformer: Reinforcement Learning via Sequence Modeling - [paper](https://arxiv.org/abs/2106.01345), [code](https://github.com/kzl/decision-transformer)
- [ ] [2022] [JoshTenenbaum,PulkitAgarwal] Decision Diffuser: Is Conditional Generative Modeling all you need for Decision-Making? - [paper](https://arxiv.org/abs/2211.15657)
- [x] [2021] [SergeyLevine] RvS: What is Essential for Offline RL via Supervised Learning? - [paper](https://arxiv.org/abs/2112.10751)
- [ ] [2022] [JimmyBa] You Can't Count on Luck: Why Decision Transformers and RvS Fail in Stochastic Environments - [paper](https://arxiv.org/abs/2205.15967)
- [ ] [2010] [jveness] MC-AIXI-CTW: Reinforcement Learning via AIXI Approximation - [paper](https://arxiv.org/abs/1007.2049)
- [ ] [2024] [jveness] Generative Reinforcement Learning with Transformers - [paper](https://openreview.net/forum?id=6qtDu7hVPF)
- [ ] [2024] [jveness] Amortized Planning with Large-Scale Transformers: A Case Study on Chess - [paper](https://arxiv.org/abs/2402.04494)
- [ ] [2023] Diffusion Policy: Visuomotor Policy Learning via Action Diffusion - [paper](https://arxiv.org/abs/2303.04137)
- [ ] [2021] [GDM,PedroOrtega,Nando] Shaking the Foundations: Delusions in Sequence Models for Interaction and Control - [paper](https://arxiv.org/abs/2110.10819)
- [ ] [2026] [PedroOrtega,Nando] CIC: Continual, Interactive, Causal Agents - [paper](https://love4all.ai/files/continual-interactive-causal-agents.pdf), [blog](https://love4all.ai/blog/continual-interactive-causal-agents/), [notebook](https://love4all.ai/files/continual-interactive-causal-agents.ipynb)

---

## [2021] [PieterAbbeel] [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)

- **Date**: 2026-05-05
- **Code**: <https://github.com/kzl/decision-transformer>

---

- **Abstract**:
  - > We introduce a framework that abstracts **Reinforcement Learning (RL) as a sequence modeling problem**. This allows us to draw upon the simplicity and scalability of the Transformer architecture, and associated advances in language modeling such as GPT-x and BERT. In particular, we present Decision Transformer, an architecture that casts the problem of RL as **conditional sequence modeling**. **Unlike prior approaches to RL that fit value functions or compute policy gradients, Decision Transformer simply outputs the optimal actions by leveraging a causally masked Transformer**. **By conditioning an autoregressive model on the desired return (reward), past states, and actions, our Decision Transformer model can generate future actions that achieve the desired return**. Despite its simplicity, Decision Transformer matches or exceeds the performance of state-of-the-art model-free offline RL baselines on Atari, OpenAI Gym, and Key-to-Door tasks.
- **Intro**:
  - > Recent work has shown transformers [1] can model high-dimensional distributions of semantic concepts at scale, including effective zero-shot generalization in language [2] and out-of-distribution image generation [3]. Given the diversity of successful applications of such models, we seek to examine their application to sequential decision making problems formalized as reinforcement learning (RL). In contrast to prior work using transformers as an architectural choice for components within traditional RL algorithms [4, 5], **we seek to study if generative trajectory modeling - i.e. modeling the joint distribution of the sequence of states, actions, and rewards - can serve as a replacement for conventional RL algorithms**.
  - > We consider the following shift in paradigm: **instead of training a policy through conventional RL algorithms like temporal difference (TD) learning [6], we will train transformer models on collected experience using a sequence modeling objective**. This will allow us to bypass the need for bootstrapping for long term credit assignment - thereby avoiding one of the "deadly triad" [6] known to destabilize RL. **It also avoids the need for discounting future rewards, as typically done in TD learning, which can induce undesirable short-sighted behaviors**. Additionally, we can make use of existing transformer frameworks widely used in language and vision that are easy to scale, utilizing a large body of work studying stable training of transformer models.
  - > In addition to their demonstrated ability to model long sequences, transformers also have other advantages. **Transformers can perform credit assignment directly via self-attention, in contrast to Bellman backups which slowly propagate rewards and are prone to "distractor" signals [7]**. This can enable transformers to still work effectively in the presence of sparse or distracting rewards. Finally, empirical evidence suggest that a transformer modeling approach can model a wide distribution of behaviors, enabling better generalization and transfer [3].
  - > We explore our hypothesis by considering offline RL, where we will task agents with learning policies from suboptimal data - producing maximally effective behavior from fixed, limited experience. This task is traditionally challenging due to error propagation and value overestimation [8]. However, it is a natural task when training with a sequence modeling objective. **By training an autoregressive model on sequences of states, actions, and returns, we reduce policy sampling to autoregressive generative modeling**. **We can specify the expertise of the policy - which "skill" to query - by selecting the desired return tokens, acting as a prompt for generation**.
  - > Illustrative example. To get an intuition for our proposal, consider the task of finding the shortest path on a directed graph, which can be posed as an RL problem. The reward is 0 when the agent is at the goal node and -1 otherwise. We train a GPT [9] model to predict next token in a sequence of returns-to-go (sum of future rewards), states, and actions. **Training only on random walk data - with no expert demonstrations - we can generate optimal trajectories at test time by adding a prior to generate highest possible returns** (see more details and empirical results in the Appendix) and subsequently generate the corresponding sequence of actions via conditioning. **Thus, by combining the tools of sequence modeling with hindsight return information, we achieve policy improvement without the need for dynamic programming**.
  - The core move is to remove much of the usual RL machinery: no Bellman backups, no value-function fitting, no explicit policy-gradient objective.
  - Offline RL datasets already contain sequences of states, actions, and rewards. Decision Transformer formats these as trajectories and trains a transformer with supervised autoregressive prediction.
  - This is closer in spirit to GPT-style conditional generation than to classic dynamic programming.
- **Trajectory modeling**:
  - Each timestep is represented by a tuple:
    - return-to-go $R_t$
    - state $s_t$
    - action $a_t$
  - The sequence is ordered as:
    - $R_1, s_1, a_1, R_2, s_2, a_2, \ldots$
  - The model predicts actions autoregressively from prior returns-to-go, states, and actions.
  - Return-to-go is the desired remaining cumulative reward from a timestep onward:
    - $R_t = \sum_{t'=t}^{T} r_{t'}$
  - At evaluation time, choose a target return, condition the model on that desired return, execute the predicted action, observe the reward, and decrement the remaining target return.
- **Training**:
  - Supervised action prediction over offline trajectories.
  - For continuous-control tasks, train with mean-squared error on actions.
  - For discrete Atari actions, train with cross-entropy.
  - Uses causal masking so each predicted action only depends on past trajectory context.
- **Why return-to-go matters**:
  - Return-to-go acts like a goal-conditioning variable: "act so as to achieve this much remaining reward."
  - This lets a single model represent different-quality behaviors from the same dataset by conditioning on different desired returns.
  - In contrast, ordinary behavior cloning averages over the dataset behavior without an explicit knob for desired performance.
- **Relation to Gato**: [[papers-generalist-agents]]
  - Relevant to [[papers-generalist-agents]] because it is a direct technical predecessor to Gato's framing of control as sequence modeling.
  - Decision Transformer shows "control can be sequence modeling."
  - Gato broadens this to multimodal, multi-task, multi-embodiment behavior: text, image observations, proprioception, discrete actions, continuous actions, and robot control.
  - Gato does not use return-to-go conditioning. It relies on high-return trajectory filtering, prompts/demonstrations, and context.
  - Return-to-go conditioning is awkward for Gato's heterogeneous setting because reward scales and meanings differ across Atari, DM Control, BabyAI, robot stacking, etc., and many vision/language datasets have no reward at all.
  - Gato's high-return filtering is a simpler alternative: use reward only to curate high-quality trajectories per task, then train a behavior-cloning model without inserting reward tokens into every sequence.
- **Limitations / open questions**:
  - Still depends on the quality and coverage of the offline dataset.
  - If high-return behavior is absent from the dataset, return conditioning cannot invent it reliably.
  - For long-horizon tasks, finite context length limits how much trajectory history and goal information the model can use.

## [2021] [SergeyLevine] RvS: What is Essential for Offline RL via Supervised Learning?](<https://arxiv.org/abs/2112.10751>)

- **Date**: 2026-07-08

---

- **One-liner**: the deflationary study of DT / conditional-imitation offline RL — ablate everything and the essence is **the conditioning, not the architecture**: a two-layer MLP maximizing action likelihood matches TD methods and Transformer sequence models; what matters is model capacity/regularization and *what* you condition on (goal vs reward).
- **Abstract**:
  - > Recent work has shown that **supervised learning alone, without temporal difference (TD) learning, can be remarkably effective for offline RL**. When does this hold true, and which algorithmic components are necessary? Through extensive experiments, we boil supervised learning for offline RL down to its essential elements. In every environment suite we consider, **simply maximizing likelihood with a two-layer feedforward MLP is competitive with state-of-the-art results of substantially more complex methods based on TD learning or sequence modeling with Transformers**. Carefully choosing model capacity (e.g., via regularization or architecture) and choosing which information to condition on (e.g., goals or rewards) are critical for performance. These insights serve as a field guide for practitioners doing Reinforcement Learning via Supervised Learning (which we coin RvS learning). They also probe the limits of existing RvS methods, which are comparatively weak on random data, and suggest a number of open problems.
- **Intro**:
  - What RvS names — RL recast as conditional imitation:
    - > Recent work has explored an alternative approach: **convert the RL problem into a conditional, filtered, or weighted imitation learning problem**. This typically uses a simple insight: **suboptimal experience for one task may be optimal for another task**. By conditioning on some piece of information, such as a goal, reward function parameterization, or reward value, such experience can be used for simple behavior cloning [...]. We refer to this set of approaches as RL VIA SUPERVISED LEARNING (RVS). These approaches commonly condition on **goals** [...] or **reward values** [...], but they can also involve reweighting or filtering [...].
  - The motivating disagreement (Q1) — prior work can't agree what's essential:
    - > RvS methods are appealing because of their algorithmic simplicity. However, **prior work has put forward conflicting hypotheses about which factors are essential** for their good performance, including online data [...], advantage weighting [...], or large Transformer sequence models [...]. The first question we study is: **what elements are essential for effective RvS learning?**
  - The limits question (Q2) — the stitching challenge:
    - > it also remains unclear on which tasks and datasets such methods work well. For example, prior work has argued that **temporal compositionality (dubbed "subtrajectory stitching")** is an important component for solving offline RL when there are few near-optimal trajectories present in the data (e.g., the Franka Kitchen and AntMaze tasks in D4RL [12]). A priori, one might expect that **dynamic programming via TD learning is needed** for these tasks. So we also ask: **what are the limits of RvS learning, and does it scale to settings with few near-optimal trajectories?**
  - The three findings:
    - > First, we show that **pure supervised learning [...] performs as well as conservative TD learning** across a diverse set of environments. Second, **simple feedforward models can match the performance of more complex sequence models** from prior work across a wide range of tasks. Finally, **choosing to condition on reward values versus goals can have a large effect on performance**, with different choices working better in different domains.
  - The pointed conclusion:
    - > These simple results **contradict the narrative put forward in many prior works that argue for more complex design decisions** [...]. To the best of our knowledge, our results match or exceed those reported by any prior RvS method.
- **§3 The RvS formulation**: outcome-conditioned policy π(a | s, ω), trained by maximum-likelihood behavior cloning. Per trajectory, sample a timestep and an outcome ω drawn from its own future, then regress the action. Two outcome types: **RvS-G** = a future goal state; **RvS-R** = *average* reward-to-go (average, not sum; max episode length in the denominator). No TD, no value function.
- **§4 Tasks**: D4RL Gym locomotion (reward-conditioned), Franka Kitchen + AntMaze + GCSL (goal-conditioned). AntMaze is the key stress test — a non-Markovian demonstrator, built to require "subtrajectory stitching," normally thought to need dynamic programming.
- **§5 Capacity & regularization** (the "field guide" core):
  - Best nets are **larger than typical in RL/IL** — the policy must represent the optimal policy *and* the policies for every other conditioning value. Over/underfitting is the main challenge.
  - Dropout is task-dependent: helps some (kitchen-complete), hurts others (hopper-medium-expert, antmaze).
  - **Categorical (discretized) action outputs beat unimodal Gaussians** — again a capacity effect.
  - **Validation loss only loosely tracks performance** ⇒ hyperparameter tuning is an open problem. Recipe: grow width until it saturates, add dropout p≈0.1.
- **§6 Results (Table 1)**: RvS-G is SOTA on AntMaze / Kitchen / GCSL; RvS-R matches Decision Transformer on Gym locomotion — with only an MLP.
  - **Stitching surprise**: RvS-G matches DP-based methods (CQL, TD3+BC) on AntMaze despite no dynamic programming. Speculation: goal-conditioning gives compositionality *in space* the way Bellman backups give it *in time*.
  - **Limits**: weak on `random` data (TD/CQL win there). RvS-R **cannot interpolate returns** — conditioning on an intermediate target just mimics the nearest demonstration mode (implicit filtering, not planning), so the reward target must be tuned per task.
- **§7 Discussion — three conclusions**: (1) with the right capacity + regularization, simple FC nets match or beat the best prior methods; (2) the conditioning choice (goal vs reward) is critical and domain-specific; (3) RvS stays competitive where there is little optimal data (Kitchen, AntMaze). Open: automatic tuning of capacity/regularization (validation loss unreliable) and of the conditioning variable.
- **Relation to Decision Transformer**: same decision rule (condition on desired outcome, imitate), differ only in model class. DT credits the win to the Transformer sequence model; RvS shows an MLP does as well ⇒ the win was the conditioning framing, not the architecture. Caveat: RvS's "history is unnecessary" holds on near-Markovian benchmarks (D4RL); context/sequence modeling still earns its keep under partial observability / long-horizon credit / rich observations — which is where DT's own gains concentrated (Key-to-Door, Atari).
- **Relation to You Can't Count on Luck** (also in this file): that paper shows return-conditioned methods fail in stochastic environments — conditioning on a high target return selects for lucky outcomes rather than good actions. RvS shares Decision Transformer's return-conditioning rule and differs only in using an MLP, so it fails the same way. The bug is in the conditioning rule, not in the choice of architecture.

## [2021] [GDM,PedroOrtega,Nando] Shaking the Foundations: Delusions in Sequence Models for Interaction and Control

- **Date**: 2026-07-02
- **Arxiv**: <https://arxiv.org/abs/2110.10819>

---

- **tl;dr**: The theory-of-failure paper for this entire file — using a sequence model *as a policy* (Decision Transformer, goal/return-conditioning, behavior cloning from demos) breaks whenever latent confounders exist, because the model treats its own sampled actions as evidence about the world; the fix is causal: condition on observations, **intervene** (do) on self-generated actions.
  - Under a hidden task parameter θ (a confounder of actions and observations), conditioning on your own sampled action collapses the posterior — P(θ | a) acts as if an expert who *saw* θ chose a — so the agent becomes certain of a world state it invented (self-delusion); prediction and imitation are only equivalent when nothing is latent.
  - Treating past self-generated actions as interventions, P(A_{t+1} | do(a_{1:t}), o_{1:t}), cuts the a→θ evidence flow so the agent learns only from its actions' *effects* — and the resulting adaptive policy is Thompson sampling, derived from first principles rather than as a heuristic. Remark 6 is a direct hit on return/goal-conditioned policies: *choosing* the goal/return to condition on is itself an action, so conditioning on it is delusion.
  - Practical recipe: **factual teaching** (log-loss on observed data) trains conditioning; **counterfactual teaching** (score the model's action prediction against the expert's revealed action, stop-gradient through the sampled action) trains intervening — but this needs an expert available *online*; doing it purely from offline demonstrations is an open problem, since demos are confounded by θ (you can't continue a trajectory whose expert action you didn't take).
- **Abstract**:
  - > The recent phenomenal success of language models has reinvigorated machine learning research, and large sequence models such as transformers are being applied to a variety of domains. One important problem class that has remained relatively elusive however is purposeful adaptive behavior. Currently there is a common perception that sequence models "lack the understanding of the cause and effect of their actions" leading them to draw incorrect inferences due to auto-suggestive delusions. In this report we explain where this mismatch originates, and show that it can be resolved by treating actions as causal interventions. Finally, we show that in supervised learning, one can teach a system to condition or intervene on data by training with factual and counterfactual error signals respectively.

## [2022] [JimmyBa] [You Can't Count on Luck: Why Decision Transformers and RvS Fail in Stochastic Environments](https://arxiv.org/abs/2205.15967)

- **Date**: 2026-07-08

---

- **Abstract**:
  - > Recently, methods such as Decision Transformer [1] that reduce reinforcement learning to a prediction task and solve it via supervised learning (RvS) [2] have become popular due to their simplicity, robustness to hyperparameters, and strong overall performance on offline RL tasks. **However, simply conditioning a probabilistic model on a desired return and taking the predicted action can fail dramatically in stochastic environments since trajectories that result in a return may have only achieved that return due to luck.** In this work, we describe the limitations of RvS approaches in stochastic environments and propose a solution. Rather than simply conditioning on the return of a single trajectory as is standard practice, our proposed method, **ESPER, learns to cluster trajectories and conditions on average cluster returns, which are independent from environment stochasticity.** Doing so allows ESPER to achieve strong alignment between target return and expected performance in real environments. We demonstrate this in several challenging stochastic offline-RL tasks including the challenging puzzle game 2048, and Connect Four playing against a stochastic opponent. In all tested domains, ESPER achieves significantly better alignment between the target return and achieved return than simply conditioning on returns. ESPER also achieves higher maximum performance than even value-based baselines.

## [2026] [PedroOrtega,Nando] [CIC: Continual, Interactive, Causal Agents](https://love4all.ai/files/continual-interactive-causal-agents.pdf)

- **Date**: 2026-07-08
- **Blog**: <https://love4all.ai/blog/continual-interactive-causal-agents/>
- **Notebook**: <https://love4all.ai/files/continual-interactive-causal-agents.ipynb>

---

- **Abstract**:
  - > **Modern language-model agents are usually built by stacking separate training regimes: pretraining, mid-training, supervised fine-tuning, preference modeling, rejection sampling, reinforcement learning, reasoning-specific tuning, self-distillation, and deployment-time patches. This is intelligence by design and engineering, as opposed to emergent intelligence. The multi-stage recipe is a research local minimum, but it has produced powerful systems. It has no single semantics for an interaction transcript: user messages, tool outputs, demonstrations, model actions, verifier judgements, and corrections are often treated as if they were the same kind of evidence. This paper studies a simpler alternative: a continual, causal interaction stream. The central rule is that world-written tokens are evidence, whereas self-written agent tokens are interventions. In LLM fine-tuning this rule becomes a loss mask: keep the agent's own attempts in the context, but remove them from the supervised target. In a small, reproducible STEM reasoning experiment, this interventional stream agent reaches held-out solve accuracy, which is comparable to that of ReST, GRPO, and SFT with oracle corrections. The result is not a claim of benchmark dominance; it is a proof of viability for a single continual learning agent that can use interaction, causality, feedback, and corrections to achieve purposeful and useful behaviour.
- **Intro**:
  - LLMs are already agents, not passive predictors:
    - > **Large language models are no longer only passive next-token predictors. They answer questions, call tools, browse documents, write code, invoke APIs, use verifiers, and receive corrections in multi-turn loops**
  - …but training is still the stacked pipeline:
    - > Yet the dominant training story for such systems is still a sequence of loosely connected stages. First we pretrain on web data; then we continue or mid-train on selected domains; then we run supervised fine-tuning on instructions; then we collect preferences, fit reward models, use RLHF or rejection sampling, add reasoning-specific procedures, and finally distill the improved behavior back into another model [2, 4, 6, 10, 22]
    - > The pipeline works, but it is conceptually fragmented: **many datasets, many objectives, many interfaces, and many opportunities for a handoff to silently change the meaning of the data.**
    - > each stage requires a **new interpretation of what counts as a target**: a web token, an instruction answer, a preference label, a sample selected by a verifier, a reward, a chain-of-thought trace, or a distilled completion.
  - The alternative — one representation for everything:
    - > can instead **keep one representation throughout deployment and training: an interaction stream** containing questions, demonstrations, agent attempts, verifier outputs, tool observations, oracle corrections, and ordinary environment feedback
  - The central move (provenance), verbatim:
    - > In such a stream, a teacher solution and an agent solution can look identical as strings. **The difference is provenance. A teacher solution is world-written evidence. The agent's own solution is an action.**
  - Fig. 1 caption — the "local minimum" framing:
    - > The field of AI is at [a] steep local minimum: The left-hand pipeline treats pretraining, mid-training, SFT, RLHF, reasoning tricks, and self-distillation as separate engineered stages with separate data interfaces. The right-hand picture is the alternative studied here: **a single interventional stream in which the agent receives an observation or question, acts, observes environment feedback, and learns from externally grounded corrections or outcomes.**
  - The Pearl connection → the loss mask (the load-bearing paragraph):
    - > This provenance distinction is exactly **Pearl's distinction between conditioning and intervention** [12, 13]. Observing an action-like event a gives the conditional distribution P(o | a): it asks what tends to be true in records where that event occurred. Setting the action ourselves gives P(o | do (a)): it asks what follows after the action mechanism has been overwritten. For a deployed agent, its own turns belong to the second category. **The agent should condition on what it did when predicting later world responses, but it should not update as if its own attempt were independent evidence about the world.** In a causal language-model loss, the translation is simple: **keep self-authored tokens in the input prefix, but mask them out of the target labels.** World-written tokens - user messages, teacher demonstrations, tool outputs, verifier verdicts, and oracle corrections - remain supervised evidence.
  - Why it matters — post-training is already imitation, so it inherits imitation's failure:
    - > The point matters because **modern post-training is already interactive imitation**. Behavior cloning and learning from demonstrations have a long history in control and robotics [1, 16, 20]. Their classic failure mode is **covariate shift**: after a learner makes a mistake, it reaches states that were rare under the teacher, and future errors compound [18, 19]. LLM agents face an analogous problem at the token level. **If we fine-tune on transcripts without marking who wrote which tokens, the model can learn from its own failed completions as if the world had endorsed them.** This is the same self-confirmation pathology studied in causal sequence-model accounts of delusion [9], and it is related to empirical failures such as **sycophancy and model collapse** under repeated self-training [14, 23, 24].
  - The CIC objective, the bandit analogy, and the (deliberately practical) contribution:
    - > The alternative developed here is a continual, interactive, causal (CIC) objective. It is close in spirit to interactive imitation and to Ortega's interactional account of agency, in which purposeful behavior can be learned from the structure of an interaction history rather than from a primitive scalar reward [8]. It is also consistent with the causal view of decision making in bandits and reinforcement learning: **an arm being pulled by someone else is evidence, whereas pulling the arm ourselves is an intervention** [3, 28]. The contribution here is deliberately practical. **We apply the same causal accounting to the ordinary token-level supervised loss used for LLM fine-tuning.**
- TODO
