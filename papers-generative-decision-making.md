# Generative Decision-Making

- **Created**: 2026-06-17
- **Last Updated**: 2026-07-02
- **Status**: `In Progress`
- **Related**:
  - [[compression]]
  - [[papers-rl]]
  - [[papers-diffusion-models]]
  - [[papers-generalist-agents]]

---

- [x] [2021] Decision Transformer: Reinforcement Learning via Sequence Modeling - [paper](https://arxiv.org/abs/2106.01345), [code](https://github.com/kzl/decision-transformer)
- [ ] [2022] Decision Diffuser: Is Conditional Generative Modeling all you need for Decision-Making? - [paper](https://arxiv.org/abs/2211.15657)
- [ ] [2021] Trajectory Transformer: Offline RL as One Big Sequence Modeling Problem - [paper](https://arxiv.org/abs/2106.02039)
- [ ] [2019] [Schmidhuber] Reinforcement Learning Upside Down: Don't Predict Rewards, Just Map Them to Actions - [paper](https://arxiv.org/abs/1912.02875), [training agents](https://arxiv.org/abs/1912.02877)
- [ ] [2021] RvS: What is Essential for Offline RL via Supervised Learning? - [paper](https://arxiv.org/abs/2112.10751)
- [ ] [2010] [jveness] MC-AIXI-CTW: Reinforcement Learning via AIXI Approximation - [paper](https://arxiv.org/abs/1007.2049)
- [ ] [2024] [jveness] Generative Reinforcement Learning with Transformers - [paper](https://openreview.net/forum?id=6qtDu7hVPF)
- [ ] [2024] [jveness] Amortized Planning with Large-Scale Transformers: A Case Study on Chess - [paper](https://arxiv.org/abs/2402.04494)
- [ ] [2023] Diffusion Policy: Visuomotor Policy Learning via Action Diffusion - [paper](https://arxiv.org/abs/2303.04137)
- [ ] [2021] [GDM,PedroOrtega,Nando] Shaking the Foundations: Delusions in Sequence Models for Interaction and Control - [paper](https://arxiv.org/abs/2110.10819)

---

## [2021] Decision Transformer: Reinforcement Learning via Sequence Modeling

- **Date**: 2026-05-05
- **Arxiv**: <https://arxiv.org/abs/2106.01345>
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

## [2021] Shaking the Foundations: Delusions in Sequence Models for Interaction and Control

- **Date**: 2026-07-02
- **Arxiv**: <https://arxiv.org/abs/2110.10819>

---

- **tl;dr**: The theory-of-failure paper for this entire file — using a sequence model *as a policy* (Decision Transformer, goal/return-conditioning, behavior cloning from demos) breaks whenever latent confounders exist, because the model treats its own sampled actions as evidence about the world; the fix is causal: condition on observations, **intervene** (do) on self-generated actions.
  - Under a hidden task parameter θ (a confounder of actions and observations), conditioning on your own sampled action collapses the posterior — P(θ | a) acts as if an expert who *saw* θ chose a — so the agent becomes certain of a world state it invented (self-delusion); prediction and imitation are only equivalent when nothing is latent.
  - Treating past self-generated actions as interventions, P(A_{t+1} | do(a_{1:t}), o_{1:t}), cuts the a→θ evidence flow so the agent learns only from its actions' *effects* — and the resulting adaptive policy is Thompson sampling, derived from first principles rather than as a heuristic. Remark 6 is a direct hit on return/goal-conditioned policies: *choosing* the goal/return to condition on is itself an action, so conditioning on it is delusion.
  - Practical recipe: **factual teaching** (log-loss on observed data) trains conditioning; **counterfactual teaching** (score the model's action prediction against the expert's revealed action, stop-gradient through the sampled action) trains intervening — but this needs an expert available *online*; doing it purely from offline demonstrations is an open problem, since demos are confounded by θ (you can't continue a trajectory whose expert action you didn't take).
- **Abstract**:
  - > The recent phenomenal success of language models has reinvigorated machine learning research, and large sequence models such as transformers are being applied to a variety of domains. One important problem class that has remained relatively elusive however is purposeful adaptive behavior. Currently there is a common perception that sequence models "lack the understanding of the cause and effect of their actions" leading them to draw incorrect inferences due to auto-suggestive delusions. In this report we explain where this mismatch originates, and show that it can be resolved by treating actions as causal interventions. Finally, we show that in supervised learning, one can teach a system to condition or intervene on data by training with factual and counterfactual error signals respectively.
