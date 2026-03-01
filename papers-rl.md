# Reinforcement Learning

- **Created**: 2019-04
- **Last Updated**: 2026-02-18
- **Status**: `In Progress`

---

- [[tutorial-rl-spinning-up-openai]]

---

- TODO papers in <https://spinningup.openai.com/en/latest/spinningup/keypapers.html>
- [ ] [2000] [AndrewNg] Algorithms for Inverse Reinforcement Learning - [paper](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf)
- [ ] [2013] Guided Policy Search - [paper](https://graphics.stanford.edu/projects/gpspaper/gps_full.pdf)
- [ ] [2014] [DavidSilver] DPG: Deterministic Policy Gradient Algorithms - [paper](https://proceedings.mlr.press/v32/silver14.pdf)
- [ ] [2015] [TimLillicrap,DavidSilver] Continuous control with deep reinforcement learning - [paper](https://arxiv.org/abs/1509.02971)
- [ ] [2015] [GregW,TimLillicrap,DavidSilver] Learning Continuous Control Policies by Stochastic Value Gradients - [paper](https://arxiv.org/abs/1510.09142)
- [ ] [2018] [blog] Deep RL doesn't work yet - [blog](https://www.alexirpan.com/2018/02/14/rl-hard.html)
- [x] [2018] [Ben Recht] A Tour of Reinforcement Learning: The View from Continuous Control - [paper](https://arxiv.org/abs/1806.09460)
- [ ] [2018] Investigating Human Priors for Playing Video Games - [paper](https://arxiv.org/abs/1802.10217)
- [ ] [2020] [deepmind] Agent57: Outperforming the Atari Human Benchmark - [paper](https://arxiv.org/abs/2003.13350), [blog](https://deepmind.google/blog/agent57-outperforming-the-human-atari-benchmark/)
- [ ] [2020] Atari 100K: Model-Based Reinforcement Learning for Atari - [paper](https://arxiv.org/abs/1903.00374)
- [ ] [2020] Revisiting Fundamentals of Experience Replay - [paper](https://arxiv.org/abs/2007.06700)
- [ ] [2023] Bigger, Better, Faster (BBF): Human-level Atari with human-level efficiency - [paper](https://arxiv.org/abs/2305.19452)
- [ ] [2023] [deepmind] A Definition of Continual Reinforcement Learning - [paper](https://arxiv.org/abs/2307.11046)
- [ ] TODO nethack
- [ ] TODO crafter
- [ ] [2017] [OpenAI] Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World - [paper](https://arxiv.org/abs/1703.06907)
- [ ] [2017] [FAIR] Intrinsic Curiosity Module (ICM): Intrinsic Motivation and Automatic Curricula via Asymmetric Self-Play - [paper](https://arxiv.org/abs/1703.05407)
- [ ] [2018] [OpenAI] RND: Exploration by Random Network Distillation - [paper](Exploration by Random Network Distillation)
- [ ] [2020] [rockt] PLR: Prioritized Level Replay - [paper](https://arxiv.org/abs/2010.03934)
- [ ] [2021] [rockt] Replay-Guided Adversarial Environment Design - [paper](https://arxiv.org/abs/2110.02439)
- [ ] [2022] [rockt] Evolving Curricula with Regret-Based Environment Design - [paper](https://arxiv.org/abs/2203.01302)
- [ ] [2022] [rockt] E3B: Exploration via elliptical episodic bonuses - [paper](https://arxiv.org/abs/2210.05805)
- [ ] [2024] Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning - [paper](https://arxiv.org/abs/2402.16801)
- [x] [2025] ScaleRL: The Art of Scaling Reinforcement Learning Compute for LLMs - [paper](https://arxiv.org/abs/2510.13786)
- [ ] TODO diplomacy
- [ ] TODO alphago etc
- [ ] TODO openai dota 5v5
- [ ] [2025] TODO Kevin Murphy RL book: <https://arxiv.org/abs/2412.05265>

---

## [2015] [TimLillicrap,DavidSilver] Continuous control with deep reinforcement learning

- **Date**: 2026-02-20
- **Arxiv**: <https://arxiv.org/abs/1509.02971>
- **Paperpile**: <https://app.paperpile.com/view/?id=a37dfe47-75a9-426d-8e47-0d73d7a07638>

---

- TODO

## [2018] [Ben Recht] A Tour of Reinforcement Learning: The View from Continuous Control

- **Date**: 2025-12-10
- **Arxiv**: <https://arxiv.org/abs/1806.09460>
- **Paperpile**: <https://app.paperpile.com/view/?id=e49dc43d-6eb4-4832-8751-443e6964d352>

---

- **Intro**:
  - > This survey aims to provide a language for the control and reinforcement learning communities to begin communicating, highlighting what each can learn from the other.  Controls is the theory of designing complex actions from well-specified models, while reinforcement learning often makes intricate, model-free predictions from data alone.  Yet both RL and control aim to design systems that  use  richly  structured  perception,  perform  planning  and  control  that  adequately  adapt  to environmental changes, and exploit safeguards when surprised by a new scenario.
  - > I try  to  put  RL  and  control  techniques  on  the  same  footing  through  a  case  study  of  the linear quadratic regulator (LQR) with unknown dynamics.  This baseline will illuminate the var- ious  trade-offs  associated  with  techniques  from  RL  and  control.
  - > “model-free”  methods  popular  in  deep  reinforcement  learning  are  considerably  less effective in both theory and practice than simple model-based schemes when applied to LQR. Per- haps surprisingly, I also show cases where these observations continue to hold on more challenging nonlinear applications.  I then argue that model-free and model-based perspectives can be unified, combining their relative merits.
- **RL - optimal control when the dynamics are unknown**:
  - > find a sequence of inputs that drives a dynamical system to maximize some objective beginning with minimal knowledge of how the system responds to inputs.
  - > Since the dynamics are stochastic, the optimal control problem typically allows a controller to observe the state before deciding upon the next action [12].  This allows a controller to continually mitigate uncertainty through feedback.  Hence, rather than optimizing over deterministic sequences of actions $a_t$, we instead optimize over policies. A control policy (or simply “a policy”) is a function, $\pi$, that takes a trajectory from a dynamical system and outputs a new control action.  Note that $\pi$ gets access only to previous states and control actions.
  - > we can’t solve this optimization problem using standard optimization methods unless we know the state transidion dynamics.  We must learn something about the dynamical system and subsequently choose the best policy based on our knowledge.
  - > The main paradigm in contemporary RL is to play the following game.  We decide on a policy $\pi$ and horizon length $L$. Then we pass this policy either to a simulation engine or to a real physical system and are returned a trajectory $\tau_L$ and a sequence of rewards. We want to find a policy that maximizes the reward with the fewest total number of samples computed by the oracle, and we are allowed to do whatever we’d like with the previously observed trajectories and reward information when computing a new policy.  If we were to run $m$ queries with horizon length $L$, we would pay a total cost of $mL$.  However, we are free to vary our horizon length for each experiment. This is our oracle model and is called **episodic reinforcement learning** (See, for example Chapter 3 of Sutton and Barto [76], Chapter 2 of Puterman [58], or Dann and Brunskill [24]).  We want the expected reward to be high for our derived policy, but we also need the number of oracle queries to be small.
  - > Do we decide an algorithm is best if it crosses some reward threshold in the fewest number of samples?  Or is it best if it achieves the highest reward given a fixed budget of samples?  Or maybe there’s a middle ground?
- **RL vs supervised learning**:
  - > A key distinguishing aspect of RL is the control action $a$.  Unlike in prediction,  the practitioner can vary $a$, which has implications both for learning (e.g., designing experiments to learn about a given system) and for control (e.g., choosing inputs to maximize reward).
  - > There is a precarious trade-off that must be carefully considered:  reinforcement learning demands interventions with the promise that these actions will directly lead to valuable returns, but the resulting complicated feedback loops are hard to study in theory, and failures can have catastrophic consequences.
- **RL Strategies**:
  - **(1) Model-based RL**:
    - fits a model of the state transitions to best match observed trajectories, and uses this to approximate the solution to the RL problem.
  - **(2) Model-free RL**:
    - eschews the need for the system's model, directly seeking a map from observations to actions.
    - > The term “model-free” almost always means “no model of the state transition function” when casually claimed in reinforcement learning research.  However, this does not mean that modeling is not heavily built into the assumptions of model-free RL algorithms.
    - **(a) Approximate Dynamic Programming / Value Based**:
      - uses Bellman’s principle of optimality to approximate the RL problem using previously observed data.
      - > Also troubling is the fact that we had to introduce the discount factor in order to get a simple Bellman equation.  One can avoid discount factors,  but this requires considerably more sophisticated analysis.  Large discount factors do in practice lead to brittle methods, and the discount becomes a hyperparameter that must be tuned to stabilize performance.
    - **(b) Policy Search / Policy Based**:
      - directly searches for policies by using data from previous episodes in order to improve the reward.
      - `REINFORCE` algorithm and log-likelihood trick.
  - > The main question is which of these approaches makes the best use of samples and how quickly do the derived policies converge to optimality.
- > **This survey has focused on “episodic” reinforcement learning and has steered clear of a much harder problem:  adaptive control.  In the adaptive setting, we want to learn the policy online.  We only get one trajectory.  The goal is, after a few steps, to have a model whose reward from here to eternity will be large.  This is very different, and much harder that what people are doing in RL. In episodic RL, you get endless access to a simulator.  In adaptive control, you get one go.**
- > as soon as a  machine  learning  system  is  unleashed  in  feedback  with  humans,  that  system  is  a  reinforcement learning system.

## [2023] Bigger, Better, Faster (BBF): Human-level Atari with human-level efficiency

- **Date**: 2025-11-28
- **Arxiv**: <https://arxiv.org/abs/2305.19452>
- **Paperpile**: <https://app.paperpile.com/view/?id=9cd1b87a-f170-4c80-bdc3-7b135a501947>
- **Code**: <https://github.com/google-research/google-research/tree/master/bigger_better_faster>

---

- **Abstract**:
  - > We introduce a value-based RL agent, which we call BBF, that achieves super-human performance in the Atari 100K benchmark. BBF relies on scaling the neural networks used for value estimation, as well as a number of other design choices that enable this scaling in a sample-efficient manner. We conduct extensive analyses of these design choices and provide insights for future work. We end with a discussion about updating the goal- posts for sample-efficient RL research on the ALE. We make our code and data publicly available.
- **Human-level sample efficiency on Atari**:
  - > The success of these RL methods has relied on large neural networks and an enormous number of environment samples to learn from – **a human player would require tens of thousands of years of game play to gather the same amount of experience as OpenAI Five or AlphaGo**.
  - > It is plausible that such large networks are necessary for the agent’s value estimation and/or policy to be expressive enough for the environment’s complexity, while large number of samples might be needed to gather enough experience so as to deter- mine the long-term effect of different action choices as well as train such large networks effectively. As such, **obtaining human-level sample efficiency with deep RL remains an outstanding goal**.
  - > **as RL continues to be used in increasingly challenging and sample-scarce scenarios, the need for scalable yet sample-efficient online RL methods becomes more pressing**. Despite the variability in problem characteristics making a one-size-fits-all solution unrealistic, there are many insights that may transfer across problem domains. As such, methods that achieve “state-of-the-art” performance on established benchmarks can provide guidance and insights for others wishing to integrate their techniques.
  - **BBF**:
    - Atari 100K benchmark: agents are constrained to 2 hours of gameplay to evaluate human-level efficiency. 100k steps (400k frames) at 60 FPS is 111 minutes.
    - EfficientZero: achieves human-level sample efficiency via model-based RL.
    - BBF: achieves this via model-free RL while being much more computationally efficient than EfficientZero.
- **Background - RL Axes**:
  - (1) Value-Based vs Policy-Based vs Actor-Critic (hybrid) - the "what do we learn?" axis.
  - (2) Model-Based vs Model-Free - the "do we understand the world?" axis.
  - (3) On-Policy vs Off-Policy
  - (4) Online RL (Environment interaction) vs Offline RL (Batch RL) - Offline RL is inherently off-policy, online RL can be either on-policy or off-policy.
- **Method**:
  - > The question driving this work is: **How does one scale networks for deep RL when samples are scarce?**

## [2024] Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning

- **Date**: 2026-02-18
- **Arxiv**: <https://arxiv.org/abs/2402.16801>
- **Paperpile**: <https://app.paperpile.com/view/?id=1a4ffec3-2196-4477-a09c-2caf9b84f365>
- **Code**: <https://github.com/MichaelTMatthews/Craftax>

---

- **Abstract**:
  - > Benchmarks play a crucial role in the development and analysis of reinforcement learning (RL) algorithms. We identify that existing benchmarks used for research into open-ended learning fall into one of two categories.  Either they are too slow  for  meaningful  research  to  be  performed without enormous computational resources, like Crafter, NetHack and Minecraft, or they are not complex enough to pose a significant challenge, like Minigrid and Procgen.  To remedy this, we first present Craftax-Classic: a ground-up rewrite of Crafter in JAX that runs up to 250x faster than the Python-native original. A run of PPO using 1 billion environment interactions finishes in under an hour using only a single GPU and averages 90% of the optimal reward.  To provide a more compelling challenge we present the main Craftax benchmark, a significant extension of the Crafter mechanics with elements inspired from NetHack. **Solving Craftax requires deep exploration, long term planning and memory, as well as continual adaptation to novel situations as more of the world is discovered**. We show that **existing methods including global and episodic exploration, as well as unsupervised environment design (UED) fail to make material progress** on the benchmark. We believe that **Craftax can for the first time allow researchers to experiment in a complex, open-ended environment with limited computational resources**.
- **Intro**:
  - **Motivation**:
    - Benchmark gap: existing open-ended RL benchmarks are either too slow (Crafter, NetHack, Minecraft) or too easy (Minigrid, Procgen).
    - Goal: a benchmark that is both fast enough for accessible research AND hard enough that existing methods fail.
    - One author completed perfect runs in ~5 hours of human gameplay (with unlimited decision time), so it's human-solvable but not agent-solvable.
  - **Jax-based environments**:
    - > While deep RL training has traditionally been split between collecting trajectories on CPU-based environments and then training policy and value networks on the GPU, **the relatively new phenomenon of JAX-based environments allows for the whole RL pipeline to be run on the GPU**. This allows for massive parallelisation of trajectory gathering (we use up to 4096 parallel environment workers), the elimination of the GPU-CPU transfer bottleneck and just-in-time (JIT) compilation of the whole training process.
  - **Crafter**:
    - > While Crafter has become a popular benchmark, the evaluation protocol proposed allocates algorithms only 1 million environment interactions, a very limiting constraint when compared to other RL benchmarks.
    - > While we reuse many of the Crafter dynamics, our aim is to provide a benchmark for investigations into open-endedness rather than sample efficiency.
    - > Open-endedness, by its very definition, should not be constrained by a fixed number of samples. In practice we have to impose some limit, but this should be suitably high as to not impact the emergence of interesting phenomena.
  - **Open-ended learning**:
    - Exploration through intrinsic rewards.
    - Unsupervised environment design (UED).
      - RL paradigm where an adversary proposes environments configurations (referred to as levels) for an agent to train on.
      - The adversary is rewarded for choosing levels that maximise the agent’s regret (difference in return between the current and optimal agent).
      - This has been empirically shown to automatically induce a curriculum of progressively harder levels that aid the performance and generalisation properties of the learned agent.
      - Different UED algorithms require different levels of access to the underlying environment state, ranging from simply being able to repeat seeds to directly editing the levels.
      - Due to the functional nature of Craftax necessitated by JAX, the entire environment state is exposed as a single object, making UED methods easy to apply.
- **Craftax Overview**:
  - **Craftax-Classic** (JAX rewrite of Crafter):
    - 257x speedup over the Python-native Crafter original.
    - PPO hits ~90% of optimal reward using 1B interactions in 51 minutes on a single GPU.
    - Pixel-based and symbolic observation variants.
  - **Craftax** (extended benchmark, Crafter + NetHack-inspired):
    - 9 procedurally generated floors: overworld, dungeons, mines, fire/ice realms, graveyard, boss floor.
    - Enemies expanded from 3 to 19 types, each requiring different strategies.
    - Potions with randomly permuted effects per episode (tests in-context learning).
    - Attribute system: XP from floor descent → specialize in dexterity, strength, or intelligence.
    - Requires deep exploration, long-term planning, memory, and continual adaptation.
    - > While in Crafter the player is confined to a single 64x64 grid, Craftax contains 9 unique procedu- rally generated floors, including caves, dungeons, fire and ice floors and a final boss floor. The player can descend and ascend through the world by finding the ladders that con- nect adjacent floors. Each floor contains distinct challenges in the forms of different terrain generation, enemies and required skills, necessitating deep exploration and generali- sation. While each floor is unique, many game mechanics are shared between them and, on a meta-level, exploration strategies that worked on earlier floors (for instance moving adjacent to a block and trying different actions to figure out its characteristics) will also work on later floors. In this way we hope to not only facilitate generalisation across different procedurally generated worlds but also generalisation of the exploration strategy through time over the learning process.
    - > the diversity in combat furthers the in-context learning el- ement provided by the procedural level generation — an agent that stumbles upon a strong weapon or armour piece should suitably change its strategy. This further extends the exploration problem as, by design, there should not be one fixed strategy (for instance, always putting experience points into strength and defeating enemies with melee attacks) that works on every level, meaning that an agent will have to explore a diverse range of strategies to achieve consistently high return.
    - > The player can find potions of varying colours spread over the 9 floors; however the effects of these potions are randomly permuted every episode. This means that an agent will need to discover which potions correspond to which effects through trial and error each episode,  further testing in-context learning and memory.
  - **Jax architecture**:
    - Up to 4,096 parallel environment workers; JIT compilation of entire pipeline eliminates CPU-GPU bottleneck.
    - Full environment state exposed as single functional object (enables UED methods directly).
    - **Optimistic environment resets** (Appendix C): since jax can't branch inside parallelized functions, both reset and step run every timestep. But resets are expensive. As an optimization, for N parallel workers, only M of those (M << N) have both resets and steps, and the other just step. For those that need sets, the initial states are sampled w/o replacement from those M based on the done bit. The only issue is that if the number of dones from the N parallel workers is larger than M at time t. Appendix C shows the probability of this occurring is extremely unlikely for M = 64 and N = 1024.
    - **Speed comparison** (steps/sec at best-case parallelism):
      - Craftax-Classic: 405,618 (4,096 workers)
      - Craftax: 266,961 (4,096 workers)
      - Procgen: 7,638 (1,024 workers)
      - NetHack: 5,628 (64 workers)
      - Crafter: 1,580 (1,024 workers)
  - **Observation space**:
    - Provides both pixel-based and symbolic observations.
    - Pixel: 63×63×3 (Craftax-Classic), 110×130×3 (Craftax).
    - Symbolic: 1,345-dim (Craftax-Classic), 8,268-dim (Craftax) — one-hot encodings for blocks/creatures + inventory.
    - Also supports textual observations for language-conditioned research.
  - **Action space**:
    - Discrete actions: 17 (Crafter-Classic), 43 (Craftax).
    - Every action can be taken at any timestep, so attempting an action without its specific prerequisites will effectively cause the agent to execute a no- op action, stepping the environment forward one timestep.
  - **Reward structure**: achievement-based tiers (Basic=1, Intermediate=3, Advanced=5, Very Advanced=8 pts) + 0.1/hp recovered, -0.1/damage taken.
  - **Evaluation Challenges**:
    - **Craftax-1B**: 1B steps on Craftax-Symbolic (~6 years of human gameplay at 5 steps/sec) — targets exploration, continual learning, and long-term planning and reasoning.
    - **Craftax-1M**: 1M steps on Craftax-Symbolic, finishes in seconds — for rapid iteration and sample efficiency research.
- **Experimental results** (Craftax-1B):
  - PPO: masters basic achievements, plateaus at intermediate.
  - PPO-RNN (with memory): best overall; significantly more dungeon entry than other methods.
  - Intrinsic motivation (RND, ICM, E3B): no improvement; E3B *reduced* reward — dense achievement rewards already provide sufficient signal.
  - No method makes appreciable progress onto floor 2 (Gnomish Mines).
  - Running 10B steps "barely improves performance" — likely an LR decay artifact, not algorithmic.
- **UED results** (Craftax-1B):
  - PLR > domain randomization.
  - Robust PLR < PLR (contradicts prior work — Craftax naturally generates high-quality, solvable levels so robustness curriculum is unnecessary).
  - ACCEL variants ≈ domain randomization.
  - ACCEL showed distribution shift: collected diamonds 30% of replay time vs 7% on normal levels.
- **Key takeaway**: no existing RL method makes meaningful progress on Craftax despite large compute budgets. Specialized exploration methods (intrinsic rewards, UED) provide no benefit. Memory (RNN) helps most.

## [2025] ScaleRL: The Art of Scaling Reinforcement Learning Compute for LLMs

- **Date**: 2026-02-22
- **Arxiv**: <https://arxiv.org/abs/2510.13786>
- **Paperpile**: <https://app.paperpile.com/view/?id=8e1e35ce-ae48-46f7-a33f-3c989ccc44a1>

---

- **Abstract**:
  - > Reinforcement learning (RL) has become central to training large language models (LLMs), yet the field lacks predictive scaling methodologies comparable to those established for pre-training. Despite rapidly rising compute budgets, there is no principled understanding of how to evaluate algorithmic improvements for scaling RL compute. We present the first large-scale systematic study, amounting to more than 400,000 GPU-hours, that defines a principled framework for analyzing and predicting RL scaling in LLMs. We fit sigmoidal compute-performance curves for RL training and ablate a wide range of common design choices to analyze their effects on asymptotic performance and compute efficiency. We observe: (1) Not all recipes yield similar asymptotic performance, (2) Details such as loss aggregation, normalization, curriculum, and off-policy algorithm primarily modulate compute efficiency without materially shifting the asymptote, and (3) Stable, scalable recipes follow predictable scaling trajectories, enabling extrapolation from smaller-scale runs. Combining these insights, we propose a best-practice recipe, ScaleRL, and demonstrate its effectiveness by successfully scaling and predicting validation performance on a single RL run scaled up to 100,000 GPU-hours. Our work provides both a scientific framework for analyzing scaling in RL and a practical recipe that brings RL training closer to the predictability long achieved in pre-training.
- **Intro**:
  - > Scaling reinforcement learning (RL) compute is emerging as a critical paradigm for advancing large language models (LLMs). While pre-training establishes the foundations of a model; the subsequent phase of RL training unlocks many of today’s most important LLM capabilities, from test-time thinking (OpenAI, 2024; Guo et al., 2025) to agentic capabilities (Kimi Team et al., 2025a).
  - > Deepseek-R1-Zero used 100,000 H800 GPU hours for RL training – 3.75% of its pre-training compute (Guo et al., 2025). This dramatic increase in RL compute is amplified across frontier LLM generations, with more than 10× increase from o1 to o3 (OpenAI, 2025) and a similar leap from Grok-3 to Grok-4 (xAI Team, 2025).
  - > This work lays the groundwork for science of RL scaling by borrowing from the well-established concept of scaling laws from pre-training. While pre-training has converged to algorithmic recipes that scale predictably with compute (Kaplan et al., 2020; Hoffmann et al., 2022; Owen, 2024), the RL landscape lacks a clear standard.
- **Setup**:
  - **Architecture**: generator–trainer split — generators handle rollout sampling via optimized inference kernels on one set of GPUs, trainers do policy updates via FSDP on another set of GPUs.
  - **Setup**: math reasoning domain (Polaris-53K). Sequence length 16,384 tokens (12,288 thinking + 2,048 solution + 2,048 prompt). Batch: 768 = 48 prompts × 16 generations.
    - 48 prompts (distinct math problems) × 16 independent completions each. The 16 completions per prompt are needed to compute the group baseline for GRPO — advantage is normalized relative to the other completions for the *same* prompt. If 1/16 is correct it gets large positive advantage; if 15/16 are correct the one failure gets large negative advantage.
  - **Base algorithm**: GRPO without KL, with asymmetric DAPO clipping to prevent entropy collapse.
    - Group-normalized advantage: $\hat{A}_i = r_i - \text{mean}(\{r_j\})$, $\hat{A}_i^G = \hat{A}_i / (\text{std}(\{r_j\}) + \epsilon)$
    - Importance sampling ratio: $\rho_{i,t}(\theta) = \pi_\text{train}(y_{i,t}) / \pi_\text{gen}(y_{i,t})$
    - Objective: $J(\theta) = \mathbb{E}\left[\frac{1}{G} \sum_i \min\left(\rho_{i,t}(\theta)\hat{A}_i^G,\ \text{clip}_\text{asym}(\rho_{i,t}(\theta))\hat{A}_i^G\right)\right]$
  - **GRPO vs PPO**: [[tutorial-rl-spinning-up-openai]]
    - Both use the same clipped surrogate objective. Key difference is the baseline.
    - PPO has a learned critic (value function $V(s)$); advantage $= R_t - V(s_t)$. For LLMs the critic must match the policy in size — doubles memory.
    - GRPO has no critic; the group mean reward *is* the baseline. No extra model needed.
    - GRPO requires multiple rollouts per prompt (to form the group); PPO does not.
    - GRPO signal is noisier (sparse outcome reward only) but much cheaper. Works well for LLM reasoning where reward is already sparse (final answer correctness) so a learned $V(s)$ wouldn't add much.
  - **Length control**: forcibly append end-of-thinking phrase to interrupt overly long generations, preventing instability.
  - **[Fig3] Scaling model** (Section 2.1): sigmoid over compute, not power-law (power-law is unbounded; reward is bounded): $R_C - R_0 = (A - R_0) \cdot \frac{1}{1 + (C_\text{mid}/C)^B}$ where
    - $A$: asymptotic pass rate (performance ceiling)
    - $B > 0$: compute efficiency (how fast you climb)
    - $C_\text{mid}$: midpoint of the curve
    - Three phases: slow growth → sharp acceleration → saturation
    - Curve fitting excludes very early regime ($< 1.5$k GPU-hours) for stability.
  - **Validation**: 1,000 held-out prompts from Polaris-53K, measured every 100 steps with 16 generations per prompt. Curves fit on validation to measure generalization, not training performance.
  - **Curve fitting**: $R_0$ is known (initial reward); fit for $A$, $B$, $C_\text{mid}$ jointly via nonlinear least squares: $\min_{A,B,C_\text{mid}} \sum_i \left(R_i - \left[R_0 + (A-R_0)\cdot\frac{1}{1+(C_\text{mid}/C_i)^B}\right]\right)^2$. Solved with Levenberg-Marquardt (e.g. `scipy.optimize.curve_fit`). $A$ is not observed — it's inferred from the shape of the partial curve.
- **Section 3: Empirical Study of RL Design Choices** (8B dense model):
  - **3.1 Asynchronous setup**:
    - *PPO-off-policy-k*: generate batch, do k gradient updates, repeat.
    - *PipelineRL-k*: generators stream continuously while trainers update immediately; generators use updated weights but stale cached KV from k steps ago. More off-policy but much faster wall-clock.
    - **Takeaway**: PipelineRL-8 substantially improves compute efficiency ($B$) with similar asymptote ($A$) → preferred infrastructure.
  - **3.2 Algorithmic choices** (what shifts $A$ vs only $B$):
    - **Loss type** (shifts $A$): DAPO < GSPO ≈ CISPO. CISPO (truncated importance sampling + vanilla policy gradient) shows longest linear reward growth → selected.
      - **DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)**: token-level importance sampling ratio $\rho_{i,t} = \pi_\text{train}(y_{i,t})/\pi_\text{gen}(y_{i,t})$ with asymmetric clip $[1-\varepsilon_\text{low}, 1+\varepsilon_\text{high}]$, $\varepsilon_\text{low} < \varepsilon_\text{high}$ (often $\varepsilon_\text{low}=0$). When $\rho$ exceeds upper bound, gradient = 0 — prevents entropy collapse but kills gradient signal when very off-policy.
      - **GSPO (Group Sequence Policy Optimization)**: sequence-level importance sampling ratio $\rho_i^\text{seq} = \prod_t \rho_{i,t}$. More principled (the whole sequence is the "action") but product of many token ratios has enormous variance for long sequences.
      - **CISPO (Clipped Importance Sampling Policy Optimization)**: token-level importance sampling but truncated — when $\rho_{i,t} > \tau$, drop the importance sampling correction and fall back to vanilla policy gradient (use $\hat{A}_i$ directly, no ratio). Uses importance sampling where it's reliable ($\rho \approx 1$), falls back to vanilla PG where it's not ($\rho \gg 1$). Avoids zero gradient (DAPO problem) and high variance (GSPO problem) when off-policy → produces cleaner gradient signal and longer linear growth before saturation.
    - **FP32 at LM head** (shifts $A$): numerical mismatch between generator/trainer kernels corrupts importance sampling ratios. FP32 fix raises $A$ from 0.52 → 0.61. Large effect.
    - **Loss aggregation** (shifts $B$ only): prompt-level > sample-level > token-level. Prompt-level selected.
    - **Advantage normalization** (shifts $B$ only): prompt-level vs batch-level vs none — all similar. Batch-level selected for theoretical soundness.
    - **Zero-variance filtering** (shifts $A$ slightly): prompts where all $G$ completions get same reward have zero policy gradient — filtering them improves efficiency and asymptote.
    - **No-positive-resampling / curriculum** (shifts $B$ only): permanently drop prompts with pass rate $\geq 0.9$ from future epochs. Improves scalability.
- **Section 4: ScaleRL Recipe**:
  - Full recipe: PipelineRL-8 + CISPO loss + FP32 at logits + prompt-level aggregation + batch-level normalization + zero-variance filtering + no-positive-resampling + length interruption.
  - **4.1 Leave-one-out (LOO) ablations** (16k GPU-hours each): removing any single component degrades efficiency or stability. FP32 fix appears less critical on 8B+CISPO but becomes essential on MoE and with other loss functions — robustness across settings justifies inclusion.
  - **4.2 Error margins**: 3 independent runs show $\pm 0.02$ variance in fitted $A$ — establishes significance threshold for comparing recipes.
  - **4.3 Extrapolation**: curves fit at 8k GPU-hours accurately predict performance at 16k GPU-hours, confirming predictability.
- **Section 5: Predictable Scaling Across Axes**:
  - **5.1 Model scale (17B×16 MoE)**: ScaleRL remains stable and predictable on larger MoE. Larger model achieves higher $A$ using only 1/6 of the RL compute needed by the 8B model.
  - **5.2 Generation length**: longer thinking budget (14k → 32k tokens) lowers early efficiency (higher $C_\text{mid}$, lower $B$) but raises $A$. **Long context is a ceiling-raising knob**, not just an efficiency tradeoff. Validated by extended runs.
  - **5.3 Batch size**: larger batches (up to 2,048 prompts) → higher $A$ and cleaner scaling. Smaller batches stagnate on downstream evals even when in-distribution validation looks fine.
  - **5.4 Generations per prompt** (fixing total batch size): varying $G \in \{8,16,24,32\}$ with matching prompt counts produces essentially identical scaling curves — second-order consideration at moderate batch scales.
  - **100k GPU-hour validation**: sigmoid fit from 50k GPU-hours accurately predicted final performance, confirming the whole framework.

## 2019-04 Reading List

1. Beginner's introduction to RL and Deep Q-Learning (DQN): <https://www.intel.ai/demystifying-deep-reinforcement-learning/#gs.ac37fu>
2. Fundamentals of Policy Gradients (forms the basis of IMPALA): <http://karpathy.github.io/2016/05/31/rl>
3. John Schulman's 4-part lecture series on Policy Gradients (lectures 2 and 3 are particularly relevant): <https://www.youtube.com/@mlsscadiz4148/search?query=john%20schulman>
4. A deeper explanation of the theory and the equations underneath Policy Gradients (follows lectures 2 and 3, a very useful read-along): <https://danieltakeshi.github.io/2017/03/28/going-deeper-into-reinforcement-learning-fundamentals-of-policy-gradients/>
5. Actor-Critic Methods:
    <https://towardsdatascience.com/understanding-actor-critic-methods-931b97b6df3f>
    <http://rail.eecs.berkeley.edu/deeprlcourse-fa17/f17docs/lecture_5_actor_critic_pdf.pdf>
6. A high-level overview/refresher of everything above (Markov Decision Processes, Temporal-Difference Learning, DQN, various Policy Gradient algorithms, Actor-Critic Methods, etc.): <https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html>
7. IMPALA Paper: <https://arxiv.org/abs/1802.01561>
8. TorchBeast code: <https://github.com/fairinternal/torchbeast>
9. TRPO
10. PPO
11. Multi-armed bandits: <https://lilianweng.github.io/lil-log/2018/01/23/the-multi-armed-bandit-problem-and-its-solutions.html>
12. Kris Jensen's [An introduction to reinforcement learning for neuroscience, 2023](https://arxiv.org/abs/2311.07315)
