# Continual Learning

- **Created**: 2026-03-01
- **Last Updated**: 2026-04-18
- **Status**: `In Progress`

---

The umbrella topic of learning from a non-stationary stream of tasks. Three sub-challenges that get conflated in the literature:

- **Catastrophic forgetting**: performance on old tasks drops when training on new ones. The classic concern; most methods target this (EWC, progressive nets, synaptic intelligence, replay).
- **Forward transfer**: new tasks should be *faster* to learn because of prior experience. Often ignored by forgetting-focused methods, which buy stability at the cost of transfer (e.g., PackNet has zero forward transfer by construction).
- **Loss of plasticity**: even without forgetting, standard SGD networks lose the *ability to learn* over long sequences. Orthogonal to forgetting and not addressed by most CL methods (Sutton et al. 2024).

---

- [ ] [1989] Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem - [paper](https://www.sciencedirect.com/science/chapter/bookseries/abs/pii/S0079742108605368)
- [ ] [2016] [deepmind] EWC: Overcoming catastrophic forgetting in neural networks - [paper](https://arxiv.org/abs/1612.00796)
- [ ] [2016] [deepmind] Progressive Neural Networks - [paper](https://arxiv.org/abs/1606.04671)
- [ ] [2017] Learning without Forgetting - [paper](https://arxiv.org/abs/1606.09282)
- [ ] [2017] [SuryaGanguli] Continual Learning Through Synaptic Intelligence - [paper](https://arxiv.org/abs/1703.04200)
- [ ] [2019] [deepmind] CLEAR: Experience Replay for Continual Learning - [paper](https://arxiv.org/abs/1811.11682)
- [ ] [2019] A-GEM: Efficient Lifelong Learning with A-GEM - [paper](https://arxiv.org/abs/1812.00420)
- [ ] [2020] PackNet: Adding Multiple Tasks to a Single Network by Iterative Pruning - [paper](https://arxiv.org/abs/1711.05769)
- [ ] [2021] Continual World: A Robotic Benchmark For Continual Reinforcement Learning - [paper](https://arxiv.org/abs/2105.10919)
- [ ] [2021] OWL: Same State, Different Task — Continual Reinforcement Learning without Interference - [paper](https://arxiv.org/abs/2106.02940)
- [ ] [2023] [deepmind] A Definition of Continual Reinforcement Learning - [paper](https://arxiv.org/abs/2307.11046)
- [ ] [2024] [sutton] Loss of plasticity in deep continual learning - [paper](https://www.nature.com/articles/s41586-024-07711-7)

---

## [2016] [deepmind] EWC: Overcoming catastrophic forgetting in neural networks

- **Date**: 2026-03-04
- **Arxiv**: <https://arxiv.org/abs/1612.00796>
- **Paperpile**: <https://app.paperpile.com/view/?id=c7eb2d35-0bef-4d45-b908-d6479731e72c>

---

- **Abstract**:
  - > **The ability to learn tasks in a sequential fashion is crucial to the development of artificial intelligence**. Neural networks are not, in general, capable of this and it has been widely thought that catastrophic forgetting is an inevitable feature of connectionist models. We show that it is possible to overcome this limitation and train networks that can maintain expertise on tasks which they have not experienced for a long time. Our approach remembers old tasks by selectively slowing down learning on the weights important for those tasks. We demonstrate our approach is scalable and effective by solving a set of classification tasks based on the MNIST hand written digit dataset and by learning several Atari 2600 games sequentially.
- TODO

## [2021] Continual World: A Robotic Benchmark For Continual Reinforcement Learning

- **Date**: 2026-04-18
- **Arxiv**: <https://arxiv.org/abs/2105.10919>
- **Website**: <https://sites.google.com/view/continualworld>
- **Code**: <https://github.com/awarelab/continual_world>

---

- **Abstract**:
  - > Continual learning (CL) --- the ability to continuously learn, building on previously acquired knowledge --- is a natural requirement for long-lived autonomous reinforcement learning (RL) agents. While building such agents, one needs to balance opposing desiderata, such as constraints on capacity and compute, the ability to not catastrophically forget, and to exhibit positive transfer on new tasks. Understanding the right trade-off is conceptually and computationally challenging, which we argue has led the community to overly focus on the catastrophic forgetting problem. In response to these issues, we advocate for the need to prioritize forward transfer and propose Continual World, a benchmark consisting of realistic and meaningfully diverse robotic tasks built on top of Meta-World as a testbed. Following an in-depth empirical evaluation of existing CL methods, we pinpoint their limitations and highlight unique algorithmic challenges in the continual RL domain.
- **Motivation**:
  - Continual-learning benchmarks prior to this were dominated by sequential image classification (split-MNIST, split-CIFAR, permuted-MNIST). Classification benchmarks reward *not forgetting* above all else, because there's nothing interesting to transfer forward between "recognize 0–4" and "recognize 5–9".
  - RL is different: the tasks in a realistic sequence share dynamics, contact primitives, and motor skills. An agent that doesn't forget but also doesn't *transfer* is missing the whole point. The paper's thesis is that the CL community was overfitting to forgetting metrics and under-measuring forward transfer.
- **Benchmark design**:
  - Built on Meta-World v2, reusing the Sawyer arm and 4D action space (so all tasks share an embodiment).
  - **CW10**: fixed sequence of 10 manipulation tasks, 1M environment steps per task, total 10M steps.
  - **CW20**: CW10 repeated twice — tests whether a second pass improves over the first (a "did the agent actually learn to learn?" signal).
  - Task sequences chosen for moderate similarity: enough shared structure that forward transfer is *possible* in principle, without being trivial.
  - **Metrics**:
    - *Average performance*: success rate over all tasks at end of training.
    - *Forgetting*: drop in success on task $i$ between end of training on $i$ and end of full sequence.
    - *Forward transfer*: normalized advantage of a continual learner over training task $i$ from scratch, when it arrives at task $i$ after $i-1$ previous tasks.
      - Concretely: compare two learning curves on task $i$ — a reference SAC trained only on $T_i$ from scratch, and the continual learner arriving at $T_i$ after prior training on $T_1, \ldots, T_{i-1}$. Both get the same per-task budget (1M steps).
      - Formula: $FT_i = \dfrac{\mathrm{AUC}^{\text{continual}}_i - \mathrm{AUC}^{\text{ref}}_i}{1 - \mathrm{AUC}^{\text{ref}}_i}$, where AUC is the area under the per-step success curve on $T_i$, normalized to $[0, 1]$.
      - Interpretation: $FT_i = 1$ → instant mastery (continual learner already at ceiling when $T_i$ starts). $FT_i = 0$ → prior experience was neutral. $FT_i < 0$ → interference; prior training made $T_i$ harder than starting fresh.
      - The denominator $(1 - \mathrm{AUC}^{\text{ref}}_i)$ normalizes by *room for improvement*, so tasks the reference already nearly solves don't dominate the metric.
      - This metric distinguishes "representation sharing helped" from "prior experience was irrelevant" from "catastrophic interference". PackNet scores $FT \approx 0$ by construction (isolated subnets per task). Naïve fine-tuning scores $FT > 0$ on CW10 — counter to classification-dominated CL folklore.
  - Separating these three metrics is the main methodological contribution — prior work collapsed them.
- **Methods evaluated**:
  - Baselines: fine-tuning (SAC continuing on each new task), single-task oracle, multi-task oracle.
  - Regularization: EWC, L2, MAS.
  - Rehearsal: A-GEM, Perfect Memory (replay everything).
  - Architectural: PackNet, Progress & Compress.
- **Key findings**:
  - Fine-tuning suffers catastrophic forgetting as expected, but its *forward transfer* is non-trivially positive — naïve fine-tuning genuinely benefits from prior tasks in the sequence, counter to the classification-dominated CL folklore.
  - PackNet prevents forgetting nearly perfectly but has *zero* forward transfer by construction — each task gets an isolated subnet. Forgetting-only methods buy stability at the cost of the thing RL benchmarks care most about.
  - EWC and regularization methods show limited benefit — in RL the reward landscape is non-stationary enough that the Fisher-information-based importance weights are noisy.
  - Perfect Memory (full replay) is strong but assumes unbounded storage, which violates the CL setting. A-GEM approximates it more cheaply and does well.
  - **The oracle gap is small**: the multi-task upper bound isn't dramatically better than sequential fine-tuning on CW10. This is a striking result — it says either (a) multi-task RL itself is weak at this scale, or (b) sequential training with forgetting is less costly than assumed. Probably both.
- **Current status (as of 2026)**:
  - Continual World remains the standard continual-RL benchmark, alongside DeepMind's more recent formal definition work.
  - The field has partially moved away from "train a single network sequentially" framing:
    - **Foundation model + adapter/LoRA**: each new task gets a lightweight adapter; the frozen base provides stability, adapters provide plasticity. Architecturally adjacent to PackNet but scales better.
    - **In-context learning**: tasks become prompts rather than sequential gradient updates. Sidesteps forgetting entirely, but the "continual" framing evaporates — the model is frozen, so it can't really *learn* a new task, only condition on demonstrations of one.
    - **Meta-learning as inoculation**: train on diverse task distributions up front so that downstream adaptation is cheap; avoid the sequential-update regime.
  - **Unsolved**: continual RL with a bounded-parameter network, long task horizons, and genuine forward transfer. Sutton et al.'s "loss of plasticity" (Nature 2024) showed that even standard SGD networks *lose the ability to learn* on long sequences — a failure mode orthogonal to forgetting that most CL methods don't address.
  - Worth reading *in conjunction with* Meta-World's v1→v2 reward lessons: Continual World inherits Meta-World's reward structure, so conclusions about forward transfer are only as clean as the underlying reward functions. Papers that reran on v2 got meaningfully different numbers than v1.
