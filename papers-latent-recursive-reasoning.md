# Latent / Recursive Reasoning

- **Created**: 2026-06-15
- **Last Updated**: 2026-06-15
- **Status**: `In Progress`
- **Related**:
  - [[papers-diffusion-models]]
  - [[papers-post-training]]

---

## Recursive Reasoning Models

- [x] [2025] HRM: Hierarchical Reasoning Model - [paper](https://arxiv.org/abs/2506.21734)
- [x] [2025] The Hidden Drivers of HRM's Performance on ARC-AGI - [blog](https://arcprize.org/blog/hrm-analysis)
- [ ] [2025] TRM: Less is More: Recursive Reasoning with Tiny Networks - [paper](https://arxiv.org/abs/2510.04871), [code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
- [ ] [2025] Tiny Recursive Models on ARC-AGI-1: Inductive Biases, Identity Conditioning, and Test-Time Compute - [paper](https://arxiv.org/abs/2512.11847)
- [ ] [2025] URM: Universal Reasoning Model - [paper](https://arxiv.org/abs/2512.14693), [code](https://github.com/zitian-gao/URM)
- [ ] [2026] GRAM: Generative Recursive Reasoning - [paper](https://arxiv.org/abs/2605.19376), [site](https://ahn-ml.github.io/gram-website/)

## Adaptive Computation / Implicit Depth

- [ ] [2016] [Graves] ACT: Adaptive Computation Time for Recurrent Neural Networks - [paper](https://arxiv.org/abs/1603.08983)
- [ ] [2019] DEQ: Deep Equilibrium Models - [paper](https://arxiv.org/abs/1909.01377)
- [ ] [2021] PonderNet: Learning to Ponder - [paper](https://arxiv.org/abs/2107.05407)

## Recurrent / Looped Transformer Architectures

- [ ] [2018] Universal Transformers - [paper](https://arxiv.org/abs/1807.03819)
- [ ] [2023] Looped Transformers as Programmable Computers - [paper](https://arxiv.org/abs/2301.13196)
- [ ] [2023] Looped Transformers are Better at Learning Learning Algorithms - [paper](https://arxiv.org/abs/2311.12424)
- [ ] [2024] Looped Transformers for Length Generalization - [paper](https://arxiv.org/abs/2409.15647)
- [ ] [2025] Reasoning with Latent Thoughts: On the Power of Looped Transformers - [paper](https://arxiv.org/abs/2502.17416)
- [ ] [2025] Scaling Latent Reasoning via Looped Language Models - [paper](https://arxiv.org/abs/2510.25741), [site](http://ouro-llm.github.io/)
- [ ] [2026] LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation - [paper](https://arxiv.org/abs/2602.11451), [site](https://loopformer.github.io/)

## Latent-CoT / Recurrent-Depth LMs

- [ ] [2025] Coconut: Training Large Language Models to Reason in a Continuous Latent Space - [paper](https://arxiv.org/abs/2412.06769)
- [ ] [2025] Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning - [paper](https://arxiv.org/abs/2505.16782)
- [ ] [2025] Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach - [paper](https://arxiv.org/abs/2502.05171)

## Algorithmic Extrapolation / Constraint Solving

- [ ] [2021] Can You Learn an Algorithm? Generalizing from Easy to Hard Problems with Recurrent Networks - [paper](https://arxiv.org/abs/2106.04537)
- [ ] [2022] End-to-End Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking - [paper](https://arxiv.org/abs/2202.05826)
- [ ] [2019] PDP: A General Neural Framework for Learning Constraint Satisfaction Solvers - [paper](https://arxiv.org/abs/1903.01969)

## [2025] HRM: Hierarchical Reasoning Model

- **Date**: 2026-06-15
- **Arxiv**: <https://arxiv.org/abs/2506.21734>
- **Code**: <https://github.com/sapientinc/HRM>

---

- **Abstract**:
  - > Reasoning, the process of devising and executing complex goal-oriented action sequences, remains a critical challenge in AI. Current large language models (LLMs) primarily employ Chain-of-Thought (CoT) techniques, which suffer from brittle task decomposition, extensive data requirements, and high latency. Inspired by the hierarchical and multi-timescale processing in the human brain, we propose the Hierarchical Reasoning Model (HRM), a novel recurrent architecture that attains significant computational depth while maintaining both training stability and efficiency. HRM executes sequential reasoning tasks in a single forward pass without explicit supervision of the intermediate process, through two interdependent recurrent modules: a high-level module responsible for slow, abstract planning, and a low-level module handling rapid, detailed computations. With only 27 million parameters, HRM achieves exceptional performance on complex reasoning tasks using only 1000 training samples. The model operates without pre-training or CoT data, yet achieves nearly perfect performance on challenging tasks including complex Sudoku puzzles and optimal path finding in large mazes. Furthermore, HRM outperforms much larger models with significantly longer context windows on the Abstraction and Reasoning Corpus (ARC), a key benchmark for measuring artificial general intelligence capabilities. These results underscore HRM's potential as a transformative advancement toward universal computation and general-purpose reasoning systems.
- **Killer Figs**:
  - **Fig 1**: Headline. Left = the brain-inspired two-timescale architecture; right = HRM (~27M params, ~1000 examples) beating SOTA CoT models on ARC-AGI and solving Sudoku-Extreme / Maze-Hard where CoT models score ~0.
  - **Fig 2**: Accuracy vs computational depth — HRM beats plain and recurrent Transformers at every depth; also shows deeper Transformers help on Sudoku but plateau far from optimal (the "shallow ceiling").
  - **Fig 3**: The evidence *for* hierarchical convergence — HRM keeps forward-pass activity (residual norm) high across many steps while a standard RNN's decays toward 0 (premature convergence).
  - **Fig 5**: ACT — (a) average compute drops vs a fixed-$M_{\max}$ model at matched accuracy; (c) inference-time scaling: raising $M_{\max}$ at test time lifts accuracy (strong on Sudoku, marginal on ARC) with no retraining.
  - **Fig 7**: Decoded intermediate predictions → task-specific emergent strategies (maze = parallel explore/prune, Sudoku = DFS + backtracking, ARC = hill-climbing).
  - **Fig 8**: Brain correspondence — dimensionality (Participation Ratio) hierarchy $z_H \gg z_L$ in trained HRM mirroring mouse cortex; untrained control shows no separation (so it's emergent).
- **Intro / Motivation**:
  - CoT reasoning externalizes intermediate steps into the token stream, which the abstract calls out as suffering from **brittle task decomposition, extensive data requirements, and high latency**. HRM instead pushes the reasoning into the model's hidden state ("latent reasoning"):
    - > we explore "latent reasoning", where the model conducts computations within its internal hidden state space. This aligns with the understanding that language is a tool for human communication, not the substrate of thought itself; the brain sustains lengthy, coherent chains of reasoning with remarkable efficiency in a latent space, without constant translation back to language. However, the power of latent reasoning is still fundamentally constrained by a model's effective computational depth.
  - **Central problem = effective computational depth**. A fixed-layer transformer has shallow depth; you can't add reasoning steps without adding layers. Naively recurring a single block (a standard RNN / looped transformer) is the obvious fix but fails because of premature convergence (see below). HRM is the mechanism to get large effective depth from a small, recurrently-applied network while staying stable.
  - **Brain inspiration**: hierarchical, multi-timescale processing — slow abstract planning in higher cortical areas, fast detailed computation in lower areas. This motivates the two-module design.
- **Architecture — two coupled recurrent modules**:
  - **High-level module $f_H$ (slow)**: abstract planning / strategy. Updated infrequently.
  - **Low-level module $f_L$ (fast)**: rapid detailed computation / search. Updated every step.
  - Both modules are transformer blocks used as recurrent cells, maintaining states $z_H$ and $z_L$. Input $x$ is injected via an input embedding.
  - **Nested timing**: computation runs for $N$ high-level cycles, each containing $T$ low-level steps.
    - Within a cycle, $f_L$ runs $T$ steps refining $z_L$, **conditioned on a fixed $z_H$** (and the input).
    - At the end of the cycle, $f_H$ performs a single update, consuming the final $z_L$ to produce a new $z_H$.
  - A small output head reads off the prediction from $z_H$ (or $z_L$) at the end.
- **Hierarchical convergence (the key idea)**:
  - The failure mode HRM is designed around — standard RNNs converge too early and stall:
    - > Hierarchical convergence Although convergence is crucial for recurrent networks, standard RNNs are fundamentally limited by their tendency to converge too early. As the hidden state settles toward a fixed point, update magnitudes shrink, effectively stalling subsequent computation and capping the network's effective depth. To preserve computational power, we actually want convergence to proceed very slowly–but engineering that gradual approach is difficult, since pushing convergence too far edges the system toward instability.
  - HRM's fix — let $f_L$ converge *within* a cycle, then have $f_H$ reset the context so $f_L$ starts a fresh convergence toward a new equilibrium:
    - > HRM is explicitly designed to counteract this premature convergence through a process we term hierarchical convergence. During each cycle, the L-module (an RNN) exhibits stable convergence to a local equilibrium. This equilibrium, however, depends on the high-level state zH supplied during that cycle. After completing the T steps, the H-module incorporates the sub-computation's outcome (the final state zL) and performs its own update. This zH update establishes a fresh context for the L-module, essentially "restarting" its computational path and initiating a new convergence phase toward a different local equilibrium.
  - Net effect: a sequence of nested, stable computations with effective depth $\sim N \cdot T$ rather than the $\sim T$ of a plain RNN:
    - > This process allows the HRM to perform a sequence of distinct, stable, nested computations, where the H-module directs the overall problem-solving strategy and the L-module executes the intensive search or refinement required for each step. Although a standard RNN may approach convergence within T iterations, the hierarchical convergence benefits from an enhanced effective depth of N T steps. As empirically shown in Figure 3, this mechanism allows HRM both to maintain high computational activity (forward residual) over many steps (in contrast to a standard RNN, whose activity rapidly decays) and to enjoy stable convergence. This translates into better performance at any computation depth, as illustrated in Figure 2.
  - **Intuition**: the L-module is an inner optimization loop solving a subproblem to local equilibrium; the H-module is the outer loop that, each time the inner loop settles, takes a step and hands the inner loop a new subproblem. The "restart" keeps forward-pass activity (residual norm) high instead of decaying to zero, which is what buys the depth.
- **Approximate gradient (1-step gradient, no BPTT)**:
  - **The problem with the naive approach (BPTT)**: a recurrent module applies the *same* function repeatedly, $z_{t+1} = f(z_t, x;\, \theta)$. To train it the obvious way you **unroll** the whole sequence $z_0 \to z_1 \to \dots \to z_T$ and backpropagate through every step (backpropagation through time). That means storing every intermediate state and chaining the Jacobian through all of them — $O(N T)$ memory/compute here, and increasingly unstable (vanishing/exploding gradients) as the number of steps grows. This is exactly the cost HRM wants to avoid, since its whole pitch is *many* recurrent steps.
  - **Fixed point**: if you keep applying $f$ and the state stops changing, you've reached a fixed point $z^\star$ where applying $f$ again does nothing:
    $$z^\star = f(z^\star, x;\, \theta)$$
    The paper's framing:
    - > Fortunately, if a recurrent neural network converges to a fixed point, we can avoid unrolling its state sequence by applying backpropagation in a single step at that equilibrium point.
  - **Key fact (implicit function theorem)**: at a fixed point the gradient depends *only on $z^\star$ and the local derivatives there* — not on the trajectory $z_0, z_1, \dots$ or how many steps it took. Differentiate $z^\star = f(z^\star, x; \theta)$ w.r.t. $\theta$:
    $$\frac{dz^\star}{d\theta} = J\,\frac{dz^\star}{d\theta} + \frac{\partial f}{\partial \theta} \;\;\Longrightarrow\;\; \frac{dz^\star}{d\theta} = (I - J)^{-1}\frac{\partial f}{\partial \theta}, \qquad J = \tfrac{\partial f}{\partial z}\big|_{z^\star}$$
    The right-hand side has no dependence on the path, so the unrolled history carries no extra gradient information once you've converged — that is *why* you're allowed to throw the trajectory away.
  - **The exact solution is still expensive**: computing $(I - J)^{-1}$ is a linear solve / inverse-Jacobian at the fixed point. **Deep Equilibrium Models (DEQ)** do exactly this — solve for the fixed point in the forward pass and the exact $(I-J)^{-1}$ in the backward pass.
  - **HRM's cheap 1-step approximation**: approximate $(I - J)^{-1} \approx I$, which collapses the gradient to
    $$\frac{dz^\star}{d\theta} \approx \frac{\partial f}{\partial \theta}\Big|_{z^\star}$$
    In code this is just **one backward pass through a single application of $f$**, run at the converged state, with the *input* state $z^\star$ treated as a constant (`detach()`ed). Feed $z^\star$ in → apply $f$ once → backprop through that one step. No unrolling, nothing earlier stored.
    - Why $(I-J)^{-1}\approx I$: the exact inverse is the Neumann series $(I-J)^{-1} = I + J + J^2 + \dots$; keeping only the leading $I$ term *is* the 1-step truncation. Intuitively, at convergence the last step's input ($z^\star$) ≈ its output, so a single application is a faithful local linearization and the earlier steps have "washed out."
  - **What you gain**: $O(1)$ memory in the number of recurrent steps (the trajectory is never stored — this is what lets HRM recurse cheaply for many steps), better stability than deep BPTT, and a more biologically plausible update (no stored forward history to replay backward).
  - **The catch — it's a biased gradient**, approximate twice over: (1) the recurrence may not have *exactly* reached $z^\star$, and (2) dropping all $J, J^2, \dots$ terms discards real curvature. Works well in practice for HRM/TRM, but expect trouble at the regime where the inner loop hasn't actually settled.
  - **One-liner**: same fixed-point / implicit-depth view as DEQ [[Deep Equilibrium Models]], but DEQ pays for the exact $(I-J)^{-1}$ while HRM takes the 1-step ($\approx I$) shortcut.
- **Deep supervision**:
  - Motivated by the idea that **periodic neural oscillations regulate when learning happens in the brain** — learning is gated to happen at intervals, not continuously.
  - **Setup**: a "segment" = one full forward pass of HRM (i.e. the entire $N$ cycles $\times\ T$ steps). For a sample $(x, y)$ the model runs $M$ segments in sequence. At each segment $m$:
    1. Forward pass from the previous segment's state: $(z_m, \hat{y}_m) \leftarrow \text{HRM}(z_{m-1}, x;\, \theta)$
    2. Loss for this segment: $L_m \leftarrow \text{LOSS}(\hat{y}_m, y)$
    3. Optimizer step: $\theta \leftarrow \text{OptimizerStep}(\theta, \nabla_\theta L_m)$
  - **The crucial detail — detach between segments**: the hidden state $z_m$ is **detached from the computation graph** before being fed in as the input state to segment $m{+}1$. So the recurrent *state* $(z_H, z_L)$ persists across segments, but **gradients do not flow from segment $m{+}1$ back into segment $m$**.
    - This makes the *outer* deep-supervision loop itself a 1-step gradient approximation (stacked on top of the inner 1-step gradient of §approximate-gradient). Net effect: a sample is supervised $M$ separate times, each backprop local to one segment.
  - **Why it helps**: gives the H-module **more frequent feedback** and acts as a **regularizer**; the paper reports it empirically outperforms (and is more stable than) the more complex Jacobian-based regularization used in deep-equilibrium models.
- **Adaptive computational time (ACT)**:
  - Motivated by **"thinking, fast and slow"** — the brain alternates System-1 (automatic) and System-2 (deliberate) reasoning over shared circuits (PFC, default mode network), modulating "runtime" by task complexity and reward. ACT lets HRM spend variable compute per input. Conceptually ACT (Graves) / PonderNet [[PonderNet]], but framed as an **RL halt/continue decision with a learned Q-head** rather than a differentiable ponder cost.
  - **Q-head**: from the final H-state of each segment, predict two Q-values via a sigmoid: $\hat{Q}_m = (\hat{Q}^{\text{halt}}_m, \hat{Q}^{\text{continue}}_m) = \sigma(\theta_Q^\top z_H^{mNT})$.
  - **Halting rule**: halt when the segment count exceeds the max $M_{\max}$ (fixed hyperparam), **or** when $\hat{Q}^{\text{halt}} > \hat{Q}^{\text{continue}}$ *and* count has reached the minimum $M_{\min}$.
    - $M_{\min}$ is a **random variable for exploration**: with prob. $\varepsilon$ it's sampled uniformly from $\{2,\dots,M_{\max}\}$ (forces longer thinking), else it's $1$.
  - **The MDP it's solving**: state $= z_m$; actions $= \{\text{halt}, \text{continue}\}$. "halt" ends the episode with **binary reward $\mathbb{1}\{\hat{y}_m = y\}$** (was the prediction correct); "continue" gives reward $0$ and transitions to $z_{m+1}$. Q-learning targets:
    - $\hat{G}^{\text{halt}}_m = \mathbb{1}\{\hat{y}_m = y\}$
    - $\hat{G}^{\text{continue}}_m = \hat{Q}^{\text{halt}}_{m+1}$ if $m \ge N_{\max}$, else $\max(\hat{Q}^{\text{halt}}_{m+1}, \hat{Q}^{\text{continue}}_{m+1})$
  - **Combined loss** per segment (added to deep supervision): $L^{\text{ACT}}_m = \text{LOSS}(\hat{y}_m, y) + \text{BCE}(\hat{Q}_m, \hat{G}_m)$ — i.e. the seq-to-seq loss plus a Q-head loss, so it learns accurate predictions *and* near-optimal stopping at once.
  - **Batching trick**: halted samples in a batch are swapped out for fresh samples from the dataloader.
  - **Payoff (Fig 5)**: ACT matches a fixed-$M_{\max}$ model's accuracy while spending much less average compute — adapts depth to difficulty.
  - **"Is this really RL?" — what the MDP actually buys** (non-obvious on a reread): it *is* Q-learning, but a degenerate MDP — deterministic transitions ($z_{m+1} = \text{HRM}(z_m, x)$), two actions, zero reward except a terminal correctness bit. It's really an **optimal-stopping** problem ("keep thinking or stop?"). The two Q-heads have very different characters:
    - **Halt head is just a classifier.** Its target $\hat{G}^{\text{halt}}_m = \mathbb{1}\{\hat{y}_m = y\}$ is a supervised binary label — "is my current answer already correct?" A learned **verifier**, no RL needed.
    - **Continue head is the only genuinely sequential piece.** "Is another segment worth it?" depends on *what would happen if I continued* — a label you don't have at segment $m$. The Bellman bootstrap $\hat{G}^{\text{continue}}_m = \max(\hat{Q}^{\text{halt}}_{m+1}, \hat{Q}^{\text{continue}}_{m+1})$ is the minimal trick to estimate that without an oracle. **This look-ahead is the whole reason for the MDP framing.**
    - **Why not pure supervision?** You could unroll all $M_{\max}$ segments and supervise "halt at the first correct one," but that's a full unroll per sample and a static label. Q-learning bootstraps the continue-value cheaply, adapts online as the model improves, and piggybacks on deep supervision (just one extra BCE term per segment).
    - **Arguably overcomplicated.** **PonderNet / Graves' ACT** [[PonderNet]] solve the identical halting problem with **no RL** — a differentiable halting distribution + a ponder-cost regularizer, trained by plain backprop. That HRM needs a whole paragraph defending Q-learning stability (no replay buffer / target network; relies on Post-Norm + RMSNorm + AdamW) is a tell that the tool is heavier than the job strictly requires.
- **Inference-time scaling**:
  - Because computation is decoupled from parameter count, HRM can be given **more compute at test time than it used during training** — run more segments / increase the number of high-level cycles $N$ — to push accuracy on harder instances, with **no retraining and no architectural change**.
  - (This is the lever TRM later pushes on — e.g. increasing the outer-step count at inference to trade compute for accuracy.)
- **Architecture / stability details**:
  - Both $f_L$ and $f_H$ are **encoder-only Transformer blocks, identical architecture/dims**, taking multiple inputs combined by simple **element-wise addition** (gating left to future work).
  - Blocks use modern-LLM (Llama-style) components: **Rotary Positional Encoding (RoPE), Gated Linear Units, RMSNorm, no bias terms**, **Post-Norm** placement, **truncated LeCun-normal init**, and the **Adam-atan2** optimizer (scale-invariant Adam) with constant LR + linear warm-up. Initial states $z_0$ sampled once from a truncated normal and kept fixed.
  - **Why ACT's Q-learning is stable without the usual crutches**: deep Q-learning normally needs replay buffers / target networks, which HRM omits. Instead it leans on a result (Gallici et al.) that Q-learning converges if parameters are **bounded**, **weight decay** is used, and **post-norm** layers are present — exactly what the **Post-Norm + RMSNorm + AdamW** setup provides (AdamW implicitly solves an $L_\infty$-constrained problem, bounding params by $1/\lambda$).
- **Results (§3)**:
  - **Headline**: ~27M params, ~1000 training examples per task, random init, **no pretraining and no CoT labels**, trained seq-to-seq on flattened input/output grids.
  - **Benchmarks (§3.1)**:
    - **ARC-AGI-1 / ARC-AGI-2**: inductive IQ-test-style grid puzzles; must infer an abstract rule from 2–3 demos and apply to a test input (2 attempts). ARC-AGI-2 adds deeper compositional / multi-step / symbolic tasks.
    - **Sudoku-Extreme**: their new hard dataset (mean **22 backtracks/puzzle** by the `tdoku` solver, vs ~0.45 for prior "hard" sets like Sudoku-Bench). 1000-example subset for the small-sample setting; Sudoku-Extreme-Full (~3.8M) for analysis. Strict split so test puzzles aren't transforms of train.
    - **Maze-Hard**: optimal pathfinding in 30×30 mazes, filtered to shortest-path length > 110. Correct = valid *and* optimal. 1000 train / 1000 test.
  - **Evaluation (§3.2)**: ARC uses heavy augmentation (translations/rotations/flips/color perms) + a **learnable per-puzzle token**; at test time solve 1000 augmented variants, invert, and majority-vote the top-2. Sudoku augments with band/digit perms; Maze uses no augmentation. Sudoku/Maze are single inference pass.
    - **Baselines**: "Direct pred" = same setup but a plain 8-layer Transformer (HRM-sized). It **fails entirely** on Sudoku-Extreme / Maze-Hard at 1000 examples (only reaches 16.9% on Sudoku with the full ~3.8M set). A 175M Transformer trained on 1M maze examples stayed below 20% (pass@64). Swapping the Transformer for HRM + ACT gives **>2×** on ARC-AGI-1 over Direct-pred.
  - **Visualization of intermediate timesteps (§3.3)** — what algorithm is HRM running? Decode the prediction $\bar{y}_i$ at each timestep and watch it evolve. The strategy is **task-dependent and emergent**:
    - **Maze**: explores several candidate paths in parallel, prunes blocked/inefficient ones, sketches an outline, then refines.
    - **Sudoku**: resembles **depth-first search with backtracking** — tries solutions, hits dead ends, backs out.
    - **ARC**: **hill-climbing** — incremental board adjustments, steady progression, little backtracking.
    - Take-away: one trained network picks an effective strategy per task rather than running a fixed procedure.
- **Brain correspondence (§4) — emergent dimensionality hierarchy**:
  - **Premise from systems neuroscience**: a region's functional repertoire correlates with the **effective dimensionality** of its representations; higher-order areas (flexible, many tasks) sit in higher-dimensional state spaces → an observable "dimensionality hierarchy."
  - **Metric — Participation Ratio (PR)**: $\mathrm{PR} = \dfrac{\left(\sum_i \lambda_i\right)^2}{\sum_i \lambda_i^2}$ over eigenvalues $\{\lambda_i\}$ of the covariance of neural trajectories. High PR = variance spread across many dims; low PR = concentrated in a few.
  - **In mouse cortex**, PR rises monotonically from low-level sensory to high-level associative areas (Spearman $\rho = 0.79$).
  - **In trained HRM** (on Sudoku-Extreme-Full): low-level $z_L$ has **PR ≈ 30.2**, high-level $z_H$ has **PR ≈ 90.0** — a clear hierarchy, $z_H$ in a much larger subspace. Scaling tasks 10→100 grows $z_H$ dimensionality while $z_L$ stays flat.
  - **Control (untrained net)**: $z_L$ PR ≈ 42.1, $z_H$ PR ≈ 40.7 — **no separation**. So the hierarchy is a **learned, emergent property of training**, not an architectural artifact.
  - $z_H/z_L$ PR ratio ≈ **2.98** ≈ mouse cortex's ≈ 2.25. Contrasts with **neural collapse** (conventional deep nets collapse last-layer features to low-dim); HRM instead keeps a high-dim top module, a hallmark of flexible PFC-like computation.
  - **Caveat (stated by authors)**: evidence is **correlational**; causal necessity of the hierarchy (e.g. by constraining $z_H$ dimensionality) is left to future work.
- **Related work (§5)**:
  - **Reasoning / algorithm learning**: Neural Turing Machines, Differentiable Neural Computer, Neural GPUs, Recurrent Relational Networks — iterative neural architectures that learn algorithms from data. HRM's angle: brain-inspired architecture with far better **data efficiency**, learning complex/diverse algorithms from ~1000 examples.
  - **Brain-inspired reasoning architectures**: **Spaun** (spiking nets, modules ↔ brain regions, but hand-designed algorithms limit learning new tasks); **Tolman-Eichenbaum Machine** (hippocampal-entorhinal, structural-knowledge basis for relational memory).
  - **Hierarchical memory**: **Hierarchical Sequential Models**, **Clockwork RNN** — multi-timescale recurrent modules for long-range dependencies / mitigating forgetting. HRM uses full attention for simplicity (it targets reasoning, not memory); adding hierarchical memory is flagged as future work.
- **Discussions (§6)**:
  - **Turing-completeness**: like the Universal Transformer, HRM is computationally universal given enough memory/time — escaping the fixed-depth Transformer's $\mathrm{AC}^0 / \mathrm{TC}^0$ ceiling. Earlier RNN reasoners had universality on paper but were crippled by premature convergence + BPTT; by fixing both and adding adaptive compute, HRM moves toward **practical** Turing-completeness (deep DFS/backtracking-style reasoning).
    - **What $\mathrm{AC}^0 / \mathrm{TC}^0$ mean** (the formal sense of "shallow"): circuit-complexity classes describing what a Boolean circuit of **constant depth** + **polynomial size** can compute. Depth = number of sequential stages; the superscript $0$ means depth is $O(1)$ — *doesn't grow with input size $n$*.
      - **$\mathrm{AC}^0$**: constant-depth, unbounded-fan-in AND/OR/NOT. Famously **cannot compute PARITY** (odd # of 1-bits), can't multiply, can't count.
      - **$\mathrm{TC}^0$**: $\mathrm{AC}^0$ **+ MAJORITY/threshold gates** (can count) → gets PARITY, multiplication, division, sorting — but is **still constant depth**. Containment: $\mathrm{AC}^0 \subsetneq \mathrm{TC}^0 \subseteq \mathrm{NC}^1 \subseteq \dots \subseteq \mathrm{P}$.
      - **Why Transformers land here**: theory (e.g. Merrill–Sabharwal; "average-hard attention ⇒ constant-depth threshold circuits") shows a **fixed-depth, log-precision** Transformer's whole forward pass collapses into a $\mathrm{TC}^0$-ish constant-depth circuit. So its **serial-computation budget is constant** — problems needing a number of sequential steps that *grows with $n$* (long algorithmic reasoning, deep search/backtracking) are out of reach in one forward pass.
      - **The escape**: grow effective depth *with the problem* instead of fixing it — CoT adds serial steps in token space; HRM/Universal/looped transformers add them in latent space via recurrence (+ ACT to scale steps with difficulty). Once depth isn't constant, you climb out of $\mathrm{TC}^0$ toward Turing-completeness. This is the theoretical motivation for the whole recurrent-depth program.
  - **Vs. RL+CoT**: recent evidence says RL mostly *unlocks existing* CoT abilities rather than discovering new reasoning, and is unstable / data-hungry with sparse rewards. HRM instead learns from **dense gradient supervision** in a **continuous** latent space (biologically plausible; doesn't waste equal compute on every token).
  - **Vs. linear attention**: recurrence has also been used to kill attention's quadratic cost (e.g. Log-linear Attention), but swapping the attention mechanism doesn't change the fact that Transformers stay **fixed-depth** and still need CoT — orthogonal to HRM's depth contribution.
- **Connections / lineage**:
  - **Vs. plain looped/recurrent transformers**: HRM's contribution is *hierarchical convergence* — the two-timescale reset is what avoids the premature-convergence ceiling that limits a single recurrently-applied block.
  - **Vs. DEQ**: shares the fixed-point / implicit-depth view, but HRM uses an explicit nested recurrence + 1-step gradient instead of a root-find.
  - **→ TRM**: TRM (below) argues the two-module *hierarchy* isn't the essential ingredient — a single tiny network recursed over a latent + answer, with deep supervision and inference-time scaling, matches or beats HRM. Read HRM first, then TRM as the simplification, then the ARC-Prize analysis for the skeptical take on *why* HRM works.

## [2025] The Hidden Drivers of HRM's Performance on ARC-AGI

- **Date**: 2026-06-16
- **Blog**: <https://arcprize.org/blog/hrm-analysis>

---

- **What it is**: ARC-Prize team reproduces HRM on ARC-AGI and ablates it to find what *actually* drives the score. The skeptical counterweight to the HRM paper's framing.
- **Top-line takeaways**:
  - **The hierarchy is *not* the driver.** Swapping HRM for a same-size vanilla Transformer (no hyperparam tuning) lands **within ~5pp** — directly undercuts the "hierarchical convergence is the key idea" claim.
  - **The outer refinement loop is what matters.** Going from 1→2 outer loops is a **+13pp** jump; refinement *during training* matters more than at inference. (So HRM's real engine is deep supervision / iterative refinement, not the two-timescale architecture.)
  - **ACT / adaptive compute**: only a minor inference-time benefit.
  - **Puzzle embeddings are a hard dependency**: the model can only process `puzzle_id`s seen during training → this *enables* test-time training but *blocks* true generalization.
  - **Cross-task transfer is minimal**: training on only 400 eval tasks → 31% vs 41% with the full set; gains come from per-puzzle training, not transferable reasoning. (~300 augmentations suffice, vs the 1000 reported.)
  - **It's fundamentally transductive test-time training.** The post concludes HRM is "a zero-pretraining test-time training approach, **similar to Liao and Gu's 'ARC-AGI without pretraining'**" — i.e. closer to a per-puzzle program-synthesis substrate than a pretrained reasoner. Speculates ~21% pass@2 if truly task-isolated. → see CompressARC in [[compression]].
  - **Numbers**: ARC-AGI-1 ≈ **32%** semi-private / 41% public-eval (claimed); ARC-AGI-2 ≈ **2%**.
- **So what**: reframes HRM's contribution — the depth/refinement loop + test-time training carry the ARC result, not the brain-inspired hierarchy. Sets up TRM's "drop the hierarchy" move and is the reason to treat the §4 dimensionality story as *correlational*.
