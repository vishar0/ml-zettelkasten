# Optimization Algorithms

- **Created**: 2022-04
- **Last Updated**: 2026-10-26
- **Status**: `In Progress`
- **Description**: First-order optimizers for deep nets, from SGD with momentum and the adaptive family through decoupled weight decay to Muon's orthogonalized updates.
- **Related**:
  - [[scaling-laws]] — batch-size and weight-decay scaling rules.
  - [[karpathy-repos]] — nanochat's Muon + AdamW.
  - [[papers-distributed-optimization]]

---

## Fundamentals

- [ ] [2016] [Ruder] An overview of gradient descent optimization algorithms — [blog](https://ruder.io/optimizing-gradient-descent/)
- [x] [2012] [Hinton] CSC321 lecture 6 slides — [slides](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) — momentum, Nesterov, adaptive optimizers, RMSProp with intuition and general bag of tricks
  - Large gradients doesn't necessarily mean it's the right direction towards global optimum.
  - Goal:
    - Move quickly in directions with small but consistent gradients
    - move slowly in directions with big bug inconsistent gradients
  - Global LR (SGD, SGD with momentum, Nesterov) vs per-param adaptive LR (AdaGrad, RMSProp, Adam)
  - Momentum: Instead of using the gradient to change the position, use it to change the velocity
    - Builds up speed in directions with gentle by consistent gradients
    - Damps oscillations in directions of high curvature (large but inconsistent gradients)
    - Start with a small momentum factor to reduce noise, and smoothly increase as things start to converge to accelerate training
  - Adaptive methods: Adapt the LR for each param based on the consistency of the gradient for that param
    - Intuition: the magnitude of the gradients are often very different for different params and keeps changing, choosing a single global LR that works well is hard
  - RMSProp: Divide the LR for each param by a running mean of magnitudes of the recent gradients for that param.
    - The idea is just to use the sign of the gradient to make the update. But directly using the sign in a stochastic mini-batch setting is problematic because the magnitude of the gradients is important too (eg., 9 consecutive updates of +0.1 and then one update of -0.9 should cancel out; this won't happen if we just take the sign).
    - RMSProp solves this problem by keeping a running mean estimate (with decay) of the magnitude of the gradients for each param, and then normalizing the incoming gradient with that estimate. This essentially gives us the sign of the gradient when averaged over several mini-batches while also avoiding the above-mentioned pathology.
    - But what does this solve? By essentially using only the sign of the gradient, we are punting the problem of having to choose a global LR that works well for a variety of gradient magnitudes (across params and across timesteps).
- [ ] [2014] [Kingma,Ba] Adam: A Method for Stochastic Optimization — [paper](https://arxiv.org/abs/1412.6980)
- [ ] [2018] [NoamShazeer] Adafactor: Adaptive Learning Rates with Sublinear Memory Cost — [paper](https://arxiv.org/abs/1804.04235)

## Weight Decay

- [ ] [2017] Decoupled Weight Decay Regularization — [paper](https://arxiv.org/abs/1711.05101) — why weight decay != L2 reg for Adam
  - SGD with L2 reg == SGD with weight decay (following a simple scaling factor)
  - Adam with L2 reg != Adam with weight decay
  - In the Adam with L2 reg case, For params with large gradients, Adam's scaling factor also downweights the regularization component. Thus not all params get regularized equally. But with decoupled weight decay, this is not a problem.
  - This makes Adam with (decoupled) weight decay perform better than Adam with L2 reg.
  - Coupled together with cosine annealing LR scheduler, things become even better empirically.
  - Section 3 of the paper has a Bayesian filtering based theoritical justification for why decoupled weight decay performs better for adaptive gradient methods.
- [ ] [2018] Three Mechanisms of Weight Decay Regularization — [paper](https://arxiv.org/abs/1810.12281)
- [ ] [2024] How to set AdamW's weight decay as you scale model and dataset size — [paper](https://arxiv.org/abs/2405.13698) — the EMA-timescale view: keep $T_{\text{epoch}} = B/(\eta\lambda D)$ constant as batch, LR and data change; nanochat's base_train.py derives its weight-decay scaling $\lambda = \lambda_{\text{ref}}\sqrt{B/B_{\text{ref}}}\,(D_{\text{ref}}/D)$ from it. See [[scaling-laws]] for the companion batch-size rule (Power Lines).

## Weight Averaging

Averaging points along one training trajectory, EMA, SWA and sparse checkpoint averaging. DiLoCo with a single outer step reduces to model souping, see [[papers-distributed-optimization]].

- [ ] [2023] [SunnySanyal] LAWA: Early Weight Averaging meets High Learning Rates for LLM Pre-training — [paper](https://arxiv.org/abs/2306.03241)

## Muon

Muon is SGD-momentum for 2-D weights where each update matrix is replaced by its nearest semi-orthogonal matrix $UV^\top$, computed with a few bf16 Newton-Schulz iterations instead of an SVD; the theory reads this as steepest descent under the spectral norm. Embeddings, the output head and 0/1-D params stay on AdamW.

- [ ] [2024] [KellerJordan] Muon: An optimizer for hidden layers in neural networks — [blog](https://kellerjordan.github.io/posts/muon/) — the original write-up: quintic Newton-Schulz with tuned coefficients (3.4445, -4.7750, 2.0315), 5 steps, Nesterov momentum, under 1% FLOP overhead; CIFAR-10 94% record 3.3 to 2.6 A100-s, 1.35x on the NanoGPT speedrun, 1.5B GPT-2 XL-level HellaSwag in 10 vs 13.3 8xH100-hours for AdamW; intuition is that SGD/Adam updates for transformer matrices are nearly low-rank and orthogonalizing boosts the rare directions; applying Muon separately to Q, K, V works better than fused
- [ ] [2024] [Bernstein] Deriving Muon — [blog](https://jeremybernste.in/writing/deriving-muon) — four-step derivation: metrize the layer with the RMS-to-RMS operator norm ($\sqrt{\text{fan-in}/\text{fan-out}}$ times spectral), bound the output change by it, minimize the linearized loss under that bound (the solution keeps the gradient's singular vectors and drops the singular values), then get $UV^\top$ via odd matrix polynomials that commute with the SVD; argues $\mu$P and Muon are two sides of the same coin and that the LR transfers across width for free
- [ ] [2024] Old Optimizer, New Norm: An Anthology — [paper](https://arxiv.org/abs/2409.20325) — with EMAs switched off, Adam is sign descent (steepest descent under the max-of-max $\ell_1 \to \ell_\infty$ norm), Shampoo without accumulation is steepest descent under the max spectral norm and its update is exactly $UV^\top$, Prodigy picks Adam's norm with an automatic step size; proposes assigning a different operator norm to each tensor by its role in the network, which is the design space Muon sits in
- [ ] [2025] [Moonshot] Muon is Scalable for LLM Training — [paper](https://arxiv.org/abs/2502.16982) — two fixes make Muon work at scale with no tuning: add weight decay and adjust the per-parameter update scale; scaling-law runs show about 2x compute efficiency over AdamW at compute-optimal training; Moonlight, a 3B/16B MoE trained on 5.7T tokens; open-source distributed Muon that is memory-optimal and communication-efficient
- [ ] [2025] [EssentialAI] Practical Efficiency of Muon for Pretraining — [paper](https://arxiv.org/abs/2505.02222) — Muon expands the Pareto frontier over AdamW on the compute-time tradeoff and keeps data efficiency at batch sizes far beyond the critical batch size; studies Muon with $\mu$P for hyperparameter transfer and a telescoping algorithm that accounts for all sources of $\mu$P error at modest overhead; models up to 4B params
- [ ] [2026] nanochat's `optim.py`, Muon for the matrices and AdamW for embeddings and head — [code](https://github.com/karpathy/nanochat/blob/4b40774/nanochat/optim.py) — one optimizer with `adamw` and `muon` param groups; the Muon step is a single compiled kernel of Nesterov momentum, then 5 Polar Express iterations (Amsel et al. 2505.16932 coefficients, bf16) in place of the quintic Newton-Schulz, then NorMuon per-row or per-column variance reduction (2510.05491), then cautious weight decay that only decays entries where update and weight share a sign; LR scaled by $\sqrt{\max(1, \text{rows}/\text{cols})}$; `DistMuonAdamW` stacks same-shape matrices, reduce-scatters grads so each rank updates its chunk with sharded state, then all-gathers, ZeRO-2 style without DDP. See [[karpathy-repos#nanochat]].

## Edge of Stability

What gradient descent does at large learning rates, where the sharpness sits at the stability threshold $2/\eta$.

- [ ] [2021] Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability — [paper](https://arxiv.org/abs/2103.00065)
- [ ] [2022] Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability — [paper](https://arxiv.org/abs/2209.15594)
- [ ] [2024] Understanding Optimization in Deep Learning with Central Flows — [paper](https://arxiv.org/abs/2410.24206)

## Misc

- [ ] [2025] [MicahGoldblum,AndrewGordonWilson] [Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful](#2025-small-batch-size-training-for-language-models-when-vanilla-sgd-works-and-why-gradient-accumulation-is-wasteful) — [paper](https://arxiv.org/abs/2507.07101)

---

## [2025] [Small Batch Size Training for Language Models: When Vanilla SGD Works, and Why Gradient Accumulation Is Wasteful](https://arxiv.org/abs/2507.07101)

- **Date**: 2026-10-22
- **Code**: <https://github.com/martin-marek/batch-size>

---

- **Abstract**:
  - > Conventional wisdom dictates that small batch sizes make language model pretraining and fine-tuning unstable, motivating gradient accumulation, which trades off the number of optimizer steps for a proportional increase in batch size. While it is common to decrease the learning rate for smaller batch sizes, other hyperparameters are often held fixed. In this work, we revisit small batch sizes all the way down to batch size one, and we propose a rule for scaling Adam hyperparameters to small batch sizes. In particular, rather than holding the decay rate of the second moment fixed across batch sizes, we propose to hold its half-life fixed in terms of tokens. We find that small batch sizes (1) train stably, (2) are consistently more robust to hyperparameter choices, (3) achieve equal or better per-FLOP performance than larger batch sizes, and (4) notably enable stable language model training with vanilla SGD, even without momentum, despite storing no optimizer state. Building on these results, we provide practical recommendations for selecting a batch size and setting optimizer hyperparameters. We further recommend against gradient accumulation unless training on multiple devices with multiple model replicas. Finally, we show that a small batch size combined with an optimizer with a small state size can provide the performance benefits of full fine-tuning while maintaining a similar memory footprint to LoRA.
- **Section 1: Introduction**:
  - Received wisdom: large batches stabilize LM training, so large-batch optimizers are standard and gradient accumulation simulates batches larger than device memory allows.
  - Small batches do spike in practice. Diagnosis: a hyperparameter artifact. The LR gets lowered for small batches but Adam's $\beta_1$, $\beta_2$ stay fixed, and a fixed $\beta_2$ means a different amount of data at every batch size (Section 2).
  - [Fig1a] Hold the half-life of $\beta_2$ fixed in tokens instead of $\beta_2$ itself, and training is stable down to batch size 1.
  - [Fig1b] Why small batches are forgiving: at fixed compute a large batch takes fewer steps, needs a large LR, so each step predicts the loss surface far from the current point, which takes a sophisticated, well-tuned optimizer. Small batches take many small steps; the LR matters less and momentum matters less.
  - At very small batch, vanilla SGD with no momentum and no weight decay is competitive, against recent claims that SGD cannot train LMs. No momentum also means no optimizer state.
  - Adafactor is revisited as a small-state alternative to Adam at small batch, which lets larger models fit a memory budget.
  - When small batches are efficient, from the convex-optimization intuition that the batch size should grow as training converges:
    - A mini-batch gradient is the true gradient plus noise, and the noise shrinks as $1/\sqrt{B}$. What a step buys depends on the ratio of the true gradient's norm to that noise. When the ratio is high, a tiny batch already points the right way, so a larger batch spends $B$ times the FLOPs on a direction you already had. When the ratio is low, the noise swamps the signal and only averaging over a big batch recovers it.
    - At a fixed FLOP budget, batch size trades against step count: $B$ times the batch is $B$ times fewer steps. So a small batch is the better deal exactly when the extra steps are worth more than the noise they add.
    - Early in training, far from any minimum, the true gradient is large and the ratio is high. Small batches win.
    - Near a minimum the true gradient's norm goes to zero while the noise does not, so the ratio collapses. Averaging is now the only way to make progress and large batches win. This is the regime classical optimization has in mind when it recommends growing the batch over training.
    - Where LM pretraining sits: compute-optimal scaling says that for a fixed budget it is better to train a larger model on fewer tokens than to push any model toward convergence. So the loss is still falling fast at the end of a run, the gradient has not shrunk toward the noise floor, and the whole run stays in the regime where small batches are efficient. Hypothesis in the paper, borne out by Section 4.
  - Recommendation previewed, for pretraining and fine-tuning alike:
    - Pick the smallest batch size that still maximizes throughput. Below a few hundred tokens per device the matmuls are too thin to keep the chip busy and tokens per second drop; above that point a larger batch adds no throughput and, by the argument above, costs steps and robustness. The batch size is set by the hardware, not by stability.
    - Do not use gradient accumulation. It exists to simulate a batch larger than fits in memory, and the claim is that the larger batch was never needed; it also keeps the summed gradients resident, a memory cost with no return. The one exception is multi-replica training where the bandwidth between replicas is the bottleneck (Section 5).
- **Section 2: Background**:
  - **SGD**: $\theta_{t+1} = \theta_t - \eta\, g_t$, $g_t$ averaged over $B$ samples, down to $B = 1$ here. Momentum is left out on purpose to show it is unnecessary at small batch.
  - **Adam**: $m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t$, $v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2$, $\theta_{t+1} = \theta_t - \eta\, \hat m_t / (\sqrt{\hat v_t} + \epsilon)$ with $\hat m_t = m_t / (1 - \beta_1^t)$ and $\hat v_t = v_t / (1 - \beta_2^t)$.
    - What survives of the gradient: per coordinate the step is $\eta\, m_i / \sqrt{v_i}$. With a steady gradient $m_i \approx g_i$, $\sqrt{v_i} \approx |g_i|$, the ratio is $\pm 1$ and every parameter moves by about $\eta$ regardless of gradient magnitude; rescaling the loss changes nothing. A mean is at most the RMS in magnitude, so $|m_i / \sqrt{v_i}| \le 1$, and it shrinks toward 0 when a coordinate's gradient flips sign. Adam keeps the sign plus a consistency factor in $[0, 1]$ and drops the magnitude. RMSProp is Adam with $\beta_1 = 0$ and no bias correction.
    - Bias correction: $m_0 = v_0 = 0$, so the unrolled weights sum to $1 - \beta^t$ and both averages start biased small. Dividing by $1 - \beta^t$ fixes that. It matters most for $v$: at step 1 with $\beta_2 = 0.999$, $\sqrt{v_1} \approx 0.03\,|g|$ while $m_1 = 0.1\, g$, so the uncorrected first step is about $3\eta$, and $v$ stays underestimated for hundreds of steps. The factors decay to 1.
    - State: $m$ and $v$ are each one float per parameter, usually fp32, so for a bf16 model Adam's state is four times the weight bytes. Dropping momentum (SGD) or factoring $v$ (Adafactor) is a real saving.
    - LM defaults: $\beta_1 = 0.9$, $\beta_2 \in [0.95, 0.98]$. The range reports practice (0.95 in GPT-3, Llama and nanochat; 0.98 elsewhere), not a principle. All are far below Adam's original 0.999, whose 700-step memory lags when the gradient scale drifts: $v$ reflects the smaller past, the step overshoots, the loss spikes. Almost nobody scales the betas with batch size; the paper's claim is that $\beta_2$ must be.
  - [Fig2] **Moment half-life**: $\beta_1, \beta_2$ reparameterized as timescales in tokens, because a $\beta$ is a per-step decay and a step is a different amount of data at every batch size.
    - After $n$ steps with $\beta^n = 1/2$ a past gradient's weight has halved: $n = \ln 2 / (-\ln \beta) \approx 0.69 / (1 - \beta)$. In steps: 0.9 gives 6.6, 0.95 gives 13.5, 0.99 gives 69, 0.999 gives 693.
    - A step sees $B \cdot T$ tokens, so the half-life in tokens satisfies $\beta^{t_{1/2} / (BT)} = 1/2$, i.e. $\beta = (1/2)^{BT / t_{1/2}}$. The paper writes $t_1$, $t_2$ for the two half-lives.
    - Why tokens: $\beta_2 = 0.95$ is 13.5 steps, about 14M tokens at $B = 512$, $T = 2048$, but 28K tokens at $B = 1$. Same $\beta_2$, a smooth average in one case and nearly the last few samples in the other. Fixing $t_2$, say 10M tokens, keeps the data the average covers constant, and $\beta_2$ follows. That is what let settings transfer across the grid without retuning.
    - "Averaged over roughly 10M tokens" is a gloss. An EMA has no window edge: half the weight is on the most recent $t_{1/2}$ tokens, a quarter on the $t_{1/2}$ before, and so on; the mean age is $\beta / (1 - \beta)$ steps, about 1.44 half-lives, and 90% of the weight lies within 3.3 half-lives. The half-life is the scale of the window, not its width.
  - **Adafactor**: Adam with both state buffers cut down.
    - Second moment factored to rank one: for a $d_1 \times d_2$ matrix keep EMAs of the row sums $R_i$ and column sums $C_j$ of $g^2$ and reconstruct $\hat v_{ij} = R_i C_j / \sum_i R_i$, $d_1 + d_2$ numbers instead of $d_1 d_2$. Exact when the squared gradients are rank one, close in practice since a weight's gradient scale is roughly an input factor times an output factor; Shazeer and Stern show it is the best rank-one fit under the natural divergence for nonnegative quantities. Vectors keep a full second moment.
    - No first moment, $\beta_1 = 0$: the update is $g / \sqrt{\hat v}$. State per matrix $d_1 + d_2$ against Adam's $2 d_1 d_2$.
    - Three stabilizers: update clipping (scale the update down when its RMS exceeds 1), relative step size (LR times the parameter tensor's RMS), and the schedule $\beta_{2,t} = 1 - t^{-0.8}$, short memory early so no bias correction is needed.
    - Section 4.5: the best performance-memory trade-off; fits the small-batch regime because momentum stops mattering there.
  - **Muon**: for weight matrices only; hidden layers on Muon, embedding and final layers on Adam.
    - Per step: momentum $M = \mu M + G$, then the nearest orthogonal matrix $O = U V^\top$ from the SVD $M = U S V^\top$ (same singular directions, every singular value set to 1), then $W \leftarrow W - \eta\, O$ with a scale for non-square shapes. The matrix is the layer's own $d_{\text{out}} \times d_{\text{in}}$ weight, one matrix at a time; a stacked kernel over same-shape matrices is only batching. Tensors of more than two dimensions are flattened to (outputs, everything else) first, as for conv kernels.
    - Newton-Schulz computes $U V^\top$ without an SVD. An odd polynomial $a M + b M (M^\top M) + c M (M^\top M)^2$ maps each singular value $\sigma \mapsto a\sigma + b\sigma^3 + c\sigma^5$ and leaves $U$, $V$ alone, so a polynomial whose iteration drives $(0, 1]$ toward 1 orthogonalizes under iteration. Classic form $X \leftarrow (3X - X X^\top X)/2$, Newton's method for the matrix sign function. Muon's quintic (3.4445, -4.7750, 2.0315) is tuned for five bf16 iterations landing singular values in about $(0.7, 1.3)$, enough since they only need to be roughly equal; under 1% of training FLOPs. nanochat uses the Polar Express coefficients.
    - Why: a transformer matrix's momentum is nearly low rank; setting all singular values to 1 moves the matrix equally along every direction it has evidence for, amplifying the rare ones Adam and SGD barely touch. Equivalently (Bernstein) steepest descent under a spectral-norm budget. Adam normalizes per element, Muon per matrix.
    - Reported: about 2x compute efficiency over AdamW at compute-optimal scale with weight decay and a per-parameter update scale (Moonshot); data efficiency kept well past the critical batch size (Essential AI). See the Muon section above.
  - **Gradient accumulation**: sum gradients over micro-batches, step periodically; effective batch is micro-batch times steps. Still costs more memory than a small batch, since the accumulated gradients are stored.
- **Section 4: Experiments**:
  - **Setup**: dense grid over LR, $\beta_1$, $\beta_2$ for SGD, Adam, Adafactor, Muon on a 30M-param decoder, 600M FineWeb-Edu tokens at Chinchilla's ratio (RoPE, QK-norm, RMSNorm, GELU, untied embeddings, context 512, GPT-2 tokenizer), batch sizes $\{1, 4, 16, 64, 256, 1024, 4096\}$. Heuristics then checked on GPT-2 (124M), GPT-3 (1.3B) pretraining and Gemma 3 (4B) fine-tuning.
  - **The prior negative results are hyperparameter artifacts.** SGD looked far slower than Adam and Adam looked bad at small batch in setups that held $\beta_1$, $\beta_2$ fixed or ran SGD at too large a batch.
  - [Fig1a] **4.1 Every optimizer is best at the smallest batch, and the gap between optimizers widens with batch size.** With everything tuned per batch size, all four degrade as the batch grows, and the spread is smallest at batch size 1.
    - Large batches need sophisticated optimizers because each step is a prediction about the loss surface, and large steps predict far from the current point.
    - Momentum matters less at small batch because its job is damping oscillations across high-curvature directions, and small steps do not overshoot them (Appendix B toy example).
    - **Outcome: vanilla SGD without momentum is competitive at small batch.**
  - [Fig3] **4.2 Small batches are robust to hyperparameter misspecification, large batches are not.** Sweeping each Adam hyperparameter around its optimum, batch size 1 stays near its best loss across the whole range of LR, $\beta_1$, $\beta_2$; large batches lose sharply off the optimum.
    - [Fig1b] **The robustness compounds.** A joint LR and $\beta_1$ sweep on GPT-2 gives batch size 1 a far broader low-loss region than batch 512.
    - **Outcome: counting the tuning budget, a smaller batch may be preferable before any per-FLOP argument.**
  - [Fig4] **4.3 Scaling Adam's hyperparameters with batch size**:
    - **LR does not follow square-root scaling.** Batch 1 to 1024 would multiply the LR by 32 under the rule; about 3 works better.
    - **$\beta_1 = 0.9$ holds across batch sizes.**
    - **A fixed $\beta_2$ fails at small batch; a fixed half-life $t_2$ works.** The default $[0.95, 0.98]$ is far too short a memory at small batch. Holding $t_2$ at its optimum transfers with no retuning: $\beta_2^* = \beta_2^{(B^*/B)}$ when the batch goes from $B$ to $B^*$ (Equation 2).
    - [Fig5] **The rule repairs a published negative result.** Xiao's setup (19M-param decoder on C4, Adam with $\beta_1 = 0.9$, $\beta_2 = 0.95$ fixed, LR tuned) had batch 16 and 32 below 128 and 256. Scaling $\beta_2$ for fixed $t_2$ with no other tuning brings them on par. So that result was incomplete tuning, and the heuristic transfers.
    - **Outcome: keep $\beta_1$ at 0.9, scale $\beta_2$ for a fixed token half-life, and small batches match or beat large ones.**
  - [Fig6] **4.4 SGD is competitive with adaptive optimizers at 124M and 1.3B**:
    - **GPT-3 (1.3B) on FineWeb**: baseline AdamW at Brown et al.'s settings (batch 512, $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay 0.1). Adam at batch 1 gets no tuning: weight decay off, LR scaled by 4.3, $\beta_2$ either fixed or set for fixed $t_2$. Adafactor and SGD get a four-LR sweep for lack of reference values. SGD at batch 1, no momentum, no state, matches the baseline; Adam and Adafactor at batch 1 beat it, and the fixed-$t_2$ rule is what makes Adam's result good.
    - **GPT-2 (124M), LR tuned per optimizer**: tuning lifts the AdamW baseline; Adam and Adafactor at batch 1 match AdamW at batch 512, SGD slightly worse.
    - [Fig7] **Memory**: GPT-3 (13B) in 16-bit with gradient checkpointing and a fused backward-plus-update fits an A100 40GB, but Adam's state and accumulated gradients do not. Small batch plus a small-state optimizer is what fits.
  - [Fig8] **4.5 Full fine-tuning on a memory budget**:
    - **Fine-tuning is memory-bound.** Few FLOPs, but model plus optimizer state must be resident. LoRA shrinks the state by freezing weights and often underperforms full fine-tuning.
    - **Small batch plus a small-state optimizer gives full fine-tuning at LoRA-like memory.** Gemma 3 (4B), not instruction-tuned, on MATH, 5 epochs, 8 LRs. Baselines: Adam at batch 16 with accumulation and fp32 everything, 16 bytes per parameter; LoRA at rank 16, about 2 bytes per parameter from the frozen bf16 checkpoint.
    - **Two steps.** Adam at batch 1 with the $\beta_2$ rule (0.95 at batch 16 becomes 0.997 at batch 1) drops the accumulated gradients; then Adafactor in place of Adam shrinks the state to near nothing, fp32 state over bf16 weights updated with stochastic rounding (Appendix A.3).
    - **Outcome: in memory-constrained settings, exploit small batches and small-state optimizers instead of emulating the industry large-batch setup.**
- **Section 5: Practical recommendations**:
  - **Summary of claims.** Small batches, down to 1, match or beat large batches across scales when tuned, in pretraining and fine-tuning; SGD at batch 1 matches the AdamW baseline on GPT-3 (1.3B) with minimal tuning; the fixed-$t_2$ rule for $\beta_2$ transfers across scales. This contradicts the common assumptions behind large-batch training and gradient accumulation.
  - **Batch size rule: the smallest batch that maximizes throughput in tokens per second, equivalently MFU.** The accelerator must spend more time computing than moving data between HBM and registers, which in practice means at least several hundred tokens per device. On a large cluster that can be millions of tokens per batch; on one device a few thousand tokens may already saturate arithmetic intensity.
  - **No gradient accumulation** unless training on multiple devices with multiple model replicas, where the bandwidth between replicas is the bottleneck.
  - **Memory-constrained: simple optimizers plus a small batch.**
  - **Open questions**: interaction with batch size schedules; weight precision in fine-tuning below 16 bits, to 8 or 4; optimizers with small state designed for the small-batch regime; a theory for why fixing the second-moment half-life in tokens transfers across scales.
