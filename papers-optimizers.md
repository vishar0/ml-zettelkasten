# Optimization Algorithms

- **Created**: 2022-04
- **Last Updated**: 2026-10-11
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

## Weight Decay

- [ ] [2017] Decoupled Weight Decay Regularization — [paper](https://arxiv.org/abs/1711.05101) — why weight decay != L2 reg for Adam
  - SGD with L2 reg == SGD with weight decay (following a simple scaling factor)
  - Adam with L2 reg != Adam with weight decay
  - In the Adam with L2 reg case, For params with large gradients, Adam's scaling factor also downweights the regularization component. Thus not all params get regularized equally. But with decoupled weight decay, this is not a problem.
  - This makes Adam with (decoupled) weight decay perform better than Adam with L2 reg.
  - Coupled together with cosine annealing LR scheduler, things become even better empirically.
  - Section 3 of the paper has a Bayesian filtering based theoritical justification for why decoupled weight decay performs better for adaptive gradient methods.
- [ ] [2018] [Zhang] Three Mechanisms of Weight Decay Regularization — [paper](https://arxiv.org/abs/1810.12281)
- [ ] [2024] How to set AdamW's weight decay as you scale model and dataset size — [paper](https://arxiv.org/abs/2405.13698) — the EMA-timescale view: keep $T_{\text{epoch}} = B/(\eta\lambda D)$ constant as batch, LR and data change; nanochat's base_train.py derives its weight-decay scaling $\lambda = \lambda_{\text{ref}}\sqrt{B/B_{\text{ref}}}\,(D_{\text{ref}}/D)$ from it. See [[scaling-laws]] for the companion batch-size rule (Power Lines).

## Muon

Muon is SGD-momentum for 2-D weights where each update matrix is replaced by its nearest semi-orthogonal matrix $UV^\top$, computed with a few bf16 Newton-Schulz iterations instead of an SVD; the theory reads this as steepest descent under the spectral norm. Embeddings, the output head and 0/1-D params stay on AdamW.

- [ ] [2024] [KellerJordan] Muon: An optimizer for hidden layers in neural networks — [blog](https://kellerjordan.github.io/posts/muon/) — the original write-up: quintic Newton-Schulz with tuned coefficients (3.4445, -4.7750, 2.0315), 5 steps, Nesterov momentum, under 1% FLOP overhead; CIFAR-10 94% record 3.3 to 2.6 A100-s, 1.35x on the NanoGPT speedrun, 1.5B GPT-2 XL-level HellaSwag in 10 vs 13.3 8xH100-hours for AdamW; intuition is that SGD/Adam updates for transformer matrices are nearly low-rank and orthogonalizing boosts the rare directions; applying Muon separately to Q, K, V works better than fused
- [ ] [2024] [Bernstein] Deriving Muon — [blog](https://jeremybernste.in/writing/deriving-muon) — four-step derivation: metrize the layer with the RMS-to-RMS operator norm ($\sqrt{\text{fan-in}/\text{fan-out}}$ times spectral), bound the output change by it, minimize the linearized loss under that bound (the solution keeps the gradient's singular vectors and drops the singular values), then get $UV^\top$ via odd matrix polynomials that commute with the SVD; argues $\mu$P and Muon are two sides of the same coin and that the LR transfers across width for free
- [ ] [2024] Old Optimizer, New Norm: An Anthology — [paper](https://arxiv.org/abs/2409.20325) — with EMAs switched off, Adam is sign descent (steepest descent under the max-of-max $\ell_1 \to \ell_\infty$ norm), Shampoo without accumulation is steepest descent under the max spectral norm and its update is exactly $UV^\top$, Prodigy picks Adam's norm with an automatic step size; proposes assigning a different operator norm to each tensor by its role in the network, which is the design space Muon sits in
- [ ] [2025] [Moonshot] Muon is Scalable for LLM Training — [paper](https://arxiv.org/abs/2502.16982) — two fixes make Muon work at scale with no tuning: add weight decay and adjust the per-parameter update scale; scaling-law runs show about 2x compute efficiency over AdamW at compute-optimal training; Moonlight, a 3B/16B MoE trained on 5.7T tokens; open-source distributed Muon that is memory-optimal and communication-efficient
- [ ] [2025] [EssentialAI] Practical Efficiency of Muon for Pretraining — [paper](https://arxiv.org/abs/2505.02222) — Muon expands the Pareto frontier over AdamW on the compute-time tradeoff and keeps data efficiency at batch sizes far beyond the critical batch size; studies Muon with $\mu$P for hyperparameter transfer and a telescoping algorithm that accounts for all sources of $\mu$P error at modest overhead; models up to 4B params
- [ ] [2026] nanochat's `optim.py`, Muon for the matrices and AdamW for embeddings and head — [code](https://github.com/karpathy/nanochat/blob/4b40774/nanochat/optim.py) — one optimizer with `adamw` and `muon` param groups; the Muon step is a single compiled kernel of Nesterov momentum, then 5 Polar Express iterations (Amsel et al. 2505.16932 coefficients, bf16) in place of the quintic Newton-Schulz, then NorMuon per-row or per-column variance reduction (2510.05491), then cautious weight decay that only decays entries where update and weight share a sign; LR scaled by $\sqrt{\max(1, \text{rows}/\text{cols})}$; `DistMuonAdamW` stacks same-shape matrices, reduce-scatters grads so each rank updates its chunk with sharded state, then all-gathers, ZeRO-2 style without DDP. See [[karpathy-repos#nanochat]].
