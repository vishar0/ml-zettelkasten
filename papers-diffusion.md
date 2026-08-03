# Diffusion

- **Created**: 2025-08-19
- **Last Updated**: 2026-08-03
- **Status**: `In Progress`

---

## 0. Overview and Basics

- [ ] [2025] [3blue1brown] <https://www.3blue1brown.com/lessons/diffusion-models/>
- [ ] [2026] [Nando] Diffusion and Flow Matching Tutorial - [blog](https://love4all.ai/blog/diffusion-and-flow-matching-tutorial/), [pdf](https://love4all.ai/files/diffusion-and-flow-matching-tutorial.pdf), [notebook](https://love4all.ai/files/diffusion-and-flow-matching-tutorial.ipynb)
  - First technical treatment: derive the losses and sampling procedures while staying close to implementable code.
- [ ] [2022] [HuggingFace] The Annotated Diffusion Model — [blog](https://huggingface.co/blog/annotated-diffusion)
- [ ] [2024] Demystifying Variational Diffusion Models - [paper](https://arxiv.org/abs/2401.06281)
  - Optional prerequisite if ELBOs, directed graphical models, or latent-variable models are rusty.
- [ ] TODO something on twitter i bookmarked, jacob shared on slack as well (about diffusion tutorial)
- [ ] TODO MIT course <https://diffusion.csail.mit.edu/2026/>
- [ ] [Flourish] Alan's diffusion tutorial slides - [slides](../../flourish/presentations/2026-05-21-diffusion/README.md)

## 1. Classical Diffusion and Likelihood

- [ ] [2020] [JonathanHo] DDPM: Denoising Diffusion Probabilistic Models - [paper](https://arxiv.org/abs/2006.11239)
  - Read closely. Be able to derive $q(x_t\mid x_0)$, the reverse posterior, and the simplified noise-prediction loss.
- [ ] [2015] [JaschaSohlDickstein] Deep Unsupervised Learning using Nonequilibrium Thermodynamics - [paper](https://arxiv.org/abs/1503.03585)
  - Read after DDPM rather than before it. Focus on the fixed forward process and learned reversal; skim older implementation details.
- [ ] [2021] [DiederikKingma] Variational Diffusion Models - [paper](https://arxiv.org/abs/2107.00630), [code](https://github.com/google-research/vdm)
  - Essential bridge among SNR, the variational bound, estimator variance, likelihood, and bits-back compression.

## 2. Score, SDE, ODE, and Flow Views

- [ ] [2019] [YangSong] Generative Modeling by Estimating Gradients of the Data Distribution - [paper](https://arxiv.org/abs/1907.05600)
  - Understand the score and why estimation at multiple noise levels makes sampling usable.
- [ ] [2021] [YangSong] Score-Based Generative Modeling through Stochastic Differential Equations - [paper](https://arxiv.org/abs/2011.13456)
  - Understand the reverse-time SDE and probability-flow ODE.
- [ ] [2023] [YaronLipman] Flow Matching for Generative Modeling - [paper](https://arxiv.org/abs/2210.02747)
  - Learn conditional flow matching and why simulation-free training recovers a marginal vector field.
- [ ] [2022] Elucidating the Design Space of Diffusion-Based Generative Models - [paper](https://arxiv.org/abs/2206.00364)
  - Separate parameterization, preconditioning, noise distribution, loss weighting, and sampler choice.

**Checkpoint:** translate among $x$-prediction, $\epsilon$-prediction, $v$-prediction, score prediction, and velocity prediction. Distinguish pure reparameterizations from genuinely different paths, objectives, or loss weightings.

## 3. Representation, Architecture, and Fast Sampling

- [ ] [2022] [BillPeebles,SainingXie] Scalable Diffusion Models with Transformers - [paper](https://arxiv.org/abs/2212.09748)
  - Standard DiT architecture and timestep / conditioning machinery underlying the Atari prototype.
- [ ] [2021] High-Resolution Image Synthesis with Latent Diffusion Models - [paper](https://arxiv.org/abs/2112.10752)
  - Study the pixel-space fidelity versus learned-latent efficiency tradeoff and the reconstruction bottleneck.
- [ ] [2025] [KaimingHe] Back to Basics: Let Denoising Generative Models Denoise - [paper](https://arxiv.org/abs/2511.13720)
  - "Just Image Transformers": clean-data $x$-prediction with simple large-patch transformers directly on pixels. Treat as a promising design lead, not settled doctrine.
- [ ] [2024] [DanijarHafner,SergeyLevine,PieterAbbeel] One-Step Diffusion via Shortcut Models - [paper](https://arxiv.org/abs/2410.12557)
- [ ] [2025] [KaimingHe] Mean Flows for One-Step Generative Modeling - [paper](https://arxiv.org/abs/2505.13447)
  - Read Shortcut Models and MeanFlow after the basic flow view; compare instantaneous velocity, average velocity, and step-conditioned prediction.
- [ ] [2024] Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction - [paper](https://arxiv.org/abs/2404.02905)
  - Important foil: coarse-to-fine autoregression may retain global-to-local generation with lower-variance likelihood training.

## 4. Discrete and Block Diffusion

- [ ] [2021] Structured Denoising Diffusion Models in Discrete State-Spaces - [paper](https://arxiv.org/abs/2107.03006)
  - Learn transition matrices, absorbing-mask corruption, and the discrete ELBO.
- [ ] [2024] Simple and Effective Masked Diffusion Language Models - [paper](https://arxiv.org/abs/2406.07524)
  - Bridge from D3PM theory to masked-token diffusion in practice.
- [ ] [2024] [YaronLipman] Discrete Flow Matching - [paper](https://arxiv.org/abs/2407.15595)
  - Connect conditional flow matching to continuous-time Markov chains and discrete rate matrices.
- [ ] [2025] Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models - [paper](https://arxiv.org/abs/2503.09573)
  - Relevant to variable-length generation, KV caching, parallel token sampling, and controlling estimator variance.
- [ ] [2026] Scaling Beyond Masked Diffusion Language Models - [paper](https://arxiv.org/abs/2602.15014)
  - Likelihood can compare models within a diffusion family yet mislead across families; evaluate the speed-quality Pareto frontier too.

## 5. Mixed-Modality and Sequence Models

- [ ] [2024] Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model - [paper](https://arxiv.org/abs/2408.11039)
  - Concrete single-transformer treatment of mixed discrete text and continuous images using modality-specific losses.
- [ ] [2024] Diffusion Forcing: Next-Token Prediction Meets Full-Sequence Diffusion - [paper](https://arxiv.org/abs/2407.01392), [project](https://boyuan.space/diffusion-forcing)
  - Independent noise levels per token unify causal next-step prediction, full-sequence diffusion, variable-horizon rollouts, planning, and guidance.

**Checkpoint:** design a corruption process for $(o_t,a_t,r_t)$ in which pixels use continuous noise, actions and rewards use a CTMC or mask process, and different frames may have different noise levels. Justify both its loss and sampling semantics.

## 6. Compression and Evaluation

- [ ] [2015] A Note on the Evaluation of Generative Models - [paper](https://arxiv.org/abs/1511.01844)
  - Mandatory: strong likelihood / compression, perceptual samples, and downstream behavior can be largely independent in high dimensions.
- [ ] [2019] Practical Lossless Compression with Latent Variables using Bits Back Coding - [paper](https://arxiv.org/abs/1901.04866), [code](https://github.com/bits-back/bits-back)
  - Understand how a variational bound becomes a realizable code rather than only an evaluation number.
- [ ] Revisit [[compression]], [[nncp]], and [Language Modeling is Compression](https://arxiv.org/abs/2309.10668)
  - Distinguish offline likelihood from prequential adaptive coding: predict once, pay the code length, then learn from the datum.

## 7. Diffusion for Control

Start this section only after the core generative-modeling path.

- [ ] [2022] Decision Diffuser: Is Conditional Generative Modeling All You Need for Decision-Making? - [paper](https://arxiv.org/abs/2211.15657)
  - Conditional trajectory generation and guidance.
- [ ] [2023] Diffusion Policy: Visuomotor Policy Learning via Action Diffusion - [paper](https://arxiv.org/abs/2303.04137), [project](https://diffusion-policy.cs.columbia.edu/)
  - Action-sequence diffusion for visuomotor control.
- [ ] [2025] Efficient Online Reinforcement Learning for Diffusion Policy - [paper](https://openreview.net/forum?id=6Anv3KB9lz)
  - Reweighted score matching for policy improvement without differentiating through the sampling chain.
- [ ] [2024] Diffusion-Based Reinforcement Learning via Q-Weighted Variational Policy Optimization - [paper](https://arxiv.org/abs/2405.16173)
  - Explicit $Q$-weighted variational objective discussed in the Universal Learner work.
- [ ] Revisit [[papers-generative-decision-making]] and [Compress and Control](https://arxiv.org/abs/1411.5326)
  - Conditional generation is not automatically causal control. Self-generated actions must be treated as interventions rather than evidence.

## 8. Language Diffusion and Iterative Reasoning

- [ ] [2025] Large Language Diffusion Models - [paper](https://arxiv.org/abs/2502.09992), [project](https://ml-gsai.github.io/LLaDA-demo/)
- [ ] [2026] Improved Large Language Diffusion Models - [paper](https://arxiv.org/abs/2606.25331), [code](https://github.com/ML-GSAI/LLaDA)
- [ ] [2026] DiffusionGemma - [model card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card), [project](https://deepmind.google/models/gemma/diffusiongemma/)
  - Read these as system-level capstones for masked / block diffusion, bidirectional attention, and parallel text decoding.
- [ ] [2024] IRED: Learning Iterative Reasoning through Energy Diffusion - [paper](https://arxiv.org/abs/2406.11179)
  - Explore diffusion as an iterative refinement process over solutions rather than only as a data generator.

## [2022] The Annotated Diffusion Model

- **Date**: 2026-04-23
- **Blog**: <https://huggingface.co/blog/annotated-diffusion>

---

- Two processes
  - Forward diffusion process: sample an image from the true distribution and gradually add gausian noise for $T$ steps until it's eventually pure noise / isotropic gaussian.
  - Reverse denoising diffusion process: neural net trained to gradually denoise an image starting from pure noise to an eventual image in the distribution.
- Forward diffusion process: $q(x_t | x_{t - 1})$. $x_0$ is the actual image and $x_T$ is pure noise.
  - At each step $t$, sample from a conditional gaussian distrubution with mean $\sqrt{1 - \beta_t}x_{t-1}$ and variance $\beta_tI$.
  - This can be done by sampling $\epsilon$ noise from the standard gaussian (0 mean, unit variance) and setting $x_t = \sqrt{1 - \beta_t}x_{t - 1} + \beta_t\epsilon$.
  - $\beta_t$ values change aross time steps following a "variance schedule" (can be linear, quadratic, cosine, etc), kinda like learning rate schedule.
- Backward denoising diffusion process:
  - In the forward diffusion process, starting with an actual sample $x_0$, if we set the schedule appropriately, we end up with pure gaussian noise at $x_T$.
- TODO
