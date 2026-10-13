# Scaling Laws

- **Created**: 2026-06-17
- **Last Updated**: 2026-10-13
- **Status**: `In Progress`
- **Description**: The empirical science of how loss scales with compute, data, and parameters, including transfer across distributions.
- **Related**:
  - [[papers-foundation-models]] - Scaling laws guide how foundation models allocate parameters, data, and compute.
  - [[papers-data-centric-ml]] - Data selection can change or beat standard scaling curves.
  - [[diffusion#6. Scaling Laws and Compute Allocation]] - Scaling laws for diffusion models.

---

- [ ] [2026] [blog] [LilianWeng] [Scaling Laws, Carefully](#2026-scaling-laws-carefully) — [blog](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/)
- [ ] [2022] [blog] [Bahdanau] The FLOPs Calculus of Language Model Training — [blog](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4) — derives $C\approx 6ND$: 2 FLOPs per parameter per token forward, 4 backward; attention FLOPs dropped as second order
- [ ] [2026] [blog] How to Build a Scientific Scaling Ladder — [blog](https://jiaxuanzou0714.github.io/en/blog/2026/how-to-build-scientific-scaling-ladder/) — a checklist for a scaling ladder: fix the decision and acceptance criteria first, then the measurement spec, baseline recipe and scaling rules, the ladder's budget and hyperparameter search, loss-law fitting and downstream prediction, and validate the extrapolation before the large run
- [x] [2017] [Deep Learning Scaling is Predictable, Empirically](#2017-deep-learning-scaling-is-predictable-empirically) — [paper](https://arxiv.org/abs/1712.00409)
- [ ] [2019] [Rosenfeld] A Constructive Prediction of the Generalization Error Across Scales — [paper](https://arxiv.org/abs/1909.12673) — first joint fit of generalization error as a function of both model size and data size, $L(N,D)\approx E+A/N^{\alpha}+B/D^{\beta}$, across ResNet, WRN, LSTM and Transformer on ImageNet, CIFAR and WikiText-103; fits on small $(N,D)$ configurations extrapolate to larger ones, which is the premise behind scaling ladders; Chinchilla's Method 3 fits this same form
- [ ] [2020] [Kaplan] Scaling Laws for Neural Language Models — [paper](https://arxiv.org/abs/2001.08361)
- [ ] [2022] Chincilla: Training Compute-Optimal Large Language Models — [paper](https://arxiv.org/abs/2203.15556)
- [ ] [2024] [Besiroglu] Chinchilla Scaling: A replication attempt — [paper](https://arxiv.org/abs/2404.10102) — refits Chinchilla's Method 3 on points read off its Figure 4; the original averaged the Huber losses instead of summing, so L-BFGS stopped early on a too-small gradient, and reported exponents were rounded to two digits, which made $A$ and $B$ look inconsistent; the refit gives $\alpha\approx0.35$, $\beta\approx0.37$, so $N_{\text{opt}}\propto C^{0.51}$ and $D_{\text{opt}}\propto C^{0.49}$, in line with Methods 1 and 2; the paper's confidence intervals would have needed about 600k runs, not the fewer than 500 run; Weng's section 4 and its toy simulation build on this
- [ ] [2025] [Cerebras] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training — [paper](https://arxiv.org/abs/2505.13738) — $B_{\mathrm{opt}} \propto D^{0.383}$ and the constant-$B/(\eta\lambda D)$ weight-decay rule; nanochat uses both to scale batch size and weight decay from its d12 reference (dev/LOG.md 2026-02-05); the weight-decay side rests on the $T_{\text{epoch}}$ paper filed under [[papers-optimizers#Weight Decay]]
- [ ] [2022] [Yang] Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP) — [paper](https://arxiv.org/abs/2203.03466) — the maximal-update parametrization makes optimal hyperparameters stable across width, so they can be tuned on a small proxy and transferred zero-shot; nanochat's base_train.py tunes LR, batch size and weight decay at the d12 reference and transfers them to deeper models "muP style", then layers the Power Lines and $T_{\text{epoch}}$ scaling rules on top
- [ ] [2021] Scaling Laws for Transfer — [paper](https://arxiv.org/abs/2102.01293)

## Data-constrained scaling

- [ ] [2022] [Hernandez] Scaling Laws and Interpretability of Learning from Repeated Data — [paper](https://arxiv.org/abs/2205.10487) — a small repeated fraction inside mostly unique data hurts far more than its share: 0.1% of tokens repeated 100 times drops an 800M model to 400M performance; the damage is a double descent in repetition count, peaking where the model memorizes the repeated subset, and it knocks out induction heads; Weng's section 3 cites it for the data-limited regime; our multi-pass sweep repeats the whole dataset uniformly instead, the setting of Muennighoff 2023
- [ ] [2023] [Muennighoff] Scaling Data-Constrained Language Models — [paper](https://arxiv.org/abs/2305.16264) — the whole dataset repeated uniformly, 400 runs up to 9B params and 1500 epochs; Chinchilla's fit with exponentially discounted effective data and params, decay constants about 15 epochs for data and 5 for excess params; up to four epochs are nearly free, then returns diminish; our 2, 4, 8 pass sweep is this setting at 9.5 tokens per param
- [ ] [2026] [Lovelace] Prescriptive Scaling Laws for Data Constrained Training — [paper](https://arxiv.org/abs/2605.01640) — adds one additive overfitting penalty to the Chinchilla form, $P\,R_D^\delta\,(N/U_D)^\kappa$ in repeats and the capacity ratio; repetition helps up to a point and past it compute is better spent on model capacity; larger models are more sensitive to repetition; strong weight decay ($\lambda=1$) cuts the penalty coefficient by about 70%, a scaling-law reason the optimal weight decay in data-constrained training is an order of magnitude above standard practice; the prescribed allocation beats Chinchilla's in the data-constrained regime

---

## [2026] [Scaling Laws, Carefully](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/)

- **Date**: 2026-10-12

---

- On Hestness et al. 2017:
  - > Interestingly, architecture changes the offset ($E$) of the power-law fit but does not change the exponent ($\alpha$). The slope of the power law appears to be a property of the problem domain rather than the model architecture.
- On Kaplan et al. 2020:
  - > Given a fixed compute budget, it is more efficient to train a very large model and stop before convergence than to train a smaller model all the way to convergence. This finding is where the Chinchilla scaling laws (the next section) disagree: Kaplan et al. overestimated the optimal model size as their fitted exponent was larger.
  - > The most influential and, in hindsight, most contested conclusion was the compute-optimal allocation. Kaplan et al. found $N_\text{opt} \propto C^{0.73}$ and concluded that model size should grow faster than dataset size. Concretely, for a 10x increase in compute they suggested scaling the model size by ~5.5x but the training tokens by only ~1.8x. The Chinchilla paper would later overturn this recommendation, arguing that it leaves large models badly undertrained.
- On section 3, the data-limited regime (papers under [[scaling-laws#Data-constrained scaling]]):
  - [Hernandez et al. 2022](https://arxiv.org/abs/2205.10487): 90% unique tokens, 10% drawn from a tiny repeated subset, 100B tokens. Loss against repetition count is a double descent: a little repetition is harmless, extreme repetition is cheap to memorize, and the middle is where the model spends capacity memorizing the subset; it also hurts OOD evals and fine-tuning. The flat or rising middle of training breaks scaling-law fits.
  - [Muennighoff et al. 2023](https://arxiv.org/abs/2305.16264): uniform repetition of the whole dataset, 10M to 9B params, up to 1500 epochs. Chinchilla's fit with discounted data and params, $D'=U_D+U_D r_D(1-e^{-R_D/r_D})$ and the same form for $N'$, where $R_D$ is epochs minus one and $r_D$ a fitted decay constant. Fitted $r_D\approx15$, $r_N\approx5$: up to about four epochs are nearly free, returns diminish after, and excess params decay faster than repeated data, so spend extra compute on epochs rather than params.
  - [Lovelace et al. 2026](https://arxiv.org/abs/2605.01640): adds an explicit overfitting penalty $P\,R_D^\delta\,(N/U_D)^\kappa$ to the Chinchilla form; larger models are more sensitive to repetition, and strong weight decay shrinks the penalty.
  - Our multi-pass sweep (2, 4, 8 passes at fixed tokens, so unique tokens cut to an eighth) is Muennighoff's setting, but at 9.5 tokens per param one pass is already far below Chinchilla's 20, so 8 passes is 1.2 unique tokens per param. That capacity ratio is Lovelace's penalty regime and explains why the nanochat GPT loses half its CORE where Muennighoff's four free epochs would predict little. MDLM block diffusion sees a fresh mask each pass, which may be why it stays flat.
- On our ladder, against the post's sections ([text diffusion scaling notebook](https://github.com/inductivebias/flourish/pull/2346)):
  - Section 1, offset vs exponent: the nanochat GPT and MDLM block diffusion ladders (d12 to d24, 9.5 tokens per param) fit val bpb against training FLOPs with the same exponent, -0.049, and different coefficients. Hestness's claim holds across an AR vs diffusion change. Fitting CORE instead gives different exponents (0.16 vs 0.20), so fit the loss, not a downstream metric.
  - Section 2.1 vs 2.3, which $N$: nanochat's ladder sets tokens from non-embedding "scaling params" and excludes its value embedding tables, which are 43% to 53% of total params. Removing them ([#2404](https://github.com/inductivebias/flourish/pull/2404)) moves val bpb by about 0.01 and CORE within noise, so the choice of $N$ changes the coefficient more than the curve.
  - Section 2.2, fit form: the ladder fixes $D=9.5N$, so the fit is Kaplan's single-variable $L(C)$, not a Chinchilla joint $L(N,D)$. The ratio is AR-tuned; the compute-optimal ratio for diffusion needs IsoFLOP curves.
  - Section 3, repetition: at 2, 4 and 8 passes over the same tokens the nanochat GPT loses about half its CORE at 8 passes while MDLM block diffusion neither overfits nor improves, as in [arXiv 2507.15857](https://arxiv.org/abs/2507.15857).
  - Section 4, noise: reruns of d12 and d14 at one seed move CORE by up to 0.011 and val bpb by 0.001. Over seven rungs the val bpb exponent's standard error is 0.001. MDLM's val bpb is the NELBO, an upper bound, so only exponents compare across trainers.

## [2017] [Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409)

- **Date**: 2026-09-03

---

- **Motivation**: Deep learning advances through a recurring recipe: model architecture search, creating large training datasets, and scaling computation. Architecture advances, however, often depend on an unreliable epiphany, creative reframing, or large hyperparameter searches with some serendipity. Growing datasets and scaling computation are a lower-risk complement over which researchers have more control.
- **Big picture**: The paper presents a large-scale empirical characterization of how generalization error and model size change as training sets grow. It asks whether small experiments can predict the data and compute requirements for advancing the state of the art.
- **Notation used below**:
  - $D$: the amount of training data, measured in examples in Hestness and usually in tokens in language-model scaling laws.
  - $N$: model size, usually measured by the number of trainable parameters.
  - $C$: training compute, usually measured in FLOPs. For a dense Transformer, the common approximation is $C\approx 6ND$.
  - $\varepsilon(D)$: Hestness's validation or generalization error for the best model selected at dataset size $D$.
  - $L(N,D)$: the loss achieved by a model of size $N$ trained on $D$ data. Kaplan uses held-out language-model cross-entropy; Chinchilla uses smoothed training loss as an estimate of test loss in its effectively infinite-data regime.
  - $A$ and $B$: positive fitted scale coefficients, not universal constants. In Hestness's one-variable formula, $A$ sets the scale of the data-dependent error. In Chinchilla's joint formula, $A$ scales the model-limited term and $B$ scales the data-limited term.
  - $E$: the irreducible error or loss remaining in the infinite-model, infinite-data limit.
  - $\alpha$ and $\beta$: positive power-law exponents in the common modern form $A/N^\alpha+B/D^\beta$.
  - $\beta_g$: Hestness's negative generalization-error exponent, so $D^{\beta_g}=1/D^{|\beta_g|}$.
  - $\beta_p$: Hestness's positive exponent describing how the best model size grows with data, $N_{\mathrm{best}}(D)\propto D^{\beta_p}$.
- For each domain, the authors train models on successively larger data subsets and search for an appropriately sized model at each scale. Generalization error follows an approximate power law:
  $$
  \varepsilon(D) \approx A D^{\beta_g} + E,
  \qquad \beta_g < 0,
  $$
  where $D$ is dataset size, $\beta_g$ is the domain's learning-curve slope, and $E$ is the eventual error floor. After accounting for $E$, the relationship is a straight line on a log-log plot.
- Prior theory commonly predicted exponents such as $\beta_g=-0.5$ or $-1$. In the real applications studied here, the empirical exponents instead lie between roughly $-0.07$ and $-0.35$, and remain unexplained by existing theory.
- **Power-law learning curves appear across all four tested domains**—machine translation, language modeling, image classification, and speech recognition—and across a broad range of models, optimizers, regularizers, and loss functions. Different applications nevertheless have different exponents and intercepts.
- [**Figure 6**](https://arxiv.org/html/1712.00409#S5.F6) is the paper's clearest summary of the full learning curve. It is a conceptual sketch, rather than another empirical fit, and separates three qualitative regimes:
  1. **Small-data regime**: there is too little signal, so performance remains near guessing.
  2. **Power-law regime**: adding data produces smooth, predictable improvements.
  3. **Irreducible-error regime**: further data no longer removes error caused by noise, task ambiguity, or other limiting factors.
- **Improved architectures and optimizers shift the power-law intercept but do not appear to change its exponent**: models within one domain show the same learning-curve steepness. The rate at which additional data helps therefore appears more characteristic of the problem domain than of a particular architecture.
- The model size needed to use a growing dataset also follows a power law and generally grows sublinearly:
  $$
  N_{\mathrm{best}}(D) \propto D^{\beta_p},
  \qquad 0 < \beta_p < 1.
  $$
- **Implications**:
  1. **The learning curves of real applications**: Real learning curves move from a small-data region dominated by best guessing, through a power-law region whose exponent measures how quickly the model family benefits from more data, and eventually toward an irreducible-error region. The exponent appears to depend on the problem domain or data distribution. The lower bound includes Bayes error and other sources of imperfect generalization such as mislabeled data; the paper did not reach this region in its real applications.
  2. **Implications for practitioners and researchers**: The robustness of the power-law region makes it useful for debugging data, architecture, and optimization—divergence from the expected trend can indicate a deeper problem. Architecture improvements can shift a curve downward without improving its exponent. Beating the power law would require learning more concepts with successively less data, or extracting more marginal information from each new sample; the authors suggest studying data filtering and augmentation, few-shot learning, experience replay, and generative adversarial networks.
  3. **Operational implications**: Projections can encounter three limits: too little training data, computation that is too slow, or irreducible error. If a smaller dataset is already in the power-law region, researchers may be able to explore architectures cheaply and then scale the best candidate while preserving relative gains. The fitted curves can also forecast the model, data, and compute required for a target accuracy; moving below a genuine irreducible floor instead requires increasing the information content of the data.
  4. **Hardware-design implications**: Predictable learning and model-size curves can guide computing-system design by estimating how much throughput and memory are required to reach a target accuracy. They also help evaluate techniques such as low-precision computation and sparsity: a method may sacrifice some accuracy per training step but still win overall if its higher throughput permits training larger models on more data.
- **What Hestness measures**: For every dataset size $D$, the experiments search over model sizes and retain the best one:
  $$
  N^*=\arg\min_N L(N,D),
  \qquad
  L^*(D)=L(N^*,D).
  $$
  Hestness then fits a scaling law for $L^*(D)$ and for how $N^*$ grows with $D$. This tells us how the best achievable error and the corresponding model size change as data grows. However, selecting only the best model for each $D$ collapses the model-size dimension: it does not characterize the full surface $L(N,D)$ or constrain the compute used by the different runs.
- **What Kaplan adds**: Kaplan specializes to Transformer language models and measures how loss scales with model size, dataset size, and training compute:
  $$
  L(N),
  \qquad
  L(D),
  \qquad
  L(C),
  \qquad
  L(N,D).
  $$
  Since training compute is approximately $C\approx 6ND$, Kaplan can ask a new operational question:
  $$
  (N^*,D^*)
  =
  \arg\min_{N,D:\,6ND=C} L(N,D).
  $$
  In words, given a fixed compute budget, should it be spent on a larger model or on processing more tokens? Hestness does not directly answer this because its runs are not compared under a fixed compute constraint.
- **What Chinchilla changes**: Chinchilla fits the joint loss model first proposed by Rosenfeld et al. 2019 ([paper](https://arxiv.org/abs/1909.12673)),
  $$
  L(N,D)\approx E+\frac{A}{N^\alpha}+\frac{B}{D^\beta},
  $$
  and keeps Kaplan's compute-allocation question but revises the estimated optimum, finding that model size and training tokens should grow at approximately equal rates under increasing training compute.
