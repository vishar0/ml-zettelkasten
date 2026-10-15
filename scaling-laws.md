# Scaling Laws

- **Created**: 2026-06-17
- **Last Updated**: 2026-10-13
- **Status**: `In Progress`
- **Description**: The empirical science of how loss scales with compute, data, and parameters, including transfer across distributions and training under a fixed data budget.
- **Related**:
  - [[papers-foundation-models]] - Scaling laws guide how foundation models allocate parameters, data, and compute.
  - [[papers-data-centric-ml]] - Data selection can change or beat standard scaling curves.
  - [[diffusion#6. Scaling Laws and Compute Allocation]] - Scaling laws for diffusion models.

---

- [x] [2026] [blog] [LilianWeng] [Scaling Laws, Carefully](#2026-scaling-laws-carefully) — [blog](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/)
- [x] [2022] [blog] [Bahdanau] The FLOPs Calculus of Language Model Training — [blog](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4)
- [ ] [2026] [blog] How to Build a Scientific Scaling Ladder — [blog](https://jiaxuanzou0714.github.io/en/blog/2026/how-to-build-scientific-scaling-ladder/)
- [x] [2017] [Deep Learning Scaling is Predictable, Empirically](#2017-deep-learning-scaling-is-predictable-empirically) — [paper](https://arxiv.org/abs/1712.00409)
- [ ] [2019] A Constructive Prediction of the Generalization Error Across Scales — [paper](https://arxiv.org/abs/1909.12673)
- [ ] [2020] [Kaplan] Scaling Laws for Neural Language Models — [paper](https://arxiv.org/abs/2001.08361)
- [ ] [2022] Chincilla: Training Compute-Optimal Large Language Models — [paper](https://arxiv.org/abs/2203.15556)
- [ ] [2025] [Cerebras] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training — [paper](https://arxiv.org/abs/2505.13738)
- [ ] [2022] muP: Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer — [paper](https://arxiv.org/abs/2203.03466)
- [ ] [2021] Scaling Laws for Transfer — [paper](https://arxiv.org/abs/2102.01293)

## Scaling Laws in Data-Constrained Settings

Chinchilla assumes every training token is unique. These papers ask what changes when compute outgrows the supply of unique data and tokens must be repeated. See also the diffusion-vs-autoregressive comparison under data constraints in [[diffusion]].

- [ ] [2022] [Anthropic] Scaling Laws and Interpretability of Learning from Repeated Data — [paper](https://arxiv.org/abs/2205.10487)
- [ ] [2023] Scaling Data-Constrained Language Models — [paper](https://arxiv.org/abs/2305.16264)
- [ ] [2026] Prescriptive Scaling Laws for Data Constrained Training — [paper](https://arxiv.org/abs/2605.01640)

---

## [2026] [Scaling Laws, Carefully](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/)

- **Date**: 2026-10-12

---

- On Hestness et al. 2017:
  - > Interestingly, architecture changes the offset ($E$) of the power-law fit but does not change the exponent ($\alpha$). The slope of the power law appears to be a property of the problem domain rather than the model architecture.
- On Kaplan et al. 2020:
  - > Given a fixed compute budget, it is more efficient to train a very large model and stop before convergence than to train a smaller model all the way to convergence. This finding is where the Chinchilla scaling laws (the next section) disagree: Kaplan et al. overestimated the optimal model size as their fitted exponent was larger.
  - > The most influential and, in hindsight, most contested conclusion was the compute-optimal allocation. Kaplan et al. found $N_\text{opt} \propto C^{0.73}$ and concluded that model size should grow faster than dataset size. Concretely, for a 10x increase in compute they suggested scaling the model size by ~5.5x but the training tokens by only ~1.8x. The Chinchilla paper would later overturn this recommendation, arguing that it leaves large models badly undertrained.
- On Hoffmann et al. 2022 (Chinchilla):
  - > Chinchilla arrives at its answer through three complementary methods whose final results agree with each other, and this is part of why the result was quite convincing.
  - > The claim in the Chinchilla paper that most large models (at the time, ~2022) were undertrained is supported by a famous demonstration: under the same compute budget as Gopher (Rae et al. 2021; 280B parameter count, 300B token budget), they trained Chinchilla (70B parameter count, 1.4T token budget), a model 4x smaller but trained on roughly 4x more tokens and it outperformed Gopher across the board.
- On reconciling Kaplan and Chinchilla:
  - > The Chinchilla scaling laws disagree with Kaplan et al. as follows:
    > - Instead of "grow the model faster than the data" ($N_\text{opt} \propto C^{0.73}$), for every doubling of model size, you should also double the number of training tokens ($N_\text{opt} \propto C^{0.5}$).
    > - Instead of "train a big model and stop before convergence," you should train a smaller model on more data.
- On section 3, the data-limited regime (papers under [[scaling-laws#Scaling Laws in Data-Constrained Settings]]):
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
