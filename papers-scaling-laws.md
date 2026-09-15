# Scaling Laws

- **Created**: 2026-06-17
- **Last Updated**: 2026-09-03
- **Status**: `In Progress`
- **Description**: The empirical science of how loss scales with compute, data, and parameters, including transfer across distributions.
- **Related**:
  - [[papers-foundation-models]] - Scaling laws guide how foundation models allocate parameters, data, and compute.
  - [[papers-data-centric-ml]] - Data selection can change or beat standard scaling curves.
  - [[diffusion#6. Scaling Laws and Compute Allocation]] - Scaling laws for diffusion models.

---

- [ ] [2026] [LilianWeng] Scaling Laws, Carefully — [blog](https://lilianweng.github.io/posts/2026-06-24-scaling-laws/)
- [x] [2017] [Deep Learning Scaling is Predictable, Empirically](#2017-deep-learning-scaling-is-predictable-empirically) — [paper](https://arxiv.org/abs/1712.00409)
- [ ] [2020] Scaling Laws for Neural Language Models — [paper](https://arxiv.org/abs/2001.08361)
- [ ] [2022] Chincilla: Training Compute-Optimal Large Language Models — [paper](https://arxiv.org/abs/2203.15556)
- [ ] [2021] Scaling Laws for Transfer — [paper](https://arxiv.org/abs/2102.01293)
- [ ] [2025] nanochat scaling analysis - <https://github.com/karpathy/nanochat/blob/master/dev/LEADERBOARD.md>

---

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
- **What Chinchilla changes**: Chinchilla fits the joint loss model
  $$
  L(N,D)\approx E+\frac{A}{N^\alpha}+\frac{B}{D^\beta},
  $$
  and keeps Kaplan's compute-allocation question but revises the estimated optimum, finding that model size and training tokens should grow at approximately equal rates under increasing training compute.
