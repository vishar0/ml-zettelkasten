# Diffusion

- **Created**: 2025-08-19
- **Last Updated**: 2026-08-10
- **Status**: `In Progress`
- **Related**:
  - [[course-mit-diffusion]] — Structured MIT course with lecture notes, slides, recordings, and labs on flow matching and diffusion models.

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
- [ ] TODO Alan's stuff
  - [ ] [Flourish] Alan's diffusion tutorial slides - [slides](../../flourish/presentations/2026-05-21-diffusion/README.md)
  - [ ] Alan's paper list <https://docs.google.com/document/d/1dgvsHthnVjYMl0nqfFWeP0GITSMz6lopmQUNP3gDQ9M/edit?usp=sharing>
  - [ ] Alan's diffusion loss notebook

## 1. Classical Diffusion and Likelihood

- [ ] [2020] [JonathanHo] DDPM: Denoising Diffusion Probabilistic Models - [paper](https://arxiv.org/abs/2006.11239)
  - Read closely. Be able to derive $q(x_t\mid x_0)$, the reverse posterior, and the simplified noise-prediction loss.
- [ ] [2021] Improved Denoising Diffusion Probabilistic Models - [paper](https://arxiv.org/abs/2102.09672), [code](https://github.com/openai/improved-diffusion)
  - Practical sequel to DDPM: cosine noise schedule, learned reverse variances, hybrid loss, improved likelihood, and substantially fewer sampling steps through timestep respacing.
- [ ] [2015] [JaschaSohlDickstein] Deep Unsupervised Learning using Nonequilibrium Thermodynamics - [paper](https://arxiv.org/abs/1503.03585)
  - Read after DDPM rather than before it. Focus on the fixed forward process and learned reversal; skim older implementation details.
- [ ] [2021] [Greg-rec] [Kingma] Variational Diffusion Models - [paper](https://arxiv.org/abs/2107.00630), [code](https://github.com/google-research/vdm)
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
- [ ] [2023] [EmielHoogeboom,JonathanHeek,TimSalimans] Simple Diffusion: End-to-End Diffusion for High Resolution Images - [paper](https://arxiv.org/abs/2301.11093), [proceedings](https://proceedings.mlr.press/v202/hoogeboom23a.html)
  - Pixel-space alternative to latent diffusion and cascades. Focus on the resolution-dependent log-SNR shift, selective low-resolution scaling and dropout, early downsampling, and the multiscale loss; the shifted cosine schedule is the part used in Nando §2.3.
- [ ] [2025] [Greg-rec] [KaimingHe] Back to Basics: Let Denoising Generative Models Denoise - [paper](https://arxiv.org/abs/2511.13720)
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
- [ ] [2024] Rolling Diffusion Models - [paper](https://arxiv.org/abs/2402.09470)
  - Sliding-window denoising assigns progressively more noise to later frames, committing to the near-term future while preserving greater uncertainty farther ahead. Focus on Figure 2's rolling noise schedule and how it differs from applying one shared noise level to an entire temporal sequence.

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
- [ ] [2025] [Bengio] Monte Carlo Tree Diffusion for System 2 Planning - [paper](https://arxiv.org/abs/2502.07202)
  - Recast denoising as tree search over partially denoised plans: evaluate, branch, prune, and revisit candidates so planning quality can improve with inference-time compute. Read as an MCTS-style search extension to diffusion planning, not as a core diffusion prerequisite.
- [ ] [2023] Diffusion Policy: Visuomotor Policy Learning via Action Diffusion - [paper](https://arxiv.org/abs/2303.04137), [project](https://diffusion-policy.cs.columbia.edu/)
  - Action-sequence diffusion for visuomotor control.
- [ ] [2023] [Greg-rec] [SergeyLevine] IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies - [paper](https://arxiv.org/abs/2304.10573)
  - Interpret IQL as an actor-critic method, represent its potentially multimodal implicit actor with a diffusion behavior policy, and use critic-derived weights to extract the intended policy.
- [ ] [2025] [Greg-rec] Efficient Online Reinforcement Learning for Diffusion Policy - [paper](https://arxiv.org/abs/2502.00361)
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

---

## [2026] [Nando] [Diffusion and Flow Matching Tutorial](https://love4all.ai/files/diffusion-and-flow-matching-tutorial.pdf)

- **Date**: 2026-08-03
- **Notebook**: <https://love4all.ai/files/diffusion-and-flow-matching-tutorial.ipynb>

---

### Section 1: Introduction

_**TL;DR:** Diffusion turns generation into a supervised denoising problem: use a fixed process to corrupt real samples, learn to reverse that corruption, then generate by starting from Gaussian noise and repeatedly applying the learned reverse process. The tutorial takes score matching as its route to the training objective and previews flow matching as an ODE-based alternative view._

- **Problem setup**:
  - Represent any modality—images, video, audio, proteins, or molecules—as a vector $x$ drawn from an unknown real distribution $p_d(x)$.
  - Learn a model distribution $p_\theta(x)$ from which new samples can be generated: $x \sim p_\theta(x)$.
  - The tutorial's slogan, "match imagination to reality," is the high-level goal. It is not yet the computable objective; the later score-matching derivation supplies that objective without requiring direct access to $p_d(x)$.
  - > While there are other derivations using variational methods, here we will simply use a fundamental learning principle: match imagination to reality. That is, what the model imagines, predicts, or generates, must match the real data. This is the principle used to train LLMs too, but while LLMs typically use the Maximum Likelihood principle, here we will use the Score Matching principle.
- **The two processes**:
  - The fixed forward process $q$ starts at clean data $z_0=x$ and progressively adds Gaussian noise until $z_T \sim \mathcal{N}(0,I)$. Its schedule is designed rather than learned.
  - The learned backward process $p_\theta$ starts at $z_T$ and denoises step by step until it produces $z_0$. Training learns this reversal; generation runs it from right to left.
  - The forward process therefore manufactures the training signal: clean examples and their noisy versions. No separately labelled targets are needed.
  - > Figure 1: The two halves of a diffusion model. The forward process $q$ (top, blue) takes a clean datapoint $x = z_0$ and gradually corrupts it into pure Gaussian noise $z_T \sim \mathcal{N}(0,I)$ by adding a small amount of noise at each step. This direction is typically hand-designed: each transition is a Gaussian whose mean and variance are fixed by a schedule, with no learned parameters. The backward process $p_\theta$ (bottom, red, dashed) goes the other way and is what the neural network learns: starting from pure noise it denoises step-by-step until it produces a sample. Sampling at inference time is just running the bottom row from right to left to produce new images, speech, videos or molecules.
- **Training and inference plan**:
  - First derive a loss, then minimize it with ordinary gradient-based optimization such as Adam.
  - At inference time, sample from a Gaussian and use the trained neural network to reverse the noising process. Here _inference_ means generative sampling, not posterior inference over a latent variable.
  - Conditioning information—text, previous video frames, pose, camera view, or a quality score—can steer the same reverse process without changing the basic construction.
  - > To understand diffusion from first principles, we first need to derive a loss function, which will then be used to train the generative model. The loss function is often reparameterised to make it numerically stable. The data for the loss function will consist of the original image and noisy samples generated by a forward diffusion process, as shown in Figure 1. Using these data, we will train a neural network to undo the process of adding noise. Finally, such a network will enable us to start with any random sample and reverse it until we get an image. We will refer to this reverse process as inference.
  - > Once we have the loss function, we can minimize it with standard gradient descent approaches, such as Adam. For inference, we will derive a Gaussian distribution for sampling (generating) any type of data using the trained neural network. The generation can be unconditional or conditioned on signals such as past video frames, text, pose, camera view, quality score, and so on.
- **Relation to other generative models**:
  - Autoregressive language models and diffusion models share the goal of matching the model's generated distribution to the data distribution, but commonly operationalize it differently: next-token maximum likelihood versus score-based denoising.
  - The promised flow-matching route replaces the stochastic denoising-chain emphasis with conditional expectations and an ODE whose time-dependent vector field can be represented by a deep network such as a transformer.
  - > a simple but very powerful approach based on conditional expectation and ordinary differential equations (ODEs). This will allow us to arrive at flow matching, where very deep neural networks can be interpreted as running ODEs with transformer blocks to generate data.

### Section 2: Training

#### 2.1 Matching imagination to reality

_**TL;DR:** The ideal target is $p_\theta=p_d$, but neither the real density $p_d(x)$ nor the normalized model density $p_\theta(x)$ is generally available pointwise. The probability-space squared error is therefore a statement of intent rather than the loss that will actually be optimized._

- The data distribution $p_d(x)$ denotes the unknown mechanism that produced the training examples. A generative model $p_\theta(x)$ approximates it and supports sampling:

$$
x \sim p_\theta(x).
$$

- > The data (images, proteins, videos, songs) will be represented with the generic vector $x$. The real data is assumed to come from an unknown distribution $p_d(x)$. Since we don’t have access to this distribution, we will try to approximate it using a model distribution $p_\theta(x)$, with parameters $\theta$. After learning the model distribution, also known as the generative model, we will be able to generate new data from it. Mathematically, the process of generation is represented as follows: $x \sim p_\theta(x)$.

- The tutorial first writes distribution matching as

$$
\mathcal{L}(\theta)
=
\mathbb{E}_{x\sim p_d}
\left[
\frac{1}{2}
\left\|p_\theta(x)-p_d(x)\right\|_2^2
\right].
\tag{1}
$$

- > We want our model to assign the same probability as the world to all data configurations $x$. That is, we want to minimize the difference between these two distributions on expectation over all the possible realizations of the data.

- This loss would attain its ideal value when the model assigns the same density as the world to real-data configurations. It is not directly computable:
  - We have samples from $p_d$, not numerical values of $p_d(x)$.
  - A flexible model may provide an unnormalized energy for $x$ while leaving its global normalizing constant intractable.
  - The next section changes _what is matched_: instead of matching density values, it matches their log-density gradients.
  - > Matching what the model imagines (generates) to the data generated by the world seems like a natural goal for learning. However, this is hard because we cannot calculate probabilities for models directly (so we’ll have to use autoregression or, as we explain here, diffusion score matching). The reason we cannot calculate the probabilities has to do with the normalizing constant,

#### 2.2 Score matching

_**TL;DR:** Write the model as a normalized energy model. Its partition function depends on $\theta$ but not on $x$, so taking $\nabla_x\log p_\theta(x)$ removes it. Score matching then compares the model and data log-density gradients, although the unknown data score remains to be handled by denoising score matching in §2.3._

**Energy-based representation**

- > The model distribution $p_\theta(x)$ can be expressed in a very general normalized exponential form:

$$
p_\theta(x)
=
\frac{1}{Z(\theta)}e^{-E_\theta(x)}.
$$

- $E_\theta(x)$ is the energy. At fixed $\theta$, lower-energy configurations have greater probability.
- $Z(\theta)$ is the partition function:

$$
Z(\theta)
=
\int_{\mathcal{X}} e^{-E_\theta(x)}\,dx.
$$

- The quantifiers matter: normalization means

$$
\forall\theta,\qquad
\int_{\mathcal{X}}p_\theta(x)\,dx=1.
$$

In words, for each fixed parameter setting $\theta$, integrate over every possible $x$. We do not integrate over $\theta$.

- > In this representation, $Z$ is known as the normalizing constant or partition function. It ensures that over the whole set of values that the data can take, the model probability sums to 1 for all values of $\theta$.

- For a discrete sample space, replace the integral with

$$
Z(\theta)=\sum_{x\in\mathcal{X}}e^{-E_\theta(x)}.
$$

- > The denominator sums over all possible images in the universe so that $p_\theta(x)$ can be interpreted as a probability:
- > This partition function is typically intractable because the sum is simply too large. It belongs to a complexity class known as sharp P, which in short means bloody hard if not impossible.

**Why maximum likelihood is hard for an unrestricted energy model**

$$
\log p_\theta(x)
=
-E_\theta(x)-\log Z(\theta).
$$

- For fixed $\theta$, minimizing $E_\theta(x)$ over $x$ is equivalent to maximizing $p_\theta(x)$ over $x$.
- When learning $\theta$, however, the partition function cannot be ignored because it also changes with $\theta$:

$$
\nabla_\theta\log p_\theta(x)
=
-\nabla_\theta E_\theta(x)
-\nabla_\theta\log Z(\theta).
$$

Thus, simply lowering the energy of training examples is not by itself maximum-likelihood learning; the normalizer accounts for what happens to all other configurations.

- > The quantity in the exponent is known as the energy. Physicists often prefer to use the terminology of minimizing energy, but clearly this is equivalent to maximising the model probability. Maximising the probability of the data by modifying the model parameters is known as maximum likelihood.

**The autoregressive LLM route**

- An LLM avoids one global normalization over all complete sequences by applying the probability chain rule:

$$
p_\theta(x_{1:T})
=
\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

- Each conditional is normalized only over the vocabulary $\mathcal{V}$:

$$
\sum_{v\in\mathcal{V}}
p_\theta(x_t=v\mid x_{<t})
=1.
$$

- Consequently, next-token maximum likelihood is tractable:

$$
-\log p_\theta(x_{1:T})
=
-\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

- > For the practical applications we care about, we cannot do this sum. A decade ago we weren’t too optimistic, but we have learned since then that it is actually possible to approximate this well for text, images, video, audio and other natural signals. One possible solution is to break the data $x$ into small blocks and process each block auto-regressively (this is basically what LLMs do).

**The score-matching route**

- > An alternative is to do what we are about to learn to do in this document. If we can’t do the sum, let’s get rid of the sum! We can do this by taking the log of the model probability and then computing its gradient with respect to the data

$$
\begin{aligned}
p_\theta(x)
&=\frac{1}{Z(\theta)}e^{-E_\theta(x)},\\
\log p_\theta(x)
&=-\log Z(\theta)-E_\theta(x),\\
\nabla_x\log p_\theta(x)
&=-\nabla_x E_\theta(x).
\end{aligned}
$$

- The last equality holds because $Z(\theta)$ has no dependence on $x$:

$$
\nabla_x\log Z(\theta)=0.
$$

- The **score** of a distribution is its log-density gradient with respect to the data:

$$
s_\theta(x)
=
\nabla_x\log p_\theta(x).
$$

- It is a vector field over data space. Locally, it points in the direction in which the model's log probability increases most quickly.
- Score matching asks the model vector field to equal the data vector field:

$$
\mathcal{L}(\theta)
=
\mathbb{E}_{x\sim p_d}
\left[
\left\|
\nabla_x\log p_\theta(x)
-
\nabla_x\log p_d(x)
\right\|_2^2
\right].
\tag{2}
$$

- > We will now reframe learning as matching the gradient of the model probability and the gradient of the distribution of the data. Intuitively, we want the rate of change in the modelled energy to match the rate of change of the real energy. This is known as score matching:

- This removes the model partition function, but it does not yet give a trainable objective because $\nabla_x\log p_d(x)$ is unknown. Section 2.3 introduces the Gaussian corruption process needed before §2.4 derives a computable objective.
  - > Getting rid of $Z$ is not enough. We still don’t have an expression for the derivative of the data distribution: $\nabla_x\log p_d(x)$. The model derivative $\nabla_x\log p_\theta(x)$, known as the score function, can be easily calculated using backpropagation.

#### 2.3 Denoised score matching

_**TL;DR:** Replace the singular, unknown data distribution with a family of smooth noisy distributions. At a randomly selected noise level $t$, mix a clean sample $x$ with known Gaussian noise $\epsilon_t$ to obtain $z_t$. Because the corruption kernel is known and the injected noise is recorded, the next section can turn score learning into supervised regression._

**Gaussian corruption at one noise level**

- Center a Gaussian corruption kernel on a scaled version of each clean sample:

$$
z_t
\sim
q_t(z_t\mid x)
=
\mathcal{N}\!\left(z_t\mid \alpha_t x,\sigma_t^2 I\right).
$$

- > Assume instead that we can place a Gaussian $q(\cdot)$ concentrated on each data point $x$ and then draw a sample $z_t$. This Gaussian will have two scalar hyperparameters taking values between 0 and 1. A hyperparameter $\alpha_t$ will be used to scale the data, e.g. scale an image $x$. A second hyperparameter $\sigma_t^2$ will determine the Gaussian variance.
- > When $\alpha_t = 0$ the Gaussian will have mean zero, and when $\alpha_t = 1$ the Gaussian will have mean $x$. Later we will show how we can parameterise $\alpha_t$ so that by modifying the subindex $t$, $\alpha_t$ will vary from 1 to 0, and $\sigma_t^2$ in turn will vary from 0 to 1.

- Use the Gaussian reparameterization

$$
\epsilon_t\sim\mathcal{N}(0,I),
\qquad
z_t=\alpha_t x+\sigma_t\epsilon_t.
\tag{3}
$$

  Here $\alpha_t$ controls how much clean signal remains and $\sigma_t$ controls the noise standard deviation. The variance is $\sigma_t^2$.

- > In other words $z_t$ is a bit like the image $x$ and a bit like Gaussian noise $\epsilon_t$.

- Nando reserves $x$ for clean data and uses $z_t$ for its noisy version. Other diffusion treatments often call the same noisy variable $x_t$.
- This is a direct marginal corruption $q_t(z_t\mid x)$: during training, one can choose any $t$ and construct $z_t$ in a single operation. There is no need to simulate $z_1,\ldots,z_{t-1}$ first.
- After mixing clean samples over the data distribution, the noisy marginal is

$$
q_t(z_t)
=
\int_{\mathcal{X}}q_t(z_t\mid x)p_d(x)\,dx.
$$

  Even when $p_d$ itself is only available through samples, Gaussian convolution makes $q_t$ smoother and gives the training procedure a known conditional corruption mechanism.

**Multiple noise scales**

- Let $t$ move between a clean endpoint and a Gaussian-noise endpoint:

$$
\begin{aligned}
t\approx0:&\qquad \alpha_t\approx1,\quad \sigma_t\approx0,\quad z_t\approx x,\\
t\approx1:&\qquad \alpha_t\approx0,\quad \sigma_t\approx1,\quad z_t\approx\epsilon_t.
\end{aligned}
$$

- Intermediate values of $t$ create intermediate corruption levels. Training across randomly sampled $t$ teaches one time-conditioned network how to denoise everywhere from nearly clean data to nearly pure noise.
- > We have introduced the index $t$ because we will allow for sampling at multiple scales. At the very noisy scale, when $\alpha \approx 0$ and $\sigma \approx 1$, $z_t$ will be basically Gaussian noise. At the other no noise extreme, when $\alpha \approx 1$ and $\sigma \approx 0$, $z_t \approx x$. We will choose a schedule to obtain samples between these two extremes.
- When the data have approximately unit variance, the common constraint

$$
\alpha_t^2+\sigma_t^2=1
$$

  keeps the overall variance of $z_t$ approximately constant while trading signal for noise.

**SNR and log-SNR**

- > The ratio of hyper-parameters is known as the signal-to-noise ratio:

- Define the signal-to-noise ratio by

$$
\operatorname{SNR}_t
=
\frac{\alpha_t^2}{\sigma_t^2},
$$

- > We often use the log-SNR:

- Define its logarithm by

$$
\lambda_t
=
\log\operatorname{SNR}_t
=
\log\frac{\alpha_t^2}{\sigma_t^2}.
\tag{5}
$$

- Interpretation:
  - $\lambda_t\gg0$: signal dominates; $z_t$ is nearly clean.
  - $\lambda_t=0$: signal and noise powers are equal.
  - $\lambda_t\ll0$: noise dominates; $z_t$ is nearly Gaussian.
- Log-SNR by itself determines only the ratio $\alpha_t^2/\sigma_t^2$. To recover the two coefficients individually, impose the variance-preserving convention

$$
\alpha_t^2+\sigma_t^2=1.
$$

- Exponentiating the log-SNR definition gives

$$
e^{\lambda_t}
=
\frac{\alpha_t^2}{\sigma_t^2},
\qquad\text{so}\qquad
\alpha_t^2=e^{\lambda_t}\sigma_t^2.
$$

  Substitute this into the variance-preserving constraint:

$$
\begin{aligned}
e^{\lambda_t}\sigma_t^2+\sigma_t^2&=1,\\
\sigma_t^2(1+e^{\lambda_t})&=1,\\
\sigma_t^2&=\frac{1}{1+e^{\lambda_t}}
=\operatorname{sigmoid}(-\lambda_t).
\end{aligned}
$$

  The complementary signal variance is therefore

$$
\alpha_t^2
=
\frac{1}{1+e^{-\lambda_t}}
=
\operatorname{sigmoid}(\lambda_t),
\tag{6}
$$

$$
\sigma_t^2
=
\frac{1}{1+e^{\lambda_t}}
=
\operatorname{sigmoid}(-\lambda_t).
\tag{7}
$$

- The corruption equation $z_t=\alpha_t x+\sigma_t\epsilon_t$ uses the signal and noise **standard-deviation coefficients**, not their squares. Taking the nonnegative roots yields the implementation:

$$
\boxed{
\alpha_t
=
\sqrt{\operatorname{sigmoid}(\lambda_t)}
},
\qquad
\boxed{
\sigma_t
=
\sqrt{\operatorname{sigmoid}(-\lambda_t)}
}.
$$

**Cosine noise schedule**

- Choose

$$
\alpha_t=\cos\!\left(\frac{\pi t}{2}\right),
\qquad
\sigma_t=\sin\!\left(\frac{\pi t}{2}\right).
$$

  Then $\alpha_t^2+\sigma_t^2=1$ and

$$
\lambda_t
=\log\operatorname{SNR}_t
=\log\frac{\alpha_t^2}{\sigma_t^2}
=\log\frac{\cos^2(\pi t/2)}{\sin^2(\pi t/2)}
=\log\frac{1}{\tan^2(\pi t/2)}
=-2\log\tan(\pi t/2).
$$

- > With $\alpha_t = \cos(\pi t/2)$ and $\sigma_t = \sin(\pi t/2)$, we have $\alpha_t^2 + \sigma_t^2 = 1$, and hence $\lambda_t = -2\log\tan(\pi t/2)$. This is known as the cosine schedule, and it is a very popular choice

**Endpoint behavior**

$$
\begin{aligned}
t\to0^+:&\quad (\alpha_t,\sigma_t)\to(1,0)
\quad\Longrightarrow\quad \operatorname{SNR}_t\to+\infty
\quad\Longrightarrow\quad \lambda_t\to+\infty,\\
t\to1^-:&\quad (\alpha_t,\sigma_t)\to(0,1)
\quad\Longrightarrow\quad \operatorname{SNR}_t\to0
\quad\Longrightarrow\quad \lambda_t\to-\infty.
\end{aligned}
$$

These are limits, not finite numerical values at the endpoints. The implementation therefore restricts log-SNR to finite bounds $[\lambda_{\min},\lambda_{\max}]$.

**Deriving the bounded cosine noise schedule**

- Write the cosine-schedule angle as

$$
u=\frac{\pi t}{2},
\qquad
\lambda=-2\log\tan u.
$$

- To find the angle corresponding to a desired finite log-SNR, invert that equation:

$$
\begin{aligned}
\lambda&=-2\log\tan u,\\
-\frac{\lambda}{2}&=\log\tan u,\\
e^{-\lambda/2}&=\tan u,\\
u&=\arctan\!\left(e^{-\lambda/2}\right).
\end{aligned}
$$

- The code uses this inverse to find safe angular endpoints:

$$
u_{\mathrm{lo}}
=
\arctan\!\left(e^{-\lambda_{\max}/2}\right),
\qquad
u_{\mathrm{hi}}
=
\arctan\!\left(e^{-\lambda_{\min}/2}\right).
$$

  Despite their names in the code, `_t_lo` and `_t_hi` are angles $u$, not normalized diffusion times.

- It maps the external time $t\in[0,1]$ linearly between those angles:

$$
u(t)
=
u_{\mathrm{lo}}
+t\left(u_{\mathrm{hi}}-u_{\mathrm{lo}}\right),
$$

  and then evaluates

$$
\lambda_t
=
-2\log\tan u(t).
$$

- With the defaults $\lambda_{\max}=15$ and $\lambda_{\min}=-15$,

$$
u_{\mathrm{lo}}
=
\arctan(e^{-7.5})
\approx0.00055,
$$

$$
u_{\mathrm{hi}}
=
\arctan(e^{7.5})
\approx\frac{\pi}{2}-0.00055.
$$

  Thus the implementation follows almost the entire cosine curve while avoiding exact zero signal, exact zero noise, and infinite log-SNR.

- The sigmoid conversion is also consistent with the original trigonometric schedule. Using $\lambda=-2\log\tan u$,

$$
\begin{aligned}
\operatorname{sigmoid}(\lambda)
&=\frac{1}{1+e^{-\lambda}}
=\frac{1}{1+\tan^2u}
=\cos^2u,\\
\operatorname{sigmoid}(-\lambda)
&=\sin^2u.
\end{aligned}
$$

  Because $u\in[0,\pi/2]$, taking the nonnegative square roots recovers $\alpha_t=\cos u$ and $\sigma_t=\sin u$.

- The implementation also supports the shifted-cosine schedule used for different image resolutions:

$$
\lambda_t^{\mathrm{shifted}}
=
\lambda_t
+2\log\!\left(\frac{d_{\mathrm{noise}}}{d_{\mathrm{image}}}\right).
$$

  Increasing image resolution relative to the reference noise resolution makes this shift negative, applying more corruption at the same nominal $t$.

**Minimal implementation for §2.3**

- This follows the paper's two §2.3 code blocks while trimming their long argument and return-value docstrings.
- `_t_lo` and `_t_hi` implement the finite-endpoint derivation above; they are angles despite their names.
- `add_noise` returns the injected noise alongside $z_t$ because that known $\epsilon_t$ becomes the regression target in the following objective.

```python
import math

import torch
from torch import Tensor


def add_noise(
    clean_sample: Tensor,
    alpha_t: Tensor,
    sigma_t: Tensor,
) -> tuple[Tensor, Tensor]:
    """Forward diffusion step: z_t = alpha_t * x + sigma_t * noise."""
    noise = torch.randn_like(clean_sample)
    noisy_sample = alpha_t * clean_sample + sigma_t * noise
    return noisy_sample, noise


class CosineNoiseSchedule:
    """Cosine log-SNR schedule with optional resolution-dependent shift."""

    def __init__(
        self,
        log_snr_min: float = -15.0,
        log_snr_max: float = 15.0,
        shift: float = 0.0,
    ) -> None:
        self.shift = shift
        self._t_lo = math.atan(math.exp(-0.5 * log_snr_max))
        self._t_hi = math.atan(math.exp(-0.5 * log_snr_min))

    def log_snr(self, t: Tensor) -> Tensor:
        if torch.any((t < 0) | (t > 1)):
            raise ValueError("t must be in [0, 1].")
        clipped_t = self._t_lo + t * (self._t_hi - self._t_lo)
        return -2.0 * torch.log(torch.tan(clipped_t)) + self.shift

    def alpha_sigma(self, log_snr_t: Tensor) -> tuple[Tensor, Tensor]:
        alpha_t = torch.sqrt(torch.sigmoid(log_snr_t))
        sigma_t = torch.sqrt(torch.sigmoid(-log_snr_t))
        return alpha_t, sigma_t
```

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
