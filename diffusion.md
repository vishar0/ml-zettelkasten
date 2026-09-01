# Diffusion

- **Created**: 2025-08-19
- **Last Updated**: 2026-08-25
- **Status**: `In Progress`
- **Related**:
  - [[course-mit-diffusion-2026]] — Structured MIT course with lecture notes, slides, recordings, and labs on flow matching and diffusion models.

---

TODO:

- TODO cvpr diffusion workshop
- TODO kaiming he diffusion stuff (papers, but also cvpr 2025 slides on meanflow)
- TODO berkeley diffusion workshop simons
- TODO ben poole
- TODO arash vahdat

---

## 0. Tutorials

- [ ] [[course-mit-diffusion-2026]]
- [ ] [2022] [HuggingFace] The Annotated Diffusion Model — [blog](https://huggingface.co/blog/annotated-diffusion)
- [ ] [2025] [3blue1brown] <https://www.3blue1brown.com/lessons/diffusion-models/>
- [ ] [2026] [Nando] Diffusion and Flow Matching Tutorial - [blog](https://love4all.ai/blog/diffusion-and-flow-matching-tutorial/), [pdf](https://love4all.ai/files/diffusion-and-flow-matching-tutorial.pdf), [notebook](https://love4all.ai/files/diffusion-and-flow-matching-tutorial.ipynb)
  - First technical treatment: derive the losses and sampling procedures while staying close to implementable code.
- [ ] [2026] [book] <https://the-principles-of-diffusion-models.github.io/>
- [ ] [2025] Demystifying Variational Diffusion Models - [paper](https://arxiv.org/abs/2401.06281)
  - Optional prerequisite if ELBOs, directed graphical models, or latent-variable models are rusty.
- [ ] TODO something on twitter i bookmarked, jacob shared on slack as well (about diffusion tutorial)
- [ ] TODO Alan's stuff
  - [ ] [Flourish] Alan's diffusion tutorial slides - [slides](../../flourish/presentations/2026-05-21-diffusion/README.md)
  - [ ] Alan's paper list <https://docs.google.com/document/d/1dgvsHthnVjYMl0nqfFWeP0GITSMz6lopmQUNP3gDQ9M/edit?usp=sharing>
  - [x] Alan's diffusion loss notebook <https://github.com/inductivebias/flourish/pull/1400>
  - [ ] Alan's diffusion ELBO notebook <https://github.com/inductivebias/flourish/pull/1597>

## 1. Classical Diffusion and Likelihood

- [ ] [2020] [JonathanHo] DDPM: Denoising Diffusion Probabilistic Models - [paper](https://arxiv.org/abs/2006.11239)
  - Read closely. Be able to derive $q(x_t\mid x_0)$, the reverse posterior, and the simplified noise-prediction loss.
- [ ] [2021] [AlexNichol,PrafullaDhariwal] Improved Denoising Diffusion Probabilistic Models - [paper](https://arxiv.org/abs/2102.09672), [code](https://github.com/openai/improved-diffusion)
  - Practical sequel to DDPM: cosine noise schedule, learned reverse variances, hybrid loss, improved likelihood, and substantially fewer sampling steps through timestep respacing.
- [ ] [2021] [PrafullaDhariwal,AlexNichol] ADM: Diffusion Models Beat GANs on Image Synthesis - [paper](https://arxiv.org/abs/2105.05233), [code](https://github.com/openai/guided-diffusion)
- [ ] [2021] [JonathanHo,TimSalimans] CFG: Classifier-Free Diffusion Guidance - [paper](https://arxiv.org/abs/2207.12598)
  - Learn conditional and unconditional scores jointly through condition dropout, then combine them at sampling time to trade diversity for condition adherence and fidelity—without a separate classifier.
- [ ] [2024] [TeroKarras] Autoguidance: Guiding a Diffusion Model with a Bad Version of Itself - [paper](https://arxiv.org/abs/2406.02507)
  - Replace CFG's unconditional model with a smaller or undertrained conditional model. Extrapolating away from this degraded model improves sample quality while better separating quality from condition alignment and diversity; it also works for unconditional generation.
- [ ] [2015] [JaschaSohlDickstein] Deep Unsupervised Learning using Nonequilibrium Thermodynamics - [paper](https://arxiv.org/abs/1503.03585)
  - Read after DDPM rather than before it. Focus on the fixed forward process and learned reversal; skim older implementation details.
- [ ] [2021] [Greg-rec] [Kingma] Variational Diffusion Models - [paper](https://arxiv.org/abs/2107.00630), [code](https://github.com/google-research/vdm)
  - Essential bridge among SNR, the variational bound, estimator variance, likelihood, and bits-back compression.

## 2. Score, SDE, ODE, and Flow Views

- [ ] [2005] [AapoHyvarinen,PeterDayan] Estimation of Non-Normalized Statistical Models by Score Matching - [paper](https://jmlr.org/papers/v6/hyvarinen05a.html)
  - Foundation of score matching: understand why the partition function disappears and how integration by parts replaces the inaccessible data score with a tractable objective.
- [ ] [2019] [YangSong] NCSN: Generative Modeling by Estimating Gradients of the Data Distribution - [paper](https://arxiv.org/abs/1907.05600)
  - Understand the score and why estimation at multiple noise levels makes sampling usable.
- [ ] [2020] [YangSong,StefanoErmon] NCSNv2: Improved Techniques for Training Score-Based Generative Models - [paper](https://arxiv.org/abs/2006.09011)
  - Practical refinement of NCSN: improved noise schedules, architecture, normalization, and annealed Langevin sampling for stable high-resolution generation.
- [ ] [2021] [YangSong] Score-Based Generative Modeling through Stochastic Differential Equations - [paper](https://arxiv.org/abs/2011.13456)
  - Understand the reverse-time SDE and probability-flow ODE.
- [ ] [2023] [YaronLipman] Flow Matching for Generative Modeling - [paper](https://arxiv.org/abs/2210.02747)
  - Learn conditional flow matching and why simulation-free training recovers a marginal vector field.
- [ ] [2024] [YaronLipman] Flow Matching Guide and Code - [paper](https://arxiv.org/abs/2412.06264), [code](https://github.com/facebookresearch/flow_matching)
  - Practical, self-contained treatment of continuous and discrete flow matching; read after the original Flow Matching paper for clearer derivations, design choices, and implementations.
- [ ] [2022] Elucidating the Design Space of Diffusion-Based Generative Models - [paper](https://arxiv.org/abs/2206.00364)
  - Separate parameterization, preconditioning, noise distribution, loss weighting, and sampler choice.

**Checkpoint:** translate among $x$-prediction, $\epsilon$-prediction, $v$-prediction, score prediction, and velocity prediction. Distinguish pure reparameterizations from genuinely different paths, objectives, or loss weightings.

## 3. Representation, Architecture, and Fast Sampling

- [x] [2022] [BillPeebles,SainingXie] DiT: Scalable Diffusion Models with Transformers - [paper](https://arxiv.org/abs/2212.09748)
  - Standard DiT architecture and timestep / conditioning machinery.
- [ ] [2021] [JonathanHo,ChitwanSaharia,TimSalimans] CDM: Cascaded Diffusion Models for High Fidelity Image Generation - [paper](https://arxiv.org/abs/2106.15282)
  - Generate images through a low-resolution base model followed by diffusion super-resolution models; conditioning augmentation makes later stages robust to errors from earlier generated stages.
- [ ] [2021] LDM: High-Resolution Image Synthesis with Latent Diffusion Models - [paper](https://arxiv.org/abs/2112.10752)
- [ ] [2022] [AdityaRamesh] Hierarchical Text-Conditional Image Generation with CLIP Latents - [paper](https://arxiv.org/abs/2204.06125)
  - DALL·E 2 / unCLIP: generate a CLIP image embedding from text, then condition a diffusion decoder on that semantic representation.
- [ ] [2022] [ChitwanSaharia,JonathanHo] Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding - [paper](https://arxiv.org/abs/2205.11487)
  - Imagen: frozen T5 text conditioning, cascaded pixel-space diffusion, and the finding that scaling the language encoder mattered more than scaling the image denoiser.
- [ ] [2023] [TimSalimans] Simple Diffusion: End-to-End Diffusion for High Resolution Images - [paper](https://arxiv.org/abs/2301.11093)
  - Pixel-space alternative to latent diffusion and cascades. Focus on the resolution-dependent log-SNR shift, selective low-resolution scaling and dropout, early downsampling, and the multiscale loss; the shifted cosine schedule is the part used in Nando §2.3.
- [ ] [[state-space-models]] [2024] SSM Meets Video Diffusion Models: Efficient Long-Term Video Generation with Structured State Spaces - [paper](https://arxiv.org/abs/2403.07711)
  - Replace quadratic temporal attention in a video-diffusion denoiser with bidirectional SSM layers for more memory-efficient long-video generation.
- [ ] [2025] [Greg-rec] [KaimingHe] Back to Basics: Let Denoising Generative Models Denoise - [paper](https://arxiv.org/abs/2511.13720)
  - "Just Image Transformers": clean-data $x$-prediction with simple large-patch transformers directly on pixels. Treat as a promising design lead, not settled doctrine.
- [ ] [2022] [TimSalimans,JonathanHo] Progressive Distillation for Fast Sampling of Diffusion Models - [paper](https://arxiv.org/abs/2202.00512)
  - Repeatedly distill two deterministic teacher sampling steps into one student step, halving the sampling budget each round. Read before Shortcut Models and MeanFlow for the foundational teacher-based route from many-step diffusion to few-step generation and for the motivation behind the $v$-parameterization.
- [ ] [2024] [DanijarHafner,SergeyLevine,PieterAbbeel] One-Step Diffusion via Shortcut Models - [paper](https://arxiv.org/abs/2410.12557)
- [ ] [2025] [KaimingHe] Mean Flows for One-Step Generative Modeling - [paper](https://arxiv.org/abs/2505.13447)
  - Read Shortcut Models and MeanFlow after the basic flow view; compare instantaneous velocity, average velocity, and step-conditioned prediction.
- [ ] [2025] [KaimingHe,JZicoKolter] iMF: Improved Mean Flows: On the Challenges of Fastforward Generative Models - [paper](https://arxiv.org/abs/2512.02012), [code](https://github.com/Lyy-iiis/imeanflow)
  - Stabilize MeanFlow by expressing training as a conventional instantaneous-velocity regression while parameterizing the average velocity, and condition explicitly on guidance settings so CFG remains adjustable at inference. Achieves one-step generation from scratch without distillation.
- [ ] [2024] Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction - [paper](https://arxiv.org/abs/2404.02905)
  - Important foil: coarse-to-fine autoregression may retain global-to-local generation with lower-variance likelihood training.
- [ ] [2025] [Greg-rec] DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation - [paper](https://arxiv.org/abs/2506.14202), [blog](https://sakana.ai/diffusion-blocks/), [code](https://github.com/SakanaAI/DiffusionBlocks)
  - Use the diffusion interpretation of residual updates and score matching to train Transformer blocks independently, reducing activation memory. Read as an adjacent application of diffusion ideas to neural-network training, not as a core generative-model prerequisite.

## 4. Discrete and Block Diffusion

- [ ] [2021] Structured Denoising Diffusion Models in Discrete State-Spaces - [paper](https://arxiv.org/abs/2107.03006)
  - Learn transition matrices, absorbing-mask corruption, and the discrete ELBO.
- [ ] [2024] Simple and Effective Masked Diffusion Language Models - [paper](https://arxiv.org/abs/2406.07524)
  - Bridge from D3PM theory to masked-token diffusion in practice.
- [ ] [2024] [YaronLipman] Discrete Flow Matching - [paper](https://arxiv.org/abs/2407.15595)
  - Connect conditional flow matching to continuous-time Markov chains and discrete rate matrices.
- [ ] [2025] Edit Flows: Variable Length Discrete Flow Matching with Sequence-Level Edit Operations - [paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/cb43f46154e750746602faaffd65fbbb-Paper-Conference.pdf)
  - Extend discrete flow matching from fixed-position token transitions to sequence-level insertions, deletions, and substitutions, enabling non-autoregressive variable-length generation through a CTMC over sequences.
- [ ] [2025] Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models - [paper](https://arxiv.org/abs/2503.09573)
  - Relevant to variable-length generation, KV caching, parallel token sampling, and controlling estimator variance.

## 5. Mixed-Modality and Sequence Models

- [ ] [2024] MAR: Autoregressive Image Generation without Vector Quantization - [paper](https://arxiv.org/abs/2406.11838)
  - Autoregressively generates image patches without discrete tokenization, using diffusion to model each patch's continuous-valued conditional distribution.
- [ ] [2024] Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model - [paper](https://arxiv.org/abs/2408.11039)
  - Concrete single-transformer treatment of mixed discrete text and continuous images using modality-specific losses.
- [ ] [2024] Diffusion Forcing: Next-Token Prediction Meets Full-Sequence Diffusion - [paper](https://arxiv.org/abs/2407.01392), [project](https://boyuan.space/diffusion-forcing)
  - Independent noise levels per token unify causal next-step prediction, full-sequence diffusion, variable-horizon rollouts, planning, and guidance.
- [ ] [2024] Rolling Diffusion Models - [paper](https://arxiv.org/abs/2402.09470)
  - Sliding-window denoising assigns progressively more noise to later frames, committing to the near-term future while preserving greater uncertainty farther ahead. Focus on Figure 2's rolling noise schedule and how it differs from applying one shared noise level to an entire temporal sequence.

**Checkpoint:** design a corruption process for $(o_t,a_t,r_t)$ in which pixels use continuous noise, actions and rewards use a CTMC or mask process, and different frames may have different noise levels. Justify both its loss and sampling semantics.

## 6. Scaling Laws and Compute Allocation

- [ ] [2025] Scaling Inference Time Compute for Diffusion Models - [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Ma_Scaling_Inference_Time_Compute_for_Diffusion_Models_CVPR_2025_paper.html)
  - Scale test-time compute through verifier-guided search over initial-noise candidates rather than merely increasing denoising steps.
- [ ] [2025] [DeepakPathak] Diffusion Beats Autoregressive in Data-Constrained Settings - [paper](https://arxiv.org/abs/2507.15857), [project](https://diffusion-scaling.github.io/)
  - Derive when masked diffusion overtakes autoregression as fixed data is reused with increasing compute; randomized masking acts as implicit augmentation over token orderings.
- [ ] [2026] Scaling Beyond Masked Diffusion Language Models - [paper](https://arxiv.org/abs/2602.15014)
  - Compare scaling across masked, uniform-state, and interpolating discrete diffusion; distinguish likelihood scaling from the practical speed-quality frontier.
- [ ] [2026] [JiamingSong] Abra: Scaling Diffusion Image Training - [paper](https://arxiv.org/abs/2608.17286)
  - Compute-optimal scaling laws for text-to-image flow-matching Transformers, including model/data allocation, overtraining robustness, generative quality, CFG, and representation quality.

## 7. Compression and Evaluation

- [ ] [2015] A Note on the Evaluation of Generative Models - [paper](https://arxiv.org/abs/1511.01844)
  - Mandatory: strong likelihood / compression, perceptual samples, and downstream behavior can be largely independent in high dimensions.
- [ ] [2019] Practical Lossless Compression with Latent Variables using Bits Back Coding - [paper](https://arxiv.org/abs/1901.04866), [code](https://github.com/bits-back/bits-back)
  - Understand how a variational bound becomes a realizable code rather than only an evaluation number.
- [ ] Revisit [[compression]], [[nncp]], and [Language Modeling is Compression](https://arxiv.org/abs/2309.10668)
  - Distinguish offline likelihood from prequential adaptive coding: predict once, pay the code length, then learn from the datum.

## 8. Diffusion for Control

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
- [ ] [2024] [Greg-rec] Diffusion-Based Reinforcement Learning via Q-Weighted Variational Policy Optimization - [paper](https://arxiv.org/abs/2405.16173)
  - Explicit $Q$-weighted variational objective discussed in the Universal Learner work.
- [ ] [2023] [SergeyLevine] DDPO: Training Diffusion Models with Reinforcement Learning - [paper](https://arxiv.org/abs/2305.13301), [project](https://rl-diffusion.github.io/)
  - Treat each reverse denoising transition as an RL action and optimize the terminal reward using policy gradients. Read before AWM, which interprets DDPO as noisy-target matching and replaces it with advantage-weighted pretraining loss.
- [ ] [2025] [Greg-rec] Advantage Weighted Matching: Aligning RL with Pretraining in Diffusion Models - [paper](https://arxiv.org/abs/2509.25050)
  - Recast diffusion policy gradients as noisy score / flow matching, then advantage-weight the original pretraining loss for a lower-variance RL objective. This is reward post-training of image generators rather than control, but directly complements the reweighted and $Q$-weighted objectives above.
- [ ] Revisit [[papers-generative-decision-making]] and [Compress and Control](https://arxiv.org/abs/1411.5326)
  - Conditional generation is not automatically causal control. Self-generated actions must be treated as interventions rather than evidence.

## 9. Language Diffusion and Iterative Reasoning

- [ ] [2025] Large Language Diffusion Models - [paper](https://arxiv.org/abs/2502.09992), [project](https://ml-gsai.github.io/LLaDA-demo/)
- [ ] [2026] Improved Large Language Diffusion Models - [paper](https://arxiv.org/abs/2606.25331), [code](https://github.com/ML-GSAI/LLaDA)
- [ ] [2026] [FloorEijkelboom] An Intuitive Introduction to Flow-Based Language Generation - [blog](https://flow-based-llms.github.io/)
  - Bridge from continuous flow matching to language: categorical endpoint distributions, cross-entropy-based Variational Flow Matching, simplex-valued denoisers, and flow-map distillation. Read before ELF.
- [ ] [2026] [Greg-rec] [KaimingHe] ELF: Embedded Language Flows - [paper](https://arxiv.org/abs/2605.10938)
  - Continuous-time flow matching in token-embedding space, remaining continuous until a final shared-weight projection to discrete tokens. Use as the continuous-language foil to masked discrete diffusion models.
- [ ] [2026] RePlaid: Continuous Diffusion Scales Competitively with Discrete Diffusion for Language - [paper](https://arxiv.org/abs/2605.18530), [project](https://research.nvidia.com/labs/genair/replaid/)
  - Likelihood-based continuous diffusion over learned token embeddings; compare its scaling laws, learned noise schedule, and embedding geometry with MDLM, Duo, and ELF.
- [ ] [2026] [Greg-rec] [SanderDieleman] Continuous Diffusion Language Models - [blog](https://sander.ai/2026/08/24/continuous-dlms.html)
  - Historical and technical synthesis of continuous language diffusion: embedding and unembedding strategies, objectives, noise schedules, self-conditioning, the shift toward discrete DLMs, and the 2026 revival represented by LangFlow, ELF, and RePlaid. Read after ELF and RePlaid.
- [ ] [2026] DiffusionGemma - [model card](https://ai.google.dev/gemma/docs/diffusiongemma/model_card), [project](https://deepmind.google/models/gemma/diffusiongemma/)
  - Read these as system-level capstones for masked / block diffusion, bidirectional attention, and parallel text decoding.
- [ ] [2026] [JunboZhao] Some Theoretical and Practical Thoughts on Diffusion Language Models - [blog](https://jzhao2024.github.io/notes/2026/08/08/diffusion-language-models.html)
  - Critical systems and scaling perspective on masked and block dLLMs: per-position marginals versus joint consistency, the parallel-commitment tradeoff, token and infrastructure efficiency, and competition with AR plus speculative decoding. Read after LLaDA and DiffusionGemma; its analysis does not directly cover continuous approaches such as ELF.
- [ ] [2024] IRED: Learning Iterative Reasoning through Energy Diffusion - [paper](https://arxiv.org/abs/2406.11179)
  - Explore diffusion as an iterative refinement process over solutions rather than only as a data generator.

---

## [2022] [BillPeebles,SainingXie] [DiT: Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)

- **Date**: 2026-08-12

---

- **Abstract**:
  > We explore a new class of diffusion models based on the Transformer architecture. We train latent diffusion models of images, replacing the commonly used U-Net backbone with a Transformer that operates on latent patches. We analyze the scalability of our Diffusion Transformers (DiTs) through the lens of forward-pass complexity, measured in Gflops. We find that DiTs with higher Gflops—through increased Transformer depth or width, or an increased number of input tokens—consistently have lower FID. In addition to possessing good scalability properties, our largest DiT-XL/2 models outperform all prior diffusion models on the class-conditional ImageNet $512\times512$ and $256\times256$ benchmarks, achieving a state-of-the-art FID of $2.27$ on the latter.
- **Question**: Can the U-Net backbone traditionally used by image diffusion models be replaced by a standard, scalable Transformer without changing the underlying diffusion method?
- **Answer**: Yes. By constructing and benchmarking the DiT design space within the Latent Diffusion Model (LDM) framework—where diffusion models are trained in a VAE's latent space—the paper successfully replaces the U-Net backbone with a homogeneous stack of Transformer blocks. It leaves the latent-diffusion process and DDPM training objective largely conventional. Its main empirical claim is a strong correlation between network complexity, measured in Gflops, and sample quality, measured by FID.
- **What is—and is not—new**:
  - DiT is primarily an **architecture paper**, not a new diffusion objective or sampler.
  - The original DiT is a DDPM-style model that predicts Gaussian noise and a reverse-process covariance. It is not a flow-matching model, even though the same backbone can later be trained to predict velocity or other targets.
  - It is a **latent diffusion model**: the Transformer denoises a learned, spatially compressed representation of the image rather than the image's raw pixels.
  - It is not an end-to-end pure Transformer. A frozen convolutional VAE encodes and decodes images; the Transformer implements the learned diffusion backbone between those two operations.
- **Historical context: why U-Nets came first**:
  - DDPM inherited its denoising backbone from the convolutional architectures used by earlier image models. Later systems improved the residual blocks, attention layers, normalization, and channel allocation without changing the basic U-Net shape.
  - In particular, Dhariwal and Nichol's ADM ablated two separate kinds of U-Net choice: **how to condition a block**, including adaptive normalization, and **how wide to make it**, including the number of convolutional channels at each resolution. Channel count controls capacity and compute; it is not information injected by adaptive normalization. DiT inherits the conditioning lesson while questioning whether the surrounding convolutional U-Net is necessary.
  - The U-Net is a natural fit for image denoising: convolutions supply locality and translation-equivariance, its resolution hierarchy cheaply builds global context, and skip connections preserve fine spatial detail while the bottleneck reasons at coarser scales.
  - Full self-attention over high-resolution pixels was historically expensive. Performing diffusion in a VAE latent reduces the spatial grid enough that a Transformer with global attention becomes practical.
  - The paper shows that the U-Net inductive bias is not crucial to diffusion-model performance and can be readily replaced by a standard design such as a Transformer. This positions diffusion to benefit from architecture unification: it can inherit training practices from other domains while retaining favorable properties such as scalability, robustness, and efficiency, and a standardized architecture opens possibilities for cross-domain research.
  - This does not establish that U-Nets are intrinsically bad. It shows that their hand-designed multiscale inductive bias is not necessary for strong image diffusion when sufficient compute, data, and a compressed latent representation are available.
- **Latent-diffusion pipeline**:
  - A pretrained VAE encoder maps an image $x$ to a spatial latent $z_0=E(x)$. For a $256\times256\times3$ image, the Stable Diffusion VAE used here produces a $32\times32\times4$ latent—an $8\times$ reduction along each spatial axis.
  - Diffusion corrupts and denoises $z$, not $x$:
    $$
    x \xrightarrow{E} z_0
    \xrightarrow{q(z_t\mid z_0)} z_t
    \xrightarrow{\mathrm{DiT}} (\hat\epsilon,\hat\Sigma)
    \xrightarrow{\text{reverse sampler}} \hat z_0
    \xrightarrow{D} \hat x.
    $$
  - "Latent" here means the learned compressed variable $z$ produced by the VAE. Merely projecting raw inputs to Transformer embeddings would not make a model a latent diffusion model; the state being diffused must itself be encoded and later decoded.
  - The VAE saves substantial diffusion compute, but it also imposes a reconstruction bottleneck: the generated image cannot contain information that the decoder cannot reconstruct from its latent representation.
- **DDPM formulation retained by DiT**:
  - Given a clean latent $z_0$, the forward process samples
    $$
    q(z_t\mid z_0)
    =\mathcal N\!\left(
      z_t;
      \sqrt{\bar\alpha_t}\,z_0,
      (1-\bar\alpha_t)I
    \right),
    $$
    or equivalently
    $$
    z_t
    =\sqrt{\bar\alpha_t}\,z_0
    +\sqrt{1-\bar\alpha_t}\,\epsilon,
    \qquad \epsilon\sim\mathcal N(0,I).
    $$
  - The reverse transition is parameterized as
    $$
    p_\theta(z_{t-1}\mid z_t)
    =\mathcal N\!\left(
      z_{t-1};
      \mu_\theta(z_t,t,c),
      \Sigma_\theta(z_t,t,c)
    \right),
    $$
    where $c$ is the ImageNet class condition.
  - As in DDPM/ADM, the principal regression target is the exact noise used to construct $z_t$:
    $$
    \mathcal L_{\mathrm{simple}}
    =\mathbb E_{z_0,t,\epsilon}
      \left[
        \left\|\epsilon-\epsilon_\theta(z_t,t,c)\right\|_2^2
      \right].
    $$
  - The model also learns the reverse covariance using the full variational objective. This is why the final layer emits $2C$ channels: $C$ for $\hat\epsilon$ and $C$ for the diagonal covariance parameterization.
- **The DiT architecture (Figures 3 and 4)**:
  1. **Patchify the noisy latent.** For $z_t\in\mathbb R^{I\times I\times C}$, split it into non-overlapping $p\times p$ patches and linearly project each flattened patch to width $d$. This gives
     $$
     T=\left(\frac{I}{p}\right)^2
     $$
     tokens, each in $\mathbb R^d$.
  2. **Add position information.** Fixed sine-cosine positional embeddings retain each patch's location on the latent grid.
  3. **Run ordinary Transformer blocks.** DiTs adhere to Vision Transformer (ViT) best practices, which had been shown to scale more effectively for visual recognition than traditional convolutional networks such as ResNets. Each block consists of multi-head self-attention and a GELU feed-forward network, with residual connections and layer normalization. The original DiT does not use SwiGLU.
  4. **Condition every block.** Embeddings of the diffusion timestep $t$ and class label $c$ modulate the computation. The paper's preferred mechanism is adaLN-Zero.
  5. **Decode and unpatchify.** A final adaptive layer normalization and linear projection turn each token into $p^2(2C)$ values; rearranging the patches recovers two $I\times I\times C$ spatial outputs for noise and covariance.
- **Patch size controls token count and compute (Figure 4)**:
  - At $256\times256$ image resolution, the VAE latent has $I=32$. Thus $p\in\{2,4,8\}$ produces respectively $256$, $64$, or $16$ tokens.
  - Halving $p$ quadruples the sequence length. It barely changes the parameter count but substantially increases Gflops, because the same Transformer parameters are evaluated over more tokens and attention also couples more token pairs.
  - Model names expose this choice: **DiT-XL/2** means the extra-large Transformer with $2\times2$ latent patches. The `/2` is not a diffusion-step count or image downsampling factor.
- **Four conditioning mechanisms (Figure 3)**:
  - **In-context conditioning** appends the timestep and class embeddings as two additional tokens. It is simple and has negligible extra compute.
  - **Cross-attention conditioning** treats timestep and class as a length-two conditioning sequence and inserts a cross-attention layer after self-attention. It adds roughly $15\%$ compute in the paper's comparison.
  - **Adaptive layer normalization (adaLN)** turns the summed timestep and class embeddings into feature-wise scale and shift parameters for layer normalization. The same global condition modulates every spatial token without adding attention layers; the modulation is feature-wise rather than spatially varying.
  - **adaLN-Zero** additionally predicts feature-wise gates for the attention and MLP residual branches, then initializes the conditioning projection to zero. It gave the best FID throughout the conditioning ablation in Figure 5 while adding negligible compute.
- **From GroupNorm to adaLN-Zero**:
  - **GroupNorm** is the normalization used inside the ADM U-Net. It normalizes groups of channels and then applies learned scale and shift parameters. Those parameters are normally fixed after training: they do not change with the diffusion timestep or class.
  - **Adaptive GroupNorm** makes the scale and shift depend on the condition $c$, which combines the timestep and class embeddings:
    $$
    \operatorname{AdaGN}(h,c)
    =(1+s(c))\odot\operatorname{GN}(h)+b(c).
    $$
    Thus different timesteps and classes can emphasize, suppress, or shift different channels. The same adjustment is broadcast across all spatial positions. ADM retains GroupNorm's ordinary learned affine parameters as well; the conditional scale and shift are additional modifications.
  - The U-Net's channel count $C$ only determines how many scale and shift values are needed. Choosing $C$ is a separate decision about model width and compute; it is not itself a way of injecting the condition.
  - **Adaptive LayerNorm (adaLN)** applies the same idea to Transformer features. DiT uses LayerNorm without its own affine parameters, then obtains the scale and shift from the timestep and class condition:
    $$
    \operatorname{AdaLN}(x,c)
    =(1+s(c))\odot\operatorname{LN}(x)+b(c).
    $$
    The $1+s$ form means that $s=b=0$ gives ordinary LayerNorm. Without the $1$, zero initialization would multiply the normalized features by zero.
  - **adaLN-Zero** additionally predicts a gate $g(c)$ for each residual branch:
    $$
    x'
    =x+g(c)\odot F\!\left(\operatorname{AdaLN}(x,c)\right),
    $$
    where $F$ is either attention or the MLP. Each branch has its own scale, shift, and gate.
  - DiT initializes the conditioning projection so that $s=b=g=0$. Because $g=0$, the attention and MLP updates initially contribute nothing, and each block starts as the identity $x'=x$. The gate can then learn to open and admit the branch's output. Only this conditioning projection is zero-initialized; the attention and MLP weights are initialized normally.
- **Classifier-free guidance**:
  - During training, some class labels are replaced with a learned null label so that one model learns both conditional and unconditional denoising.
  - At sampling time their noise estimates are combined as
    $$
    \hat\epsilon_{\mathrm{CFG}}
    =\epsilon_\theta(z_t,t,\varnothing)
    +s\left(
      \epsilon_\theta(z_t,t,c)
      -\epsilon_\theta(z_t,t,\varnothing)
    \right).
    $$
    Increasing $s$ generally improves condition adherence and perceptual fidelity at the cost of sample diversity and recall.
- **Model family (Table 1)**:

  | Model | Blocks $N$ | Width $d$ | Heads |
  | --- | ---: | ---: | ---: |
  | DiT-S | 12 | 384 | 6 |
  | DiT-B | 12 | 768 | 12 |
  | DiT-L | 24 | 1024 | 16 |
  | DiT-XL | 28 | 1152 | 16 |

  - Each size is trained with patch sizes $p\in\{2,4,8\}$. This independently varies network capacity and token count, allowing the paper to ask whether compute predicts quality better than parameter count alone.
- **Training recipe**:
  - Class-conditional ImageNet at $256\times256$ and $512\times512$, using the frozen Stable Diffusion VAE with an $8\times$ spatial downsampling factor.
  - The diffusion process follows ADM: $1000$ steps, a linear variance schedule from $10^{-4}$ to $2\times10^{-2}$, noise prediction, and learned reverse covariance.
  - AdamW, learning rate $10^{-4}$, no weight decay, batch size $256$, horizontal flips as the only data augmentation, and an exponential moving average of model weights with decay $0.9999$.
  - The paper deliberately avoids elaborate Transformer regularization and optimization recipes—no learning-rate warmup—so that model size, token count, and compute remain the central variables.
- **Main empirical findings**:
  - **Conditioning matters (Figure 5):** adaLN-Zero is substantially better than in-context, cross-attention, and ordinary adaLN conditioning; the paper attributes the gap chiefly to its zero initialization.
  - **All three routes to more compute help (Figure 6):** increasing depth, increasing width, or decreasing patch size improves FID over the course of training.
  - **Forward-pass compute predicts quality (Figure 8):** model Gflops has a strong negative correlation with FID, reported as $-0.93$. Parameter count alone misses the patch-size effect because more tokens increase compute without materially increasing parameters. Simply scaling the LDM to the high-capacity, $118.6$-Gflop DiT-XL/2 backbone yields the paper's headline FID of $2.27$ on class-conditional ImageNet $256\times256$.
  - **Large models are more training-compute-efficient (Figure 9):** extending the training of a small DiT eventually costs more total compute than reaching the same or better FID with a larger model sooner.
  - **Extra sampler steps do not rescue a small backbone (Figure 10):** spending inference compute on more denoising steps for DiT-L/2 does not close the quality gap to DiT-XL/2. Model compute and sampling compute are not interchangeable.
  - **Visual quality scales consistently (Figure 7):** holding the initial noise and class label fixed makes the effects of model/patch scaling directly visible rather than confounding them with different samples.
- **Headline results, read historically**:
  - On class-conditional ImageNet $256\times256$, guided DiT-XL/2 reported FID $2.27$ at classifier-free guidance scale $1.5$, compared with the prior latent-diffusion result of $3.60$ in Table 2. The unguided DiT-XL/2 result was FID $9.62$.
  - On ImageNet $512\times512$, guided DiT-XL/2 reported FID $3.04$ in Table 3.
  - These are claims relative to the models and evaluations available in 2022–2023, not current state-of-the-art claims. The more durable contribution is the architecture and scaling evidence, not the absolute benchmark ranking.
- **How to read the figures and tables**:
  - **Figure 3:** the full latent-DiT pipeline and the four ways of injecting timestep/class conditioning.
  - **Figure 4:** patchification and why the patch-size suffix controls sequence length and compute.
  - **Figure 5:** the case for adaLN-Zero.
  - **Figure 6:** separate scaling effects of depth, width, and token count.
  - **Figure 7:** qualitative comparison with noise and class held fixed.
  - **Figure 8:** the paper's central Gflops-versus-FID result.
  - **Figure 9:** why larger models can be more efficient for a fixed training-compute budget.
  - **Figure 10:** why more sampling steps are not a substitute for a more capable denoiser.
  - **Table 1:** the S/B/L/XL configurations and patch-size compute.
  - **Tables 2 and 3:** headline ImageNet benchmarks and the fidelity/diversity effect of guidance.
  - **Appendix A and Table 4:** implementation details, including the timestep embedding, adaLN projections, initialization, optimizer, and diffusion hyperparameters.
- **Connections to later DiT-style implementations**:
  - The reusable idea is the conditional Transformer backbone, not a commitment to $\epsilon$-prediction. Later systems retain patch/token processing and adaLN-Zero while changing the generated modality, attention pattern, MLP, positional encoding, corruption path, or regression target.
  - A trajectory model can therefore be recognizably DiT-derived while using rotary embeddings, SwiGLU, packed observation/action/reward tokens, structured attention, and a flow/velocity target. Those are architectural descendants, not the exact ImageNet DiT described in this paper.
  - When reading code, separate three choices that are often bundled together: **representation** (pixels, VAE latents, or trajectory tokens), **backbone** (U-Net or Transformer), and **training target** ($x$, $\epsilon$, $v$, score, or flow velocity). DiT's central intervention is only the backbone choice.
- **Core takeaway**: DiT made diffusion architecture look more like modern language-model architecture: a uniform Transformer whose quality improves as model and token compute increase. Its latent VAE, DDPM corruption process, noise target, covariance learning, and sampler remain conventional; the decisive replacement is U-Net $\rightarrow$ conditioned Transformer.

## [2026] [Greg-rec] [KaimingHe] [ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938)

- **Date**: 2026-08-20

---

- **Abstract**:
  > Diffusion and flow-based models have become the de facto approaches for generating continuous data, e.g., in domains such as images and videos. Their success has attracted growing interest in applying them to language modeling. Unlike their image-domain counterparts, today's leading diffusion language models (DLMs) primarily operate over discrete tokens. In this paper, we show that continuous DLMs can be made effective with minimal adaptation to the discrete domain.
  >
  > We propose Embedded Language Flows (ELF), a class of diffusion models in continuous embedding space based on continuous-time Flow Matching. Unlike existing DLMs, ELF predominantly stays within the continuous embedding space until the final time step, where it maps to discrete tokens using a shared-weight network. This formulation makes it straightforward to adapt established techniques from image-domain diffusion models, e.g., classifier-free guidance (CFG).
  >
  > Experiments show that ELF substantially outperforms leading discrete and continuous DLMs, achieving better generation quality with fewer sampling steps. These results suggest that ELF offers a promising path toward effective continuous DLMs.

- **Continuous versus discrete DLMs**: DLMs are commonly formulated in one of two ways: continuous or discrete. Continuous DLMs map discrete tokens into continuous representations and perform denoising in the resulting continuous space. Discrete DLMs, in contrast, operate directly in token space and formulate a probabilistic diffusion model over discrete random variables.
  - More precisely, “continuous” versus “discrete” describes the **random state being corrupted and generated**, not merely whether the neural network has continuous hidden activations. Every Transformer already represents tokens with continuous vectors internally.
  - A continuous DLM such as ELF embeds the sequence as $x\in\mathbb R^{L\times d}$, corrupts and denoises these continuous vectors, and converts the final vectors to token probabilities only at the end.
  - A discrete DLM such as [D3PM](https://arxiv.org/abs/2107.03006), [MDLM](https://arxiv.org/abs/2406.07524), or [Duo](https://arxiv.org/abs/2506.10892) corrupts the categorical token identities themselves. At an intermediate time, each position still contains a vocabulary symbol—perhaps the original token, `[MASK]`, or a random replacement token.
- **The motivating question**: Recent advances in discrete DLMs have substantially improved their generation quality and sampling efficiency. By contrast, continuous DLMs have seen relatively limited progress. It remains an open question whether the current performance gap is due to the inherently discrete nature of language modeling or to underexplored algorithmic design choices for continuous models.
- **ELF is continuous in two senses**:
  - It operates in **continuous embedding space**, directly denoising continuous representations throughout the flowing process and considering discretization only at the final time step.
  - It is formulated in **continuous time**, following Flow Matching, which allows the velocity field to be defined through a time derivative. This formulation lets ELF benefit directly from advances in Flow Matching.
- **Constructing the embedding space**: Following Latent Diffusion Models, ELF constructs its continuous embedding space by applying an encoder model to the input discrete tokens. The encoder can be pretrained, jointly trained, or even frozen with random weights. The paper uses pretrained bidirectional T5 contextual embeddings by default:
  $$
  (s_1,\ldots,s_L)\in\mathcal V^L
  \xrightarrow{E}
  x\in\mathbb R^{L\times512}.
  $$
  The default encoder is a frozen pretrained T5-small encoder. Its output is **contextual and token-position-aligned**: there is one vector per sequence position, but the vector for a token depends on the entire input sentence rather than being a fixed vocabulary lookup. The encoder supplies clean continuous targets during training. It is not needed for unconditional inference, because generation starts directly from Gaussian noise with the same shape as $x$; conditional generation still uses it to encode the source sequence.
- **“Latent” does not necessarily mean “compressed”**: A latent is simply an unobserved internal representation. Latent diffusion typically chooses a deliberately compressed bottleneck to reduce denoising compute. For example, image latent diffusion maps
  $$
  256\times256\times3
  \xrightarrow{E}
  32\times32\times4,
  $$
  performs diffusion in the smaller representation, and relies on a separately trained decoder to reconstruct pixels. A language latent model can similarly reduce sequence length, feature width, or both:
  $$
  (s_1,\ldots,s_L)
  \xrightarrow{E}
  (h_1,\ldots,h_M),
  \qquad M<L,
  $$
  after which a decoder must expand the generated latent sequence back into $L$ tokens.
- **ELF's latent is high-dimensional and token-position-aligned**: ELF retains one contextual embedding per token position rather than shortening the sequence into a compressed latent code:
  $$
  s_i\longleftrightarrow x_i\in\mathbb R^{512}.
  $$
  The model does use a $128$-dimensional internal channel bottleneck before projecting to the Transformer width, but this is not the kind of sequence compression used by latent-diffusion autoencoders: every token position remains represented and the model projects back to one output vector per position. This makes the continuous flow more expensive than operating on a short latent sequence, but preserves a direct positional interface to vocabulary prediction. The trade-off is therefore
  $$
  \begin{aligned}
  \text{compressed latent diffusion:}&\quad
  \text{cheaper denoising}+\text{separate decoder}+\text{information bottleneck},\\
  \text{ELF:}&\quad
  \text{higher-dimensional denoising}+\text{token alignment}+\text{no separate decoder}.
  \end{aligned}
  $$
- **No separate decoder**: A conventional language latent-diffusion pipeline would be
  $$
  s
  \xrightarrow{E}
  x
  \xrightarrow{\text{diffusion or flow}}
  \hat x
  \xrightarrow{D_\phi}
  \hat s,
  $$
  where $D_\phi$ is another learned model that must run during generation. Unlike these methods, ELF does not require such a separately parameterized decoder. It repurposes the final Flow Matching time as a continuous-to-discrete decoding step: the same Transformer performs denoising at $t<1$ and decoding at $t=1$. See **Figure 2**.
  - During denoising,
    $$
    x_\theta
    =
    \operatorname{net}_\theta(z_t,t,\mathrm{denoise}).
    $$
  - During decoder training, ELF corrupts a clean embedding to create a nontrivial endpoint input $\tilde z$, then uses
    $$
    h
    =
    \operatorname{net}_\theta(\tilde z,1,\mathrm{decode}),
    \qquad
    \operatorname{logits}=Wh,
    $$
    with token-level cross-entropy against $s$. Corruption prevents the final branch from learning only an identity map and prepares it to correct the imperfect endpoint produced by numerical sampling.
  - At inference,
    $$
    z_0\sim\mathcal N(0,I)
    \xrightarrow[\text{multiple ODE steps}]{\operatorname{net}_\theta\text{ in denoise mode}}
    z_1
    \xrightarrow{\operatorname{net}_\theta(\cdot,1,\mathrm{decode})}
    h
    \xrightarrow{W}
    \text{tokens}.
    $$
  “No decoder” therefore does **not** mean that no decoding computation occurs. ELF still performs one final shared-Transformer call and applies a learned unembedding matrix $W$; the claim is that it needs no second generative network with separate parameters. The encoder is training-only, whereas the shared Transformer and unembedding layer are used at inference.
- **A deliberately minimal continuous DLM**: ELF builds on prior continuous DLMs but aims for a minimalist design focused on the interface between continuous and discrete spaces.
  - In contrast to pioneering continuous DLMs and many later methods that employ a per-step discretization loss such as cross-entropy, ELF performs denoising in continuous embedding space at nearly all steps, offering maximal flexibility for the flow dynamics.
  - Unlike latent-diffusion methods that operate in a compressed latent space and rely on a separate decoder, ELF directly operates in a high-dimensional latent space and requires no extra decoder. Here “latent space” means the sequence of contextual token embeddings, not a compressed representation with its own decoder.
  - These two choices are connected: because the continuous state remains high-dimensional and token-aligned, the denoiser can share its weights with the final token decoder. A heavily compressed, non-token-aligned latent would generally require a more expressive separate decoder.
- **Headline empirical result**: Following the evaluation protocols of prior work, ELF outperforms leading discrete DLMs and existing continuous DLMs (**Figure 1**). It achieves better generation quality with fewer sampling steps than discrete models such as MDLM and Duo and concurrent continuous models such as FLM and LangFlow. It reports this performance using $10\times$ fewer training tokens and without distillation. The authors take these results as evidence that continuous DLMs can be highly competitive while requiring only minimal treatment of discretization.
- **Flow-matching recap for ELF**: Let $x$ be the clean sequence of contextual token embeddings and let $\epsilon\sim\mathcal N(0,I)$ have the same shape. ELF uses the straight conditional path
  $$
  z_t=t x+(1-t)\epsilon,
  \qquad 0\leq t\leq1,
  $$
  so $t=0$ is Gaussian noise and $t=1$ is the clean embedding. Holding the sampled pair $(x,\epsilon)$ fixed and differentiating gives the conditional velocity
  $$
  \frac{dz_t}{dt}=x-\epsilon.
  $$
  A Transformer is trained on these easy conditional targets. As in ordinary conditional flow matching, regression over many examples makes its output approximate the marginal vector field—the average conditional velocity appropriate at the current $(z_t,t)$.
- **ELF predicts the clean embedding**: The network outputs $x_\theta(z_t,t)$ rather than velocity directly. Since
  $$
  z_t=t x+(1-t)\epsilon
  \quad\Longrightarrow\quad
  x-\epsilon=\frac{x-z_t}{1-t},
  $$
  ELF converts the clean-embedding prediction to
  $$
  v_\theta(z_t,t)=\frac{x_\theta(z_t,t)-z_t}{1-t}.
  $$
  Consequently, velocity matching and clean-embedding regression are the same objective with a time-dependent weight:
  $$
  \left\|v_\theta-(x-\epsilon)\right\|^2
  =
  \frac{1}{(1-t)^2}\left\|x_\theta-x\right\|^2.
  $$
  Generation starts from Gaussian $z_0$ and numerically integrates the learned ODE $dz_t/dt=v_\theta(z_t,t)$ toward $t=1$, using Euler's method or another ODE solver.
  - Thus the **direct network output** is the current clean-endpoint estimate $x_\theta$, while the **quantity used by the ODE solver** is the derived velocity $v_\theta$. Predicting $x$ does not remove iterative sampling: under MSE, $x_\theta(z_t,t)$ approximates $\mathbb E[x\mid z_t]$, so a highly noisy input does not identify a unique final sample. The model recomputes the estimate and velocity after every step.
  - $x$-prediction also aligns the two shared-weight tasks: both continuous denoising and final token decoding ask the network to recover a clean embedding. The paper reports that direct $v$-prediction works poorly when the same weights must also perform final discretization.
- **Final token objective**: If $W$ is the vocabulary unembedding matrix, the decoder-mode prediction for position $i$ has logits
  $$
  \operatorname{logits}(s_i)=W x_{\theta,i}.
  $$
  A softmax and categorical sample or argmax then produce the token. Training selects continuous denoising with the MSE objective for $80\%$ of examples and endpoint decoding with token-level cross-entropy for $20\%$.
- **Training algorithm**: **Algorithm 1** is easiest to read as two supervised tasks that share the same Transformer parameters. The first learns the continuous flow; the second teaches the endpoint to spell its continuous prediction as tokens.

  ```python
  clean = encoder(tokens)  # frozen T5; training only

  if random() < 0.8:  # continuous denoising branch
      t = sample_denoising_time()
      noise = 2 * normal_like(clean)
      state = t * clean + (1 - t) * noise

      clean_hat = model(state, t=t, mode="denoise")
      velocity = clean - noise
      velocity_hat = (clean_hat - state) / (1 - t)
      loss = mse(velocity_hat, velocity)
  else:  # endpoint discretization branch
      endpoint_input = corrupt_each_token(clean)
      decoded = model(endpoint_input, t=1, mode="decode")
      logits = unembedding(decoded)
      loss = cross_entropy(logits, tokens)
  ```

  The `if` statement is conceptual. The implementation places both kinds of example in a batch, masks the inapplicable operations and losses, and provides a learned **mode token** so the shared network knows which task it is performing. The two branches are not two sequential stages: each training example contributes to one of the two losses.
- **Endpoint corruption and time sampling**:
  - Before corruption, the clean T5 embeddings are normalized with the mean and standard deviation estimated on OpenWebText.
  - For denoising, ELF samples $t' \sim \mathcal N(-1.5,0.8^2)$ and uses $t=\operatorname{sigmoid}(t')$. Gaussian noise is scaled by $2$. This logit-normal schedule emphasizes the difficult, noisy region near $t=0$ more than uniform sampling does.
  - The decoder cannot simply be trained on exactly clean $x$ because the numerical sampler will arrive at an imperfect approximation to $x$. ELF therefore samples a separate corruption strength $p_i$ for each token position,
    $$
    p_i=\operatorname{sigmoid}(p'_i),
    \qquad
    p'_i\sim\mathcal N(0.8,0.8^2),
    $$
    and constructs
    $$
    \tilde z_i=p_i x_i+(1-p_i)\epsilon_i.
    $$
    The decoder noise scale is $5$ on OpenWebText and $1$ on the conditional tasks. Varying $p_i$ within a sequence makes the endpoint branch repair different amounts of local corruption instead of learning an identity map.
- **Inference algorithm**: **Algorithm 2** uses ordinary Euler integration for the clearest version of ELF. The model repeatedly predicts the clean endpoint, converts that prediction into the velocity at the current state, and takes one ODE step. Only after the final continuous step does it produce vocabulary logits.

  ```python
  state = normal(embedding_shape)

  for t, t_next in adjacent_pairs(time_grid):
      clean_hat = model(state, t=t, mode="denoise")
      velocity = (clean_hat - state) / (1 - t)
      state = state + (t_next - t) * velocity

  decoded = model(state, t=1, mode="decode")
  logits = unembedding(decoded)
  tokens = logits.argmax(dim=-1)
  ```

  `clean_hat` is an estimate of the final clean embedding, not the next state. The next state is obtained by following its implied velocity for only the current time interval. This is why $x$-prediction still requires iterative inference.
- **Self-conditioning**: A first denoising prediction $\hat x'$ is fed back as context for a second prediction:
  $$
  \hat x
  =
  \operatorname{net}_\theta(z_t\mid \operatorname{stopgrad}(\hat x'),t).
  $$
  Concretely, ELF concatenates $[z_t,\hat x']$ along the channel dimension and linearly projects the result back to the original embedding dimension. During training, half of the examples use $\hat x'$ and half replace it with zeros. During sampling, the previous timestep's clean prediction becomes the next step's self-condition, so self-conditioning does not require an additional inference pass at every step. Decoder-mode examples always use a zero self-condition.
- **Classifier-free guidance from self-conditioning**: In unconditional language generation there is no prompt or class label, so ELF treats the intermediate self-conditioning prediction as the condition $c$. Standard CFG would form
  $$
  v_{\mathrm{cfg}}(z_t\mid c,\omega)
  =
  \omega v(z_t\mid c)
  +(1-\omega)v(z_t\mid\varnothing),
  $$
  where $\omega$ is the guidance scale and $\varnothing$ is the zero condition. For $\omega>1$, this extrapolates away from the unconditional prediction and makes samples more strongly follow the self-condition. As usual, stronger guidance tends to improve apparent quality while reducing diversity.
  - Ordinary CFG needs conditional and unconditional network calls at every sampling step. ELF instead uses **training-time CFG**: it conditions the model on $\omega$ and trains one call to predict the already-guided output.
  - For a self-conditioned example, the velocity target is
    $$
    v_{\mathrm{target}}
    =
    v
    +
    \left(1-\frac{1}{\omega}\right)
    \left(v_{\mathrm{sc}}-v_{\mathrm{no\text{-}sc}}\right),
    $$
    with `stopgrad` applied to the target. When $\omega=1$, this reduces to the ordinary flow-matching target $v=x-\epsilon$.
  - Training samples $\omega\in[0.5,5]$ from a distribution biased toward smaller values. Time, guidance scale, and denoise/decode mode are represented by four prepended control tokens apiece. This in-context conditioning is slightly better in the ablation and reduces ELF-B from $148$M parameters with adaLN-Zero to $105$M.
  - The full denoising branch in **Algorithm 3** can be reduced to the following skeleton:

    ```python
    no_sc_input = project(concat(state, zeros_like(state)))
    x_no_sc = model(no_sc_input, t=t, cfg_scale=w, mode="denoise")

    sc_input = project(concat(state, stopgrad(x_no_sc)))
    x_sc = model(sc_input, t=t, cfg_scale=w, mode="denoise")

    v_no_sc = (x_no_sc - state) / (1 - t)
    v_sc = (x_sc - state) / (1 - t)
    guided_target = velocity + (1 - 1 / w) * (v_sc - v_no_sc)

    use_sc = random_per_example() < 0.5
    prediction = where(use_sc, v_sc, v_no_sc)
    target = where(use_sc, guided_target, velocity)
    loss = mse(prediction, stopgrad(target))
    ```

    The first pass supplies both the zero-self-conditioned prediction and the detached context for the second pass. The mask also keeps ordinary, unguided examples in training; otherwise the network would not learn what its null condition means.
- **Conditional text-to-text generation**: For translation or summarization, the clean contextual embeddings of the source sequence are prepended to the target and are never corrupted. Full self-attention lets the noisy target positions condition on this clean prefix. The CFG condition then includes both the prefix and the self-conditioning prediction; the unconditional counterpart zeros these conditions. During training, ELF zeros the source embeddings with probability $0.1$ so the same model learns the conditional and unconditional cases required for CFG. The same generative Transformer and final decoding mechanism therefore extend to sequence-to-sequence tasks; the frozen T5 encoder is additionally run at inference to embed the source.
- **SDE-inspired sampler**: ELF's baseline sampler follows the deterministic flow ODE. The paper also uses a practical stochastic approximation that re-injects noise at each step and asks the denoiser to evaluate a slightly noisier state:

  ```python
  def sde_like_step(state, t, dt, gamma):
      noise = normal_like(state)
      alpha = 1 - gamma * dt
      t_back = alpha * t
      noisy_state = alpha * state + (1 - alpha) * noise

      clean_hat = model(noisy_state, t=t_back, mode="denoise")
      velocity = (clean_hat - state) / (1 - t)
      return state + dt * velocity
  ```

  This is deliberately called **SDE-inspired**, not an exact numerical integration of the corresponding SDE. $\gamma$ controls the re-injected noise; $\gamma=0$ recovers the deterministic Euler update. The authors hypothesize that moderate stochasticity can correct early mistakes instead of letting a deterministic trajectory amplify them. Empirically it helps most when the step budget is small, usually lowering generative perplexity while slightly lowering entropy.
- **Discrete diffusion uses categorical Markov dynamics**: An ordinary ODE moves a point infinitesimally in $\mathbb R^d$, and a diffusion SDE adds infinitesimal Gaussian increments. Neither operation is directly defined for token identities: there is no token “slightly between” `cat` and `dog`. A discrete DLM instead specifies probabilities of jumping between vocabulary symbols.
  - In discrete time, a transition matrix $Q_t$ defines a categorical Markov chain:
    $$
    q(s_t\mid s_{t-1})=\operatorname{Cat}(Q_t s_{t-1}),
    \qquad
    q(s_t\mid s_0)=\operatorname{Cat}(\bar Q_t s_0),
    $$
    where tokens are represented as one-hot vectors and $\bar Q_t=Q_tQ_{t-1}\cdots Q_1$.
  - In continuous time, the analogue is a **continuous-time Markov chain** (CTMC) with a rate matrix $R_t$. With $p_t$ written as a row vector, its distribution follows the master equation
    $$
    \frac{dp_t}{dt}=p_tR_t.
    $$
    An individual sample remains at one token for a while and then jumps to another. This is not a Brownian SDE, even though the literature still calls the overall construction “diffusion.”
- **State space and time are separate choices**: “Discrete diffusion” means that the state being generated is discrete; it does not imply that the time variable must also be discrete. Likewise, the original DDPM uses continuous-valued states but a finite set of timesteps. Continuous-time models are still evaluated with finitely many numerical steps on a computer.

  | Model | State space | Mathematical time | Representative dynamics |
  | --- | --- | --- | --- |
  | Original DDPM | continuous, $X_t\in\mathbb R^d$ | discrete, $t\in\{0,\ldots,T\}$ | $q(X_t\mid X_{t-1})=\mathcal N(\sqrt{1-\beta_t}\,X_{t-1},\beta_t I)$ |
  | Score SDE | continuous, $X_t\in\mathbb R^d$ | continuous, $t\in[0,1]$ | $dX_t=f_t(X_t)\,dt+g_t\,dW_t$ |
  | D3PM | discrete, $S_t\in\mathcal V$ | usually discrete | $q(S_t\mid S_{t-1})=\operatorname{Cat}(Q_tS_{t-1})$ |
  | MDLM | discrete, $S_t\in\mathcal V\cup\{\mathrm{MASK}\}$ | continuous | $q(S_t\mid S_0)=\alpha_t\delta_{S_0}+(1-\alpha_t)\delta_{\mathrm{MASK}}$ |
  | Duo / uniform-state diffusion | discrete, $S_t\in\mathcal V$ | continuous | $q(S_t\mid S_0)=\alpha_t\delta_{S_0}+(1-\alpha_t)\operatorname{Uniform}(\mathcal V)$ |
  | ELF | continuous, $Z_t\in\mathbb R^{L\times d}$ | continuous | $dZ_t/dt=v_\theta(Z_t,t)$ |

  MDLM and Duo therefore have continuous schedules and continuously evolving categorical probabilities, but a realized sample is always a token and changes through jumps. ELF has both continuous time and a continuous state, so its realized trajectory has an ordinary velocity $dZ_t/dt$.
- **What replaces velocity for discrete tokens**: A continuous point can move a small distance in the direction $u_t(x)$, but there is no meaningful direction or intermediate value between two token identities such as `cat` and `dog`. A discrete model instead specifies a jump rate $R_t(i,j)$: the probability per unit time that a sample currently at token $i$ switches to token $j$. All the rates together determine how the categorical probabilities change:
  $$
  \frac{dp_t}{dt}=p_tR_t.
  $$
  [Discrete Flow Matching](https://arxiv.org/abs/2407.15595) therefore does not learn an ODE velocity over token IDs. It learns the transition rates needed to move probability from one token to another, and generation samples the resulting token jumps. “Flow” here refers to probability moving among categories, not to individual tokens moving continuously through space.
- **Masked or absorbing diffusion (MDLM)**: For one clean token $S_0$, the forward marginal is
  $$
  q(S_t\mid S_0)
  =
  \alpha_t\,\delta_{S_0}
  +(1-\alpha_t)\,\delta_{\mathrm{MASK}},
  $$
  where:
  - $q(S_t\mid S_0)$ is the distribution of the token at time $t$, conditioned on the original clean token;
  - $\delta_a$ is a point mass that assigns probability $1$ to symbol $a$ and $0$ to every other symbol;
  - $\alpha_t\in[0,1]$ is the **clean-token survival probability**, or noise schedule. It is a decreasing function of time, with $\alpha_0\approx1$ and $\alpha_1\approx0$.

  Thus the mixture says exactly:
  $$
  S_t=
  \begin{cases}
  S_0, & \text{with probability }\alpha_t,\\
  \mathrm{MASK}, & \text{with probability }1-\alpha_t.
  \end{cases}
  $$
  For example, if $\alpha_t=0.7$, then a token remains $S_0$ with probability $0.7$ and is `[MASK]` with probability $0.3$. At the endpoints,
  $$
  \alpha_0=1
  \;\Longrightarrow\;
  q(S_0\mid S_0)=\delta_{S_0},
  \qquad
  \alpha_1=0
  \;\Longrightarrow\;
  q(S_1\mid S_0)=\delta_{\mathrm{MASK}}.
  $$
  This $\alpha_t$ is not a Gaussian variance or standard deviation. It is a categorical mixture weight, although it plays the analogous role of measuring how much clean signal remains. The displayed equation is a **marginal** from the clean token directly to time $t$, rather than a one-step transition. For two times $s<t$, an unmasked token survives from $s$ to $t$ with conditional probability $\alpha_t/\alpha_s$; if it has already become `[MASK]`, it remains masked. For sequences, the forward corruption applies this mechanism independently at each token position.
- **Useful taxonomy**:

  | Model family | Corrupted state | Forward dynamics | Learned object | Sampling path |
  | --- | --- | --- | --- | --- |
  | D3PM | categorical tokens | discrete Markov chain | reverse categorical transitions / clean-token prediction | token-to-token steps |
  | MDLM | tokens and `[MASK]` | absorbing-mask Markov process | clean-token distribution at masked positions | progressive unmasking |
  | Duo | categorical tokens | uniform-state Markov process | reverse categorical transitions | repeated token revision |
  | SEDD | categorical tokens | CTMC | probability ratios / reverse rates | continuous-time token jumps |
  | Discrete Flow Matching | categorical tokens | discrete probability path | posterior or transition rates | token jumps |
  | ELF | continuous contextual embeddings | Euclidean interpolation | clean embedding, converted to velocity | ODE through embedding space, then one token decode |

- **Architecture and default training recipe**:
  - The clean targets come from a frozen $35$M-parameter T5-small encoder with embedding dimension $512$. A learned linear bottleneck maps each embedding through $128$ channels and then into the Transformer's hidden width. The bottleneck changes channel dimension, not sequence length.
  - The denoiser-decoder is a DiT-style bidirectional Transformer with SwiGLU, RMSNorm, RoPE, and query-key normalization. Unlike a causal LM, it uses full attention because all token positions are refined together.

    | Model | Layers | Hidden width | Heads | Parameters | OWT epochs |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | ELF-B | 12 | 768 | 12 | 105M | 5 |
    | ELF-M | 24 | 1056 | 16 | 342M | 4 |
    | ELF-L | 32 | 1280 | 16 | 652M | 3 |

  - The OpenWebText configuration uses sequences of $1024$ tokens, global batch size $512$, Muon with learning rate $0.002$, no weight decay, a constant learning rate after $0.5$ warmup epochs, and EMA decay $0.9999$. The reported setup uses $64$ TPU v5p chips and takes about $1.5$ hours per epoch.
  - The training allocation is $80\%$ denoising and $20\%$ endpoint decoding. Self-conditioning is used with probability $0.5$. The default SDE noise-reinjection scale is $\gamma=1$, though the system comparison tunes it by sampling budget.
- **Evaluation protocol**:
  - Unconditional ELF is trained on OpenWebText, about $9$B tokens, and evaluated on $1{,}000$ generated sequences. **Generative perplexity** is the perplexity that a separate pretrained GPT-2 Large assigns to those samples. It is a fluency proxy supplied by an external evaluator, not ELF's own likelihood.
  - The paper pairs perplexity with per-sample unigram entropy. For a generated sequence $S$ of length $L$,
    $$
    \hat p_S(w)
    =
    \frac{\#\{i:s_i=w\}}{L},
    \qquad
    H(S)
    =
    -\sum_w \hat p_S(w)\log \hat p_S(w),
    $$
    and it averages $H(S)$ across the $1{,}000$ samples. Higher entropy means that an individual sample uses a broader token vocabulary. It is mainly a repetition/collapse check: it ignores word order, semantics, and diversity across different samples.
  - Most ablation plots therefore show a **Gen. PPL–entropy frontier**. Lower perplexity and higher entropy are preferred; pushing CFG harder often improves one while worsening the other. The paper treats entropy below $5$ as typically repetitive or degenerate and generative perplexity above $300$ as typically meaningless or ungrammatical.
  - The authors do not report validation perplexity from ELF itself because exact likelihood evaluation for a flow can require additional likelihood-specific training. For conditional generation they instead use BLEU on WMT14 German-to-English and ROUGE-1/2/L on XSum.
- **Unconditional-generation results**: **Figure 7** compares the $105$M-parameter ELF-B with roughly $170$M-parameter discrete models MDLM and Duo and continuous models FLM and LangFlow, all on OpenWebText.
  - With the SDE-inspired sampler and self-conditioning CFG scale $3$, ELF-B reaches generative perplexity $24.08\pm0.16$ and entropy $5.15\pm0.002$ at $32$ sampling steps using $\gamma=1.5$. The corresponding $8$- and $16$-step perplexities are $67.32$ and $33.66$, using the stronger $\gamma=2$ correction.
  - Undistilled ELF is also better in the paper's few-step comparison than the distilled MDLM+SDTT, Duo+DCD, and FMLM baselines.
  - The reported effective training budget is $45.2$B tokens—five passes over OpenWebText—versus more than $500$B for the compared baselines. This accounting measures training examples consumed by the diffusion models; it does **not** include the prior cost of pretraining the frozen T5 encoder.
- **Conditional-generation results**: ELF-B uses $64$-step ODE sampling, self-conditioning CFG scale $1$, and source-condition CFG scale $2$. In the paper's matched-scale comparison it reports the best result on both tasks:

  | Model | WMT14 De-En BLEU | XSum R1 | XSum R2 | XSum R-L |
  | --- | ---: | ---: | ---: | ---: |
  | Autoregressive baseline | 25.2 | 30.5 | 10.2 | 24.4 |
  | MDLM | 18.4 | 33.4 | 11.6 | 25.8 |
  | Duo | 21.3 | 31.4 | 10.1 | 25.0 |
  | ELF-B | **26.4** | **36.0** | **12.2** | **27.8** |

  WMT14 uses $64$ clean source positions and $64$ generated target positions; XSum uses up to $1024$ source positions and $64$ generated target positions. The table is encouraging evidence that the method is not restricted to unconditional text, though it covers small, conventional sequence-to-sequence benchmarks rather than modern instruction-following evaluation.
- **Main-method ablations**:
  - **Guidance — Figure 4**: increasing self-conditioning CFG scale lowers generative perplexity but also lowers entropy. CFG is therefore a quality–diversity control, not an unqualified improvement.
  - **Embedding choice — Figure 5a**: frozen pretrained contextual T5 embeddings give the best frontier. A contextual encoder trained from scratch on OpenWebText is close but slightly worse. Among noncontextual choices, frozen pretrained T5 token lookups beat frozen Gaussian vectors; jointly learned token embeddings perform worst, which the authors attribute to the difficulty of moving the target representation while simultaneously learning its denoiser.
    - Thus ELF's best model is **not an end-to-end, from-scratch language model** in the GPT sense. It bootstraps its clean representation from a pretrained frozen encoder. The ablation shows that the ELF mechanism does not logically require this choice, but it materially helps.
    - “Token-position-aligned” should not be confused with “token-identity-aligned.” Each position has a vector that can be decoded to a token, but contextual T5 means the same token can have different clean vectors in different sentences.
  - **Shared versus separate decoder — Figure 5b**: a separately trained T5-shaped decoder and the shared-weight design trace similar quality–diversity curves, but weight sharing reaches lower perplexity and removes a separate training stage and inference module. This supports the minimalist design, though it does not show that separate decoding is fundamentally incapable.
  - **ODE versus SDE-inspired sampling — Figure 5c**: stochastic noise reinjection gives substantially lower perplexity in the few-step regime. Its advantage shrinks with more steps.
  - **Model scaling — Figure 6**: ELF-B, M, and L improve the frontier with scale. At matched entropy, larger models achieve lower perplexity; at matched perplexity, they retain higher entropy.
- **Additional ablations**:
  - **Prediction target — Figure 11**: $x$-prediction is stable for T5-small/base/large embedding dimensions $512/768/1024$. Direct $v$-prediction is competitive at $512$ but degrades at $768$ and $1024$; $\epsilon$-prediction collapses at every tested width. Besides the empirical result, $x$-prediction is the only target that naturally makes denoising and endpoint decoding ask the shared network for the same semantic object.
  - **Bottleneck — Figure 12**: $32$ channels can obtain low perplexity but often collapses into the low-entropy region; $512$ preserves more entropy but has substantially worse perplexity. The default $128$-channel bottleneck gives the best balance.
  - **Denoise/decode allocation — Figure 13**: too few denoising examples hurts the frontier, particularly with the SDE sampler. The tested optimum is the default $0.8/0.2$ denoising/decoding split.
  - **Conditioning mechanism — Figure 14**: prepended in-context control tokens are slightly better than adaLN-Zero while reducing ELF-B's parameter count from $148$M to $105$M.
  - **Optimizer — Figure 15**: tuned Muon outperforms tuned AdamW at matched entropy, especially with SDE-inspired sampling, but both optimizers remain better than the paper's DLM baselines. The headline result is therefore not solely an optimizer artifact.
  - **Time grid — Figure 16a**: a logit-normal inference grid beats a uniform grid at every tested step count, especially with few steps. It both matches the training distribution and places finer intervals in the noisy early portion of the path.
  - **Noise reinjection — Figure 16b**: moderate increases in $\gamma$ lower perplexity while slightly lowering entropy; $\gamma=0$ is the ODE sampler and $\gamma=1$ is the default.
  - **Conditional CFG — Figure 17**: increasing source guidance from $1$ to $2$ improves translation and summarization, while stronger guidance begins to hurt. The default source-condition scale is therefore $2$.
- **Progressive distillation extension (ELF+PD)**: Base ELF is strong at $8$–$32$ steps but deteriorates below that range. Appendix B compresses a fixed $64$-step teacher into a few-step student. If $K$ teacher substeps move $z_t$ to $z_r$, the displacement is converted back into the clean-prediction parameterization:
  $$
  \tilde x
  =
  z_t
  +
  \frac{1-t}{r-t}(z_r-z_t),
  \qquad
  \mathcal L_{\mathrm{distill}}
  =
  \mathbb E\left[
  \left\|x_\theta(z_t,t)-\tilde x\right\|^2
  \right].
  $$
  This follows by asking which clean prediction would make one Euler step over $[t,r]$ reproduce the teacher displacement. The distillation loss replaces the denoising MSE, while the shared endpoint decoder keeps its original cross-entropy loss.

  | Round | Student steps | Teacher substeps per student step |
  | --- | ---: | ---: |
  | 1 | 16 | 4 |
  | 2 | 8 | 8 |
  | 3 | 4 | 16 |
  | 4 | 2 | 32 |
  | 5 | 1 | 64 |

  Each round trains for one epoch; later students initialize from the previous round. **Figure 9** reports that the final ELF+PD student beats the distilled MDLM+SDTT, Duo+DCD, and FMLM baselines from $1$ through $32$ steps. Its one-step generative perplexity is $136.10$ at entropy $5.26$, while $8$ steps reach $23.18$ at entropy $5.07$. The curriculum matters: early-round students collapse when evaluated far below their trained step budget, while later rounds progressively improve $1$–$4$-step generation. ELF+PD consumes about $90$B effective tokens, still excluding the frozen encoder's pretraining cost.
- **What is actually novel**: None of contextual embeddings, flow matching, $x$-prediction, self-conditioning, CFG, shared weights, or endpoint cross-entropy is individually new. ELF's contribution is the unusually clean combination:
  $$
  \text{frozen contextual embedding space}
  +
  \text{unrestricted continuous flow}
  +
  \text{no intermediate token projection or CE}
  +
  \text{final-only shared-weight discretization}.
  $$
  Previous continuous DLMs commonly tie intermediate states back to the vocabulary through per-step cross-entropy or projection. Latent DLMs can keep the path continuous but generally need a separate latent-to-text decoder. ELF occupies the missing design point: keep the whole generative path continuous, then make the denoiser itself perform the one discrete endpoint operation.
- **Limitations and interpretation**:
  - The default representation comes from a frozen pretrained T5 encoder, so the strongest result does not yet demonstrate joint end-to-end learning of the language representation and generative flow. The poor learnable-embedding ablation makes this an important open problem rather than a bookkeeping detail.
  - The claimed $10\times$ data efficiency excludes T5 pretraining and compares effective token counts across systems with different training pipelines. It shows that the ELF stage is data-efficient, not that the complete system was trained from scratch on one tenth the total compute or data.
  - Generative perplexity under GPT-2 Large and unigram entropy are convenient historical DLM metrics but weak measures of semantics, factuality, instruction following, long-range coherence, and cross-sample mode collapse.
  - Undistilled ELF still needs iterative sampling. One-step generation becomes competitive only after a five-round progressive-distillation curriculum.
  - The experiments establish a strong small-model result on OpenWebText and two sequence-to-sequence benchmarks. They do not yet establish GPT-scale pretraining behavior, long-context scaling, or parity with modern autoregressive foundation models.
- **Core takeaway**: ELF's main question is not whether continuous vectors can represent language—all Transformers already use them—but whether the **generative state can remain continuous for the entire iterative trajectory**. The paper's answer is yes: with a good frozen contextual representation, $x$-prediction, a shared endpoint decoder, self-conditioning, guidance, and a carefully chosen sampler, continuous language flows can outperform the paper's discrete and continuous DLM baselines. The most consequential unresolved question is whether the representation and flow can be learned together at scale without relying on a separately pretrained encoder.

## [2026] [Nando] [Diffusion and Flow Matching Tutorial](https://love4all.ai/files/diffusion-and-flow-matching-tutorial.pdf)

- **Date**: 2026-08-03
- **Notebook**: <https://love4all.ai/files/diffusion-and-flow-matching-tutorial.ipynb>

---

### Section 1: Introduction

_**TL;DR:** Diffusion turns generation into a supervised denoising problem: use a fixed process to corrupt real samples, learn to reverse that corruption, then generate by starting from Gaussian noise and repeatedly applying the learned reverse process. The tutorial takes score matching as its route to the training objective and previews flow matching as an ODE-based alternative view._

**Notation used in these notes**

The PDF calls clean data $x$, a noisy state $z_t$, and the clean prediction $x_\theta$. These notes rename them to $z$, $x_t$, and $D_\theta$, respectively, to match the roles used in the [MIT course notes](course-mit-diffusion-2026.md):

| Symbol | Meaning |
| --- | --- |
| $Z\sim p_{\mathrm{data}}$, $z$ | Random clean data point and one realized value |
| $X_t$, $x_t$ | Random state at noise level $t$ and one realized value |
| $p_t(x_t\mid z)$ and $p_t(x_t)$ | Conditional and marginal probability paths |
| $s_\theta(x_t,t)$ | Learned marginal score |
| $D_\theta(x_t,t)$ | Learned denoiser / clean-data prediction |
| $\epsilon_\theta(x_t,t)$ | Learned noise prediction |
| $v_\theta(x_t,t)$ | Diffusion $v$-parameterization, not a velocity field |
| $u_t(x)$ | Velocity field, when flow matching is introduced |

Two differences from the MIT course remain because they are built into Nando's formulas and code:

1. **Time runs in the opposite direction.** Nando uses $t=0$ for clean data and $t=1$ for noise; the MIT flow-matching notes use $t=0$ for noise and $t=1$ for data. Nando's sampler therefore runs from large $t$ to small $t$.
2. **The Gaussian noise amplitude is $\sigma_t$.** The MIT notes call it $\beta_t$, but standard DDPM notation also uses $\beta_t$ for a one-step noise variance. Keeping $\sigma_t$ avoids that collision.

The network can be conditioned on either $t$ or the equivalent log-SNR $\lambda_t$. The mathematical exposition writes $t$ for simplicity; the tutorial's implementation passes $\lambda_t$.

- **Problem setup**:
  - Represent any modality—images, video, audio, proteins, or molecules—as a vector $z$ drawn from an unknown real distribution $p_{\mathrm{data}}(z)$.
  - Learn a model distribution $p_\theta(z)$ from which new samples can be generated: $z \sim p_\theta(z)$.
  - The tutorial's slogan, "match imagination to reality," is the high-level goal. It is not yet the computable objective; the later score-matching derivation supplies that objective without requiring direct access to $p_{\mathrm{data}}(z)$.
  - > While there are other derivations using variational methods, here we will simply use a fundamental learning principle: match imagination to reality. That is, what the model imagines, predicts, or generates, must match the real data. This is the principle used to train LLMs too, but while LLMs typically use the Maximum Likelihood principle, here we will use the Score Matching principle.
- **The two processes**:
  - The fixed forward process $q$ starts at clean data $x_0=z$ and progressively adds Gaussian noise until $x_T \sim \mathcal{N}(0,I)$. Its schedule is designed rather than learned.
  - The learned backward process $p_\theta$ starts at $x_T$ and denoises step by step until it produces $x_0$. Training learns this reversal; generation runs it from right to left.
  - The forward process therefore manufactures the training signal: clean examples and their noisy versions. No separately labelled targets are needed.
  - > Figure 1: The two halves of a diffusion model. The forward process $q$ (top, blue) takes a clean datapoint $z=x_0$ and gradually corrupts it into pure Gaussian noise $x_T\sim\mathcal{N}(0,I)$ by adding a small amount of noise at each step. This direction is typically hand-designed: each transition is a Gaussian whose mean and variance are fixed by a schedule, with no learned parameters. The backward process $p_\theta$ (bottom, red, dashed) goes the other way and is what the neural network learns: starting from pure noise it denoises step-by-step until it produces a sample. Sampling at inference time is just running the bottom row from right to left to produce new images, speech, videos or molecules.
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

_**TL;DR:** The ideal target is $p_\theta=p_{\mathrm{data}}$, but neither the real density $p_{\mathrm{data}}(z)$ nor the normalized model density $p_\theta(z)$ is generally available pointwise. The probability-space squared error is therefore a statement of intent rather than the loss that will actually be optimized._

- The data distribution $p_{\mathrm{data}}(z)$ denotes the unknown mechanism that produced the training examples. A generative model $p_\theta(z)$ approximates it and supports sampling:

$$
z \sim p_\theta(z).
$$

- > The data (images, proteins, videos, songs) will be represented with the generic vector $z$. The real data is assumed to come from an unknown distribution $p_{\mathrm{data}}(z)$. Since we don’t have access to this distribution, we will try to approximate it using a model distribution $p_\theta(z)$, with parameters $\theta$. After learning the model distribution, also known as the generative model, we will be able to generate new data from it. Mathematically, the process of generation is represented as follows: $z \sim p_\theta(z)$.

- The tutorial first writes distribution matching as

$$
\mathcal{L}(\theta)
=
\mathbb{E}_{z\sim p_{\mathrm{data}}}
\left[
\frac{1}{2}
\left\|p_\theta(z)-p_{\mathrm{data}}(z)\right\|_2^2
\right].
\tag{1}
$$

- > We want our model to assign the same probability as the world to all data configurations $z$. That is, we want to minimize the difference between these two distributions on expectation over all the possible realizations of the data.

- This loss would attain its ideal value when the model assigns the same density as the world to real-data configurations. It is not directly computable:
  - We have samples from $p_{\mathrm{data}}$, not numerical values of $p_{\mathrm{data}}(z)$.
  - A flexible model may provide an unnormalized energy for $z$ while leaving its global normalizing constant intractable.
  - The next section changes _what is matched_: instead of matching density values, it matches their log-density gradients.
  - > Matching what the model imagines (generates) to the data generated by the world seems like a natural goal for learning. However, this is hard because we cannot calculate probabilities for models directly (so we’ll have to use autoregression or, as we explain here, diffusion score matching). The reason we cannot calculate the probabilities has to do with the normalizing constant,

#### 2.2 Score matching

_**TL;DR:** Write the model as a normalized energy model. Its partition function depends on $\theta$ but not on $z$, so taking $\nabla_z\log p_\theta(z)$ removes it. Score matching then compares the model and data log-density gradients, although the unknown data score remains to be handled by denoising score matching in §2.3._

**Energy-based representation**

- > The model distribution $p_\theta(z)$ can be expressed in a very general normalized exponential form:

$$
p_\theta(z)
=
\frac{1}{\mathcal{Z}(\theta)}e^{-E_\theta(z)}.
$$

- $E_\theta(z)$ is the energy. At fixed $\theta$, lower-energy configurations have greater probability.
- $\mathcal{Z}(\theta)$ is the partition function:

$$
\mathcal{Z}(\theta)
=
\int_{\mathbb R^d} e^{-E_\theta(z)}\,dz.
$$

- The quantifiers matter: normalization means

$$
\forall\theta,\qquad
\int_{\mathbb R^d}p_\theta(z)\,dz=1.
$$

In words, for each fixed parameter setting $\theta$, integrate over every possible $z$. We do not integrate over $\theta$.

- > In this representation, $\mathcal{Z}$ is known as the normalizing constant or partition function. It ensures that over the whole set of values that the data can take, the model probability sums to 1 for all values of $\theta$.

- For a discrete sample space, replace the integral with

$$
\mathcal{Z}(\theta)=\sum_{z\in\mathcal{X}}e^{-E_\theta(z)}.
$$

- > The denominator sums over all possible images in the universe so that $p_\theta(z)$ can be interpreted as a probability:
- > This partition function is typically intractable because the sum is simply too large. It belongs to a complexity class known as sharp P, which in short means bloody hard if not impossible.

**Why maximum likelihood is hard for an unrestricted energy model**

$$
\log p_\theta(z)
=
-E_\theta(z)-\log \mathcal{Z}(\theta).
$$

- For fixed $\theta$, minimizing $E_\theta(z)$ over $z$ is equivalent to maximizing $p_\theta(z)$ over $z$.
- When learning $\theta$, however, the partition function cannot be ignored because it also changes with $\theta$:

$$
\nabla_\theta\log p_\theta(z)
=
-\nabla_\theta E_\theta(z)
-\nabla_\theta\log \mathcal{Z}(\theta).
$$

Thus, simply lowering the energy of training examples is not by itself maximum-likelihood learning; the normalizer accounts for what happens to all other configurations.

- > The quantity in the exponent is known as the energy. Physicists often prefer to use the terminology of minimizing energy, but clearly this is equivalent to maximising the model probability. Maximising the probability of the data by modifying the model parameters is known as maximum likelihood.

**The autoregressive LLM route**

- An LLM avoids one global normalization over all complete sequences by applying the probability chain rule:

$$
p_\theta(z_{1:L})
=
\prod_{i=1}^{L}p_\theta(z_i\mid z_{<i}).
$$

- Each conditional is normalized only over the vocabulary $\mathcal{V}$:

$$
\sum_{v\in\mathcal{V}}
p_\theta(z_i=v\mid z_{<i})
=1.
$$

- Consequently, next-token maximum likelihood is tractable:

$$
-\log p_\theta(z_{1:L})
=
-\sum_{i=1}^{L}\log p_\theta(z_i\mid z_{<i}).
$$

- > For the practical applications we care about, we cannot do this sum. A decade ago we weren’t too optimistic, but we have learned since then that it is actually possible to approximate this well for text, images, video, audio and other natural signals. One possible solution is to break the data $z$ into small blocks and process each block auto-regressively (this is basically what LLMs do).

**The score-matching route**

- > An alternative is to do what we are about to learn to do in this document. If we can’t do the sum, let’s get rid of the sum! We can do this by taking the log of the model probability and then computing its gradient with respect to the data

$$
\begin{aligned}
p_\theta(z)
&=\frac{1}{\mathcal{Z}(\theta)}e^{-E_\theta(z)},\\
\log p_\theta(z)
&=-\log \mathcal{Z}(\theta)-E_\theta(z),\\
\nabla_z\log p_\theta(z)
&=-\nabla_z E_\theta(z).
\end{aligned}
$$

- The last equality holds because $\mathcal{Z}(\theta)$ has no dependence on $z$:

$$
\nabla_z\log \mathcal{Z}(\theta)=0.
$$

- The **score** of a distribution is its log-density gradient with respect to the data:

$$
s_\theta(z)
=
\nabla_z\log p_\theta(z).
$$

- It is a vector field over data space. Locally, it points in the direction in which the model's log probability increases most quickly.
- Score matching asks the model vector field to equal the data vector field:

$$
\mathcal{L}(\theta)
=
\mathbb{E}_{z\sim p_{\mathrm{data}}}
\left[
\left\|
\nabla_z\log p_\theta(z)
-
\nabla_z\log p_{\mathrm{data}}(z)
\right\|_2^2
\right].
\tag{2}
$$

- > We will now reframe learning as matching the gradient of the model probability and the gradient of the distribution of the data. Intuitively, we want the rate of change in the modelled energy to match the rate of change of the real energy. This is known as score matching:

- This removes the model partition function, but it does not yet give a trainable objective because $\nabla_z\log p_{\mathrm{data}}(z)$ is unknown. Section 2.3 introduces the Gaussian corruption process needed before §2.4 derives a computable objective.
  - > Getting rid of $\mathcal{Z}$ is not enough. We still don’t have an expression for the derivative of the data distribution: $\nabla_z\log p_{\mathrm{data}}(z)$. The model derivative $\nabla_z\log p_\theta(z)$, known as the score function, can be easily calculated using backpropagation.

#### 2.3 Denoising Score Matching

_**TL;DR:** Replace the singular, unknown data distribution with a family of smooth noisy distributions. At a randomly selected noise level $t$, mix a clean sample $z$ with known Gaussian noise $\epsilon$ to obtain $x_t$. Because the corruption kernel is known and the injected noise is recorded, the next section can turn score learning into supervised regression._

**Gaussian corruption at one noise level**

- Center a Gaussian corruption kernel on a scaled version of each clean sample:

$$
x_t
\sim
p_t(x_t\mid z)
=
\mathcal{N}\!\left(x_t\mid \alpha_t z,\sigma_t^2 I\right).
$$

- > Assume instead that we can place a Gaussian concentrated on each data point $z$ and then draw a sample $x_t$. This Gaussian will have two scalar hyperparameters taking values between 0 and 1. A hyperparameter $\alpha_t$ will be used to scale the data, e.g. scale an image $z$. A second hyperparameter $\sigma_t^2$ will determine the Gaussian variance.
- > When $\alpha_t = 0$ the Gaussian will have mean zero, and when $\alpha_t = 1$ the Gaussian will have mean $z$. Later we will show how we can parameterise $\alpha_t$ so that by modifying the subindex $t$, $\alpha_t$ will vary from 1 to 0, and $\sigma_t^2$ in turn will vary from 0 to 1.

- Use the Gaussian reparameterization

$$
\epsilon\sim\mathcal{N}(0,I),
\qquad
x_t=\alpha_t z+\sigma_t\epsilon.
\tag{3}
$$

  Here $\alpha_t$ controls how much clean signal remains and $\sigma_t$ controls the noise standard deviation. The variance is $\sigma_t^2$.

- > In other words $x_t$ is a bit like the image $z$ and a bit like Gaussian noise $\epsilon$.

- In the notation established above, $z$ is clean data and $x_t$ is its noisy version at time $t$.
- This is a direct marginal corruption $p_t(x_t\mid z)$: during training, one can choose any $t$ and construct $x_t$ in a single operation. There is no need to simulate $x_1,\ldots,x_{t-1}$ first.
- After mixing clean samples over the data distribution, the noisy marginal is

$$
p_t(x_t)
=
\int_{\mathbb R^d}p_t(x_t\mid z)p_{\mathrm{data}}(z)\,dz.
$$

  Even when $p_{\mathrm{data}}$ itself is only available through samples, Gaussian convolution makes $p_t$ smoother and gives the training procedure a known conditional corruption mechanism.

**Multiple noise scales**

- Let $t$ move between a clean endpoint and a Gaussian-noise endpoint:

$$
\begin{aligned}
t\approx0:&\qquad \alpha_t\approx1,\quad \sigma_t\approx0,\quad x_t\approx z,\\
t\approx1:&\qquad \alpha_t\approx0,\quad \sigma_t\approx1,\quad x_t\approx\epsilon.
\end{aligned}
$$

- Intermediate values of $t$ create intermediate corruption levels. Training across randomly sampled $t$ teaches one time-conditioned network how to denoise everywhere from nearly clean data to nearly pure noise.
- > We have introduced the index $t$ because we will allow for sampling at multiple scales. At the very noisy scale, when $\alpha \approx 0$ and $\sigma \approx 1$, $x_t$ will be basically Gaussian noise. At the other no noise extreme, when $\alpha \approx 1$ and $\sigma \approx 0$, $x_t \approx z$. We will choose a schedule to obtain samples between these two extremes.
- When the data have approximately unit covariance and are independent of the injected unit Gaussian noise,

$$
\begin{aligned}
\operatorname{Cov}(x_t)
&=\alpha_t^2\operatorname{Cov}(z)
+\sigma_t^2\operatorname{Cov}(\epsilon)\\
&\approx(\alpha_t^2+\sigma_t^2)I.
\end{aligned}
$$

  The common variance-preserving constraint

$$
\alpha_t^2+\sigma_t^2=1
$$

  therefore keeps $\operatorname{Cov}(x_t)\approx I$ while trading signal variance for noise variance. The squares appear because $\alpha_t$ and $\sigma_t$ multiply the samples, so their contributions to variance are $\alpha_t^2$ and $\sigma_t^2$.

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
  - $\lambda_t\gg0$: signal dominates; $x_t$ is nearly clean.
  - $\lambda_t=0$: signal and noise powers are equal.
  - $\lambda_t\ll0$: noise dominates; $x_t$ is nearly Gaussian.
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

  These expressions satisfy both requirements:

$$
\alpha_t^2+\sigma_t^2
=\frac{e^{\lambda_t}}{1+e^{\lambda_t}}
+\frac{1}{1+e^{\lambda_t}}
=1,
\qquad
\frac{\alpha_t^2}{\sigma_t^2}=e^{\lambda_t}.
$$

  For example, $\lambda_t=\log 9$ means a signal-to-noise variance ratio of $9:1$, and hence

$$
\alpha_t^2=\frac{9}{10},
\qquad
\sigma_t^2=\frac{1}{10}.
$$

- The corruption equation $x_t=\alpha_t z+\sigma_t\epsilon$ uses the signal and noise **standard-deviation coefficients**, not their squares. Taking the nonnegative roots yields the implementation:

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

- A variance-preserving pair $(\alpha_t,\sigma_t)$ lies on the unit circle. The cosine schedule moves through its first quadrant by setting the angle $u=\pi t/2$ and choosing

$$
\alpha_t=\cos\!\left(\frac{\pi t}{2}\right),
\qquad
\sigma_t=\sin\!\left(\frac{\pi t}{2}\right).
$$

  Thus $t=0$ is the all-signal endpoint, $t=1$ is the all-noise endpoint, and the Pythagorean identity automatically gives $\alpha_t^2+\sigma_t^2=1$. At $t=1/2$, $\alpha_t=\sigma_t=1/\sqrt2$, so signal and noise each contribute half the variance. The corresponding log-SNR is

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

- The implementation also supports the resolution-dependent shifted-cosine schedule from Simple Diffusion. Let

$$
r=\frac{d_{\mathrm{noise}}}{d_{\mathrm{image}}},
$$

  where $d_{\mathrm{noise}}$ is the reference spatial resolution associated with the base schedule and $d_{\mathrm{image}}$ is the target image resolution. The shifted log-SNR is

$$
\lambda_t^{\mathrm{shifted}}
=
\lambda_t
+2\log r.
$$

  Adding a constant in log-SNR space multiplies the SNR:

$$
\operatorname{SNR}_t^{\mathrm{shifted}}
=e^{\lambda_t^{\mathrm{shifted}}}
=e^{\lambda_t}e^{2\log r}
=r^2\operatorname{SNR}_t.
$$

  Thus, if the target resolution doubles while the reference stays fixed, $r=1/2$ and the SNR at every nominal $t$ is divided by four. Increasing image resolution therefore makes the shift negative and applies more corruption at the same $t$. Equations (6) and (7) then convert the shifted $\lambda_t$ back to $\alpha_t^2$ and $\sigma_t^2$ while preserving their sum of one. The factor of two appears because SNR is a ratio of squared amplitude coefficients.

**Minimal implementation for §2.3**

- This follows the paper's two §2.3 code blocks while trimming their long argument and return-value docstrings.
- `_t_lo` and `_t_hi` implement the finite-endpoint derivation above; they are angles despite their names.
- `add_noise` returns the injected noise alongside $x_t$ because that known $\epsilon$ becomes the regression target in the following objective.

```python
import math

import torch
from torch import Tensor


def add_noise(
    clean_sample: Tensor,
    alpha_t: Tensor,
    sigma_t: Tensor,
) -> tuple[Tensor, Tensor]:
    """Forward diffusion step: x_t = alpha_t * z + sigma_t * noise."""
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

#### 2.4 The Score-Matching Objective

_**TL;DR:** At each noise level, the network should predict the score of the noisy marginal $p_t(x_t)$. That marginal score is an intractable posterior average, but denoising score matching lets us regress against the tractable Gaussian conditional score $\nabla_{x_t}\log p_t(x_t\mid z)$. The two squared-error objectives differ only by a term independent of the network parameters._

**The time-conditioned score**

At time $t$, the model should approximate the marginal score:

$$
s_\theta(x_t,t)
\approx
\nabla_{x_t}\log p_t(x_t).
\tag{8}
$$

The network receives the current noisy point $x_t$ and its time $t$. The implementation passes the equivalent log-SNR $\lambda_t$ because it directly identifies the physical corruption level.

The target is the score of the noisy **marginal**

$$
p_t(x_t)
=
\int p_t(x_t\mid z)p_{\mathrm{data}}(z)\,dz,
$$

  not the score of the clean empirical distribution. Gaussian corruption makes this marginal smooth, but its density and score still require integrating over the unknown data distribution.

**From the marginal score to a conditional Gaussian target**

Start with score matching at a fixed noise level:

$$
\begin{aligned}
\mathcal L_t^{\mathrm{SM}}(\theta)
&=
\mathbb E_{x_t\sim p_t}
\left[
\left\|
s_\theta(x_t,t)
-\nabla_{x_t}\log p_t(x_t)
\right\|_2^2
\right]\\
&=
\mathbb E_{x_t\sim p_t}
\left[
\left\|
s_\theta(x_t,t)
-
\frac{
\nabla_{x_t}\int p_t(x_t\mid z)p_{\mathrm{data}}(z)\,dz
}{p_t(x_t)}
\right\|_2^2
\right]\\
&=
\mathbb E_{x_t\sim p_t}
\left[
\left\|
s_\theta(x_t,t)
-
\frac{
\int p_t(x_t\mid z)
\nabla_{x_t}\log p_t(x_t\mid z)
p_{\mathrm{data}}(z)\,dz
}{p_t(x_t)}
\right\|_2^2
\right]\\
&=
\mathbb E_{x_t\sim p_t}
\left[
\left\|
s_\theta(x_t,t)
-
\mathbb E_{z\sim p_t(\cdot\mid x_t)}
\left[
\nabla_{x_t}\log p_t(x_t\mid z)
\right]
\right\|_2^2
\right]\\
&=
\mathbb E_{\substack{z\sim p_{\mathrm{data}}\\x_t\sim p_t(\cdot\mid z)}}
\left[
\left\|
s_\theta(x_t,t)
-
\nabla_{x_t}\log p_t(x_t\mid z)
\right\|_2^2
\right]
+\mathrm{const}.
\end{aligned}
$$

The missing steps are:

1. Use $\nabla_{x_t}\log p_t(x_t)=\nabla_{x_t}p_t(x_t)/p_t(x_t)$ and substitute $p_t(x_t)=\int p_t(x_t\mid z)p_{\mathrm{data}}(z)\,dz$.
2. Move the gradient through the integral and use

   $$
   \nabla_{x_t}p_t(x_t\mid z)
   =
   p_t(x_t\mid z)\nabla_{x_t}\log p_t(x_t\mid z).
   $$

3. Apply Bayes' rule:

   $$
   \frac{p_t(x_t\mid z)p_{\mathrm{data}}(z)}{p_t(x_t)}
   =p_t(z\mid x_t).
   $$

   Thus the marginal score is a posterior average of conditional scores:

   $$
   \boxed{
   \nabla_{x_t}\log p_t(x_t)
   =
   \mathbb E_{z\sim p_t(\cdot\mid x_t)}
   \left[
   \nabla_{x_t}\log p_t(x_t\mid z)
   \right].
   }
   $$

4. Replace the inaccessible posterior-average target with a sampled conditional target. Filling in the omitted square expansion gives

   $$
   \begin{aligned}
   &\mathbb E_{x_t\sim p_t}
   \left[
   \left\|
   s_\theta(x_t,t)
   -
   \mathbb E_{z\sim p_t(\cdot\mid x_t)}
   [\nabla_{x_t}\log p_t(x_t\mid z)]
   \right\|_2^2
   \right]\\
   &=
   \mathbb E_{x_t\sim p_t}
   \mathbb E_{z\sim p_t(\cdot\mid x_t)}
   \left[
   \left\|
   s_\theta(x_t,t)
   -\nabla_{x_t}\log p_t(x_t\mid z)
   \right\|_2^2
   \right]
   +\mathrm{const},
   \end{aligned}
   $$

   where

   $$
   \begin{aligned}
   \mathrm{const}
   =\mathbb E_{x_t\sim p_t}\Bigg[
   &\left\|
   \mathbb E_{z\sim p_t(\cdot\mid x_t)}
   [\nabla_{x_t}\log p_t(x_t\mid z)]
   \right\|_2^2\\
   &-
   \mathbb E_{z\sim p_t(\cdot\mid x_t)}
   \left[
   \left\|
   \nabla_{x_t}\log p_t(x_t\mid z)
   \right\|_2^2
   \right]
   \Bigg].
   \end{aligned}
   $$

   This is the negative conditional variance of the conditional-score target. It depends on the data and corruption process but contains no $s_\theta$, so it does not affect $\nabla_\theta\mathcal L$ or the minimizer. Finally,

   $$
   p_t(x_t)p_t(z\mid x_t)
   =p_t(x_t\mid z)p_{\mathrm{data}}(z)
   $$

   turns the nested expectation into the joint expectation used in the last line.

**The conditional score is known analytically**

For

$$
p_t(x_t\mid z)
=
\mathcal N(x_t\mid\alpha_t z,\sigma_t^2I),
$$

the log-density terms that depend on $x_t$ are

$$
\log p_t(x_t\mid z)
=
\mathrm{const}
-\frac{1}{2\sigma_t^2}
\|x_t-\alpha_t z\|_2^2.
$$

Therefore,

$$
\boxed{
\nabla_{x_t}\log p_t(x_t\mid z)
=
-\frac{x_t-\alpha_t z}{\sigma_t^2}
=
-\frac{\epsilon}{\sigma_t}
}.
$$

The final sign must be negative: the score of a Gaussian points back toward its mean $\alpha_t z$. The tutorial's explanatory line below the derivation drops this minus sign, while the loss itself correctly uses the corresponding plus sign:

$$
\mathcal L_t^{\mathrm{DSM}}(\theta)
=
\mathbb E_{p_{\mathrm{data}}(z)}
\mathbb E_{x_t\sim\mathcal N(\alpha_t z,\sigma_t^2I)}
\left[
\left\|
s_\theta(x_t,t)
+\frac{x_t-\alpha_t z}{\sigma_t^2}
\right\|_2^2
\right]
+\mathrm{const}.
$$

Although the marginal score is intractable, training needs only a sampled clean $z$, a sampled $x_t\mid z$, and this known Gaussian conditional score. Under squared error, the best network prediction given $(x_t,t)$ is their posterior average, which is exactly the marginal score.

**Combining noise levels**

For discrete noise levels $t=1,\ldots,T$, combine the per-level objectives using positive weights:

$$
\mathcal L(\theta)
=
\sum_{t=1}^{T}w_t\mathcal L_t^{\mathrm{DSM}}(\theta),
\qquad w_t>0.
\tag{9}
$$

The weights decide how much different SNR regions matter numerically. In code, this sum is usually implemented by sampling a noise level $t$ for each training example and applying the corresponding importance or loss weight.

#### 2.5 Reparameterizations of the Loss

_**TL;DR:** Score prediction, noise prediction, clean-data prediction, and $v$-prediction encode the same Gaussian-path information but scale errors differently. The common $\epsilon$ objective turns score matching into ordinary supervised MSE. The $v$-parameterization retains that target while avoiding the unstable division by $\alpha_t$ that appears when reconstructing clean data from a high-noise $\epsilon$-prediction._

**From score prediction to noise prediction**

The known injected noise satisfies

$$
\epsilon
=
\frac{x_t-\alpha_t z}{\sigma_t}.
$$

Since the conditional score target is $-\epsilon/\sigma_t$, parameterize the model by

$$
\boxed{
\epsilon_\theta(x_t,t)
=
-\sigma_t s_\theta(x_t,t)
}.
$$

Then the per-noise-level score loss becomes

$$
\mathcal L_t^{\mathrm{DSM}}(\theta)
=
\mathbb E
\left[
\frac{1}{\sigma_t^2}
\left\|
\epsilon_\theta(x_t,t)-\epsilon
\right\|_2^2
\right].
$$

Choosing $w_t=\sigma_t^2$ cancels this scale factor and gives the familiar diffusion objective

$$
\boxed{
\mathcal L_{\epsilon}(\theta)
=
\mathbb E_{z,\epsilon,t}
\left[
\left\|
\epsilon_\theta(x_t,t)-\epsilon
\right\|_2^2
\right]
},
\qquad
x_t=\alpha_t z+\sigma_t\epsilon.
\tag{10}
$$

This is supervised regression: the training procedure injected $\epsilon$, so its exact target is already known.

**What one network call contains — Figure 2**

- The noisy latent $x_t$ enters the denoising network, commonly a [DiT](#2022-billpeeblessainingxie-dit-scalable-diffusion-models-with-transformers).
- The scalar $\lambda_t$ is lifted with Fourier or sinusoidal features followed by an MLP, then injected through adaptive normalization or a related conditioning mechanism.
- A text caption $c$, if present, is encoded separately and supplied through cross-attention or adaptive normalization.
- The network emits $\hat\epsilon_\theta$ in the $\epsilon$-parameterization. At training time it is compared with the known $\epsilon$; at sampling time it is converted into a clean prediction and used to construct the next reverse transition. The same architecture can instead emit $\hat D_\theta$ or $\hat v_\theta$.

**Recovering the clean-data prediction**

Impose the model analogue of $x_t=\alpha_t z+\sigma_t\epsilon$:

$$
\boxed{
D_\theta(x_t,t)
=
\frac{x_t-\sigma_t\epsilon_\theta(x_t,t)}{\alpha_t}
=
\frac{x_t+\sigma_t^2s_\theta(x_t,t)}{\alpha_t}
}.
\tag{11}
$$

The second equality follows from $\epsilon_\theta=-\sigma_t s_\theta$; the corresponding equality in the tutorial has another sign typo. Comparing

$$
D_\theta-z
=
-\frac{\sigma_t}{\alpha_t}
(\epsilon_\theta-\epsilon)
$$

shows the equivalent loss weightings:

$$
\frac{1}{\sigma_t^2}
\|\epsilon_\theta-\epsilon\|_2^2
=
\frac{\alpha_t^2}{\sigma_t^4}
\|D_\theta-z\|_2^2.
$$

After applying $w_t=\sigma_t^2$, the simple $\epsilon$-MSE is equivalently

$$
\|\epsilon_\theta-\epsilon\|_2^2
=
\frac{\alpha_t^2}{\sigma_t^2}
\|D_\theta-z\|_2^2
=
\operatorname{SNR}_t\|D_\theta-z\|_2^2.
$$

Thus changing the prediction parameterization without also tracking the induced weighting can change the numerical training problem even when the ideal predictions are algebraically convertible.

**Why $\epsilon$-prediction becomes fragile for large sampling jumps**

At low SNR, $\alpha_t$ is small and $\sigma_t$ is large. If

$$
\epsilon_\theta=\epsilon+\delta_t,
$$

then

$$
D_\theta-z
=
-\frac{\sigma_t}{\alpha_t}\delta_t.
\tag{12}
$$

A small noise-prediction error is magnified by $\sigma_t/\alpha_t$. The amplification itself does not depend on the number of sampling steps; fewer steps make it more damaging because each reverse step must jump across a larger SNR interval. For a deterministic jump from noisy $t$ to cleaner $s<t$,

$$
x_s^\theta-x_s
=
\left(
\sigma_s-\frac{\alpha_s\sigma_t}{\alpha_t}
\right)\delta_t.
$$

When $s\approx t$, the two terms nearly cancel. With few large jumps they need not, and there are fewer later iterations available to correct the error. This is the numerical motivation emphasized by [Progressive Distillation](https://arxiv.org/abs/2202.00512).

**The $v$-parameterization**

Define

$$
\boxed{
v_\theta(x_t,t)
=
\alpha_t\epsilon_\theta(x_t,t)
-\sigma_t D_\theta(x_t,t)
}.
\tag{13}
$$

Together with $x_t=\alpha_t z+\sigma_t\epsilon$ and $\alpha_t^2+\sigma_t^2=1$, this is a rotation:

$$
\begin{bmatrix}x_t\\v_t\end{bmatrix}
=
\begin{bmatrix}
\alpha_t&\sigma_t\\
-\sigma_t&\alpha_t
\end{bmatrix}
\begin{bmatrix}z\\\epsilon\end{bmatrix}.
$$

The inverse rotation gives

$$
\boxed{D_\theta=\alpha_tx_t-\sigma_tv_\theta},
\tag{14}
$$

$$
\boxed{\epsilon_\theta=\sigma_tx_t+\alpha_tv_\theta}.
\tag{15}
$$

All coefficients remain bounded. At the high-noise endpoint $(\alpha_t,\sigma_t)\approx(0,1)$,

$$
D_\theta\approx-v_\theta,
\qquad
\epsilon_\theta\approx x_t.
\tag{16}
$$

At the low-noise endpoint $(\alpha_t,\sigma_t)\approx(1,0)$,

$$
D_\theta\approx x_t,
\qquad
\epsilon_\theta\approx v_\theta.
\tag{17}
$$

The sampler uses the first conversion to recover $D_\theta$; the score-matching loss uses the second to compare the implied $\epsilon_\theta$ with the known injected noise.

**Training with either $\epsilon$- or $v$-prediction**

```python
from typing import Literal

import torch.nn as nn
import torch.nn.functional as F


Parameterisation = Literal["v", "eps"]


def diffusion_loss(
    model: nn.Module,
    clean_batch: Tensor,
    schedule: CosineNoiseSchedule,
    parameterisation: Parameterisation = "v",
    condition: Tensor | None = None,
) -> Tensor:
    """Denoising score-matching loss for a diffusion model."""
    batch_size = clean_batch.shape[0]
    device = clean_batch.device

    # Sample an independent noise level for every training example.
    t = torch.rand(batch_size, device=device)
    log_snr_t = schedule.log_snr(t)
    alpha_t, sigma_t = schedule.alpha_sigma(log_snr_t)

    # Broadcast each per-example schedule scalar over its feature dimensions.
    feature_shape = (-1,) + (1,) * (clean_batch.ndim - 1)
    alpha_t = alpha_t.view(feature_shape)
    sigma_t = sigma_t.view(feature_shape)

    noisy_sample, true_noise = add_noise(clean_batch, alpha_t, sigma_t)
    network_output = model(noisy_sample, log_snr_t, condition)

    # Convert either parameterization to epsilon before applying the same MSE.
    if parameterisation == "v":
        predicted_noise = sigma_t * noisy_sample + alpha_t * network_output
    elif parameterisation == "eps":
        predicted_noise = network_output
    else:
        raise ValueError(f"Unknown parameterisation: {parameterisation!r}")

    return F.mse_loss(predicted_noise, true_noise)
```

Each training example draws its own $t$, so the network sees scattered supervised examples across the whole noisy probability path. It does not simulate an entire forward trajectory during each optimization step.

### Section 3: Inference

_**TL;DR:** Ancestral sampling starts at $x_T\sim\mathcal N(0,I)$ and draws successively cleaner states. To obtain those reverse conditionals, choose a Gaussian Markov forward process whose marginals match $p_t(x_t\mid z)=\mathcal N(\alpha_t z,\sigma_t^2I)$. Bayes' rule gives the exact reverse posterior when clean $z$ is known; generation replaces $z$ with the denoiser $D_\theta$ and samples backward to $x_0$._

**What ancestral sampling means**

The learned reverse process factorizes as

$$
p_\theta(x_{0:T})
=
p(x_T)
\prod_{t=0}^{T-1}
p_\theta(x_t\mid x_{t+1}).
$$

Sampling follows this directed chain from its root toward its descendants:

$$
x_T\sim\mathcal N(0,I),
\qquad
x_{T-1}\sim p_\theta(x_{T-1}\mid x_T),
\qquad
\ldots,
\qquad
x_0\sim p_\theta(x_0\mid x_1).
$$

This sequential procedure is **ancestral sampling**. In diffusion language, each conditional takes the current noisy latent and samples a slightly cleaner one. “Ancestral” describes sampling through a directed probabilistic model; it does not mean retrieving an ancestral training example.

The corruption formula from training,

$$
p_t(x_t\mid z)
=
\mathcal N(x_t\mid\alpha_t z,\sigma_t^2I),
$$

specifies a separate noisy **snapshot distribution** at each $t$. It does not by itself specify how $x_t$ and $x_{t+1}$ on one trajectory are coupled. To sample backward, Nando therefore first chooses a forward Markov transition connecting adjacent snapshots, then reverses that transition with Bayes' rule:

In this section, $p_t$ names the probability path, as in the MIT notes, while $q$ names the particular forward Markov chain chosen to realize it. Its marginals satisfy $q(x_t\mid z)=p_t(x_t\mid z)$.

$$
\boxed{
\text{noisy marginals}
\longrightarrow
\text{forward Markov transitions}
\longrightarrow
\text{exact reverse posterior given }z
\longrightarrow
\text{replace }z\text{ with }D_\theta.
}
$$

**Step 1: construct a Markov process with the desired marginals**

To connect the snapshot marginals above into a Markov chain, choose

$$
\boxed{
q(x_{t+1}\mid x_t)
=
\mathcal N\left(
x_{t+1}
\mathrel{\Big|}
\frac{\alpha_{t+1}}{\alpha_t}x_t,
\left[
\sigma_{t+1}^2
-\frac{\alpha_{t+1}^2}{\alpha_t^2}\sigma_t^2
\right]I
\right)
}.
\tag{18}
$$

Equivalently,

$$
x_{t+1}
=
\frac{\alpha_{t+1}}{\alpha_t}x_t
+
\sqrt{
\sigma_{t+1}^2
-\frac{\alpha_{t+1}^2}{\alpha_t^2}\sigma_t^2
}\,\eta_{t+1},
\qquad
\eta_{t+1}\sim\mathcal N(0,I).
$$

Substitute $x_t=\alpha_t z+\sigma_t\epsilon$:

$$
\mathbb E[x_{t+1}\mid z]
=
\frac{\alpha_{t+1}}{\alpha_t}\alpha_t z
=
\alpha_{t+1}z,
$$

$$
\begin{aligned}
\operatorname{Var}(x_{t+1}\mid z)
&=
\frac{\alpha_{t+1}^2}{\alpha_t^2}\sigma_t^2I
+
\left(
\sigma_{t+1}^2
-\frac{\alpha_{t+1}^2}{\alpha_t^2}\sigma_t^2
\right)I\\
&=\sigma_{t+1}^2I.
\end{aligned}
$$

So this adjacent transition reproduces the required probability-path marginal $p_{t+1}(x_{t+1}\mid z)=\mathcal N(\alpha_{t+1}z,\sigma_{t+1}^2I)$. The successive noises are coupled into one Markov trajectory even though training can sample any marginal $x_t\mid z$ directly.

**Step 2: derive the exact reverse posterior when $z$ is known**

The Markov property gives

$$
q(x_{t+1}\mid x_t,z)=q(x_{t+1}\mid x_t).
$$

Bayes' rule therefore yields

$$
\boxed{
q(x_t\mid x_{t+1},z)
=
\frac{
q(x_{t+1}\mid x_t)q(x_t\mid z)
}{q(x_{t+1}\mid z)}
}.
$$

All three terms on the right are Gaussian, so $(x_t,x_{t+1})$ is jointly Gaussian when conditioned on $z$. Let

$$
r_t=\frac{\alpha_{t+1}}{\alpha_t}.
$$

From the forward transition $x_{t+1}=r_tx_t+\text{independent noise}$,

$$
\begin{aligned}
\mathbb E[x_t\mid z]&=\alpha_t z,
&\operatorname{Var}(x_t\mid z)&=\sigma_t^2I,\\
\mathbb E[x_{t+1}\mid z]&=\alpha_{t+1}z,
&\operatorname{Var}(x_{t+1}\mid z)&=\sigma_{t+1}^2I,
\end{aligned}
$$

and their cross-covariance is

$$
\begin{aligned}
\operatorname{Cov}(x_t,x_{t+1}\mid z)
&=
\operatorname{Cov}\left(
x_t,
r_tx_t+\text{independent noise}
\mathrel{\Big|}z
\right)\\
&=r_t\sigma_t^2I.
\end{aligned}
$$

For jointly Gaussian variables, conditioning uses

$$
\mathbb E[A\mid B]
=
\mathbb E[A]
+\operatorname{Cov}(A,B)
\operatorname{Var}(B)^{-1}
(B-\mathbb E[B]),
$$

$$
\operatorname{Var}(A\mid B)
=
\operatorname{Var}(A)
-\operatorname{Cov}(A,B)
\operatorname{Var}(B)^{-1}
\operatorname{Cov}(B,A).
$$

Applying these with $A=x_t$ and $B=x_{t+1}$ gives

$$
\begin{aligned}
\mu_{t\mid t+1,z}
&=
\alpha_t z
+\frac{r_t\sigma_t^2}{\sigma_{t+1}^2}
(x_{t+1}-\alpha_{t+1}z),\\
\Sigma_{t\mid t+1,z}
&=
\left(
\sigma_t^2
-\frac{r_t^2\sigma_t^4}{\sigma_{t+1}^2}
\right)I.
\end{aligned}
$$

These are already the reverse mean and variance. The tutorial introduces $c$ only to make them more compact. Define

$$
c
=
1-e^{\lambda_{t+1}-\lambda_t},
\qquad
1-c
=
e^{\lambda_{t+1}-\lambda_t}
=
\frac{\alpha_{t+1}^2\sigma_t^2}
{\alpha_t^2\sigma_{t+1}^2}.
$$

Because $t+1$ is noisier, $\lambda_{t+1}<\lambda_t$, so $0<c<1$. The coefficients above become

$$
\frac{r_t\sigma_t^2}{\sigma_{t+1}^2}
=
\alpha_t\frac{1-c}{\alpha_{t+1}},
$$

$$
\alpha_t
-\frac{r_t\sigma_t^2}{\sigma_{t+1}^2}\alpha_{t+1}
=
\alpha_tc,
$$

$$
\sigma_t^2
-\frac{r_t^2\sigma_t^4}{\sigma_{t+1}^2}
=
c\sigma_t^2.
$$

Substituting these three identities into the Gaussian conditional mean and variance gives

$$
\boxed{
q(x_t\mid x_{t+1},z)
=
\mathcal N\left(
x_t
\mathrel{\Big|}
\alpha_t
\left[
\frac{1-c}{\alpha_{t+1}}x_{t+1}
+c z
\right],
c\sigma_t^2I
\right)
}.
\tag{19}
$$

The mean interpolates between two sources of information:

- $x_{t+1}$ carries the particular noisy trajectory being reversed.
- The clean endpoint $z$ says where that trajectory should ultimately go.
- The posterior variance accounts for information destroyed by the stochastic forward step; reversing it requires sampling rather than only taking a mean.
- For nearby noise levels, $c$ is small: the posterior stays close to a rescaled $x_{t+1}$ and adds little fresh noise. For a larger jump, $c$ grows: the clean endpoint matters more and the reverse conditional has greater uncertainty.

**Step 3: replace the unknown clean endpoint with the model**

During training, the exact posterior can condition on the known example $z$. During generation, there is no clean input to reconstruct—we are trying to create a new one. The network supplies

$$
D_\theta(x_{t+1},t+1)
=
\alpha_{t+1}x_{t+1}
-\sigma_{t+1}v_\theta(x_{t+1},t+1)
\tag{20}
$$

or, under $\epsilon$-prediction,

$$
D_\theta(x_{t+1},t+1)
=
\frac{
x_{t+1}-\sigma_{t+1}\epsilon_\theta(x_{t+1},t+1)
}{\alpha_{t+1}}.
$$

Plugging this estimate into the exact posterior defines the learned reverse transition:

$$
\boxed{
p_\theta(x_t\mid x_{t+1})
=
\mathcal N\left(
x_t
\mathrel{\Big|}
\alpha_t
\left[
\frac{1-c}{\alpha_{t+1}}x_{t+1}
+c\,D_\theta(x_{t+1},t+1)
\right],
c\sigma_t^2I
\right)
}.
\tag{21}
$$

The tutorial continues to label this distribution with $q$, but once the true $z$ has been replaced by a learned prediction, $p_\theta$ makes the distinction clearer: it is now the model's approximate reverse process.

For the code's more general jump from a current noisy level $t$ to a cleaner level $s<t$, define

$$
c=1-e^{\lambda_t-\lambda_s}.
$$

Then

$$
\mu_{s\mid t}
=
\alpha_s
\left[
\frac{1-c}{\alpha_t}x_t
+c\,D_\theta(x_t,t)
\right],
\qquad
\Sigma_{s\mid t}=c\sigma_s^2I.
$$

Drawing from this Gaussian means

$$
\boxed{
x_s
=
\mu_{s\mid t}
+\sqrt{c}\,\sigma_s\eta,
\qquad
\eta\sim\mathcal N(0,I).
}
$$

This single line is the ancestral sampling step: compute a denoised mean using $D_\theta$, then add the amount of fresh randomness prescribed by the reverse posterior.

**One ancestral sampling step**

```python
@torch.no_grad()
def ancestral_sampling_step(
    noisy_sample: Tensor,
    network_output: Tensor,
    log_snr_t: Tensor,
    log_snr_s: Tensor,
    parameterisation: Parameterisation,
    clip_to: tuple[float, float] | None = (-1.0, 1.0),
) -> tuple[Tensor, Tensor]:
    """Reverse one step from noisy level t to cleaner level s < t."""
    c = -torch.expm1(log_snr_t - log_snr_s)

    alpha_t = torch.sqrt(torch.sigmoid(log_snr_t))
    alpha_s = torch.sqrt(torch.sigmoid(log_snr_s))
    sigma_t = torch.sqrt(torch.sigmoid(-log_snr_t))
    sigma_s = torch.sqrt(torch.sigmoid(-log_snr_s))

    if parameterisation == "v":
        predicted_clean = alpha_t * noisy_sample - sigma_t * network_output
    elif parameterisation == "eps":
        predicted_clean = (
            noisy_sample - sigma_t * network_output
        ) / alpha_t
    else:
        raise ValueError(f"Unknown parameterisation: {parameterisation!r}")

    if clip_to is not None:
        predicted_clean = predicted_clean.clamp(*clip_to)

    posterior_mean = alpha_s * (
        noisy_sample * (1.0 - c) / alpha_t + c * predicted_clean
    )
    posterior_variance = sigma_s.square() * c
    return posterior_mean, posterior_variance
```

`-torch.expm1(a)` computes $1-e^a$ accurately when $a$ is close to zero. Clipping $D_\theta$ to the normalized data range, such as $[-1,1]$ for images, can prevent implausibly large clean predictions from destabilizing later steps, although clipping also changes the exact model distribution.

**The full reverse loop**

This snippet shares the schedule, parameterization, and imports above and assumes that `DEVICE` names the configured PyTorch device.

```python
@torch.no_grad()
def generate_samples(
    model: nn.Module,
    schedule: CosineNoiseSchedule,
    sample_shape: tuple[int, ...],
    num_steps: int = 100,
    parameterisation: Parameterisation = "v",
    condition: Tensor | None = None,
    clip_to: tuple[float, float] | None = None,
    device: torch.device = DEVICE,
) -> Tensor:
    sample = torch.randn(sample_shape, device=device)

    for step in reversed(range(1, num_steps + 1)):
        t_now = torch.tensor(step / num_steps, device=device)
        t_next = torch.tensor((step - 1) / num_steps, device=device)

        log_snr_t = schedule.log_snr(t_now)
        log_snr_s = schedule.log_snr(t_next)
        log_snr_t_batched = log_snr_t.expand(sample.shape[0])
        network_output = model(
            sample, log_snr_t_batched, condition
        )

        mean, variance = ancestral_sampling_step(
            noisy_sample=sample,
            network_output=network_output,
            log_snr_t=log_snr_t,
            log_snr_s=log_snr_s,
            parameterisation=parameterisation,
            clip_to=clip_to,
        )

        if step > 1:
            sample = mean + torch.randn_like(mean) * torch.sqrt(variance)
        else:
            sample = mean

    return sample
```

The sampling algorithm is:

1. Draw $x_T\sim\mathcal N(0,I)$.
2. At the current noise level, run the same network to predict $v_\theta$ or $\epsilon_\theta$.
3. Convert that output into $D_\theta$ and the Gaussian reverse mean and variance.
4. Sample the next cleaner latent. At the final step, use the mean so fresh noise is not added to the output.
5. Repeat until $x_0\approx D_\theta$.

**Relation to the MIT ODE and SDE samplers**

Nando's ancestral transition is a finite-step Gaussian update:

$$
x_s
=
\mu_\theta(x_t,t,s)
+\sqrt{\Sigma(t,s)}\,\eta.
$$

The reverse-SDE sampler in the MIT notes expresses the same kind of stochastic motion locally:

$$
X_{t+\Delta t}
\approx
X_t
+b_t(X_t)\Delta t
+g_t\sqrt{|\Delta t|}\,\eta.
$$

As $s$ approaches $t$, the ancestral posterior mean becomes the current state plus a small reverse drift, and its variance becomes proportional to the time interval. It therefore approaches an Euler-Maruyama step for the corresponding reverse SDE. The presentations differ mainly in what they derive first:

- **Ancestral DDPM view:** derive an exact finite Gaussian conditional using Bayes' rule, then plug in the network's denoising prediction.
- **Reverse-SDE view:** derive an infinitesimal drift involving the score, then numerically integrate it with fresh Brownian noise.
- **Probability-flow ODE / DDIM view:** alter the dynamics so no fresh noise is added after initialization, then integrate a deterministic vector field.

With an exact model, appropriately paired reverse SDE and probability-flow ODE samplers can have the same marginal distribution at every time even though their individual trajectories differ. Ancestral sampling belongs to the stochastic side: every non-final step draws fresh Gaussian noise.

Nando presents this classical discrete construction first because it follows from elementary Bayes and Gaussian conditioning and directly explains the posterior coefficients used by DDPM schedulers. The shorter-looking SDE and ODE formulas rely on the reverse-time SDE and Fokker-Planck results developed explicitly in the MIT course.

The part worth retaining is the generative structure, not the coefficients:

$$
\boxed{
x_T\sim\mathcal N(0,I)
\quad\longrightarrow\quad
\text{predict }D_\theta
\quad\longrightarrow\quad
\text{sample a cleaner Gaussian state}
\quad\longrightarrow\quad
x_0.
}
$$

**Inference as recurrent refinement — Figure 3**

Unrolling the sampling loop gives a tied-parameter recurrent network. The same DiT, with the same weights $\theta$, is applied at every step; what changes between calls is the input latent $x_t$ and the noise level $\lambda_t$. A text caption can be encoded once and the same conditioning reused by every DiT call.

Figure 3 is also a useful way to view recent recurrent-depth LLMs. **The diffusion sampler is a loop over latent state:** the same Transformer weights are reused, and each pass refines a hidden object—here, the noisy latent $x_t$.

- In [Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741), the same idea is applied inside an LLM: a shared recurrent block performs several latent “thinking” updates before emitting tokens, and the model can allocate more recurrent depth to harder examples.
- The earlier Feedback Transformer uses a related feedback mechanism along the sequence direction: representations for future tokens can read high-level representations computed for previous tokens, rather than only lower-layer states.

The analogy is not exact, but it is instructive. In diffusion, the recurrence index is diffusion time and the state is an external sample $x_t$. In looped LMs, the recurrence index is latent reasoning depth. In feedback transformers, the recurrent signal is memory from earlier sequence positions. **In all three cases, the core move is parameter sharing across repeated computation, which turns a feed-forward Transformer block into an iterative refinement machine.** See also [[papers-latent-recursive-reasoning]].

### Section 4: The Important Details

_**TL;DR:** Sections 2 and 3 contain the essential training and sampling mechanics, but a useful diffusion system also needs a representation, a neural-network architecture, sufficient scale and data, and a way to condition generation. This section focuses on the last item through classifier-free and classifier guidance._

The tutorial has so far hidden several choices behind the score network:

- Images or videos may be compressed by an encoder before diffusion and decoded afterward, as in latent diffusion. Pixel-space diffusion does not need this extra representation model.
- The denoiser may be a U-Net, DiT, or another architecture. The diffusion equations do not determine the architecture.
- Model and dataset scale matter substantially. For text-to-image systems, caption quality is particularly important because the model can only learn the correspondence supplied by its image–text pairs.

#### 4.1 Classifier-Free Guidance

Suppose $y$ is a condition such as a text caption and $x_t$ is the current noisy latent. Bayes' rule gives

$$
\log p_t(y\mid x_t)=\log p_t(x_t\mid y)+\log p_t(y)-\log p_t(x_t),
$$

so differentiating with respect to the noisy latent gives

$$
\nabla_{x_t}\log p_t(y\mid x_t)
=
\nabla_{x_t}\log p_t(x_t\mid y)
-
\nabla_{x_t}\log p_t(x_t).
$$

The right-hand side is the difference between the conditional and unconditional scores. Adding a scaled version of this direction to the unconditional score gives

$$
\begin{aligned}
s_{\mathrm{CFG}}(x_t,t,y)
&=s_{\mathrm{uncond}}(x_t,t)+w\bigl(s_{\mathrm{cond}}(x_t,t,y)-s_{\mathrm{uncond}}(x_t,t)\bigr)\\
&=(1-w)s_{\mathrm{uncond}}(x_t,t)+w s_{\mathrm{cond}}(x_t,t,y).
\end{aligned}
$$

This same linear combination can be applied to a consistently parameterized model output—score, noise, denoiser, or diffusion $v$. The tutorial uses $v$-prediction:

$$
\widetilde v_t
=
v_\theta(x_t,t,\varnothing)
+w\left[v_\theta(x_t,t,y)-v_\theta(x_t,t,\varnothing)\right].
$$

Here $\varnothing$ means “no condition.” The scale $w$ controls inference-time guidance:

- $w=0$: use the unconditional prediction.
- $w=1$: use the ordinary conditional prediction.
- $w>1$: extrapolate beyond the conditional prediction, usually strengthening prompt adherence at some cost to diversity or realism.

Only one network is needed. During training, independently replace $y$ by $\varnothing$ with a conditioning-dropout probability $p_{\mathrm{drop}}$. The resulting network learns both calls needed at inference. **The dropout probability $p_{\mathrm{drop}}$ and guidance scale $w$ are different hyperparameters**; the tutorial unfortunately uses $w$ while describing both.

For $w>1$, the guided output should be understood as a useful extrapolation rather than the exact score of the original conditional distribution $p_t(x_t\mid y)$. Sampling also requires both conditional and unconditional network evaluations at each step, although they can be evaluated together in one batch.

#### 4.2 Classifier Guidance

Classifier guidance obtains the same conditioning direction from a separate classifier. Train $p_\phi(y\mid x_t,t)$ to recognize the condition from inputs corrupted to every noise level. Bayes' rule gives the conditional score

$$
\nabla_{x_t}\log p_t(x_t\mid y)
=
\nabla_{x_t}\log p_t(x_t)
+
\nabla_{x_t}\log p_\phi(y\mid x_t,t),
$$

and the scaled version used in practice is

$$
s_{\mathrm{guided}}(x_t,t,y)
=
s_{\mathrm{uncond}}(x_t,t)
+w\nabla_{x_t}\log p_\phi(y\mid x_t,t).
$$

The classifier gradient points in a direction that makes the current noisy sample more likely to receive label $y$. The unconditional score keeps the sample near the data distribution; the classifier term steers it toward the requested condition.

For the Gaussian corruption path, score and clean-data prediction satisfy

$$
s_\theta(x_t,t)=\frac{\alpha_t D_\theta(x_t,t)-x_t}{\sigma_t^2}
\quad\Longleftrightarrow\quad
D_\theta(x_t,t)=\frac{x_t+\sigma_t^2s_\theta(x_t,t)}{\alpha_t}.
$$

Therefore adding classifier guidance to the score is equivalent to shifting the denoised prediction:

$$
D_{\mathrm{guided}}(x_t,t,y)
=
D_{\mathrm{uncond}}(x_t,t)
+
\frac{w\sigma_t^2}{\alpha_t}
\nabla_{x_t}\log p_\phi(y\mid x_t,t).
$$

Classifier guidance has several costs:

1. It requires a separate classifier in addition to the generative model.
2. That classifier must work on noisy inputs at every diffusion time, not merely on clean images.
3. Sampling requires a classifier forward pass and an input gradient at every step.
4. A strong guidance scale can exploit peculiarities of the classifier rather than improve the image naturally.

The tutorial compares the last failure mode to Google's Inception/DeepDream work. DeepDream deliberately optimized an image to excite classifier features, producing exaggerated textures and repeated eyes or fur. Classifier guidance can similarly amplify features that strongly activate the classifier. Such artifacts illustrate classifier exploitation, although their presence is not a reliable way to identify how an arbitrary image was generated.

Classifier-free guidance avoids the separate noisy classifier and derives both terms from one generative model, which is why it became the more common approach.

### Section 5: Connection with the DDPM Parameterization

_**TL;DR:** Nando's $(\alpha_t,\sigma_t)$ notation and the standard DDPM $(\beta_t,\bar\alpha_t)$ notation describe the same variance-preserving Gaussian corruption. Nando specifies the total amount of signal and noise present at time $t$; DDPM specifies the amount of new noise added by each adjacent step._

The tutorial writes a noisy sample at any time directly as

$$
x_t=\alpha_t z+\sigma_t\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I).
$$

For the variance-preserving schedule used here,

$$
\alpha_t^2+\sigma_t^2=1.
$$

Classic DDPM notation instead begins with the one-step transition

$$
q(x_t\mid x_{t-1})
=
\mathcal N\!\left(x_t\,\middle|\,\sqrt{1-\beta_t}\,x_{t-1},\beta_t I\right),
$$

where $\beta_t$ is the variance of the new noise added at step $t$. Define the cumulative signal retention

$$
\bar\alpha_t
=
\prod_{i=1}^t(1-\beta_i).
$$

Composing the one-step transitions gives the familiar DDPM marginal

$$
q(x_t\mid z)
=
\mathcal N\!\left(x_t\,\middle|\,\sqrt{\bar\alpha_t}\,z,(1-\bar\alpha_t)I\right),
$$

or equivalently

$$
x_t=\sqrt{\bar\alpha_t}\,z+\sqrt{1-\bar\alpha_t}\,\epsilon.
$$

The square roots appear because $\bar\alpha_t$ and $1-\bar\alpha_t$ specify **variance fractions**, while the coefficients multiply the random variables themselves. Multiplying a random variable by $a$ multiplies its variance by $a^2$.

The notations correspond as follows:

| Quantity | Nando's notation | Standard DDPM notation |
| --- | --- | --- |
| Signal amplitude at time $t$ | $\alpha_t$ | $\sqrt{\bar\alpha_t}$ |
| Signal variance fraction | $\alpha_t^2$ | $\bar\alpha_t$, called $\rho_t$ in the tutorial |
| Noise amplitude at time $t$ | $\sigma_t$ | $\sqrt{1-\bar\alpha_t}$ |
| Noise variance fraction | $\sigma_t^2$ | $1-\bar\alpha_t$ |
| One-step signal retention | $\alpha_t/\alpha_{t-1}$ | $\sqrt{1-\beta_t}$ |
| One-step added-noise variance | $1-\alpha_t^2/\alpha_{t-1}^2$ | $\beta_t$ |

In particular,

$$
\sqrt{1-\beta_t}
=
\frac{\alpha_t}{\alpha_{t-1}}
\quad\Longrightarrow\quad
\boxed{\beta_t=1-\frac{\alpha_t^2}{\alpha_{t-1}^2}}.
$$

Using $\sigma_t^2=1-\alpha_t^2$, the tutorial's final correspondence is the same identity written in terms of consecutive noise levels:

$$
\sigma_t^2-\frac{\alpha_t^2}{\alpha_{t-1}^2}\sigma_{t-1}^2
=(1-\alpha_t^2)-\frac{\alpha_t^2}{\alpha_{t-1}^2}(1-\alpha_{t-1}^2)
=1-\frac{\alpha_t^2}{\alpha_{t-1}^2}
=\beta_t.
$$

Thus the DDPM forward transition, its cumulative marginal, and the Gaussian reverse posterior derived in Section 3 are not different diffusion models. They are the same construction with different bookkeeping and occasionally an index shift.

This section introduces $\beta_t$, $\bar\alpha_t$, and $\rho_t$ only to translate DDPM papers. The rest of these notes return to $(\alpha_t,\sigma_t)$.

### Section 6: The Conditional-Expectation Trick

_**TL;DR:** Under squared-error loss, the best prediction from an input is the conditional mean of the target given that input. Diffusion exploits this by regressing on easy, sample-specific targets such as injected noise or conditional scores. Although each target depends on the particular clean example and noise draw, the optimal network output is their posterior average—the marginal score, denoiser, or vector field needed for generation._

The notation in the tutorial obscures a standard regression result. Let $Y$ be any vector-valued target and let $U$ be the information given to the predictor. Under squared error, the optimal predictor is

$$
\boxed{
f^*(U)=\mathbb E[Y\mid U],
\qquad
f^*
=
\arg\min_f
\mathbb E\left[\|Y-f(U)\|_2^2\right].
}
$$

For the proof, write $m(U)=\mathbb E[Y\mid U]$ and insert and subtract it:

$$
Y-f(U)=\bigl(Y-m(U)\bigr)+\bigl(m(U)-f(U)\bigr).
$$

Expanding the square and taking expectations gives

$$
\begin{aligned}
\mathbb E\|Y-f(U)\|_2^2
&=\mathbb E\|Y-m(U)\|_2^2+\mathbb E\|m(U)-f(U)\|_2^2\\
&\quad+2\mathbb E\left[(Y-m(U))^\top(m(U)-f(U))\right].
\end{aligned}
$$

The cross term is zero because, after conditioning on $U$, the second factor is fixed and

$$
\mathbb E[Y-m(U)\mid U]
=
\mathbb E[Y\mid U]-m(U)
=0.
$$

Therefore

$$
\mathbb E\|Y-f(U)\|_2^2
=
\underbrace{\mathbb E\|Y-m(U)\|_2^2}_{\text{irreducible error; independent of }f}
+
\mathbb E\|m(U)-f(U)\|_2^2.
$$

The first term is irreducible error and does not depend on $f$. Section 2.4 rearranges this identity in the opposite direction, so its “const” is the negative of this term. Either way, removing it does not change the minimizer: squared-error regression predicts $m(U)=\mathbb E[Y\mid U]$.

**Application to denoising score matching**

Training samples

$$
z\sim p_{\mathrm{data}},
\qquad
\epsilon\sim\mathcal N(0,I),
\qquad
x_t=\alpha_t z+\sigma_t\epsilon.
$$

The network receives $U=(x_t,t)$, but not the clean $z$ or sampled $\epsilon$. The analytically known conditional-score target is

$$
Y
=
\nabla_{x_t}\log p_t(x_t\mid z)
=
-\frac{\epsilon}{\sigma_t}.
$$

Squared-error regression therefore learns

$$
\begin{aligned}
s^*(x_t,t)
&=\mathbb E\left[-\frac{\epsilon}{\sigma_t}\,\middle|\,x_t,t\right]\\
&=-\frac{1}{\sigma_t}\mathbb E[\epsilon\mid x_t,t]\\
&=\nabla_{x_t}\log p_t(x_t),
\end{aligned}
$$

where the last equality is the posterior-average identity derived in Section 2.4. **The target for one training example is a conditional score, but the function learned from many examples is the marginal score.** The model does not recover the exact noise draw from an ambiguous noisy input; it returns the mean of all noise values compatible with that input.

The same result explains the common prediction targets:

| Sampled regression target $Y$ | Optimal prediction from $(x_t,t)$ | Meaning |
| --- | --- | --- |
| Clean data $z$ | $\mathbb E[z\mid x_t,t]$ | Denoiser / posterior-mean clean prediction |
| Injected noise $\epsilon$ | $\mathbb E[\epsilon\mid x_t,t]$ | Noise prediction |
| Conditional score $-\epsilon/\sigma_t$ | $\nabla_{x_t}\log p_t(x_t)$ | Marginal score |
| Conditional velocity $u_t^{\mathrm{target}}(X_t\mid Z)$ | $\mathbb E[u_t^{\mathrm{target}}(X_t\mid Z)\mid X_t=x_t]$ | Marginal vector field $u_t^{\mathrm{target}}(x_t)$ |

For the final row, differentiate the Gaussian path while holding $z$ and $\epsilon$ fixed:

$$
u_t^{\mathrm{target}}(x_t\mid z)
=
\frac{d x_t}{dt}
=
\frac{d\alpha_t}{dt}z
+
\frac{d\sigma_t}{dt}\epsilon.
$$

Regressing a network on this easy conditional velocity makes it learn its conditional expectation given the current state, which is the marginal vector field. This is the same marginalization principle as score matching and is the reason the tutorial introduces the trick immediately before flow matching.

### Section 7: Flow Matching

_**TL;DR:** The Gaussian corruption formula defines a probability distribution at every time. Flow matching learns a deterministic velocity field whose ODE transports samples through those same marginal distributions. Training remains ordinary regression: sample a clean point and noise, construct an intermediate point, and predict the analytically known velocity of that conditional path._

#### 7.1 The Key Insight

Continue for the moment with Nando's convention, in which $t=0$ is data and $t=1$ is noise:

$$
Z\sim p_{\mathrm{data}},
\qquad
\epsilon\sim\mathcal N(0,I),
\qquad
X_t=\alpha_t Z+\sigma_t\epsilon.
$$

For one fixed pair $(Z,\epsilon)$, differentiating the interpolant gives its conditional velocity:

$$
\frac{dX_t}{dt}
=
\dot\alpha_t Z+\dot\sigma_t\epsilon.
$$

At inference time, however, the ODE state $x$ does not reveal which clean point and noise produced it. The velocity that depends only on the current state must therefore average over all compatible pairs:

$$
\boxed{
u_t(x)
=
\mathbb E\left[
\dot\alpha_t Z+\dot\sigma_t\epsilon
\,\middle|\,
X_t=x
\right]
=
\dot\alpha_t\mathbb E[Z\mid X_t=x]
+
\dot\sigma_t\mathbb E[\epsilon\mid X_t=x].
}
$$

This is the same marginalization pattern used earlier for the score:

| Tractable conditional target | Function learned by squared-error regression |
| --- | --- |
| $\nabla_x\log p_t(x\mid z)$ | $\nabla_x\log p_t(x)$ |
| $u_t(x\mid z)$ | $u_t(x)$ |

The deterministic ODE

$$
\frac{dX_t}{dt}=u_t(X_t)
$$

has the same marginal probability distribution $p_t$ at every time as the Gaussian construction above. **It does not reproduce the same individual trajectories.** A Gaussian sample path is indexed by a particular $(Z,\epsilon)$; an ODE trajectory follows the posterior-averaged velocity at its current location. Appendix B proves that both constructions nevertheless evolve the same density.

This is the result that makes flow matching generative. Start from the easy endpoint and solve the ODE toward the data endpoint. With Nando's initial convention this means integrating backward from $t=1$ to $t=0$:

$$
X_1\sim\mathcal N(0,I)
\quad\xrightarrow[\text{solve backward}]{dX_t/dt=u_t(X_t)}\quad
X_0\sim p_{\mathrm{data}}.
$$

A simple Euler step is

$$
x_{t+\Delta t}
=
x_t+\Delta t\,u_\theta(x_t,t).
$$

Here $\Delta t<0$ when using Nando's data-to-noise convention for generation. Higher-order solvers such as RK4 obtain a more accurate trajectory with several velocity evaluations per step.

#### 7.2 Conditional Flow-Matching Training

The exact marginal field $u_t(x)$ contains posterior expectations over the unknown data distribution, so it cannot be evaluated directly. The conditional velocity is easy because training supplies both $z$ and $\epsilon$:

$$
u_t(x_t\mid z)
=
\dot\alpha_t z+\dot\sigma_t\epsilon,
\qquad
x_t=\alpha_t z+\sigma_t\epsilon.
$$

The conditional flow-matching loss is therefore

$$
\boxed{
\mathcal L_{\mathrm{CFM}}(\theta)
=
\mathbb E_{t,Z,\epsilon}
\left[
\left\|
u_\theta(X_t,t)
-
\left(\dot\alpha_t Z+\dot\sigma_t\epsilon\right)
\right\|_2^2
\right].
}
$$

By the conditional-expectation result from Section 6, the optimal predictor is

$$
u_\theta^*(x,t)
=
\mathbb E\left[
\dot\alpha_t Z+\dot\sigma_t\epsilon
\mid X_t=x
\right]
=
u_t(x).
$$

Thus the tractable conditional target teaches the desired marginal vector field without calculating $p_t(x)$ or the posterior $p_t(z\mid x)$.

**The convention used by the code**

The tutorial now reverses time to match the standard CondOT/CFM convention used in the [MIT course notes](course-mit-diffusion-2026.md):

$$
t=0:\text{ noise},
\qquad
t=1:\text{ data}.
$$

Its slightly softened straight path is

$$
X_t
=
\left[1-(1-\sigma_{\min})t\right]\epsilon+tZ.
$$

The endpoints and constant conditional velocity are

$$
X_0=\epsilon,
\qquad
X_1=Z+\sigma_{\min}\epsilon\approx Z,
\qquad
u_t(X_t\mid Z)
=
\frac{dX_t}{dt}
=
Z-(1-\sigma_{\min})\epsilon.
$$

When $\sigma_{\min}=0$, this is exactly the MIT notes' CondOT path:

$$
X_t=(1-t)\epsilon+tZ,
\qquad
u_t(X_t\mid Z)=Z-\epsilon.
$$

So Nando's CFM is not a different learning principle. The differences are only the reversed time convention and a small residual endpoint noise $\sigma_{\min}$.

```python
def conditional_flow_matching_loss(model, data, sigma_min=1e-3):
    batch_size = data.shape[0]

    # Stratify the times across the batch to reduce estimator variance.
    base_t = torch.rand(1, device=data.device)
    t = (base_t + torch.arange(batch_size, device=data.device) / batch_size) % 1.0
    t_view = t.view((-1,) + (1,) * (data.ndim - 1))

    noise = torch.randn_like(data)
    x_t = (1.0 - (1.0 - sigma_min) * t_view) * noise + t_view * data
    target_velocity = data - (1.0 - sigma_min) * noise

    predicted_velocity = model(x_t, t)
    return F.mse_loss(predicted_velocity, target_velocity)
```

Each training iteration samples a new $Z$, $\epsilon$, and $t$. The network is not asked to remember a particular straight line; across training it sees points from many conditional paths and learns their posterior-averaged velocity at every location.

#### 7.3 Flow-Matching Inference

Under the code's noise-to-data convention, generation simply integrates

$$
\frac{dX_t}{dt}=u_\theta(X_t,t),
\qquad
X_0\sim\mathcal N(0,I),
\qquad
t:0\rightarrow1.
$$

The tutorial uses RK4 rather than the ancestral Gaussian sampler from Section 3:

```python
def rk4_step(velocity, x, t, dt):
    k1 = velocity(x, t)
    k2 = velocity(x + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = velocity(x + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = velocity(x + dt * k3, t + dt)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def sample_flow(velocity, shape, num_steps, device):
    x = torch.randn(shape, device=device)
    dt = 1.0 / num_steps
    t = torch.tensor(0.0, device=device)

    def field(x, t_scalar):
        return velocity(x, t_scalar.expand(x.shape[0]))

    for _ in range(num_steps):
        x = rk4_step(field, x, t, dt)
        t = t + dt
    return x
```

Integrating the same ODE from $t=1$ to $t=0$ encodes a data point back into its noise-space coordinate. Exact inversion assumes an exact ODE solve; numerical error makes finite-step encode/decode only approximately inverse.

#### 7.4 Relation Between the Score and Velocity Field

For the Gaussian path

$$
X_t=\alpha_tZ+\sigma_t\epsilon,
$$

the score and velocity are two algebraic encodings of the same posterior information. Conditioned on $X_t=x$,

$$
x
=
\alpha_t\mathbb E[Z\mid x]
+
\sigma_t\mathbb E[\epsilon\mid x],
$$

$$
u_t(x)
=
\dot\alpha_t\mathbb E[Z\mid x]
+
\dot\sigma_t\mathbb E[\epsilon\mid x].
$$

Multiply the second line by $\alpha_t$ and subtract $\dot\alpha_t$ times the first:

$$
\alpha_tu_t(x)-\dot\alpha_t x
=
(\alpha_t\dot\sigma_t-\dot\alpha_t\sigma_t)
\mathbb E[\epsilon\mid x].
$$

Because the marginal Gaussian-path score is

$$
s_t(x)
=
\nabla_x\log p_t(x)
=
-\frac{1}{\sigma_t}\mathbb E[\epsilon\mid X_t=x],
$$

we obtain

$$
\boxed{
s_t(x)
=
\frac{
\alpha_tu_t(x)-\dot\alpha_t x
}{
\sigma_t(\dot\alpha_t\sigma_t-\alpha_t\dot\sigma_t)
}.
}
$$

Conversely,

$$
\boxed{
u_t(x)
=
\frac{\dot\alpha_t}{\alpha_t}x
+
\sigma_t
\left(
\frac{\dot\alpha_t\sigma_t}{\alpha_t}-\dot\sigma_t
\right)s_t(x).
}
$$

These conversions are specific to the Gaussian affine path, not to arbitrary probability paths. They explain why a Gaussian diffusion model trained to predict a score can supply an ODE velocity, and why a flow model can supply the score needed by an SDE sampler. Numerical implementations normally use the parameterization that avoids division by a vanishing endpoint coefficient.

### Section 8: Training LLMs Jointly with Diffusion Losses over Text and Images

_**TL;DR:** Continuous image patches can be incorporated into an autoregressive Transformer without quantizing them into vocabulary tokens. MAR keeps noisy patches outside the Transformer and uses a separate diffusion head for each next patch. Transfusion inserts a whole noisy image block into the Transformer, which denoises its patches jointly while also learning next-token prediction for text._

There are two separate time indices in this section:

- $\tau$ is an autoregressive position among image patches or sequence elements.
- $t$ is the diffusion noise level used while generating one patch or image block.

#### 8.1 MAR: Noise Enters a Separate Diffusion Head

For a sequence of continuous image patches $Z_1,\ldots,Z_N$, autoregressive factorization gives

$$
p(z_{1:N})
=
\prod_{\tau=1}^N p(z_\tau\mid z_{<\tau}).
$$

The Transformer summarizes the clean prefix in a hidden state

$$
h_\tau=f_\theta(z_{<\tau}),
$$

but a softmax is inappropriate for the next patch because $z_\tau$ is a continuous vector rather than a discrete token. MAR models the conditional distribution $p(z_\tau\mid h_\tau)$ with a small diffusion head. During training,

$$
X_{\tau,t}=\alpha_tZ_\tau+\sigma_t\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I),
$$

and the head receives the noisy target plus the clean-prefix context:

$$
\mathcal L_{\mathrm{MAR}}
=
\mathbb E
\left[
\left\|
\epsilon_\phi(X_{\tau,t},t,h_\tau)-\epsilon
\right\|_2^2
\right].
$$

**The main Transformer never sees $X_{\tau,t}$.** It processes the clean previously generated patches once. The changing noisy state is confined to the diffusion head, which is repeatedly evaluated to sample the next patch. After that patch is generated, it is appended to the prefix and the process advances to $\tau+1$.

Figure 4 in the tutorial depicts exactly this separation: `LLM(clean prefix) -> h_tau`, followed by `diffusion head(noisy target, h_tau) -> noise prediction`.

#### 8.2 Transfusion: Noise Enters the Transformer

Transfusion makes the opposite architectural choice. It constructs one mixed sequence containing discrete text tokens and continuous image-latent patches:

$$
\texttt{A cat <BOI> }z_1,\ldots,z_N\texttt{ <EOI> is cute}.
$$

The image is usually first compressed by a VAE and split into patch vectors, but those vectors are **not quantized into a finite codebook**. During training, all patches in one image block are corrupted at the same sampled time:

$$
X_{\tau,t}=\alpha_tZ_\tau+\sigma_t\epsilon_\tau,
\qquad
\epsilon_\tau\sim\mathcal N(0,I),
\qquad
\tau=1,\ldots,N.
$$

The Transformer receives clean text embeddings, noisy image-patch embeddings, and a diffusion-time embedding. Its attention mask is:

- causal across text and modality blocks;
- bidirectional among patches within the current image block.

The same Transformer weights support two losses:

$$
\mathcal L_{\mathrm{Transfusion}}
=
\mathcal L_{\mathrm{text\;CE}}
+
\lambda_{\mathrm{img}}
\mathbb E
\left[
\sum_{\tau=1}^N
\left\|
\epsilon_\theta(X_{1:N,t},t,\text{prefix})_\tau
-
\epsilon_\tau
\right\|_2^2
\right].
$$

Text positions use a vocabulary softmax and cross-entropy. Image positions use a continuous output adapter and diffusion MSE. The input and output adapters are modality-specific, while the large Transformer body is shared.

At inference time, the model generates text autoregressively until it emits `<BOI>`. It then inserts a block of Gaussian image latents and repeatedly runs the full Transformer to denoise that block. The image patches are updated together, not generated left to right. After denoising, `<EOI>` returns the model to text generation.

Figure 5 shows the important wiring: text and noisy image patches enter one Transformer; text outputs receive cross-entropy, while image outputs receive diffusion loss.

#### 8.3 MAR versus Transfusion

| Design choice | MAR | Transfusion |
| --- | --- | --- |
| What is autoregressive? | Individual image patches | Text and modality blocks; image patches within a block are parallel |
| Where noisy patches enter | Small diffusion head | Full Transformer input sequence |
| Main Transformer attention | Standard causal prefix | Causal across blocks, bidirectional within an image block |
| Repeated work during denoising | Small head; Transformer context can be cached | Full Transformer runs at every denoising step |
| Conditioning bottleneck | Diffusion head receives $h_\tau$ | Every image patch can attend to the available prefix |
| Benefit | Cheap iterative head and simple causal backbone | Shared end-to-end multimodal representations and parallel image-patch generation |
| Cost | Sequential generation across patches | More expensive denoising step |

Both methods replace discrete image-token cross-entropy with a continuous diffusion loss; they disagree about where iterative denoising should happen. The same wiring could use score, noise, clean-data, diffusion-$v$, or flow-matching targets as long as training and sampling use a consistent conversion.

### Appendix A: Deriving the Backward-Sampling Gaussian

Appendix A supplies the complete-the-square algebra behind the reverse Gaussian already derived more intuitively in [Section 3](#section-3-inference) using the standard conditional formula for jointly Gaussian variables. The two derivations produce the same distribution.

Let

$$
A=\frac{\alpha_{t+1}}{\alpha_t},
\qquad
B=\sigma_{t+1}^2-A^2\sigma_t^2.
$$

Then

$$
q(x_{t+1}\mid x_t)
=
\mathcal N(x_{t+1}\mid Ax_t,BI),
\qquad
q(x_t\mid z)
=
\mathcal N(x_t\mid\alpha_tz,\sigma_t^2I).
$$

Bayes' rule removes the denominator because it is constant with respect to the variable $x_t$ whose Gaussian shape we are deriving:

$$
q(x_t\mid x_{t+1},z)
\propto
q(x_{t+1}\mid x_t)q(x_t\mid z).
$$

Ignoring constants independent of $x_t$, its exponent is

$$
-\frac12
\left[
\frac{1}{B}\|x_{t+1}-Ax_t\|_2^2
+
\frac{1}{\sigma_t^2}\|x_t-\alpha_tz\|_2^2
\right].
$$

Collecting the quadratic and linear terms in $x_t$ gives the posterior precision and natural parameter:

$$
\Sigma^{-1}
=
\left(\frac{A^2}{B}+\frac{1}{\sigma_t^2}\right)I,
\qquad
\Sigma^{-1}\mu
=
\frac{A}{B}x_{t+1}+\frac{\alpha_t}{\sigma_t^2}z.
$$

Since $A^2\sigma_t^2+B=\sigma_{t+1}^2$,

$$
\Sigma
=
\frac{B\sigma_t^2}{\sigma_{t+1}^2}I,
\qquad
\mu
=
\frac{A\sigma_t^2}{\sigma_{t+1}^2}x_{t+1}
+
\frac{B\alpha_t}{\sigma_{t+1}^2}z.
$$

Writing

$$
c=1-e^{\lambda_{t+1}-\lambda_t},
\qquad
\lambda_t=\log\frac{\alpha_t^2}{\sigma_t^2},
$$

reduces these coefficients to the sampler form

$$
\boxed{
q(x_t\mid x_{t+1},z)
=
\mathcal N\left(
x_t
\mathrel{\Big|}
\alpha_t\left[
\frac{1-c}{\alpha_{t+1}}x_{t+1}+cz
\right],
c\sigma_t^2I
\right).
}
$$

Generation substitutes the denoiser $D_\theta(x_{t+1},t+1)$ for the unknown clean endpoint $z$, exactly as shown in Section 3.

### Appendix B: Why Conditional Flow Matching Produces the Correct Probability Path

_**Claim:** Let $X_t=\alpha_tZ+\sigma_t\epsilon$ define the desired marginal density $p_t(x)$, and let_

$$
u_t(x)
=
\mathbb E\left[
\dot\alpha_tZ+\dot\sigma_t\epsilon
\mid X_t=x
\right].
$$

_Then the ODE $dX_t/dt=u_t(X_t)$ has marginal density $p_t$ at every time._

The proof has two parts:

1. Show that the Gaussian construction's density $p_t$ obeys the continuity equation with velocity $u_t$.
2. Show that the ODE transports its density according to that same continuity equation. With the same endpoint distribution, uniqueness makes the two density paths identical.

**Part 1: the Gaussian construction obeys the continuity equation**

Use the characteristic function, the Fourier transform of the density:

$$
\widehat p_t(k)
=
\mathbb E[e^{ik^\top X_t}]
=
\int e^{ik^\top x}p_t(x)\,dx.
$$

Differentiate with respect to time:

$$
\begin{aligned}
\partial_t\widehat p_t(k)
&=ik^\top\mathbb E\left[\frac{dX_t}{dt}e^{ik^\top X_t}\right]\\
&=ik^\top\mathbb E\left[
\mathbb E\left[\frac{dX_t}{dt}\middle|X_t\right]
e^{ik^\top X_t}
\right]\\
&=\int ik^\top u_t(x)e^{ik^\top x}p_t(x)\,dx.
\end{aligned}
$$

The second line conditions on the current state; by definition, that conditional expectation is $u_t(X_t)$. Since

$$
\nabla_x e^{ik^\top x}=ik\,e^{ik^\top x},
$$

integration by parts moves the spatial derivative from the exponential onto the probability current $p_tu_t$:

$$
\partial_t\widehat p_t(k)
=
-\int e^{ik^\top x}\nabla_x\cdot\left(p_t(x)u_t(x)\right)\,dx.
$$

But differentiating the integral definition of $\widehat p_t$ also gives

$$
\partial_t\widehat p_t(k)
=
\int e^{ik^\top x}\partial_t p_t(x)\,dx.
$$

The two Fourier transforms agree for every frequency $k$, so their underlying functions agree:

$$
\boxed{
\partial_t p_t(x)
+
\nabla_x\cdot\left(p_t(x)u_t(x)\right)
=0.
}
$$

This is the continuity equation: probability is transported by the velocity field without being created or destroyed.

**Part 2: the ODE has the same density evolution**

Expand the divergence in the continuity equation:

$$
\partial_t p_t(x)
+
u_t(x)\cdot\nabla_xp_t(x)
+
p_t(x)\nabla_x\cdot u_t(x)
=0.
$$

Now follow one ODE trajectory $x(t)$ satisfying

$$
\frac{dx(t)}{dt}=u_t(x(t)).
$$

The chain rule gives the change in density observed along that trajectory:

$$
\begin{aligned}
\frac{d}{dt}p_t(x(t))
&=\partial_t p_t(x(t))
+
\frac{dx(t)}{dt}\cdot\nabla_xp_t(x(t))\\
&=\partial_t p_t(x(t))
+
u_t(x(t))\cdot\nabla_xp_t(x(t))\\
&=-p_t(x(t))\nabla_x\cdot u_t(x(t)).
\end{aligned}
$$

Therefore

$$
\frac{d}{dt}\log p_t(x(t))
=
-\nabla_x\cdot u_t(x(t)),
$$

and integrating along the trajectory from $t_0$ to $t$ gives

$$
p_t(x(t))
=
p_{t_0}(x(t_0))
\exp\left(
-\int_{t_0}^{t}
\nabla_x\cdot u_s(x(s))\,ds
\right).
$$

The exponential corrects density for local volume change. Positive divergence expands nearby trajectories, so the same probability occupies more volume and density falls; negative divergence contracts them, so density rises.

Equivalently, if $\Phi_{t,t_0}$ is the ODE flow map, then

$$
x(t)=\Phi_{t,t_0}(x(t_0)),
\qquad
p_t(x(t))
=
p_{t_0}(x(t_0))
\left|
\det\frac{\partial x(t)}{\partial x(t_0)}
\right|^{-1}.
$$

This is precisely the density evolution encoded by the continuity equation. The analytically constructed $p_t$ and the ODE-induced density therefore solve the same equation with the same endpoint distribution. Under the usual regularity and uniqueness assumptions,

$$
\boxed{
X_t=\alpha_tZ+\sigma_t\epsilon
\quad\text{and}\quad
\frac{dX_t}{dt}=u_t(X_t)
\quad\text{have the same marginal }p_t\text{ for every }t.
}
$$

Again, this is equality of **marginal distributions**, not equality of sample-wise trajectories. That is exactly enough for generation: integrating the learned ODE from the simple Gaussian endpoint transports the whole population to the data distribution.

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
