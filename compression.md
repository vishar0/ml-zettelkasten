# Compression

- **Created**: 2026-06-07
- **Last Updated**: 2026-06-17
- **Status**: `In Progress`

---

- [ ] [hutter-prize] [2019] Rationale for a Large Text Compression Benchmark, <https://mattmahoney.net/dc/rationale.html>
- [ ] [hutter-prize] Human Knowledge Compression Benchmark FAQ, <http://prize.hutter1.net/hfaq.htm>
- [ ] [hutter-prize] [gwern] [2026] Towards a Better Hutter Prize, <https://gwern.net/hutter-prize>
- [ ] [hutter-prize] [byronknoll] [2011] PAQ8: A Machine Learning Perspective on Predictive Coding with PAQ, <https://arxiv.org/abs/1108.3298>
- [ ] [hutter-prize] [byronknoll] cmix: <https://github.com/byronknoll/cmix>, <https://www.byronknoll.com/cmix.html>
- [ ] [hutter-prize] [[nncp]] NNCP: Lossless Data Compression with Neural Networks
- [ ] [[papers-gln]] GLN: Gated Linear Networks
- [ ] [jveness] [2023] Language Modeling is Compression - [paper](https://arxiv.org/abs/2309.10668)
- [ ] [byronknoll] gmix: <https://github.com/byronknoll/gmix>
- [ ] [jveness] [2014] Compress and Control - [paper](https://arxiv.org/abs/1411.5326), [slides](https://www.hutter1.net/publ/scnc.pdf)
- [ ] [albertgu] [2025] CompressARC: ARC-AGI Without Pretraining - [blog](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html), [paper](https://arxiv.org/abs/2512.06104). cf. [[papers-latent-recursive-reasoning]]
- [ ] TODO UAI book

---

## [2014] [jveness] Compress and Control

- **Date**: 2026-06-17
- **Arxiv**: <https://arxiv.org/abs/1411.5326>
- **Slides**: <https://www.hutter1.net/publ/scnc.pdf>
- **Authors**: Veness, Bellemare, Hutter, Chua, Desjardins (Google DeepMind / ANU)

---

- **Abstract**:
  - > This paper describes a new information-theoretic policy evaluation technique for reinforcement learning. This technique converts any compression or density model into a corresponding estimate of value. Under appropriate stationarity and ergodicity conditions, we show that the use of a sufficiently powerful model gives rise to a consistent value function estimator. We also study the behavior of this technique when applied to various Atari 2600 video games, where the use of suboptimal modeling techniques is unavoidable. We consider three fundamentally different models, all too limited to perfectly model the dynamics of the system. Remarkably, we find that our technique provides sufficiently accurate value estimates for effective on-policy control. We conclude with a suggestive study highlighting the potential of our technique to scale to large problems.
- **One-liner**: Reduce **policy evaluation to density estimation** — *any* compressor/density model that yields a coding distribution becomes a value estimator. CNC (Compress aNd Control) learns two conditional density models and combines them with Bayes' rule to read off $Q^\pi$, with no learned value head and no forward rollout.
- **Framing / motivation (intro)**:
  - The pitch is to carry the compression-based-classification tradition (Frank/Chui/Witten 2000; Bratko 2006; Cilibrasi & Vitányi 2005) over to RL:
    - > In this paper we show how a similarly inspired approach can be applied to reinforcement learning, or more specifically, to the tasks of policy evaluation and on-policy control.
  - **Why policy evaluation is the target**: it's the shared bottleneck of the major RL algorithm families, and CNC recasts it as a density-modeling problem (the thesis sentence):
    - > The performance of well-known reinforcement learning techniques such as policy iteration [...], approximate dynamic programming [...] and actor-critic methods [...] all crucially depend on how well policy evaluation can be performed. In this paper we introduce a model-based approach to policy evaluation, which transforms the task of estimating a value function to that of learning a particular kind of probabilistic state model.
  - **The dichotomy CNC positions against** — two flavors of model-based RL. *Simulation-based* (learn a forward model, plan by search) suffers from compounding rollout error over long horizons (Talvitie 2014); *planning-as-inference* sidesteps simulation by turning planning into inference in a generative model:
    - > Simulation based techniques involve learning some kind of forward model of the environment from which future samples can be generated. Given access to such models, planning can be performed directly using search.
    - > In contrast, another family of techniques, referred to in the literature as planning as inference, attempt to side-step the issue of needing to perform accurate simulations by reducing the planning task to one of probabilistic inference within a generative model of the system.
  - CNC is the planning-as-inference branch, made tractable via compression — this is the stated contribution (and motivates why the Atari forward-model-for-MCTS comparison below is the natural baseline to beat):
    - > Our main contribution in this paper is to show how to set up a particularly tractable form of inference problem by generalizing compression-based classification to reinforcement learning.
- **The core idea**:
  - Want $Q^\pi(s,a) = \sum_z z\, P(Z = z \mid s, a)$, where $Z$ is the (finite, $m$-horizon) return. Instead of modeling $P(Z \mid s,a)$ directly, **flip it with Bayes**:
    $$\hat{Q}^\pi(s,a) = \sum_{z \in \mathcal{Z}} z\,\frac{\rho_S(s \mid z, a)\,\rho_Z(z \mid a)}{\sum_{z' \in \mathcal{Z}} \rho_S(s \mid z', a)\,\rho_Z(z' \mid a)}$$
    - $\rho_S(s \mid z, a)$ — a density/compression model over **states**, conditioned on the return-action pair ("what do states that led to return $z$ under action $a$ look like?").
    - $\rho_Z(z \mid a)$ — a model over **returns** given action.
  - **The counterintuitive part**: $\rho_S$ conditions on the *future* return. This is made rigorous via the **augmented "snake" Markov chain** (Lemmas 1–2): stack $(A_t, S_t, R_t)$ tuples over an $m$-window into a single HMC state. Under (IR+EA+PR) — irreducible, essentially-aperiodic, positive-recurrent — that chain has a unique **stationary** distribution $\nu$ with a well-defined joint over $(Z, S, A)$. Conditioning on the future is fine once you reason about a stationary distribution rather than a forward simulation. This is in the spirit of **planning-as-inference** (Attias 2003; Botvinick & Toussaint 2012), but with the conditioning done against an explicitly-constructed stationary distribution.
- **Connection to compression-based classification (Frank/Chui/Witten 2000)**: CNC is the RL generalization of compression-based classification. There you classify by training one compressor per class and asking which compresses the input best ($2^{-\ell_z(x)}$ as a coding distribution); here the "classes" are return-action buckets and the compressed object is the state.
- **Algorithm (online, embarrassingly simple)**:
  - Maintain $|\mathcal{Z}|\cdot|\mathcal{A}|$ buckets, each holding an instance of compressor $\rho_S$, plus $|\mathcal{A}|$ buckets of $\rho_Z$.
  - As experience streams in, route each state into the bucket matching its realized $(z, a)$ and update that compressor; route each return into its $a$-bucket.
  - To evaluate: query each bucket for its code-length of the candidate state, exponentiate ($\rho = 2^{-\ell}$), normalize per Eq. above. Cost $O(|\mathcal{Z}|)$ per query.
- **Theory**: consistent value estimation if $\rho_S, \rho_Z$ are consistent density estimators (Thm 1), with absolute error $\in O_P(n^{-1/2})$. Holds for the **frequency estimator** (Thm 2, tabular) and for **factored multi-alphabet Context Tree Weighting (CTW)** (Thm 3), which scales to larger state spaces. Caveat: the clean theory is for policy *evaluation* of a **stationary** policy; on-policy control violates stationarity and is empirical only.
- **Experiments**:
  - **Blackjack** (validation): CNC tracks first-visit Monte Carlo, slightly better early due to Dirichlet smoothing; MSE $\to 0$ as predicted.
  - **Atari / ALE on-policy control** ($\epsilon$-greedy, horizon $m=80 \approx 5$s): swap different models in for $\rho_S$ — factored **SAD** (Sparse Adaptive Dirichlet), autoregressive **logistic regression**, **Lempel-Ziv**, and **SkipCTS**. Under Lempel-Ziv, $\rho_S(s \mid \text{hist}) := 2^{-[\ell_{LZ}(\text{hist}\cdot s) - \ell_{LZ}(\text{hist})]}$ — literally the marginal code-length of appending the state. SkipCTS reaches **near-optimal Pong**; competitive on Freeway and Q\*bert.
  - **The striking result**: SkipCTS as a *forward model* for MCTS was useless (couldn't beat $-14$ in Pong), but the **same model** under CNC worked well with orders of magnitude less compute. CNC never rolls the model forward, so modeling errors don't **compound** over the horizon (the Talvitie 2014 problem) — it appears "more forgiving of modeling inaccuracies."
- **Why it matters (compression $=$ control)**: this is the cleanest demonstration that value estimation can be *entirely* reduced to coding length. Choosing a density model $=$ "committing to a particular kind of compression-based similarity metric over the state space." It opens RL to the full toolbox of density modeling / statistical compression.
- **Limitations / open questions**:
  - Return space $\mathcal{Z}$ must be **small and finite** — cost scales with $|\mathcal{Z}|$, and **discounting introduces exponential dependence on the horizon**. Proposed fix: tree discretization of the return space (depth $d \gtrsim \log_2(m(r_{\max}-r_{\min})/\epsilon)$) or Monte Carlo approximation of Eq. 4.
  - **No bootstrapping** — pure Monte Carlo return as the only learning signal; incorporating TD-style bootstrapping is open.
  - Whole approach rests on the **quality of the density estimator**, itself a hard problem; no guidance on when CNC beats model-free function approximation.
  - **Adaptive / non-stationary** extension flagged: convert a stationary coder into a piecewise-stationary one via expert-tracking meta-algorithms (György–Linder–Lugosi 2012; Partition Tree Weighting).
- **Connections / lineage**:
  - **Return-conditioned generative control**: the $\rho_S(s \mid z, a)$ "condition on a desired return, invert to a policy" move is the tabular/CTW ancestor of upside-down RL and **Decision Transformer / Decision Diffuser** (return-conditioned sequence/diffusion models for control).
  - **AIXI / algorithmic IT**: the authors note the open question of a formal link to Hutter's AIXI (2005) unification of algorithmic information theory and RL. cf. [[nncp]], the Hutter-prize line above.
  - **cf. CompressARC** above — the other face of "compression as objective": CompressARC compresses a single task's structure for reasoning (per-instance, no pretraining); CNC compresses the state-distribution-given-return for control. Both replace a learned task head with a coding-length computation. See also [[papers-latent-recursive-reasoning]].
