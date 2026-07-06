# Compression

- **Created**: 2026-06-07
- **Last Updated**: 2026-07-02
- **Status**: `In Progress`
- **Related**:
  - [[papers-aixi]]
  - [[papers-generative-decision-making]]
  - [[nncp]]
  - [[papers-gln]]

---

- [ ] [hutter-prize] [2019] Rationale for a Large Text Compression Benchmark, <https://mattmahoney.net/dc/rationale.html>
- [ ] [hutter-prize] Human Knowledge Compression Benchmark FAQ, <http://prize.hutter1.net/hfaq.htm>
- [ ] [hutter-prize] [gwern] [2026] Towards a Better Hutter Prize, <https://gwern.net/hutter-prize>
- [ ] [hutter-prize] [byronknoll] [2011] PAQ8: A Machine Learning Perspective on Predictive Coding with PAQ, <https://arxiv.org/abs/1108.3298>
- [ ] [hutter-prize] [byronknoll] cmix: <https://github.com/byronknoll/cmix>, <https://www.byronknoll.com/cmix.html>
- [ ] [hutter-prize] [[nncp]]
- [ ] [[papers-gln]]
- [ ] [byronknoll] gmix: <https://github.com/byronknoll/gmix>
- [x] [talk] [ilya] [2023] An Observation on Generalization (Simons Institute) - [video](https://www.youtube.com/live/AKMuA_TVz3A)
- [ ] [talk] [jackrae] [2023] Compression for AGI (Stanford MLSys) - [video](https://www.youtube.com/watch?v=dO4TPJkeaaU)
- [ ] [talk] [3blue1brown] [2026] Reinventing Entropy: Compression is Intelligence Part 1 - [video](https://www.youtube.com/watch?v=l6DKRf-fAAM&t=824s)
- [ ] [jveness] [2023] Language Modeling is Compression - [paper](https://arxiv.org/abs/2309.10668)
- [ ] [2024] Compression Represents Intelligence Linearly - [paper](https://arxiv.org/abs/2404.09937)
- [ ] [jveness] [2014] CNC: Compress and Control - [paper](https://arxiv.org/abs/1411.5326), [slides](https://www.hutter1.net/publ/scnc.pdf)
- [ ] [jveness] [2025] ActivePTW: Partition Tree Weighting for Non-Stationary Stochastic Bandits - [paper](https://arxiv.org/abs/2502.19325), [code](https://github.com/google-deepmind/active_ptw)
- [ ] [albertgu] [2025] CompressARC: ARC-AGI Without Pretraining - [blog](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html), [paper](https://arxiv.org/abs/2512.06104). cf. [[papers-latent-recursive-reasoning]]
- [ ] [2022] Less is More: Parameter-Free Text Classification with Gzip - [paper](https://arxiv.org/abs/2212.09410)
- [x] [2025] zip2zip: Inference-Time Adaptive Tokenization via Online Compression - [paper](https://arxiv.org/abs/2506.01084)
- [ ] [schmidhuber] [2009] Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes - [paper](https://arxiv.org/abs/0812.4360). cf. [[papers-open-ended-learning]]
- [ ] [2018] [FAIR] Description Length of Deep Learning Models — [papers](https://arxiv.org/abs/1802.07044)
- [ ] [2019] BB-ANS: Practical Lossless Compression with Latent Variables using Bits Back Coding - [paper](https://arxiv.org/abs/1901.04866). orig. Hinton & van Camp 1993; bridges VAE/diffusion ELBO → real compression; cf. [[papers-vae]] [[papers-diffusion-models]]
- [ ] UAI book (Hutter et al. 2024) → see [[papers-aixi]]
- [ ] TODO <https://www.adaptiveagents.org/_media/universal-ai-as-imitation.pdf>

---

## [2014] [jveness] [CNC: Compress and Control](https://arxiv.org/abs/1411.5326)

- **Date**: 2026-06-17
- **Slides**: <https://www.hutter1.net/publ/scnc.pdf>

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
    - Both are *model-based* — the axis is how the model yields behavior (forward rollout + search vs. inference), not model-based-vs-model-free. Canonical: **MuZero / Dreamer** (simulation) vs. **control-as-inference** (Levine 2018; Toussaint & Storkey 2006), which conditions on an *optimality* variable (treat reward/return as evidence) and infers the action posterior. CNC's "condition on return $z$" is exactly this move — but note it learns **no forward model** $p(s'\mid s,a)$, only $p(s\mid z,a)$ and $p(z\mid a)$, which is what lets it sidestep compounding rollout error.
  - CNC is the planning-as-inference branch, made tractable via compression — this is the stated contribution (and motivates why the Atari forward-model-for-MCTS comparison below is the natural baseline to beat):
    - > Our main contribution in this paper is to show how to set up a particularly tractable form of inference problem by generalizing compression-based classification to reinforcement learning.
- **Background (§2) — coding distributions & compression-based classification** (the blueprint §3 generalizes):
  - **Coding ≡ probability** (§2.2): a compressor and a distribution are the *same object*. Arithmetic coding turns any coding distribution $\rho$ into a code of length $\approx -\log_2 \rho(x_{1:n})$; run backwards, any compressor $z$ defines a distribution $\rho(x) = 2^{-\ell_z(x)}$. "Bits to encode $x$" = "how (im)probable the model thinks $x$ is" — few bits ⇒ familiar/high-probability.
    - > Given a coding distribution $\rho$ and a data sequence $x_{1:n}$, arithmetic encoding constructs a code $a_\rho$ which produces a binary codeword whose length is essentially $-\log_2 \rho(x_{1:n})$.
  - **Compression-based classification** (§2.3, Frank/Chui/Witten 2000): a *generative* classifier where each class's input-model is a compressor. Train one coding distribution $\rho_C$ per class on that class's inputs; classify new $Y$ by Bayes $P(C\mid Y,D) \propto \rho_C(Y)\,P(C\mid D)$ — i.e. **assign $Y$ to the class whose compressor encodes it in the fewest bits** (the class that finds $Y$ least surprising). Prior $P(C\mid D)$ from empirical class frequencies. (E.g. spam filtering, Bratko 2006: a spam-trained compressor squeezes a new spam mail shorter than a ham-trained one.) It's the MDL principle — the model that describes the data most cheaply is the best explanation.
    - > The main idea behind compression-based classification is to model $P[Y\mid C,D]$ using a coding distribution for the inputs that is trained on the subset of examples from $D$ that match class $C$. [...] Thus the overall accuracy of the classifier essentially depends upon how well the inputs can be modeled by the class conditional coding distribution.
  - **So what (why §2.3 is here)**: it recasts a *discriminative* task (classification) as *generative modeling + Bayes* — model $P(Y\mid C)$ per class instead of learning $P(C\mid Y)$ directly. CNC is **literally this, run per action $a$, with the "class" = the return $z$**:

| Compression-based classification | CNC (policy evaluation), *per action $a$* |
|---|---|
| classify input $Y$ | classify state $s$ |
| class $C$ | return $z$ |
| — | action $a$: conditioning context that *selects the classifier* (indexes the $(z,a)$ buckets), not a predicted label — like $D$ in the Bayes formula |
| class-conditional model $\rho_C(Y)$ | $\rho_S(s \mid z, a)$ |
| class prior $P(C)$ | return prior $\rho_Z(z \mid a)$ |
| Bayes → **argmax** over classes | Bayes → **expectation** over $z$: $\sum_z z\,P(z\mid s,a) = Q$ |

- $a$ is on the *conditioning* side of the bar (you're given it — it's the action being evaluated), so it isn't a class; the posterior $P(z\mid s,a)$ normalizes over $z$ with $a$ fixed (Eq. 5). The $|\mathcal{Z}||\mathcal{A}|$ buckets $=$ $|\mathcal{A}|$ separate classifiers (one per action), each with $|\mathcal{Z}|$ return-classes.
- The only twist vs. classification: take the **expectation** over return-classes, not the **argmax**. So $Q(s,a)$ = return-weighted "which return-class's compressor finds this state least surprising?".
- Headline payoff: **no feature engineering** — operate on raw bytes; the compressor finds task-relevant structure itself. Wins where features are hard to specify (formatted text, DNA, game frames).
- **The catch** (advantages/disadvantages): generative modeling of the input is *harder* than learning a discriminative boundary — you model the whole input distribution, far more than a decision boundary needs. CNC inherits exactly this trade-off vs. model-free value approximation: more general, possibly harder.
  - > On one hand, it is straightforward to apply generic compression techniques [...] to complicated input types such as richly formatted text or DNA strings [...]. On the other hand, learning a probabilistic model of the input may be significantly more difficult than directly applying standard discriminative classification techniques. Our approach to policy evaluation [...] raises similar questions.
- **§3 at a glance (plain terms)** — one line per subsection (details in the bullets below):
  - **§3.1 Overview** — value = average return, $Q(s,a)=\sum_z z\,P(z\mid s,a)$. Don't model $P(Z\mid S,A)$ directly; model states-per-return $P(S\mid Z,A)$ + return-frequencies $P(Z\mid A)$ and invert with Bayes. The only weird bit (justified in §3.2) is conditioning on the *future* return.
  - **§3.2 Transformation** — the return spans $m$ steps, so it's not a property of one moment. Bundle an $m$-step window into one "super-state" (the snake); now the return lives inside a single state, the bundled process settles to a unique equilibrium, and $P(Z\mid S,A)$ is read off from it. (This is what *earns the right* to condition on the future return.)
  - **§3.3 Online Policy Evaluation** — the bucket algorithm: keep one compressor per (return, action) bucket; file each state under the return it actually got; to value an action, see which return-bucket compresses the current state most cheaply, weight by return-frequency, average the returns. Online, no train/test split; an experience can only be filed once its $m$ future rewards are seen.
  - **§3.4 Analysis** — if the compressors are *consistent*, the value estimate converges to the true $Q$ at rate $\sim 1/\sqrt n$; both counting (frequency) and CTW qualify.
- **Markov-chain glossary** (the jargon the lemmas use):
  - **(Homogeneous) Markov chain (HMC)** — a memoryless random process (next state depends only on the current one) whose transition rule doesn't change over time.
  - **Stationary distribution $\nu$** — the long-run equilibrium: the fraction of time spent in each state if you run forever; a fixed point of the dynamics ($\nu P=\nu$). (Like a deck of cards after enough shuffles — uniform regardless of the start.)
  - **IR (irreducible)** — every state is reachable from every other; one connected piece, no stranded islands.
  - **PR (positive recurrent)** — from any state you always return, with *finite expected return time*; nothing drifts off and never comes back.
  - **AP / EA (aperiodic / essentially aperiodic)** — returns aren't locked to a fixed cycle, so the chain actually *settles* to $\nu$ rather than oscillating forever. (EA is a mild relaxation that also tolerates transient states.)
  - **IR + EA + PR ⇒ "ergodic"** — there's a *unique* $\nu$, the chain converges to it from any start, and — the property CNC relies on — **time-averages along one long run equal averages under $\nu$** (the *ergodic theorem*), which is what lets CNC pool experience over time into buckets.
- **The core idea**:
  - Want $Q^\pi(s,a) = \sum_z z\, P(Z = z \mid s, a)$, where $Z$ is the (finite, $m$-horizon) return. Instead of modeling $P(Z \mid s,a)$ directly, **flip it with Bayes**:
    $$\hat{Q}^\pi(s,a) = \sum_{z \in \mathcal{Z}} z\,\frac{\rho_S(s \mid z, a)\,\rho_Z(z \mid a)}{\sum_{z' \in \mathcal{Z}} \rho_S(s \mid z', a)\,\rho_Z(z' \mid a)}$$
    - $\rho_S(s \mid z, a)$ — a density/compression model over **states**, conditioned on the return-action pair ("what do states that led to return $z$ under action $a$ look like?").
    - $\rho_Z(z \mid a)$ — a model over **returns** given action.
    - > In the spirit of compression-based classification, CNC estimates this distribution by using Bayes rule to combine learnt density models of both $P(S \mid Z, A)$ and $P(Z \mid A)$. Although it might seem initially strange to learn a model that conditions on the future return, the next section shows how this counterintuitive idea can be made rigorous. (§3.1)
  - **Predictive vs. generative (terminology)**: $\rho_S, \rho_Z$ are *predictive* coding distributions (compressors) — $\rho_S(s\mid z,a)$ is queried as the predictive probability of the next state given the past states in bucket $(z,a)$. "Generative" here refers to the generative-*classifier* strategy (model $P(s\mid z,a)$, Bayes-invert), implemented via those predictors *used as density estimators* — **no sampling**. The paper calls them both "coding distributions" (§3.3, predictive) and "density models" (§3.1, generative); the equivalence is the point. See the **Predictive vs. generative** section at the end of this note for the full discussion.
  - **The counterintuitive part**: $\rho_S$ conditions on the *future* return. This is made rigorous via the **augmented "snake" Markov chain** (Lemmas 1–2): stack $(A_t, S_t, R_t)$ tuples over an $m$-window into a single HMC state. Under (IR+EA+PR) — irreducible, essentially-aperiodic, positive-recurrent — that chain has a unique **stationary** distribution $\nu$ with a well-defined joint over $(Z, S, A)$. Conditioning on the future is fine once you reason about a stationary distribution rather than a forward simulation. This is in the spirit of **planning-as-inference** (Attias 2003; Botvinick & Toussaint 2012), but with the conditioning done against an explicitly-constructed stationary distribution.
    - > [closing remarks, §6] The most interesting aspect of this approach is the way in which it uses a learnt probabilistic model that conditions on the future return; remarkably, this counterintuitive idea can be justified both in theory and in practice.
  - **The snake construction unpacked (§3.2)** — two augmentations turn a multi-step return into a function of *one* Markov state:
    - **Aug 1 (Lemma 1) — fold the reward in.** Reward isn't normally part of the state; it's a function of the transition. Glue it on: $X_t=(A_t,S_t) \to Y_t=(A_t,S_t,R_t)$. Lemma 1's content is that $Y_t$ is *still* an (IR+EA+PR) HMC — ergodicity preserved.
    - **Aug 2 (Lemma 2) — stack a window (the "snake").** The $m$-horizon return $Z=\sum R$ spans $m$ steps, so it's a function of *no single* $Y_t$. Pack a sliding window into one super-state $W_t=(Y_t,\dots,Y_{t+m})$; now $Z$ is a deterministic function of $W_t$. Lemma 2: $W_t$ is also an (IR+EA+PR) HMC.
    - **Payoff.** Ergodic ⇒ unique stationary $\nu'$ over $(\mathcal A\times\mathcal S\times\mathcal R)^{m+1}$ ⇒ a joint $\nu$ over $\mathcal Z\times(\dots)$ ⇒ $P(Z\mid S_0,A_1)$ is well-defined & time-independent (Eq. 2–3). The point: a *trajectory* question (multi-step return) becomes a *stationary-distribution* question about one bigger chain.
    - **Why $m$ must be finite.** Structurally, $W_t$ is only a finite-dimensional Markov state if $m<\infty$ (infinite horizon ⇒ infinite-dim state, construction collapses). And finite $m$ + finite $\mathcal R$ ⇒ finite return space $|\mathcal Z|\le m\,|r_{\max}-r_{\min}|$, which the bucketing + $O(|\mathcal Z|)$ value sum need (discounting/continuous returns blow this up — see Limitations).
  - **Why time-independence is required**: CNC pools experience across *all* timesteps into $(z,a)$ buckets, so $P(Z\mid S,A)$ must be *one fixed distribution* (not $P_t$) for the pooled estimate to converge — by the ergodic theorem, time-averages → stationary-distribution expectations. It's manufactured by **time-homogeneous MDP + stationary policy + fixed $m$-horizon** (always summing $m$ rewards, so no shrinking return-to-go), which makes the snake chain time-homogeneous → unique stationary $\nu$ → time-independent conditional → time-independent $Q(s,a)$. NB: "stationary policy" (no $t$) ≠ "stationary distribution" (fixed point $\nu P=\nu$); CNC needs both. **Breaks under on-policy control**: $\epsilon$-greedy with decaying $\epsilon$ is non-stationary, so Thm 1 doesn't apply (empirical only) — the lossless/offline → adaptive gap (see *Adaptive / non-stationary* under Limitations).
- **Algorithm (online, embarrassingly simple)**:
  - Maintain $|\mathcal{Z}|\cdot|\mathcal{A}|$ buckets, each holding an instance of compressor $\rho_S$, plus $|\mathcal{A}|$ buckets of $\rho_Z$.
  - As experience streams in, route each state into the bucket matching its realized $(z, a)$ and update that compressor; route each return into its $a$-bucket.
  - To evaluate: query each bucket for its code-length of the candidate state, exponentiate ($\rho = 2^{-\ell}$), normalize per Eq. above. Cost $O(|\mathcal{Z}|)$ per query.
- **Theory**: consistent value estimation if $\rho_S, \rho_Z$ are consistent density estimators (Thm 1), with absolute error $\in O_P(n^{-1/2})$. Holds for the **frequency estimator** (Thm 2, tabular) and for **factored multi-alphabet Context Tree Weighting (CTW)** (Thm 3), which scales to larger state spaces. Caveat: the clean theory is for policy *evaluation* of a **stationary** policy; on-policy control violates stationarity and is empirical only.
- **Experiments**:
  - **§4.1 Blackjack** (validation of the theory): CNC tracks first-visit Monte Carlo, slightly better early due to Dirichlet smoothing; MSE $\to 0$ as predicted by §3.4. Small, exactly-solvable problem to confirm consistency.
  - **§4.2 On-policy control (Atari/ALE) — the core "compression for control" demonstration**:
    - **Goal & caveat.** Show CNC does *real on-policy control* and *scales* across very different density estimators. **Theorem 1 does not apply** here — ε-greedy + an improving policy violates the stationary-policy assumption, so all of §4.2 is **empirical**, outside the guarantee. The loop is implicit **generalized policy iteration**: act ε-greedily w.r.t. the current $\hat Q$ read from the buckets → collect experience → update buckets → $\hat Q$ sharpens → policy improves → repeat.
    - **Setup.** ALE Atari, mainly **Pong** (3 actions {UP, DOWN, NOOP}; reward ±1 per point; episode ends at 21; score $\in[-21,21]$). 4-frame time steps; **ε decays $1.0 \to 0.02$ over 200k steps**; **horizon $m=80$ ($\approx$5s)**; **10 trials $\times$ 2M steps**. $\rho_Z$ = SAD for *all* agents (the cheap, small piece) — all variation is in the state model $\rho_S$.
    - **Four $\rho_S$ models — the "any compressor works" point.** The *same* CNC machinery, four deliberately different compressors:
      - **Factored SAD** — count-based: 16×16 screen regions, a per-region SAD estimator over patches, screen prob = product over patches.
      - **Autoregressive logistic regression** — discriminative/online: per-pixel prob from local context (online ADAGRAD, random-search hyperparams), screen prob = product over pixels.
      - **Lempel-Ziv** — dictionary compressor; $\rho_S(s\mid\text{hist}) = 2^{-[\ell_{LZ}(\text{hist}\cdot s)-\ell_{LZ}(\text{hist})]}$ — a *non-probabilistic* compressor turned into a density via codelength.
      - **SkipCTS** — a Context Tree Weighting derivative with an ALE-tailored context function (the strongest model).
    - **Results (last-50-episode average in Pong).** **Factored SAD +3.29** (std err 2.49) — *the simplest model, best of the three*; **Lempel-Ziv −0.09** (std err 1.79) — roughly even, ~50% win rate; **logistic regression −17.87** (std err 0.38) — *failed* (authors blame insufficient training). All ran **real-time or better**. **CNC+SkipCTS → near-optimal Pong**, and competitive on **Freeway / Q\*bert** vs DQN and BASS (DQN is a different training regime, included only illustratively). Notable: the *count-based* model beat the *learned/discriminative* one — echoes "simple compression-style models are surprisingly strong; online discriminative density modeling is finicky."
    - **The result that matters most — CNC vs. forward-model planning.** The **same SkipCTS model** used as a *forward model for MCTS* (even with double progressive widening) was **useless**: the best simulation agent couldn't beat **−14** in Pong and was **no better than random** on Q\*bert/Freeway. Inside CNC the same model was **near-optimal, with orders of magnitude less compute**. Same model, opposite outcomes — forward rollout *compounds* error over the horizon (Talvitie 2014), CNC *never rolls forward* (one Bayes inversion of a stationary distribution), so it is **"more forgiving of modeling inaccuracies."** This is the empirical backbone of the planning-as-inference $>$ simulation argument in the Framing section.
    - **What it implies (and the gaps to target).** Existence proof that *any* compressor in the $\rho_S$ slot + ε-greedy yields a controller — the modular generality the abstract promises. But it also maps the exact limits: **stationarity violated** (buckets accumulate stale early-random experience, *no forgetting* → the online-but-not-adaptive gap); **small finite action & return spaces** (Pong: 3 actions enumerated for the argmax, score $\in[-21,21]$) — the favorable regime, large/continuous spaces break it (§5); **ε-greedy is the only exploration** (no principled exploration). These are precisely the openings for an *adaptive* compression-for-control objective.
- **Why it matters (compression $=$ control)**: this is the cleanest demonstration that value estimation can be *entirely* reduced to coding length. Choosing a density model $=$ "committing to a particular kind of compression-based similarity metric over the state space." **It opens RL to the full toolbox of density modeling / statistical compression.**
- **Limitations / open questions**:
  - Return space $\mathcal{Z}$ must be **small and finite** — cost scales with $|\mathcal{Z}|$, and **discounting introduces exponential dependence on the horizon**. Proposed fix: tree discretization of the return space (depth $d \gtrsim \log_2(m(r_{\max}-r_{\min})/\epsilon)$) or Monte Carlo approximation of Eq. 4.
    - > So far we have only applied CNC to undiscounted, finite horizon problems with finite action spaces, and more importantly, finite (and rather small) return spaces. This setting is favorable for CNC, since the per-step running time depends on $|\mathcal{Z}| \le m|r_{\max} - r_{\min}|$ [...]. However, even modest changes to the above setting can change the situation drastically. For example, using discounted return can introduce an exponential dependence on the horizon. Thus an important topic for future work is to further develop the CNC approach for large or continuous return spaces. (§5)
  - **No bootstrapping** — pure Monte Carlo return as the only learning signal; incorporating TD-style bootstrapping is open.
  - Whole approach rests on the **quality of the density estimator**, itself a hard problem; no guidance on when CNC beats model-free function approximation.
  - **Adaptive / non-stationary** extension flagged: convert a stationary coder into a piecewise-stationary one via expert-tracking meta-algorithms (György–Linder–Lugosi 2012; Partition Tree Weighting).
- **Connections / lineage**:
  - **Return-conditioned generative control**: the $\rho_S(s \mid z, a)$ "condition on a desired return, invert to a policy" move is the tabular/CTW ancestor of upside-down RL and **Decision Transformer / Decision Diffuser** (return-conditioned sequence/diffusion models for control).
  - **AIXI / algorithmic IT**: the authors note the open question of a formal link to Hutter's AIXI (2005) unification of algorithmic information theory and RL. cf. [[nncp]], the Hutter-prize line above.
  - **cf. CompressARC** above — the other face of "compression as objective": CompressARC compresses a single task's structure for reasoning (per-instance, no pretraining); CNC compresses the state-distribution-given-return for control. Both replace a learned task head with a coding-length computation. See also [[papers-latent-recursive-reasoning]].

### Predictive vs. generative, and what CNC actually models

> Context: working through CNC (Compress and Control). Question that kept coming up — the "better predictor = better compressor" duality is about *predictive* models, but CNC's formula is described as *generative*. Which is it? Resolution below.

**Predictive and generative aren't opposites.**

The thing the duality cares about is: does the model assign a likelihood ρ(x) to data, so that −log₂ ρ(x) is a codelength? Autoregressive/predictive models are the cleanest case, via the chain rule:

```
ρ(x₁:ₙ) = ∏ᵢ ρ(xᵢ | x<ᵢ)        →    −log ρ(x₁:ₙ) = −Σᵢ log ρ(xᵢ | x<ᵢ)
   (joint = "generative")              (codelength = sum of per-step prediction surprisals)
```

A next-symbol predictor is a generative model of the joint sequence — you recover the joint by multiplying conditionals, and you can sample it left-to-right. GPT is simultaneously "a next-token predictor" and "a generative model of text." So "predictive" (the conditionals) and "generative" (the joint) are the same object read two ways; the duality "better predictor = better compressor" is really "better likelihood = shorter code," and predictive models supply the likelihood by chaining.

(Where it gets subtle: VAEs/diffusion are "generative" but don't give an exact tractable likelihood — they give a bound (ELBO), so they only compress via bits-back/bound coding. Flows and autoregressive models give exact likelihoods, so they compress cleanly. CNC's models are in the clean autoregressive camp.)

**What CNC's models actually are.**

They're predictive. The paper defines ρ_S and ρ_Z (§3.3) as coding distributions — sequences of conditional PMFs ρ(xₙ | x<ₙ). Concretely:

- ρ_S(s | z, a) is queried as ρ_S(s | s^{z,a}_{0:n−1}) — the predictive probability of the next state s given the past states that fell in bucket (z, a). The Lempel-Ziv version is literally a codelength difference 2^(−[ℓ(hist·s)−ℓ(hist)]). Pure sequential predictor.
- ρ_Z(z | a) likewise predicts the next return given the past returns in action-bucket a.

So: yes, the machinery is predictive coding distributions / compressors, and the duality applies to them directly. There's no separate "generative model" being trained.

**So why do the paper say "generative"?** Two reasons, both legitimate, neither about sampling:

1. **"Generative classifier" is a factorization claim, not a sampling claim.** The discriminative-vs-generative distinction (Ng & Jordan) is: discriminative learns P(C | Y) directly; generative learns P(Y | C) and P(C), then inverts with Bayes. CNC does the latter — it models P(state | return, action) and P(return | action) and Bayes-flips. That is the textbook meaning of "generative" here, and it's what "generative classifier" / "generative decision-making" refer to. It has nothing to do with whether you sample.
2. **The predictor is being used as a density estimator.** What CNC needs from ρ_S is an estimate of the class-conditional density ν(s | z, a) — "how probable is state s among states of class (z,a)." A sequential predictor over a bucket's (roughly exchangeable) stream converges to exactly that marginal: the frequency estimator ρ(s|hist)=count(s)/(n−1) literally is the empirical class-conditional density; CTW/Dirichlet are smoothed versions (that's what Theorems 2–3 prove). So a predictive object is doing a generative/density-estimation job.

And note CNC only ever evaluates ρ_S at the observed s (a likelihood query) and Bayes-combines — it never samples. So "generative" here means "models the input distribution P(s | z, a)," not "produces samples."

**The clean statement.**

- The duality is about likelihood: better predictor ⇒ better likelihood ⇒ shorter code. ✓
- CNC's ρ_S, ρ_Z are predictive coding distributions (compressors) — the duality applies to them as-is.
- CNC is a generative classifier: it uses those predictors as estimates of the class-conditional density P(state | return, action) and the prior P(return | action), then inverts with Bayes. "Generative" = the modeling-the-inputs-and-Bayes-inverting strategy, implemented with predictive/compression models.

The paper signals exactly this by calling them both "coding distributions" (§3.3, predictive) and "density models" (§3.1, generative) — the equivalence is the whole point.

## [2025] [jveness] [ActivePTW: Partition Tree Weighting for Non-Stationary Stochastic Bandits](https://arxiv.org/abs/2502.19325)

- **Date**: 2026-07-02
- **Code**: <https://github.com/google-deepmind/active_ptw>

---

- **Abstract**:
  - > This paper considers a generalisation of universal source coding for interaction data, namely data streams that have actions interleaved with observations. Our goal will be to construct a coding distribution that is both universal *and* can be used as a control policy. Allowing for action generation needs careful treatment, as naive approaches which do not distinguish between actions and observations run into the self-delusion problem in universal settings. We showcase our perspective in the context of the challenging non-stationary stochastic Bernoulli bandit problem. Our main contribution is an efficient and high performing algorithm for this problem that generalises the Partition Tree Weighting universal source coding technique for passive prediction to the control setting.
- **One-liner**: derive the agent itself from **universal source coding** — build a coding distribution over the whole interaction stream (actions *and* percepts) and **sample actions from it**. Where CNC (above) used compression for policy *evaluation*, this uses it for the full *policy*, in a **non-stationary** setting. Same lineage (Veness/Hutter): CTW → PTW → CNC → this.
- **Framing**: agent design from Maximum Expected Utility (RL) vs. from *minimizing the expected bits to losslessly describe agent-environment interactions*. Loss = code length $-\log_2 \nu^\pi(h)$; regret = **redundancy**, which decomposes additively into *environment redundancy* (percept prediction) + *policy redundancy* (distance from the desired policy).
- **The self-delusion problem** (the durable conceptual point):
  - Naive approach: one Bayesian mixture over interaction measures, condition on everything observed — *including your own actions*. This fails: the posterior treats the agent's own actions as **evidence about the environment**, so the agent "confirms" hypotheses by acting as those hypotheses' policies would — believing its own outputs as if the world had produced them (Ortega et al. 2021).
  - Fix: the $\Vert$ notation — actions are **interventions (given), never coded as evidence**: $\nu(e_{1:t} \Vert a_{1:t})$. The posterior over environments updates **only on percepts**.
  - Resulting policy = **Bayesian Control Rule** (Ortega & Braun 2008): $\hat\pi(a_t \mid h) = \sum_\rho w^\rho_{t-1}\, \pi_\rho(a_t \mid h)$ — mix each candidate environment's *desired policy*, weighted by a percept-only posterior. Sampling from this **is Thompson sampling** — TS drops out of pure coding principles.
  - Design constraint for any generative decision-making system: **actions must enter the model as interventions, not observations** — a conditional generative model that conditions on actions the same way it conditions on frames walks into this trap the moment it goes on-policy.
- **PTW construction** (non-stationarity *inside* the model class, not bolted on):
  - Per-arm **KT estimators** (Beta(½,½) sequential predictors) code a stationary Bernoulli arm; redundancy ≤ (|A|/2)·log n + |A|.
  - **Partition Tree Weighting**: Bayesian mixture over *all binary temporal partitions* of time (all ways the world might have segmented into stationary regimes), prior weight $2^{-\Gamma_D(\mathcal{P})}$ — shorter tree description ⇒ higher weight, i.e. a **compression prior over change-point structures**. Fresh KT estimators within each segment.
  - Tractability: ~$2^{2^D}$ partitions, but at time $t$ only $D{+}1$ **active segments** matter (lengths 1, 2, 4, …, $2^D$, from the binary structure of $t$) ⇒ exact posterior in **O(log T) time/space per step**.
  - **ActivePTW** = generalized Thompson sampling: sample an active segment from the PTW posterior ("when did the current regime start?") → sample arm parameters from that segment's Beta posteriors → act with that environment's reference policy (greedy MEU, or MEU + forced exploration).
- **Forced exploration serves the coder**: constructed failure mode — a change-point where the previously-best arm's payoff stays the same while another arm silently becomes better; greedy play yields *zero evidence* of the change. Forced exploration at rate $1/\sqrt{l}$ within a segment fixes it (and is what the concentration analysis needs). Pure exploitation can starve the compressor of the data needed to detect change.
- **Results**: generally beats Sliding-Window UCB (even with oracle window), MASTER, Thompson sampling, UCB across change-point regimes. In stationary environments it *collapses to Thompson sampling* (posterior concentrates on the single-segment partition) — the adaptivity costs essentially nothing when the world is static.
- **Limits**: bandits only — no state, no long-horizon credit assignment; regret theory for the full algorithm deferred (redundancy bounds + concentration lemmas proven); needs a *reference policy per environment* (trivial for bandits, nontrivial for MDPs — the gap where CNC-style Q-estimation would plug in). BCR mixes reference policies one step at a time, which may under-explore over long horizons (cf. Leike et al. 2016 general TS, BayesEXP).
- **Connections**: [[nncp]] (adaptive coding of a passive stream ↔ this: adaptive coding of an *interaction* stream); CNC above (evaluation → full policy); PTW's partition posterior *forgets* old segments when the world changes — a compression-native answer to non-stationary/continual adaptation. Wiener's cybernetics (agent as entropy-constrained adaptive process) cited as the spiritual ancestor.

## [talk] [ilya] [2023] [An Observation on Generalization (Simons Institute)](https://www.youtube.com/live/AKMuA_TVz3A)

- **Date**: 2026-06-21

---

- **One-liner**: a mathematical account of *why unsupervised learning works* — compressing data **jointly** extracts the shared structure downstream tasks need; formalized as (conditional) Kolmogorov complexity, with SGD-over-nets as the tractable stand-in. Compression = prediction = a theory of unsupervised learning.
- A mathematical formulation with guarantees exists for supervised learning (low train error + more data than parameters ⇒ low test error). What's the equivalent for unsupervised learning?
- **Unsupervised learning: you optimize one objective, but you care about a different objective. And yet it works. How?**
  - Distribution matching as an example (eg., substitution ciphers, unsupervised machine translation).
    - Given datasets X and Y, find F such that distribution(F(X)) ~ distribution(Y).
- Compression to the rescue
  - **Compression is prediction, every compressor can be a predictor and vice versa**.
  - One-to-one correspondence between all predictors and all compressors.
- **Compression for reasoning about unsupervised learning**
  - Given: two datasets X and Y, and a good compression algorithm C(data).
  - Compress X and Y jointly.
  - What will a "sufficiently good compressor" do?
    - Use patterns that exist in X to help compress Y (and vice versa)
    - $\lvert C(\text{concat}(X,Y)) \rvert \le \lvert C(X) \rvert + \lvert C(Y) \rvert + O(1)$ (this upper bound always holds; the strict gain shows up only when there's structure to share)
    - Any additional compression that was gained by concatenation was some kind of shared structure the compressor knows. The better your compressor is, there is more shared structure to extract.
    - Gap = "shared structure" = algorithmic mutual information.
    - Generalizes distribution matching. If there exists an F such that distribution(F(X)) ~= Distribution(Y), then a good compressor will notice and exploit this.
- Can we formalize this?
  - Consider an algorithm A that tries to compress Y. Say it has access to X.
  - What is our regret of using this algorithm?
    - And regret relative to what?
    - Low regret = "we got all the value" out of the unlabelled data X. And nobody could get much more value that we did!
    - X can be a uniform distribution that we can learn nothing from, or X actually has structure that's useful to compress Y. Either way, a low-regret algorithm will have done the maximum to exploit X to compress Y.
- **Kolmogorov complexity as the ultimate compressor**
  - Gives the ultimate low-regret algorithm (ideal, not computable).
  - K(X) = length of the shortest program that outputs X.
  - If C is a computable compressor, then, for all X, $K(X) \lt \lvert C(X) \rvert + K(C) + O(1)$
    - See connection to Hutter Prize and [[nncp]].
  - K(X) is not tractable as it searches over all programs.
  - But training a neural network with SGD is not unlike doing a program search.
  - Simulation argument. A neural net is a simulator of computer programs. Architecture research is thus hard (one neural net can simulate another) except in rare cases (eg., RNN to transformer, as RNN has a severe bottelneck) when there's a big jump.
- **Conditional Kolmogorov complexity as the solution to unsupervised learning**
  - $K(Y|X) \lt \lvert C(Y|X) \rvert + K(C) + O(1)$
  - What is the absolute shortest way to describe dataset Y, assuming I have complete access to dataset X?
  - This is ultimate low-regret solution to unsupervised learning except that it's not computable.
- **"Just compress everything" also works**
  - $K(X,Y) = K(X) + K(Y|X) + O(\log(K(X,Y)))$
  - Chain rule (symmetry of information): the joint decomposes into $K(X)$ + the *conditional* $K(Y|X)$. So a good **joint** compressor automatically captures $K(Y|X)$ — i.e. plain next-token pretraining on one big concatenated pile picks up the transferable conditional structure *for free*, with no explicit conditioning and no paired data. That's why "just compress everything together" is already unsupervised learning that transfers.
- **Can we show universality of GPT-compression?**
  - Can we expect it to always work? (text clearly works; does the compression story generalize across modalities?)
  - Vision — lots of work on SSL for vision.
  - **iGPT (Image GPT, Chen et al. 2020)**: a GPT trained to autoregressively predict pixels with no labels; its features (linear probe) rival self-supervised CNNs on ImageNet. Evidence the *AR-prediction → good-representation* story isn't text-specific — it transfers to vision, supporting universality of the compression account.
- Linear representations
  - The compression theory does not immediately explain why representations are nice and linearly separable.
  - But linear representations are so pervasive that the reason for their formation must be deep and profound. (Flagged as an open puzzle.)
  - AR models seem to have better representations than BERT. **Intuition**: next-token prediction uses *left context only*, so the *hardest* predictions force integrating long-range structure; BERT's masked infilling sees *both sides*, so most masks are locally determined and easy. Representation quality is driven by the hardest prediction problems → AR's are harder → richer representations. (Offered as intuition, not proof.)
- Anything that turns a neural net into a probabilistic model assigning probabilities to inputs is **implicitly maximum likelihood = compression**, so the compression account applies to it — not just to autoregressive models. The differences between methods (AR, BERT, diffusion) are then about *how well/efficiently* they compress and *what representations* that induces, not whether the theory covers them.
- On diffusion
  - The other big family of likelihood models is diffusion. The diffusion models used in high-quality image generators don't actually maximize the likelihood of their inputs — they optimize a different (denoising) objective — but their original formulation *is* likelihood maximization.
  - Speculation: diffusion should also have *worse* representations than next-token prediction, for the same reason as BERT (the denoising/infilling task is easier than the hardest next-token prediction).
- **Connection**: the *theory* leg of the compression-thesis cluster in this file — pairs with *Language Modeling is Compression* (empirical: LLMs are SOTA compressors), *Compression Represents Intelligence Linearly* (compression rate ⇒ capability), and Jack Rae's *Compression for AGI* talk. Note the whole account is **passive / offline / lossless**; extending it to **control / decision-making** (cf. CNC above) and to **bounded-resource efficiency** (compression *per unit compute*, not just ratio) are the natural open directions.

## [2025] [zip2zip: Inference-Time Adaptive Tokenization via Online Compression](https://arxiv.org/abs/2506.01084)

- **Date**: 2026-07-02

---

- **tl;dr**: runs LZW online over the BPE token stream so the tokenizer adapts to each input at inference — merging recurring runs into "hypertokens" that cut sequence length 15–40% — i.e. adaptive compression pushed down to the *tokenizer* layer for efficiency.
  - LZW runs online over BPE tokens, merging recurring runs into per-input "hypertokens" that shorten the sequence at inference.
  - Not free / not magic: a one-time (~10 GPU-hr) finetune makes the model fluent in hypertokens; ~50% fewer tokens but only ~5–30% real speedup.
  - Why LZW over a bigger fixed BPE vocab (causal, per-input recurrence, self-synchronizing) is the same static-vs-adaptive lesson as [[nncp]] / continual learning.
- **Abstract**:
  - > Tokenization efficiency plays a critical role in the performance and cost of large language models (LLMs), yet most models rely on static tokenizers optimized on general-purpose corpora. These tokenizers' fixed vocabularies often fail to adapt to domain- or language-specific inputs, leading to longer token sequences and higher computational costs. We introduce zip2zip, a novel method for achieving context-adaptive tokenization in LLMs at inference time. Leveraging an online data compression algorithm (Lempel-Ziv-Welch), zip2zip dynamically expands its active vocabulary at inference time by continuously replacing fragmented token sequences with more compact hypertokens, which it can immediately output during generation. In doing so, the model refines its internal tokenization scheme to match the token distribution of the current context, reducing redundancy and improving representational efficiency. zip2zip consists of three key components: (1) a tokenizer based on Lempel-Ziv-Welch compression that incrementally merges co-occurring tokens into reusable hypertokens on the fly; (2) a dynamic embedding (and unembedding) layer that computes embeddings for newly formed hypertokens at runtime; and (3) a variant of autoregressive language modeling that pretrains the model to handle hypertokenized, compressed text sequences as inputs and outputs. We show that an existing LLM can be uptrained for zip2zip in 10 GPU-hours via parameter-efficient finetuning. The resulting LLM performs test-time adaptation, learning to use hypertokens in unseen contexts and reducing input and output tokens by 15-40%.
- **Pipeline**: BPE tokens → LZW forms hypertokens online (dictionary grows within each input) → a small **hyper-embedder** composes each hypertoken's embedding from its constituent token embeddings → transformer → next-token prediction over the static ∪ dynamic vocab → decode. A one-time (~10 GPU-hr) finetune on LZW-compressed data teaches the model to read/write hypertokens; the hyper-embedder is *trained then, only run at inference* (not test-time training).
- **Result**: up to ~50% fewer tokens, <1% perplexity hit — but composing embeddings + a dynamic vocab adds overhead, so real **speedup is much smaller than the token reduction** (~5–30%, hardware-dependent). Token count ≠ latency.
- **Why LZW, not a fixed/expanded BPE vocab** (the design crux, static vs. adaptive):
  - BPE captures **global frequency** over a *training corpus* (fit once, frozen); LZW captures **recurrence within this specific input** — local repetition (a phrase repeated 80× in one document) that no fixed vocab can pre-enumerate regardless of how it's trained.
  - LZW is **online/causal** — it builds its dictionary left-to-right, so it works *during* generation when the document doesn't exist yet; per-document BPE is two-pass/batch and can't.
  - LZW is **self-synchronizing** — decoder rebuilds the identical dictionary from the stream (no side-channel), and hypertokens self-describe via their constituents (so the hyper-embedder can compose them). Per-document BPE would need a transmitted merge table + embeddings for arbitrary new tokens.
  - Note LZW tracks **recurrence, not frequency** — no counters; a pattern earns a shorter code by recurring, gradually, not by being globally frequent.
- **Pushbacks**:
  - On *raw ratio* for a single, complete, known document, BPE-fit-on-that-doc can **beat** LZW (global view, no LZW warm-up cost). LZW is chosen for causality + decoder-sync, **not** compression ratio.
  - "If you finetune anyway, why not just expand the BPE vocab from the finetune data?" — only helps if the finetune corpus already covers the deployment domain; misses per-document repetition either way. LZW is what *physically shortens* the sequence (the payoff); the finetune only makes the model *tolerate* shortened input.
  - Is the added complexity worth it over a fixed vocab, and does data-dependent tokenization break clean apples-to-apples eval (BPE is fixed a priori; a dynamic tokenizer is input-dependent)? Open.
- **Connection**: classic LZ-family compressor bolted onto an NLP task — same genre as the gzip text-classification entry above, and the adaptive-coding sibling of [[nncp]]. Open directions: pruning (not just growing) the dynamic vocab; lossy/learned alternatives to LZW.
