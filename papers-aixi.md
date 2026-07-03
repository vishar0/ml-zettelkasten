# Universal AI (AIXI)

- **Created**: 2026-07-02
- **Last Updated**: 2026-07-02
- **Status**: `In Progress`
- **Related**:
  - [[compression]]
  - [[nncp]]
  - [[papers-generative-decision-making]]
  - [[papers-continual-learning]]

---

AIXI = **Solomonoff induction (prediction) + expectimax (control)**: the formal, incomputable upper bound of machine intelligence (the "universal AI" framework). Intelligence is measured by the **Legg-Hutter score** (expected cumulative reward over all computable environments, weighted by the universal prior), and AIXI is by definition its maximizer. Incomputable, so it can only be **approximated from below** with more compute — which is the thread connecting this note to [[compression]] and [[papers-generative-decision-making]].

---

## Theory / Foundations

- [ ] [2026] [GDM] From AGI to ASI - [paper](https://arxiv.org/abs/2606.12683). Uses AIXI as the formal endpoint of the AGI→ASI continuum.
- [ ] [2007] [Legg,Hutter] Universal Intelligence: A Definition of Machine Intelligence - [paper](https://arxiv.org/abs/0712.3329)
- [ ] [2024] [book] [Hutter] An Introduction to Universal Artificial Intelligence
- [ ] [1964] [Solomonoff] A Formal Theory of Inductive Inference

## Approximations / Practical

- [ ] TODO <https://www.lesswrong.com/posts/TtYuY2QBug3dn2wuo/the-problem-with-aixi>
- [ ] [2010] [jveness] MC-AIXI-CTW: Reinforcement Learning via AIXI Approximation - [paper](https://arxiv.org/abs/1007.2049)
- [ ] [2002] [Schmidhuber] The Speed Prior — computable prior restoring tractability - [paper](TODO link), <https://www.alignmentforum.org/posts/bC5xd7wQCnTDw7Kyx/getting-up-to-speed-on-the-speed-prior-in-2022>
- [ ] [2024] [Grau-Moya et al.] Learning Universal Predictors - [paper](https://arxiv.org/abs/2401.14953). Amortized Bayesian predictor via log-loss → universal limit in principle. Key "pretraining ≈ resource-bounded universal compression" citation.
- [ ] [2023/2026] [Catt et al.; Kim & Lee] Pushing the heavy lifting of AIXI into the predictor - [paper](TODO link)
- [ ] [2025] [Meulemans et al.] Embedded / multi-agent extension of AIXI (agent inside the environment class) - [paper](TODO link)

---

## [2026] [Genewein et al.] From AGI to ASI

- **Date**: 2026-07-02
- **Arxiv**: <https://arxiv.org/abs/2606.12683>
- **Zotero**: `Genewein et al. - 2026 - From AGI to ASI.pdf`

---

- **What it is**: GDM position/survey paper on AI progress *beyond* human-level AGI. Characterizes ASI, then maps four (parallel, non-exclusive) technological pathways AGI→ASI — (1) scaling compute/data/models, (2) algorithmic paradigm shifts, (3) recursive self-improvement, (4) ASI via multi-agent collectives — plus the frictions/bottlenecks along each (data wall, resource demand, paradigm insufficiency, research-gets-harder, **abstraction barrier**, deliberate slowdown). Uses AIXI as the theoretical frame throughout.
- **What AIXI says (§4)**: the incomputable optimal agent — maximize expected reward over all computable environments under Solomonoff's universal prior (prefer shorter programs = Occam). The **upper bound** of the Legg-Hutter intelligence measure and the **most data-efficient possible learner** (inherits Solomonoff's mistake bound), but bound by fundamental limits (physics, real-time, complexity-theory, Gödel/halting) — "**neither omniscient nor omnipotent**" (Table 2).
  - Three problems AIXI solves: acting under uncertainty (Bayesian mixture over environments), credit assignment (general RL), exploration-exploitation (resolved implicitly — explores only while useful).
- **Data-efficiency result**: Solomonoff induction has lowest cumulative prediction error / fewest mistakes on average over all computable environments — total surprise above optimal ≤ ~K(μ) (description length of the true environment), a constant independent of stream length. cf. [[compression]], [[nncp]].
- **Remark I (the continual-learning key)**: the correct comparison for AIXI is *an LLM architecture + training algorithm under continual-learning (cumulative lifetime) evaluation*, **NOT a frozen trained LLM**. i.e. the data-efficiency ideal is intrinsically **prequential/lifelong** — "most data-efficient" and "continual learner" are the same property viewed from the data vs. time axis. Frozen offline pretraining is a *departure* from the ideal, which is why it forgets and can't accumulate. cf. [[papers-generative-decision-making]] — CIC (Continual, Interactive, Causal Agents) is a concrete prequential-stream method: self-authored tokens are *interventions* (kept as context, masked from the target), the causal form of learning from one's own stream.
- **Approximation from below (the pretraining bridge)**: AIXItl / speed prior restore computability but stay impractical; more promisingly, most of the heavy lifting can be pushed into the *predictor* (Catt, Kim & Lee), and **an amortized Bayesian predictor trained by log-loss with a large parametric model could in principle reach the universal limit** (Grau-Moya 2024) — so massive pretraining = resource-bounded approximation of universal compression that improves with scale. Tentative theoretical license that the current paradigm *could* reach ASI without fundamental blockers (inconclusive; continual learning, long-context, robust planning remain clear practical gaps).
  - **Explicit vs. amortized Bayes**: Solomonoff/CTW *represent and update the mixture*; NNCP/LLMs *train one net (a point estimate) to mimic the mixture's outputs* — the prior survives only implicitly (architecture + SGD simplicity bias). Amortization is lossy: point estimate ≠ posterior → forgetting + OOD collapse.
