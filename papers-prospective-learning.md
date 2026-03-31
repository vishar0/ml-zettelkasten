# Prospective Learning

- **Created**: 2025-10-27
- **Last Updated**: 2026-03-31
- **Status**: `Done`

---

- [x] [2020] Simple Lifelong Learning Machines - [paper](https://arxiv.org/abs/2004.12908)
- [x] [2022] Prospective Learning: Principled Extrapolation to the Future - [paper](https://arxiv.org/abs/2201.07372)
- [x] [2024] Prospective Learning: Learning for a Dynamic Future - [paper](https://arxiv.org/abs/2411.00109)
- [x] [2025] Prospective Learning in Retrospect - [paper](https://arxiv.org/abs/2507.07965)
- [x] [2025] Optimal control of the future via prospective learning with control - [paper](https://arxiv.org/abs/2511.08717)

---

## [2020] Simple Lifelong Learning Machines

- **Date**: 2025-11-24
- **Arxiv**: <https://arxiv.org/abs/2004.12908>
- **Paperpile**: <https://app.paperpile.com/view/?id=e9a7adf4-4ac3-46de-98aa-0a09593bddfd>

---

- **Abstract**:
  - > In  lifelong  learning,  data  are  used  to  improve  performance not only on the present task, but also on past and future (unencountered) tasks. While typical transfer learning algorithms can improve performance on future tasks, their performance on prior tasks degrades upon learning new tasks (called forgetting). Many  recent  approaches  for  continual  or  lifelong  learning  have attempted   to maintain performance   on   old   tasks   given   new tasks. **But striving to avoid forgetting sets the goal unnecessarily low.  The  goal  of  lifelong  learning  should  be  to  use  data  to improve  performance  on  both  future  tasks  (forward  transfer) and past tasks (backward transfer)**. In this paper, we show that a   simple   approach—representation   ensembling—demonstrates both  forward  and  backward  transfer  in  a  variety  of  simulated and benchmark data scenarios, including tabular, vision (CIFAR- 100, 5-dataset, Split Mini-Imagenet, Food1k, and CORe50), and speech (spoken digit), in contrast to various reference algorithms, which  typically  failed  to  transfer  either  forward  or  backward, or  both.  Moreover,  our  proposed  approach  can  flexibly  operate with  or  without  a  computational  budget.
- **Intro**:
  - > While  it  is  relatively easy to simultaneously optimize for multiple tasks (multi-task learning) [4], it has proven much more difficult to sequentially optimize  for  multiple  tasks.
  - > For example, in humans, learning a second language often improves performance in an individual’s native language.
  - > In  lifelong  learning,  where  tasks  arrive  sequentially,  the ability to transfer knowledge across tasks is characterized by two complementary objectives: forward transfer and backward transfer. Forward  transfer facilitates  accelerated  learning  in new  tasks  using  previous  knowledge.  In  contrast, backward transfer evaluates  the  impact  of  new  learning  on  previously encountered tasks. Achieving both positive forward and back- ward transfer is crucial for an effective lifelong learner. How- ever, as we will demonstrate, many existing lifelong learning algorithms do not enable forward transfer to future tasks, and most  do  not  exhibit  positive  backward  transfer  to  previously learned tasks.
- **Section 2 — Related Work**:
  - Three camps: regularization/resource-constrained (EWC, LwF), modular/resource-growing (ProgNN, DF-CNN), replay-based
  - Key distinction: prior modular methods freeze backward transfer; SiLLy avoids this by allowing cross-task encoder interaction
- **Section 3 — Mathematical Framework**:
  - **Setup**: sequential tasks $t \in \{1,\ldots,T\}$, shared input space $\mathcal{X} \subset \mathbb{R}^D$, per-task label sets $\mathcal{Y}_t$, data i.i.d. from $\mathcal{D}_t$
  - **Objective**: $\min_{f \in \mathcal{F}} \sum_{t=1}^{T} \mathcal{E}_f^t\!\left(\bigcup_{t'=1}^{T} S^{t'}\right)$
  - **Transfer metrics** (log-ratios of errors; positive = good transfer):
    - $\text{Overall}^t = \log\left[\mathcal{E}_f^t(S^t) \,/\, \mathcal{E}_f^t\!\left(\bigcup_{t'} S^{t'}\right)\right]$
    - $\text{Forward}^t = \log\left[\mathcal{E}_f^t(S^t) \,/\, \mathcal{E}_f^t\!\left(\bigcup_{t' \leq t} S^{t'}\right)\right]$
    - $\text{Backward}^t = \log\left[\mathcal{E}_f^t\!\left(\bigcup_{t' \leq t} S^{t'}\right) \,/\, \mathcal{E}_f^t\!\left(\bigcup_{t'} S^{t'}\right)\right]$
    - Identity: $\text{Overall}^t = \text{Forward}^t + \text{Backward}^t$
  - Log-ratios (vs. raw differences) make metrics scale-invariant across tasks — most prior work doesn't decompose transfer this way
  - Most continual learning papers only track accuracy + forgetting avoidance, missing both transfer directions entirely
  - **Computational taxonomy**:
    - Parametric (EWC, SI, LwF): fixed capacity — cheap but limited
    - Semi-parametric (ProgNN, DER, Model Zoo): capacity grows with $T$, $O(nT)$ training
    - Non-parametric (IBP-WF): capacity adapts to data complexity
    - SiLLy operates in semi-parametric (SiLLy-N) or non-parametric (SiLLy-F) mode depending on budget
- **Section 4 — Representation Ensembling (SiLLy)**:
  - Hypothesis decomposed into encoder → channel → decoder; channels ensemble across all encoders, enabling both forward and backward interaction
  - **SiLLy-N (growing)**: adds a new deep net encoder per task; channels use decision forests over all encoder outputs
  - **SiLLy-N (constrained)**: freezes encoder addition after budget; only learns new channels over existing encoders
  - **SiLLy-F**: uses random forests as encoders; capacity scales with task complexity, not fixed per task
- **Section 5 — Simulation Study**:
  - Gaussian XOR/XNOR: SiLLy-N achieves positive forward + backward transfer; standard deep nets forget catastrophically
  - Adversarial tasks (rotated XOR): graceful degradation; backward transfer becomes nonlinear function of source sample size
- **Section 6 — Benchmark Study**:
  - 16 baselines including EWC, ProgNN, Model Zoo, Total Replay
  - CIFAR 10×10: SiLLy-N positive transfer on all tasks; competitors fail backward transfer
  - Also: Spoken Digit, FOOD1k 50×20, CORe50
  - Ablations: replay helps backward but not forward; pretrained encoders help forward but slightly hurt backward; performance saturates ~30 encoders on FOOD1k
  - Task/label order shuffling: SiLLy transfer stats nearly invariant; baselines sensitive
- **Section 7 — Discussion**:
  - Noted as suitable for federated learning (independent encoders per client)
  - Limitation: assumes task identity is known at inference (task-aware setting)
  - Code: <http://proglearn.neurodata.io/>
- **Takeaways (practitioner)**:
  - Forgetting avoidance is the wrong goal — you should demand *positive backward transfer*, i.e. new data actually improves old task performance.
  - Forward transfer is also largely unsolved — most baselines (EWC, LwF, ProgNN) fail at it; SiLLy is one of the few that achieves both directions.
  - The mechanism: ensemble representations across independent per-task encoders, not weights — decoupling is what prevents interference.
  - Budget-constrained mode works: freeze encoder addition, only learn new routing over existing encoders; performance degrades gracefully.
  - Replay doesn't solve this — it only helps backward transfer, not forward; not a general fix.
  - Log-ratio metrics matter practically: transfer scores are comparable across tasks with very different base error rates, unlike raw accuracy.
  - **Key limitation**: task identity must be known at train and inference time — no clean task boundaries = method breaks down; this is the gap Prospective Learning addresses next.

## [2022] Prospective Learning: Principled Extrapolation to the Future

- **Date**: 2025-10-27
- **Arxiv**: <https://arxiv.org/abs/2201.07372>
- **Paperpile**: <https://app.paperpile.com/view/?id=757687a3-71b3-4b28-9a6e-8e1376258296>

---

- **Abstract**:
  - > Learning is a process which can update decision rules, based on past experience, such that future performance improves. Traditionally, machine learning is often evaluated under the assumption that the future will be identical to the past in distribution or change adversarially. But these assumptions can be either too optimistic or pessimistic for many problems in the real world. Real world scenarios evolve over multiple spatiotemporal scales with partially predictable dynamics. Here we reformulate the learning problem to one that centers around this idea of dynamic futures that are partially learn- able.  We conjecture that certain sequences of tasks are not retrospectively learnable (in which the data distribution is fixed), but are prospectively learnable (in which distributions may be dynamic), suggesting that prospective learning is more difficult in kind than retrospective learning.  We argue that prospective learning more accurately characterizes many real world problems that (1) currently stymie existing artificial intelligence solutions and/or (2) lack adequate explanations for how nat- ural intelligences solve them.  Thus, studying prospective learning will lead to deeper insights and solutions to currently vexing challenges in both natural and artificial intelligences.
- **Character**: primarily a position/theory paper — defines the problem space, leaves "how" largely open (that's the 2024 paper's job)
- **Definition 1 — Retrospective Learning (PAC)**:
  - Assumes $P_t \equiv P$ for all $t$ (stationary distribution)
  - A hypothesis class $\mathcal{H}$ is PAC-learnable if $\exists$ a learner achieving $|R(\hat{h}) - R(h^*)| < \varepsilon$ w.p. $\geq 1-\delta$ with sufficient i.i.d. samples
  - Key failure mode: no single hypothesis can achieve small error on both $P_A$ and $P_B$ when tasks alternate — PAC learner is stuck
- **Definition 2 — Prospective Learning**:
  - A hypothesis class $\mathcal{H}$ is prospectively-learnable w.r.t. sequence $\mathbf{P} = \{P_t\}_{t \geq 0}$ if a learner $L$ can output a time-indexed sequence $\hat{h} = \{\hat{h}_t\}_{t \geq 0}$ such that:
  $$\lim_{T \to \infty} \frac{1}{T - \bar{t}} \int_{\bar{t}}^{T} \mathbb{P}\left[|R_t(\hat{h}) - R_t(h^*)| < \varepsilon\right] \geq 1 - \delta$$
  - Learner must achieve bounded error on an **infinite future time horizon** beyond some finite $\bar{t}$
- **Anticipation vs. Adaptation** (the key distinction):
  - Continual learner: eventually adapts to $P_A \to P_B$ switch after seeing new samples
  - Prospective learner: **models the temporal dynamics** and switches hypothesis without needing new samples at switch point
  - Example: if tasks alternate every $N$ steps, maintain two hypotheses and alternate — requires learning the dynamics, not just the tasks
- **Conjecture 1 — Continuum Hypothesis of Learning**:
  - Three complexity classes: PAC-learnable $\subset$ Prospectively-learnable $\subset$ Not learnable
  - Prospective learnability depends on the *complexity of task sequence dynamics*, not just individual tasks
  - Suggested measure: predictive information between past and future of the sequence
- **Where existing methods fall short**:
  - Online learning (FTL, OGD): minimizes regret but makes no predictions about future; exploits no temporal structure
  - Meta-learning (MAML): samples uniformly from seen tasks, ignores temporal order entirely
  - Continual learning: no formal learnability theory; just "avoid forgetting"
  - OOD generalization: special case where distribution shifts once then freezes — not general
  - RL: adapts as policy evolves but doesn't anticipate future distribution shifts
- **Biological grounding**: natural intelligences proposed as existence proofs of prospective learners; motivates neuro/AI collaboration around prospection across species and substrates
- **Takeaways (practitioner)**:
  - The core insight: adapting *after* a distribution shift is not enough — you need to *predict* when and how it will shift
  - PAC/ERM are provably insufficient for non-stationary problems even when the non-stationarity is perfectly regular (e.g. periodic tasks)
  - The framework subsumes OOD, continual learning, and online learning as special cases — it's a more general problem statement
  - **Key open question this paper leaves**: how do you actually build a prospective learner? → answered in the 2024 paper

## [2024] Prospective Learning: Learning for a Dynamic Future

- **Date**: 2025-10-27
- **Arxiv**: <https://arxiv.org/abs/2411.00109>
- **Paperpile**: <https://app.paperpile.com/view/?id=191169a0-5c20-4ad0-ba27-355f9c2ffb97>
- **NeurIPS Poster**: <https://neurips.cc/virtual/2024/poster/94786>
- **OpenReview Discussion**: <https://openreview.net/forum?id=XEbPJUQzs3&noteId=74fm5Z4Lk6>
- **Code**: <https://github.com/neurodata/prolearn>

---

- **Abstract**:
  - > In  real-world  applications,  the  distribution  of  the  data,  and  our  goals,  evolve over time.  The prevailing theoretical framework for studying machine learning, namely probably approximately correct (PAC) learning, largely ignores time. As a consequence, existing strategies to address the dynamic nature of data and goals exhibit poor real-world performance. This paper develops a theoretical framework called “Prospective Learning” that is tailored for situations when the optimal hypothesis changes over time. In PAC learning, empirical risk minimization (ERM) is known to be consistent. We develop a learner called Prospective ERM, which returns a sequence of predictors that make predictions on future data. We prove that the risk of prospective ERM converges to the Bayes risk under certain assumptions on the stochastic process generating the data. Prospective ERM, roughly speaking, incorporates time as an input in addition to the data. We show that standard ERM as done in PAC learning, without incorporating time, can result in failure to learn when distributions are dynamic. Numerical experiments illustrate that prospective ERM can learn synthetic and visual recognition problems constructed from MNIST and CIFAR-10. Code at <https://github.com/neurodata/prolearn>.
- **Character**: delivers the algorithm the 2022 paper left open — models data as a stochastic process $\{Z_t\}$ rather than i.i.d. samples
- **Prospective Risk**:
  $$R_t(h) = \mathbb{E}[\bar{\ell}_t(h, Z) \mid z_{\leq t}]$$
  where $\bar{\ell}_t$ aggregates future losses via a limsup — error measured over all future time, not just the next step
- **Prospective ERM — the algorithm**:
  - Incorporate time as input via sinusoidal embeddings: $\phi(t) = (\sin(\omega_1 t), \ldots, \sin(\omega_{d/2} t), \cos(\omega_1 t), \ldots, \cos(\omega_{d/2} t))$, $\omega_i = \pi/i$
  - Concatenate $\phi(s)$ with $x_s$, train network to predict $y_s$ — no need to explicitly maintain infinite predictor sequences
  - Train with cross-entropy; evaluate with zero-one error
  - Note: *where* you concatenate matters — before softmax vs. directly to image gives ~0.2 risk difference on CIFAR-10
- **Theorem 1**: Prospective ERM converges to Bayes risk under (1) consistency of expanding hypothesis classes and (2) uniform concentration of empirical limsup estimate
- **Proposition 1**: Standard time-agnostic ERM provably fails even in the *weak* sense — not just suboptimal, but broken in principle
- **Four scenarios**:

  | Scenario | Data structure | Standard ERM | Prospective ERM |
  |---|---|---|---|
  | IID | stationary | works | reduces to PAC |
  | Independent, non-identical | distribution shifts | fails | converges |
  | Dependent, non-stationary | Markov chain | chance | converges |
  | Action-dependent | MDP | — | Q-learning variant |

- **Experimental tasks**:
  - *Scenario 2*: 4 tasks from overlapping MNIST/CIFAR-10 class subsets (1-5, 4-7, 6-9, 8-10); cycle every 10 timesteps deterministically
  - *Scenario 3*: same 4 tasks but governed by a hierarchical HMM — two Markov chains (one over {task1,task2}, one over {task3,task4}) switching every 10 steps; **no stationary distribution** → time-agnostic ERM provably broken
  - 50k samples total; train on first $t$ steps, evaluate prospective risk on remainder
- **”Isn't this just transformers with positional embeddings?”**
  - Same sinusoidal encoding, but fundamentally different purpose:
    - Transformer positional embeddings: tell the model *where in the sequence* a token is within a fixed context window; distribution is still assumed stationary; training on shuffled data would work just as well
    - Prospective ERM time embedding: $t$ is a signal about *which distribution the data is coming from right now*; the model learns $P(y \mid x, t)$ where the optimal decision boundary shifts with wall-clock time
  - The architecture contribution is thin — it's a proof-of-concept that the theoretical framework is operationalizable
  - The real contributions are: (1) formal definition of prospective risk, (2) convergence proof for time-conditioned ERM, (3) impossibility result for time-agnostic ERM
- **Open question**: whether strong and weak prospective learnability are equivalent; sample complexity bounds remain future work
- **Takeaways (practitioner)**:
  - “Just add time as a feature” is basically the answer — but the paper justifies *why* this works and *when* it's guaranteed to converge
  - The architecture is trivial; the non-trivial part is that time-agnostic ERM is provably broken on Markov-structured non-stationarity, not just worse
  - Doesn't scale to real non-stationarity yet — tasks are hand-constructed with known switching structure; the hard open problem is when you don't know the dynamics

## [2025] Prospective Learning in Retrospect - [paper](https://arxiv.org/abs/2507.07965)

- **Date**: 2025-10-28
- **Arxiv**: <https://arxiv.org/abs/2507.07965>
- **Paperpile**: <https://app.paperpile.com/view/?id=60c02eab-3fa7-458d-acf9-cc4b174770ec>
- **Code**: <https://github.com/neurodata/prolearn2>

---

- **Abstract**:
  - > In most real-world applications of artificial intelligence, the distributions of the data and the goals of the learners tend to change over time. The Probably Approximately Correct (PAC) learning framework, which underpins most machine learning algorithms, fails to account for dynamic data distributions and evolving objectives, often resulting in suboptimal performance. Prospective learning is a recently introduced mathematical framework that overcomes some of these limitations. We build on this framework to present preliminary results that improve the algorithm and numerical results, and extend prospective learning to sequential decision-making scenarios, specifically foraging. Code is available at: <https://github.com/neurodata/prolearn2>.
- **Character**: consolidation + extension paper — three contributions over the 2024 work; explicitly labeled "preliminary results"
- **Contribution 1 — Empirical improvements to Prospective-MLP**:
  - Irregular/Poisson-sampled data: Prospective-MLP handles it; time-agnostic baselines degrade
  - Time embedding choice matters for different process types:
    - Fourier embeddings → best for periodic processes
    - Monomial $\phi_m(t) = (t, t^2, \ldots, t^d)$ → best for linear/infinite-task processes
  - Online/streaming training works but needs ~10x more samples than batch (250 vs. 2500 to reach same performance)
- **Contribution 2 — Prospective Forests** (tree-based learners for tabular data):
  - Prospective CART: greedy tree minimizing weighted future loss
  - Prospective GBTs: ensemble boosting with prospective objective
  - GBTs converge *faster* than Prospective-MLP toward Bayes risk; time-agnostic GBTs fail
- **Contribution 3 — Prospective Foraging** (extension to sequential decision-making):
  - **Task**: 1×7 linear track, two reward patches A and B; rewards alternate every 10 steps, decay exponentially after activation; agent takes ≥3 steps to travel between patches; **single lifetime, no resets**
  - Optimal strategy: leave current patch *before* it depletes, arrive at next patch *when rewards peak* — requires pure anticipation, not reaction
  - **Architecture**: actor-critic, both sharing a 128-unit ReLU hidden layer then splitting into separate heads; time embedding $\phi(t)$ concatenated to state input before shared layer
  - **Critic**: input $x_t + \phi(t)$, output scalar $V(x_t, t)$; trained with TD error via GAE: $\delta_t = r_t + \gamma V(x_{t+1}, t+1) \cdot \text{mask}_t - V(x_t, t)$
  - **Actor**: input $x_t + \phi(t)$, output policy $\pi(a \mid x_t, t)$; trained with policy gradient using critic advantages + entropy regularization
  - Both updated with Adam + gradient clipping; trained with standard on-policy RL (real environment rewards, no supervised signal)
  - **Results**: with time embedding → converges to Bayes risk; without time embedding → suboptimal but still beats retrospective actor-critic; standard actor-critic fails
- **Critical question: is this actually non-stationary?**
  - The periodic switching (every 10 steps, deterministic) means the full process $P(y \mid x, t)$ is **stationary in a lifted space** — conditioning on $t \bmod 10$ gives a fixed distribution
  - Standard ERM fails not because the problem is fundamentally non-stationary, but because it *ignores time entirely*; a sufficiently expressive time-agnostic model with enough data could implicitly learn the period
  - The Scenario 3 hierarchical HMM is closer to genuine non-stationarity — no stationary distribution by construction — but even there the switching rules are fixed and learnable
  - **The truly hard case** (unstructured, unpredictable drift) is not addressed anywhere in this paper series — these papers are best understood as *"what you can do when non-stationarity has learnable structure"*, not a general solution to distribution shift

## [2025] Optimal control of the future via prospective learning with control

- **Date**: 2026-03-31
- **Arxiv**: <https://arxiv.org/abs/2511.08717>
- **Paperpile**: <https://app.paperpile.com/view/?id=3f1af382-e2c5-4ae0-bc3d-294d3e142a8c>

---

- **Character**: biggest conceptual jump in the series — prior papers assumed decisions have no impact on the world; this one drops that assumption
- **PL-C vs PL+C**:
  - *PL-C* (prior work): learner observes $(x_t, y_t, t)$ fully — no behavioral impact, like supervised classification
  - *PL+C* (this paper): learner only observes rewards at *visited* locations — actions determine what you observe, creating a missing data / counterfactual problem
- **Loss functions**:
  - Instantaneous loss — negative reward at next position:
    $$\bar{\ell}(h_s(x_s, s),\, y_{s+1}) = -y_{s+1}(x_{s+1})$$
  - Prospective (cumulative) loss — weighted sum of future instantaneous losses:
    $$\sum_{s > t} w_{s-t}\, \bar{\ell}(h_s(x_s, s),\, y_{s+1}), \quad \sum_t w_t = 1,\; w_t \in [0,1]$$
  - Theoretical objective (intractable — requires future data):
    $$\hat{h}^t = \arg\max_{h \in \mathcal{H}_t} \int \sum_{s>t} w_{s-t} \cdot y_{s+1}^h(x_{s+1}^h)\; d\mathbb{P}_{Z \mid z_{\leq t}}$$
  - Empirical objective (tractable — sums over observed past):
    $$\hat{h}^t \approx \arg\min_{h^t \in \mathcal{H}_t} \sum_{t'=1}^{t} \sum_{s > t'} w_{s-t'} \cdot y_{s+1}(x_{s+1})$$
  - Counterfactual handling — only observe rewards where you actually went:
    $$\tilde{y}_s(x_s) = \begin{cases} y_s(x_s) & \text{if } x'_s = x_s \text{ (visited)}\\ \hat{y}_s(x'_s) & \text{if } x'_s \neq x_s \text{ (estimated)} \end{cases}$$
- **Algorithm — ProForg (Prospective Foraging)**:
  - Two regressors trained with supervised learning: $\hat{g}^i$ estimates instantaneous reward at $(x_s, s)$; $\hat{g}^p$ estimates cumulative prospective loss from $(x_s, s)$ forward
  - **Warm-start**: pretrain both regressors on batch data; missing counterfactual rewards filled with surrogate estimates
  - **Online**: enumerate all length-$H$ trajectories, score with $\hat{g}^i + \hat{g}^p$, execute first action of best trajectory, retrain regressors incrementally
- **Why not Markovian / why standard RL fails**:
  - Markov assumption: $P(x_{t+1} \mid x_t, a_t, \ldots) = P(x_{t+1} \mid x_t, a_t)$ — future depends only on current state
  - Bellman backup: $Q(x,a,t) = r(x,t) + \gamma \max_{a'} Q(x',a',t+1)$ — requires bootstrap targets to be stable
  - Here reward at patch A depends on *when* you're there, not just that you're there: $r_t(x) = f(x,t)$, not $f(x)$
  - You could augment state with time $(x_t, t)$ to recover Markov property — but Bellman still assumes transition dynamics and reward function are **stationary** across time; they're not here (rewards decay, patches switch)
  - Time-aware FQI and SAC still fail: Bellman backups on a non-stationary reward landscape chase moving bootstrap targets
- **Why ProForg works — algorithm step by step**:
  - Never bootstraps from its own predictions — only does supervised regression on actual observed rewards
  - **Step 1 — learn instantaneous reward model**: fit $\hat{g}^i(x, t) \approx r(x, t)$ via regression on observed $(x_s, t_s, r_s)$ tuples; impute unvisited positions with regressor
  - **Step 2 — learn cumulative reward model**: fit $\hat{g}^p(x, t) \approx \sum_{s>t} w_{s-t} \cdot r(x_s, s)$ via regression on observed trajectories; no value iteration
  - **Step 3 — plan by enumeration**: at each timestep, score all length-$H$ trajectories as $\hat{g}^i(x_1, t) + \hat{g}^p(x_H, t+H)$; execute first action of best trajectory; retrain both regressors on new observation; repeat
  - Non-stationarity becomes just "time $t$ is a feature in your regression" — no special handling needed as long as reward dynamics are learnable
- **Key departure from RL**: no Markov assumption, no episodic resets, no exploration policy — treats control as supervised learning with explicit temporal reasoning; counterfactuals estimated, not explored
- **Theorem 5.1**: ERM converges to Bayes-optimal policy under expressiveness + concentration assumptions, without requiring Markovian structure
- **Results** (same 1×7 foraging task):
  - ProForg converges to Bayes-optimal regret orders of magnitude faster than RL baselines
  - Time-aware Fitted Q-Iteration and Soft Actor-Critic either fail or converge 4-20x slower
  - Online ProForg: ~20 timesteps post-warmup; offline: ~80 timesteps
  - Combining $\hat{g}^i$ and $\hat{g}^p$ outperforms either regressor alone
