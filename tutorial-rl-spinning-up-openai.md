# [Spinning Up in RL, OpenAI](https://spinningup.openai.com/)

- **Created**: 2025-12-13
- **Last Updated**: 2026-01-05
- **Status**: `Done`

---

- [[papers-rl]]

---

- [x] Overview: <https://spinningup.openai.com/en/latest/user/algorithms.html>
- [x] Key Concepts in RL: <https://spinningup.openai.com/en/latest/spinningup/rl_intro.html>
- [x] A Taxonomy of RL Algorithms: <https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html>
- [x] Intro to Policy Optimization: <https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html>
- [x] Algorithms
  - [x] VPG: <https://spinningup.openai.com/en/latest/algorithms/vpg.html>
  - [x] TRPO: <https://spinningup.openai.com/en/latest/algorithms/trpo.html>
  - [x] PPO: <https://spinningup.openai.com/en/latest/algorithms/ppo.html>
  - [x] DDPG: <https://spinningup.openai.com/en/latest/algorithms/ddpg.html>
  - [x] TD3: <https://spinningup.openai.com/en/latest/algorithms/td3.html>
  - [x] SAC: <https://spinningup.openai.com/en/latest/algorithms/sac.html>

---

## [0. RL Algorithms Covered](https://spinningup.openai.com/en/latest/user/algorithms.html)

1. Vanilla Policy Gradients (VPG)
2. Trust Region Policy Optimization (TRPO)
3. Proximal Policy Optimization (PPO)
4. Deep Deterministic Policy Gradient (DDPG)
5. Twin Delayed DDPG (TD3)
6. Soft Actor-Critic (SAC)

## [1. Key Concepts in RL](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html)

- **State $s$**: Complete description of the world/environment. There is no information about the world which is hidden from the state.
- **Observation $o$**: What the agent observes about the state.
  - **(a) Fully Observed**: The agent can observe the state completely.
  - **(b) Partially Observed**: The agent can only see a partial observation of the state.
- **Action Space**: Set of all valid actions in a given environment.
  - **(a) Discrete action space**
  - **(b) Continuous action space**
- **Policy $\pi$**: Rule used by an agent to decide what actions to take.
  - **(a) Deterministic Policy** $a_t = \mu_\theta(s_t)$
  - **(b) Stochastic Policy**: $a_t \sim \pi_\theta( \cdot | s_t)$
    - **Categorical Policy**: For discrete action spaces. Outputs a categorical distribution over actions to sample from.
    - **Diagonal Gaussian Policy**: For continuous actions spaces. "Diagonal" in the sense that covariance matrix of the multivariate gaussian distribution is a diagonal matrix. Mean actions, $\mu_\theta(s)$, is output by a neural net. Log std of the actions, $\log \sigma_\theta(s)$, is either output by the neural net or are standalone params. Action is sampled from the gaussian distribution with this mean and std $a \sim \mathcal{N}(\mu_\theta(s), \sigma_\theta(s))$, implemented as $a = \mu_{\theta}(s) + \sigma_{\theta}(s) \odot z$ where $z \sim \mathcal{N}(0, I)$.
- **Trajectory/Rollout/Episode $\tau$**: Sequence of states and actions in the world: $\tau = (s_0, a_0, s_1, a_1, ...)$.
  - Initial state $s_0 \sim \rho_0(.)$.
  - State transitions depend only the previous state and action.
  - State transitions can either be deterministic ($s_{t+1} = f(s_t, a_t)$) or stochastic ($s_{t+1} \sim P(\cdot|s_t, a_t)$) depending on the environment.
- **Reward Function $R$**: $r_t = R(s_t, a_t, s_{t+1})$, although frequently simplified to $r_t = R(s_t, a_t)$.
- **Return $R(\tau)$**: The goal of the agent is to maximize some notion of cumulative reward $R(\tau)$ over a trajectory $\tau$.
  - **Finite-horizon undiscounted return**: $R(\tau) = \sum_{t=0}^T r_t$
  - **Infinite-horizon discounted return**: $R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t$, where $\gamma \in (0,1)$ is the discount factor.
- **The RL Problem**: Select a policy which **maximizes expected return** when the agent acts according to it.
  - **Probability of a $T$-step trajectory** $P(\tau|\pi) = \rho_0 (s_0) \prod_{t=0}^{T-1} \pi(a_t | s_t) P(s_{t+1} | s_t, a_t)$.
  - **Expected return** $J(\pi) = \underset{\tau\sim \pi}{\mathbb{E}}[R(\tau)] = \int_{\tau} P(\tau|\pi) R(\tau)$.
  - **Optimal policy** $\pi^* = \arg \max_{\pi} J(\pi)$. This is the central optimization problem in RL.
- **On-Policy**: Old data not used $\rightarrow$ weaker sample efficiency, but more stability.
  - **Algorithms**: VPG, TRPO, PPO.
- **Off-Policy**: Reuses old data by exploiting Bellman’s equations for optimality -> sample efficient, but can be unstable.
  - **Algorithms**: Q-learning, DDPG, TD3, SAC.
  - > But problematically, there are no guarantees that doing a good job of satisfying Bellman’s equations leads to having great policy performance. Empirically one can get great performance—and when it happens, the sample efficiency is wonderful—but the absence of guarantees makes algorithms in this class potentially brittle and unstable. TD3 and SAC are descendants of DDPG which make use of a variety of insights to mitigate these issues.
- **Value Function**: The expected return if you start in that state or state-action pair, and then act according to a particular policy forever after.
  - **(a) On-Policy Action-Value Function**: $Q^{\pi}(s,a) = \underset{\tau \sim \pi}{\mathbb{E}}[R(\tau) | s_0 = s, a_0 = a]$. The expected return if you start in state $s$, take an arbitrary action $a$ (which may not have come from the policy), and then always act according to policy $\pi$.
  - **(b) On-Policy Value Function**: $V^{\pi}(s) = \underset{\tau \sim \pi}{\mathbb{E}}[R(\tau) | s_0 = s] = \underset{a \sim \pi}{\mathbb{E}}[Q^{\pi}(s,a)]$. The expected return if you start in state $s$ and always act according to policy $\pi$.
  - **(c) Optimal Action-Value Function**: $Q^*(s,a) = \max_\pi Q^{\pi}(s,a)$. The expected return if you start in state $s$, take an arbitrary action $a$ (which may not have come from the policy), and then always act according to the optimal policy.
  - **(d) Optimal Value Function**: $V^*(s) = \max_\pi V^{\pi}(s) = \max_a Q^*(s,a)$. The expected return if you start in state $s$ and always act according to the optimal policy.
- **Advantage Function**: $A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s)$. Captures the **relative advantage of an action**.
  - Describes how much better it is to take a specific action $a$ in state $s$, over randomly sampling an action according to the distribution $\pi(\cdot|s)$, assuming you act according to $\pi$ forever after.
  - We don't always need to know how good an action is in an absolute sense, but only how much better it is than others on average.
- **Optimal Action**: $a^*(s) = \arg \max_a Q^* (s,a)$.
- **Bellman Equations**: Basic idea - the value of your starting point is the reward you expect to get from being there, plus the value of wherever you land next.
  - **Bellman Equations for On-Policy Value Functions**:
    - $Q^{\pi}(s,a) = \underset{s' \sim P}{\mathbb{E}}[{r(s,a) + \gamma \underset{a'\sim \pi}{\mathbb{E}}[{Q^{\pi}(s',a')}]}]$
    - $V^{\pi}(s) = \underset{a \sim \pi; s'\sim P}{\mathbb{E}}[{r(s,a) + \gamma V^{\pi}(s')}]$
  - **Bellman Equations for Optimal Value Functions**:
    - $Q^*(s,a) = \underset{s'\sim P}{\mathbb{E}}[{r(s,a) + \gamma \max_{a'} Q^*(s',a')}]$
    - $V^*(s) = \max_a \underset{s'\sim P}{\mathbb{E}}[{r(s,a) + \gamma V^*(s')}]$
- **Markov Decision Processes (MDP)**:
  - System obeys the Markov property - transitions only depend on the most recent state and action, and no prior history.
  - $\langle S, A, R, P, \rho_0 \rangle$, where
    - $S$ is the set of all valid states,
    - $A$ is the set of all valid actions,
    - $R : S \times A \times S \to \mathbb{R}$ is the reward function, with $r_t = R(s_t, a_t, s_{t+1})$,
    - $P : S \times A \to \mathcal{P}(S)$ is the transition probability function, with $P(s_{t+1}|s_t,a_t)$ being the probability of transitioning into state $s_{t+1}$ if you start in state $s_t$ and take action $a_t$,
    - $\rho_0$ is the starting state distribution, that is, $s_0 \sim \rho_0(\cdot)$.
- **Model-Based vs Model-Free RL**:
  - Whether the agent has access to (or learns) a model of the environment (a function which predicts state transitions and rewards).
  - > The main upside to having a model is that it **allows the agent to plan by thinking ahead**, seeing what would happen for a range of possible choices, and explicitly deciding between its options. Agents can then distill the results from planning ahead into a learned policy. A particularly famous example of this approach is AlphaZero. When this works, it can result in a **substantial improvement in sample efficiency over methods that don’t have a model**.
  - > The main downside is that **a ground-truth model of the environment is usually not available to the agent**. If an agent wants to use a model in this case, it has to learn the model purely from experience, which creates several challenges. The biggest challenge is that **bias in the model can be exploited by the agent, resulting in an agent which performs well with respect to the learned model, but behaves sub-optimally (or super terribly) in the real environment**. Model-learning is fundamentally hard, so even intense effort—being willing to throw lots of time and compute at it—can fail to pay off.

## [2. A Taxonomy of RL Algorithms](https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)

### RL Axes

1. **Does the agent have access to or learn a model of the environment?**
   1. Model-Based RL
   2. Model-Free RL
2. **What do we learn?**
   1. Policies (deterministic or stochastic)
   2. Action-value functions (Q-functions)
   3. Value functions
   4. World/environment models

### RL Algorithms

- **Model-Based RL**
  - **Model Given**
    - AlphaZero
  - **Model Learned**
    - World Models
    - I2A
    - MBMF
    - MBVE
- **Model-Free RL**
  - **Policy Optimization (On-Policy)**
    - Vanilla Policy Gradient
    - TRPO
    - PPO
    - A2C / A3C
  - **Q-Learning (Off-Policy)**
    - DQN
    - C51
    - QR-DQN
    - HER
  - **Hybrid**
    - DDPG
    - TD3
    - SAC

![RL Algorithms Taxonomy](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)

## [3. Intro to Policy Optimization](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)

### Deriving the Simplest Policy Gradient

We aim to maximize the expected return $J(\pi_\theta) = \underset{\tau\sim \pi_\theta}{\mathbb{E}}[R(\tau)] = \int_{\tau} P(\tau|\pi_\theta) R(\tau)$,
where

- $\pi_\theta$ is the policy,
- $\theta$ denotes the parameters of the policy,
- $\tau$ is the trajectory sampled from the policy.

by optimizing the parameters of the policy by gradient ascent: $\theta_{k+1} = \theta_k + \alpha \left. \nabla_{\theta} J(\pi_{\theta}) \right|_{\theta_k}$.

> The gradient of policy performance, **$\nabla_{\theta} J(\pi_{\theta})$, is called the policy gradient**, and algorithms that optimize the policy this way are called **policy gradient algorithms**. (Examples include Vanilla Policy Gradient and TRPO. PPO is often referred to as a policy gradient algorithm, though this is slightly inaccurate.)

> To actually use this algorithm, we need an expression for the policy gradient which we can numerically compute. This involves two steps: 1) deriving **the analytical gradient of policy performance, which turns out to have the form of an expected value**, and then 2) forming a sample estimate of that expected value, which can be computed with data from a finite number of agent-environment interaction steps.

- **(1) Probability of a trajectory**: For a trajectory $\tau = (s_0, a_0, ..., s_{T+1})$, $P(\tau|\theta) = \rho_0 (s_0) \prod_{t=0}^{T} \pi_{\theta}(a_t |s_t) P(s_{t+1}|s_t, a_t)$.
- **(2) Log-probability of a trajectory**: $\log P(\tau|\theta) = \log \rho_0 (s_0) + \sum_{t=0}^{T} \bigg( \log \pi_{\theta}(a_t |s_t) + \log P(s_{t+1}|s_t, a_t) \bigg)$.
- **(3) Gradient of log-probability of a trajectory**: $\nabla_\theta \log P(\tau|\theta) = \sum_{t=0}^{T} \nabla_\theta \log \pi_{\theta}(a_t |s_t)$ (since the initial state and state transition functions don't depend on $\theta$).
- **(4) Log-derivative trick**: Since the derivative of $\log x$ is $1/x$, we can rewrite $\nabla_{\theta} P(\tau | \theta) = P(\tau | \theta) \nabla_{\theta} \log P(\tau | \theta)$.
- **(5) Policy gradient derivation** (putting everything together):
  - $\nabla_\theta J(\pi_\theta) = \nabla_\theta \underset{\tau\sim \pi_\theta}{\mathbb{E}}[R(\tau)]$
  - $\nabla_\theta J(\pi_\theta) = \nabla_\theta \int_{\tau} P(\tau|\pi_\theta) R(\tau)$
  - $\nabla_\theta J(\pi_\theta) = \int_{\tau} \nabla_\theta P(\tau|\pi_\theta) R(\tau)$
  - $\nabla_\theta J(\pi_\theta) = \int_{\tau} P(\tau | \theta) \nabla_{\theta} \log P(\tau | \theta) R(\tau)$ (log-derivative trick)
  - $\nabla_\theta J(\pi_\theta) = \underset{\tau\sim \pi_\theta}{\mathbb{E}}\bigg[\nabla_{\theta} \log P(\tau | \theta) R(\tau)\bigg]$
  - $\nabla_\theta J(\pi_\theta) = \underset{\tau\sim \pi_\theta}{\mathbb{E}}\bigg[\sum_{t=0}^{T} \nabla_\theta \log \pi_{\theta}(a_t |s_t) R(\tau)\bigg]$

The final expectation above can be estimated with a sample mean over a set of trajectories collected by running the policy $\pi_\theta$ (which makes it on-policy) in the environment.

> A loss function usually evaluates the performance metric that we care about. Here, we care about expected return, $J(\pi_{\theta})$, but our “loss” function does not approximate this at all, even in expectation. This “loss” function is only useful to us because, when evaluated at the current parameters, with data generated by the current parameters, it has the negative gradient of performance.
>
> But after that first step of gradient descent, there is no more connection to performance. This means that minimizing this “loss” function, for a given batch of data, has no guarantee whatsoever of improving expected return. You can send this loss to $-\infty$ and policy performance could crater; in fact, it usually will. Sometimes a deep RL researcher might describe this outcome as the policy “overfitting” to a batch of data. This is descriptive, but should not be taken literally because it does not refer to generalization error.
>
> We raise this point because it is common for ML practitioners to interpret a loss function as a useful signal during training—”if the loss goes down, all is well.” In policy gradients, this intuition is wrong, and you should only care about average return. The loss function means nothing.

### Fix 1: Reward-to-go Policy Gradient

$\nabla_\theta J(\pi_\theta) = \underset{\tau\sim \pi_\theta}{\mathbb{E}}\bigg[\sum_{t=0}^{T} \nabla_\theta \log \pi_{\theta}(a_t |s_t) R(\tau)\bigg]$

This final expression from the policy gradient derivation leads to the log-probabilities of each action being pushed up in proportion to $R(\tau)$, the sum of all rewards ever obtained. But each action only impacts future rewards, not the past rewards. This can be fixed by making sure for each timestep in summation, we only include the future rewards.

$\nabla_\theta J(\pi_\theta) = \underset{\tau\sim \pi_\theta}{\mathbb{E}}\bigg[\sum_{t=0}^{T} \nabla_\theta \log \pi_{\theta}(a_t |s_t) \hat{R}_t\bigg]$, where $\hat{R}_t$ is the sum of rewards from time $t$.

> But how is this better? **A key problem with policy gradients is how many sample trajectories are needed to get a low-variance sample estimate** for them. The formula we started with included terms for reinforcing actions proportional to past rewards, all of which had zero mean, but nonzero variance: as a result, they would just add noise to sample estimates of the policy gradient. By removing them, we reduce the number of sample trajectories needed.

### Fix 2: Baselines in Policy Gradients

**Expected Grad-Log-Prob (EGLP) Lemma**:

- $\int_x P_\theta(x) = 1$ (any normalized probability distribution)
- $\nabla_\theta \int_x P_\theta(x) = 0$ (taking gradient on both sides)
- $\int_x \nabla_\theta P_\theta(x) = 0$ (rearranging)
- $\int_x P_\theta(x) \nabla_\theta \log P_\theta(x) = 0$ (log-derivative trick)
- **$\underset{x \sim P_\theta}{\mathbb{E}}\bigg[\nabla_\theta \log P_\theta(x)\bigg] = 0$ (EGLP Lemma)**

As a consequence of the above lemma, for any function $b$ which depends only on state (and not on action), $\underset{a_t \sim \pi_{\theta}}{\mathbb{E}}\bigg[{\nabla_{\theta} \log \pi_{\theta}(a_t|s_t) b(s_t)}\bigg] = 0$.

As a result, the following doesn't change the expectation:

$\nabla_\theta J(\pi_\theta) = \underset{\tau\sim \pi_\theta}{\mathbb{E}}\bigg[\sum_{t=0}^{T} \nabla_\theta \log \pi_{\theta}(a_t |s_t) (\hat{R}_t - b(s_t))\bigg]$.

Any function $b$ used in the above manner is called a **baseline**. A common choice of baseline is the **on-policy value function $V^{\pi}(s_t)$**, which is the average an agent gets if it starts at state $s_t$ and then acts according to the policy $\pi$ for the rest of its life.

$V^{\pi}(s_t)$ cannot be computed exactly and is approximated with a neural network, $V_\phi(s_t)$, updated concurrently with the policy (so that the value network always approximates the value function of the most recent policy), typically with a mean-squared-error objective between $V_\phi(s_t)$ and $\hat{R}_t$.

> Empirically, the choice $b(s_t) = V^{\pi}(s_t)$ has the desirable effect of reducing variance in the sample estimate for the policy gradient. This results in faster and more stable policy learning. It is also appealing from a conceptual angle: it encodes the intuition that if an agent gets what it expected, it should “feel” neutral about it.

### Policy Gradient General Form

$\nabla_\theta J(\pi_\theta) = \underset{\tau\sim \pi_\theta}{\mathbb{E}}\bigg[\sum_{t=0}^{T} \nabla_\theta \log \pi_{\theta}(a_t |s_t) \Phi_t \bigg]$,
where $\Phi_t$ could be any of:

- **Basic**: $\Phi_t = R(\tau)$. Full trajectory return.
- **Reward-to-go**: $\Phi_t = \hat{R}_t$. Only include rewards from time $t$ onwards for action $a_t$, not the past rewards.
- **Reward-to-go with baseline**: $\Phi_t = \hat{R}_t - b(s_t)$. Baseline to reduce variance (and thus improve sample efficiency). Typically this is the value function $V^{\pi}(s_t)$ that provides the average expected return from a given state.
- **On-policy value function**: $\Phi_t = Q^{\pi_{\theta}}(s_t, a_t)$. Proof in <https://spinningup.openai.com/en/latest/spinningup/extra_pg_proof2.html>.
- **Advantage function**: $\Phi_t = A^{\pi_\theta}(s_t,a_t) = Q^{\pi_\theta}(s_t,a_t) - V^{\pi_\theta}(s_t)$

## 4. Algorithms

### [VPG: Vanilla Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/vpg.html)

- **On-policy**
- **Action spaces**: discrete, continuous
- **Intuition**: Push up the probabilities of actions that lead to higher return, and push down the probabilities of actions that lead to lower return.
- Policy Gradient general form with Advantage function: $\Phi_t = A^{\pi_\theta}(s_t,a_t) = Q^{\pi_\theta}(s_t,a_t) - V^{\pi_\theta}(s_t)$.

![VPG pseudocode](https://spinningup.openai.com/en/latest/_images/math/262538f3077a7be8ce89066abbab523575132996.svg)

### [TRPO: Trust Region Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/trpo.html)

- **On-policy**
- **Action spaces**: discrete, continuous
- **TRPO: VPG, but with a KL-divergence constraint on how far the new policy can jump from the old policy**.
  - > even seemingly small differences in parameter space can have very large differences in performance—so a single bad step can collapse the policy performance. This makes it dangerous to use large step sizes with vanilla policy gradients, thus hurting its sample efficiency. TRPO nicely avoids this kind of collapse, and tends to quickly and monotonically improve performance.
- **Theoretical TRPO Update**: $\theta_{k+1} = \arg \max_{\theta} {\mathcal L}(\theta_k, \theta) \; \text{s.t.} \; \bar{D}_{KL}(\theta || \theta_k) \leq \delta$, where
  - **Surrogate Advantage** ${\mathcal L}(\theta_k, \theta) = \underset{s,a \sim \pi_{\theta_k}}{\mathbb E}{\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a)}$ is a measure of how policy $\pi_{\theta}$ performs relative to the old policy $\pi_{\theta_k}$ using data from the old policy.
  - **Average KL-Divergence** $\bar{D}_{KL}(\theta || \theta_k) = \underset{s \sim \pi_{\theta_k}}{\mathbb E}{D_{KL}\left(\pi_{\theta}(\cdot|s) || \pi_{\theta_k} (\cdot|s) \right)}$ between policies across states visited by the old policy.
- **At $\theta = \theta_k$**,
  - **(a) Surrogate advantage = 0**.
    - ${\mathcal L}(\theta_k, \theta_k) = \underset{s,a \sim \pi_{\theta_k}}{\mathbb E} \left[A^{\pi_{\theta_k}}(s,a)\right]$
    - ${\mathcal L}(\theta_k, \theta_k) = \underset{s,a \sim \pi_{\theta_k}}{\mathbb E} \left[Q^{\pi_{\theta_k}}(s,a) - V^{\pi_{\theta_k}}(s)\right]$
    - ${\mathcal L}(\theta_k, \theta_k) = \underset{s,a \sim \pi_{\theta_k}}{\mathbb E} \left[Q^{\pi_{\theta_k}}(s,a)\right] - \underset{s \sim \pi_{\theta_k}}{\mathbb E} \left[V^{\pi_{\theta_k}}(s)\right]$
    - ${\mathcal L}(\theta_k, \theta_k) = \underset{s \sim \pi_{\theta_k}}{\mathbb E} \left[V^{\pi_{\theta_k}}(s)\right] - \underset{s \sim \pi_{\theta_k}}{\mathbb E} \left[V^{\pi_{\theta_k}}(s)\right]$
    - ${\mathcal L}(\theta_k, \theta_k) = 0$
  - **(b) Gradient of surrogate advantage wrt $\theta$ = vanilla policy gradient**.
    - $\nabla_{\theta} \mathcal{L}(\theta_k, \theta) \Big|_{\theta=\theta_k} = \mathbb{E}_{s,a \sim \pi_{\theta_k}} \left[ \frac{\nabla_{\theta} \pi_{\theta}(a|s) \Big|_{\theta=\theta_k}}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a) \right]$
    - $\nabla_{\theta} \mathcal{L}(\theta_k, \theta) \Big|_{\theta=\theta_k} = \mathbb{E}_{s,a \sim \pi_{\theta_k}} \left[ \frac{\pi_{\theta_k}(a|s) \nabla_{\theta} \log \pi_{\theta}(a|s) \Big|_{\theta=\theta_k}}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a) \right]$ (log-derivative trick)
    - $\nabla_{\theta} \mathcal{L}(\theta_k, \theta) \Big|_{\theta=\theta_k} = \mathbb{E}_{s,a \sim \pi_{\theta_k}} \left[ \nabla_{\theta} \log \pi_{\theta}(a|s) \Big|_{\theta=\theta_k} A^{\pi_{\theta_k}}(s,a) \right]$ (vanilla policy gradient)
  - **(c) KL-divergence = 0**.
  - **(d) Gradient of KL-divergence wrt $\theta$ = 0** since at $\theta = \theta_k$, $D_{KL}(\theta || \theta_k)$ is at a global minimum.
- **Approximating the theoretical TRPO update**:
  - **Taylor expansion**: ${\mathcal L}(\theta_k, \theta) \approx g^T (\theta - \theta_k)$ (follows from (b) above at $\theta = \theta_k$).
  - **Taylor expansion**: $\bar{D}_{KL}(\theta || \theta_k) \approx \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k)$ (follows from (c) and (d) above at $\theta = \theta_k$).
  - And therefore, $\theta_{k+1} = \arg \max_{\theta} \; g^T (\theta - \theta_k) \; \text{s.t.} \; \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k) \leq \delta$
  - **Lagrangian duality**: $\theta_{k+1} = \theta_k + \sqrt{\frac{2 \delta}{g^T H^{-1} g}} H^{-1} g$
  - **Backtracking line search**: The above might not  satisfy the KL constraint or actually improve the surrogate advantage due to the approximation errors introduced by Taylor expansion. TRPO adds backtracking line search, $\theta_{k+1} = \theta_k + \alpha^j \sqrt{\frac{2 \delta}{g^T H^{-1} g}} H^{-1} g$, where $\alpha \in (0,1)$ is the backtracking coefficient, and $j$ is the smallest nonnegative integer such that $\pi_{\theta_{k+1}}$ satisfies the KL constraint and produces a positive surrogate advantage.
  - **Conjugate gradient**: To solve for $Hx = g$ and obtain $x = H^{-1}g$ as computing $H^{-1}$ directly is expensive.

![TRPO pseudocode](https://spinningup.openai.com/en/latest/_images/math/5808864ea60ebc3702704717d9f4c3773c90540d.svg)

### [PPO: Proximal Policy Optimization](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

- **On-policy**
- **Action spaces**: discrete, continuous
- **PPO: Same goal as TRPO (taking the biggest possible policy improvement step but not being too far from old policy), but achieved via first-order methods instead of second-order**.
- **PPO Variants**:
  - **(1) PPO-Penalty**: Approximately solves a KL-constrained update like TRPO, but instead of a hard constrait, penalizes KL-divergence in the objective, and automatically adjusts the penalty coefficient over the course of training.
  - **(2) PPO-Clip**: Neither a KL-divergence term in the objective nor a hard constraint. Instead, relies on specialized clipping in the objective function to remove incentives for the new policy to get far from the old policy.
    - Updates policy via $\theta_{k+1} = \arg \max_{\theta} \underset{s,a \sim \pi_{\theta_k}}{{\mathbb E}}\left[L(s,a,\theta_k, \theta)\right]$ by taking multiple gradient steps, where
      - $L(s,a,\theta_k,\theta) = \min\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\;\text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1+\epsilon \right) A^{\pi_{\theta_k}}(s,a)\right)$.
      - $\epsilon$ is a (small) hyperparameter which roughly says how far away the new policy is allowed to go from the old.

![PPO pseudocode](https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg)

### [DDPG: Deep Deterministic Policy Gradient](https://spinningup.openai.com/en/latest/algorithms/ddpg.html)

- **Off-policy**
- **Action spaces**: continuous
- **DDPG: Extends DQN to continuous action spaces**.
  - In Q-learning, if we know the optimal action-value function $Q^*(s,a)$, then the optimal action at a given state can be found as $a^*(s) = \arg \max_a Q^*(s,a)$.
  - The above works for discrete action spaces, but **$\text{argmax}$ is intractable when the action space is continuous**.
  - But, if we can learn a policy $\mu(s)$ such that $a^*(s) = \mu(s)$, then $\max_a Q^*(s,a) \approx Q^*(s,\mu(s))$.
  - And since $a$ is a continuous variable, we can differentiate $Q^*(s,a)$ wrt $a$, and in turn, we can differentiate $Q^*(s,\mu(s))$ wrt the parameters of $\mu$ to learn the policy.
- **Learns both the $Q(s,a)$ function (critic) and the policy $\mu(s)$ (actor) via alternating optimization**.
  - **(1) Q-Learning Side of DDPG**: The parameters of the Q network $Q(s,a)$ can be updated via **gradient descent using Mean-Squared Bellman Error (MSBE) loss** $\underset{(s,a,r,s') \sim {\mathcal D}}{{\mathbb E}}\left[ \Bigg( Q(s,a) - \left(r + \gamma (1 - \text{done}) Q^\text{target}(s', \mu^\text{target}(s')) \right) \Bigg)^2 \right]$, where $D$ is the replay buffer.
  - **(2) Policy-Learning Side of DDPG**: We want to learn a deterministic policy $\mu(s)$ that maximizes $Q(s,\mu(s))$. The parameters of the policy network $\mu(s)$ can be updated via **gradient ascent to maximize $\underset{s \sim {\mathcal D}}{\mathbb E}\left[ Q(s,\mu(s)) \right]$**.
- **Leverages Replay Buffer and Target Network tricks from DQN**.
  - In DDPG, there's both a target Q network and a target policy network.
  - In DQN, the target network is just copied over from the main network every few steps. In DDPG, the target network parameters are updated every step via polyak averaging: $\theta_\text{target} = \rho\theta_\text{target} + (1-\rho)\theta$, where $\theta$ denotes the most recent parameters, $\theta_\text{target}$ denotes the target network parameters, and $\rho$ is a hyperparam between 0 and 1.
- **Exploration vs Exploitation**: The policy network $\mu(s)$ in DDPG outputs a deterministic action value within the continuous action space. Given that the policy is not stochastic, we need to explicitly add some noise to encourage exploration early on, such as uncorrelated Gaussian with zero mean.
  
![DDPG pseudocode](https://spinningup.openai.com/en/latest/_images/math/5811066e89799e65be299ec407846103fcf1f746.svg)

### [TD3: Twin Delayed DDPG](https://spinningup.openai.com/en/latest/algorithms/td3.html)

- **Off-policy**
- **Action spaces**: continuous
- **TD3: Tricks to address brittleness in DDPG**.
  - **Trick 1 ("Twin") - Clipped Double-Q Learning**:
    - **DDPG Failure Mode**: The learned Q-function can dramatically overestimate Q-values.
    - **TD3 Fix**: Learns two Q-functions instead of one, and uses the smaller of the two Q-values to form the targets in the Bellman error loss: $\underset{(s,a,r,s') \sim {\mathcal D}}{{\mathbb E}}\left[ \Bigg( Q(s,a) - \left(r + \gamma (1 - \text{done}) \min(Q^\text{target}_1(s',a'), Q^\text{target}_2(s',a')) \right) \Bigg)^2 \right]$.
  - **Trick 2 ("Delayed") - Delayed Policy Updates**:
    - **DDPG Failure Mode**: Training volatility.
    - **TD3 Fix**: Policy (and target networks) updated less frequently than the Q-function. The paper recommends one policy update for every two Q-function updates.
  - **Trick 3 - Target Policy Smoothing**:
    - **DDPG Failure Mode**: If the Q-function approximator develops an incorrect sharp peak for some actions, the policy will quickly exploit that peak and then have brittle or incorrect behavior.
    - **TD3 Fix**: Regularization via smoothing out the Q-function over similar actions by adding clipped noise to the output of the target policy network.

![TD3 pseudocode](https://spinningup.openai.com/en/latest/_images/math/b7dfe8fa3a703b9657dcecb624c4457926e0ce8a.svg)

### [SAC: Soft Actor-Critic](https://spinningup.openai.com/en/latest/algorithms/sac.html)

- **Off-policy**
- **Action spaces**: discrete, continuous
- **SAC: TD3 but with stochastic policy instead of deterministic policy**. Differences from TD3:
  - **(1) Stochastic Policy**: Unlike TD3, policy is stochastic instead of deterministic. For continuous action spaces, the policy outputs mean and std action values to sample from via the reparametrization trick.
  - **(2) No Target Policy Smoothing**: Since the actions are sampled from a stochastic policy already in SAC, no external stochastic noise needs to be added to smooth the action values output by the policy as done in TD3.
  - **(3) Entropy Regularization**: Instead of the goal being just to maximize reward, a weighted entropy regularization term is added.
  - **(4) Actions sampled from current policy**: Unlike in TD3, the next-state actions used in the target come from the current policy instead of a target policy.

![SAC pseudocode](https://spinningup.openai.com/en/latest/_images/math/c01f4994ae4aacf299a6b3ceceedfe0a14d4b874.svg)
