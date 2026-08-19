# [Introduction to Flow Matching and Diffusion Models, MIT](https://diffusion.csail.mit.edu/2026/index.html)

- **Created**: 2026-08-04
- **Last Updated**: 2026-08-19
- **Status**: `In Progress`
- **Related**:
  - [[papers-diffusion]] — Broader reading list covering the foundations, objectives, architectures, and applications of diffusion models.

---

- **Course**: <https://diffusion.csail.mit.edu/2026/index.html>
- [Lecture Notes](assets/course-mit-diffusion-2026/lecture-notes.pdf)

---

| Done | Lecture | Topic | Slides | Recording | Lecture Notes | Lab | Additional Material |
| :--: | :--: | --- | :--: | :--: | --- | --- | --- |
| ☑ | 1 | **Flow and Diffusion Models**<br>• Introduction to generative models<br>• Ordinary and stochastic differential equations<br>• Sampling from flow and diffusion models | [slides](assets/course-mit-diffusion-2026/lecture-01-flow-and-diffusion-models.pdf) | [recording](https://www.youtube.com/watch?v=9eJQQVrUUoI) | [§§1-2](assets/course-mit-diffusion-2026/lecture-notes.pdf)<br>[Appendix A: A Reminder on Probability Theory](assets/course-mit-diffusion-2026/lecture-notes.pdf#page=70) | [Lab 1: ODEs and SDEs](https://colab.research.google.com/drive/18W-IB1QwdK7zuKlWrx5uB76FM-9Ak3eW?usp=sharing) | [3blue1brown: ODEs](https://www.3blue1brown.com/lessons/differential-equations/)<br>[3blue1brown: PDEs](https://www.3blue1brown.com/lessons/pdes/) |
| ☑ | 2 | **Flow Matching**<br>• Conditional and marginal probability path<br>• Conditional and marginal vector field<br>• Flow matching training objective | [slides](assets/course-mit-diffusion-2026/lecture-02-flow-matching.pdf) | [recording](https://www.youtube.com/watch?v=PNkMKWW8Khw) | [§3](assets/course-mit-diffusion-2026/lecture-notes.pdf)<br>[Appendix B: A Proof of the Fokker-Planck Equation](assets/course-mit-diffusion-2026/lecture-notes.pdf#page=72) | [Lab 2: Flow and Score Matching](https://colab.research.google.com/drive/1Rb9pjn-lEH2r9F0UvIos7W0IWsBUs_kX?usp=sharing) | [Mario Gemoll: Flow Matching](https://mariogemoll.com/flow-matching)<br>[Khan Academy: Divergence](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/divergence-and-curl-articles/a/divergence)<br>[Khan Academy: Intuition for the Divergence Formula](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/divergence-and-curl-articles/a/intuition-for-divergence-formula)<br>[Greg Wayne: Flow Matching Notes](../flourish/assets/2026-greg-wayne-sessions/Notes_Flow_Matching.pdf) |
| ☐ | 3-A | **Score Functions and Score Matching**<br>• Score functions<br>• Denoising score matching<br>• SDE sampling | [slides](assets/course-mit-diffusion-2026/lecture-03-score-matching-and-guidance.pdf) | [recording](https://www.youtube.com/watch?v=ngC3QnYSVNM) | [§4](assets/course-mit-diffusion-2026/lecture-notes.pdf) | [Lab 2: Flow and Score Matching](https://colab.research.google.com/drive/1Rb9pjn-lEH2r9F0UvIos7W0IWsBUs_kX?usp=sharing) | [Mario Gemoll: Diffusion](https://mariogemoll.com/diffusion) |
| ☐ | 3-B | **Classifier-free Guidance**<br>• Guided generation<br>• Classifier guidance<br>• Classifier-free guidance | [slides](assets/course-mit-diffusion-2026/lecture-03-score-matching-and-guidance.pdf) | [recording](https://www.youtube.com/watch?v=8oWZ1bHwyRI) | [§5](assets/course-mit-diffusion-2026/lecture-notes.pdf) | — | — |
| ☐ | 4 | **Latent Spaces and Neural Network Architectures**<br>• Variational autoencoders and latent spaces<br>• Diffusion Transformer and U-Nets<br>• Case studies: Large-scale models | [slides](assets/course-mit-diffusion-2026/lecture-04-latent-spaces-and-neural-network-architectures.pdf) | [recording](https://www.youtube.com/watch?v=g0MB1CCBmsI) | [§6](assets/course-mit-diffusion-2026/lecture-notes.pdf)<br>[Appendix D: Additional Perspectives on VAEs](assets/course-mit-diffusion-2026/lecture-notes.pdf#page=77) | [Lab 3: Diffusion Transformer and VAEs](https://github.com/eje24/iap-diffusion-labs/blob/2026/labs/lab_three.ipynb) | — |
| ☐ | 5 | **Discrete Diffusion Models**<br>• Continuous-time Markov chains (CTMCs)<br>• Sampling from CTMC models<br>• Training CTMC models | [slides](assets/course-mit-diffusion-2026/lecture-05-discrete-diffusion-models.pdf) | [recording](https://www.youtube.com/watch?v=d0kmyEJN2hI) | [§7](assets/course-mit-diffusion-2026/lecture-notes.pdf)<br>[Appendix C: Existence and Uniqueness of Continuous-time Markov Chains](assets/course-mit-diffusion-2026/lecture-notes.pdf#page=74) | — | — |
| ☐ | — | **A Guide to the Diffusion Model Literature** | — | — | [Appendix E](assets/course-mit-diffusion-2026/lecture-notes.pdf#page=81) | — | — |

---

## [Lecture1] Flow and Diffusion Models

### Generative Modeling as Sampling

_**TL;DR:** Represent an object as a point in a high-dimensional space, represent plausible objects by a probability distribution over that space, and define generation as drawing a new point from that distribution._

The central direction of travel is:

$$
\text{simple noise} \longrightarrow \text{structured data}.
$$

Destroying structure by adding noise is easy. Generative modeling asks us to learn the much harder reverse transformation: starting from something easy to sample, such as Gaussian noise, produce a plausible image, video, or molecule.

The first abstraction is to represent each object numerically as a vector $z \in \mathbb{R}^d$:

- An RGB image with height $H$ and width $W$ is an array in $\mathbb{R}^{H \times W \times 3}$, which can be flattened into a vector with $d = 3HW$ coordinates.
- A video with $T$ frames is an element of $\mathbb{R}^{T \times H \times W \times 3}$.
- A simple representation of a molecule with $N$ atoms is an element of $\mathbb{R}^{3 \times N}$ containing the atoms' spatial coordinates.

The ambient space $\mathbb{R}^d$ contains every possible assignment of coordinates, including mostly meaningless images or physically impossible molecules. Plausible data occupies only a highly structured part of this space. We express that structure with an unknown data distribution $p_{\mathrm{data}}$.

Generation then means sampling

$$
z \sim p_{\mathrm{data}}.
$$

This formulation preserves diversity. There is no single best image of a dog; there are many valid samples, and $p_{\mathrm{data}}$ describes how the probability mass is distributed among them. Strictly, for continuous data, $p_{\mathrm{data}}(z)$ is a **density**, not the probability of the exact point $z$; probabilities are obtained by integrating the density over regions.

We do not know $p_{\mathrm{data}}$ directly. We only have a finite dataset that is modeled as iid draws:

$$
z_1,\ldots,z_N \overset{\mathrm{iid}}{\sim} p_{\mathrm{data}}.
$$

The dataset is therefore evidence about the distribution, not the distribution itself. The learning problem is to construct an algorithm that can produce new samples distributed approximately like the population that generated the dataset, rather than merely return training examples.

For conditional generation, the target distribution also depends on conditioning information $y$:

$$
z \sim p_{\mathrm{data}}(\cdot \mid y).
$$

Here $y$ might be a text prompt, class label, another image, or some scientific constraint. Unconditional generation learns what objects are plausible; conditional generation restricts plausibility to objects compatible with $y$. The underlying sampling machinery can be studied first without conditioning and extended later.

**Connection to language models.** An autoregressive language model also defines generation as sampling, but over a discrete sequence:

$$
p(x_{1:L}) = \prod_{i=1}^{L} p(x_i \mid x_{<i}).
$$

It generates one token at a time from these conditional distributions. The models in this course instead focus mainly on continuous vectors and generate by evolving an entire state through continuous time. Section 7 later returns to discrete diffusion for text.

### From Sampling to Continuous Dynamics

_**TL;DR:** Begin with a sample from an easy distribution and move it continuously so that the population of endpoints follows the data distribution._

Directly sampling from the unknown, complicated $p_{\mathrm{data}}$ is hard. Sampling from a simple initial distribution is easy, usually

$$
X_0 \sim p_{\mathrm{init}},
\qquad
p_{\mathrm{init}} = \mathcal{N}(0,I_d).
$$

We would like to construct a continuous evolution from time $t=0$ to $t=1$ such that

$$
X_0 \sim p_{\mathrm{init}}
\quad\Longrightarrow\quad
X_1 \sim p_{\mathrm{data}}.
$$

The interval $[0,1]$ is just a convenient convention. Here $t$ is **generative time** or **noise time**, not necessarily physical time. For a video, for example, the coordinates of $X_t$ already contain the video's frame-time dimension; the subscript $t$ describes how the whole video sample is being transformed by the generative process.

There are two main ways to specify this evolution:

- A **flow model** uses an ordinary differential equation (ODE), giving a deterministic trajectory once $X_0$ is fixed.
- A **diffusion model** uses a stochastic differential equation (SDE), injecting additional randomness throughout the trajectory.

The common idea is iterative generation: many simple local changes compose into a complicated global transformation from noise to data.

### ODEs: Trajectories, Vector Fields, and Flows

_**TL;DR:** A vector field gives the instantaneous velocity at every state and time; solving its ODE produces trajectories, and the flow collects the trajectories for every possible starting state._

A **trajectory** is a time-indexed position:

$$
X : [0,1] \to \mathbb{R}^d,
\qquad
t \mapsto X_t.
$$

A **time-dependent vector field** assigns a velocity to every possible position and time:

$$
u : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d,
\qquad
(x,t) \mapsto u_t(x).
$$

The vector $u_t(x)$ answers a local question: _if the state were at $x$ at time $t$, in which direction and how quickly should it move?_ It does not directly tell us the final destination.

Given a fixed initial state $x_0$, the solution of the initial-value problem is one trajectory $X$. The ODE requires this trajectory's instantaneous velocity to equal the vector field evaluated at its current location:

$$
\frac{dX_t}{dt} = u_t(X_t),
\qquad
X_0=x_0.
$$

The dependence $u_t(X_t)$ is important. As $X_t$ moves, it enters a new part of the field, receives a new velocity, and bends accordingly. This is the higher-dimensional version of a slope field: arrows describe local derivatives, while a solution curve follows those arrows.

![A trajectory following the arrows of a time-dependent vector field](assets/course-mit-diffusion-2026/media/lecture-01/ode-trajectory.png)

_A trajectory follows the local arrows as it moves through the field. Source: [Mario Gemoll's interactive visualization](https://mariogemoll.com/flow-matching), via the [MIT Lecture 1 slides](https://diffusion.csail.mit.edu/2026/docs/20260120_Lecture_01.pdf)._

The same equation can be written in integral form:

$$
X_t
=
x_0 + \int_0^t u_s(X_s)\,ds.
$$

This says that the current position equals the starting position plus all the infinitesimal displacements accumulated so far. It is also the form from which numerical solvers such as Euler's method follow naturally.

The **flow** gathers the trajectories for every possible initial state into one solution operator:

$$
\psi : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d,
\qquad
(x_0,t) \mapsto \psi_t(x_0),
$$

where

$$
\boxed{X_t=\psi_t(X_0)=\psi_t(x_0)}.
$$

Thus, after fixing $x_0$, the function $t\mapsto\psi_t(x_0)$ is exactly that initial point's solution trajectory. Because this must hold for every $x_0$, the flow satisfies

$$
\frac{d}{dt}\psi_t(x_0)
=
u_t\!\left(\psi_t(x_0)\right),
\qquad
\psi_0(x_0)=x_0
\qquad
\text{for every }x_0.
$$

This resolves the apparently competing terminology in the lecture notes: the **trajectory is the individual solution for one initial condition**, while the **flow is the family, or solution operator, containing the solutions for all initial conditions**. It is common to call the flow a solution of the ODE as shorthand because $t\mapsto\psi_t(x_0)$ obeys the ODE for each fixed $x_0$.

![A vector field progressively warping a grid through its flow](assets/course-mit-diffusion-2026/media/lecture-01/flow-warp.png)

_The blue arrows specify local velocities; integrating them progressively warps the entire red grid. Source: [Lipman et al., Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264), reproduced as Figure 1 in the MIT lecture notes._

For a random initial condition $X_0$, we select one of these trajectories at random:

$$
X_t=\psi_t(X_0).
$$

The three concepts differ mainly in viewpoint:

- $u_t(x)$ is the full field of local instructions.
- $X_t$ is one solution path obtained from one fixed initial condition $X_0=x_0$.
- $\psi_t$ maps every possible $x_0$ to its position at time $t$, so it collects all the solution paths.

In the wind analogy, $u_t(x)$ is the wind velocity everywhere, $X_t$ is the path of one balloon released at $x_0$, and $\psi_t$ is the map saying where a balloon released from every possible starting point would be at time $t$.

So the causal chain is

$$
\text{vector field}
\longrightarrow
\text{ODE}
\longrightarrow
\text{flow}.
$$

The model will eventually learn or parameterize the **vector field**, while an ODE solver computes the resulting trajectory/flow. The neural network is not usually asked to output $X_1$ in a single jump.

### Existence, Uniqueness, and Invertibility

_**TL;DR:** A sufficiently regular vector field gives exactly one trajectory from each initial state, so the resulting finite-time flow is reversible._

The lecture notes give a convenient sufficient condition: if $u$ is continuously differentiable with bounded derivative, then the ODE has a unique flow $\psi_t$, and each finite-time map $\psi_t$ is a diffeomorphism—a differentiable, invertible warping of space with a differentiable inverse.

A slightly more general way to remember the relevant condition is that $u_t(x)$ should be sufficiently regular in time and **Lipschitz in $x$**:

$$
\lVert u_t(x)-u_t(y)\rVert
\leq
L\lVert x-y\rVert.
$$

This limits how abruptly nearby velocities can diverge. The practical consequences are more important here than the technical theorem:

- the same starting point cannot branch into multiple solutions;
- two distinct trajectories cannot collide at the same finite time and then continue as one, because running the ODE backward would violate uniqueness;
- knowing the endpoint and the vector field lets us integrate backward to recover the start;
- the flow can continuously warp, stretch, contract, and rotate space without tearing or gluing it.

These statements concern the exact ODE. A coarse numerical approximation may still introduce error or apparent crossings, which is one reason step size matters.

### Linear Vector Field Example

_**TL;DR:** The field $u_t(x)=-\theta x$ points toward the origin and produces exponential contraction._

Consider the time-independent linear vector field

$$
u_t(x)=-\theta x,
\qquad
\theta>0.
$$

Every arrow points opposite the current position, and its magnitude grows linearly with the distance from the origin. The proposed flow is

$$
\psi_t(x_0)=e^{-\theta t}x_0.
$$

It satisfies the initial condition because

$$
\psi_0(x_0)=e^0x_0=x_0,
$$

and it satisfies the ODE because

$$
\begin{aligned}
\frac{d}{dt}\psi_t(x_0)
&= \frac{d}{dt}\left(e^{-\theta t}x_0\right) \\
&= -\theta e^{-\theta t}x_0 \\
&= -\theta\psi_t(x_0) \\
&= u_t\!\left(\psi_t(x_0)\right).
\end{aligned}
$$

For every finite $t$, multiplication by $e^{-\theta t}>0$ remains invertible. Points approach the origin exponentially but reach it only in the limit $t\to\infty$, so finite-time invertibility is not contradicted.

### Simulating an ODE with Euler's Method

_**TL;DR:** Euler's method repeatedly freezes the current velocity for a short time and takes a straight step in that direction._

In general, we cannot compute the flow $\psi_t$ directly. Euler's method approximates it by dividing $[0,1]$ into $n$ short steps of size

$$
h=\frac{1}{n},
$$

and repeatedly applying

$$
\boxed{
X_{t+h}=X_t+h\,u_t(X_t)
}
\qquad
t=0,h,2h,\ldots,1-h.
$$

Here $u_t(X_t)$ is the current velocity, so $h\,u_t(X_t)$ is the displacement obtained by following that velocity for time $h$. Each step therefore:

1. reads the arrow at the current position;
2. moves a short distance in that direction;
3. advances time by $h$;
4. repeats until $t=1$.

Smaller steps usually follow a curved trajectory more accurately, but they require more vector-field evaluations. In a flow model, each evaluation is a neural-network forward pass, so accuracy is traded against sampling cost.

![Euler trajectories with a large and small step size](assets/course-mit-diffusion-2026/media/lecture-01/euler-step-size.png)

_A large step uses fewer field evaluations but follows the curved trajectory less accurately. Source: [Mario Gemoll](https://mariogemoll.com/flow-matching), via the [MIT Lecture 1 slides](https://diffusion.csail.mit.edu/2026/docs/20260120_Lecture_01.pdf)._

### Improving Euler with Heun's Method

_**TL;DR:** Euler uses only the velocity at the beginning of a step; Heun predicts the endpoint, evaluates the velocity there too, and moves using the average of the two velocities._

Euler uses the velocity at the beginning of the step for the entire update, so it can miss changes in direction. Heun's method first takes an Euler step to predict the endpoint:

$$
\widetilde{X}_{t+h}
=
X_t+h\,u_t(X_t).
$$

It then evaluates the velocity at that predicted endpoint and performs the actual update using the average of the two velocities:

$$
\boxed{
X_{t+h}
=
X_t+
\frac{h}{2}
\left[
u_t(X_t)
+
u_{t+h}\!\left(\widetilde{X}_{t+h}\right)
\right]
}.
$$

The two stages are therefore:

1. **Predict:** follow the initial arrow for one Euler step.
2. **Correct:** compare the initial arrow with the arrow at the predicted endpoint and use their average.

Heun usually follows curved trajectories more accurately than Euler at the same step size, but it requires two vector-field evaluations per step instead of one.

### Flow Models

_**TL;DR:** A flow model learns a neural vector field that deterministically transports a random draw from a simple initial distribution into a data sample._

We now parameterize the vector field with a neural network:

$$
u^\theta:
\mathbb{R}^d\times[0,1]
\to
\mathbb{R}^d,
\qquad
(x,t)\mapsto u_t^\theta(x),
$$

where $\theta$ denotes the network parameters. A flow model is the initial-value problem

$$
\boxed{
\begin{aligned}
X_0 &\sim p_{\mathrm{init}}, \\
\frac{dX_t}{dt} &= u_t^\theta(X_t).
\end{aligned}
}
$$

The ODE is deterministic **conditional on $X_0$**: fixing the initial noise fixes the entire trajectory and endpoint. The generated sample is nevertheless random because the initial state is random. Usually,

$$
p_{\mathrm{init}}=\mathcal{N}(0,I_d),
$$

although any distribution that is easy to sample at inference time could be used.

Let $\psi_t^\theta$ be the flow induced by $u_t^\theta$. Then

$$
X_t=\psi_t^\theta(X_0),
$$

and the distribution of states at time $t$ is the pushforward

$$
p_t^\theta
=
(\psi_t^\theta)_{\#}p_{\mathrm{init}}.
$$

The learning goal is to make the endpoint distribution equal, or closely approximate, the data distribution:

$$
\boxed{
X_1\sim p_{\mathrm{data}}
\quad\Longleftrightarrow\quad
\psi_1^\theta(X_0)\sim p_{\mathrm{data}}
}
$$

or equivalently

$$
p_1^\theta\approx p_{\mathrm{data}}.
$$

![A simple initial distribution and a more structured data distribution](assets/course-mit-diffusion-2026/media/lecture-01/initial-and-data-distributions.png)

_The random initial sample $X_0$ comes from a simple distribution; the desired endpoint $X_1$ follows the structured data distribution. Figure credit: Yaron Lipman, from the [MIT Lecture 1 slides](https://diffusion.csail.mit.edu/2026/docs/20260120_Lecture_01.pdf)._

The source of randomness and the transformation play separate roles:

$$
\underbrace{X_0\sim p_{\mathrm{init}}}_{\text{random seed}}
\xrightarrow{\text{deterministic flow }\psi_1^\theta}
\underbrace{X_1}_{\text{generated sample}}.
$$

This is the same general pattern as a latent-variable generator $z\mapsto G_\theta(z)$: a deterministic function transforms a random latent variable into a random output. In a flow model, however, the transformation is defined implicitly by integrating a time-dependent vector field.

**The network parameterizes the vector field, not the flow.** A single evaluation of $u_t^\theta(x)$ produces only the instantaneous velocity at one state and time. The global map $\psi_t^\theta$ is obtained by repeatedly evaluating that network inside an ODE solver. Training must therefore learn suitable local velocities; numerical integration composes them into the final transformation.

**Sampling from a flow model with Euler's method.** Given $u_t^\theta$ and a number of steps $n$:

1. Set $t=0$ and $h=1/n$.
2. Draw $X_0\sim p_{\mathrm{init}}$.
3. For $n$ steps, update

   $$
   X_{t+h}=X_t+h\,u_t^\theta(X_t),
   \qquad
   t\leftarrow t+h.
   $$

4. Return $X_1$.

With Euler, sampling takes $n$ neural-network evaluations. The returned $X_1$ approximates $\psi_1^\theta(X_0)$; changing the numerical solver or step size changes the simulation error without changing the underlying learned continuous-time model.

At this point the remaining question is how to choose $\theta$ so that $p_1^\theta\approx p_{\mathrm{data}}$. Section 3 will answer that with flow matching. Section 2.2 first introduces the stochastic counterpart: diffusion models built from SDEs.

### Stochastic Processes and Random Trajectories

_**TL;DR:** An ODE assigns one trajectory to an initial point; an SDE assigns a distribution over trajectories because fresh randomness enters throughout the evolution._

For an ODE, fixing the initial condition $X_0=x_0$ fixes the complete trajectory. An SDE instead produces a **stochastic process**

$$
(X_t)_{0\leq t\leq 1}.
$$

There are two complementary ways to look at this object:

- For a fixed time $t$, $X_t$ is a random variable in $\mathbb{R}^d$ with some marginal distribution $p_t$.
- For one fixed outcome of all the randomness, $t\mapsto X_t$ is one random trajectory, or **sample path**.

More formally, a stochastic process is a function

$$
X:\Omega\times[0,1]\to\mathbb{R}^d,
\qquad
(\omega,t)\mapsto X_t(\omega),
$$

where $\omega\in\Omega$ represents one outcome of the random experiment. Holding $t$ fixed and varying $\omega$ gives the random variable $X_t$; holding $\omega$ fixed and varying $t$ gives one trajectory.

This distinction prevents a common confusion: the process is not one randomly chosen curve. It is the entire probabilistic rule that can produce many curves. Running the same SDE twice from the same $x_0$ can therefore produce different endpoints.

### Brownian Motion

_**TL;DR:** Brownian motion is continuous-time Gaussian noise: an interval of length $h$ contributes an independent Gaussian displacement with variance $h$ and therefore typical size $\sqrt{h}$._

A $d$-dimensional **Brownian motion**, or **Wiener process**, is a stochastic process $(W_t)_{t\geq0}$ satisfying:

1. It starts at the origin:

   $$
   W_0=0.
   $$

2. Its paths $t\mapsto W_t$ are continuous.

3. Its increments are Gaussian. For $0\leq s<t$,

   $$
   W_t-W_s\sim\mathcal{N}\!\left(0,(t-s)I_d\right).
   $$

4. Increments over disjoint time intervals are independent.

Taking $s=t$ and advancing by a small step $h$ gives

$$
W_{t+h}-W_t\sim\mathcal{N}(0,hI_d).
$$

Consequently, we can simulate its values on a time grid using

$$
\boxed{
W_{t+h}=W_t+\sqrt{h}\,\epsilon_t,
\qquad
\epsilon_t\sim\mathcal{N}(0,I_d)
}
$$

with a fresh independent $\epsilon_t$ at every step.

![Three independently sampled two-dimensional Brownian paths](assets/course-mit-diffusion-2026/media/lecture-01/brownian-motion.gif)

_The paths share the same initial point but use independent Gaussian increments. Locally rendered from the update above, adapted from [Mario Gemoll's Brownian-motion visualization](https://mariogemoll.com/flow-matching)._

**Why $\sqrt{h}$ rather than $h$?** Consider simulating one unit of time with $n=1/h$ independent increments. Because $\epsilon_t\sim\mathcal{N}(0,I_d)$, a single correctly scaled increment has variance

$$
\operatorname{Var}(\sqrt{h}\,\epsilon_t)=hI_d.
$$

Independent variances add, so after all $n$ steps the total variance is

$$
\frac{1}{h}\cdot hI_d=I_d.
$$

The alternatives fail as the step size shrinks:

- With $h\epsilon_t$, each step has variance $h^2I_d$, so the total variance is $hI_d\to0$: the randomness disappears.
- With unscaled $\epsilon_t$, each step has variance $I_d$, so the total variance is $I_d/h\to\infty$: the randomness explodes.

Thus $\sqrt h$ produces a finite, nonzero amount of randomness over a fixed time interval.

It also explains why Brownian motion is not differentiable. Over one short step, the deterministic drift in an SDE moves by a quantity proportional to $h$, while the Brownian increment typically has size proportional to $\sqrt h$:

$$
\text{drift displacement}\sim h,
\qquad
\text{Brownian displacement}\sim\sqrt h.
$$

Both go to zero as $h\to0$, which is compatible with a continuous path. But the Brownian finite-difference slope behaves like

$$
\frac{W_{t+h}-W_t}{h}
\stackrel{d}{=}
\frac{\epsilon_t}{\sqrt h}.
$$

Its standard deviation is $1/\sqrt h$, which diverges as $h\to0$. The path therefore remains continuous while its local slope never settles to a finite value; with probability one, Brownian paths are nowhere differentiable. An SDE must be interpreted through Brownian increments or stochastic integrals, not as an ODE containing an ordinary derivative $dW_t/dt$.

### From ODEs to SDEs

_**TL;DR:** An SDE combines a directed drift with Brownian spreading: locally it moves by $h$ times the drift plus $\sqrt h$ times fresh Gaussian noise._

Using a step size $h$, Euler simulates an ODE with the update

$$
X_{t+h}
=
X_t+h\,u_t(X_t).
$$

Euler-Maruyama simulates an SDE by adding a Brownian increment:

$$
X_{t+h}
=
X_t
+h\,u_t(X_t)
+\sigma_t(W_{t+h}-W_t).
$$

Because

$$
W_{t+h}-W_t
\stackrel{d}{=}
\sqrt h\,\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,I_d),
$$

the same update can be written as

$$
\boxed{
X_{t+h}
=
X_t
+h\,u_t(X_t)
+\sigma_t\sqrt h\,\epsilon_t
}.
$$

The continuous-time shorthand for these small-step dynamics is

$$
\boxed{
dX_t=u_t(X_t)\,dt+\sigma_t\,dW_t,
\qquad
X_0=x_0.
}
$$

The two terms have distinct roles:

- $u_t(X_t)\,dt$ is the **drift**. It gives the systematic local direction of travel.
- $\sigma_t\,dW_t$ is the **diffusion** term. The nonnegative coefficient $\sigma_t$ controls how strongly trajectories spread at time $t$.

A useful local interpretation is

$$
\mathbb{E}[X_{t+h}-X_t\mid X_t=x]
\approx
h\,u_t(x),
$$

and, for the scalar diffusion coefficient used in the notes,

$$
\operatorname{Cov}(X_{t+h}-X_t\mid X_t=x)
\approx
h\sigma_t^2I_d.
$$

Thus the drift describes the conditional mean displacement, while the diffusion coefficient describes the conditional covariance of the random displacement.

The $dX_t$ notation is symbolic shorthand for the small-step dynamics above; it should not be interpreted using an ordinary derivative $dW_t/dt$.

**What happened to the flow map?** An ODE has a deterministic map $x_0\mapsto\psi_t(x_0)$. For an SDE, $X_t$ is not determined by $x_0$ alone; it also depends on the Brownian path. One can define a random flow after fixing that Brownian path, but there is no single deterministic map of $x_0$ that gives every outcome.

### Existence and Uniqueness for SDEs

As in the ODE case, regularity assumptions prevent ambiguous dynamics. The notes use the sufficient conditions that $u$ is continuously differentiable with bounded derivative and $\sigma_t$ is continuous. Under these assumptions, the SDE has a unique solution.

Operationally, uniqueness means that if we fix both

$$
X_0=x_0
\quad\text{and}\quad
(W_t)_{0\leq t\leq1},
$$

the SDE determines one path $X_t$. Changing the Brownian realization changes the path; ambiguity in the differential equation does not.

Every ODE is the special case

$$
\sigma_t=0.
$$

The hierarchy is therefore

$$
\text{flow model}
\subset
\text{SDE model},
$$

with the flow model obtained by turning off stochasticity in the dynamics.

### Ornstein-Uhlenbeck Process

_**TL;DR:** The Ornstein-Uhlenbeck process balances an inward linear drift against continual Gaussian noise, producing a stationary Gaussian rather than collapsing to zero._

The lecture's main SDE example is

$$
\boxed{
dX_t=-\theta X_t\,dt+\sigma\,dW_t,
\qquad \theta>0.
}
$$

This is the **Ornstein-Uhlenbeck (OU) process**. Its two forces compete:

- The drift $-\theta X_t$ pulls the state toward zero. Farther points receive a stronger restoring force.
- The diffusion term $\sigma dW_t$ continually injects noise and spreads trajectories apart.

![Ornstein-Uhlenbeck trajectories under increasing diffusion strength](assets/course-mit-diffusion-2026/media/lecture-01/ornstein-uhlenbeck-diffusion.png)

_As $\sigma$ increases from left to right, the restoring drift remains but the paths fluctuate more strongly. Source: [MIT Lecture 1 slides](https://diffusion.csail.mit.edu/2026/docs/20260120_Lecture_01.pdf)._

For a fixed initial state $X_0=x_0$,

$$
X_t
=e^{-\theta t}x_0+\sigma\int_0^t e^{-\theta(t-s)}\,dW_s
\sim\mathcal{N}\!\left(e^{-\theta t}x_0,\;\sigma^2\int_0^t e^{-2\theta(t-s)}\,ds\,I_d\right)
=\mathcal{N}\!\left(e^{-\theta t}x_0,\;\frac{\sigma^2}{2\theta}\left(1-e^{-2\theta t}\right)I_d\right)
\xrightarrow[t\to\infty]{d}
\mathcal{N}\!\left(0,\frac{\sigma^2}{2\theta}I_d\right).
$$

The first term is the fading memory of the initial state; the stochastic integral is the accumulated Gaussian noise. The inward drift keeps that accumulated noise from spreading the process without bound.

This limiting behavior explains the trajectories in Figure 3 of the lecture notes:

- With $\sigma=0$, all paths smoothly contract to zero, exactly as in the earlier linear ODE.
- With $\sigma>0$, trajectories keep fluctuating, but mean reversion prevents them from wandering arbitrarily far.
- Increasing $\sigma$ increases the equilibrium variance; increasing $\theta$ strengthens mean reversion and decreases it.

The OU process is a useful prototype for diffusion-model noising processes: the signal from the initial condition decays while Gaussian uncertainty accumulates.

### Simulating an SDE with Euler-Maruyama

_**TL;DR:** Euler-Maruyama is Euler's method plus a correctly scaled, independent Gaussian increment at every step._

Choose $n$ steps and set

$$
h=\frac1n.
$$

Replacing the Brownian increment by its exact grid distribution,

$$
W_{t+h}-W_t=\sqrt h\,\epsilon_t,
\qquad
\epsilon_t\sim\mathcal{N}(0,I_d),
$$

gives the **Euler-Maruyama** update

$$
\boxed{
X_{t+h}
=
X_t
+h\,u_t(X_t)
+\sigma_t\sqrt h\,\epsilon_t,
\qquad
\epsilon_t\sim\mathcal{N}(0,I_d).
}
$$

Compare the two numerical methods directly:

$$
\begin{aligned}
\text{Euler:}\qquad
X_{t+h}
&=X_t+h\,u_t(X_t),\\
\text{Euler-Maruyama:}\qquad
X_{t+h}
&=X_t+h\,u_t(X_t)+\sigma_t\sqrt h\,\epsilon_t.
\end{aligned}
$$

At each step:

1. Evaluate the drift at the current state.
2. Move a distance $h\,u_t(X_t)$ in that direction.
3. Draw a fresh independent $\epsilon_t$.
4. Add the random displacement $\sigma_t\sqrt h\,\epsilon_t$.

The independent Gaussian draw is part of simulating the modeled dynamics, not merely numerical error. Reusing the same $\epsilon$ at every step would create strongly correlated increments and would not simulate Brownian motion.

As with Euler's method, a smaller $h$ produces a more faithful approximation but requires more evaluations of the neural drift. Because Brownian paths are rough, the general strong convergence rate of Euler-Maruyama is lower than Euler's rate for smooth ODEs; the simulation still converges even though individual paths never become differentiable.

### Diffusion Models as SDE Generators

_**TL;DR:** A diffusion model learns the drift of an SDE whose endpoint distribution should match the data; sampling numerically evolves Gaussian noise while injecting fresh noise along the path._

The lecture now parameterizes the drift with a neural network

$$
u^\theta:
\mathbb{R}^d\times[0,1]
\to
\mathbb{R}^d,
\qquad
(x,t)\mapsto u_t^\theta(x),
$$

and treats the diffusion coefficient $\sigma_t$ as fixed. Its abstract diffusion model is

$$
\boxed{
\begin{aligned}
X_0&\sim p_{\mathrm{init}},\\
dX_t&=u_t^\theta(X_t)\,dt+\sigma_t\,dW_t.
\end{aligned}
}
$$

The objective is to choose $\theta$ so that

$$
X_1\sim p_{\mathrm{data}}.
$$

Euler-Maruyama sampling is:

1. Set $h=1/n$ and draw $X_0\sim p_{\mathrm{init}}$.
2. For $t=0,h,\ldots,1-h$:

   $$
   \epsilon_t\sim\mathcal{N}(0,I_d),
   $$

   $$
   X_{t+h}
   =
   X_t
   +h\,u_t^\theta(X_t)
   +\sigma_t\sqrt h\,\epsilon_t.
   $$

3. Return $X_1$.

Unlike a flow model, randomness now enters in two places:

$$
\underbrace{X_0\sim p_{\mathrm{init}}}_{\text{random initial state}}
\quad\text{and}\quad
\underbrace{dW_t}_{\text{fresh randomness during evolution}}.
$$

Setting $\sigma_t=0$ removes the second source and recovers the flow model exactly.

**What does the neural network predict?** In this abstract presentation, it directly parameterizes the SDE drift $u_t^\theta(x)$. Practical diffusion models are often trained to predict a score, noise $\epsilon$, velocity $v$, or clean data $x_0$ instead. Those quantities can be converted algebraically into the drift required by the sampler; §4 develops the score-based construction.

**Time-direction convention.** This lecture labels the generative direction as

$$
X_0\sim p_{\mathrm{init}}
\longrightarrow
X_1\sim p_{\mathrm{data}}.
$$

DDPM papers use the opposite time convention: $x_0$ is data and $x_1$ is noise, so generation runs from $t=1$ back to $t=0$. This is only a relabeling of time.

Lecture 1 has now specified what flow and diffusion generative models **are** and how to simulate them. It has not yet explained how to learn a drift that reaches $p_{\mathrm{data}}$: §3 introduces flow matching for deterministic flows, while §4 introduces score matching and reverse-SDE sampling for diffusion models.

## [Lecture2] Flow Matching

### From a Flow Model to a Training Problem

_**TL;DR:** A flow model can already turn an initial sample into an endpoint; flow matching supplies a tractable target vector field whose endpoint distribution is the data distribution._

Lecture 1 defined a flow model but did not say how to choose its neural vector field. The model is

$$
X_0\sim p_{\mathrm{init}},
\qquad
dX_t=u_t^\theta(X_t)\,dt,
$$

or equivalently

$$
\frac{dX_t}{dt}=u_t^\theta(X_t).
\tag{10}
$$

Generation means numerically simulating this ODE from $t=0$ to $t=1$ and returning $X_1$. Training must therefore choose $\theta$ so that

$$
X_0\sim p_{\mathrm{init}}
\quad\Longrightarrow\quad
X_1\sim p_{\mathrm{data}}.
$$

This is a distribution-level requirement: there is no distinguished correct endpoint for any particular initial noise sample. Many different flows can carry the same initial distribution to the same data distribution. Flow matching resolves this freedom by first choosing a convenient path of intermediate distributions and then constructing a vector field that realizes it.

Although this section trains a deterministic ODE, its Gaussian corruption paths will look almost identical to the noising formulas used in diffusion models. The later distinction is in the dynamics used for generation: flow matching here learns an ODE, whereas score-based diffusion constructs a reverse SDE or its associated probability-flow ODE.

### Conditional and Marginal Probability Paths

_**TL;DR:** A conditional path ends at one chosen datum $z$; mixing those paths over $z\sim p_{\mathrm{data}}$ gives the marginal path from noise to the full data distribution._

A **probability path** $(p_t)_{0\leq t\leq1}$ specifies the desired distribution of $X_t$ at every time. It describes a sequence of population snapshots, not how any particular particle moves between snapshots. A vector field, introduced below, supplies those dynamics.

For a fixed data point $z\in\mathbb R^d$, a conditional probability path $p_t(\cdot\mid z)$ satisfies

$$
p_0(\cdot\mid z)=p_{\mathrm{init}},
\qquad
p_1(\cdot\mid z)=\delta_z.
\tag{11}
$$

Here $\delta_z$ is the Dirac point mass: a sample from $\delta_z$ equals $z$ with probability one. Thus the conditional path starts with the same noise distribution for every $z$ and gradually collapses onto that one selected data point.

The word **conditional** means "for one fixed endpoint $z$." To recover the whole dataset distribution, first draw the endpoint and then draw a noisy point conditional on it:

$$
Z\sim p_{\mathrm{data}},
\qquad
X_t\sim p_t(\cdot\mid Z).
\tag{12}
$$

The resulting marginal density is

$$
p_t(x)
=
\int p_t(x\mid z)p_{\mathrm{data}}(z)\,dz.
\tag{13}
$$

"Marginal" means that the latent choice of $z$ has been averaged out. The [sampling construction above](#conditional-and-marginal-probability-paths) is tractable with a dataset: sample a training example and corrupt it. The corresponding [marginal-density integral](#conditional-and-marginal-probability-paths) is generally intractable because evaluating $p_t(x)$ requires integrating over all possible clean data points.

The endpoints follow directly from the conditional endpoints. At $t=0$,

$$
\begin{aligned}
p_0(x)
&=\int p_0(x\mid z)p_{\mathrm{data}}(z)\,dz\\
&=\int p_{\mathrm{init}}(x)p_{\mathrm{data}}(z)\,dz\\
&=p_{\mathrm{init}}(x),
\end{aligned}
$$

because $p_{\mathrm{data}}$ integrates to one. At $t=1$,

$$
\begin{aligned}
p_1(x)
&=\int \delta_z(x)p_{\mathrm{data}}(z)\,dz\\
&=p_{\mathrm{data}}(x).
\end{aligned}
$$

Therefore

$$
p_0=p_{\mathrm{init}},
\qquad
p_1=p_{\mathrm{data}}.
\tag{14}
$$

The conditional and marginal views are two levels of the same construction:

- $p_t(\cdot\mid z)$: what noisy versions of one particular $z$ look like at time $t$;
- $p_t$: what noisy versions of the entire data distribution look like at time $t$.

![Conditional and marginal Gaussian probability paths](assets/course-mit-diffusion-2026/media/lecture-02/conditional-and-marginal-probability-path.png)

_The conditional path (top) contracts toward one fixed $z$; the marginal path (bottom) spreads toward the full data distribution as $z$ varies. Source: Figure 5 of the [lecture notes](assets/course-mit-diffusion-2026/lecture-notes.pdf)._

### The Gaussian Conditional Probability Path

_**TL;DR:** Form an intermediate sample by scaling a clean datum and a Gaussian-noise sample; changing their coefficients moves the distribution continuously from noise to data._

Let $\alpha_t$ and $\beta_t$ be differentiable schedules satisfying

$$
\alpha_0=0,
\quad
\alpha_1=1,
\qquad
\beta_0=1,
\quad
\beta_1=0.
$$

The Gaussian conditional path is

$$
p_t(\cdot\mid z)
=
\mathcal N(\alpha_tz,\beta_t^2I_d).
\tag{15}
$$

It can be sampled by

$$
Z\sim p_{\mathrm{data}},
\qquad
\epsilon\sim\mathcal N(0,I_d),
\qquad
X_t=\alpha_tZ+\beta_t\epsilon.
\tag{16}
$$

![Images sampled along a Gaussian conditional probability path](assets/course-mit-diffusion-2026/media/lecture-02/gaussian-image-probability-path.png)

_As $t$ increases, $\alpha_t$ preserves more of each image and $\beta_t$ retains less Gaussian noise. Source: Figure 4 of the [lecture notes](assets/course-mit-diffusion-2026/lecture-notes.pdf)._

Conditioned on $Z=z$,

$$
\alpha_tz+\beta_t\epsilon
\sim
\mathcal N(\alpha_tz,\beta_t^2I_d),
$$

because scaling a standard Gaussian by $\beta_t$ gives covariance $\beta_t^2I_d$ and adding $\alpha_tz$ shifts its mean.

At the endpoints,

$$
X_0=\epsilon\sim\mathcal N(0,I_d),
\qquad
X_1=Z\sim p_{\mathrm{data}}.
$$

A particularly simple choice is the **conditional optimal transport**, or **CondOT**, schedule

$$
\alpha_t=t,
\qquad
\beta_t=1-t,
$$

which gives the straight interpolation

$$
X_t=tZ+(1-t)\epsilon.
$$

**Why is this called conditional optimal transport?** Fix one data point $z$. The conditional problem is then to move the initial Gaussian distribution to the point mass $\delta_z$. For each initial noise sample $\epsilon$, the minimum-energy path between the two fixed endpoints is the straight line

$$
\underset{\substack{x_0=\epsilon\\x_1=z}}{\operatorname{argmin}}
\int_0^1\left\|\frac{dx_t}{dt}\right\|^2\,dt
\quad\Longrightarrow\quad
x_t=tz+(1-t)\epsilon.
$$

It moves at the constant velocity

$$
\frac{dx_t}{dt}=z-\epsilon.
$$

Since $\epsilon\sim\mathcal N(0,I_d)$, the population of these straight trajectories has

$$
X_t\mid Z=z
\sim
\mathcal N\!\left(tz,(1-t)^2I_d\right),
$$

which is exactly the Gaussian path obtained from $\alpha_t=t$ and $\beta_t=1-t$. It is **conditional** OT because this argument is made separately for every fixed endpoint $z$.

This name does **not** imply that independently pairing $\epsilon\sim p_{\mathrm{init}}$ with $Z\sim p_{\mathrm{data}}$ gives the globally optimal transport coupling between the full noise and data distributions. The conditional trajectories are straight, but after averaging over all $z$, trajectories of the learned marginal vector field need not be straight or globally OT-optimal.

The original [Flow Matching paper](https://arxiv.org/abs/2210.02747) calls these **conditional OT paths**, while [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) uses the shorthand **CondOT** and exposes `CondOTPath` and `CondOTScheduler` in its library. The term is therefore standard within flow matching, but not universal across generative modeling. Other presentations may say **straight-line path** or **linear interpolation path**; closely related rectified-flow work uses the same interpolation under its own terminology.

**Variance along the path.** Let

$$
X_t=\alpha_tZ+\beta_t\epsilon,
\qquad
\epsilon\sim\mathcal N(0,I_d),
$$

with $Z$ and $\epsilon$ independent. If $Z$ has mean $\mu_{\mathrm{data}}$ and covariance $\Sigma_{\mathrm{data}}$, then

$$
\mathbb E[X_t]=\alpha_t\mu_{\mathrm{data}}
$$

and

$$
\begin{aligned}
\operatorname{Cov}(X_t)
&=\operatorname{Cov}(\alpha_tZ+\beta_t\epsilon)\\
&=\alpha_t^2\operatorname{Cov}(Z)
+\beta_t^2\operatorname{Cov}(\epsilon)\\
&=\alpha_t^2\Sigma_{\mathrm{data}}+\beta_t^2I_d.
\end{aligned}
$$

The cross-covariance terms vanish because $Z$ and $\epsilon$ are independent. The squares appear because scaling a random vector by $c$ scales its covariance by $c^2$:

$$
\operatorname{Cov}(cY)=c^2\operatorname{Cov}(Y).
$$

If the data has been whitened, or is idealized as isotropic, so that $\Sigma_{\mathrm{data}}=I_d$, this reduces to

$$
\operatorname{Cov}(X_t)
=(\alpha_t^2+\beta_t^2)I_d.
$$

Therefore, the condition

$$
\alpha_t^2+\beta_t^2=1
$$

keeps the **marginal covariance** equal to $I_d$ throughout the path.

CondOT instead uses $\alpha_t=t$ and $\beta_t=1-t$. Under the same unit-covariance assumption,

$$
\operatorname{Cov}(X_t)
=\left(t^2+(1-t)^2\right)I_d.
$$

This equals $I_d$ at the endpoints but contracts to $\tfrac12I_d$ at $t=\tfrac12$. Flow matching does not require constant variance, so this contraction is allowed; CondOT chooses constant-speed straight trajectories instead.

**A simple variance-preserving schedule**, using the course's noise-to-data time convention, is

$$
\alpha_t=\sqrt t,
\qquad
\beta_t=\sqrt{1-t},
$$

giving

$$
X_t=\sqrt t\,Z+\sqrt{1-t}\,\epsilon.
$$

Its covariance is

$$
\operatorname{Cov}(X_t)
=t\Sigma_{\mathrm{data}}+(1-t)I_d
=I_d
\qquad\text{when }\Sigma_{\mathrm{data}}=I_d.
$$

### Conditional Vector Fields

_**TL;DR:** A conditional vector field supplies dynamics whose population of trajectories has the prescribed conditional distribution at every time._

For each fixed $z$, a conditional vector field $u_t^{\mathrm{target}}(x\mid z)$ must satisfy

$$
X_0\sim p_{\mathrm{init}},
\qquad
\frac{dX_t}{dt}
=u_t^{\mathrm{target}}(X_t\mid z)
\quad\Longrightarrow\quad
X_t\sim p_t(\cdot\mid z)
\quad(0\leq t\leq1).
\tag{17}
$$

The distinction between a probability path and a vector field is essential:

- $p_t(x\mid z)$ is a scalar density describing where probability mass is found at time $t$;
- $u_t(x\mid z)\in\mathbb R^d$ is a velocity describing the direction and speed of motion at $(x,t)$.

Consequently,

$$
\frac{p_{t+\Delta t}(x\mid z)-p_t(x\mid z)}{\Delta t}
$$

approximates the scalar density derivative $\partial_tp_t(x\mid z)$, not the vector velocity. The continuity equation, explained after Example 10, connects density change to particle motion through conservation of probability mass. The density change does not by itself determine a unique vector field.

For example, a rotational field can move particles around inside an isotropic Gaussian without changing the Gaussian density at all. The same probability snapshots can therefore admit different particle trajectories.

The conditional field is analytically convenient but not itself a useful unconditional generator: it must be given a data point $z$, and every trajectory conditioned on it ends at that same already-known example rather than producing a new sample from $p_{\mathrm{data}}$. Its role is to provide tractable building blocks and, later, tractable regression targets.

### The Marginalization Trick

_**TL;DR:** At a noisy location $x$, average the conditional velocities using the posterior probability that each clean datum $z$ generated that $x$._

The marginalization result is easiest to understand as follows:

- Each possible $z$ proposes a conditional velocity $u_t^{\mathrm{target}}(x\mid z)$.
- At the current $x$, some values of $z$ are more plausible than others.
- Average the proposed velocities according to $p_t(z\mid x)$, the probability of $z$ given the current $x$.

Thus the marginal vector field is

$$
\boxed{
u_t^{\mathrm{target}}(x)
=
\int
u_t^{\mathrm{target}}(x\mid z)
p_t(z\mid x)\,dz
=
\int
u_t^{\mathrm{target}}(x\mid z)
\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}\,dz
}.
\tag{18}
$$

Equivalently, it is the conditional expectation

$$
u_t^{\mathrm{target}}(x)
=
\mathbb E\!\left[
u_t^{\mathrm{target}}(x\mid Z)
\mid X_t=x
\right].
$$

Why not average directly with $p_{\mathrm{data}}(z)$? The prior $p_{\mathrm{data}}(z)$ says only how common $z$ is globally. It does not account for whether $z$ could plausibly have produced this particular $x$. Bayes' rule gives the required posterior:

$$
p_t(z\mid x)
=
p_t(x\mid z)\frac{p_{\mathrm{data}}(z)}{p_t(x)}.
$$

The likelihood $p_t(x\mid z)$ increases the weight of data points compatible with $x$ and decreases the weight of unrelated data points.

The posterior-averaged field follows the marginal probability path. Therefore

$$
X_0\sim p_{\mathrm{init}},
\qquad
\frac{dX_t}{dt}=u_t^{\mathrm{target}}(X_t)
\quad\Longrightarrow\quad
X_t\sim p_t,
\tag{19}
$$

and in particular $X_1\sim p_{\mathrm{data}}$.

### Gaussian Conditional Vector Field and Its Derivation (Example 10)

_**TL;DR:** Write a Gaussian sample as $X_t=\alpha_tz+\beta_t\epsilon$, differentiate it, and then eliminate $\epsilon$ in favor of the current point $x$._

For the Gaussian path

$$
p_t(\cdot\mid z)=\mathcal N(\alpha_tz,\beta_t^2I_d),
$$

the target conditional vector field is

$$
\boxed{
u_t^{\mathrm{target}}(x\mid z)
=
\left(
\dot\alpha_t
-\frac{\dot\beta_t}{\beta_t}\alpha_t
\right)z
+
\frac{\dot\beta_t}{\beta_t}x.
}
\tag{20}
$$

The proof can be written directly from the Gaussian sampling formula. Draw one $\epsilon\sim\mathcal N(0,I_d)$ and write

$$
X_t
=
\alpha_tz+\beta_t\epsilon.
$$

At every $t$,

$$
X_t
\sim
\mathcal N(\alpha_tz,\beta_t^2I_d)
=
p_t(\cdot\mid z).
$$

Differentiate $X_t=\alpha_tz+\beta_t\epsilon$ while holding $z$ and $\epsilon$ fixed. The resulting derivative is the conditional vector field evaluated at the current position $X_t$:

$$
u_t^{\mathrm{target}}(X_t\mid z)
=
\frac{dX_t}{dt}
=
\dot\alpha_tz+\dot\beta_t\epsilon.
$$

To express this velocity using the current point $X_t=x$, solve the sampling equation for the noise:

$$
\epsilon
=
\frac{x-\alpha_tz}{\beta_t}.
$$

Substitute this into the velocity:

$$
\begin{aligned}
u_t^{\mathrm{target}}(x\mid z)
&=\dot\alpha_tz
+\dot\beta_t\epsilon\\
&=\dot\alpha_tz
+\dot\beta_t\frac{x-\alpha_tz}{\beta_t}\\
&=\left(
\dot\alpha_t
-\frac{\dot\beta_t}{\beta_t}\alpha_t
\right)z
+\frac{\dot\beta_t}{\beta_t}x,
\end{aligned}
$$

which is the [Gaussian conditional vector field](#gaussian-conditional-vector-field-and-its-derivation-example-10).

For CondOT, $\alpha_t=t$ and $\beta_t=1-t$, so $\dot\alpha_t=1$ and $\dot\beta_t=-1$. The [general Gaussian conditional vector field](#gaussian-conditional-vector-field-and-its-derivation-example-10) simplifies to

$$
u_t^{\mathrm{target}}(x\mid z)
=
\left(1-\frac{-1}{1-t}t\right)z
+\frac{-1}{1-t}x
=
\frac{z-x}{1-t}
=
\frac{z-\left[tz+(1-t)\epsilon\right]}{1-t}
=
z-\epsilon.
$$

The final form is especially intuitive: the conditional trajectory is a straight line from initial noise $\epsilon$ to data $z$, so its constant velocity is simply endpoint minus starting point. The expression $(z-x)/(1-t)$ appears singular at $t=1$ only because it infers that constant velocity from a vanishing remaining displacement and time. Along a valid trajectory the ratio has the finite limit $z-\epsilon$.

![Conditional and marginal probability paths simulated with ODEs](assets/course-mit-diffusion-2026/media/lecture-02/conditional-and-marginal-odes.png)

_Top: for one fixed $z$, the conditional ODE follows the conditional probability path and every trajectory ends at $z$. Bottom: the marginal ODE follows the full marginal probability path and transports noise toward every mode of the data distribution without requiring a data-point label. Source: Figure 6 of the [lecture notes](assets/course-mit-diffusion-2026/lecture-notes.pdf)._

For any sampled data point $z$, both $u_t^{\mathrm{target}}(x\mid z)$ and $p_t(x\mid z)$ are easy to compute. The marginal field, however, requires the posterior average

$$
u_t^{\mathrm{target}}(x)
=
\int u_t^{\mathrm{target}}(x\mid z)
\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}\,dz,
\qquad
p_t(x)=\int p_t(x\mid z)p_{\mathrm{data}}(z)\,dz.
$$

The data distribution is normally available only through samples, not as a density that can be integrated analytically. Computing $p_t(x)$ would therefore require averaging $p_t(x\mid z)$ over every possible clean data point $z$; the same problem makes the posterior weights and the exact marginal field intractable. Conditional flow matching avoids evaluating any of these integrals: it trains on sampled $z$ using the easy conditional target, and the network learns the marginal field by averaging across training examples.

### The Divergence Operator

A vector field $v:\mathbb R^d\to\mathbb R^d$ assigns a vector $v(x)$ to every point $x$. Its **divergence** is the scalar

$$
\operatorname{div}(v)(x)
=
\sum_{i=1}^d
\frac{\partial v^i(x)}{\partial x_i}.
\tag{22}
$$

It measures local net outflow. Positive divergence means that more of the field flows out of a tiny region around $x$ than flows in; negative divergence means net inflow.

![Vector fields with positive, negative, and zero divergence](assets/course-mit-diffusion-2026/media/lecture-02/divergence-positive-negative-zero.jpg)

_Positive divergence behaves locally like a source, negative divergence like a sink, and zero divergence means no net local expansion or compression; it does not necessarily mean no motion. Diagram by Bfoshizzle1, via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Divergence_(captions).svg), licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)._

In one dimension there is only one component, so divergence is simply the ordinary spatial derivative

$$
\operatorname{div}(v)(x)=\frac{dv(x)}{dx}.
$$

For example, if $v(x)$ increases with $x$, the arrow at the right side of a small interval is larger than the arrow at the left side: more flows out than in, so the divergence is positive. The continuity equation applies this idea to the flow of probability.

For more visual explanations, see Khan Academy's articles on [divergence](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/divergence-and-curl-articles/a/divergence) and the [intuition behind the divergence formula](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/divergence-and-curl-articles/a/intuition-for-divergence-formula).

### The Continuity Equation

**Continuity equation.** Consider a flow model with vector field $u_t^{\mathrm{target}}$ and

$$
X_0\sim p_{\mathrm{init}}=p_0.
$$

Then $X_t\sim p_t$ for every $0\leq t\leq1$ if and only if

$$
\boxed{
\partial_t p_t(x)
=
-\operatorname{div}\!\left(p_tu_t^{\mathrm{target}}\right)(x)
}.
\tag{23}
$$

where $\partial_t p_t(x)=\frac{d}{dt}p_t(x)$ is the time derivative of the density at $x$. This is the **continuity equation**.

The left-hand side describes how much the probability density at $x$ changes over time. This change must equal the net inflow of probability mass. A particle $X_t$ follows the vector field $u_t^{\mathrm{target}}$, while divergence measures net outflow. Therefore, negative divergence measures net inflow.

The velocity must also be weighted by how much probability is present. The product $p_tu_t^{\mathrm{target}}$ is the probability flow: a region containing more probability transports more probability at the same velocity. Thus $-\operatorname{div}(p_tu_t^{\mathrm{target}})(x)$ is the net inflow of probability mass at $x$. Since probability mass is conserved, it must equal $\partial_t p_t(x)$.

<img src="assets/course-mit-diffusion-2026/media/lecture-02/continuity-equation-inflow-outflow.png" alt="Inflow and outflow around a small region in a vector field" width="420">

_The small box around $x$ gains or loses probability according to the balance between flow entering and leaving it. In the continuity equation, that flow is the probability current $p_tu_t$. Source: [Lecture 2 slides](assets/course-mit-diffusion-2026/lecture-02-flow-matching.pdf#page=20), slide 20._

#### The Fokker-Planck Equation

The continuity equation describes how a probability density evolves under the deterministic motion of an ODE. The **Fokker-Planck equation** generalizes it to the SDE

$$
X_0\sim p_{\mathrm{init}},
\qquad
dX_t=u_t(X_t)\,dt+\sigma_t\,dW_t,
$$

whose density evolves according to

$$
\boxed{
\partial_t p_t(x)
=
-\operatorname{div}(p_tu_t)(x)
+
\frac{\sigma_t^2}{2}\Delta p_t(x)
}.
\tag{108}
$$

There are now two ways for the density to change:

- The drift $u_t$ transports probability, giving the same $-\operatorname{div}(p_tu_t)$ term as in the continuity equation.
- Brownian noise spreads probability into nearby locations, giving the diffusion term $\frac{\sigma_t^2}{2}\Delta p_t$. The **Laplacian**

  $$
  \Delta p_t(x)
  =
  \sum_{i=1}^d\frac{\partial^2p_t(x)}{\partial x_i^2}
  $$

  adds the density's curvature along every coordinate. At a local peak it is typically negative, so diffusion lowers the density; at a local dip it is typically positive, so diffusion raises the density. In this way, the Laplacian describes probability spreading from concentrated regions into their surroundings.

When $\sigma_t=0$, no random noise is added and the diffusion term disappears. The Fokker-Planck equation then reduces exactly to the continuity equation.

The formal proof of the Fokker-Planck equation, including the continuity equation as its deterministic special case, is in [Appendix B](assets/course-mit-diffusion-2026/lecture-notes.pdf#page=72).

#### Proof That the Marginal Vector Field Generates the Marginal Probability Path

Recall that averaging the conditional probability paths over data points gives the [marginal probability path](#conditional-and-marginal-probability-paths)

$$
p_t(x)
=
\int p_t(x\mid z)p_{\mathrm{data}}(z)\,dz,
$$

and averaging their conditional velocities according to the posterior $p_t(z\mid x)$ gives the [marginal vector field](#the-marginalization-trick)

$$
u_t^{\mathrm{target}}(x)
=
\int u_t^{\mathrm{target}}(x\mid z)p_t(z\mid x)\,dz
=
\int
u_t^{\mathrm{target}}(x\mid z)
\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}\,dz.
$$

We want to prove that the ODE

$$
\frac{dX_t}{dt}=u_t^{\mathrm{target}}(X_t)
$$

actually moves samples through the marginal distributions $p_t$. In particular, it must turn $p_0=p_{\mathrm{init}}$ into $p_1=p_{\mathrm{data}}$. This is what turns the posterior-averaged field into an unconditional generative model rather than merely a formal average of conditional velocities.

The [continuity equation](#the-continuity-equation) gives the condition we need to check:

$$
\partial_t p_t(x)
=
-\operatorname{div}\!\left(p_tu_t^{\mathrm{target}}\right)(x).
$$

Each conditional pair $p_t(\cdot\mid z)$ and $u_t^{\mathrm{target}}(\cdot\mid z)$ already satisfies its own continuity equation. The calculation below averages those conditional equations over $z$ and shows that the result is exactly the marginal continuity equation:

$$
\begin{aligned}
\partial_t p_t(x)
&\overset{(i)}{=}
\partial_t\int p_t(x\mid z)p_{\mathrm{data}}(z)\,dz\\
&=
\int \partial_t p_t(x\mid z)p_{\mathrm{data}}(z)\,dz\\
&\overset{(ii)}{=}
-\int
\operatorname{div}\!\left(
p_t(\cdot\mid z)u_t^{\mathrm{target}}(\cdot\mid z)
\right)(x)
p_{\mathrm{data}}(z)\,dz\\
&\overset{(iii)}{=}
-\operatorname{div}\!\left(
\int
p_t(x\mid z)u_t^{\mathrm{target}}(x\mid z)p_{\mathrm{data}}(z)\,dz
\right)(x)\\
&\overset{(iv)}{=}
-\operatorname{div}\!\left(
p_t(x)
\int
u_t^{\mathrm{target}}(x\mid z)
\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}\,dz
\right)(x)\\
&\overset{(v)}{=}
-\operatorname{div}\!\left(p_tu_t^{\mathrm{target}}\right)(x).
\end{aligned}
$$

In $(i)$ we used the [sampling definition of the marginal path](#conditional-and-marginal-probability-paths); in $(ii)$, the continuity equation for each conditional probability path $p_t(\cdot\mid z)$; in $(iii)$, linearity to exchange the integral and divergence; in $(iv)$, multiplication and division by $p_t(x)$; and in $(v)$, the [definition of the marginal vector field](#the-marginalization-trick). Therefore $u_t^{\mathrm{target}}$ satisfies the continuity equation and follows the marginal probability path, proving the [marginalization result](#the-marginalization-trick).

### Learning the Marginal Vector Field

_**TL;DR:** Marginal flow matching states the ideal learning problem but cannot be evaluated. Conditional flow matching is computable from individual training examples and has exactly the same population gradient._

<img src="assets/course-mit-diffusion-2026/media/lecture-02/flow-matching-matrix.png" alt="The conditional and marginal rows of the flow matching construction" width="700">

_The flow-matching construction has two rows. A probability path defines the distributions from noise to data; its vector field defines the target dynamics; and its loss trains the neural network. Source: [Lecture 2 slides](assets/course-mit-diffusion-2026/lecture-02-flow-matching.pdf#page=32), slide 32._

<img src="assets/course-mit-diffusion-2026/media/lecture-02/conditional-flow-matching-summary-slide.png" alt="Conditional probability path, vector field, and flow matching loss" width="700">

_Conditional probability path, vector field, and flow-matching loss. Source: [Lecture 2 slides](assets/course-mit-diffusion-2026/lecture-02-flow-matching.pdf#page=33), slide 33._

<img src="assets/course-mit-diffusion-2026/media/lecture-02/marginal-flow-matching-summary-slide.png" alt="Marginal probability path, vector field, and flow matching loss" width="700">

_Marginal probability path, vector field, and flow-matching loss. Source: [Lecture 2 slides](assets/course-mit-diffusion-2026/lecture-02-flow-matching.pdf#page=34), slide 34._

The two rows serve different purposes:

- The **marginal row** describes the distribution and vector field we actually want for generation, but evaluating its density, vector field, or loss requires intractable integrals over the data distribution.
- The **conditional row** fixes one sampled data point $z$. Its probability path, vector field, and loss all have tractable analytical formulas.

There is one useful nuance: although the marginal density $p_t(x)$ cannot generally be evaluated, sampling from it is easy. Draw $z\sim p_{\mathrm{data}}$ and then $x\sim p_t(\cdot\mid z)$.

#### The Ideal but Intractable Objective

Ideally, the neural network would directly match the marginal vector field:

$$
\mathcal L_{\mathrm{FM}}(\theta)
=
\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\x\sim p_t}}
\left[
\left\|u_t^\theta(x)-u_t^{\mathrm{target}}(x)\right\|^2
\right].
\tag{24}
$$

Samples from $p_t$ are easy to obtain by first sampling $z\sim p_{\mathrm{data}}$ and then $x\sim p_t(\cdot\mid z)$, so the same loss can be written as

$$
\mathcal L_{\mathrm{FM}}(\theta)
=
\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\z\sim p_{\mathrm{data}}\\x\sim p_t(\cdot\mid z)}}
\left[
\left\|u_t^\theta(x)-u_t^{\mathrm{target}}(x)\right\|^2
\right].
\tag{25}
$$

The remaining obstacle is the label $u_t^{\mathrm{target}}(x)$: the [marginalization formula](#the-marginalization-trick) requires an intractable integral over the data distribution. Thus we can sample the input $x$, but we cannot compute the marginal velocity that should label it.

#### The Tractable Objective

The tractable alternative is the **conditional flow matching loss**

$$
\boxed{
\mathcal L_{\mathrm{CFM}}(\theta)
=
\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\z\sim p_{\mathrm{data}}\\x\sim p_t(\cdot\mid z)}}
\left[
\left\|u_t^\theta(x)-u_t^{\mathrm{target}}(x\mid z)\right\|^2
\right]
}.
\tag{26}
$$

| | Marginal flow matching | Conditional flow matching |
| --- | --- | --- |
| Regression target | Marginal velocity $u_t^{\mathrm{target}}(x)$ | Conditional velocity $u_t^{\mathrm{target}}(x\mid z)$ |
| Can the target be evaluated? | No: it requires the posterior average over the data distribution | Yes: it is known analytically for the sampled $z$ |
| Role | The ideal objective we want to minimize | The objective we can actually train with |
| Population result | Learns the marginal vector field directly | Learns the same marginal vector field through regression |

“Tractable” has a concrete meaning here: one training example can be produced by sampling $t$, sampling a dataset example $z$, sampling $x\sim p_t(\cdot\mid z)$, and evaluating $u_t^{\mathrm{target}}(x\mid z)$. This requires no evaluation of $p_{\mathrm{data}}(z)$, $p_t(x)$, or $p_t(z\mid x)$; no integral over the dataset; and no ODE simulation.

The network receives only $(x,t)$, not $z$. The sampled $z$ is used to construct the input and its target. Because the network does not know which $z$ produced $x$, squared-error regression forces it to average the compatible conditional velocities.

#### Why Conditional Flow Matching Learns the Marginal Vector Field

The key result is that the marginal and conditional flow-matching losses differ only by a constant that does not depend on $\theta$:

$$
\mathcal L_{\mathrm{FM}}(\theta)
=
\mathcal L_{\mathrm{CFM}}(\theta)+C.
$$

Consequently, their gradients are equal, so minimizing the tractable conditional loss also minimizes the intractable marginal loss. The lecture notes prove this by expanding the marginal mean-squared error into three terms:

$$
\begin{aligned}
\mathcal L_{\mathrm{FM}}(\theta)
&=
\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\x\sim p_t}}
\left[
\left\|u_t^\theta(x)-u_t^{\mathrm{target}}(x)\right\|^2
\right]\\
&=
\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\x\sim p_t}}
\left[\left\|u_t^\theta(x)\right\|^2\right]
-2\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\x\sim p_t}}
\left[u_t^\theta(x)^Tu_t^{\mathrm{target}}(x)\right]
+
\underbrace{
\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\x\sim p_t}}
\left[\left\|u_t^{\mathrm{target}}(x)\right\|^2\right]
}_{C_1}\\
&=
\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\z\sim p_{\mathrm{data}}\\x\sim p_t(\cdot\mid z)}}
\left[\left\|u_t^\theta(x)\right\|^2\right]
-2\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\x\sim p_t}}
\left[u_t^\theta(x)^Tu_t^{\mathrm{target}}(x)\right]
+C_1.
\end{aligned}
$$

The second line uses $\lVert a-b\rVert^2=\lVert a\rVert^2-2a^Tb+\lVert b\rVert^2$. The final term is called $C_1$ because it does not depend on $\theta$. In the last line, the first expectation is rewritten using the sampling procedure for the marginal path.

The crucial remaining step is rewriting the cross term that contains the intractable marginal vector field. Substitute the [posterior average that defines the marginal vector field](#the-marginalization-trick):

$$
\begin{aligned}
&\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\x\sim p_t}}
\left[u_t^\theta(x)^Tu_t^{\mathrm{target}}(x)\right]\\
&=
\int_0^1\!\int
p_t(x)u_t^\theta(x)^Tu_t^{\mathrm{target}}(x)\,dx\,dt\\
&=
\int_0^1\!\int
p_t(x)u_t^\theta(x)^T
\left[
\int
u_t^{\mathrm{target}}(x\mid z)
\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}\,dz
\right]dx\,dt\\
&=
\int_0^1\!\int\!\int
u_t^\theta(x)^Tu_t^{\mathrm{target}}(x\mid z)
p_t(x\mid z)p_{\mathrm{data}}(z)
\,dz\,dx\,dt\\
&=
\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\z\sim p_{\mathrm{data}}\\x\sim p_t(\cdot\mid z)}}
\left[u_t^\theta(x)^Tu_t^{\mathrm{target}}(x\mid z)\right].
\end{aligned}
$$

This equality is the heart of the proof: its left side uses the marginal vector field, while its right side uses the tractable conditional vector field. Substitute it back into the expanded marginal loss, then add and subtract the squared norm of the conditional vector field:

$$
\begin{aligned}
\mathcal L_{\mathrm{FM}}(\theta)
&=
\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\z\sim p_{\mathrm{data}}\\x\sim p_t(\cdot\mid z)}}
\left[
\left\|u_t^\theta(x)\right\|^2
-2u_t^\theta(x)^Tu_t^{\mathrm{target}}(x\mid z)
\right]+C_1\\
&=
\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\z\sim p_{\mathrm{data}}\\x\sim p_t(\cdot\mid z)}}
\left[
\left\|u_t^\theta(x)-u_t^{\mathrm{target}}(x\mid z)\right\|^2
-\left\|u_t^{\mathrm{target}}(x\mid z)\right\|^2
\right]+C_1\\
&=
\mathcal L_{\mathrm{CFM}}(\theta)
+
\underbrace{
\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\z\sim p_{\mathrm{data}}\\x\sim p_t(\cdot\mid z)}}
\left[-\left\|u_t^{\mathrm{target}}(x\mid z)\right\|^2\right]
}_{C_2}
+C_1\\
&=
\mathcal L_{\mathrm{CFM}}(\theta)+C,
\qquad C=C_1+C_2.
\end{aligned}
$$

Both $C_1$ and $C_2$ are independent of $\theta$. Therefore,

$$
\nabla_\theta\mathcal L_{\mathrm{CFM}}(\theta)
=
\nabla_\theta\mathcal L_{\mathrm{FM}}(\theta).
$$

Thus conditional flow matching and direct flow matching have the same population gradient and the same minimizers. CFM is not an approximation to the marginal objective: it is the tractable loss whose parameter-dependent part is exactly the same.

The lecture notes emphasize three striking features of this training algorithm:

- **Simulation-free:** We never actually simulate any ODE during training. This makes training extremely cheap because we do not have to roll out ODE trajectories, which take many steps.
- **Similar to supervised learning:** Training is a simple regression objective against $u_t^{\mathrm{target}}(x\mid z)$.
- **Extremely simple algorithm:** It is hard to think of a much simpler training objective. This makes flow matching especially appealing for large-scale machine-learning models.

#### What Is Used at Generation Time

After training, generation uses the learned marginal field:

$$
X_0\sim p_{\mathrm{init}},
\qquad
dX_t=u_t^\theta(X_t)\,dt.
\tag{27}
$$

Numerically solving this ODE from $t=0$ to $t=1$ produces the generated sample $X_1$.

### Gaussian and CondOT Flow Matching

For the Gaussian conditional path, sample

$$
z\sim p_{\mathrm{data}},
\qquad
\epsilon\sim\mathcal N(0,I_d),
\qquad
x_t=\alpha_tz+\beta_t\epsilon.
\tag{28}
$$

The [Gaussian conditional vector field derived above](#gaussian-conditional-vector-field-and-its-derivation-example-10), evaluated at this sampled $x_t$, is

$$
u_t^{\mathrm{target}}(x_t\mid z)
=
\dot\alpha_tz+\dot\beta_t\epsilon.
$$

Therefore, Gaussian conditional flow matching trains on

$$
\boxed{
\mathcal L_{\mathrm{CFM}}(\theta)
=
\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\z\sim p_{\mathrm{data}}\\\epsilon\sim\mathcal N(0,I_d)}}
\left[
\left\|
u_t^\theta(x_t)
-(\dot\alpha_tz+\dot\beta_t\epsilon)
\right\|^2
\right]
}.
\tag{31}
$$

For the [straight-line CondOT schedule derived above](#gaussian-conditional-vector-field-and-its-derivation-example-10), this becomes

$$
x_t=tz+(1-t)\epsilon,
\qquad
u_t^{\mathrm{target}}(x_t\mid z)=z-\epsilon,
$$

and the objective becomes

$$
\boxed{
\mathcal L_{\mathrm{CFM}}(\theta)
=
\mathbb E_{\substack{t\sim\operatorname{Unif}[0,1]\\z\sim p_{\mathrm{data}}\\\epsilon\sim\mathcal N(0,I_d)}}
\left[
\left\|u_t^\theta(x_t)-(z-\epsilon)\right\|^2
\right]
}.
$$

The input is a linear interpolation of noise and data, and the target is the constant velocity from that noise sample to that data sample.

### Flow Matching Training Procedure

For the Gaussian CondOT path, one training step is:

1. Sample a data example $z$ from the dataset.
2. Sample $t\sim\operatorname{Unif}[0,1]$.
3. Sample $\epsilon\sim\mathcal N(0,I_d)$.
4. Form $x_t=tz+(1-t)\epsilon$.
5. Compute $\|u_t^\theta(x_t)-(z-\epsilon)\|^2$ and update $\theta$.

These steps instantiate the simulation-free regression objective described above. ODE simulation is needed only after training, when generation starts from fresh Gaussian noise and follows the learned marginal field to $t=1$.

The lecture highlights Stable Diffusion 3 and Meta Movie Gen as large-scale examples trained with flow-matching-style objectives. Their architectures and datasets are much more elaborate, but the basic training signal is the regression objective above.

### Flow Matching Summary

Flow matching separates the construction into tractable conditional objects and useful marginal objects:

1. Choose $p_t(x\mid z)$, a conditional probability path from noise to one data point $z$.
2. Derive $u_t^{\mathrm{target}}(x\mid z)$, a tractable conditional vector field that follows that path.
3. Train $u_t^\theta(x)$ with conditional flow matching. Although the targets depend on sampled $z$, the network learns the marginal field because squared-error regression averages over compatible data points.
4. Generate by sampling $X_0\sim p_{\mathrm{init}}$ and solving $dX_t=u_t^\theta(X_t)\,dt$ to $t=1$.

For the Gaussian CondOT path, the complete training pair is simply

$$
\boxed{
x_t=tz+(1-t)\epsilon,
\qquad
\text{target}=z-\epsilon
}.
$$

<img src="assets/course-mit-diffusion-2026/media/lecture-02/learned-flow-matching-path.png" alt="Ground-truth and learned marginal probability paths" width="520">

_The ground-truth marginal path (top) and samples produced by the trained flow-matching ODE (bottom) closely agree. Source: Figure 7 of the [lecture notes](assets/course-mit-diffusion-2026/lecture-notes.pdf)._
