# [Introduction to Flow Matching and Diffusion Models, MIT](https://diffusion.csail.mit.edu/2026/index.html)

- **Created**: 2026-08-04
- **Last Updated**: 2026-08-11
- **Status**: `In Progress`
- **Related**:
  - [[papers-diffusion]] — Broader reading list covering the foundations, objectives, architectures, and applications of diffusion models.

---

- **Course**: <https://diffusion.csail.mit.edu/2026/index.html>
- [Lecture Notes](assets/course-mit-diffusion-2026/lecture-notes.pdf)

---

| Done | Lecture | Topic | Slides | Recording | Lecture Notes | Lab | Additional Material |
| :--: | :--: | --- | :--: | :--: | --- | --- | --- |
| ☐ | 1 | **Flow and Diffusion Models**<br>• Introduction to generative models<br>• Ordinary and stochastic differential equations<br>• Sampling from flow and diffusion models | [slides](assets/course-mit-diffusion-2026/lecture-01-flow-and-diffusion-models.pdf) | [recording](https://www.youtube.com/watch?v=9eJQQVrUUoI) | [§§1-2](assets/course-mit-diffusion-2026/lecture-notes.pdf) | [Lab 1: ODEs and SDEs](https://github.com/eje24/iap-diffusion-labs/blob/2026/labs/lab_one.ipynb) | [3blue1brown: ODEs](https://www.3blue1brown.com/lessons/differential-equations/)<br>[3blue1brown: PDEs](https://www.3blue1brown.com/lessons/pdes/)<br>[Khan: ODE basics](https://www.khanacademy.org/math/differential-equations/first-order-differential-equations) |
| ☐ | 2 | **Flow Matching**<br>• Conditional and marginal probability path<br>• Conditional and marginal vector field<br>• Flow matching training objective | [slides](assets/course-mit-diffusion-2026/lecture-02-flow-matching.pdf) | [recording](https://www.youtube.com/watch?v=PNkMKWW8Khw) | [§3](assets/course-mit-diffusion-2026/lecture-notes.pdf) | [Lab 2: Flow and Score Matching](https://github.com/eje24/iap-diffusion-labs/blob/2026/labs/lab_two.ipynb) | — |
| ☐ | 3-A | **Score Functions and Score Matching**<br>• Score functions<br>• Denoising score matching<br>• SDE sampling | [slides](assets/course-mit-diffusion-2026/lecture-03-score-matching-and-guidance.pdf) | [recording](https://www.youtube.com/watch?v=ngC3QnYSVNM) | [§4](assets/course-mit-diffusion-2026/lecture-notes.pdf) | [Lab 2: Flow and Score Matching](https://github.com/eje24/iap-diffusion-labs/blob/2026/labs/lab_two.ipynb) | — |
| ☐ | 3-B | **Classifier-free Guidance**<br>• Guided generation<br>• Classifier guidance<br>• Classifier-free guidance | [slides](assets/course-mit-diffusion-2026/lecture-03-score-matching-and-guidance.pdf) | [recording](https://www.youtube.com/watch?v=8oWZ1bHwyRI) | [§5](assets/course-mit-diffusion-2026/lecture-notes.pdf) | — | — |
| ☐ | 4 | **Latent Spaces and Neural Network Architectures**<br>• Variational autoencoders and latent spaces<br>• Diffusion Transformer and U-Nets<br>• Case studies: Large-scale models | [slides](assets/course-mit-diffusion-2026/lecture-04-latent-spaces-and-neural-network-architectures.pdf) | [recording](https://www.youtube.com/watch?v=g0MB1CCBmsI) | [§6](assets/course-mit-diffusion-2026/lecture-notes.pdf) | [Lab 3: Diffusion Transformer and VAEs](https://github.com/eje24/iap-diffusion-labs/blob/2026/labs/lab_three.ipynb) | — |
| ☐ | 5 | **Discrete Diffusion Models**<br>• Continuous-time Markov chains (CTMCs)<br>• Sampling from CTMC models<br>• Training CTMC models | [slides](assets/course-mit-diffusion-2026/lecture-05-discrete-diffusion-models.pdf) | [recording](https://www.youtube.com/watch?v=d0kmyEJN2hI) | [§7](assets/course-mit-diffusion-2026/lecture-notes.pdf) | — | — |
| ☐ | — | Supplementary mathematical background | — | — | [Appendices A-C](assets/course-mit-diffusion-2026/lecture-notes.pdf) | — | — |
| ☐ | — | Additional VAE perspectives and diffusion literature guide | — | — | [Appendices D-E](assets/course-mit-diffusion-2026/lecture-notes.pdf) | — | — |

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

A **trajectory** is a time-indexed position. To make its dependence on the starting point explicit, write

$$
X^{x_0} : [0,1] \to \mathbb{R}^d,
\qquad
t \mapsto X_t^{x_0}.
$$

A **time-dependent vector field** assigns a velocity to every possible position and time:

$$
u : \mathbb{R}^d \times [0,1] \to \mathbb{R}^d,
\qquad
(x,t) \mapsto u_t(x).
$$

The vector $u_t(x)$ answers a local question: _if the state were at $x$ at time $t$, in which direction and how quickly should it move?_ It does not directly tell us the final destination.

Given a fixed initial state $x_0$, the solution of the initial-value problem is one trajectory $X^{x_0}$. The ODE requires this trajectory's instantaneous velocity to equal the vector field evaluated at its current location:

$$
\frac{dX_t^{x_0}}{dt} = u_t\!\left(X_t^{x_0}\right),
\qquad
X_0^{x_0}=x_0.
$$

The dependence $u_t(X_t^{x_0})$ is important. As $X_t^{x_0}$ moves, it enters a new part of the field, receives a new velocity, and bends accordingly. This is the higher-dimensional version of a slope field: arrows describe local derivatives, while a solution curve follows those arrows.

The same equation can be written in integral form:

$$
X_t^{x_0}
=
x_0 + \int_0^t u_s\!\left(X_s^{x_0}\right)\,ds.
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
\boxed{\psi_t(x_0)=X_t^{x_0}}.
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

For a random initial condition $X_0$, we select one of these trajectories at random:

$$
X_t=\psi_t(X_0).
$$

The three concepts differ mainly in viewpoint:

- $u_t(x)$ is the full field of local instructions.
- $X_t^{x_0}$ is one solution path obtained from one fixed initial condition $x_0$.
- $\psi_t$ maps every possible $x_0$ to its position at time $t$, so it collects all the solution paths.

In the wind analogy, $u_t(x)$ is the wind velocity everywhere, $X_t^{x_0}$ is the path of one balloon released at $x_0$, and $\psi_t$ is the map saying where a balloon released from every possible starting point would be at time $t$.

So the causal chain is

$$
\text{vector field}
\longrightarrow
\text{ODE}
\longrightarrow
\text{flow}.
$$

The model will eventually learn or parameterize the **vector field**, while an ODE solver computes the resulting trajectory/flow. The neural network is not usually asked to output $X_1$ in a single jump.

**Sample path versus distribution.** The ODE moves each individual sample. If $X_0 \sim p_0$, applying the flow to all possible initial samples induces a time-dependent distribution

$$
p_t = (\psi_t)_{\#}p_0,
$$

read as “$p_t$ is the pushforward of $p_0$ through $\psi_t$.” At the population level this density obeys the continuity equation

$$
\partial_t p_t(x)
=
-\nabla \cdot \bigl(p_t(x)u_t(x)\bigr).
$$

Thus an ODE describes the motion of a sample, whereas a PDE describes how the entire probability density changes. This is the bridge between the first two 3Blue1Brown videos linked in the table.

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

This is also a simple **gradient flow**. For

$$
V(x)=\frac{\theta}{2}\lVert x\rVert^2,
$$

we have $u(x)=-\nabla V(x)$. The system moves downhill in the quadratic potential toward its minimum. Gradient descent will reappear below as a discrete update with the same shape as Euler's method.

### Simulating an ODE with Euler's Method

_**TL;DR:** Euler's method repeatedly freezes the current velocity for a short time and takes a straight step in that direction._

For a neural vector field, there is generally no closed-form expression for $\psi_t$. We approximate the trajectory on a grid

$$
t_k=kh,
\qquad
h=\frac{1}{n},
\qquad
k=0,\ldots,n.
$$

Starting from the exact integral over one step,

$$
X_{t+h}
=
X_t+\int_t^{t+h}u_s(X_s)\,ds,
$$

Euler's method approximates the changing velocity throughout the interval by its value at the left endpoint:

$$
u_s(X_s)\approx u_t(X_t)
\qquad
(s\in[t,t+h]).
$$

Therefore,

$$
\boxed{
X_{t+h}=X_t+h\,u_t(X_t)
}
\qquad
t=0,h,2h,\ldots,1-h.
$$

Equivalently, this is the first-order Taylor approximation

$$
\begin{aligned}
X_{t+h}
&=X_t+h\frac{dX_t}{dt}+O(h^2) \\
&=X_t+h\,u_t(X_t)+O(h^2).
\end{aligned}
$$

The geometric interpretation is:

1. evaluate the arrow $u_t(X_t)$ at the current location;
2. multiply it by the time interval $h$ to turn velocity into displacement;
3. move by that displacement;
4. evaluate the field again at the new state.

With a smooth vector field, one Euler step has local truncation error $O(h^2)$, while the accumulated error over the fixed interval $[0,1]$ is $O(h)$. More, smaller steps usually improve the trajectory but require more vector-field evaluations. In a generative model, each evaluation is normally a neural-network forward pass, so solver accuracy is traded against sampling cost.

For the linear example $u(x)=-\theta x$, Euler gives

$$
X_{(k+1)h}
=
X_{kh}-h\theta X_{kh}
=
(1-\theta h)X_{kh}.
$$

After $k$ steps,

$$
X_{kh}=(1-\theta h)^k x_0.
$$

Using $h=1/n$ and taking $n$ steps to $t=1$:

$$
X_1^{\mathrm{Euler}}
=
\left(1-\frac{\theta}{n}\right)^n x_0
\xrightarrow[n\to\infty]{}
e^{-\theta}x_0,
$$

which recovers the exact flow. It also shows why a step can be too large: if $\theta h$ is large, the multiplier $1-\theta h$ can overshoot, oscillate in sign, or even grow in magnitude instead of representing smooth decay.

**Connection to residual networks.** The Euler update has exactly the form of a residual block:

$$
x_{k+1}=x_k+h\,u_\theta(x_k,t_k).
$$

A residual network composes a finite sequence of small learned changes; a neural ODE can be viewed as the continuous-depth limit, with an ODE solver choosing how to discretize that continuous vector field. Euler's method performs the simulation—it does not tell us how to train $u_\theta$. Flow matching will provide that learning objective later.

### Improving Euler with Heun's Method

_**TL;DR:** Euler uses only the velocity at the beginning of a step; Heun predicts the endpoint, evaluates the velocity there too, and moves using the average of the two velocities._

Euler's method approximates

$$
\int_t^{t+h}u_s(X_s)\,ds
$$

using a rectangle whose height is the starting velocity $u_t(X_t)$. If the trajectory curves or the vector field changes during the interval, that single velocity can be a poor description of the whole step.

Heun's method instead approximates the integral with a trapezoid. The true endpoint $X_{t+h}$ is not yet known, so it first uses Euler to predict one:

$$
\widetilde{X}_{t+h}
=
X_t+h\,u_t(X_t).
$$

It evaluates the vector field again at the predicted endpoint and corrects the step using the average velocity:

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

For a sufficiently smooth ODE, Heun's method has local error $O(h^3)$ and accumulated global error $O(h^2)$, compared with Euler's $O(h^2)$ local and $O(h)$ global errors. The cost is two vector-field evaluations per step instead of one. For a neural vector field, this normally means twice as many network evaluations for the same number of steps.

For the linear field $u(x)=-\theta x$, the predictor is

$$
\widetilde{X}_{t+h}=(1-\theta h)X_t.
$$

The corrected step becomes

$$
\begin{aligned}
X_{t+h}
&=X_t+\frac{h}{2}
\left[-\theta X_t-\theta\widetilde{X}_{t+h}\right] \\
&=\left(1-\theta h+\frac{\theta^2h^2}{2}\right)X_t.
\end{aligned}
$$

This matches the first three terms of

$$
e^{-\theta h}
=
1-\theta h+\frac{\theta^2h^2}{2}+O(h^3),
$$

whereas Euler retains only $1-\theta h$. Heun is thus a concrete example of spending an additional model evaluation to follow curvature more accurately.

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

**Why $\sqrt{h}$ rather than $h$?** If $\epsilon_t\sim\mathcal{N}(0,I_d)$, then

$$
\operatorname{Var}(\sqrt{h}\,\epsilon_t)=hI_d.
$$

Over $n=1/h$ independent steps spanning one unit of time, the variances add:

$$
n h I_d=I_d.
$$

Scaling the random increment by $h$ would instead produce total variance $nh^2=h\to0$, causing the randomness to disappear as the grid is refined. Using unscaled noise would make the total variance diverge.

The $\sqrt{h}$ scaling has an important consequence. Over a short interval,

$$
\text{deterministic displacement}=O(h),
\qquad
\text{Brownian displacement}=O(\sqrt{h}).
$$

Brownian paths are continuous but, with probability one, nowhere differentiable. This is why an SDE cannot be interpreted as an ordinary differential equation driven by a conventional derivative $dW_t/dt$.

### From ODEs to SDEs

_**TL;DR:** An SDE combines a directed drift with Brownian spreading: locally it moves by $h$ times the drift plus $\sqrt h$ times fresh Gaussian noise._

For an ODE, a small-time update is

$$
X_{t+h}
=
X_t+h\,u_t(X_t)+hR_t(h),
$$

where $R_t(h)\to0$ as $h\to0$. Adding a Brownian increment gives

$$
X_{t+h}
=
X_t
+h\,u_t(X_t)
+\sigma_t(W_{t+h}-W_t)
+hR_t(h).
$$

This motivates the symbolic SDE notation

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

The notation $dW_t$ is symbolic. The corresponding integral equation is

$$
X_t
=
X_0
+\int_0^t u_s(X_s)\,ds
+\int_0^t\sigma_s\,dW_s.
$$

The last term is an Itô stochastic integral rather than an ordinary Riemann integral. The course avoids developing the full stochastic-calculus machinery and works through the simulation rule instead.

**What happened to the flow map?** An ODE has a deterministic map $x_0\mapsto\psi_t(x_0)$. For an SDE, $X_t$ is not determined by $x_0$ alone; it also depends on the Brownian path. One can define a random flow after fixing that Brownian path, but there is no single deterministic map of $x_0$ that gives every outcome.

The marginal distribution $p_t$ is nevertheless well-defined and evolves deterministically. For state-independent scalar $\sigma_t$, its evolution is described by the Fokker-Planck equation

$$
\frac{\partial p_t(x)}{\partial t}
=
-\nabla\cdot\left(p_t(x)u_t(x)\right)
+\frac{\sigma_t^2}{2}\Delta p_t(x).
$$

The first term transports probability according to the drift, just as in the ODE continuity equation. The second term spreads probability through diffusion. Setting $\sigma_t=0$ recovers the ODE case.

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

Using the integrating factor $e^{\theta t}$ gives the exact solution

$$
X_t
=
e^{-\theta t}X_0
+\sigma\int_0^t e^{-\theta(t-s)}\,dW_s.
$$

For fixed $X_0=x_0$, the stochastic integral is Gaussian, so

$$
X_t\mid X_0=x_0
\sim
\mathcal{N}\!\left(
e^{-\theta t}x_0,
\frac{\sigma^2}{2\theta}
\left(1-e^{-2\theta t}\right)I_d
\right).
$$

The mean decays toward zero:

$$
\mathbb{E}[X_t\mid X_0=x_0]=e^{-\theta t}x_0,
$$

while the variance grows from zero toward a finite limit:

$$
\operatorname{Var}(X_t\mid X_0=x_0)
=
\frac{\sigma^2}{2\theta}
\left(1-e^{-2\theta t}\right)I_d.
$$

Hence, as $t\to\infty$,

$$
X_t
\xrightarrow{d}
\mathcal{N}\!\left(0,\frac{\sigma^2}{2\theta}I_d\right).
$$

This explains the trajectories in Figure 3 of the lecture notes:

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

DDPM papers and the `offline_atari` implementation usually label the easy corruption direction oppositely:

$$
x_0\sim p_{\mathrm{data}}
\longrightarrow
x_1\approx\mathcal{N}(0,I),
$$

and generate by integrating from $t=1$ back to $t=0$. The substance is the same after reversing or relabeling time; the symbols $x_0$ and $x_1$ do not intrinsically mean “noise” and “data.” In particular, `ConditionalOTFlow` in `offline_atari` is a flow model in the lecture's terminology: its sampling dynamics contain no Brownian term, even though it lives inside the broader diffusion interface.

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

"Marginal" means that the latent choice of $z$ has been averaged out. Equation (12) is tractable with a dataset: sample a training example and corrupt it. Equation (13) is generally intractable because evaluating $p_t(x)$ requires integrating over all possible clean data points.

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

A particularly simple choice is the conditional optimal-transport, or CondOT, schedule

$$
\alpha_t=t,
\qquad
\beta_t=1-t,
$$

which gives the straight interpolation

$$
X_t=tZ+(1-t)\epsilon.
$$

The schedules need not satisfy $\alpha_t^2+\beta_t^2=1$ for general flow matching. Variance-preserving diffusion schedules often impose such a relation, but CondOT instead prioritizes straight conditional trajectories.

**Time convention.** In these notes, $t=0$ is noise and $t=1$ is data. Many DDPM presentations call clean data $x_0$ and increase noise with $t$. The formulas can be translated by reversing or relabeling time; always check which endpoint is noise.

**A path is not yet a dynamics.** Equation (16) can be used in two ways:

- Drawing a fresh independent $\epsilon$ for every $t$ produces correct snapshots from $p_t(\cdot\mid z)$ but does not connect them into a trajectory.
- Holding the same $\epsilon$ fixed while varying $t$ couples the snapshots into the path $X_t=\alpha_tz+\beta_t\epsilon$. This coupling will produce the convenient conditional vector field below.

The fixed $\epsilon$ here is an initial latent variable, not a stream of Brownian increments. An SDE simulation instead adds fresh noise throughout time.

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

The distinction between a path and a vector field is essential:

- $p_t(x\mid z)$ is a scalar density describing where probability mass is found at time $t$;
- $u_t(x\mid z)\in\mathbb R^d$ is a velocity describing the direction and speed of motion at $(x,t)$.

Consequently,

$$
\frac{p_t(x\mid z)-p_{t-\Delta t}(x\mid z)}{\Delta t}
$$

approximates the scalar density derivative $\partial_tp_t(x\mid z)$, not the vector velocity. The density derivative constrains a vector field through the continuity equation

$$
\partial_tp_t(x\mid z)
=
-\nabla\cdot\bigl(p_t(x\mid z)u_t(x\mid z)\bigr),
$$

but it does not uniquely determine $u_t$. For example, a rotational field can move particles around inside an isotropic Gaussian without changing the Gaussian density at all. The same probability snapshots can therefore admit different particle trajectories.

By contrast, if $X_t$ and $X_{t-\Delta t}$ are coupled points on the same trajectory, then

$$
\frac{X_t-X_{t-\Delta t}}{\Delta t}
\longrightarrow
\frac{dX_t}{dt}
$$

does give the trajectory's velocity. The Gaussian construction obtains its vector field from exactly such a shared-$\epsilon$ coupling.

The conditional field is analytically convenient but not itself a useful unconditional generator: it requires knowing $z$, and every trajectory ends at that already-known $z$. Its role is to provide tractable building blocks and, later, tractable regression targets.

### The Marginalization Trick

_**TL;DR:** At a noisy location $x$, average the conditional velocities using the posterior probability that each clean datum $z$ generated that $x$._

Theorem 9 defines the marginal vector field

$$
u_t^{\mathrm{target}}(x)
=
\int
u_t^{\mathrm{target}}(x\mid z)
\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)}
\,dz.
\tag{18}
$$

By Bayes' rule,

$$
p_t(z\mid x)
=
\frac{p_t(x\mid z)p_{\mathrm{data}}(z)}{p_t(x)},
$$

so Equation (18) is the conditional expectation

$$
\boxed{
u_t^{\mathrm{target}}(x)
=
\mathbb E\!\left[
u_t^{\mathrm{target}}(x\mid Z)
\mid X_t=x
\right].
}
$$

The weighting must use the posterior $p_t(z\mid x)$, not just the prior $p_{\mathrm{data}}(z)$. The prior says how common $z$ is globally; the posterior says how much of the probability mass currently located at this particular $x$ came from the component indexed by $z$.

A probability-current view makes both the likelihood factor and the denominator unavoidable. The component associated with $z$ contributes local density

$$
\rho_z(x)=p_t(x\mid z)p_{\mathrm{data}}(z)
$$

and local probability current

$$
J_z(x)=\rho_z(x)u_t(x\mid z).
$$

After forgetting the component label $z$, currents and densities add:

$$
J(x)=\int J_z(x)\,dz,
\qquad
p_t(x)=\int\rho_z(x)\,dz.
$$

The effective velocity is current divided by density,

$$
u_t(x)
=
\frac{J(x)}{p_t(x)}
=
\frac{\int p_t(x\mid z)p_{\mathrm{data}}(z)u_t(x\mid z)\,dz}{p_t(x)},
$$

which is Equation (18). Matching the total current makes the marginal density obey the same continuity equation as the mixture of conditional paths. Therefore

$$
X_0\sim p_{\mathrm{init}},
\qquad
\frac{dX_t}{dt}=u_t^{\mathrm{target}}(X_t)
\quad\Longrightarrow\quad
X_t\sim p_t,
\tag{19}
$$

and in particular $X_1\sim p_{\mathrm{data}}$.

For a concrete two-mode CondOT example, suppose $Z\in\{-a,+a\}$ with equal prior probability. The conditional velocities derived below are

$$
u_t(x\mid z)=\frac{z-x}{1-t}.
$$

Using only the prior would give

$$
\frac12u_t(x\mid+a)+\frac12u_t(x\mid-a)
=
-\frac{x}{1-t},
$$

which pushes everything toward the mean zero. The correct marginal field is instead

$$
u_t(x)
=
\frac{\mathbb E[Z\mid X_t=x]-x}{1-t}.
$$

Positive $x$ values that are more plausibly noisy versions of $+a$ receive more of the $+a$ velocity; negative values receive more of the $-a$ velocity. The averaging is local and $x$-dependent, so the marginal flow can produce multiple modes rather than collapse to the global mean.

The theorem guarantees the correct distribution at every time. It does not say that a trajectory of the marginal ODE secretly chooses one fixed $z$ and follows that conditional trajectory. Once $z$ is marginalized out, the deterministic ODE uses only the posterior-averaged velocity at its current location.

### Gaussian Conditional Vector Field and Its Derivation (Example 10)

_**TL;DR:** Couple all times with the same initial noise, differentiate that explicit flow, and rewrite the velocity in terms of the current state._

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

To derive it without guessing, first define an explicit conditional flow map using an initial point $x_0$:

$$
\psi_t^{\mathrm{target}}(x_0\mid z)
=
\alpha_tz+\beta_tx_0.
\tag{21}
$$

If $X_0\sim\mathcal N(0,I_d)$, then

$$
X_t
=
\psi_t^{\mathrm{target}}(X_0\mid z)
=
\alpha_tz+\beta_tX_0
\sim
\mathcal N(\alpha_tz,\beta_t^2I_d)
=
p_t(\cdot\mid z).
$$

Thus this flow has the desired conditional snapshots. Differentiate it while holding $x_0$ and $z$ fixed:

$$
\frac{d}{dt}\psi_t^{\mathrm{target}}(x_0\mid z)
=
\dot\alpha_tz+\dot\beta_tx_0.
$$

This is a trajectory velocity expressed using its starting point $x_0$. A vector field must instead be a function of the current state

$$
x=\psi_t^{\mathrm{target}}(x_0\mid z)
=
\alpha_tz+\beta_tx_0.
$$

For $0\leq t<1$ with $\beta_t>0$, solve for the initial point:

$$
x_0
=
\frac{x-\alpha_tz}{\beta_t}.
$$

Substitution gives

$$
\begin{aligned}
u_t^{\mathrm{target}}(x\mid z)
&=\dot\alpha_tz
+\dot\beta_t\frac{x-\alpha_tz}{\beta_t}\\
&=\left(
\dot\alpha_t
-\frac{\dot\beta_t}{\beta_t}\alpha_t
\right)z
+\frac{\dot\beta_t}{\beta_t}x,
\end{aligned}
$$

which is Equation (20).

Equivalently, along the shared-noise coupling $X_t=\alpha_tz+\beta_t\epsilon$,

$$
\frac{X_t-X_{t-\Delta t}}{\Delta t}
\longrightarrow
\dot\alpha_tz+\dot\beta_t\epsilon.
$$

Replacing $\epsilon=(x-\alpha_tz)/\beta_t$ yields the same vector field. This is why differencing **coupled sample locations** can recover velocity whereas differencing density values cannot.

For CondOT, $\alpha_t=t$ and $\beta_t=1-t$, so $\dot\alpha_t=1$ and $\dot\beta_t=-1$. Equation (20) simplifies to

$$
\begin{aligned}
u_t^{\mathrm{target}}(x\mid z)
&=\frac{z-x}{1-t}\\
&=z-\epsilon
\qquad\text{when }x=tz+(1-t)\epsilon.
\end{aligned}
$$

The second form is especially intuitive: the conditional trajectory is a straight line from initial noise $\epsilon$ to data $z$, so its constant velocity is simply endpoint minus starting point. The expression $(z-x)/(1-t)$ appears singular at $t=1$ only because it infers that constant velocity from a vanishing remaining displacement and time. Along a valid trajectory the ratio has the finite limit $z-\epsilon$.

Up to this point, the analytically known objects are conditional on $z$, while the useful generative field $u_t^{\mathrm{target}}(x)$ involves an intractable integral over all $z$. The next part of Lecture 2 explains how conditional flow matching trains a neural network on tractable conditional targets while implicitly learning this marginal field.
