# [Introduction to Flow Matching and Diffusion Models, MIT](https://diffusion.csail.mit.edu/2026/index.html)

- **Created**: 2026-08-04
- **Last Updated**: 2026-08-08
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
