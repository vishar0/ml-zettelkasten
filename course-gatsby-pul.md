# [Probabilistic and Unsupervised Learning, Gatsby](https://www.gatsby.ucl.ac.uk/teaching/courses/ml1/)

- **Created**: 2026-05-09
- **Last Updated**: 2026-05-24
- **Status**: `Not Started`

---

- <https://www.gatsby.ucl.ac.uk/teaching/courses/ml1/>

---

**Probabilistic and Unsupervised Learning**:

- [x] Lecture 1: [Introduction to Probabilistic Learning](assets/course-gatsby-pul-slides-2025/lecture-01-05-probabilistic-unsupervised-learning.pdf)
- [ ] Lecture 2: [Latent Variable Models](assets/course-gatsby-pul-slides-2025/lecture-01-05-probabilistic-unsupervised-learning.pdf)
- [ ] Lecture 3: [EM Algorithm and Latent Chain Models](assets/course-gatsby-pul-slides-2025/lecture-01-05-probabilistic-unsupervised-learning.pdf)
- [ ] Lecture 4: [Markov Chains and MCMC](assets/course-gatsby-pul-slides-2025/lecture-01-05-probabilistic-unsupervised-learning.pdf)
- [ ] Lecture 5: [Optimisation](assets/course-gatsby-pul-slides-2025/lecture-01-05-probabilistic-unsupervised-learning.pdf)

**Approximate Inference**:

- [ ] Lecture 6: [Graphical Models](assets/course-gatsby-pul-slides-2025/lecture-06-graphical-models.pdf)
- [ ] Lecture 7: [Factored Variational Approximations and Variational Bayes](assets/course-gatsby-pul-slides-2025/lecture-07-factored-variational-approximations-vb.pdf)
- [ ] Lecture 8: [Bayesian Model Selection, Hyperparameter Optimisation, and Gaussian Processes](assets/course-gatsby-pul-slides-2025/lecture-08-bayesian-model-selection-gps.pdf)
- [ ] Lecture 9: [Expectation Propagation](assets/course-gatsby-pul-slides-2025/lecture-09-expectation-propagation.pdf)
- [ ] Lecture 10: [Belief Propagation](assets/course-gatsby-pul-slides-2025/lecture-10-belief-propagation.pdf)
- [ ] Lecture 11: [Exponential Families: Convexity, Duality, and Free Energies](assets/course-gatsby-pul-slides-2025/lecture-11-exponential-families-free-energies.pdf)
- [ ] Lecture 12: [Parametric Variational Methods and Recognition Models](assets/course-gatsby-pul-slides-2025/lecture-12-parametric-variational-methods-recognition-models.pdf)

---

## Lecture 1: [Introduction to Probabilistic Learning](assets/course-gatsby-pul-slides-2025/lecture-01-05-probabilistic-unsupervised-learning.pdf)

### Three Learning Problems

_**TL;DR:** Supervised learning predicts labels, unsupervised learning models structure in observations, and reinforcement learning chooses actions for reward._

Slide 3 frames ML around three different problem types:

- **Supervised learning**: observe input/output pairs $(x_i, y_i)$ and predict $y_*$ for a new $x_*$. Examples: classification and regression.
- **Unsupervised learning**: observe inputs $x_1, x_2, \ldots$ without labels and try to describe the structure of the data. This often means modelling $p(x)$ or inferring hidden structure behind the observations.
- **Reinforcement learning**: choose actions $a_i$ that affect future rewards $r_i$, and learn a policy that maximizes payoff.

The key course direction is the unsupervised/probabilistic one: **if no labels are provided, we need a model of how the observations themselves are generated**.

### Representing a Data Source

_**TL;DR:** A model is a family of possible data-generating distributions; learning uses observed data to narrow down which distribution explains what we saw._

Slide 9 introduces the core probabilistic modelling stance:

$$
P(\text{data}|\text{parameters})
$$

or more concretely:

$$
P(x|\theta)
\quad \text{or} \quad
P(y|x,\theta)
$$

The idea:

- We observe data $D = (x_1, \ldots, x_n)$ from some source.
- We describe the source with a probability distribution $P$ over a sample space $\mathcal{X}$.
- A statistical model is a set of candidate probability distributions:

$$
M = \{P(\cdot|\theta) \mid \theta \in T\}
$$

- Saying "assume model $M$" means "assume the real data-generating distribution is one of the distributions inside $M$."
- Example: $M$ could be all Gaussian distributions with fixed variance 1 and unknown mean $\theta$:

$$
M = \{\mathcal{N}(\theta, 1) \mid \theta \in \mathbb{R}\}
$$

### Basic Probability Rules

_**TL;DR:** Probability algebra lets us move between joint, marginal, and conditional probabilities; Bayes' rule updates beliefs by combining prior plausibility with data fit._

Slide 10 is the basic algebra we keep using:

- Probabilities are non-negative:

$$
P(x) \ge 0
$$

- Probabilities normalize:

$$
\sum_x P(x) = 1
\quad \text{or} \quad
\int p(x)\,dx = 1
$$

- The **joint probability** $P(x,y)$ describes the probability of $x$ and $y$ together.
- The **marginal probability** $P(x)$ sums/integrates out the other variable:

$$
P(x) = \sum_y P(x,y)
$$

- The **conditional probability** $P(x|y)$ means probability of $x$ after assuming $y$ happened:

$$
P(x|y) = \frac{P(x,y)}{P(y)}
$$

**Bayes' rule** follows from writing the joint in two ways:

$$
P(x,y) = P(x)P(y|x) = P(y)P(x|y)
$$

So:

$$
P(y|x) = \frac{P(x|y)P(y)}{P(x)}
$$

The usual Bayesian vocabulary:

$$
\text{posterior}
=
\frac{\text{likelihood} \times \text{prior}}{\text{evidence}}
$$

Meaning:

- **prior**: what we believed before seeing $x$;
- **likelihood**: how compatible the observed data $x$ is with each hypothesis $y$;
- **evidence**: normalizing constant that makes the posterior sum/integrate to 1;
- **posterior**: updated belief after seeing the data.

### Independent and Identically Distributed (IID)

_**TL;DR:** IID means each observation is drawn independently from the same distribution, which lets dataset probabilities factor into products._

Slide 11 defines independence. Two random variables $X_1$ and $X_2$ are independent if knowing one tells us nothing about the other.

Mathematically, the joint distribution factorizes:

$$
P(x_1, x_2) = P_1(x_1)P_2(x_2)
$$

Equivalently:

$$
P(x_1|x_2) = P_1(x_1)
$$

The conditional version is the intuition: after learning $x_2$, our belief about $x_1$ is unchanged.

For many observations, we often assume the data are **iid**, meaning independent and identically distributed:

$$
P(x_1,\ldots,x_n) = \prod_{i=1}^n P(x_i)
$$

This assumption says:

- **independent**: each data point is sampled separately, so one observation does not affect another;
- **identically distributed**: all data points come from the same distribution.

This assumption is often unrealistic in detail, but it makes the math tractable and is the starting point for many models.

### Exponential Families

_**TL;DR:** Exponential families are distributions where data affects learning through compact sufficient statistics like counts, sums, and squared sums._

Core intuition:

- Exponential families are common distributions where learning from data reduces to updating a few simple summaries: counts, sums, or sums of squares.
- The raw data sequence often matters less than these summaries.
  - Coin flips: number of heads and tails.
  - Gaussian data: if only the mean is unknown, the key statistic is the sum of observations; if variance is also unknown, we also need the sum of squared observations.
- These summaries are called **sufficient statistics** because they contain all the information in the data that is relevant for learning the parameter, within the assumed model.

The course writes an exponential family as:

$$
p(x|\theta) = f(x)g(\theta)e^{\phi(\theta)^T T(x)}
$$

Read this as:

$$
\text{probability of data under parameter}
=
\text{data-only background}
\times
\text{parameter-only normalization}
\times
\text{parameter/data-summary interaction}
$$

Notation:

- $x$: observed data point.
- $\theta$: parameter of the distribution.
- $T(x)$: sufficient statistic; the summary of $x$ that matters for learning $\theta$.
- $\phi(\theta)$: natural parameter; a convenient reparameterization of $\theta$.
- $\phi(\theta)^T T(x)$: dot product between the natural parameter and the sufficient statistic. If both are scalars, this is just multiplication.
- $f(x)$: base measure; the part that depends only on the data.
- $g(\theta)$: normalization term; the part that depends only on the parameter and makes the distribution integrate or sum to 1.

**The main payoff appears with iid data**:

$$
\prod_i p(x_i|\theta)
$$

The exponential terms multiply:

$$
\prod_i e^{\phi(\theta)^T T(x_i)}
=
e^{\phi(\theta)^T \sum_i T(x_i)}
$$

So all the observations enter the likelihood through:

$$
\sum_i T(x_i)
$$

That is the little mechanism behind why these models are so convenient.

Slide 13 says two things "informally":

- A sample space often has a **natural statistic**. This means the type of data usually suggests the obvious summary to track.
  - Coin flips live in $\{0, 1\}$, so the natural summary is the number of 1s.
  - Category labels live in $\{1, \ldots, K\}$, so the natural summary is a count for each category.
  - Real-valued Gaussian data live on $\mathbb{R}$, so sums and squared sums are natural summaries.
- The exponential family defined by that statistic is often the **natural distribution** for simple data sources on that sample space. In plain English: once you decide what summary matters, there is usually a standard distribution whose probabilities are controlled by that summary.
  - If the summary is number of successes, you get Bernoulli/binomial-style models.
  - If the summary is category counts, you get multinomial-style models.
  - If the summaries are sum and sum of squares, you get Gaussian-style models.

So the slide is saying: first look at what kind of objects your data are; that usually tells you what summaries are sensible; those summaries usually point to the right basic probability model.

#### Exponential Families: Bernoulli Example

_**TL;DR:** For Bernoulli data, the sufficient statistic is the number of heads and the natural parameter is the log-odds of heads._

Let $x \in \{0, 1\}$ be the observed outcome of one coin flip, where $x = 1$ means heads and $x = 0$ means tails. Let $q \in [0, 1]$ be the parameter of the distribution: the probability of heads, $q = P(x = 1)$. The Bernoulli distribution is:

$$
p(x|q) = q^x(1-q)^{1-x}
$$

Rewrite:

$$
p(x|q)
=
(1-q)e^{x \log(q/(1-q))}
$$

Matching to the exponential-family form:

- $f(x) = 1$
- $g(q) = 1-q$
- $T(x) = x$
- $\phi(q) = \log \frac{q}{1-q}$

So the sufficient statistic for one coin flip is $x$. For many flips:

$$
\sum_i T(x_i) = \sum_i x_i
$$

which is the number of heads. The number of tails is then determined by $n - \sum_i x_i$.

Intuition:

- We do not need to remember the whole sequence of flips.
- For estimating the coin bias $q$, the relevant summary is how many heads occurred.
- This is why the Beta prior updates so cleanly: observing a head increments one count, and observing a tail increments the other count.

#### Exponential Families: Gaussian Example

_**TL;DR:** For a 1D Gaussian with unknown mean and variance, the sufficient statistics are the sum of observations and the sum of squared observations._

For a 1D Gaussian with unknown mean $\mu$ and unknown variance $\sigma^2$:

$$
p(x|\mu,\sigma^2)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

Expand the square:

$$
-\frac{(x-\mu)^2}{2\sigma^2}
=
-\frac{x^2}{2\sigma^2}
+ \frac{\mu x}{\sigma^2}
- \frac{\mu^2}{2\sigma^2}
$$

So the Gaussian can be rewritten to separate the parameter-only term from the parameter/data interaction:

$$
p(x|\mu,\sigma^2)
=
\underbrace{
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{\mu^2}{2\sigma^2}\right)
}_{g(\mu,\sigma^2)}
\underbrace{
\exp\left(
\left(
\underbrace{
\left(\frac{\mu}{\sigma^2}, -\frac{1}{2\sigma^2}\right)
}_{\phi(\mu,\sigma^2)}
\right)^T
\underbrace{
\left(x, x^2\right)
}_{T(x)}
\right)
}_{e^{\phi(\mu,\sigma^2)^T T(x)}}
$$

One valid exponential-family mapping is:

- $T(x) = (x, x^2)$
- $\phi(\mu,\sigma^2) = \left(\frac{\mu}{\sigma^2}, -\frac{1}{2\sigma^2}\right)$
- $f(x) = 1$
- $g(\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{\mu^2}{2\sigma^2}\right)$

For many observations, the sufficient statistics are:

$$
\sum_i x_i
\quad \text{and} \quad
\sum_i x_i^2
$$

Intuition:

- $\sum_i x_i$ tells us about the center/mean.
- $\sum_i x_i^2$ tells us about spread/variance.
- Together with $n$, these are enough to compute the sample mean and variance.
- The full dataset can be compressed into these totals, assuming the Gaussian model is correct.

**Takeaway**:

- Exponential families look abstract because of the notation, but the practical idea is simple: **counts and sums are enough** for many common distributions.
- This makes Bayesian updating, conjugate priors, maximum likelihood, and later variational inference much cleaner algebraically.

### Learning Parameters

_**TL;DR:** MLE chooses the parameter that best explains the data, MAP chooses the parameter that best explains the data while respecting a prior, and Bayesian inference keeps a posterior distribution over plausible parameters._

Slide 14 distinguishes two ways to think about an unknown parameter $\theta$.

#### Bayesian Inference

_**TL;DR:** Bayesian inference treats the parameter as uncertain, updating a prior $p(\theta)$ into a posterior $p(\theta|D)$ after seeing data._

- Treat the unknown parameter as uncertain and represent that uncertainty with a probability distribution.
- Before data: prior $p(\theta)$.
- After data: posterior $p(\theta|D)$.
- Instead of returning one best value, Bayesian inference asks how plausible each $\theta$ is after seeing the data.
- $p(\theta|D)$ is called the **posterior** because it is the distribution over $\theta$ after observing data $D$:

$$
p(\theta|D) = \frac{p(D|\theta)p(\theta)}{p(D)}
$$

  The names are temporal: prior = before data, posterior = after data.

- For a coin, Bayesian inference does not just say "the best estimate is $q = 0.7$." It represents uncertainty over $q$, for example with a Beta posterior concentrated near plausible coin biases.
- A fully Bayesian predictive distribution averages over parameter uncertainty:

$$
p(y|x,D) = \int p(y|x,\theta)p(\theta|D)d\theta
$$

- Standard deep learning usually uses $p(y|x,\hat{\theta})$ instead. Bayesian inference would use the whole posterior $p(\theta|D)$.

**Bayesian model = statistical model plus parameter prior.** In symbols, it is the pair $(M, \pi)$, where $M = \{p(\cdot|\theta) \mid \theta \in T\}$ is the family of possible data-generating distributions and $\pi(\theta)$ is the prior distribution over the parameter. The prior is usually chosen and tractable; the hard part is often the posterior denominator/evidence, which requires integrating over all possible $\theta$ values.

**Bayesian inference usually proceeds in one of three ways:**

1. **Exact/conjugate posterior**

   This is the cleanest case of the Bayesian model idea: choose a likelihood model $M$ and a prior $\pi$ that are algebraically compatible, then Bayes' rule updates the prior into a posterior of the same form.

   Sometimes Bayes' rule gives a closed-form posterior that can be written down exactly.

   Example:

   $$
   q \sim \text{Beta}(\alpha,\beta),
   \quad
   x_i \sim \text{Bernoulli}(q)
   $$

   If we observe $H$ heads and $T$ tails, the posterior is:

   $$
   q|D \sim \text{Beta}(\alpha + H, \beta + T)
   $$

   This is called **conjugacy**: the posterior is in the same distribution family as the prior. In this case, Bayesian updating is basically bookkeeping over counts.

2. **Sampling / MCMC**

   Sometimes we cannot represent the posterior with a simple formula (no closed-form normalized $p(\theta|D)$ from $p(D|\theta)p(\theta)$, and often no closed-form predictive integral $\int p(y|x,\theta)p(\theta|D)d\theta$), but we can generate samples from it:

   $$
   \theta_1, \theta_2, \ldots, \theta_N \sim p(\theta|D)
   $$

   Then we use the sample cloud to estimate posterior quantities. For example:

   $$
   \mathbb{E}[\theta|D] \approx \frac{1}{N}\sum_{i=1}^N \theta_i
   $$

   Credible intervals come from quantiles of the samples, and predictions can be made by averaging predictions over sampled parameter values. MCMC methods are designed to produce these approximate posterior samples even when direct sampling is hard.

3. **Variational inference**

   Often the true posterior $p(\theta|D)$ is too complicated to compute exactly or sample from efficiently, mainly because its normalizing denominator/evidence is hard. Variational inference chooses a simpler family of distributions $q_\lambda(\theta)$ and finds the member of that family that best approximates the posterior:

   $$
   p(\theta|D) \approx q_\lambda(\theta)
   $$

   The key move is turning inference into optimization: choose $\lambda$ so that $q_\lambda$ is close to $p(\theta|D)$.

   Here, $\lambda$ denotes the parameters of the approximation, not the original model parameters. Example: if $q_\lambda(\theta)$ is Gaussian, then $\lambda$ might contain its mean and covariance.

   After choosing the variational family, we optimize $\lambda$ to get an approximate posterior $q_{\lambda^*}(\theta)$. In practice this is usually done by maximizing an ELBO or equivalently minimizing a KL divergence to the true posterior. Then $q_{\lambda^*}$ is used like the posterior: compute expectations, uncertainty intervals, or predictions by averaging over $q_{\lambda^*}$.

   It is called **variational** inference because we are optimizing over candidate distributions, not just over a single number. Historically, "variational" refers to optimization over functions; here the function/distribution being optimized is $q_\lambda(\theta)$.

   Example intuition: the true posterior might be weird and high-dimensional, but we approximate it with a Gaussian whose mean and covariance are learned.

Examples of successful Bayesian-style methods:

- **Bayesian optimization**: uses a probabilistic model, often a Gaussian process, to decide which expensive experiment or hyperparameter setting to try next. It works well when uncertainty should guide exploration.
- **Kalman filters / particle filters**: maintain beliefs over hidden states in robotics, navigation, tracking, and sensor fusion.
- **Thompson sampling**: uses posterior uncertainty for exploration in bandits and online decision-making.

#### MAP Estimate

_**TL;DR:** MAP is still a point estimate, but unlike MLE it includes the prior; regularized training is often MAP-like._

$$
\theta_{MAP} = \arg\max_\theta p(\theta|D)
= \arg\max_\theta p(D|\theta)p(\theta)
$$

- MAP uses the Bayesian posterior, but then collapses it back to one parameter value.
- So MAP is still a point estimate.
- Practical translation:
  - MLE fits the data.
  - MAP fits the data while also respecting a prior.
- Example: if we observe 7 heads out of 10 coin flips, MLE gives $\hat{q}_{MLE} = 0.7$. If we have a strong prior belief that the coin is close to fair, MAP will be pulled back toward $0.5$.
- Neural-network training without explicit regularization is often MLE-like: choose weights that make the training labels likely.
- Neural-network training with weight decay is MAP-like:

$$
\min_\theta \text{loss}(D,\theta) + \lambda \|\theta\|^2
$$

- The $\lambda \|\theta\|^2$ term behaves like a Gaussian prior over weights, favoring smaller weights. So regularized training can be read as "fit the data, but prefer parameters that were plausible under the prior."

**Diffusion-model aside**:

- Standard diffusion models are probabilistic generative models, but they are usually not Bayesian over parameters.
- They learn one set of neural-network parameters $\hat{\theta}$ and then sample from $p_{\hat{\theta}}(x)$.
- A Bayesian diffusion model would maintain a posterior over parameters $p(\theta|D)$ and average or sample over that uncertainty, which is not how mainstream diffusion models are usually trained.

### Basic Bayesian Learning

_**TL;DR:** Given observed data, Bayesian learning scores each possible parameter by likelihood, combines that with the prior, and normalizes into a posterior._

Slide 15 makes the Bayesian setup explicit. The notation is doing several jobs at once:

- $X$ is a random variable: the data before we observe its actual value.
- $x$ is an observed value of that random variable.
- $D = \{x_1,\ldots,x_n\}$ is the observed dataset.
- $\theta$ is a particular possible parameter value.
- $\Theta$ is the parameter treated as a random variable.
- $T$ is the parameter space, the set of possible values $\theta$ can take. For a coin bias, $T = [0,1]$; for an unknown Gaussian mean, $T = \mathbb{R}$.
- $p(\cdot|\theta)$ is a probability distribution over possible data values for a fixed parameter $\theta$.
- $M = \{p(\cdot|\theta) \mid \theta \in T\}$ is the statistical model: the family of possible data-generating distributions.
- $\pi$ is the prior distribution over the parameter space $T$.

Bernoulli example:

- Let $\theta$ be the coin bias, i.e. the probability of heads.
- The parameter space is $T = [0,1]$ because $\theta$ can be any probability between 0 and 1.
- Each random coin flip is:

$$
X_i|\theta \sim \text{Bernoulli}(\theta)
$$

- Before observing the flip, $X_i$ is random.
- After observing the flip, $x_i$ is the realized value: $x_i = 1$ for heads, $x_i = 0$ for tails.
- If the observed dataset is heads, heads, tails, then:

$$
D = \{1,1,0\}
$$

- For a fixed possible bias $\theta$, the likelihood is:

$$
p(D|\theta) = p(1|\theta)p(1|\theta)p(0|\theta)
= \theta^2(1-\theta)
$$

So $X_i|\theta \sim p(\cdot|\theta)$ describes how data would be generated, while $p(x_i|\theta)$ evaluates how plausible an observed value $x_i$ is under a particular $\theta$.

With those pieces in place, a **Bayesian model** is the pair:

$$
(M, \pi)
$$

This means: specify both the possible likelihoods and the prior belief over which parameter values are plausible. The likelihood family $M$ alone is not yet a full Bayesian model; Bayesian inference also needs a prior $\pi$. The prior is usually not the hard part because we choose it. The posterior can be hard because normalizing it requires the evidence integral over all $\theta$.

In practice, the data $D$ is already observed. For each possible parameter value $\theta$, we compute the likelihood:

$$
p(D|\theta)
$$

This asks: how plausible would the observed dataset be if $\theta$ were the parameter?

If the observations are iid given $\theta$, the dataset likelihood factorizes:

$$
p(D|\theta) = \prod_{i=1}^n p(x_i|\theta)
$$

Bayesian learning combines that likelihood with the prior. Bayes' rule gives the posterior:

$$
\pi(\theta|D)
=
\frac{
\prod_{i=1}^n p(x_i|\theta)\pi(\theta)
}{
\int_T \prod_{i=1}^n p(x_i|\theta)\pi(\theta)d\theta
}
$$

The important version to remember:

$$
\pi(\theta|D) \propto \left[\prod_{i=1}^n p(x_i|\theta)\right]\pi(\theta)
$$

In words:

$$
\text{posterior} \propto \text{likelihood} \times \text{prior}
$$

In the above,

- **Numerator**: parameter values are plausible after seeing data if they both explain the data well and were plausible before seeing the data.
- **Denominator**: evidence/normalizer; it integrates over all possible $\theta$ values so that the posterior is a valid distribution. **Often intractable in realistic Bayesian models**.

### Conjugate Priors

_**TL;DR:** Conjugacy makes Bayesian updating algebraic: posterior stored quantities equal prior stored quantities plus data summaries._

Slide 24 explains why exponential families pair so neatly with conjugate priors.

Start with an exponential-family likelihood:

$$
P(x|\theta) = g(\theta)f(x)e^{\phi(\theta)^T T(x)}
$$

For $n$ iid observations $x_1,\ldots,x_n$, the likelihood is:

$$
P(\{x_i\}_{i=1}^n|\theta)
=
\prod_{i=1}^n P(x_i|\theta)
=
g(\theta)^n e^{\phi(\theta)^T \sum_{i=1}^n T(x_i)}\prod_{i=1}^n f(x_i)
$$

The important part for learning $\theta$ is the likelihood as a function of $\theta$:

$$
P(\{x_i\}_{i=1}^n|\theta)
\propto
g(\theta)^n e^{\phi(\theta)^T \sum_{i=1}^n T(x_i)}
$$

The $\prod_i f(x_i)$ term depends only on the observed data, not on $\theta$, so it is not important for the posterior's shape as a function of $\theta$.

A conjugate prior is designed to have the same algebraic shape as this likelihood:

$$
P(\theta) = F(\tau,\nu)g(\theta)^\nu e^{\phi(\theta)^T \tau}
$$

Here:

- $F(\tau,\nu)$ is the normalizer that makes the prior integrate to 1.
- $\tau$ stores prior evidence in the same "units" as the sufficient statistic $T(x)$.
- $\nu$ stores prior strength or concentration, roughly how much weight to give the prior.

When we multiply likelihood by prior:

$$
P(\theta|\{x_i\}_{i=1}^n)
\propto
P(\{x_i\}_{i=1}^n|\theta)P(\theta)
$$

we get:

$$
P(\theta|\{x_i\}_{i=1}^n)
\propto
g(\theta)^{\nu+n}
e^{\phi(\theta)^T\left(\tau+\sum_{i=1}^n T(x_i)\right)}
$$

After adding the right normalizer, the posterior is:

$$
P(\theta|\{x_i\}_{i=1}^n)
=
F\left(\tau+\sum_{i=1}^n T(x_i), \nu+n\right)
g(\theta)^{\nu+n}
e^{\phi(\theta)^T\left(\tau+\sum_{i=1}^n T(x_i)\right)}
$$

After observing $n$ data points, the posterior updates by simple addition:

$$
\tau \rightarrow \tau + \sum_{i=1}^n T(x_i)
$$

$$
\nu \rightarrow \nu + n
$$

Intuition:

$$
\underbrace{\tau+\sum_{i=1}^n T(x_i)}_{\text{posterior statistic}}
=
\underbrace{\tau}_{\text{prior accumulated statistic}}
+
\underbrace{\sum_{i=1}^n T(x_i)}_{\text{data accumulated statistic}}
$$

$$
\underbrace{\nu+n}_{\text{posterior strength}}
=
\underbrace{\nu}_{\text{prior strength}}
+
\underbrace{n}_{\text{number of real observations}}
$$

The concentration $\nu$ specifies how much weight we assign to the prior belief $\tau$. Large $\nu$ means a strong prior assumption. When the pseudo-count interpretation applies, $\nu$ behaves like the number of fictitious observations behind the prior.

So conjugate Bayesian updating is bookkeeping:

$$
\text{posterior stored quantities} = \text{prior stored quantities} + \text{data summaries}
$$

This is why the Beta-Bernoulli update is so clean. For coin flips, the sufficient statistic is the number of heads. Observing a head increments the heads evidence; observing a tail increments the total count without incrementing heads. The posterior is just the prior counts plus observed counts.

The broader lesson: conjugacy works because the prior is written in the same language as the likelihood's sufficient statistics. This is the clean case in [Bayesian Inference](#bayesian-inference), item 1: exact/conjugate posterior. Data does not force a new kind of distribution; it just updates the prior's stored statistics.

#### Conjugate Priors: Bernoulli Example

_**TL;DR:** For Bernoulli data, the Beta posterior is prior heads/tails evidence plus observed heads/tails counts._

Slide 26 specializes the conjugate-prior story to coin flips.

Let $x \in \{0,1\}$ be one observed coin flip, where $x=1$ means heads and $x=0$ means tails. Let $\theta \in [0,1]$ be the coin bias, i.e. the probability of heads:

$$
\theta = P(x=1)
$$

The Bernoulli likelihood for one observation is:

$$
P(x|\theta) = \theta^x(1-\theta)^{1-x}
$$

This can be rewritten in exponential-family form:

$$
P(x|\theta)
=
(1-\theta)e^{x\log(\theta/(1-\theta))}
$$

So:

- $T(x) = x$: the sufficient statistic is whether this flip was heads.
- $\phi(\theta) = \log\frac{\theta}{1-\theta}$: the natural parameter is the log-odds of heads.

For many flips, the observed heads and tails counts are:

$$
\sum_{i=1}^n x_i = \text{number of heads}
\quad
\text{and}
\quad
n - \sum_{i=1}^n x_i = \text{number of tails}
$$

The conjugate prior for a Bernoulli likelihood is the Beta distribution:

$$
p(\theta) = \text{Beta}(\theta|\alpha_1,\alpha_2)
$$

The update is just count accumulation:

$$
p(\theta|D)
=
\text{Beta}\left(\theta \,\middle|\,
\alpha_1 + \underbrace{\sum_{i=1}^n x_i}_{\text{observed heads}},
\alpha_2 + \underbrace{n - \sum_{i=1}^n x_i}_{\text{observed tails}}
\right)
$$

Intuition:

- $\alpha_1$ behaves like prior heads evidence.
- $\alpha_2$ behaves like prior tails evidence.
- Observing a head increments $\alpha_1$.
- Observing a tail increments $\alpha_2$.

So slide 26 is mainly showing that the familiar Beta-Bernoulli update is exactly the general conjugate-prior formula applied to the Bernoulli exponential-family form.

### Maximum Likelihood Estimation (MLE)

_**TL;DR:** MLE is point estimation from data alone: treat the observed data as fixed and choose the parameter value under which that dataset is most likely._

Slide 29 defines MLE for iid data. Assume:

- observed data $D = \{x_1,\ldots,x_n\}$;
- a model $M = \{p(x|\theta) \mid \theta \in T\}$;
- iid observations under a fixed but unknown parameter $\theta$.

The likelihood of the dataset is:

$$
p(D|\theta) = p(x_1,\ldots,x_n|\theta)
$$

With the iid assumption:

$$
p(D|\theta) = \prod_{i=1}^n p(x_i|\theta)
$$

The maximum likelihood estimate is:

$$
\hat{\theta}_{MLE}
=
\arg\max_{\theta \in T} p(D|\theta)
=
\arg\max_{\theta \in T}\prod_{i=1}^n p(x_i|\theta)
$$

Intuition:

- The data $D$ is already observed.
- For each possible $\theta$, compute how likely $D$ would have been under that $\theta$.
- Pick the $\theta$ with the highest likelihood.

Example: if a coin lands heads 7 times out of 10, the MLE for the probability of heads is $\hat{\theta}_{MLE}=0.7$.

Most standard predictive models, including ordinary neural nets and ImageNet classifiers, learn point estimates of their weights:

$$
p(y|x,\hat{\theta})
$$

The softmax output may be a probability distribution over labels, but the model parameters are still one fitted value $\hat{\theta}$. That is different from Bayesian uncertainty over possible parameter values.

#### Logarithm Trick

_**TL;DR:** Maximizing likelihood is equivalent to maximizing log-likelihood, and logs turn products over data points into sums._

Slide 30 uses the log-likelihood because products are awkward and sums are easier:

$$
\log\prod_{i=1}^n p(x_i|\theta)
=
\sum_{i=1}^n \log p(x_i|\theta)
$$

The logarithm is monotonically increasing, so it does not change the maximizing $\theta$:

$$
\arg\max_\theta p(D|\theta)
=
\arg\max_\theta \log p(D|\theta)
$$

Therefore:

$$
\hat{\theta}_{MLE}
=
\arg\max_\theta \sum_{i=1}^n \log p(x_i|\theta)
$$

If the objective is differentiable, the MLE satisfies the first-order condition:

$$
\sum_{i=1}^n \nabla_\theta \log p(x_i|\theta) = 0
$$

This is the same MLE problem, just written in a form that is easier to differentiate and optimize numerically.

#### Law of Large Numbers

_**TL;DR:** The law of large numbers lets us replace unknown expectations with observable sample averages when we have enough iid data._

Slide 31 recalls that for iid random variables $X_1,X_2,\ldots$ and a function $f$:

$$
\frac{1}{n}\sum_{i=1}^n f(X_i)
\xrightarrow[n\to\infty]{}
\mathbb{E}[f(X_1)]
$$

The practical meaning:

- **Expectations are population quantities**: what happens on average under the data-generating distribution.
- **Sample averages are observable quantities**: what we can compute from data.
- **With enough iid data, sample averages approximate expectations**.

For parameter estimation, the strategy is:

1. Find a function $g(X,\theta)$ such that:

   $$
   \mathbb{E}_{\theta_0}[g(X,\theta)] = 0
   $$

   only when $\theta = \theta_0$.

2. Estimate that expectation with the sample average:

   $$
   \frac{1}{n}\sum_{i=1}^n g(X_i,\theta) = 0
   $$

3. Solve this equation for $\theta$.

So the law of large numbers gives a bridge from the true parameter $\theta_0$ to an estimator computable from data.

#### MLE from the Law of Large Numbers

_**TL;DR:** MLE comes from using the score function as the estimating equation whose expectation is zero at the true parameter._

Slide 32 identifies the relevant function:

$$
g(x,\theta) = \nabla_\theta \log p(x|\theta)
$$

This is called the **score function** or **Fisher score**. It is connected to **Fisher information**, which measures how sensitive the distribution $p(x|\theta)$ is to changes in $\theta$. Intuitively, the Fisher score for one data point is that data point's local "push" on the parameter: the direction changing $\theta$ would increase its log-likelihood. For a dataset, the scores add up; at the MLE, the total score is zero, meaning the data's pushes on $\theta$ balance out.

Under regularity conditions, the score has expectation zero at the true parameter:

$$
\mathbb{E}_{\theta_0}[\nabla_\theta \log p(X|\theta)] = 0
$$

The law of large numbers says we can approximate this expectation with the sample average:

$$
\frac{1}{n}\sum_{i=1}^n \nabla_\theta \log p(x_i|\theta) = 0
$$

Multiplying by $n$ does not change the solution, so this is equivalent to:

$$
\sum_{i=1}^n \nabla_\theta \log p(x_i|\theta) = 0
$$

That is exactly the first-order condition for maximizing the log-likelihood. In this sense, MLE is the parameter value that makes the average score on the observed data equal zero.

### Tools: Gaussian Distributions

_**TL;DR:** A Gaussian is controlled by a mean and a scale; in multiple dimensions, the covariance matrix controls both the marginal spread of each coordinate and the dependence between coordinates._

Slides 33-39 pause the main learning story to review Gaussian distributions, because Gaussians will appear repeatedly in probabilistic modelling, latent-variable models, approximate inference, and optimization.

#### One-Dimensional Gaussian

Slide 34 recalls the 1D Gaussian density:

$$
p(x;\mu,\sigma)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right)
$$

The parameters are:

- $\mu$: expected value or mean.
- $\sigma^2$: variance.
- $\sigma$: standard deviation.

The standardized quantity

$$
\frac{x-\mu}{\sigma}
$$

measures how far $x$ is from its mean in units of standard deviation. So $\sigma$ defines the length scale of the distribution.

The cumulative distribution function is:

$$
\Phi(z) = P(Z \le z) = \int_{-\infty}^z p(z')dz'
$$

For any 1D Gaussian, intervals of 1, 2, and 3 standard deviations around the mean contain fixed probability mass:

- 1 standard deviation: $[\mu-\sigma,\mu+\sigma]$ contains about $68.27\%$ of the mass.
- 2 standard deviations: $[\mu-2\sigma,\mu+2\sigma]$ contains about $95.45\%$.
- 3 standard deviations: $[\mu-3\sigma,\mu+3\sigma]$ contains about $99.73\%$.

The important point is that $\mu$ shifts the density left or right, while $\sigma$ stretches or compresses it.

#### Components of a 1D Gaussian

Slide 35 decomposes the Gaussian shape into simpler pieces.

Starting from $x$, the Gaussian first centers the value:

$$
x - \mu
$$

Then it squares the centered distance:

$$
(x-\mu)^2
$$

Then it rescales by the variance:

$$
\left(\frac{x-\mu}{\sigma}\right)^2
$$

Then it applies the negative quadratic exponent:

$$
-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2
$$

Finally, exponentiating gives the bell shape:

$$
\exp\left(
-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2
\right)
$$

Intuition:

- Points close to $\mu$ have small squared standardized distance, so they receive high density.
- Points far from $\mu$ have large squared standardized distance, so the negative exponent makes their density shrink quickly.
- The Gaussian is therefore an exponentiated negative quadratic.

#### Covariance Matrices

Slide 36 generalizes variance to multiple dimensions.

For two random variables $X_1$ and $X_2$, covariance is:

$$
\operatorname{Cov}[X_1,X_2]
=
\mathbb{E}\left[
(X_1-\mathbb{E}[X_1])
(X_2-\mathbb{E}[X_2])
\right]
$$

If $X_1 = X_2 = X$, covariance reduces to variance:

$$
\operatorname{Cov}[X,X] = \operatorname{Var}[X]
$$

For a random vector

$$
X = (X_1,\ldots,X_D) \in \mathbb{R}^D
$$

the covariance matrix collects every pairwise covariance:

$$
\operatorname{Cov}[X]
=
\left(\operatorname{Cov}[X_i,X_j]\right)_{i,j}
=
\begin{pmatrix}
\operatorname{Cov}[X_1,X_1] & \cdots & \operatorname{Cov}[X_1,X_D] \\
\vdots & \ddots & \vdots \\
\operatorname{Cov}[X_D,X_1] & \cdots & \operatorname{Cov}[X_D,X_D]
\end{pmatrix}
$$

The usual notation is:

$$
\Sigma = \operatorname{Cov}[X]
$$

Interpretation:

- Diagonal entries $\Sigma_{ii}$ are variances of individual coordinates.
- Off-diagonal entries $\Sigma_{ij}$ are covariances between coordinates.
- Positive covariance means the two coordinates tend to move together.
- Negative covariance means one tends to be high when the other is low.
- Zero covariance means no linear dependence, though not necessarily full independence in general.

#### Multivariate Gaussian

Slide 37 replaces the 1D squared standardized distance with a quadratic form.

In 1D, the exponent is:

$$
-\frac{(x-\mu)^2}{2\sigma^2}
=
-\frac{1}{2}(x-\mu)(\sigma^2)^{-1}(x-\mu)
$$

In $D$ dimensions, the scalar variance $\sigma^2$ becomes the covariance matrix $\Sigma$:

$$
p(x;\mu,\Sigma)
=
\frac{1}{\sqrt{(2\pi)^D|\Sigma|}}
\exp\left(
-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)
\right)
$$

where $\Sigma$ must be positive definite.

The term

$$
(x-\mu)^T\Sigma^{-1}(x-\mu)
$$

is the squared **Mahalanobis distance** from $x$ to $\mu$. It is the multivariate analogue of:

$$
\left(\frac{x-\mu}{\sigma}\right)^2
$$

The covariance matrix controls the shape of the Gaussian:

- Large variance in a direction means the density spreads out in that direction.
- Small variance means the density is narrow in that direction.
- Nonzero covariance rotates the density contours away from the coordinate axes.

Assuming a multivariate Gaussian model means assuming that all stochastic dependence between dimensions is captured by the covariance matrix.

#### Gaussian Density Example and Contours

Slides 38-39 show a 2D Gaussian with:

$$
\mu = (0,0)
\quad
\text{and}
\quad
\Sigma =
\begin{pmatrix}
2 & 1 \\
1 & 2
\end{pmatrix}
$$

The covariance has positive off-diagonal entries, so the two coordinates tend to increase together. That creates tilted elliptical contours rather than axis-aligned circles.

A contour line is made by slicing the density surface at a fixed height and projecting the intersection down to the input plane. For a Gaussian, these contours are ellipses.

To see where the contour equation comes from, start with the 1D Gaussian:

$$
p(x;\mu,\sigma)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(
-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2
\right)
$$

A contour in 1D means "points with the same density." Since the normalizing constant is fixed for a given Gaussian, equal density means equal exponent:

$$
\left(\frac{x-\mu}{\sigma}\right)^2 = c
$$

Equivalently:

$$
\frac{(x-\mu)^2}{\sigma^2} = c
$$

So in 1D, a Gaussian contour picks points the same number of standard deviations away from the mean:

$$
x = \mu \pm \sqrt{c}\sigma
$$

The multivariate Gaussian replaces the 1D standardized squared distance with the covariance-adjusted squared distance:

$$
\left(\frac{x-\mu}{\sigma}\right)^2
\quad
\longrightarrow
\quad
(x-\mu)^T\Sigma^{-1}(x-\mu)
$$

So for a Gaussian with mean $\mu$, the contour equation is:

$$
(x-\mu)^T\Sigma^{-1}(x-\mu) = c
$$

Changing $c$ changes the size of the ellipse. These ellipses are the multivariate analogue of intervals around the mean in 1D.

The useful intuition:

- In 1D, "distance from the mean" is measured in standard deviations.
- In multiple dimensions, distance from the mean is measured by Mahalanobis distance.
- Gaussian contours collect points at equal Mahalanobis distance from the mean.
- The covariance matrix determines which directions count as "long" or "short" directions for the distribution.

### Tools: Spectra and Geometry of Gaussians

_**TL;DR:** The eigenvectors of a Gaussian covariance matrix give the main axes of variation, and the eigenvalues give the variances along those axes._

Slides 40-64 review the linear algebra behind multivariate Gaussians and then use it for Gaussian MLE and linear regression.

#### Eigenvalues and Eigenvectors

For a square matrix $A \in \mathbb{R}^{D \times D}$, an eigenvector $v$ and eigenvalue $\lambda$ satisfy:

$$
Av = \lambda v
$$

Applying $A$ to $v$ does not change its direction; it only rescales it by $\lambda$.

The set of eigenvalues is called the **spectrum** of $A$.

Symmetric matrices have especially clean eigenstructure:

- all eigenvalues are real;
- eigenvectors for distinct eigenvalues are orthogonal;
- if $A$ has full rank, its eigenvectors form an orthonormal basis of $\mathbb{R}^D$.

Why distinct eigendirections are orthogonal:

If $A$ is symmetric, then $A^T=A$, so it can be moved across an inner product:

$$
\langle Ax,y\rangle = \langle x,Ay\rangle
$$

because:

$$
\langle Ax,y\rangle
=
(Ax)^T y
=
x^T A^T y
=
x^T A y
=
\langle x,Ay\rangle
$$

Let $v_i$ and $v_j$ be eigenvectors with eigenvalues $\lambda_i$ and $\lambda_j$:

$$
Av_i = \lambda_i v_i,
\quad
Av_j = \lambda_j v_j
$$

Then:

$$
\langle Av_i,v_j\rangle
=
\langle v_i,Av_j\rangle
$$

Substitute the eigenvalue equations:

$$
\lambda_i \langle v_i,v_j\rangle
=
\lambda_j \langle v_i,v_j\rangle
$$

So:

$$
(\lambda_i-\lambda_j)\langle v_i,v_j\rangle = 0
$$

If $\lambda_i \ne \lambda_j$, then:

$$
\langle v_i,v_j\rangle = 0
$$

So eigenvectors with distinct eigenvalues are orthogonal.

Definiteness can be read from the eigenvalues:

- **positive definite**: all eigenvalues are $>0$;
- **positive semi-definite**: all eigenvalues are $\ge 0$;
- **negative definite**: all eigenvalues are $<0$;
- **negative semi-definite**: all eigenvalues are $\le 0$;
- **indefinite**: mixed signs.

#### Orthonormal Bases

An orthonormal basis $\{v_1,\ldots,v_D\}$ satisfies:

$$
\langle v_i,v_j\rangle =
\begin{cases}
1 & i=j \\
0 & i \ne j
\end{cases}
$$

So the basis vectors are mutually perpendicular and each has length 1.

If a symmetric matrix has eigenvectors $v_1,\ldots,v_D$ forming an orthonormal basis, then any vector $x$ can be written as:

$$
x = \sum_{j=1}^D \alpha_j v_j
$$

Applying $A$ gives:

$$
Ax = \sum_{j=1}^D \alpha_j \lambda_j v_j
$$

**Any matrix scales its eigenvectors. For a symmetric matrix, the eigenvectors can be chosen as an orthonormal basis, so the whole transformation is scaling along perpendicular eigen-directions.**

Repeated application emphasizes the largest eigenvalue direction:

$$
A^n x = \sum_{j=1}^D \alpha_j \lambda_j^n v_j
$$

The components with larger $|\lambda_j|$ dominate as $n$ grows.

#### Quadratic Forms

A symmetric matrix $A$ defines a quadratic form:

$$
q_A(x) = \langle x, Ax\rangle = x^T A x
$$

This is the vector analogue of a scalar quadratic $ax^2$.

Eigenvalues determine the shape:

- If all eigenvalues are positive, the quadratic form curves upward in every direction.
- If all eigenvalues are negative, it curves downward in every direction.
- If some eigenvalues are positive and some are negative, the surface has a saddle point.

For example:

$$
A =
\begin{pmatrix}
2 & 1 \\
1 & 2
\end{pmatrix}
$$

has positive eigenvalues, so its quadratic-form contours are ellipses.

Changing the sign of one eigenvalue flips the curvature along that eigenvector direction, producing a saddle.

#### Covariance Geometry

For a Gaussian $X \in \mathbb{R}^D$ with density $p(x;\mu,\Sigma)$:

$$
\operatorname{Cov}[X] = \Sigma
$$

Covariance matrices are symmetric because:

$$
\operatorname{Cov}[X_i,X_j] = \operatorname{Cov}[X_j,X_i]
$$

Since $\Sigma$ is symmetric, it has an eigenvector orthonormal basis. In that basis:

$$
\Sigma = \operatorname{diag}(\lambda_1,\ldots,\lambda_D)
$$

Equivalently, in the original coordinates, the covariance matrix can be reconstructed from its eigenvalues and eigenvectors:

$$
\Sigma
=
\sum_{i=1}^D \lambda_i v_i v_i^T
=
V\Lambda V^T
$$

where:

- $v_i$ are the orthonormal eigenvectors;
- $\lambda_i$ are the corresponding eigenvalues;
- $V = [v_1,\ldots,v_D]$;
- $\Lambda = \operatorname{diag}(\lambda_1,\ldots,\lambda_D)$.

The geometric interpretation:

- eigenvectors of $\Sigma$ give the principal directions of the Gaussian;
- eigenvalues $\lambda_i$ give the variances along those directions;
- rotating into the eigenbasis diagonalizes the covariance matrix.

Variance along a unit direction $a$ is:

$$
\operatorname{Var}[a^T X]
=
\mathbb{E}\left[
\left(a^T(X-\mu)\right)^2
\right]
=
\mathbb{E}\left[
a^T(X-\mu)(X-\mu)^T a
\right]
=
a^T\mathbb{E}\left[
(X-\mu)(X-\mu)^T
\right]a
=
a^T\Sigma a
$$

For an eigenvector direction $v_i$, the variance is the corresponding eigenvalue $\lambda_i$:

$$
\operatorname{Var}[v_i^T X]
=
v_i^T\Sigma v_i
=
v_i^T\lambda_i v_i
=
\lambda_i
$$

More generally, if $a=\sum_i \alpha_i v_i$ and $\|a\|=1$, then:

$$
a^T\Sigma a
=
\left(\sum_i \alpha_i v_i\right)^T
\Sigma
\left(\sum_j \alpha_j v_j\right)
=
\sum_i\sum_j
\alpha_i\alpha_j
v_i^T\Sigma v_j
=
\sum_i\sum_j
\alpha_i\alpha_j
\lambda_j
v_i^T v_j
=
\sum_i \alpha_i^2 \lambda_i
$$

**So directions with larger eigenvalues are higher-variance directions.**

After shifting by $\mu$ and rotating into the eigenbasis, define:

$$
X' = V^T(X-\mu)
$$

Then:

$$
\operatorname{Cov}[X']
=
\operatorname{Cov}[V^T(X-\mu)]
=
V^T\operatorname{Cov}[X]V
=
V^T\Sigma V
=
\Lambda
=
\operatorname{diag}(\lambda_1,\ldots,\lambda_D)
$$

For a Gaussian, diagonal covariance implies independent coordinates. So a multivariate Gaussian is a collection of independent scalar Gaussians in the right coordinate system:

$$
X_i' \sim \mathcal{N}(0,\lambda_i)
$$

#### Gaussian MLE

For data:

$$
D = (x_1,\ldots,x_n), \quad x_i \in \mathbb{R}^d
$$

under a Gaussian model:

$$
M = \{g(\cdot|\mu,\Sigma) \mid \mu \in \mathbb{R}^d,\ \Sigma \succ 0\}
$$

the maximum likelihood estimates are:

$$
\hat{\mu}_{ML} = \frac{1}{n}\sum_{i=1}^n x_i
$$

and

$$
\hat{\Sigma}_{ML}
=
\frac{1}{n}\sum_{i=1}^n
(x_i-\hat{\mu}_{ML})(x_i-\hat{\mu}_{ML})^T
$$

The mean estimate does not depend on $\Sigma$, so we can estimate $\hat{\mu}_{ML}$ first and then plug it into the covariance estimate.

Slides 56-58 give the matrix-derivative details behind this result. The important identities are that derivatives of trace terms handle quadratic forms, and the derivative of $\log |A|$ produces $A^{-T}$.

#### Multivariate Linear Regression

Slides 59-63 switch from modelling $p(x)$ to modelling a conditional distribution $p(y|x)$.

For paired data:

$$
D = \{(x_1,y_1),\ldots,(x_N,y_N)\}
$$

linear-Gaussian regression assumes:

$$
y = Wx + \epsilon
$$

with Gaussian noise:

$$
p(y|x,W,\Sigma_y)
=
|2\pi\Sigma_y|^{-\frac{1}{2}}
\exp\left(
-\frac{1}{2}(y-Wx)^T\Sigma_y^{-1}(y-Wx)
\right)
$$

The conditional log-likelihood is:

$$
\ell
=
\sum_i \log p(y_i|x_i,W,\Sigma_y)
$$

Maximizing it gives:

$$
\hat{W}
=
\left(\sum_i y_i x_i^T\right)
\left(\sum_i x_i x_i^T\right)^{-1}
$$

This is ordinary least squares written in matrix form.

#### Bayesian Linear Regression and Ridge

For scalar $y_i$, write the weights as a vector $w$ and use a Gaussian prior:

$$
p(w|A) = \mathcal{N}(0,A^{-1})
$$

With Gaussian observation noise of variance $\sigma_y^2$, the posterior is also Gaussian:

$$
p(w|D,A,\sigma_y)
=
\mathcal{N}(\mu_w,\Sigma_w)
$$

where:

$$
\Sigma_w
=
\left(
A + \sigma_y^{-2}\sum_i x_i x_i^T
\right)^{-1}
$$

and

$$
\mu_w
=
\Sigma_w
\left(
\sigma_y^{-2}\sum_i y_i x_i
\right)
$$

Because the posterior is Gaussian, the MAP estimate and posterior mean are the same:

$$
w_{MAP}
=
\left(
A\sigma_y^2 + \sum_i x_i x_i^T
\right)^{-1}
\sum_i y_i x_i
$$

Compare this with the maximum-likelihood estimate:

$$
w_{ML}
=
\left(
\sum_i x_i x_i^T
\right)^{-1}
\sum_i y_i x_i
$$

The prior shrinks weights toward the prior mean, here zero. If $A=\alpha I$, this is ridge regression: ordinary linear regression with a squared L2 weight penalty $\alpha\|w\|_2^2$.

The key modeling distinction:

- Linear regression models $p(y|x)$.
- If we also model $p(x)$, then $(x,y)$ can be treated as a joint generative model.
- If $p(x)$ is Gaussian and $p(y|x)$ is linear-Gaussian, then $(x,y)$ is jointly Gaussian.

---

## Lecture 2: [Latent Variable Models](assets/course-gatsby-pul-slides-2025/lecture-01-05-probabilistic-unsupervised-learning.pdf)

### Latent Variable Models

_**TL;DR:** Latent-variable models explain structure in observed data $x$ by introducing hidden variables $z$ that generate or influence the observations._

Slides 65-68 introduce the basic setup:

$$
z \sim p(z;\theta_z)
$$

$$
x|z \sim p(x|z;\theta_x)
$$

The joint distribution factorizes as:

$$
p(x,z;\theta_x,\theta_z)
=
p(x|z;\theta_x)p(z;\theta_z)
$$

The observed-data distribution is obtained by marginalizing out the latent variable:

$$
p(x;\theta_x,\theta_z)
=
\int p(x|z;\theta_x)p(z;\theta_z)dz
$$

Core idea:

- $x$ is observed.
- $z$ is hidden.
- Dependence and structure in $x$ can be explained by shared dependence on $z$.

Why introduce $z$?

- To model structured high-dimensional distributions with fewer effective parameters.
- To represent an underlying generative process, where $z$ may correspond to causes like object identity, pose, illumination, or hidden state.
- To separate signal from noise.
- To combine simple distributions into richer marginal distributions over $x$.

Important caveat:

- $p(z)$, $p(x|z)$, and even $p(x,z)$ may be simple distributions, often exponential-family distributions.
- But the marginal $p(x)$ after integrating out $z$ is often more complex.
- Linear-Gaussian models are a special case where the marginal remains Gaussian.

#### Latent Variables and Gaussian Correlation

Slide 68 shows the Gaussian intuition: correlation in $x$ can be generated by shared latent causes plus independent noise.

Instead of directly writing a correlated Gaussian covariance for $x$, we can write:

$$
z \sim \mathcal{N}(0,1)
$$

and then let $x$ depend on $z$ with additional uncorrelated Gaussian noise.

The shared latent variable creates correlation between the observed dimensions of $x$. The independent noise accounts for dimension-specific variation.

So a latent-variable model can explain observed correlation as:

$$
\text{shared latent component} + \text{independent noise}
$$

#### Probabilistic PCA

_**TL;DR:** PPCA is PCA written as a latent-variable Gaussian model: low-dimensional latent coordinates generate high-dimensional observations plus isotropic Gaussian noise._

Slides 69-70 define PPCA.

$$
x_i \in \mathbb{R}^D,
\quad
z_i \in \mathbb{R}^K,
\quad
K < D
$$

The linear generative model is:

$$
x = Wz + \epsilon
$$

where:

$$
z \sim \mathcal{N}(0,I)
$$

and

$$
\epsilon \sim \mathcal{N}(0,\psi I)
$$

Equivalently:

$$
p(z) = \mathcal{N}(0,I)
$$

$$
p(x|z) = \mathcal{N}(Wz,\psi I)
$$

Here:

- $W$ is a $D \times K$ loading matrix that maps low-dimensional latent coordinates into observation space.
- $\psi I$ is isotropic noise: same independent noise variance in every observed dimension.
- $K<D$, so the structured variation is lower-dimensional than the observation space.

Because this is a linear-Gaussian model, marginalizing out $z$ keeps $p(x)$ Gaussian:

$$
p(x)
=
\int p(z)p(x|z)dz
=
\mathcal{N}(0,WW^T+\psi I)
$$

So the covariance of $x$ decomposes into:

$$
\underbrace{WW^T}_{\text{shared low-dimensional structure}}
+
\underbrace{\psi I}_{\text{independent isotropic noise}}
$$

This gives two views of the same kind of covariance:

- A full Gaussian $p(x)=\mathcal{N}(0,\Sigma)$ is a descriptive model: correlations are stored directly in $\Sigma$.
- PPCA is a latent causal model: correlations arise because observed dimensions share the same latent variables.

The tradeoff is:

- A full covariance matrix has $\frac{D(D+1)}{2}$ free parameters.
- PPCA has $DK+1$ covariance parameters through $WW^T+\psi I$.
- For $K<D$, PPCA is more constrained and more interpretable, but maximum-likelihood estimation is less direct.

#### PPCA Likelihood and PCA Limit

Slide 71 writes the PPCA likelihood using the marginal Gaussian:

$$
p(x) = \mathcal{N}(0,WW^T+\psi I)
$$

The sample covariance is:

$$
S = \frac{1}{N}\sum_n x_nx_n^T
$$

If the data are not centered, first estimate and subtract the mean:

$$
\hat{\mu} = \frac{1}{N}\sum_n x_n
$$

Then:

$$
S = \frac{1}{N}\sum_n (x_n-\hat{\mu})(x_n-\hat{\mu})^T
$$

The course notes that PPCA parameters can be optimized numerically, or later with EM.

Slide 72 gives the connection to PCA: as $\psi \to 0$, the model has no isotropic noise left to explain residual variance. The latent model can only represent $K$ dimensions of variance, so maximum likelihood chooses the $K$-dimensional subspace with the most variance.

#### Principal Components Analysis

Slide 73 states the PCA algorithm for zero-mean data.

PCA finds directions of maximum variance, where $n$ indexes the data points $x_1,\ldots,x_N$:

$$
v^{(1)}
=
\arg\max_{\|v\|=1}
\sum_{n=1}^N (x_n^T v)^2
$$

Then it finds the next direction with greatest variance subject to being orthogonal to the previous directions, and repeats.

PCA vs. PPCA relationship:

- PCA finds a low-dimensional subspace that captures maximal variance.
- PPCA gives a probabilistic model whose zero-noise limit recovers that PCA subspace.
- In PPCA, the latent variables $z$ are the low-dimensional coordinates and $W$ maps them back into the observed space.
