# Gated Linear Networks (GLN)

- **Created**: 2026-01-23
- **Last Updated**: 2026-01-24
- **Status**: `In Progress`

---

- [ ] [2017] Online Learning with Gated Linear Networks - [paper](https://arxiv.org/abs/1712.01897)
- [ ] [2019] Gated Linear Networks - [paper](https://arxiv.org/abs/1910.01526)
- [ ] [2020] Gaussian Gated Linear Networks - [paper](https://arxiv.org/abs/2006.05964)
- [ ] [2020] Online Learning in Contextual Bandits using Gated Linear Networks - [paper](https://arxiv.org/abs/2002.11611)
- [ ] [2020] A Combinatorial Perspective on Transfer Learning - [paper](https://arxiv.org/abs/2010.12268)
- [ ] [2021] A Rapid and Efficient Learning Rule for Biological Neural Circuits - [paper](https://www.biorxiv.org/content/10.1101/2021.03.10.434756)

---

## [2017] [Core] Online Learning with Gated Linear Networks - [paper](https://arxiv.org/abs/1712.01897)

- **Date**: 2026-01-23
- **Paper**: <https://arxiv.org/abs/1712.01897>
- **Paperpile**: <https://app.paperpile.com/view/?id=9c182ef6-6b82-4f11-8a5b-e69b9c1ef5fb>
- **Assistant**: TODO

---

- **Abstract**:
  - > This paper describes a family of probabilistic architectures designed for online learning under the logarithmic loss.  Rather than relying on non-linear transfer functions, our method gains representational power by the use of data conditioning.  We state under general conditions a learnable capacity theorem that shows this approach can in principle learn any bounded Borel-measurable function on a compact subset of euclidean space; the result is stronger than many universality results for connectionist architectures because we provide both the model and the learning procedure for which convergence is guaranteed.
- **1. Intro**:
  - > This paper explores the use of **techniques from the online learning and data compression communities for the purpose of high dimensional density modeling, with a particular emphasis on image density modeling**.
  - > Our main contribution is to show that a certain family of neural networks, composed of techniques originating from the data compression and online learning communities, can **in some circumstances match the performance of deep learning based approaches in just a single pass through the data**, while also enjoying universal source coding guarantees.
  - > Foerster et al. (2017) recently introduced a novel recurrent neural architecture whose modeling power was derived from using a data-dependent affine transformation as opposed to  a  non-linear  activation  function.   As  we  shall  see  later,  this  is  similar  in  spirit  to  our approach;  we use a product of data-dependent weight matrices to provide representation power  instead  of  a  non-linear  activation  function.   Our  work  differs  in  that  we  consider an  online  setting,  and  use  a  local  learning  rule  instead  of  backpropagation  to  adjust  the weights.
  - **Contributions**:
    - **(1) Gated Linear Networks**: A family of models consisting of a sequence of data dependent linear networks coupled with an appropriate choice of gating function.
    - **(2) Theoretical Justification**: Theoretical baiss for the local weight learning mechanism in these architectures. While gating was originally introduced for computational reasons, it also adds meaningful representational power.
    - **(3) Adaptive Regularization**: A technique that allows a GLN to have competitive loss guarantees with respect to all possible sub-networks obtained via pruning the original network.
    - **(4) Effective Capacity Theorem**: It proves that given a large enough network and the right gating function, these networks can learn any continuous density function to an arbitrary level of accuracy using local learning rules.
- **2.2 Geometric Mixing**:
  - an adaptive, online ensemble technique to combine predictions from multiple probabilistic models into a single, unified conditional probability estimate.
  - > Given $m$ sequential, probabilistic, binary models $\rho_1,...,\rho_m$, Geometric Mixing provides a principled way of combining the $m$ associated conditional probability distributions into a single conditional probability distribution, giving rise to a probability measure on binary sequences that has a number of desirable properties.
  - **Geometric Mixture**: $\text{GEO}_w(x_t = 1; p_t) = \frac{\prod_{i=1}^m p_{t,i}^{w_i}}{\prod_{i=1}^m p_{t,i}^{w_i} + \prod_{i=1}^m (1 - p_{t,i})^{w_i}}$, where
    - $x_t \in \{0,1\}$ denotes the boolean target at time $t$,
    - $p_t = (\rho_1(x_t = 1 | x_{<t}), \ldots, \rho_m(x_t = 1 | x_{<t}))$,
    - $\rho_1,...,\rho_m$ are $m$ sequential, probabilistic, binary models,
    - $w \in W$ is the parameter vector and $W \subset \mathbb{R}^m$ is a convex set.
  - **Properties of Geometric Mixture**:
    - $\text{GEO}_w(x_t = 0; p_t) = 1 - \text{GEO}_w(x_t = 1; p_t)$.
    - Setting $w_i = 1/m$ for $i \in [1, m]$ is equivalent to taking the geometric mean of the $m$ input probabilities.
    - [Fig1] Higher absolute values of $w_i$ translate into an increased belief into model $i$'s prediction; for negative values of $w_i$, the prediction needs to be reversed.
    - If $w_i = 0$, then the contribution of the model $i$ is ignored (since $p_{t,i}^{w_i} = p_{t,i}^0 = 1$).
    - If $w_i = 0$ for all $i \in \{1,..,m\}$, then $\text{GEO}_w(x_t = 1; p_t) = 1/2$.
    - Due to the product formulation, every model also has **the right of veto**: a single $p_{t,i}$ close to $0$ coupled with a $w_i > 0$ drives $\text{GEO}_w(x_t = 1; p_t)$ close to zero.
  - **Alternate Form**: $\text{GEO}_w(x_t = 1; p_t) = \sigma(w \cdot \text{logit}(p_t))$, where
    - $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function,
    - $\text{logit}(x) = \log\left(\frac{x}{1 - x}\right)$ is the inverse of the sigmoid function.
    - > This form is **best suited for numerical implementation. Furthermore, the property of **having an input non-linearity that is the inverse of the output non-linearity is the reason why a linear network is obtained when layers of geometric mixers are stacked on top of each other**.
  - **Alternate Form Derivation**:
    - Start with the original form: $\text{GEO}_w(x_t = 1; p_t) = \frac{\prod p_{t,i}^{w_i}}{\prod p_{t,i}^{w_i} + \prod (1 - p_{t,i})^{w_i}}$
    - Divide numerator and denominator by $\prod (1 - p_{t,i})^{w_i}$: $\text{GEO}_w(x_t = 1; p_t) = \frac{\prod (p_{t,i} / (1-p_{t,i}))^{w_i}}{\prod (p_{t,i} / (1-p_{t,i}))^{w_i} + 1}$
    - Define intermediate term $Q = \prod_{i=1}^m \left( \frac{p_{t,i}}{1 - p_{t,i}} \right)^{w_i}$
    - Taking log: $\log(Q) = \sum_{i=1}^m w_i \log \left( \frac{p_{t,i}}{1 - p_{t,i}} \right)$
    - Substituting logit definition: $\log(Q) = \sum w_i \cdot \text{logit}(p_{t,i}) = w \cdot \text{logit}(p_t)$
    - Taking exp: $Q = e^{w \cdot \text{logit}(p_t)}$
    - Substitute $Q$ back into the equation: $\text{GEO}_w(x_t = 1; p_t) = \frac{e^{w \cdot \text{logit}(p_t)}}{e^{w \cdot \text{logit}(p_t)} + 1}$
    - Multiply by $e^{-x}/e^{-x}$ to reach sigmoid form: $\text{GEO}_w(x_t = 1; p_t) = \frac{1}{1 + e^{-w \cdot \text{logit}(p_t)}}$
    - Final Result: **$\text{GEO}_w(x_t = 1; p_t) = \sigma(w \cdot \text{logit}(p_t))$**.
  - **Logarithmic Loss (Binary Cross-Entropy Loss)**:
    - At each time $t$, the predictor outputs a binary distribution: $\text{GEO}_w(.; p_t) \to [0,1]$.
    - $x_t \in \{0,1\}$: binary ground-truth observation at time $t$.
    - Logarithmic loss (binary cross-entropy loss) $l_t(\text{GEO}_w(.; p_t), x_t)$ between the predictor's output and the ground-truth observation.
    - **Online learning**: Loss applied to the predictor before moving to time $t+1$.
  - **Properties under Logarithmic Loss**:
    - TODO
- **2.3 Gated Geometric Mixture**:
  - TODO
