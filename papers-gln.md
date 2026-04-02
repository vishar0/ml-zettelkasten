# Gated Linear Networks (GLN)

- **Created**: 2026-01-23
- **Last Updated**: 2026-01-30
- **Status**: `In Progress`

---

- [x] [2017] Online Learning with Gated Linear Networks - [paper](https://arxiv.org/abs/1712.01897)
- [x] [2019] Gated Linear Networks - [paper](https://arxiv.org/abs/1910.01526)
- [x] [2020] Gaussian Gated Linear Networks - [paper](https://arxiv.org/abs/2006.05964)
- [ ] [2020] Online Learning in Contextual Bandits using Gated Linear Networks - [paper](https://arxiv.org/abs/2002.11611)
- [ ] [2020] A Combinatorial Perspective on Transfer Learning - [paper](https://arxiv.org/abs/2010.12268)
- [ ] [2021] A Rapid and Efficient Learning Rule for Biological Neural Circuits - [paper](https://www.biorxiv.org/content/10.1101/2021.03.10.434756)

---

## [2017] Online Learning with Gated Linear Networks

- **Date**: 2026-01-23
- **Arxiv**: <https://arxiv.org/abs/1712.01897>
- **Paperpile**: <https://app.paperpile.com/view/?id=9c182ef6-6b82-4f11-8a5b-e69b9c1ef5fb>
- **Assistant**: <https://gemini.google.com/share/4d45791afe36>
- **Code**: <https://github.com/google-deepmind/deepmind-research/tree/master/gated_linear_networks>

---

- **Abstract**:
  - > This paper describes a family of probabilistic architectures designed for online learning under the logarithmic loss.
  - > Rather than relying on non-linear transfer functions, our method gains representational power by the use of data conditioning.
    - Rather than using non-linear activations, representational power comes input/context-dependent weights (such as in hypernetworks).
  - > We state under general conditions a learnable capacity theorem that shows this approach can in principle learn any bounded Borel-measurable function on a compact subset of euclidean space;
    - Borel-measurable functions: Larger set than just continuous functions that standard neural nets are good at. Can have discontinuities.
  - > the result is stronger than many universality results for connectionist architectures because we provide both the model and the learning procedure for which convergence is guaranteed.
- **1. Intro**:
  - > This paper explores the use of **techniques from the online learning and data compression communities for the purpose of high dimensional density modeling, with a particular emphasis on image density modeling**.
  - > Our main contribution is to show that a certain family of neural networks, composed of techniques originating from the data compression and online learning communities, can **in some circumstances match the performance of deep learning based approaches in just a single pass through the data**, while also enjoying universal source coding guarantees.
  - > Foerster et al. (2017) recently introduced a novel recurrent neural architecture whose modeling power was derived from using a data-dependent affine transformation as opposed to  a  non-linear  activation  function.   As  we  shall  see  later,  this  is  similar  in  spirit  to  our approach;  we use a product of data-dependent weight matrices to provide representation power  instead  of  a  non-linear  activation  function.   Our  work  differs  in  that  we  consider an  online  setting,  and  use  a  local  learning  rule  instead  of  backpropagation  to  adjust  the weights.
  - **Contributions**:
    - **(1) Gated Linear Networks**: A family of models consisting of a sequence of data dependent linear networks coupled with an appropriate choice of gating function.
    - **(2) Theoretical Justification**: Theoretical baiss for the local weight learning mechanism in these architectures. While gating was originally introduced for computational reasons, it also adds meaningful representational power.
    - **(3) Adaptive Regularization**: A technique that allows a GLN to have competitive loss guarantees with respect to all possible sub-networks obtained via pruning the original network.
    - **(4) Effective Capacity Theorem**: It proves that given a large enough network and the right gating function, these networks can learn any continuous density function to an arbitrary level of accuracy using local learning rules.
- **2. Geometric Mixing**:
  - an adaptive, online ensemble technique to combine predictions from multiple probabilistic models into a single, unified conditional probability estimate.
  - > Given $m$ sequential, probabilistic, binary models $\rho_1,...,\rho_m$, Geometric Mixing provides a principled way of combining the $m$ associated conditional probability distributions into a single conditional probability distribution, giving rise to a probability measure on binary sequences that has a number of desirable properties.
  - **Geometric Mixture**: $$\text{GEO}_w(x_t = 1; p_t) = \frac{\prod_{i=1}^m p_{t,i}^{w_i}}{\prod_{i=1}^m p_{t,i}^{w_i} + \prod_{i=1}^m (1 - p_{t,i})^{w_i}}$$ where
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
  - **Alternate Form**: $$\text{GEO}_w(x_t = 1; p_t) = \sigma(w \cdot \text{logit}(p_t))$$ where
    - $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function,
    - $\text{logit}(x) = \log\left(\frac{x}{1 - x}\right)$ is the inverse of the sigmoid function.
    - > This form is best suited for numerical implementation. Furthermore, the property of **having an input non-linearity that is the inverse of the output non-linearity is the reason why a linear network is obtained when layers of geometric mixers are stacked on top of each other**.
  - **Alternate Form Derivation**:
    - Start with the original form: $$\text{GEO}_w(x_t = 1; p_t) = \frac{\prod p_{t,i}^{w_i}}{\prod p_{t,i}^{w_i} + \prod (1 - p_{t,i})^{w_i}}$$
    - Divide numerator and denominator by $\prod (1 - p_{t,i})^{w_i}$: $$\text{GEO}_w(x_t = 1; p_t) = \frac{\prod (p_{t,i} / (1-p_{t,i}))^{w_i}}{\prod (p_{t,i} / (1-p_{t,i}))^{w_i} + 1}$$
    - Define intermediate term $Q$: $$Q = \prod_{i=1}^m \left( \frac{p_{t,i}}{1 - p_{t,i}} \right)^{w_i}$$
    - Taking log: $$\log(Q) = \sum_{i=1}^m w_i \log \left( \frac{p_{t,i}}{1 - p_{t,i}} \right)$$
    - Substituting logit definition: $$\log(Q) = \sum w_i \cdot \text{logit}(p_{t,i}) = w \cdot \text{logit}(p_t)$$
    - Taking exp: $$Q = e^{w \cdot \text{logit}(p_t)}$$
    - Substitute $Q$ back into the equation: $$\text{GEO}_w(x_t = 1; p_t) = \frac{e^{w \cdot \text{logit}(p_t)}}{e^{w \cdot \text{logit}(p_t)} + 1}$$
    - Multiply by $e^{-x}/e^{-x}$ to reach sigmoid form: $$\text{GEO}_w(x_t = 1; p_t) = \frac{1}{1 + e^{-w \cdot \text{logit}(p_t)}}$$
    - Final Result: **$$\text{GEO}_w(x_t = 1; p_t) = \sigma(w \cdot \text{logit}(p_t))$$**
  - **Logarithmic Loss (Binary Cross-Entropy Loss)**:
    - At each time $t$, the predictor outputs a binary distribution: $\text{GEO}_w(.; p_t) \to [0,1]$.
    - $x_t \in \{0,1\}$: binary ground-truth observation at time $t$.
    - Logarithmic loss (binary cross-entropy loss) $l_t(\text{GEO}_w(.; p_t), x_t)$ between the predictor's output and the ground-truth observation.
    - **Online learning**: Loss applied to the predictor before moving to time $t+1$.
  - **Properties under Logarithmic Loss**:
    - The log loss of a geometric mixer w.r.t. weights $w$ is **convex** in $w$. This is the key property that makes local learning tractable — there are no local minima to get stuck in, so simple online gradient descent is sufficient.
    - **Proposition** — for all $t$, $x_t \in \{0,1\}$, $p_t \in (0,1)^m$, $w \in \mathcal{W}$:
      1. **Gradient**: $\nabla \ell^\text{geo}_t(w) = \left(\text{GEO}_w(1;p_t) - x_t\right)\text{logit}(p_t)$ — prediction error × logit of inputs, same structure as logistic regression.
      2. **Gradient norm**: $\|\nabla \ell^\text{geo}_t(w)\|_2 \leq \|\text{logit}(p_t)\|_2$
      3. **Convexity**: $\ell^\text{geo}_t(w)$ is convex in $w$.
      4. If inputs are clipped: $p_t \in [\epsilon, 1-\epsilon]^m$:
         - **Exp-concavity**: $\ell^\text{geo}_t$ is $\alpha$-exp-concave (a strictly stronger condition than convexity that allows curvature-aware updates).
         - **Tighter gradient norm**: $\|\nabla \ell^\text{geo}_t(w)\|_2 \leq \sqrt{m}\log(1/\epsilon)$
    - **Optimization options**:
      - Convexity alone → Online Gradient Descent → $O(\sqrt{T})$ regret (average error vanishes).
      - Exp-concavity → Online Newton Step → $O(\log T)$ regret (much faster convergence, but more compute per step).
- **2. Gated Geometric Mixing Neuron [Fig2]**: Contextual Gating + Geometric Mixing
  - **Contextual Gating**: Mapping particular examples to particular sets of weights. **Similar in concept to hypernetworks**, except that the paper doesn't use a neural network but a hashtable of weights.
  - **Context Function** $c \colon Z \to C$, where $Z$ is the set of possible side information and $C = {0,...,k − 1}$ for some $k \in N$ is the context space.
    - Given a piece of side information $z_t \in Z$, $c(z_t)$ outputs an index into an a weight table $W \subset \mathbb{R}^d$, where each entry outputs the weight $w_{c(z_t)}$ to use with standard geometric mixing.
  - **Gated Geometrix Mixer** $$\text{GEO}_W^c(x_t = 1; p_t, z_t) = \text{GEO}_{w_{c(z_t)}}(x_t = 1; p_t) = \sigma (w_{c(z_t)} \cdot \text{logit}(p_t))$$
    - > The key idea is that **our neuron can now specialize its weighting of the input predictions based on some property of the side information** $z_t$. The side information can be arbitrary, for example it could be some additional input features, or even functions of $p_t$.  Ideally the choice of context function should be informative in the sense that it simplifies the probability combination task.
  - **Classes of Context Functions** (inexhaustive):
    - **(a) Half-space contexts**: For real-valued side information.
      - **Concept**: It uses a hyperplane (a flat boundary in space defined by a normal vector $v$ and an offset $b$) to slice the input space into two halves.
      - **Mechanism**: It checks if a point $z$ falls on one side of the boundary or the other (specifically, if the dot product $x \cdot v \geq b$).
      - **Result**: This creates a binary "yes/no" context. By combining many of these random half-space cuts, the network can partition a complex continuous space (like an image or coordinate system) into fine-grained regions, assigning different weights to each region.
    - **(b) Skip-gram contexts**: For binary or categorical inputs.
      - **Concept**: It focuses on specific components (dimensions) of the input vector.
      - **Mechanism**: It checks if the $i$-th bit of the input is active (i.e., if $z_i = 1$).
      - **Result**: This allows the model to learn specific weights for the presence of specific features. For example, in text processing, it might learn a specific weight adjustment whenever the word "not" appears (the bit for "not" is 1).
  - **Context Function Composition**: Multiple context functions can be combined into a higher-order context function with the total context space being the product of the individual context spaces.
- **3. GLN: Gated Linear Networks [Fig3]**:
  - Feedforward networks composed of gated geometric mixing neurons. Each neuron in layer $i$ outputs a gated geometric mixture over predictions from layer $i-1$.
  - **Note**: The **input logit in layer $i$ cancels out the output sigmoid of layer $i-1$** (since the logit function is the inverse of sigmoid function). With fixed weights, this would be a fully linear MLP without non-linearity. But expressivity arises due to the weights being input dependant.
    - This structure allows the network to **effectively be a linear network for a fixed input context $z$ for the purpose of training (which guarantees convexity and easy optimization), while being a non-linear network for the purpose of modeling complex data**.
- **3. Learning in GLN**:
  - > While architecturally **a GLN appears superficially similar to the well-known multilayer perception (MLP), what and how it learns is very different**. The key difference is that **every neuron in a GLN probabilistically predicts the target**. This allows us to associate **a loss function to each neuron**. This loss function will be defined in terms of just the parameters of the neuron itself; thus, **unlike backpropagation, learning will be local**.
  - > Furthermore, **this loss function will be convex**, which will allow us to avoid many of the difficulties associated with training typical deep architectures. For example, we **can get away with simple deterministic weight initializations, which aids the reproducibility** of empirical results. The **convexity allows us to learn from correlated inputs in an online fashion** without suffering significant degradations in performance. And as we shall see later, GLNs are extremely data efficient, and can produce state of the art results in a single pass through the data.
  - > One should **think of each layer as being responsible for trying to directly improve the predictions of the previous layer, rather than a form of implicit non-linear feature/filter construction** as is the case with MLPs trained offline with back-propagation (Rumelhart et al., 1988).
  - **Weight Init**:
    - Given the loss is convex, unlike in non-convex optimization, choice of weight selection is less critical.
    - Weights are restricted to some scaled hypercube: $w_{ijc} \in [-b,b]^{K_{i-1}}$, where $w_{ijc}$ is the weight vector for neuron $j$ in layer $i$ with context id $c$, $K_{i-1}$ is the number of neurons in layer $i-1$, and $b \ge 1$.
    - **(a) Zero Init**: $w_{ijc} = 0$. Acts as a **sparsity prior**.
    - **(b) Geometric Average Init**: $w_{ijc} = 1/K^{i-1}$. Geometric mean of inputs.
    - **(c) Small Random Init**: Little practical difference, negative impact on reproducibility.
  - **Weight Update**:
    - **Local Learning**: Each neuron in each layer probabilistically predicts the target, and loss is applied to each neuron locally.
    - **Online Gradient Descent**: Log loss applied to each neuron, gradient computed, and weight updated applied with a suitable learning rate. After update, weights are clipped to $[-b,b]$ to restrict to project onto the hypercube $[-b,b]^{K_{i-1}}$.
    - **Time Complexity**: $O(K_{i-1})$ to update the weight of any neuron in layer $i$.
  - **Performance Guaratees**:
    - Regret (difference between actual loss and best possible loss) $R_{ik}(n) \leq 3 b K_{i-1} \sqrt{|C|n} \log \left( \frac{1}{\epsilon} \right)$, where
      - $R_{ik}(n)$: The total regret for a specific neuron (layer $i$, index $k$) after seeing $n$ examples. It measures how much worse the neuron performed compared to the best possible fixed set of weights chosen in hindsight.
      - $b$: The weight bound. The weights are constrained to lie within a specific range $[-b, b]$. A larger $b$ means the search space is bigger, making learning harder.
      - $K_{i-1}$: The number of inputs to the neuron n. This represents the "width" of the previous layer ($i-1$). The regret grows linearly with this because having more inputs to combine makes the optimization problem more complex.
      - $|C|$: The number of contexts. This is the size of the set of all possible contexts (e.g., if you have 6 binary features, $|C| = 2^6 = 64$). More contexts split the data into more "buckets," affecting how quickly each bucket collects enough data to learn.
      - $n$: The number of data points (time steps) processed so far. The fact that the bound includes $\sqrt{n}$ means the average regret ($R/n$) shrinks toward zero as $n$ increases.
      - $\epsilon$: The clipping parameter. The input probabilities from the previous layer are clipped to be within $[\epsilon, 1-\epsilon]$ to prevent numerical instability. $\log(1/\epsilon)$: This term arises from the bound on the gradient. If inputs get too close to 0 or 1 (small $\epsilon$), the gradients can explode, making learning unstable and increasing the regret.
- **3. Computational Properties of GLN**:
  - **Complexity of a single online learning step**: $O(\sum_{i=1}^L K_i K_{i-1})$ for $L$ layers and $K_i$ being the number of neurons in layer $i$. Same complexity for forward and backward.
  - **Parallelism**:
    - > When generating a prediction, parallelism can occur within a layer, similar to an MLP. The local training rule however enables all the neurons to be updated simul- taneously, as they have no need to communicate information to each other.  This compares favorably  to  back-propagation  and  significantly  simplifies  any  possible  distributed  imple- mentation.  Furthermore, as the bulk of the computation is primarily matrix multiplication, large speedups can be obtained straightforwardly using GPUs.
- **4. Effective Capacity of GLN**:
  - Standard universality proofs for neural nets say: *there exist weights* that can approximate any function. They say nothing about whether training will find those weights. The GLN result is stronger: it proves that a specific learning rule (OGD) will *converge* to the right answer — this is called **effective capacity**.
  - **What the theorem says** (informally): Given enough data and a sufficiently rich set of context functions, a GLN trained with a no-regret algorithm will converge to the true function $f$ almost everywhere.
  - The convergence is layerwise: each neuron learns to predict $f$ *averaged over the input regions its contexts carve out*. With more layers (or more neurons per layer), those regions get finer, and the approximation improves. In the limit of infinite depth, all neurons in a layer converge to the same output — and if the contexts are rich enough, that output equals $f$.
  - **What counts as "rich enough" contexts?** Half-space contexts (the kind used in practice) are sufficient, as long as you use enough of them. A two-layer network with many half-space neurons in the first layer can approximate any continuous function on a compact domain.
  - **Effective $\neq$ Capacity**: For a *fixed* architecture, there may be functions that *can* be represented but that OGD will never find. The failure case is XOR: neurons in the same layer optimize selfishly and never coordinate to solve it, so OGD gets stuck at $1/2$ — even though correct weights exist.
- **5. Adaptive Regularization via Sub-network Switching**:
  - **Problem — Catch-Up Phenomenon**: Early in training, lower-layer neurons predict better (they have a simpler job — fewer inputs to combine). Higher layers catch up as more data arrives, but by then the model has already wasted predictions deferring to them. Naively using the top neuron's output is suboptimal.
  - **Solution**: Since every neuron already outputs a probability for the target, we can maintain a weighted mixture over *all* neurons across all layers, and let the weights track which neuron is currently predicting best. This is a **switching ensemble**.
  - **Prior design**: Weight the mixture using a run-length encoding prior — essentially, sequences of neurons that stick to one neuron for a long time are assigned higher prior probability. This biases the ensemble toward stable predictions, and only "switches" when another neuron clearly starts outperforming.
  - **Guarantee**: If the best strategy is to follow a sequence of neurons that switches $s$ times, the ensemble's regret is only $O(s \log n)$ — logarithmic in data, not linear. If a single neuron is always best, the cost of running the ensemble is essentially zero.
  - **In practice**: On a 6-layer GLN, the ensemble weight starts concentrated on a mid-level (3rd layer) neuron, then gradually shifts up through layers 4, 5, and finally the top — automatically adapting the "effective depth" of the model as training progresses.
- **6. Experiments**:
  - **6.1. Non-Linear Decision Boundaries**:
    - To empirically verify GLNs can model non-linearity.
    - Ensemble of 3 GLNs to construct a one-vs-all classifier.
    - **Half-space context** by sampling 2D normal vectors, with (x,y) coordinates as side information.
    - Each component of all weight vectors were constrained to lie within $[−200, 200]$ (hypercube).
    - Input to GLN: component-wise sigmoid of (x,y) values as GLNs require the input to be within $[0,1]$.
  - **6.2. Online MNIST Classification**:
    - Ensemble of 10 GLNs to construct a one-vs-all classifier.
    - **6 half-space context functions** resulting in a weight table with 64 entries.
    - Each component of all weight vectors were constrained to lie within $[−200, 200]$ (hypercube).
    - Input: Preprocessed by applying mean-subtraction and de-skewing operation (Ghosh and Wan, 2017).
  - **6.3. Online MNIST Density Modeling**:
    - **Task**: Density model over binarized MNIST ($28 \times 28$ binary images).
    - **Autoregressive Factorization**: Use chain rule $P(X_1, \ldots, X_d) = \prod_i P(X_i | X_{<i})$ with row-major pixel ordering. Train 784 GLNs, one per pixel.
    - **Base Layer**: Up to 600 skip-gram predictions per pixel (exact count depends on pixel position). Geometric patterns known to help lossless image compression + randomly sampled pixel locations. Probabilities estimated online using the **Zero-Redundancy Estimator** (per-context, online).
    - **Context Functions**:
      - **Skip-gram**: Checks if specific earlier pixels are active — allows conditioning on the presence of nearby features.
      - **Max-pool**: Returns a binary-encoded index from max-pooled regions of the image — captures coarser spatial structure.
      - **Distance**: Returns the index of the nearest active pixel under various scan orderings (row, column, diagonal) — captures local texture/edge information.
    - **Network**: 4 layers, shape 35-60-35-70. 200 context functions randomly assigned across neurons. Learning rate: $\min(25/t,\ 0.005)$.
    - **Results**: **79.0 nats/image** on a single pass over the full dataset (train + val + test). Matches the state-of-the-art for batch-trained exact density models (PixelCNN) at the time, with no re-use of data.

## [2019] Gated Linear Networks

- **Date**: 2026-03-04
- **Arxiv**: <https://arxiv.org/abs/1910.01526>
- **Paperpile**: <https://app.paperpile.com/view/?id=3406470a-562f-418e-98af-eb9b1d566886>
- **Code**: <https://github.com/google-deepmind/deepmind-research/tree/master/gated_linear_networks>

---

- **Abstract**:
  - > This paper presents a new family of backpropagation-free neural architectures, Gated Linear Networks (GLNs). What distinguishes GLNs from contemporary neural networks is the **distributed and local nature of their credit assignment mechanism* each neuron directly predicts the target, forgoing the ability to learn feature representations in favor of rapid online learning**. Individual neurons can model nonlinear functions via the use of data-dependent gating in conjunction with online convex optimization. We show that *this architecture gives rise to universal learning capabilities in the limit, with effective model capacity increasing as a function of network size** in a manner comparable with deep ReLU networks. Furthermore, we demonstrate that the **GLN learning mechanism possesses extraordinary resilience to catastrophic forgetting**, performing comparably to a MLP with dropout and Elastic Weight Consolidation on standard benchmarks. These desirable theoretical and empirical properties position GLNs as a complementary technique to contemporary offline deep learning methods.
- **Intro**:
  - > Contemporary neural networks trained via backpropagation require many epochs of training over massive datasets, limiting their effectiveness for data-efficient online learning.
  - > Their effectiveness is further limited in the continual learning set- ting by their tendency to catastrophically forget previously learnt tasks.
  - > GLNs possess excellent online learning capabilities, which we demonstrate by showing performance competitive with batch-trained MLPs on a variety of standard classification, regression and density modeling tasks, using only a single online pass through the data. In terms of interpretibility, we show how the data-dependent linearity of the predictions can be exploited to trivialise the process of constructing mean- ingful saliency maps, which can be of great reassurance to practitioners that the model is predicting well for the right reasons.  Perhaps most interestingly, **we demonstrate that our credit assignment mechanism is extraordinarily resilient to catastrophic forgetting, achieving performance competitive with EWC on a standard continual learning benchmark with no knowledge of the task boundaries**.
- Delta from previous paper:
  - **Halfspace Sampling**:
    - Sample the normal vector $v$ uniformly from the surface of the unit sphere: draw i.i.d. $\mathcal{N}(0,1)$ components, then normalize. Sample the offset $b$ from $\mathcal{N}(0,1)$.
    - In high dimensions, randomly sampled hyperplanes are nearly orthogonal with high probability — so a set of $m$ halfspaces tends to cut the space in complementary, non-redundant ways.
    - The binary vector $g = (c_1(z), \ldots, c_m(z))$ is called the **signature** of input $z$. By locality-sensitive hashing theory, inputs close in cosine similarity map to similar signatures — so inputs that are geometrically nearby activate similar weight matrices and predict similarly. This is the inductive bias of halfspace-gated GLNs.
  - **Linear Interpretability & Saliency Maps**:
    - For a fixed input $z$, the product of data-dependent weight matrices $W_L(z) \cdots W_1(z)$ collapses to a single vector of the same dimension as the input (since $W_L$ has 1 row and $W_1$ has $K_0$ columns). The final prediction is $\sigma(\text{collapsed vector} \cdot \text{logit}(p_0))$.
    - This collapsed vector is a **multilinear polynomial of degree $L$** in the learned weights — it directly encodes how the network weights the input features for this specific example.
    - Saliency maps come for free: no gradient tricks or post-hoc analysis needed. Just read off the collapsed weight vector for any given input. Empirically, saliency maps on MNIST clearly preserve the characteristic digit shapes — the model is predicting for the right reasons.
  - **Empirical Capacity**:
    - The 2017 paper proves GLNs have universal capacity theoretically. This paper verifies it empirically: GLNs can memorize randomly labelled MNIST and pure noise datasets comparably to an MLP with the same total number of weights.
    - Capacity scales with both network width and context dimension — increasing either allows the model to fit more complex (or random) mappings.
  - **Resilience to Catastrophic Forgetting**:
    - Tested on permuted MNIST: 8 sequential tasks, each a different random pixel permutation of MNIST. No task boundaries given to the model.
    - GLN outperforms a standard MLP and matches EWC (Elastic Weight Consolidation) in a single pass per task. EWC only surpasses the GLN when given 10 passes per task.
    - **Why it works**: Inputs from different tasks have very different signatures (different halfspaces fire) since cosine distance between permuted images is large. This means different tasks activate nearly disjoint subsets of the weight table — gating acts as **implicit weight hashing**, giving tasks their own effective parameter space without any explicit mechanism.
  - **Convergence Rates**:
    - The loss $\ell_t(w)$ is not strongly convex everywhere — it's flat in directions orthogonal to $\text{logit}(p_t)$. But it *is* strongly convex in the subspace spanned by the observed gradients. Once $n > d$, this subspace is likely all of $\mathbb{R}^d$, so SGD with a $1/t$ learning rate achieves $O(\log T)$ regret (same as Online Newton Step but cheaper).
    - Learning proceeds layer by layer: after the first layer converges (in $\tilde{O}(1/\epsilon^2)$ steps), its outputs become approximately i.i.d. inputs for the second layer, which then converges, and so on. Overall convergence to $\epsilon$-accuracy is $O(L/\epsilon^2)$.
  - **Benchmarks**:
    - **MNIST classification**: 98% accuracy in a single pass (one-vs-all, 10 GLNs, 128 neurons/layer, context dimension 4).
    - **UCI datasets**: Single-pass GLN competitive with SVM, Gradient Boosting, and MLP trained for 100 epochs across diverse small-data classification tasks.
    - **MNIST density modeling**: 79.0 nats/image online; 80.74 nats if weights frozen at test time. Matches state-of-the-art batch-trained exact density models (PixelCNN). Additionally, unlike batch models, the online GLN can be directly coupled to an arithmetic decoder for lossless compression — batch models would need to first encode their (large) parameters, making them impractical for compression.

## [2020] Gaussian Gated Linear Networks

- **Date**: 2026-04-02
- **Arxiv**: <https://arxiv.org/abs/2006.05964>
- **Paperpile**: <https://app.paperpile.com/view/?id=84d4f164-fcfa-4b5c-a1de-babfcb7493e4>

---

- **Abstract**: Extends GLNs from binary (Bernoulli) to real-valued outputs by replacing geometric mixing with a **weighted Product of Gaussians**. All the desirable properties of B-GLNs carry over: local learning, convex loss, data-dependent gating, universality, catastrophic forgetting resilience.
- **Core Extension — Weighted Product of Gaussians (PoG)**:
  - The Gaussian analogue of geometric mixing. Given $m$ Gaussian experts $\mathcal{N}(\mu_i, \sigma_i^2)$ with weights $w \in \mathbb{R}_+^m$, their weighted product is itself a Gaussian (exponential families are closed under multiplication):
    $$\sigma^2_\text{PoG}(w) = \left[\sum_i \frac{w_i}{\sigma_i^2}\right]^{-1}, \qquad \mu_\text{PoG}(w) = \sigma^2_\text{PoG}(w) \sum_i \frac{w_i \mu_i}{\sigma_i^2}$$
  - Intuition: the output precision is a weighted sum of input precisions; the output mean is a precision-weighted average of input means.
  - The mean of a PoG must lie in the **convex hull** of the input means — an important constraint (see Bias Models below).
  - Multivariate case: replace scalars with covariance matrices — $\Sigma_\text{PoG}^{-1}(w) = \sum_i w_i \Sigma_i^{-1}$. For isotropic inputs ($\Sigma_i^{-1} = \tau_i \mathcal{I}$), the product is also isotropic, enabling $O(d)$ computation.
- **Weight Space**: Constrained to $w \in [0, b]^m$ with $\|w\| \geq \epsilon$ (non-negative, unlike B-GLN's $[-b,b]^m$). Non-negativity ensures variance stays well-defined and positive; the norm lower bound prevents the degenerate all-zero case.
- **Bias Models**: Since $\mu_\text{PoG}$ lies in the convex hull of input means, a G-GLN neuron can't predict outside the range spanned by its inputs. To let the network predict any target in $[-r, r]^D$, constant Gaussian PDFs with means $\pm r$ (or $\pm r$ along each basis vector in the multivariate case) are concatenated to every neuron's input. These act like learned intercepts.
- **The linear cancellation still holds**: Each neuron has a log input non-linearity and exp output non-linearity (from the PoG definition). These are inverses, so stacking layers cancels them — a G-GLN is also a gated linear network in the logspace of densities.
- **Experiments**:
  - **UCI Regression**: Beats variational inference, probabilistic backprop, and MC dropout on 7/9 standard benchmarks — in 40 epochs, no tricks.
  - **SARCOS** (21-dim → 7-dim robot inverse dynamics): MSE of 0.10, vs 0.14 for TabNet-L and 2.13 for MLP. Best result on the benchmark.
  - **Contextual Bandits (continuous rewards)**: Extends GLCB (from the 4th paper in this list) to real-valued rewards using G-GLN. Best mean rank across 3 bandit tasks vs 9 Bayesian deep learning methods, in a fully online regime.
  - **Denoising / Density Estimation**: Train as a denoising autoencoder (add noise, regress to clean). At convergence, $(x - \mu_{L,1}(x))/\lambda \approx \nabla_x \log p(x)$ — the score function. Feed into HMC to sample from the implied distribution. G-GLN recovers Swiss Roll and MNIST structure from a single online pass; MLPs require larger batches and more data.
