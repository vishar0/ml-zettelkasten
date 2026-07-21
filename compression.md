# Compression

- **Created**: 2026-06-07
- **Last Updated**: 2026-07-17
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
- [ ] [jveness] [2023] Language Modeling is Compression - [paper](https://arxiv.org/abs/2309.10668), [code](https://github.com/google-deepmind/language_modeling_is_compression)
- [x] [blog] [2026] Can gzip be a language model? - [blog](https://nathan.rs/posts/gzip-lm/), [code](https://github.com/nathanrs/gzipt)
- [ ] [2024] Compression Represents Intelligence Linearly - [paper](https://arxiv.org/abs/2404.09937)
- [ ] [jveness] [2014] CNC: Compress and Control - [paper](https://arxiv.org/abs/1411.5326), [slides](https://www.hutter1.net/publ/scnc.pdf)
- [x] [jveness] [2025] ActivePTW: Partition Tree Weighting for Non-Stationary Stochastic Bandits - [paper](https://arxiv.org/abs/2502.19325), [code](https://github.com/google-deepmind/active_ptw)
- [ ] [albertgu] [2025] CompressARC: ARC-AGI Without Pretraining - [blog](https://iliao2345.github.io/blog_posts/arc_agi_without_pretraining/arc_agi_without_pretraining.html), [paper](https://arxiv.org/abs/2512.06104). cf. [[papers-latent-recursive-reasoning]]
- [ ] [[papers-small-language-models]] [2025] [jxmo,FAIR] How Much Do Language Models Memorize? - [paper](https://arxiv.org/abs/2505.24832)
- [ ] [2022] Less is More: Parameter-Free Text Classification with Gzip - [paper](https://arxiv.org/abs/2212.09410)
- [x] [2025] zip2zip: Inference-Time Adaptive Tokenization via Online Compression - [paper](https://arxiv.org/abs/2506.01084)
- [ ] [schmidhuber] [2009] Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes - [paper](https://arxiv.org/abs/0812.4360). cf. [[papers-open-ended-learning]]
- [ ] [2018] [FAIR] Description Length of Deep Learning Models — [papers](https://arxiv.org/abs/1802.07044)
- [ ] [2019] BB-ANS: Practical Lossless Compression with Latent Variables using Bits Back Coding - [paper](https://arxiv.org/abs/1901.04866). orig. Hinton & van Camp 1993; bridges VAE/diffusion ELBO → real compression; cf. [[papers-vae]] [[papers-diffusion]]
- [ ] TODO <https://www.adaptiveagents.org/_media/universal-ai-as-imitation.pdf>

---

## [2023] [jveness] [Language Modeling is Compression](https://arxiv.org/abs/2309.10668)

- **Date**: 2026-07-07
- **Code**: <https://github.com/google-deepmind/language_modeling_is_compression>

---

- **Abstract**:
  - > **It has long been established that predictive models can be transformed into lossless compressors and vice versa**. Incidentally, in recent years, the machine learning community has focused on training increasingly large and powerful self-supervised (language) models. Since these large language models exhibit impressive predictive capabilities, they are well-positioned to be strong compressors. In this work, we advocate for viewing the prediction problem through the lens of compression and evaluate the compression capabilities of large (foundation) models. We show that **large language models are powerful general-purpose predictors and that the compression viewpoint provides novel insights into scaling laws, tokenization, and in-context learning**. For example, Chinchilla 70B, while trained primarily on text, compresses ImageNet patches to 43.4% and LibriSpeech samples to 16.4% of their raw size, beating domain-specific compressors like PNG (58.5%) or FLAC (30.3%), respectively. Finally, we show that the **prediction-compression equivalence allows us to use any compressor (like gzip) to build a conditional generative model**.
- **Intro**:
  - The Shannon foundation — likelihood maximization ≡ code-length minimization:
    - > Information theory and machine learning are inextricably linked and have even been referred to as "two sides of the same coin" (MacKay, 2003). One particularly elegant connection is the essential equivalence between probabilistic models of data and lossless compression. The source coding theorem (Shannon, 1948) is the fundamental theorem describing this idea, i.e., **the expected message length in bits of an optimal entropy encoder is equal to the negative log2-likelihood of the statistical model**. In other words, **maximizing the log2-likelihood (of the data) is equivalent to minimizing the number of bits required per message**. Indeed, lossless compression with a probabilistic model can be achieved in a variety of different ways, including Huffman coding (Huffman, 1952), arithmetic coding (Pasco, 1977; Rissanen, 1976), and asymmetric numeral systems (Duda, 2009).
  - Why arithmetic coding: the coder is already optimal, so compression quality = model quality:
    - > **Arithmetic coding, in particular, is known to be optimal in terms of coding length, meaning that the overall compression performance depends on the capabilities of the probabilistic model** (see Fig. 1 for an overview of arithmetic coding)
  - Online vs offline — the paper studies the *offline* side (NNCP/Bellard is the online side); frozen weights ⇒ all per-file adaptation happens in-context:
    - > **In the online setting, a pseudo-randomly initialized model is directly trained on the stream of data that is to be compressed, while the offline setting, which we consider in our work, trains the model on an external dataset before employing it to compress a (potentially different) data stream. Consequently, offline compression is performed in-context, with a fixed set of model parameters**
  - Context length is the binding constraint of offline compression (and evaporates at chunk boundaries — 2048-byte chunks are compressed independently):
    - > **The context length is a key limiting factor in offline compression, as it dictates the maximum number of bytes a model can compress at a time.** Transformers can only compress a few kilobytes (each "token" being coded with 2 or 3 bytes), while requiring a lot of compute. Correspondingly, many challenging predictive tasks (e.g., algorithmic reasoning or long-term memory) require long contexts (Delétang et al., 2023), and thus extending these models' context lengths is a key challenge which is gaining increased attention (Zaheer et al., 2020; Guo et al., 2022; Bulatov et al., 2023). **The in-context compression view provides insights into the failure modes of current foundation models.**
  - Scaling laws with the L(M) twist — the compression view is the log-loss view *plus model-size accounting*:
    - > we shed new light on scaling laws (Kaplan et al., 2020), showing that they also hold true for compression but that measuring the adjusted compression rates instead of the log loss adds a twist: **Scaling beyond a certain point will deteriorate the compression performance since the model parameters need to be accounted for in the compressed output.**
  - The thesis line:
    - > we advocate for framing (self-supervised) prediction through the lens of compression as it encompasses generalization: **a model that compresses well generalizes well** (Hutter, 2006).
  - Tokenization = pre-compression (contribution bullet; detailed in §3.6):
    - > We demonstrate that **tokenization, which can be viewed as a pre-compression, does, in general, not improve compression performance, but allows models to increase the information content in their context** and is thus generally employed to improve prediction performance.
- **Background (§2)**:
  - **Notation / setup**:
    - Data is a finite sequence $x_{1:n} := x_1x_2\ldots x_n \in \mathcal{X}^n$ over a finite alphabet $\mathcal{X}$. $x_{<j}=x_{1:j-1}$ is the prefix before symbol $j$; $\epsilon$ is the empty string; $sr$ denotes concatenation.
  - **Coding distributions** — a probabilistic model over the whole sequence tree:
    - $\rho$ is a sequence of probability mass functions satisfying the prefix-consistency condition
      $$\rho_n(x_{1:n})=\sum_{y\in\mathcal{X}}\rho_{n+1}(x_{1:n}y),\qquad \rho_0(\epsilon)=1.$$
      The probability assigned to a prefix must equal the total probability of all its one-symbol continuations: probability mass is neither created nor destroyed as the tree branches.
    - This makes the next-symbol conditional well-defined:
      $$\rho(x_n\mid x_{<n})=\frac{\rho(x_{1:n})}{\rho(x_{<n})}.$$
      Repeatedly applying it gives the chain rules
      $$\rho(x_{1:n})=\prod_{i=1}^n\rho(x_i\mid x_{<i}),\qquad
      \rho(x_{j:k}\mid x_{<j})=\prod_{i=j}^k\rho(x_i\mid x_{<i}).$$
      For an autoregressive LM, these are exactly its successive next-token predictions.
  - **Lossless compression**:
    - A binary source code $c:\mathcal{X}^*\to\{0,1\}^*$ maps every possible input sequence $x$ to a bitstring $c(x)$ of integer length $\ell_c(x)$. Lossless means the mapping is uniquely decodable: the original input is recoverable exactly. (Unique decodability is also what forces the Shannon bound below, via Kraft–McMillan: any uniquely decodable code satisfies $\sum_x 2^{-\ell_c(x)}\le 1$.)
    - The goal is to minimize expected length $L=\mathbb{E}_{x\sim\rho}[\ell_c(x)]$: frequent sequences receive short descriptions and rare sequences long ones.
    - Shannon's source-coding lower bound is
      $$L\ge H(\rho),\qquad H(\rho)=\mathbb{E}_{x\sim\rho}[-\log_2\rho(x)].$$
      This is an *expected* lower bound. The per-sequence ideal $-\log_2\rho(x)$ is real-valued, while a physical codeword has an integer number of bits.
  - **Arithmetic Coding**:
    - > Arithmetic Coding Given a coding distribution ρ and a sequence x1:n, arithmetic coding (Pasco, 1977; Rissanen, 1976) constructs a code with almost optimal length. It directly connects coding and compression with prediction and modeling: compressing well means modeling well in a logloss sense and vice-versa. Assuming infinite precision for the arithmetic operations involved, the arithmetic code has length −⌈log ρ(x1:n)⌉ + 1 bits, whereas the optimal code length is − log ρ(x1:n) bits. A practical implementation that is subject to B bit precision adds further O(n2−B) bits (Howard & Vitter, 1991), which is negligible for 32- or 64-bit arithmetic. In the following we consider infinite precision arithmetic coders and refer to Witten et al. (1987) for the finite-precision implementation.
  - **Why arithmetic coding turns prediction loss into bits**:
    - Begin with $I_0=[0,1)$. At step $k$, partition the current interval $I_{k-1}=[l_{k-1},u_{k-1})$ into one subinterval per possible next symbol $y$, with width
      $$|I_k(y)|=|I_{k-1}|\,\rho(y\mid x_{<k}).$$
      Keep the subinterval for the actual symbol $x_k$ and repeat. Therefore
      $$|I_n|=\prod_{i=1}^n\rho(x_i\mid x_{<i})=\rho(x_{1:n}).$$
    - An interval of width $w$ takes about $\log_2(1/w)=-\log_2w$ bits to identify. Hence
      $$-\log_2\rho(x_{1:n})
      =-\log_2\prod_{i=1}^n\rho(x_i\mid x_{<i})
      =\sum_{i=1}^n-\log_2\rho(x_i\mid x_{<i}).$$
      The right-hand side is cumulative next-token log-loss in bits; arithmetic coding is the mechanical bridge from that sum to an actual bitstream.
  - **Arithmetic encoder — what the emitted bits mean**:
    - Under Fig. 1's convention, an $\ell$-bit prefix names a rigid *dyadic interval*
      $$D_{k,\ell}=\left[k2^{-\ell},(k+1)2^{-\ell}\right)$$
      of width $2^{-\ell}$. For example, `010` names $[0.010_2,0.011_2)=[0.25,0.375)$. These cells lie on a fixed binary grid; they cannot be slid to an arbitrary position.
    - The selected dyadic cell must be fully contained in the final arithmetic interval $I_n$. Mere overlap is ambiguous: some binary continuations would lie outside $I_n$ and could decode to another sequence.
  - **The constant per-stream termination/alignment cost**:
    - Width alone is insufficient because of grid alignment. If $I=[0.30,0.55)$ has width $w=1/4$, no width-$1/4$ two-bit cell fits: $[0.25,0.50)$ protrudes left and $[0.50,0.75)$ protrudes right.
    - A cell width $d\le w/2$ is always sufficient. From $I$'s left edge $a$, the next grid boundary is at most $d$ away; the complete cell after it ends by $a+2d\le a+w$, so it lies inside $I$.
    - Since $d=2^{-\ell}$, the worst-case guarantee is
      $$2^{-\ell}\le\frac{w}{2}
      \quad\Longrightarrow\quad
      \ell\ge-\log_2w+1,$$
      hence a code of at most
      $$\boxed{\ell\le\left\lceil-\log_2\rho(x_{1:n})\right\rceil+1
      <-\log_2\rho(x_{1:n})+2}$$
      bits is guaranteed. The ceiling pays for whole bits; the extra bit halves the dyadic cell to survive worst-case grid alignment. This is an upper bound from width alone, not necessarily the shortest code for a favorably aligned interval.
    - **Fig. 1 check**: `AIXI` ends at $I=[0.322,0.341)$, so $w=0.019$ and $-\log_2w\approx5.72$. The six-bit cell `010101` is $[0.328125,0.34375)$ and protrudes past $I$; the seven-bit cell `0101010` is $[0.328125,0.3359375)\subset I$. Thus $\lceil5.72\rceil+1=7$ bits, matching the figure.
    - **Apparent formula typo/inconsistency**: the printed $-\lceil\log\rho(x)\rceil+1$ gives 6 bits for the figure's $\rho(x)=0.019$, whereas Fig. 1's full-cell-containment convention requires 7. The consistent worst-case bound is $\lceil-\log_2\rho(x)\rceil+1$, i.e. less than two bits above ideal log-loss. Termination conventions can shift the exact constant, but not the $O(1)$ conclusion.
    - This cost is paid **once per independently terminated arithmetic-coding stream**, because fractional symbol costs accumulate before the final interval is rounded to a legal binary prefix. One 1GB stream pays it once; 1,000 independently reset chunks can pay it 1,000 times. This paper's own experiments do exactly that (§3.1: independent 2048-byte chunks), and the cost is still nothing: ~2 bits per 2048-byte chunk ≈ 0.001 bpb. Contrast Huffman coding, whose integer rounding can waste up to nearly one bit *per symbol*.
  - **Finite-precision cost — separate from termination**:
    - The mathematical encoder divides real intervals exactly. A practical coder represents bounds and cumulative probabilities on a $B$-bit integer grid, so each subdivision must be rounded. With a deliberately tiny $B=3$, a desired probability $0.3$ corresponds to $2.4$ of 8 units and might become $2/8=0.25$, changing the step cost from $-\log_2 0.3\approx1.737$ to $-\log_2 0.25=2$ bits. At $B=32$, the grid has $2^{32}$ units, so an analogous one-unit error is on the scale of $2^{-32}$.
    - Tiny rounding errors can occur at every one of the $n$ coding steps, giving the paper's additional $O(n2^{-B})$ bits. This is a scaling statement with hidden constants, not an exact equality; at 32 or 64 bits it is negligible for ordinary stream sizes.
    - The interval does not eventually become too small for the register: whenever the lower and upper bounds acquire fixed common leading bits, the coder emits those bits and **renormalizes** the unresolved suffix back across the integer range. Witten–Neal–Cleary (1987) supplies the bounded-integer algorithm, including the middle-half case and deferred/pending bits.
    - Clean distinction: **termination/alignment** is $O(1)$ once per stream; **finite precision** is a minute $O(2^{-B})$ rounding loss per step, accumulating to $O(n2^{-B})$.
  - **Arithmetic decoder**:
    - Given the encoded value/prefix and the same $\rho$, start again at $I_0=[0,1)$. At step $k$, find which model-defined subinterval contains the encoded value; its label is $x_k$. Restrict to that interval and repeat.
    - Encoder and decoder must produce exactly the same conditional probabilities, symbol ordering, quantization, and updates. No symbols need to be transmitted separately: the shared model plus the bits reconstructs them, which is why the procedure is lossless.
  - **Likelihood maximization = compression-rate minimization**:
    - > Likelihood Maximization In practice, the source distribution $\rho$ is usually unknown and is instead estimated with a parametric probabilistic model $\hat\rho$. Thus, instead of achieving code length $-\sum_{i=1}^n \log_2 \rho(x_i \mid x_{<i})$ for the sequence $x_{1:n}$, we obtain the suboptimal length $-\sum_{i=1}^n \log_2 \hat\rho(x_i \mid x_{<i})$. As a result, the expected (suboptimal) number of bits is the cross-entropy:
      > $$H(\rho,\hat\rho):=\mathbb{E}_{x\sim\rho}\left[\sum_{i=1}^n-\log_2\hat\rho(x_i\mid x_{<i})\right]. \tag{2}$$
      > Thus, we can minimize the expected length of the encoded data stream with symbols distributed according to $\rho$ by minimizing the cross-entropy with respect to some $\hat\rho$, which is equivalent to likelihood maximization (MacKay, 2003). However, Eq. (2) is exactly the same objective used to train current foundation models, i.e., the log-loss. Thus, minimizing the log-loss is equivalent to minimizing the compression rate of that model used as a lossless compressor with arithmetic coding, i.e., current language model training protocols use a maximum-compression objective.
    - The true source $\rho$ is unknown, so use a model $\hat\rho$. Encoding data drawn from $\rho$ with $\hat\rho$ costs
      $$H(\rho,\hat\rho)
      =\mathbb{E}_{x\sim\rho}\left[\sum_{i=1}^n-\log_2\hat\rho(x_i\mid x_{<i})\right].$$
    - **Why cross-entropy = entropy + KL** — let $x$ denote a complete sequence and expand the expectation:
      $$H(\rho,\hat\rho)=-\sum_x\rho(x)\log_2\hat\rho(x).$$
      Add and subtract the true log-probability $\log_2\rho(x)$:
      $$\begin{aligned}
      H(\rho,\hat\rho)
      &=-\sum_x\rho(x)\log_2\rho(x)
      +\sum_x\rho(x)\log_2\frac{\rho(x)}{\hat\rho(x)}\\
      &=H(\rho)+D_{\mathrm{KL}}(\rho\Vert\hat\rho).
      \end{aligned}$$
    - **Compression meaning**: $H(\rho)$ is the unavoidable expected cost when the true source is known; $D_{\mathrm{KL}}(\rho\Vert\hat\rho)$ is the expected number of **extra bits caused by using the imperfect model** $\hat\rho$. KL is nonnegative and is zero when the model matches the source. Because $H(\rho)$ does not depend on $\hat\rho$, minimizing cross-entropy is exactly minimizing this excess-bit penalty.
    - For an autoregressive model, the sequence-level probability ratio splits across tokens:
      $$\log_2\frac{\rho(x_{1:n})}{\hat\rho(x_{1:n})}
      =\sum_{i=1}^n\log_2\frac{\rho(x_i\mid x_{<i})}{\hat\rho(x_i\mid x_{<i})}.$$
      Thus the total KL/compression penalty accumulates from the model's next-token probability errors at each context.
    - Minimizing cross-entropy/log-loss in $\hat\rho$ is therefore the same optimization as minimizing its expected arithmetic-coded length. Standard next-token maximum-likelihood training is already a **maximum-compression objective**, even if no coder is run.
  - **Compression-Based Sequence Prediction — run the equivalence backwards**:
    - > Compression-Based Sequence Prediction Analogous to how a predictive distribution can be used for lossless compression via arithmetic coding (described above), any compressor can be employed for sequence prediction (Frank et al., 2000). The main idea is to define $\rho(x_{1:n})$ as the coding distribution $2^{-\ell_c(\cdot)}$, where $\ell_c(x_{1:n})$ is the length of sequence $x_{1:n}$ when encoded with compressor $c$ (e.g., gzip). We thus recover the conditional distribution $\rho(x_i \mid x_{<i})$ by computing $2^{\ell_c(x_{<i})-\ell_c(x_{<i}x_i)}$, for all $x_i$.
    - If a compressor $c$ gives sequence $x$ a code length $\ell_c(x)$, associate short codes with high probability via $\rho(x)\propto2^{-\ell_c(x)}$. The paper writes the induced conditional as
      $$\rho(x_i\mid x_{<i})
      =2^{\ell_c(x_{<i})-\ell_c(x_{<i}x_i)}.$$
      The exponent is the *incremental compressed length* of appending candidate $x_i$: candidates that add fewer bits receive higher predictive weight. In practice, normalize across candidates if the compressor-derived scores are not already a proper probability mass function.
    - This is how a non-generative-looking compressor such as gzip can be used as a next-symbol predictor/generative model: tentatively append each candidate, measure the change in compressed length, convert bit costs to weights with $2^{-\Delta\ell}$, then sample or choose from the resulting distribution.
  - **Universal Coding**:
    - > Universal Coding Above we discussed optimal (arithmetic) coding with respect to data sampled from a fixed distribution $\rho$. In contrast, universal (optimal) source coding with respect to all computable sampling distributions can, in theory, be achieved by choosing $\ell_c(x_{1:n})$ as the Kolmogorov complexity of $x_{1:n}$ (Kolmogorov, 1998; Li & Vitányi, 2019). For this choice, the conditional distribution described above is universally optimal over $x_{<i}$, recovering the Solomonoff predictor (Solomonoff, 1964a;b; Rathmanner & Hutter, 2011). The Solomonoff predictor is a Bayesian mixture of all predictors that can be programmed in a chosen Turing-complete programming language. More precisely, for a predictor $q$ of program-length $\ell_c(q)$ bits, the Solomonoff predictor assigns a prior weight of $2^{-\ell_c(q)}$ to predictor $q$. That is, if $Q$ is the set of all predictors that can be programmed and computed, the Solomonoff predictor assigns probability $S(x_{1:n}) = \sum_{q\in Q} 2^{-\ell_c(q)}q(x_{1:n})$ to a sequence $x_{1:n}$. Therefore, $S(x_{1:n}) \ge 2^{-\ell_c(q)}q(x_{1:n})$ for all $q \in Q$, and thus $-\log_2 S(x_{1:n}) \le -\log_2 q(x_{1:n}) + \ell_c(q)$. Observe that $\ell_c(q)$ is a constant of $q$ that is independent of the sequence length. Therefore, compressing optimally is equivalent to predicting optimally and vice versa (Hutter, 2005).
    - **Fixed-distribution vs. universal coding**: ordinary arithmetic coding is optimal *relative to a supplied $\rho$*. It does not tell us which $\rho$ to use when the source is unknown. Universal coding asks for one predictor/code that competes with every distribution in a model class; here the class is all computable predictors.
    - **Kolmogorov complexity** $K(x)$ is the length of the shortest program for a fixed universal Turing machine that outputs $x$. Using $K(x)$ as codelength embodies the strongest possible version of “regular data gets a short description”: whichever computable pattern generated $x$, its shortest program supplies the code. The choice of universal machine changes $K$ only by an additive constant, but $K$ is uncomputable (a consequence of the halting problem), so this is an ideal limit rather than an implementable compressor.
    - **Solomonoff induction is Bayesian model averaging over programs**:
      $$S(x_{1:n})=\sum_{q\in Q}\underbrace{2^{-\ell_c(q)}}_{\text{simplicity prior}}\underbrace{q(x_{1:n})}_{\text{predictor likelihood}}.$$
      Short programs receive exponentially more prior weight: every extra program bit halves the prior. With prefix-free program descriptions, Kraft's inequality ensures the weights fit within total mass 1 (more precisely, Solomonoff induction is generally a semimeasure).
    - **Why the dominance inequality matters**: the mixture contains the nonnegative contribution of every individual $q$, so for any chosen computable predictor,
      $$S(x_{1:n})\ge 2^{-\ell_c(q)}q(x_{1:n}).$$
      Applying $-\log_2$ reverses the inequality and turns the product into a sum:
      $$\boxed{-\log_2S(x_{1:n})\le -\log_2q(x_{1:n})+\ell_c(q).}$$
      The left side is Solomonoff's cumulative log-loss/codelength. The right side is a **two-part description**: transmit the predictor in $\ell_c(q)$ bits, then encode the data under it in $-\log_2q(x_{1:n})$ bits.
    - **The universal guarantee**: relative to any computable $q$, Solomonoff pays at most the one-time model-description penalty $\ell_c(q)$—a constant independent of $n$. Per symbol,
      $$\frac{-\log_2S(x_{1:n})}{n}
      \le
      \frac{-\log_2q(x_{1:n})}{n}+\frac{\ell_c(q)}{n},$$
      and $\ell_c(q)/n\to0$. Thus the universal predictor asymptotically matches the average log-loss/compression rate of whichever computable predictor is best for the stream, without being told that predictor in advance.
    - **Why optimal compression = optimal prediction again**: $S$ induces next-symbol probabilities $S(x_i\mid x_{<i})=S(x_{1:i})/S(x_{<i})$, and their cumulative log-loss telescopes:
      $$\sum_{i=1}^n-\log_2S(x_i\mid x_{<i})=-\log_2S(x_{1:n}).$$
      A universal codelength therefore gives a universal sequential predictor, while a universal predictor gives a universal arithmetic code. This is the theoretical endpoint; practical universal compressors such as Lempel–Ziv or CTW make computable guarantees only for restricted source classes.
  - **Connection**: the Solomonoff dominance bound $-\log_2 S(x)\le-\log_2 q(x)+\ell_c(q)$ *is* the two-part code $L(M)+L(X\mid M)$ — the same accounting as this paper's "adjusted compression rate" (§3.2) and the $L(M)$ caveats elsewhere in this file; the universal-coding paragraph is the theory behind the metric. The Solomonoff mixture is the ideal-row anchor for the compression=prediction thesis (cf. [[papers-aixi]]). Witten–Neal–Cleary 1987 is the reference implementation for a finite-precision coder (renormalization, carry/pending bits); cf. [[nncp]].

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
- **One-liner**: **ActivePTW is Thompson sampling with uncertainty about when the world last changed.** It uses compression to maintain a soft belief over which portion of the past still belongs to the current regime, samples one such history, and acts from it.
- **Problem being solved**:
  - A Bernoulli bandit has several arms, each returning success/failure with an unknown probability. Here those probabilities occasionally jump at **unknown change-points**. Between jumps the world is stationary; the agent knows neither when a jump occurred nor the new probabilities.
  - A stationary learner remembers too much and is poisoned by stale observations. A sliding-window learner forgets, but requires a hand-chosen window size. A hard-reset learner must decide exactly when to restart. **ActivePTW avoids committing to one memory length or one restart schedule.**
- **The compression framing**:
  - Describe the entire agent-environment interaction as a code. Its excess length relative to an environment-specific agent splits naturally into two costs:
    - **environment redundancy** — extra bits because the agent predicts the world's observations poorly;
    - **policy redundancy** — extra bits because its actions differ from the policy desired for that environment.
  - A universal agent tries to make both costs sublinear for every environment in its chosen class. “Universal” is therefore **relative to a model class** — here piecewise-stationary Bernoulli bandits — not “universally intelligent.”
  - The paper's useful reframing is that a good compressor should not merely learn the environment; its predictive distribution can also be turned into a distribution over actions. But this requires two pieces that compression alone does not supply: a causal separation between actions and observations, and a desired **reference policy** for every candidate environment.
- **The causality wall / self-delusion problem**:
  - Observations come from the world and are evidence about it. Actions come from the agent and are **interventions**. The agent must remember which action preceded an observation — a reward after pulling arm 2 tells us about arm 2 — but the fact that it chose arm 2 cannot itself count as evidence that the world favors arm 2.
  - A naive joint sequence model can get this wrong: “I acted as though hypothesis X were true, therefore my action confirms X.” Its belief then tracks its own outputs instead of the environment.
  - The paper's $\Vert$ notation marks this distinction: actions are given to the percept predictor, but the posterior over environments is updated only by percepts. This is a general design constraint for learned models of interaction, not a bandit-specific trick.
- **Where the goal enters — an important qualification to “control from compression”**:
  - For each possible environment $\rho$, the construction assumes a **reference policy** $\pi_\rho$: the behavior we would want if $\rho$ were known to be true. In a bandit this is easy — play the arm with the highest success probability.
  - The universal policy averages these reference policies according to the current posterior over environments. Equivalently: sample a plausible environment, then behave as though it were true. With stationary Beta-Bernoulli bandits and greedy reference policies, this is exactly **Thompson sampling**.
  - So compression supplies the beliefs and the uncertainty-aware hedge; the reference policy supplies the **goal and what to do with those beliefs**. ActivePTW relocates rather than eliminates the control objective. This becomes a major obstacle beyond bandits, where “the best action if the environment were known” may itself require expensive planning.
- **PTW intuition — a soft, learned memory length**:
  - Imagine many explanations of the past running in parallel: “nothing has changed,” “the current regime began recently,” “it began much earlier,” etc. Each explanation starts fresh per-arm success/failure counts at its proposed segment boundary.
  - Each segment is scored by how well its per-arm **KT estimator** compresses the rewards observed within it. KT is just a smoothed Beta-Bernoulli predictor: an arm begins uncertain and becomes confident as successes and failures arrive.
  - Simpler change histories receive more prior weight. A new segment must earn its extra complexity by compressing the recent data enough better than the old, longer segment. This is the MDL trade-off in concrete form: **pay bits for declaring a change; recover those bits if the post-change data really behave differently.**
  - PTW restricts the candidate histories to a binary tree of nested time segments. That seems restrictive, but any arbitrary segmentation can be covered with only a logarithmic increase in the number of segments. The payoff is large: only about $\log T$ candidate segments are active at any moment rather than an exploding number of complete change histories.
  - Operationally this behaves like **soft resetting**. In a stable world, posterior mass moves toward one long segment and nearly all history is reused. After a change, shorter segments explain the recent data better, gain posterior mass, and old observations automatically stop influencing actions. No explicit change detector fires and no data are irrevocably deleted.
- **ActivePTW in plain English** — each step:
  1. Sample a plausible active segment: “How far back does the current regime extend?”
  2. Using only observations inside that segment, sample a success probability for every arm from its Beta posterior.
  3. Apply the reference policy for that sampled environment — normally play the sampled-best arm.
  4. Observe the reward and update all relevant segment statistics.
  - Compared with ordinary Thompson sampling, the only conceptual addition is **sampling the current segment before sampling the arm values**.
  - The tree shares computations across the huge mixture of possible histories: model updates cost $O(\log T)$ per step, memory is $O(|\mathcal A|\log T)$, and sampling an action costs $O(\log T+|\mathcal A|)$.
- **Why prediction/compression is not enough — exploration controls what can be learned**:
  - The paper constructs the decisive failure case: the old best arm keeps exactly the same reward probability after a change, while a previously bad arm silently becomes excellent. Continuing to pull the old arm produces perfectly unsurprising data, so neither a compressor nor a change detector can discover the improvement elsewhere.
  - The forced-exploration reference policy occasionally pulls a random arm, with exploration decaying roughly as the inverse square root of the inferred segment length. The plots call this more cautious variant **ParanoidPTW**; the greedier variant is **ActivePTW**.
  - This exposes a fundamental difference between passive and active compression: **the agent chooses which data enter its compressor**. A perfect predictor of collected data can still support a bad policy when the policy never collects the discriminating data.
  - Forced exploration is necessary for the paper's concentration argument and its hard example, but it is a hand-designed addition rather than something derived from source coding. It helps after hidden changes and costs reward in genuinely stationary settings.
- **Empirical picture**:
  - In million-step bandits with geometrically distributed change-points, the PTW variants usually beat ordinary Thompson sampling, Sliding-Window UCB, and MASTER when regimes last long enough to be learnable. Sliding-Window UCB is given the unusually favorable oracle setting $W=1/p$ — advance knowledge of the mean regime length — and ActivePTW still often wins.
  - There is no unconditional victory: with many arms and very frequent changes, there is too little time to rediscover all arm values, and UCB can win. The problem itself becomes increasingly hostile as “number of arms × number of regimes” grows.
  - In a stationary environment, greedy ActivePTW becomes almost indistinguishable from Thompson sampling. Its posterior favors the single long segment, so the mechanism for change adds little empirical cost when nothing changes.
  - In the deliberately hidden-change example, greedy ActivePTW fails alongside Thompson sampling because the previously best arm remains unchanged. The forced-exploration variant detects the newly good arm and substantially reduces regret. This is the clearest experiment in the paper because it isolates the information-gathering issue rather than just comparing leaderboards.
- **What is actually proved**:
  - The PTW-KT **environment model** has a coding-redundancy guarantee relative to any piecewise-stationary Bernoulli bandit, and the per-arm Beta posterior concentrates when an arm is sampled often enough.
  - The paper does **not** prove a complete regret bound for the full ActivePTW agent. The missing step is showing that interaction causes the posterior over active segments to concentrate quickly enough on useful change histories. The authors explicitly leave this for future work.
  - The experiments use a modified PTW prior that increasingly favors simpler segmentations as the number of arms grows. Intuition: with more arms, every fresh segment is more expensive to learn, so the evidence threshold for declaring a new regime should be higher.
- **What ActivePTW adds — and what it inherits**:
  - **Inherited**: KT/Beta-Bernoulli prediction, the PTW mixture over tree-structured partitions, Bayesian model averaging, the Bayesian Control Rule, and its stationary-bandit equivalence to Thompson sampling.
  - **New synthesis**: extend passive PTW to action-conditioned interaction data, compute a posterior over the currently active segment efficiently, and use that posterior inside the Bayesian Control Rule to obtain a practical non-stationary bandit policy.
  - The algorithmic novelty is therefore not a new compressor or a new general planning method. It is the **efficient coupling of change-point uncertainty to action selection**, plus an empirical demonstration that the result is competitive and reduces to Thompson sampling in the stationary limit.
  - The conceptual contribution is a clean worked example of how universal coding must change when the data stream contains the learner's own actions: separate environment prediction from policy generation or invite self-delusion.
- **Relation to CNC / Compress and Control above**:
  - **CNC** uses compressors to estimate $Q(s,a)$: compare how well return-conditioned models explain a state, recover a value by Bayes inversion, then use an external action-selection rule such as $\epsilon$-greedy. Its main contribution is **compression as policy evaluation** in stateful, finite-horizon problems.
  - **ActivePTW** has no state or delayed credit assignment. It compresses the bandit's observed rewards, maintains uncertainty over non-stationary regimes, and generates actions by mixing supplied reference policies. Its contribution is **adaptive environment inference plus a universal policy** for the bandit class.
  - Thus ActivePTW is not simply “CNC made non-stationary.” They solve complementary halves. CNC has a way to turn state/return density estimates into values but no native forgetting; ActivePTW has compression-native forgetting and posterior sampling but assumes the environment-specific policy is easy to obtain.
  - A natural synthesis would put PTW-style segmentation around CNC's density models, or use CNC-like value estimation to construct reference policies in richer environments. That would still leave exploration and long-horizon planning unresolved.
- **Other connections**:
  - **CTW → PTW → ActivePTW**: CTW mixes predictive contexts; PTW mixes temporal restart structures; ActivePTW makes the selected structure influence actions. It is the transition from passive adaptive coding to **active adaptive coding**.
  - **Thompson sampling**: ActivePTW is best remembered as hierarchical Thompson sampling — first sample the regime's age, then sample its arm parameters. In the stationary limit the first draw becomes irrelevant and ordinary Thompson sampling remains.
  - **MASTER / restart methods**: rather than choose or randomly trigger one restart schedule, ActivePTW Bayesian-averages many schedules and lets compression evidence weight them.
  - **[[nncp]] / online compression**: in a passive stream the learner receives whatever comes next; in interaction its current model changes the future training distribution through its actions. This adds the causal-wall and exploration problems that passive online compression does not face.
  - **Cybernetics**: the paper explicitly connects the view to Wiener — an adaptive process coupled to an input/output channel, with agency framed in terms of entropy and information flow rather than beginning from reward maximization.
- **Implications for adaptive / continual agents**:
  - PTW is a useful model of **adaptive memory**: do not choose globally between remembering and forgetting; maintain hypotheses at several timescales and let predictive evidence decide which history is relevant now.
  - Non-stationarity can be represented inside the model class rather than handled by an external reset heuristic. This is attractive for continual learning because the posterior can return to long memory when the world is stable and shorten memory only when the data pay for the additional change-point complexity.
  - Yet the strongest negative lesson is equally important: **better compression of experienced data does not imply better actions**. The learner also needs a goal-bearing reference policy and actions that expose the information required to distinguish worlds.
  - ActivePTW is therefore evidence for a narrower claim than “compression is sufficient for agency”: universal coding can provide principled belief updating, uncertainty, and adaptive forgetting, and can induce a strong controller when the environment-specific action problem is trivial. General control still contains the hard problems of objectives, exploration, state, credit assignment, and planning.
- **Limits / open directions**:
  - Bernoulli bandits only: no state, representation learning, delayed reward, or long-horizon credit assignment.
  - Abrupt piecewise stationarity: gradual drift and recurring latent regimes are not modeled explicitly.
  - A reference policy must be available for every hypothesized environment; trivial for bandits, potentially as hard as solving the original problem for MDPs.
  - The Bayesian Control Rule mixes reference policies one step at a time and may not seek enough information over long horizons. More general Thompson-sampling agents and BayesEXP address this with substantially more expensive multi-step reasoning.
  - Full control-regret theory remains open, and the forced-exploration schedule is supplied rather than derived.

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
