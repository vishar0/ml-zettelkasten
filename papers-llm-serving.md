# LLM Serving

- **Created**: 2026-07-26
- **Last Updated**: 2026-08-17
- **Status**: `In Progress`
- **Description**: Systems and inference-time techniques for efficiently and reliably generating outputs from deployed language models.
- **Related**:
  - [[papers-ml-fundamentals]] — Core architectures, training methods, and language-model concepts.
  - [[papers-foundation-models]] — Foundation-model families and their technical reports.
  - [[book-huggingface-ultrascale-llm-training]] — Large-scale training systems with techniques that also inform inference infrastructure.
  - [[ml-hardware]] — Hardware constraints and hardware-software co-design.

---

## Decoding

- [x] [2023] Speculative Decoding: Fast Inference from Transformers via Speculative Decoding - [paper](https://arxiv.org/abs/2211.17192)
- [x] [2026] Speculative Speculative Decoding - [paper](https://arxiv.org/abs/2603.03251), [code](https://github.com/tanishqkumar/ssd)
- [ ] [2023] Constrained Decoding: Efficient Guided Generation for Large Language Models - [paper](https://arxiv.org/abs/2307.09702), [code](https://github.com/dottxt-ai/outlines)

## Kernels & Model Execution

- [ ] [2022] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness - [paper](https://arxiv.org/abs/2205.14135), [code](https://github.com/Dao-AILab/flash-attention)

## Serving Systems

- [ ] [2023] vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention - [paper](https://arxiv.org/abs/2309.06180), [code](https://github.com/vllm-project/vllm)
- [ ] [2024] SGLang: Efficient Execution of Structured Language Model Programs - [paper](https://arxiv.org/abs/2312.07104), [code](https://github.com/sgl-project/sglang)

---

## [2023] [Speculative Decoding: Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)

- **Date**: 2026-08-17

---

- **One-liner**: Use a cheap approximation model to propose several tokens, verify those guesses concurrently with the target model, and preserve the target model's output distribution exactly.
- **Motivation (§1)**:
  - Large autoregressive models decode serially: generating $K$ tokens ordinarily requires $K$ runs of the model.
  - Existing efficiency methods often change the architecture or training procedure, require retraining, or change the output distribution. Speculative decoding instead exploits otherwise available compute to increase concurrency, especially when inference is limited by memory bandwidth or communication rather than arithmetic.
  - > Speculative execution (Burton, 1985; Hennessy & Patterson, 2012) is an optimization technique, common in processors, where a task is performed in parallel to verifying if it’s actually needed - the payoff being increased concurrency. A well-known example of speculative execution is branch prediction. For speculative execution to be effective, we need an efficient mechanism to suggest tasks to execute that are likely to be needed.
  - > In this work, we generalize speculative execution to the stochastic setting - where a task might be needed with some probability. Applying this to decoding from autoregressive models like Transformers, we sample generations from more efficient approximation models as speculative prefixes for the slower target models. With a novel sampling method, speculative sampling, we maximize the probability of these speculative tasks to be accepted, while guaranteeing that the outputs from our system have the same distribution as those from the target model alone.
- **Contributions**:
  - > To summarize, our main contributions are: (1) A generalization of speculative execution to the stochastic setting, with a novel sampling method we call speculative sampling, and (2) A decoding mechanism we call speculative decoding that can accelerate decoding from autoregressive models, without any change to the model architectures, training regimes and output distributions.
- **Overview (§2.1)**:
  - $M_p$ is the target/verifier model, with next-token distribution $p(x_t \mid x_{<t})$. $M_q$ is the cheaper approximation/draft model, with distribution $q(x_t \mid x_{<t})$.
  - One speculative-decoding step:
    - (1) Run $M_q$ autoregressively to generate $\gamma \in \mathbb{Z}^+$ candidate tokens.
    - (2) Run $M_p$ once to evaluate the guesses and their probabilities in parallel, accepting the longest prefix that preserves the distribution of $M_p$.
    - (3) Sample one additional token: from an adjusted distribution to correct the first rejected guess, or directly from $M_p$ after the final guess if all $\gamma$ guesses are accepted.
  - A target-model run therefore produces at least one token and at most $\gamma+1$ tokens. In the worst case it is no more serial than ordinary autoregressive decoding; when guesses are accepted, it advances several tokens per target-model run.
- **Standardized Sampling (§2.2)**:
  - Argmax, top-$k$, nucleus sampling, and temperature sampling can all be represented as ordinary sampling after transforming and normalizing the model's probability distribution.
  - For example, argmax is equivalent to zeroing every non-maximal probability and normalizing, yielding a point mass on the maximal token.
  - From this point onward, $p(x)$ and $q(x)$ mean the distributions from $M_p$ and $M_q$ *after* applying the chosen sampling transformation. This lets the speculative-sampling argument operate on probability distributions without separate cases for every decoding method.
- **Notation used below**:
  - The paper uses $p=p_{\text{verifier}}$ for the target model $M_p$ and $q=p_{\text{draft}}$ for the approximation model $M_q$.
  - All next-token distributions are conditional on the current prefix even when that conditioning is suppressed.
  - $\gamma$ is the speculative lookahead: the number of tokens proposed autoregressively by the draft model before one target-model verification.
- **Speculative sampling (§2.3)**:
  - For a draft proposal $X=x \sim p_{\text{draft}}$, accept with probability
    $$
    a(x)=\min\left(1,\frac{p_{\text{verifier}}(x)}{p_{\text{draft}}(x)}\right).
    $$
  - If $p_{\text{draft}}(x)\le p_{\text{verifier}}(x)$, every proposal of $x$ is accepted. If the draft overestimates $x$, only the fraction $p_{\text{verifier}}(x)/p_{\text{draft}}(x)$ is accepted.
  - On rejection, sample a correction from the residual distribution
    $$
    p'(x)=\operatorname{norm}\left(\max\left(0,p_{\text{verifier}}(x)-p_{\text{draft}}(x)\right)\right).
    $$
  - This is not ordinary rejection sampling: there is only one proposal, and a rejection is repaired by sampling from the verifier's missing probability mass rather than restarting the procedure.
- **Algorithm 1 — one speculative-decoding step**:
  - Draft $x_1,\ldots,x_\gamma$ autoregressively, saving the conditional draft distributions $q_i$.
  - Evaluate $p_1,\ldots,p_{\gamma+1}$ in one target-model pass. Here $p_i$ verifies $x_i$, while $p_{\gamma+1}$ predicts the token following the final draft token.
  - Draw independent $r_i\sim U(0,1)$ and accept the longest prefix for which
    $$
    r_i\le \frac{p_i(x_i)}{q_i(x_i)}.
    $$
    The missing explicit $\min(1,\cdot)$ is harmless because $r_i\le1$.
  - If the first rejection is at position $n+1$, discard all later guesses and sample the final emitted token from
    $$
    p'(x)=\operatorname{norm}\left(\max\left(0,p_{n+1}(x)-q_{n+1}(x)\right)\right).
    $$
  - If all $\gamma$ guesses are accepted, sample one extra **bonus token** directly from $p_{\gamma+1}$. This is why a successful step emits $\gamma+1$ tokens without another target-model pass: the logits after $x_\gamma$ were already produced during verification.
  - Every step returns the accepted draft prefix plus exactly one target-supplied token, so it always advances by at least one token and never uses more serial target calls than baseline decoding.
- **Correctness — probability-mass view (Appendix A.1)**:
  - Fix a prefix $x_{<t}$ and let
    $$
    X\sim p_{\text{draft}}(\cdot\mid x_{<t})
    $$
    be the random token proposed by the draft model. For each possible proposal $x$, define its **conditional acceptance probability**
    $$
    a_{x_{<t}}(x)
    :=\Pr(\text{accept}\mid X=x,x_{<t})
    =\min\left(1,
    \frac{p_{\text{verifier}}(x\mid x_{<t})}
         {p_{\text{draft}}(x\mid x_{<t})}
    \right).
    $$
    Thus $a_{x_{<t}}(x)$ answers: *if the draft happens to propose token $x$ at this prefix, what is the probability that the verifier accepts it?*
  - The probability that $x$ is both proposed and accepted is therefore
    $$
    \begin{aligned}
    &\Pr(X=x,\text{ accept}\mid x_{<t})\\
    &\quad=p_{\text{draft}}(x\mid x_{<t})a_{x_{<t}}(x)\\
    &\quad=\min\left(
       p_{\text{draft}}(x\mid x_{<t}),
       p_{\text{verifier}}(x\mid x_{<t})
    \right).
    \end{aligned}
    $$
  - Acceptance therefore retains exactly the overlap of the two distributions. Where the draft is too large, the accept/reject test trims its mass down to the verifier's mass; where the draft is too small, accepting every proposal still supplies only the draft's smaller mass.
  - Definition 3.1's $\beta_{x_{<t}}$ is the **unconditional acceptance probability at the prefix**: the probability of acceptance before knowing which token $X$ will be proposed. Averaging the conditional probability above over all possible proposals gives
    $$
    \begin{aligned}
    \beta_{x_{<t}}
    &:=\Pr(\text{accept}\mid x_{<t})\\
    &=\mathbb{E}_{X\sim p_{\text{draft}}(\cdot\mid x_{<t})}
      \left[a_{x_{<t}}(X)\right]\\
    &=\sum_x p_{\text{draft}}(x\mid x_{<t})a_{x_{<t}}(x)\\
    &=\sum_x\min\left(
       p_{\text{draft}}(x\mid x_{<t}),
       p_{\text{verifier}}(x\mid x_{<t})
    \right).
    \end{aligned}
    $$
    The sum is implicit in the paper's prose definition. The implementation never needs to compute it: it samples one $X$ and evaluates only $a_{x_{<t}}(X)$. The aggregate $\beta_{x_{<t}}$ is used for analysis. Below, the fixed-prefix conditioning is suppressed again: $a(x)$ denotes $a_{x_{<t}}(x)$ and $\beta$ denotes $\beta_{x_{<t}}$.
  - The unnormalized residual mass can be rewritten as
    $$
    \max\left(0,p_{\text{verifier}}(x)-p_{\text{draft}}(x)\right)
    =p_{\text{verifier}}(x)-\min\left(p_{\text{draft}}(x),p_{\text{verifier}}(x)\right).
    $$
  - Let $Z$ be the residual distribution's normalizer. Using the identity above and the fact that $p_{\text{verifier}}$ sums to one,
    $$
    \begin{aligned}
    Z
    &:=\sum_x\max\left(0,p_{\text{verifier}}(x)-p_{\text{draft}}(x)\right)\\
    &=\sum_x\left[p_{\text{verifier}}(x)
      -\min\left(p_{\text{draft}}(x),p_{\text{verifier}}(x)\right)\right]\\
    &=\underbrace{\sum_x p_{\text{verifier}}(x)}_{1}
      -\underbrace{\sum_x\min\left(p_{\text{draft}}(x),p_{\text{verifier}}(x)\right)}_{\beta}\\
    &=1-\beta.
    \end{aligned}
    $$
  - The rejection probability can be calculated independently by averaging $1-a(x)$ over draft proposals:
    $$
    \begin{aligned}
    \Pr(\text{reject})
    &=\sum_x p_{\text{draft}}(x)(1-a(x))\\
    &=\sum_x\left[p_{\text{draft}}(x)
      -\min\left(p_{\text{draft}}(x),p_{\text{verifier}}(x)\right)\right]\\
    &=1-\beta.
    \end{aligned}
    $$
    Therefore $Z=\Pr(\text{reject})=1-\beta$. Rejection removes mass where the draft exceeds the verifier, while the residual supplies mass where the verifier exceeds the draft. These quantities are not equal token by token, but their totals are equal because both distributions sum to one.
  - Thus the rejection-and-correction path contributes
    $$
    (1-\beta)p'(x)
    =(1-\beta)\frac{\max\left(0,p_{\text{verifier}}(x)-p_{\text{draft}}(x)\right)}{1-\beta}
    =\max\left(0,p_{\text{verifier}}(x)-p_{\text{draft}}(x)\right)
    =p_{\text{verifier}}(x)-\min\left(p_{\text{draft}}(x),p_{\text{verifier}}(x)\right).
    $$
  - Adding the accepted and corrected routes gives
    $$
    \begin{aligned}
    \Pr(\text{final token}=x)
    &=\Pr(X=x,\text{ accept})
      +\Pr(\text{reject and correction token}=x)\\
    &=p_{\text{draft}}(x)a(x)+(1-\beta)p'(x)\\
    &=\min\left(p_{\text{draft}}(x),p_{\text{verifier}}(x)\right)\\
    &\quad+\left(p_{\text{verifier}}(x)
      -\min\left(p_{\text{draft}}(x),p_{\text{verifier}}(x)\right)\right)\\
    &=p_{\text{verifier}}(x).
    \end{aligned}
    $$
    The output does not reproduce an independent verifier sample; it constructs a coupling whose marginal distribution is exactly the verifier distribution.
- **Acceptance rate and distribution overlap (§3.1–§3.2)**:
  - $\beta_{x_{<t}}$ is prefix-specific. The paper defines $\alpha=\mathbb{E}[\beta]$ as its average over encountered prefixes and then assumes successive acceptance events are i.i.d. for the runtime analysis.
  - The paper's $D_{\mathrm{LK}}$ is exactly total variation distance:
    $$
    D_{\mathrm{LK}}(p,q)
    =\frac12\sum_x|p(x)-q(x)|
    =1-\sum_x\min(p(x),q(x)).
    $$
    Hence $\beta=1-D_{\mathrm{LK}}(p,q)$ and $\alpha=1-\mathbb{E}[D_{\mathrm{LK}}(p,q)]$.
  - Under the i.i.d. approximation, accepting a token has probability $\alpha$ and the first rejection has probability $1-\alpha$. The number $N$ of emitted tokens is a capped geometric random variable:
    $$
    \mathbb{E}[N]
    =1+\alpha+\cdots+\alpha^\gamma
    =\frac{1-\alpha^{\gamma+1}}{1-\alpha}.
    $$
    The paper calls $1-\alpha$ the geometric distribution's “success probability”; here “success” means the first rejection/stopping event, not a successful draft acceptance.
- **Walltime and compute analysis (§3.3–§3.5)**:
  - Let $c$ be the latency of one draft-model step divided by the latency of one target-model step. One speculative round costs $\gamma c+1$ target-step units and emits the expectation above, giving speedup
    $$
    S(\alpha,\gamma,c)
    =\frac{1-\alpha^{\gamma+1}}{(1-\alpha)(\gamma c+1)}.
    $$
  - A larger draft can raise $\alpha$ but also raises $c$; the fastest draft model is therefore not necessarily the smallest or the most accurate one. The paper found models roughly two orders of magnitude smaller than the target often gave the best balance.
  - Let $\hat c$ be the draft-to-target ratio in arithmetic operations per token. The expected operation multiplier is
    $$
    \frac{(1-\alpha)(\gamma\hat c+\gamma+1)}{1-\alpha^{\gamma+1}}.
    $$
    Speculative decoding is a latency optimization, not generally a FLOP reduction: rejected branches waste computation, while the benefit comes from parallel work and fewer target-weight/KV-cache reads.
  - The optimal $\gamma$ maximizes $S$ and depends on both $\alpha$ and $c$. Because prefix difficulty varies, adapting $\gamma$ to a predicted local $\beta$ could improve over a fixed lookahead; the paper leaves this open.
- **Approximation models (§3.6)**:
  - Exactness does not require a particular draft architecture. The draft may be a smaller Transformer, an n-gram lookup, a heuristic that copies repeated context, a non-autoregressive model, or even a random proposal distribution.
  - Negligible-cost drafts can still help with modest acceptance. On English–German translation, a bigram draft achieved $\alpha\approx0.2$ and an estimated $1.25\times$ speedup.
- **Experiments (§4)**:
  - Main walltime tests use T5-XXL 11B as the target on WMT English–German translation and CNN/DailyMail summarization, batch size 1 on one TPU-v4. Drafts are T5-small 77M, T5-base 250M, and T5-large 800M.
  - T5-small gave the best measured balance: $3.4\times$ / $2.6\times$ on translation and $3.1\times$ / $2.3\times$ on summarization for temperature 0 / 1 respectively.
  - Larger drafts increased $\alpha$ but often reduced end-to-end speed because their latency cost $c$ grew. Across tested model families, smaller neural drafts commonly produced $\alpha$ values between $0.5$ and $0.9$, with sharper sampling distributions generally yielding higher acceptance.
  - The empirical speedups broadly matched the simplified theory, though implementation effects and non-i.i.d. acceptance caused deviations.
- **Limitations and extensions**:
  - The method needs spare parallel compute and is unattractive when arithmetic throughput, rather than memory bandwidth or communication, is already the bottleneck.
  - It generally increases total arithmetic work even while reducing latency and memory traffic.
  - The main experiments are batch size 1, use older/smaller model regimes, and implement end-to-end walltime only for T5; results on GPT-like and LaMDA models are acceptance-rate measurements rather than full serving benchmarks.
  - Appendix A.4 sketches a more expensive beam-search extension. Appendix A.5 introduces a lenience factor that trades exactness for acceptance and speed; the core results use the strict, distribution-preserving algorithm.
- **Exposition note**:
  - The paper's proof is more compressed than the underlying idea. It suppresses prefix conditioning, overloads $x$ as both random variable and vocabulary value, hides the key normalizer inside `norm`, and introduces $D_{\mathrm{LK}}$ instead of naming total variation distance.
  - The shortest correct mental model is
    $$
    \underbrace{p_{\text{verifier}}(x)}_{\text{desired mass}}
    =\underbrace{\min(p_{\text{draft}}(x),p_{\text{verifier}}(x))}_{\text{accepted overlap}}
    +\underbrace{\max\left(0,p_{\text{verifier}}(x)-p_{\text{draft}}(x)\right)}_{\text{rejection correction}}.
    $$
- **Takeaway**: Speculative decoding converts agreement between a cheap draft and an expensive verifier into fewer serial verifier calls. Its exactness comes from a maximal overlap coupling; its speed comes from high acceptance, a cheap draft, and hardware on which extra parallel arithmetic is cheaper than another serial target-model step.

---

## [2026] [Speculative Speculative Decoding](https://arxiv.org/abs/2603.03251)

- **Date**: 2026-08-17
- **Code**: <https://github.com/tanishqkumar/ssd>

---

- **One-liner**: Ordinary speculative decoding parallelizes token verification but still alternates draft then verify; speculative speculative decoding (SSD) runs drafting and verification asynchronously by precomputing next-round drafts for likely verification outcomes.
- **What is being speculated about?**:
  - At round $T$, ordinary SD verifies a draft sequence and returns a **verification outcome**
    $$
    v^T=(k,t^*),
    $$
    where $k\in\{0,\ldots,K\}$ is the number of accepted draft tokens and $t^*$ is the verifier-sampled bonus/correction token.
  - The next draft cannot normally begin until both $k$ and $t^*$ are known, because they determine the next prefix. SSD uses otherwise idle draft hardware to guess several likely $(k,t^*)$ outcomes while verification is still running and drafts a continuation for each one.
  - The “second” speculation is therefore not guessing target tokens directly; it is guessing the result of the first speculative-decoding verification step.
- **Relationship to ordinary speculative decoding (§2)**:
  - The paper uses $p_{\text{target}}$ and $p_{\text{draft}}$. Its background theorem writes the per-prefix acceptance rate as
    $$
    \alpha
    =\sum_x\min\{p_{\text{target}}(x),p_{\text{draft}}(x)\}
    =1-\frac12\|p_{\text{target}}-p_{\text{draft}}\|_1.
    $$
  - The correction/bonus token after a rejection comes from
    $$
    r(x)\propto\max\left(0,p_{\text{target}}(x)-p_{\text{draft}}(x)\right).
    $$
    Predicting this token is one of SSD's central difficulties, especially at nonzero temperature.
  - SSD remains lossless because cached continuations are only proposals: whichever continuation is selected is still verified by the target using ordinary speculative decoding. A cache miss falls back to another lossless proposal mechanism.
- **SSD framework (§3)**:
  - Put the target/verifier and draft/speculator on separate hardware and run them concurrently.
  - A speculation cache $S^T$ maps predicted verification outcomes to precomputed next-round draft sequences:
    $$
    S^T:v^T\mapsto s^T.
    $$
  - On a **cache hit**, the actual verification outcome is a key in $S^T$, so its next draft can be sent to the verifier immediately and draft latency is hidden.
  - On a **cache miss**, invoke a backup speculator after the outcome becomes known. The quality/latency of this fallback determines how expensive misses are.
  - Let $p_{\text{hit}}$ be cache-hit probability; $T_p,T_b$ be primary/backup draft latency relative to a verifier step; and $E_{\text{hit}},E_{\text{miss}}$ be expected tokens emitted after hit/miss drafts. The paper's speed model is
    $$
    \operatorname{speedup}_{\text{SSD}}
    =\frac{p_{\text{hit}}E_{\text{hit}}+(1-p_{\text{hit}})E_{\text{miss}}}
    {p_{\text{hit}}\max(1,T_p)+(1-p_{\text{hit}})(1+T_b)}.
    $$
  - If primary and backup are the same draft model, SSD cannot be slower than synchronous SD in this model and is strictly faster when a nonzero fraction of rounds hit the cache. The ideal gain is bounded by how much draft latency can be hidden and how much extra draft-time compute improves the proposal.
- **Saguaro — optimized SSD (§4)**:
  - The paper identifies three design problems: which verification outcomes to cache, how to make the bonus token more predictable, and how to recover from cache misses. Saguaro supplies one optimization for each.
- **1. Predict outcomes with a geometrically shaped cache (§4.1)**:
  - A verification outcome is approximately one of $(K+1)V$ possibilities: any accepted-prefix length $k$ paired with a vocabulary token $t^*$. Exhaustive precomputation is impossible.
  - Given cache budget $B$, choose a fan-out $F_k$: the number of bonus-token guesses cached for the outcome “exactly $k$ draft tokens were accepted.” Saguaro uses the top-$F_k$ draft logits at each position as bonus-token guesses.
  - Since the number of accepted tokens is approximately capped geometric, cache capacity should not be uniform across positions. Under a power-law cache-miss model, Theorem 12 gives
    $$
    F_k=F_0a_p^{k/(1+r)}\quad(k<K),
    $$
    with a separate capped-final-position factor
    $$
    F_K=F_0a_p^{K/(1+r)}(1-a_p)^{-1/(1+r)},
    $$
    where $a_p$ is primary-draft acceptance and $r$ is the empirical power-law exponent. Intuitively, spend fewer guesses on accepted-prefix lengths that are unlikely to occur.
  - Cache misses empirically fall approximately as a power law in fan-out. Geometric allocation improves cache-hit rate and decode speed over uniform allocation, especially at higher temperatures.
- **2. Shape the residual with Saguaro sampling (§4.2)**:
  - This is the surprising connection to Appendix A.1 of Leviathan et al.: residual mass is determined by where the target exceeds the draft,
    $$
    r(t)\propto\max\left(0,p_{\text{target}}(t)-p_{\text{draft}}(t)\right).
    $$
  - If a likely cached token has high draft probability, it may have little or no residual mass and therefore be unlikely to appear as the correction token. Saguaro deliberately *reduces* draft probability on the cached top-$F$ tokens so that the residual distribution concentrates on them.
  - Given draft logits $z\in\mathbb{R}^V$, fan-out $F$, and $C\in[0,1]$,
    $$
    \sigma_{F,C}(z)_t\propto
    \begin{cases}
    C\exp(z_t), & t\in\operatorname{top}_F(z),\\
    \exp(z_t), & \text{otherwise}.
    \end{cases}
    $$
  - As $C\to0$, cache-hit probability increases monotonically, but the modified draft moves farther from the target and ordinary speculative acceptance can fall. $C$ explicitly trades acceptance rate against predictability of the verification outcome.
  - This does not bias final outputs: the verifier uses the same modified draft distribution in its accept/reject and residual calculations, so the ordinary speculative-decoding correction still restores $p_{\text{target}}$ exactly.
- **3. Use a batch-dependent fallback (§4.3)**:
  - At low batch size, a cache miss is uncommon enough that a slower, accurate neural backup can be worthwhile.
  - At large batch size, the probability that at least one request misses approaches one and the whole batch stalls on fallback latency. Saguaro switches above a critical batch size $b^*$ to a near-zero-latency backup (random tokens in the implementation; an n-gram draft is an obvious improvement).
  - The fast fallback sacrifices accepted tokens on a miss to keep the verifier from waiting. This is another latency-versus-quality-of-proposal tradeoff, not a change to the target output distribution.
- **Systems design (Appendix B)**:
  - Custom PyTorch engine with PagedAttention, continuous batching, tensor parallelism, BF16, compilation, and CUDA Graphs.
  - Target model uses four H100s; the asynchronous draft uses a separate fifth H100. Target and draft exchange only the verification result and next speculative tokens/logits over NCCL; the target never receives the draft speculation cache, and no KV cache is transferred between devices.
  - All fan-out branches are decoded together with a custom sparse attention mask. Mask materialization and non-coalesced attention are significant overheads, so most measured gain comes from hiding draft latency rather than increasing lookahead.
  - SSD uses still more speculative compute than SD: entire continuations prepared for unrealized outcomes are discarded. It also stores $O(BK(K+1)F)$ speculative tokens/logits plus their draft KV state.
- **Evaluation (§5, Appendix B)**:
  - Four datasets: HumanEval, UltraFeedback, Alpaca, and GSM8K; 128 prompts each and 512 decoded tokens per prompt. Measurements exclude prefill. The headline baseline comparisons use greedy decoding; temperature sweeps use vanilla sampling rather than top-$p$/top-$k$.
  - Main comparison: Llama-3.1-70B-Instruct target with Llama-3.2-1B-Instruct draft; replication: Qwen3-32B with Qwen3-0.6B. AR/SD use four H100s; SSD uses the same four target GPUs plus one draft H100.
  - Llama average: AR $54.7$ tok/s, SD $161.8$ tok/s, SSD $255.8$ tok/s — $1.58\times$ over the authors' SD and $4.68\times$ over their AR baseline.
  - Qwen average: AR $88.8$ tok/s, SD $136.8$ tok/s, SSD $203.8$ tok/s — $1.49\times$ over SD and $2.29\times$ over AR.
  - Across the paper's strongest open-source SD baselines, Saguaro is reported as about $30\%$ faster on average and improves the throughput–latency Pareto frontier, with the largest gains at low batch size.
- **Caveats / questions**:
  - SSD spends an additional GPU on the draft while primary AR/SD baselines use four GPUs. The paper reports per-GPU throughput for the Pareto comparison, but raw latency gains should still be read as a hardware-for-latency trade.
  - Results are decode-only, use fixed 512-token generations, and concentrate on one node of H100s. Prefill-heavy workloads, shorter outputs, other interconnects, and production request-length distributions may change the balance.
  - Speculative methods remain unattractive for already throughput-bound workloads such as large-scale RL or offline data generation because they add compute. SSD improves the measured frontier but does not remove that fundamental limitation.
  - Cache construction assumes empirical regularity in acceptance lengths and bonus-token ranks. Temperature and batch size reduce hit rates; Saguaro mitigates rather than eliminates misses.
  - The custom attention path is itself a bottleneck. Better multi-branch kernels, non-neural fallbacks, cluster-level shared draft services, and composition with EAGLE/token-tree methods are open directions.
- **Connection to the original paper**:
  - Leviathan et al. remove serial target-model calls by verifying several candidate tokens together.
  - Kumar et al. remove the remaining draft→verify synchronization by predicting the verifier's complete outcome $(k,t^*)$ and preparing the next draft before that outcome exists.
  - The original residual correction guarantees exactness; SSD turns that same residual into a systems bottleneck because its sampled token determines the next prefix. Saguaro sampling manipulates the draft distribution to make the residual easier to cache while leaving the verifier marginal unchanged.
- **Takeaway**: SSD is speculative execution applied one level higher. It predicts not only future tokens but which speculative branch the verifier will commit to, then converts correct outcome predictions into hidden draft latency. Saguaro's core insight is that the best draft distribution for asynchronous serving need not be the one with maximum acceptance—it may be worth lowering acceptance slightly to make the verifier's correction token predictable enough to precompute the next round.
