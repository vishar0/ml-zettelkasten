# Karpathy Curriculum

- **Created**: 2025-10-14
- **Last Updated**: 2026-10-14
- **Status**: `In Progress`

---

- [ ] [micrograd](https://github.com/karpathy/micrograd)
- [x] [minbpe](https://github.com/karpathy/minbpe)
- [ ] [llama2.c](https://github.com/karpathy/llama2.c)
- [ ] [llm.c](https://github.com/karpathy/llm.c)
- [ ] [nanochat](https://github.com/karpathy/nanochat)
  - [x] [[papers-data-centric-ml]] ClimbMix pretraining dataset
  - [x] [[scaling-laws]] Scaling ladder, Chinchilla optimality, IsoFLOP curves, 9.5 tokens per param, power lines
  - [ ] [[papers-optimizers#Muon]] Muon optimizer

---

## [minbpe](https://github.com/karpathy/minbpe)

- **Date**: 2025-10-24

---

- Original BPE (Byte Pair Encoding) algorithm in [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) [[papers-ml-fundamentals]], and then popularized by GPT-2.
  - Basic BPE implementation in <https://github.com/karpathy/minbpe/blob/master/minbpe/basic.py>
  - See my independent impl at <https://gist.github.com/vishar0/164380801d7b21ad57e3de789fea5315>
    - **Known issues in my impl** (checked 2026-07-03 vs `minbpe/basic.py`):
      - *Base vocab init bug.* I use `chr(token).encode()`, which UTF-8-encodes the *code point* — correct only for 0–127. Tokens 128–255 become 2 bytes (e.g. `chr(200).encode()` → `b'\xc3\x88'` not `b'\xc8'`). Result: `tokenize` hits `assert False` on any non-ASCII input, and `detokenize` round-trip is wrong (`"café"` → `"cafÃ©"`). Defeats the point of *byte-level* BPE; only ASCII happens to work. Fix: `bytes([token])` (what Karpathy does).
      - *Not true BPE encoding.* My `tokenize` does greedy longest-ish match over the vocab (`reversed(vocab.items())`). Karpathy's `encode` replays learned merges in order of lowest merge index. Different algorithms → can yield different tokenizations for the same string (mine still round-trips, but won't match GPT-2). To fix, store the `merges` map (pair → id, in learned order) — my `fit` currently only builds `vocab` — and replay it. Also O(n·vocab) vs his O(n·merges).
- Couple of additional things to handle:
  - RegexTokenizer: Preprocesses the input text by splitting it into categories (letters, numbers, puncutation) before tokenization. Avoids merges across cateogry boundaries. See regex based splitting and chunk handling in `train()` and `encode_ordinary()` of <https://github.com/karpathy/minbpe/blob/master/minbpe/regex.py>.
  - Handling special tokens like `<|promptstart|>` or `<|endoftext|>`, etc.
- > Tokenization is at the heart of a lot of weirdness in LLMs and I would advise that you do not brush it off. A lot of the issues that may look like issues with the neural network architecture actually trace back to tokenization. Here are just a few examples:
  - > Why can't LLM spell words? **Tokenization**.
  - > Why can't LLM do super simple string processing tasks like reversing a string? **Tokenization**.
  - > Why is LLM worse at non-English languages (e.g. Japanese)? **Tokenization**.
  - > Why is LLM bad at simple arithmetic? **Tokenization**.
  - > Why did GPT-2 have more than necessary trouble coding in Python? **Tokenization**.
  - > Why did my LLM abruptly halt when it sees the string "<|endoftext|>"? **Tokenization**.
  - > What is this weird warning I get about a "trailing whitespace"? **Tokenization**.
  - > Why did the LLM break if I ask it about "SolidGoldMagikarp"? **Tokenization**.
  - > Why should I prefer to use YAML over JSON with LLMs? **Tokenization**.
  - > Why is LLM not actually end-to-end language modeling? **Tokenization**.
  - > What is the real root of suffering? **Tokenization**.

## [nanochat](https://github.com/karpathy/nanochat)

- **Date**: 2026-07-03

---

Reading checklist — file tree of the local checkout at commit `4b40774` (Run 4, 2026-03-04; README comments verbatim), checked off as read:

- [ ] root
  - [ ] `README.md`
  - [x] `pyproject.toml` — deps; `cpu`/`gpu` conflicting extras, `default-groups = []`
- [x] `nanochat/` (the core library)
  - [x] `checkpoint_manager.py` — Save/Load model checkpoints
  - [x] `common.py` — Misc small utilities, quality of life
  - [x] `core_eval.py` — Evaluates base model CORE score (DCLM paper)
  - [x] `dataloader.py` — Tokenizing Distributed Data Loader
  - [x] `dataset.py` — Download/read utils for pretraining data
  - [ ] `engine.py` — Efficient model inference with KV Cache
  - [x] `execution.py` — Allows the LLM to execute Python code as tool
  - [x] `flash_attention.py` — FA3 wrapper, SDPA fallback
  - [ ] `fp8.py` — fp8 support
  - [x] `gpt.py` — The GPT nn.Module Transformer
  - [x] `loss_eval.py` — Evaluate bits per byte (instead of loss)
  - [ ] `optim.py` — AdamW + Muon optimizer, 1GPU and distributed
  - [x] `report.py` — Utilities for writing the nanochat Report
  - [x] `tokenizer.py` — BPE Tokenizer wrapper in style of GPT-4
- [ ] `scripts/` (the pipeline stages)
  - [x] `tok_train.py` — Tokenizer: train it
  - [x] `tok_eval.py` — Tokenizer: evaluate compression rate
  - [x] `base_train.py` — Base model: train
  - [x] `base_eval.py` — Base model: CORE score, bits per byte, samples
  - [ ] `chat_sft.py` — Chat model: train SFT
  - [ ] `chat_rl.py` — Chat model: reinforcement learning
  - [ ] `chat_eval.py` — Chat model: eval tasks
  - [ ] `chat_cli.py` — Chat model: talk to over CLI
  - [ ] `chat_web.py` — Chat model: talk to over WebUI
- [ ] `tasks/` (evals)
  - [ ] `common.py` — TaskMixture | TaskSequence
  - [ ] `arc.py` — Multiple choice science questions
  - [ ] `customjson.py` — Make Task from arbitrary jsonl convos
  - [ ] `gsm8k.py` — 8K Grade School Math questions
  - [ ] `humaneval.py` — Misnomer; Simple Python coding task
  - [ ] `mmlu.py` — Multiple choice questions, broad topics
  - [ ] `smoltalk.py` — Conglomerate dataset of SmolTalk from HF
  - [ ] `spellingbee.py` — Task teaching model to spell/count letters
- [x] `runs/`
  - [x] `speedrun.sh` — Train the ~$100 nanochat d20
  - [x] `miniseries.sh` — Miniseries training script
  - [x] `scaling_laws.sh` — Scaling laws experiments
  - [x] `runcpu.sh` — Small example of how to run on CPU/MPS
- [ ] `dev/`
  - [x] `repackage_data_reference.py` — Pretraining data shard generation (FineWebEdu-100B, ClimbMix-400B)
  - [x] `gen_synthetic_data.py` — Example synthetic data for identity
  - [ ] `LOG.md` — dev log
  - [ ] `LEADERBOARD.md` — speedrun leaderboard
  - [ ] `scaling_analysis.ipynb`, `estimate_gpt3_core.ipynb`, `scaling_laws_jan26.png` — scaling-laws analysis

## [The FLOPs Calculus of Language Model Training](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4)

- **Author**: Dzmitry Bahdanau
- **Date**: 2026-10-14
- Related: [[scaling-laws]]; nanochat's `estimate_flops` in `gpt.py`.

---

### Why training costs approximately 6ND FLOPs

For a dense model, the dominant weight-matrix arithmetic is approximately $C = 6ND$, where $N$ is the number of participating weights and $D$ is the total number of training tokens. A FLOP is one floating-point operation; a multiply-add counts as two FLOPs even if implemented as one fused instruction.

For one connection with input $x$, weight $w$, and output $y = \sum_i w_i x_i$, let $g = \partial L_{\text{loss}}/\partial y$ be the incoming gradient. Each weight contributes six operations per example:

| Step | Operation | Purpose |
| --- | --- | --- |
| 1 | Multiply $wx$ | Compute the forward contribution. |
| 2 | Add it into $y$ | Sum contributions from inputs. |
| 3 | Multiply $wg$ | Compute the input-gradient contribution. |
| 4 | Add it into $\partial L_{\text{loss}}/\partial x$ | Sum contributions from outputs influenced by this input. |
| 5 | Multiply $xg$ | Compute the weight-gradient contribution. |
| 6 | Add it into $\partial L_{\text{loss}}/\partial w$ | Accumulate contributions across examples/tokens sharing the weight. |

Thus: **2 forward + 2 input-gradient + 2 weight-gradient FLOPs**. For $x=3$, $w=2$, and $g=4$, the three products are respectively 6, 8, and 12; each is then accumulated into its corresponding sum. The optimizer's weight update is separate from these six operations.

### Why attention can be a smaller correction

Here $d$ is the hidden dimension (features per token), and $L$ is the sequence length, not the number of layers. Big-O describes growth while ignoring constant factors.

- **Learned weight multiplications: $O(d^2)$ per token per layer.** Multiplying a $d$-element vector by a $d\times d$ matrix produces $d$ outputs, each combining $d$ inputs. Feed-forward layers have the same scaling when their intermediate width is a fixed multiple of $d$.
- **Attention interactions: $O(Ld)$ per token per layer.** A token's query is compared with $L$ keys, then the resulting scores are used to combine $L$ value vectors. Summing across heads gives $O(Ld)$ for each of these two operations, assuming total head width $d$.

The distinction is **learned weight matrices versus matrices of activations**. The learned query/key/value and output projections are already included in weight FLOPs. The additional interactions $QK^\top$ and $\operatorname{softmax}(QK^\top/\sqrt{d_k})V$ are also matrix multiplications, but they do not multiply by learned parameter matrices. The data-dependent attention scores are not model parameters.

| Work | Per token, per layer | Whole sequence, per layer |
| --- | --- | --- |
| Learned weight multiplications | $O(d^2)$ | $O(Ld^2)$ |
| Attention interactions | $O(Ld)$ | $O(L^2d)$ |

Attention is quadratic in **sequence length**; weight computation is quadratic in **model width**. Their relative growth terms satisfy $dL/d^2=L/d$. For $d=8192$ and $L=1024$, $d^2\approx67.1$ million and $dL\approx8.4$ million: the former growth term is eight times larger. These are illustrative growth terms, not exact FLOP totals.

Therefore, ignoring attention is most defensible when $L/d$ is small. Increasing parameter count through more layers alone does not make this ratio smaller, and long contexts can make attention important or dominant. The article's claim about large models assumes sufficiently large width relative to context length.

### Where nanochat's attention factor of 12 comes from

Using just hidden dimension $d$ and sequence length $L$, attention adds approximately **$12dL$ FLOPs per token per layer** for forward + backward.

There are two attention matrix multiplications:

1. **Compute scores: $QK^\top$.** For one token, compare its query with $L$ keys. Each comparison combines $d$ features using multiplication and addition, giving approximately $2dL$ forward FLOPs.
2. **Combine values: $PV$.** Here $P$ contains the attention probabilities after softmax. Each of the $d$ output features is a weighted sum of $L$ values, again giving approximately $2dL$ forward FLOPs.

Each forward matrix multiplication needs **two backward matrix multiplications**, one to compute the gradient for each input. Each has the same leading arithmetic cost as the forward operation, so forward + backward costs three times forward alone.

| Attention operation | Forward | Backward | Total per token |
| --- | --- | --- | --- |
| Query–key scores | $2dL$ | $4dL$ | $6dL$ |
| Weighted value combination | $2dL$ | $4dL$ | $6dL$ |
| **Total** | **$4dL$** | **$8dL$** | **$12dL$** |

Thus **$12 = 2$ attention multiplications $\times 3$ (forward + two backward) $\times 2$ FLOPs per multiply-add**. This is the explicit constant behind the earlier $O(dL)$ scaling; across a full sequence it becomes $12dL^2$.

Nanochat sums this contribution across layers, replacing $L$ with the window-capped effective sequence length for sliding-window attention. Its comment mentions query–key multiplication, but the factor of 12 also includes the weighted value combination. The learned query/key/value and output projections are already counted in the separate $6N$ term. Softmax arithmetic and recomputation are omitted; causal masking and the actual kernel can change the executed FLOP count.

Reference: [PaLM, Appendix B](https://arxiv.org/html/2204.02311v5#A2).

### Why LSTMs can have similar FLOPs but slower training

The same weight-operation accounting approximately applies to LSTMs. However, **training time depends on achieved throughput**, not just operation count: $T\approx C/\tau$, where $\tau$ is the achieved FLOPs per second. FLOPs measures work; FLOP/s measures speed.

An LSTM's hidden and cell states at position $t$ depend on those at position $t-1$: $(h_t,c_t)=\operatorname{LSTM}(x_t,h_{t-1},c_{t-1})$. Separate sequences can run in parallel, but the recurrent computation within each sequence must proceed in order.

For batch size $B=32$, sequence length $L=256$, and hidden width $d=1600$:

| | LSTM recurrent multiplication | Transformer weight multiplication |
| --- | --- | --- |
| Representative matrix shapes | $[32\times1600][1600\times6400]$ | $[8192\times1600][1600\times6400]$ |
| Rows available together | $B=32$ | $BL=8192$ |
| Organization across the sequence | 256 dependent recurrent multiplications | Positions grouped into a large multiplication |

For the LSTM, $6400=4\times1600$ represents its four gate/candidate groups. A Transformer feed-forward projection can have the same dimensions for a different reason: expanding its hidden width by a factor of four.

One 8192-row multiplication has the same arithmetic as 256 separate 32-row multiplications. The larger operation gives a GPU more parallel work and better opportunities to reuse loaded weights and amortize overhead. Bahdanau reports below 20 teraFLOP/s for the small shape, over ten times lower throughput than the large shape. This compares **arithmetic per second**, not the duration of one small multiplication with one large multiplication; the exact rates depend on hardware and implementation.

During training, ordinary backpropagation saves intermediate activations with memory growing roughly as $BLd$ per layer. Under a fixed memory budget, longer sequences can force a smaller batch. This affects both architectures, but an LSTM's recurrent multiplication then has fewer rows ($B$), whereas Transformer weight multiplications can still group $BL$ positions together. Some LSTM input projections and gradient computations can also be grouped across time; the previous-state dependency remains in the recurrent path.

Transformer positions can be processed together **during training** because all input tokens are known; causal masking restricts information flow without requiring sequential processing of positions. Layers remain sequential, and ordinary autoregressive generation still produces tokens in order. Recurrence's limit on parallelism is one reason Transformers displaced LSTMs for large language models, rather than a complete explanation by itself.

### Limits of the estimate

The $6ND$ rule estimates dominant dense weight FLOPs, not an exact total or guaranteed runtime. Attention, normalization, activations, memory access, communication, and recomputation affect actual cost and throughput. An extra full forward pass for activation checkpointing adds roughly $2ND$, giving about $8ND$ under the same approximation; the amount depends on the recomputation strategy. For MoE models, use active weights per token rather than total stored weights.

Further references: [Transformer attention equations](https://arxiv.org/html/1706.03762v7#S3.SS2), [Megatron-LM](https://arxiv.org/abs/1909.08053), [NVIDIA matrix multiplication performance](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html), and [LSTM equations](https://docs.pytorch.org/docs/2.14/generated/torch.nn.LSTM.html).
