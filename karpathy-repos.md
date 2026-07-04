# Karpathy Curriculum

- **Created**: 2025-10-14
- **Last Updated**: 2026-07-03
- **Status**: `In Progress`

---

- [ ] [micrograd](https://github.com/karpathy/micrograd)
- [x] [minbpe](https://github.com/karpathy/minbpe)
- [ ] [llama2.c](https://github.com/karpathy/llama2.c)
- [ ] [llm.c](https://github.com/karpathy/llm.c)
- [ ] [nanochat](https://github.com/karpathy/nanochat)

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

Reading checklist — file tree from the README (comments verbatim), checked off as read:

- [ ] root
  - [ ] `README.md`
  - [x] `pyproject.toml` — deps; `cpu`/`gpu` conflicting extras, `default-groups = []`
  - [x] `uv.lock`
  - [x] `LICENSE`
- [ ] `nanochat/` (the core library)
  - [x] `__init__.py` — empty
  - [ ] `checkpoint_manager.py` — Save/Load model checkpoints
  - [ ] `common.py` — Misc small utilities, quality of life
  - [ ] `core_eval.py` — Evaluates base model CORE score (DCLM paper)
  - [ ] `dataloader.py` — Tokenizing Distributed Data Loader
  - [ ] `dataset.py` — Download/read utils for pretraining data
  - [ ] `engine.py` — Efficient model inference with KV Cache
  - [ ] `execution.py` — Allows the LLM to execute Python code as tool
  - [ ] `flash_attention.py` — FA3 wrapper, SDPA fallback
  - [ ] `fp8.py` — fp8 support
  - [ ] `gpt.py` — The GPT nn.Module Transformer
  - [ ] `loss_eval.py` — Evaluate bits per byte (instead of loss)
  - [ ] `optim.py` — AdamW + Muon optimizer, 1GPU and distributed
  - [ ] `tokenizer.py` — BPE Tokenizer wrapper in style of GPT-4
- [ ] `scripts/` (the pipeline stages)
  - [ ] `tok_train.py` — Tokenizer: train it
  - [ ] `tok_eval.py` — Tokenizer: evaluate compression rate
  - [ ] `base_train.py` — Base model: train
  - [ ] `base_eval.py` — Base model: CORE score, bits per byte, samples
  - [ ] `chat_sft.py` — Chat model: train SFT
  - [ ] `chat_rl.py` — Chat model: reinforcement learning
  - [ ] `chat_eval.py` — Chat model: eval tasks
  - [ ] `chat_cli.py` — Chat model: talk to over CLI
- [ ] `tasks/` (evals)
  - [ ] `common.py` — TaskMixture | TaskSequence
  - [ ] `arc.py` — Multiple choice science questions
  - [ ] `mmlu.py` — Multiple choice questions, broad topics
  - [ ] `gsm8k.py` — 8K Grade School Math questions
  - [ ] `humaneval.py` — Misnomer; Simple Python coding task
  - [ ] `smoltalk.py` — Conglomerate dataset of SmolTalk from HF
- [ ] `runs/`
  - [ ] `speedrun.sh` — Train the ~$100 nanochat d20
  - [ ] `miniseries.sh` — Miniseries training script
  - [ ] `scaling_laws.sh` — Scaling laws experiments
  - [ ] `runcpu.sh` — Small example of how to run on CPU/MPS
- [ ] `dev/`
  - [ ] `repackage_data_reference.py` — Pretraining data shard generation
  - [ ] `LOG.md` — dev log
  - [ ] `LEADERBOARD.md` — speedrun leaderboard
  - [ ] `scaling_analysis.ipynb`, `estimate_gpt3_core.ipynb`, `scaling_laws_jan26.png` — scaling-laws analysis
- [ ] `tests/`
  - [ ] `test_engine.py`
  - [ ] `test_attention_fallback.py` — attention fallback (FA3 vs SDPA)
