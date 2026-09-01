# Linear Attention

- **Created**: 2026-08-11
- **Last Updated**: 2026-08-11
- **Status**: `Not Started`
- **Description**: Attention-like sequence models whose recurrent form updates a fixed-size memory, with emphasis on fast weights, gating, delta-rule updates, and hardware-efficient parallel training.
- **Related**:
  - [[state-space-models]] — The neighboring recurrent-model lineage; structured state-space duality makes the relationship especially explicit.

---

- [ ] [1991] [Schmidhuber] The 1991 Unnormalized Linear Transformer (ULTRA) — [blog](https://people.idsia.ch/~juergen/1991-unnormalized-linear-transformer.html)
- [ ] [2020] Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention — [paper](https://arxiv.org/abs/2006.16236)
- [ ] [2021] [Schmidhuber] Linear Transformers Are Secretly Fast Weight Programmers — [paper](https://arxiv.org/abs/2102.11174)
- [ ] [2023] RWKV: Reinventing RNNs for the Transformer Era — [paper](https://arxiv.org/abs/2305.13048)
- [ ] [2023] Retentive Network: A Successor to Transformer for Large Language Models — [paper](https://arxiv.org/abs/2307.08621)
- [ ] [2023] [ChrisRe] Zoology: Measuring and Improving Recall in Efficient Language Models — [paper](https://arxiv.org/abs/2312.04927)
- [ ] [2023] Gated Linear Attention Transformers with Hardware-Efficient Training — [paper](https://arxiv.org/abs/2312.06635)
- [ ] [2024] [ChrisRe] The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry — [paper](https://arxiv.org/abs/2402.04347)
- [ ] [2024] [ChrisRe] Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff — [paper](https://arxiv.org/abs/2402.18668)
- [ ] [[state-space-models]] [2024] Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality — [paper](https://arxiv.org/abs/2405.21060)
- [ ] [2024] DeltaNet Explained, Part I: The Model — [blog](https://sustcsonglin.github.io/blog/2024/deltanet-1/)
- [ ] [2024] DeltaNet Explained, Part II: The Algorithm — [blog](https://sustcsonglin.github.io/blog/2024/deltanet-2/)
- [ ] [2024] DeltaNet Explained, Part III: The Neural Architecture — [blog](https://sustcsonglin.github.io/blog/2024/deltanet-3/)
- [ ] [2024] DeltaNet: Parallelizing Linear Transformers with the Delta Rule over Sequence Length — [paper](https://arxiv.org/abs/2406.06484)
- [ ] [2024] Gated DeltaNet: Improving Mamba2 with Delta Rule — [paper](https://arxiv.org/abs/2412.06464)
- [ ] [2025] [SamGershman] Fast Weight Programming and Linear Transformers: From Machine Learning to Neurobiology — [paper](https://arxiv.org/abs/2508.08435)
- [ ] [2025] Kimi Linear: An Expressive, Efficient Attention Architecture — [paper](https://arxiv.org/abs/2510.26692)
- [ ] [2026] Sliding-Window Beats Linear Attention — [paper](https://arxiv.org/abs/2608.28444)
  - Finds that sliding-window attention with attention sinks matches or outperforms post-trained linear-attention replacements, especially on long-context retrieval, without requiring post-training.
