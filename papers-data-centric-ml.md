# Data-Centric ML

- **Created**: 2025-07-21
- **Last Updated**: 2026-09-15
- **Status**: `In Progress`
- **Description**: Data selection, filtering, pruning, and mixture optimization for pretraining; the curated web corpora that result from it.
- **Related**:
  - [[papers-scaling-laws]] — Data selection can change or beat standard scaling curves; nanochat's leaderboard is the concrete case where swapping the corpus (FineWeb-Edu → ClimbMix) beat every architecture tweak.
  - [[karpathy-repos]] — nanochat pretrains on ClimbMix via `nanochat/dataset.py`; `dev/repackage_data_reference.py` documents how the shards were built.
  - [[papers-small-language-models]] — BabyLM and LittleLearner-style corpora are the developmentally-plausible counterpart to general web mixtures.

---

- [ ] Beyond neural scaling laws: beating power law scaling via data pruning — [paper](https://arxiv.org/abs/2206.14486)
- [ ] T-MARS: Improving Visual Representations by Circumventing Text Feature Learning — [paper](https://arxiv.org/abs/2307.03132)
- [X] Datology premise — [blog](https://blog.datologyai.com/introducing-datologyai-making-models-better-through-better-data-automatically)
- [X] CLIP improvement 1 (image-text) — [blog](https://blog.datologyai.com/productionized-multimodal-data-curation-at-the-billion-sample-scale)
- [X] CLIP improvement 2 (image-text, Ricardo) — [blog](https://blog.datologyai.com/multimodal-plus-blogpost)

Pretraining corpora:

- [ ] [2024] [HF] The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale — [paper](https://arxiv.org/abs/2406.17557). FineWeb (15T) and FineWeb-Edu (1.3T, classifier-filtered for educational content); nanochat's original corpus.
- [ ] [2024] DCLM: DataComp-LM: In search of the next generation of training sets for language models — [paper](https://arxiv.org/abs/2406.11794). Source of the CORE eval used by nanochat and F1.
- [ ] [2025] [NVIDIA] [CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](#2025-nvidia-climb-clustering-based-iterative-data-mixture-bootstrapping-for-language-model-pre-training) — [paper](https://arxiv.org/abs/2504.13161). Produces ClimbMix (400B), nanochat's current corpus and Flourish F1's P0 text dataset.

---

## [2025] [NVIDIA] CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training

- **Date**: 2026-09-15
- **Arxiv**: <https://arxiv.org/abs/2504.13161> (arXiv title: *Nemotron-CLIMB*; Diao et al.)
- **Datasets**: [nvidia/Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix) (400B tokens, stored as GPT-2 token ids), ClimbLab (the larger 1.2T-token clustered pool it was selected from)

---

**Premise** (from the abstract, not yet read in full): instead of hand-picking domain weights, embed and cluster a large web corpus into semantic groups, then iteratively search over cluster mixtures using small proxy models and a learned performance predictor. Each round refines the mixture toward the target ("bootstrapping"). ClimbMix is the compact 400B-token mixture that falls out of this; the claim is that it beats equal-token training on the unmixed pool and on prior curated corpora.

**Why it matters for me**

- **nanochat** ([[karpathy-repos]]): leaderboard run 4 (2026-03-03) switched FineWeb-Edu 100B → ClimbMix 400B and cut time-to-GPT-2-CORE from 2h46m to 2h01m (−27%), enough to also shrink the model from d26 to d24. Karpathy notes it was his *sixth* attempt to beat FineWeb-Edu on CORE (FineWeb, DCLM, Olmo all failed) and the first that worked. Only ~150 shards (~7B tokens) are needed for GPT-2 capability.
  - Repackaged as `karpathy/climbmix-400b-shuffle`: 6,543 zstd Parquet shards of ~92 MB / ~86k docs each, single `text` column, decoded back from NVIDIA's GPT-2 ids and shuffled once with seed 42. Last shard is nanochat's val split.
- **Flourish F1 Sprint 2** (Sep 2026): ClimbMix is the P0 text corpus for the text-diffusion scaling runs, chosen for direct comparison with nanochat's CORE numbers. The pretokenization question is live: nanochat trains its own 32k BPE on ClimbMix, while F1's datalayer stores GPT-2 uint16 tokens, so tokens-per-parameter is not directly comparable across the two.

**Questions to answer when reading**

- How much of the gain is *mixture* vs the underlying pool quality (Nemotron-CC + others)? Is there an ablation against a uniform mixture over the same clusters?
- Does the mixture found with small proxies transfer to larger models, i.e. is it a scaling-law-stable choice ([[papers-scaling-laws]])?
- Multi-epoch behavior: nanochat sees <2% of the corpus for GPT-2; diffusion training will want multiple passes. Does the paper say anything about repetition?
