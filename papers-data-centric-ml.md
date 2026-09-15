# Data-Centric ML

- **Created**: 2025-07-21
- **Last Updated**: 2026-09-15
- **Status**: `In Progress`
- **Description**: Data selection, filtering, pruning, and mixture optimization for pretraining, and the curated web corpora that result from it.
- **Related**:
  - [[papers-scaling-laws]] — Data selection can change or beat standard scaling curves.
  - [[karpathy-repos]] — nanochat pretrains on ClimbMix.

---

- [ ] Beyond neural scaling laws: beating power law scaling via data pruning — [paper](https://arxiv.org/abs/2206.14486)
- [ ] T-MARS: Improving Visual Representations by Circumventing Text Feature Learning — [paper](https://arxiv.org/abs/2307.03132)
- [X] Datology premise — [blog](https://blog.datologyai.com/introducing-datologyai-making-models-better-through-better-data-automatically)
- [X] CLIP improvement 1 (image-text) — [blog](https://blog.datologyai.com/productionized-multimodal-data-curation-at-the-billion-sample-scale)
- [X] CLIP improvement 2 (image-text, Ricardo) — [blog](https://blog.datologyai.com/multimodal-plus-blogpost)
- [ ] [2024] [HF] The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale — [paper](https://arxiv.org/abs/2406.17557)
- [ ] [2024] DCLM: DataComp-LM: In search of the next generation of training sets for language models — [paper](https://arxiv.org/abs/2406.11794)
- [ ] [2025] [NVIDIA] [CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](#2025-nvidia-climb-clustering-based-iterative-data-mixture-bootstrapping-for-language-model-pre-training) — [paper](https://arxiv.org/abs/2504.13161)

---

## [2025] [NVIDIA] CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training

- **Date**: 2026-09-15
- **Arxiv**: <https://arxiv.org/abs/2504.13161>
- **Data**: <https://research.nvidia.com/labs/lpr/climb/>, [nvidia/Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix), [nvidia/ClimbLab](https://huggingface.co/datasets/nvidia/ClimbLab)

---

- **Abstract**:
  - > Pre-training datasets are typically collected from web content and lack inherent domain divisions. For instance, widely used datasets like Common Crawl do not include explicit domain labels, while manually curating labeled datasets such as The Pile is labor-intensive. Consequently, identifying an optimal pre-training data mixture remains a challenging problem, despite its significant benefits for pre-training performance. To address these challenges, we propose CLustering-based Iterative Data Mixture Bootstrapping (Nemotron-CLIMB), an automated framework that discovers, evaluates, and refines data mixtures in a pre-training setting. Specifically, **Nemotron-CLIMB embeds and clusters large-scale datasets in a semantic space and then iteratively searches for optimal mixtures using a smaller proxy model and a predictor**. When continuously trained on 400B tokens with this mixture, our 1B model exceeds the state-of-the-art Llama-3.2-1B by 2.0%. Moreover, we observe that optimizing for a specific domain (e.g., Social Sciences) yields a 5% improvement over random sampling. Finally, we introduce Nemotron-ClimbLab, a filtered 1.2-trillion-token corpus with 20 clusters as a research playground, and Nemotron-ClimbMix, a compact yet powerful 400-billion-token dataset designed for efficient pre-training that delivers superior performance under an equal token budget. We analyze the final data mixture, elucidating the characteristics of an optimal data mixture. Our data is available at: <https://research.nvidia.com/labs/lpr/climb/>
