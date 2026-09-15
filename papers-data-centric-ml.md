# Data-Centric ML

- **Created**: 2025-07-21
- **Last Updated**: 2026-09-15
- **Status**: `In Progress`
- **Description**: Data selection, filtering, pruning, and mixture optimization for pretraining, and the curated web corpora that result from it.
- **Related**:
  - [[papers-scaling-laws]] — Data selection can change or beat standard scaling curves.
  - [[karpathy-repos]] — nanochat pretrains on ClimbMix.

---

## Datalogy

- [ ] Beyond neural scaling laws: beating power law scaling via data pruning — [paper](https://arxiv.org/abs/2206.14486)
- [ ] T-MARS: Improving Visual Representations by Circumventing Text Feature Learning — [paper](https://arxiv.org/abs/2307.03132)
- [X] Datology premise — [blog](https://blog.datologyai.com/introducing-datologyai-making-models-better-through-better-data-automatically)
- [X] CLIP improvement 1 (image-text) — [blog](https://blog.datologyai.com/productionized-multimodal-data-curation-at-the-billion-sample-scale)
- [X] CLIP improvement 2 (image-text, Ricardo) — [blog](https://blog.datologyai.com/multimodal-plus-blogpost)

## LLM Pretraining Corpus - Offline Data Mixture Search

- [ ] [2024] [HF] The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale — [paper](https://arxiv.org/abs/2406.17557)
- [ ] [2024] DCLM: DataComp-LM: In search of the next generation of training sets for language models — [paper](https://arxiv.org/abs/2406.11794)
- [x] [2025] [NVIDIA] [CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training](#2025-nvidia-climb-clustering-based-iterative-data-mixture-bootstrapping-for-language-model-pre-training) — [paper](https://arxiv.org/abs/2504.13161)

## Online / Adaptive Data Selection

- [ ] [2017] [Graves,deepmind] Automated Curriculum Learning for Neural Networks — [paper](https://arxiv.org/abs/1704.03003)
- [ ] [2023] DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining — [paper](https://arxiv.org/abs/2305.10429)
- [ ] [2023] ODM: Efficient Online Data Mixing For Language Model Pre-Training — [paper](https://arxiv.org/abs/2312.02406)
- [ ] [2023] Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models — [paper](https://arxiv.org/abs/2307.14430)
- [ ] [2024] Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance — [paper](https://arxiv.org/abs/2403.16952)
- [ ] [2024] ADO: Adaptive Data Optimization: Dynamic Sample Selection with Scaling Laws — [paper](https://arxiv.org/abs/2410.11820)

---

## [2025] [NVIDIA] CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training

- **Date**: 2026-09-15
- **Arxiv**: <https://arxiv.org/abs/2504.13161>
- **Data**: <https://research.nvidia.com/labs/lpr/climb/>, [nvidia/Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix), [nvidia/ClimbLab](https://huggingface.co/datasets/nvidia/ClimbLab)

---

- **Abstract**:
  - > Pre-training datasets are typically collected from web content and lack inherent domain divisions. For instance, widely used datasets like Common Crawl do not include explicit domain labels, while manually curating labeled datasets such as The Pile is labor-intensive. Consequently, identifying an optimal pre-training data mixture remains a challenging problem, despite its significant benefits for pre-training performance. To address these challenges, we propose CLustering-based Iterative Data Mixture Bootstrapping (Nemotron-CLIMB), an automated framework that discovers, evaluates, and refines data mixtures in a pre-training setting. Specifically, **Nemotron-CLIMB embeds and clusters large-scale datasets in a semantic space and then iteratively searches for optimal mixtures using a smaller proxy model and a predictor**. When continuously trained on 400B tokens with this mixture, our 1B model exceeds the state-of-the-art Llama-3.2-1B by 2.0%. Moreover, we observe that optimizing for a specific domain (e.g., Social Sciences) yields a 5% improvement over random sampling. Finally, we introduce Nemotron-ClimbLab, a filtered 1.2-trillion-token corpus with 20 clusters as a research playground, and Nemotron-ClimbMix, a compact yet powerful 400-billion-token dataset designed for efficient pre-training that delivers superior performance under an equal token budget. We analyze the final data mixture, elucidating the characteristics of an optimal data mixture. Our data is available at: <https://research.nvidia.com/labs/lpr/climb/>
- **Venue**: NeurIPS 2025 Datasets and Benchmarks track.
- **Killer Figs**:
  - **Fig 1**: 1B model pretrained from scratch on ClimbMix vs. ClimbLab-random, Nemotron-CC-HQ, SmolLM, DCLM-baseline, FineWeb-Edu at 32B to 400B tokens, averaged over 12 benchmarks. ClimbMix is on top at every budget; this is the plot that justifies using it as a pretraining corpus.
  - **Fig 4**: Framework overview. Top: embed, cluster, merge. Bottom: sample mixtures, train proxy LMs, fit predictor, prune, repeat.
  - **Fig 3**: t-SNE of the mixture configs sampled at each iteration, showing the search space contracting from iter 1 to iter 3.
  - **Fig 8**: Final ClimbMix cluster weights across iterations. From-scratch pretraining wants a far more balanced mixture than continued pretraining does.
- **Core idea**:
  - Web corpora have no domain labels, and even labeled corpora (The Pile) have a nonlinear, non-obvious mapping from mixture to downstream performance. CLIMB removes the need for labels by **clustering in embedding space**, then treats the mixture as a search problem over cluster weights.
  - > CLIMB consists of three key steps: (1) embedding and clustering large-scale datasets, (2) constructing mixture-performance pairs by sampling and pruning data mixtures and training proxy models, and (3) fitting a predictor.
  - The differentiator vs. RegMix/DoReMi is **iteration**: rather than sampling mixtures uniformly once and fitting one predictor, alternate between sampling near the predictor's current best guesses and refitting the predictor. RegMix is the special case of a single iteration.
- **Methodology**:
  - **Phase 1, data preprocessing** (Sec 2.1):
    - Embed every document with an embedding model (stella_en_400M_v5).
    - K-means (FAISS) into a deliberately large $K_{init} = 1000$ fine-grained clusters.
    - Prune low-quality clusters at the cluster level using fastText quality classifiers (overall quality, educational value, informational value, advertisement score, 1 to 5), each distilled from 1M Nemotron-340B annotations. Loose threshold of 3.0 leaves $K_{pruned} = 240$.
    - Merge remaining clusters by centroid distance (Euclidean threshold 1.5) into $K_{enhanced} \approx 20$ super-clusters. $K_{enhanced} < K_{pruned} < K_{init}$. Merging exists to shrink the number of "domains" the mixture search has to weight.
  - **Phase 2, iterative bootstrapping** (Sec 2.2):
    - Framed as bi-level optimization: find mixture weights $\alpha$ on the simplex that minimize validation loss of the model trained under $\alpha$ (Eq. 1). Training a model per $\alpha$ is prohibitive, so replace $\ell(\alpha, \omega)$ with a predictor $f_\theta(\alpha)$ fit on (mixture, performance) pairs under a sampling budget $C$ (Eq. 2). $C$ is the proxy training compute.
    - Solve by coordinate descent alternating two subroutines (Eqs. 3 and 4):
      1. **Configuration sampling**: score every unsampled config with the current predictor, take the top-$N$, randomly pick $M$ of them (exploit vs. explore), train proxies on those, add to the sample set.
      2. **(Weak) predictor fitting**: refit the predictor on all samples so far, current and past iterations.
    - Iteration 1 is initialized by sampling from a Dirichlet parameterized by each cluster's token count, biased toward sparse weights. Dirichlet init beats random init in ablation.
    - Predictor is LightGBM with L1/L2 regularization, max depth 4, at least 5 samples per leaf, early stopping. Any regressor works in principle.
    - Default budget is 3 iterations with 64 / 32 / 16 searches (4:2:1), 112 proxy runs total. Proxy is 350M (62M also works, 5x cheaper, same important clusters). A proxy run is about 45 H100-hours; the 1B target on 400B tokens is about 6,400.
    - Final mixture is the argmax of the last predictor.
  - **Objective**: validation accuracy on PIQA, ARC-E, HellaSwag only. Gains transfer to the other 9 benchmarks, which they read as evidence the search captures general reasoning rather than overfitting the three.
- **Results (high level)**:
  - Continued pretraining for 40B tokens on Nemotron-CC + smollm-corpus (21 super-clusters, 800B tokens): CLIMB beats Random, DoReMi, and RegMix at both 350M and 1B (60.41 vs. 59.37 for RegMix at 1B, avg of 6 tasks).
  - Scaled to 400B tokens, the 950M CLIMB model beats Llama-3.2-1B by 2.0 points average on 12 benchmarks.
  - Domain-targeted search (MMLU STEM / Humanities / Social Sciences) improves monotonically across iterations and beats a Best@N baseline that searches with the target-size model.
  - Ablations: more search compute keeps helping (150% and 200% budgets); 4:2:1 allocation beats both a "fat" 6:1 and a "tall" 2:2:1:1 tree; number of clusters is fairly flat around 15 to 30.
  - Final weights are sparse and concentrated on a few clusters (C8, C9, C18, C19 for general reasoning) that are both **relevant** to the target and **diverse** among themselves. Different domains pick different clusters.
- **ClimbLab and ClimbMix** (Sec 6):
  - Apply CLIMB-clustering to Nemotron-CC + smollm-corpus, filter and reorganize into 20 clusters: **ClimbLab**, 1.2T tokens, released as a research playground for mixture work.
  - Run CLIMB-search on those clusters and extract a 400B-token subset under the found mixture: **ClimbMix**.
  - > since the experiments here are conducted under a pre-training-from-scratch setting, a more balanced cluster distribution is required compared to continuous pre-training.
  - Cluster topics (Table 4) are GPT-4o summaries of 100 sampled docs each; e.g. C17 is "Role-Playing, Problem Solving, Mathematics, Algorithms" and C20 is "Python, Code".
- **Limitations** (Appendix A, theirs):
  - Proxy training is still non-negligible compute; they suggest distillation or zero-shot evaluation to cut it further.
  - Domain evaluation uses MMLU's coarse categories only, so no evidence yet for real-world domains like finance or healthcare.
- **Our opinion: downsides for true active / continual learning** ([[papers-continual-learning]], [[papers-open-ended-learning]]):
  - **Static mixture, moving optimum.** One $\alpha$ for the whole run, yet the paper's own Sec 6 shows the optimum shifts with model state (balanced from scratch, concentrated when continuing). A continual learner needs $\alpha(t \mid \text{state})$, as in [Automated Curriculum Learning for Neural Networks](https://arxiv.org/abs/1704.03003) (Graves et al., 2017) or ADO/ODM above.
  - **External target, not the learner's signal.** Optimizes proxy accuracy on three fixed benchmarks known in advance; active learning selects by the current model's uncertainty or progress, and open-ended agents have no such eval to point at.
  - **Too costly to re-run.** About 5k H100-hours of proxy search vs. 6.4k for the target run: fine once offline, not a loop the learner can close as it changes.
  - **Coarse, learner-agnostic control.** ~20 weights, uniform sampling within a cluster, clusters from a frozen external embedder. No per-example novelty or redundancy.
  - **Mixture is not schedule.** Forgetting and interference are about ordering and replay, which CLIMB scopes out; repetition is likewise out of scope.
  - **Search is biased toward sparse mixtures.** Candidate $\alpha$ are deliberately sampled sparse (App. D.8), so the result puts most weight on 4 of ~20 clusters and near-zero on the rest, i.e. whole regions of the corpus are effectively dropped. Sensible when the target is fixed and known; for a learner whose future tasks are unknown it forfeits coverage, and the paper's own from-scratch result (needs a balanced mixture) shows the sparse answer fails once the model cannot lean on prior knowledge. Quality filtering by a fixed LLM-judge prior has the same flavor.
