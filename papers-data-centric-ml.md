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
- **My takeaways**:
  - The recipe is generic: cluster, score clusters, search mixture weights with a proxy plus a regressor, iterate. Nothing is specific to text beyond the embedding model, which is why it could plausibly apply to a multimodal or Atari-plus-text mixture.
  - The mixture is optimized for a 1B AR model and a reasoning objective. There is no evidence it is optimal for other architectures, objectives, or scales below 350M; the transfer across proxy sizes (62M vs 350M) is the only scale-transfer evidence.
  - ClimbMix as a released corpus bakes in the from-scratch mixture. ClimbLab keeps the cluster labels and is the right artifact if one wants to re-search the mixture for a different target.
- **Our opinion: downsides in the context of true active / continual learning** ([[papers-continual-learning]], [[papers-open-ended-learning]]):
  - **The mixture is a constant, but the optimal data distribution is a function of the learner's state.** CLIMB outputs one $\alpha$ and holds it fixed for the whole run. The paper's own Sec 6 observation, that from-scratch training wants a balanced mixture while continued pretraining wants a concentrated one, is direct evidence that the optimum moves as the model learns. They respond by running the search once per regime rather than making $\alpha$ a policy over training. A continual learner has no "regime"; it needs $\alpha(t \mid \text{model state})$. Graves et al. 2017 ([paper](https://arxiv.org/abs/1704.03003)) already framed this as a bandit over tasks driven by learning progress; ADO ([paper](https://arxiv.org/abs/2410.11820)) and ODM ([paper](https://arxiv.org/abs/2312.02406)) do it online for LM pretraining by fitting per-domain loss curves during the run.
  - **Selection is driven by an external, fixed target, not by the learner's own signal.** The objective is proxy-model accuracy on three benchmark validation sets known in advance. Active learning selects what the current model finds uncertain or surprising; CLIMB selects what a different, smaller model found useful on yesterday's benchmarks. This presupposes that the target task is known and stationary, which is the opposite of the open-ended setting where tasks arrive over time and the agent's own experience (interaction logs, frames) has no curated eval to optimize toward. It also risks a curriculum overfit to today's benchmark suite.
  - **The search is too expensive to re-run as the model changes.** 112 proxy runs at about 45 H100-hours each is roughly 5k GPU-hours, close to the 6.4k of the 400B target run. That is affordable once, offline. It is not a mechanism a learner can invoke continuously, and the predictor cannot be trusted far from the proxies it was fit on: the only scale-transfer evidence is 62M vs. 350M proxies choosing the same top clusters.
  - **Control is coarse: 20 weights, uniform within cluster.** The mixture has about 20 degrees of freedom, and inside a cluster documents are sampled uniformly regardless of whether the model has mastered them. There is no per-example or per-batch notion of novelty, learning progress, or redundancy, which is the granularity at which active learning operates. The clusters themselves come from a frozen external embedder (stella), not from the learner's representation, so "similar" is defined by a different model than the one being trained.
  - **Mixture is not schedule.** CL's central problems are forgetting and interference between domains, which are properties of ordering, interleaving, and revisiting. CLIMB explicitly scopes itself to simultaneous mixing and leaves sequencing to curriculum work. Skill-it ([paper](https://arxiv.org/abs/2307.14430)) is the closest attempt to model dependencies between data "skills" and order them; CLIMB has no analogue. Nothing here says how to fold new clusters in later, or how much old data to replay when you do.
  - **Quality is a fixed prior distilled from an LLM judge.** Cluster pruning uses fastText scores for educational value, informativeness, and ads, distilled from Nemotron-340B ratings. That is a strong human-flavored prior about what good text looks like, applied before any learning happens. For an agent that has to learn from its own experience, there is no such oracle; quality has to be defined by what improves the learner, which is what learning-progress signals try to capture.
  - **The sparsity bias trades away coverage.** Sampling is deliberately biased toward sparse $\alpha$, so a few clusters dominate. That is fine when the target is fixed and known. For a learner that must stay ready for unknown future tasks, diversity is the hedge, and the paper's own from-scratch result shows the sparse solution was wrong for that setting.
  - **Repetition is out of scope.** ClimbMix is a one-pass 400B extract. How many times to see a token, and which tokens deserve a second pass, is exactly the knob a replay-based continual learner turns, and the knob diffusion training cares about ([[papers-scaling-laws]]).
  - **What survives.** The outer loop is sound and reusable: cluster, evaluate mixtures cheaply, fit a predictor, iterate. To make it a continual-learning method the pieces to replace are the objective (external benchmark to the learner's own loss or learning progress, as in DoReMi's excess loss ([paper](https://arxiv.org/abs/2305.10429)) or Data Mixing Laws' fitted loss predictors ([paper](https://arxiv.org/abs/2403.16952))), the proxy (offline small model to the live model's recent gradients or losses), and the output (a constant $\alpha$ to a policy updated every few thousand steps). ClimbLab, with its cluster labels intact, is the right substrate for that experiment; ClimbMix is the answer to a question we are not asking.
