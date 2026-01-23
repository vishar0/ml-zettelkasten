# Gated Linear Networks (GLN)

- **Created**: 2026-01-23
- **Last Updated**: 2026-01-23
- **Status**: `In Progress`

---

- [ ] [2016] [Foundational] The Forget-me-not Process - [paper](https://papers.nips.cc/paper/6055-the-forget-me-not-process)
- [ ] [2017] [Core] Online Learning with Gated Linear Networks - [paper](https://arxiv.org/abs/1712.01897)
- [ ] [2019] [Core] Gated Linear Networks - [paper](https://arxiv.org/abs/1910.01526)
- [ ] [2020] [Extension] Gaussian Gated Linear Networks - [paper](https://arxiv.org/abs/2006.05964)
- [ ] [2020] [Extension] Online Learning in Contextual Bandits using Gated Linear Networks - [paper](https://arxiv.org/abs/2002.11611)
- [ ] [2020] [Extension] A Combinatorial Perspective on Transfer Learning - [paper](https://arxiv.org/abs/2010.12268)
- [ ] [2021] [Neuro] A Rapid and Efficient Learning Rule for Biological Neural Circuits - [paper](https://www.biorxiv.org/content/10.1101/2021.03.10.434756)

---

## [2016] [Foundational] The Forget-me-not Process - [paper](https://papers.nips.cc/paper/6055-the-forget-me-not-process)

- **Date**: 2026-01-23
- **Paper**: <https://papers.nips.cc/paper_files/paper/2016/hash/f26dab9bf6a137c3b6782e562794c2f2-Abstract.html>
- **Paperpile**: <https://app.paperpile.com/view/?id=aa1e26d4-0626-48c5-bce9-a97d3f1cb8e5>
- **Assistant**: <https://gemini.google.com/share/34b6cc27a5ad>

---

- **Abstract**:
  - > We  introduce  the  Forget-me-not  Process,  an  efficient,  non-parametric  meta- algorithm for online probabilistic sequence prediction for piecewise stationary, repeating sources. Our method works by taking a Bayesian approach to partition- ing a stream of data into postulated task-specific segments, while simultaneously building a model for each task. We provide regret guarantees with respect to piece- wise stationary data sources under the logarithmic loss, and validate the method empirically across a range of sequence prediction and task identification problems.
- **Intro**:
  - > A key limitation of these aforementioned techniques is that they can perform poorly when there exist multiple segments of data that are similarly distributed. For example, consider data generated according to the schedule depicted in Figure 1. For all these methods, once a change-point occurs, the base (stationary) model is invoked from scratch, even if the task repeats, which is clearly undesirable in many situations of interest.
  - Baseline for this problem: $O(n^2)$ - checking every single possible start and end position to detect change points.
  - **Partition Tree Weighting (PTW) Meta-Algorithm** reduces this to $O(n \log n)$ by contructing a binary tree: <https://gemini.google.com/share/34b6cc27a5ad>
    - But PTW as no long-term memory and can't detect tasks/patterns/distributions that repeat. If "Task A" happens, then "Task B", then "Task A" happens again, PTW treats the second "Task A" as a completely new, stranger event. It re-learns it from scratch.
  - **Forget-me-not Process**: Introduces the ability to avoid having to relearn repeated tasks, while still maintaining essentially the same theoretical performance guarantees as PTW on piecewise stationary sources.
- TODO

## [2017]
