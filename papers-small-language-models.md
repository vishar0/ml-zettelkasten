# Small Language Models

- **Created**: 2026-05-15
- **Last Updated**: 2026-06-10
- **Status**: `In Progress`
- **Related**:
  - [[papers-reversal-curse]]
  - [[papers-memory-architectures]]

---

- [x] [2025] FLM: Towards Fundamental Language Models: Does Linguistic Competence Scale with Model Size? - [paper](https://arxiv.org/abs/2509.02225)
- [x] [2026] IKP: Incompressible Knowledge Probes: Estimating Black-Box LLM Parameter Counts via Factual Capacity - [paper](https://arxiv.org/abs/2604.24827)
- [x] [2025] [Position] Small Language Models are the Future of Agentic AI - [paper](https://arxiv.org/abs/2506.02153)
- [ ] [2025] BabyLM: Findings of the BabyLM Challenge: Sample-Efficient Pretraining on Developmentally Plausible Corpora - [paper](https://arxiv.org/abs/2504.08165)
- [ ] [2025] BabyVLM: Data-Efficient Pretraining of VLMs Inspired by Infant Learning - [paper](https://arxiv.org/abs/2504.09426)
- [ ] [2026] EgoBabyVLM: Benchmarking Cross-Modal Learning from Naturalistic Egocentric Video Data - [paper](https://arxiv.org/abs/2605.19130)
- [ ] [2022] [GDM] Transformers Generalize Differently from Information Stored in Context vs in Weights - [paper](https://arxiv.org/abs/2210.05675)

## [2025] Towards Fundamental Language Models: Does Linguistic Competence Scale with Model Size?

- **Date**: 2026-05-15
- **Arxiv**: <https://arxiv.org/abs/2509.02225>

---

- **Abstract**:
  - > Large Language Models offer impressive language capabilities but suffer from well-known limitations, including hallucinations, biases, privacy concerns, and high computational costs. These issues are largely driven by the combination of linguistic competence and factual memorization within a single monolithic model. This paper introduces and empirically supports the Fundamental Language Model (FLM) paradigm, which advocates for smaller, linguistically competent models that offload factual retrieval to external tools. We evaluate models ranging from 135M to 32B parameters across three dimensions: linguistic competence, external factual knowledge, and internal factual knowledge. Our findings reveal that **while both linguistic competence and factual knowledge improve with scale, internal factual knowledge grows significantly faster, suggesting that model size is more closely tied to memorization than to core language ability. These results support a modular approach to language modeling, where compact, linguistically proficient models serve as the foundation for tool-augmented systems**. The FLM paradigm offers a path toward more efficient, interpretable, and sustainable NLP solutions.
- **Killer Figs**:
  - **Fig 2**: Main thesis figure. Regression slopes against log model size show internal factual knowledge rising more steeply than linguistic competence.
  - **Fig 1**: Raw score-vs-size scatter behind the argument; useful for checking model-family variation and whether the trend is clean.
- **Core idea**:
  - The paper proposes **Fundamental Language Models (FLMs)**: smaller models that preserve core language competence while delegating factual knowledge to external sources.
  - The motivating split is between:
    - **Linguistic competence**: lexical, grammatical, and semantic ability.
    - **Factual knowledge**: facts stored in weights or extracted from provided context.
  - This is basically a language-model version of the fluid/crystallized distinction:
    - Fluid-ish: reusable language processing and task execution competence.
    - Crystallized: memorized factual content.
- **Why this matters**:
  - Current LLMs entangle language ability and factual memory in the same parameter budget.
  - The authors argue that many LLM problems (hallucination, stale knowledge, privacy leakage, bias, compute cost) come from trying to store too much factual knowledge in the model.
  - FLMs are meant to be modular: compact language engines plus retrieval/tools for up-to-date facts.
- **Evaluation setup**:
  - Models: SmolLM2, Qwen2.5, Llama 3, OLMo-2, Falcon3, Gemma-2, Yi-1.5, ranging from 135M to 32B parameters.
  - Zero-shot evaluation via LM Evaluation Harness.
  - **Linguistic competence**:
    - Lexical: WiC.
    - Grammatical: BLiMP.
    - Semantic: RTE, MNLI, QQP.
  - **External factual knowledge**: answering/reasoning from provided context via LAMBADA, BoolQ, COPA, MultiRC, ReCoRD.
  - **Internal factual knowledge**: memorized facts via TriviaQA and TruthfulQA.
- **Main results**:
  - Linguistic competence improves with scale, but much less steeply than internal factual knowledge.
  - Qwen2.5-32B has the best linguistic score, but Qwen2.5-7B and even Qwen2.5-3B are already competitive with larger models.
  - External factual knowledge does not monotonically reward size: Gemma-2-9B and Falcon3-10B beat Qwen2.5-32B on the paper's aggregate EFK score.
  - Internal factual knowledge is the most size-linked category: OLMo-2-32B is best, and the regression against log model size has much higher explanatory power for IFK than for linguistic competence.
  - Reported slope vs log(size): internal factual knowledge 0.059, linguistic competence 0.029. Median split large-vs-small gap: IFK 39.50%, linguistic competence 18.29%, EFK 8.47%.
- **Takeaway**:
  - Strong language competence may not require extremely large parameter counts.
  - A lot of what model scale buys is factual storage, not necessarily proportionally more reusable intelligence.
  - This supports an architecture where the model is a compact linguistic/reasoning interface and external systems provide facts, retrieval, memory, and tool outputs.
- **Caveats / questions**:
  - The paper treats linguistic competence as close to the reusable core, but "fluid intelligence" is broader than language competence.
  - Semantic tasks can still require world knowledge, so the linguistic/factual boundary is leaky.
  - The evaluation is English-centric and benchmark-centric; it does not actually build and test a deployed FLM + retrieval/tool system.

## [2026] Incompressible Knowledge Probes: Estimating Black-Box LLM Parameter Counts via Factual Capacity

- **Date**: 2026-05-15
- **Arxiv**: <https://arxiv.org/abs/2604.24827>
- **Code**: <https://github.com/19PINE-AI/ikp>
- **Website**: <https://01.me/research/ikp>

---

- **Abstract**:
  - > Closed-source frontier labs do not disclose parameter counts, and the standard alternative -- inference economics -- carries 2×+ uncertainty from hardware, batching, and serving-stack assumptions external to the model. We exploit a tighter intrinsic bound: storing F facts requires at least F/(bits per parameter) weights, so measuring how much a model **knows** lower-bounds how many parameters it **has**. We introduce **Incompressible Knowledge Probes (IKPs)**, a benchmark of 1,400 factual questions spanning 7 tiers of obscurity, designed to isolate knowledge that cannot be derived by reasoning or compressed by architectural improvements.
    >
    > We calibrate a log-linear mapping from IKP accuracy to parameter count on 89 open-weight models (135M-1,600B) spanning 19 vendors, achieving R2=0.917; leave-one-out cross-validation confirms generalization (median fold error 1.59×, 68.5% within 2× and 87.6% within 3×). For Mixture-of-Experts models, total parameters predict knowledge (R2=0.79) far better than active parameters (R2=0.51). We evaluate 188 models from 27 vendors and estimate effective knowledge capacity for all major proprietary frontier models; for heavily safety-tuned models the estimates are lower bounds, since refusal policy can hide tens of percentage points of "refused but known" capacity.
    >
    > The widely-reported saturation of reasoning benchmarks does not imply the end of scaling. Procedural capability compresses under the "Densing Law," but across 96 dated open-weight models the IKP time coefficient is −0.0010/month (95% CI [−0.0031,+0.0008]) -- indistinguishable from zero, and rejecting the Densing prediction of +0.0117/month at p<10−15. Factual capacity continues to scale log-linearly with parameters across generations and across vendors.
- **Killer Figs**:
  - **Fig 1**: Main calibration curve. IKP accuracy grows log-linearly with parameter count and gives the core evidence for knowledge-as-capacity.
  - **Fig 2**: MoE sanity check. Total parameters predict factual knowledge better than active parameters.
  - **Fig 6**: Densing Law falsification. At fixed parameter count, IKP does not improve over time the way procedural benchmarks are claimed to.
  - **Fig 4**: Tier behavior. Shows T1-T2 saturation, T3-T5 discrimination, and T6-T7 frontier-only separation.
- **Core idea**:
  - Factual knowledge is treated as an **incompressible storage problem**.
  - If a model knows a rare arbitrary fact, the answer could not have been derived from general reasoning; some information about that fact must be stored in the weights.
  - Therefore, measuring rare-fact recall gives a black-box estimate of a model's effective factual capacity, which can be mapped to an approximate parameter count.
- **Fluid intelligence angle**:
  - This paper draws the sharpest version of the fluid/crystallized boundary:
    - **Procedural capability**: reasoning, parsing, instruction following, tool use; can get denser over time.
    - **Factual capacity**: arbitrary stored facts; bounded by information content and parameter budget.
  - The Densing Law may hold for procedural capabilities, but the authors argue it should not hold for rare factual memory.
  - In this framing, scaling has not stopped mattering; it has just stopped mattering as much for benchmarks that mostly measure compressible procedural skill.
- **Densing Law**:
  - Claim: model capability per parameter is increasing rapidly over time, so newer small models can match older much-larger models on common benchmarks.
  - Example shape: a 2026 7B model matching a 2023 70B model on MMLU/HELM-style reasoning benchmarks.
  - The paper's critique is that this mixes two resources:
    - **Compressible procedures**: better architectures/training can pack reasoning, parsing, and instruction-following into fewer parameters.
    - **Incompressible facts**: arbitrary rare facts still require storage bits and should not become freely denser just because training recipes improve.
  - IKP is designed to test the second resource, so the authors expect no Densing-style time trend at fixed parameter count.
- **Methodology**:
  - Dataset: 1,400 probes, 200 per tier, across seven obscurity tiers T1-T7.
  - Probe sources:
    - LLM-generated candidates mostly for easy tiers T1-T2.
    - Corpus-grounded Wikidata probes for founding years and entity attributes.
    - DBLP / arXiv researcher probes asking for subfield plus a verifiable artifact.
  - Tiers are assigned empirically with a landmark ladder from Qwen 2.5 0.5B through Gemini 3.1 Pro.
  - Scoring penalizes confident wrong answers more than refusals:
    - Correct: +1.0.
    - Weak researcher answer: +0.5.
    - Refusal: 0.
    - Wrong: -1.0.
  - Calibration fits aggregate penalized accuracy against `log10(parameter count in billions)` on open-weight models.
- **Main results**:
  - Calibration on 89 open-weight models from 19 vendors reaches `R^2 = 0.917`.
  - Leave-one-out cross-validation gives median fold error of 1.59x; 68.5% of models are within 2x and 87.6% within 3x.
  - Each 10x increase in parameters adds about 14.7 percentage points of IKP score.
  - For MoE models, **total parameters** predict factual capacity better than active parameters: `R^2 = 0.79` vs `0.51`.
  - Across 96 dated open-weight models, the IKP time coefficient is approximately zero, rejecting the Densing Law prediction for factual capacity.
- **Other interesting pieces**:
  - The paper estimates effective knowledge capacity for proprietary frontier models by projecting their IKP scores onto the open-model calibration curve.
  - It introduces a knowledge-fingerprinting idea: compare rare-fact overlap and shared wrong answers to identify weight-sharing siblings, post-training lineages, or full retrains.
  - Refusal policy matters a lot: safety-tuned models may know facts but refuse to answer, making IKP estimates lower bounds.
  - T1-T2 are mostly saturated; T3-T5 carry much of the discrimination; T6-T7 distinguish only the strongest frontier models.
- **Takeaway**:
  - This is the complement to the FLM paper.
  - FLM says: maybe small models retain much of the reusable linguistic competence.
  - IKP says: stored factual knowledge remains capacity-limited and continues to scale with total parameters.
  - Together: large models may be big partly because they are giant crystallized-knowledge stores, not because all of that size is necessary for fluid competence.
- **Caveats / questions**:
  - The parameter-count estimates for closed models are really **effective factual-capacity estimates**, not direct measurements of physical parameter count.
  - Training data quality, deduplication, domain coverage, and refusal behavior can all move IKP scores without changing parameter count.
  - The probe set itself may become contaminated once released, although the authors argue the tiering procedure can regenerate probes.
  - The paper assumes rare factual recall is mostly parametric storage; retrieval-augmented or tool-using models would need to be evaluated carefully to avoid measuring external memory.
  - How would IKP behave on an explicit FLM + retrieval system? It should have high factual performance without large parametric factual capacity, unless the evaluation forbids retrieval.
  - Does "incompressible" apply only to arbitrary facts, or also to skills that require memorizing many irreducible cases?

## [2025] Small Language Models are the Future of Agentic AI

- **Date**: 2026-05-17
- **Arxiv**: <https://arxiv.org/abs/2506.02153>
- **Website**: <https://research.nvidia.com/labs/lpr/slm-agents>

---

- **Abstract**:
  - > Large language models (LLMs) are often praised for exhibiting near-human performance on a wide range of tasks and valued for their ability to hold a general conversation. The rise of agentic AI systems is, however, ushering in a mass of applications in which language models perform a small number of specialized tasks repetitively and with little variation.
    >
    > Here we lay out the position that small language models (SLMs) are sufficiently powerful, inherently more suitable, and necessarily more economical for many invocations in agentic systems, and are therefore the future of agentic AI. Our argumentation is grounded in the current level of capabilities exhibited by SLMs, the common architectures of agentic systems, and the economy of LM deployment. We further argue that in situations where general-purpose conversational abilities are essential, heterogeneous agentic systems (i.e., agents invoking multiple different models) are the natural choice. We discuss the potential barriers for the adoption of SLMs in agentic systems and outline a general LLM-to-SLM agent conversion algorithm.
    >
    > Our position, formulated as a value statement, highlights the significance of the operational and economic impact even a partial shift from LLMs to SLMs is to have on the AI agent industry. We aim to stimulate the discussion on the effective use of AI resources and hope to advance the efforts to lower the costs of AI of the present day. Calling for both contributions to and critique of our position, we commit to publishing all such correspondence at this https URL.
- **Killer Figs**:
  - **Fig 1**: Useful systems framing. Contrasts language-model agency (LM orchestrates tool calls) with code agency (controller code orchestrates and LMs serve narrower roles).
- **What kind of paper is this?**:
  - Position paper, not a new benchmark or method paper.
  - The claim is architectural/economic: agentic systems should become **SLM-first**, using LLMs selectively rather than by default.
- **Core argument**:
  - Agent workflows decompose broad tasks into many narrow, repeated, structured LM calls.
  - Many of those calls do not need broad conversational/generalist ability; they need reliable instruction following, formatting, tool-call generation, extraction, classification, summarization, or code-interface behavior.
  - SLMs are often good enough for these narrow roles and are cheaper, lower-latency, easier to fine-tune, easier to deploy locally, and easier to specialize.
  - When general open-ended reasoning or conversation is needed, use a heterogeneous system: default to SLMs and route selectively to larger models.
- **Alternative views**:
  - **LLM generalists may always win on language understanding**:
    - The strongest counterargument is scaling-law intuition: larger models have broader language/world understanding, so even narrow tasks may benefit from the large model's general semantic competence.
    - The paper's rebuttal: agent tasks are decomposed into simpler subtasks, SLMs can be fine-tuned cheaply for those subtasks, and SLM test-time compute is cheaper.
  - **Centralized LLM inference may remain cheaper in practice**:
    - Generalist LLM endpoints may have better batching, utilization, and infra economics than many specialized SLM endpoints.
    - The paper mostly concedes this is case-specific; its bet is that inference systems and deployment tooling will make SLM specialization cheaper over time.
  - **LLM-first and SLM-first worlds may both be viable**:
    - The LLM-first world has a huge head start from infrastructure, products, and developer habits.
    - The paper argues this is inertia, not a fundamental technical reason.
- **Barriers to adoption**:
  - Large sunk investment in centralized LLM infrastructure.
  - SLM development still often optimizes for generalist LLM-style benchmarks instead of agentic utility.
  - SLMs have weaker mindshare/marketing despite being operationally attractive.
- **Why it belongs here**:
  - Complements FLM: if compact models retain enough reusable competence, they can be the default language interface for agents.
  - Complements IKP: if large models mainly buy lots of crystallized factual capacity, agents should externalize facts/memory/tools and reserve large models for genuinely hard cases.
  - This is the systems version of the fluid/crystallized split: keep fluid-ish control/interpretation in smaller models; move knowledge and actions into external tools, memory, APIs, and code.
- **LLM-to-SLM conversion recipe**:
  - Log agent calls and outcomes.
  - Curate and sanitize the data.
  - Cluster recurring task types.
  - Pick candidate SLMs for each task.
  - Fine-tune/distill specialized SLMs.
  - Route calls to SLMs by default, escalate to LLMs when needed, and iterate.
- **Caveats / questions**:
  - "SLMs are enough" depends heavily on reliability requirements, tail cases, and cost of mistakes.
  - Routing is the hard part: a cheap SLM-first system still needs to know when it is out of depth.
  - Centralized LLM inference may retain economic advantages for some workloads because of batching, utilization, and infrastructure maturity.
  - The paper is persuasive as a design direction, but it mostly argues from trends and examples rather than proving replacement rates empirically.
