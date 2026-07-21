# Memory Architectures

- **Created**: 2026-07-02
- **Last Updated**: 2026-07-21
- **Status**: `Not Started`
- **Description**: Architectural mechanisms for storing and retrieving information beyond standard feedforward weights, including working, episodic, external, and parametric memory.
- **Related**:
  - [[papers-small-language-models]] — External memory expands capacity without scaling model weights.
  - [[papers-reversal-curse]] — Retrieval architecture may address asymmetric factual recall.

---

## Foundations: Memory-Augmented Networks (2014–2016)

The original external-memory wave: a neural controller learns to read/write a separate memory via differentiable (or supervised) addressing.

- [ ] [[papers-ml-fundamentals]] [2014] [AlexGraves,GregWayne] Neural Turing Machines - [paper](https://arxiv.org/abs/1410.5401)
- [ ] [2014] [FAIR] Memory Networks - [paper](https://arxiv.org/abs/1410.3916)
- [ ] [2015] [FAIR] End-To-End Memory Networks - [paper](https://arxiv.org/abs/1503.08895)
- [ ] [2016] [AlexGraves,GregWayne] DNC: Hybrid Computing Using a Neural Network with Dynamic External Memory - [paper](https://www.nature.com/articles/nature20101)

## Memory in Agents / Episodic Memory

- [ ] [2017] [GDM] Neural Episodic Control - [paper](https://arxiv.org/abs/1703.01988)
- [ ] [2018] [GregWayne] MERLIN: Unsupervised Predictive Memory in a Goal-Directed Agent - [paper](https://arxiv.org/abs/1803.10760)

## External Memory in Language Models

- [x] [2025] LMLM: Pre-training Limited Memory Language Models with Internal and External Knowledge - [paper](https://arxiv.org/abs/2505.15962)
- [ ] [2025] [McClelland] Thought Gestalt Model: Modeling Language as a Sequence of Thoughts - [paper](https://arxiv.org/abs/2512.25026)

---

## [2025] LMLM: Pre-training Limited Memory Language Models with Internal and External Knowledge

- **Date**: 2026-06-10
- **Arxiv**: <https://arxiv.org/abs/2505.15962>
- **Code**: <https://github.com/kilian-group/LMLM>

---

- **Abstract**:
  - > Neural language models are black-boxes--both linguistic patterns and factual knowledge are distributed across billions of opaque parameters. This entangled encoding makes it difficult to reliably inspect, verify, or update specific facts. We introduce Limited Memory Language Models (LMLM), a new class of language models that externalizes factual knowledge to external database during pre-training rather than memorizing them. Our pre-training approach strategically masks externally retrieved factual values from the training loss, thereby teaching the model to perform targeted lookups rather than relying on memorization in model weights. Our experiments demonstrate that LMLMs achieve competitive performance compared to significantly larger LLMs on standard benchmarks, while offering the advantages of explicit, editable, and verifiable knowledge bases.
- **Killer Figs**:
  - **Fig 2**: Pipeline figure. Shows the full LMLM loop: annotate entity facts, store triples in an external DB, train on lookup-augmented text with returned values masked from loss, then interleave generation with DB lookups at inference.
  - **Fig 4**: Main thesis figure. LMLM gets lower pretraining perplexity, much higher factual precision at similar NLU performance, and cleaner machine unlearning than weight-editing baselines.
  - **Fig 6**: Detailed TOFU unlearning result. Deleting forget-set DB entries makes LMLM forget targeted facts without retraining and without the retain-set damage seen in NPO-style unlearning.
- **Core idea**:
  - LMLM tries to make the FLM idea concrete at pretraining time: keep language competence in weights, but push entity-level factual memory into an external database.
  - The model is still a causal LM, but the pretraining corpus is rewritten to contain explicit lookup calls of the form `(entity, relation) -> value`.
  - Crucial trick: the retrieved factual value is visible in context but excluded from the training loss.
  - This means the model is trained to generate the lookup query and continue fluent text, but is not rewarded for memorizing the answer tokens themselves.
  - Slogan version: it is easier and more controllable to learn **how to look up facts** than to store all facts in parameters.
- **Why this matters**:
  - Standard LM pretraining entangles linguistic ability and factual memory in the same opaque weights.
  - That makes facts hard to inspect, update, delete, or verify without changing the whole model.
  - LMLM gives a route to models whose factual knowledge is editable by DB operations rather than by retraining or fragile weight-editing.
  - It is especially relevant to unlearning/compliance: deleting facts from a database is much cleaner than trying to make a neural model forget a narrow slice of its parameters.
  - Fits the small-model thesis: if factual storage can be externalized, smaller models may retain enough language/control ability while using external memory for long-tail facts.
- **Methodology**:
  - **Data annotation**:
    - Start with raw Wikipedia-style pretraining text.
    - Use GPT-4o to annotate a small seed corpus with inline lookup calls, e.g. `[dblookup('Beyonce Giselle Knowles-Carter', 'Birth Date') -> September 4, 1981]`.
    - Train a lightweight Annotator model, based on LLaMA-3.1-8B-Instruct, to imitate this annotation behavior at scale.
    - Use a Corrector model to filter noisy annotations. High-loss entity/relation calls are removed because they are often malformed, overly specific, unsupported, or not inferable from prior left-to-right context.
    - Run the final Annotator over the full pretraining corpus.
  - **Two outputs from annotation**:
    - External DB: extracted `(entity, relation, value)` triples.
    - Training corpus: original text interleaved with lookup calls.
  - **Lookup-token format**:
    - The annotated text is converted into special-token spans such as:
      - `Napoleon was born on <|db_start|> Napoleon <|sep|> Birth_Date <|db_retrieve|> August 15, 1769 <|db_end|> August 15, 1769.`
    - The model learns when to emit `<|db_start|>`, what entity/relation query to emit, and how to continue after the DB value is inserted.
  - **Training objective**:
    - Standard autoregressive next-token prediction over normal text and lookup-query tokens.
    - Retrieved value tokens and `<|db_end|>` are masked out of the loss.
    - This discourages parametric memorization of the factual value.
  - **Retrieval**:
    - The query is string-structured, but lookup is not pure exact string matching.
    - The implementation uses fuzzy matching with cosine similarity over `all-MiniLM-L6-v2` sentence embeddings, with a rejection threshold of `0.6`.
    - They also discuss prefix-tree constrained generation, but fuzzy embedding matching is the default reported setup.
  - **Models / scale**:
    - GPT-2 and LLaMA2-style decoder-only models trained from scratch.
    - 1024-token context, 8 epochs, mixed precision.
    - Main LLaMA2 variants: 176M and 382M parameters.
    - Database from annotated pretraining corpus: 54.6M knowledge triples.
- **Main results**:
  - **Perplexity**:
    - LMLM has consistently lower normalized evaluation perplexity than the standard counterpart during pretraining.
    - Interpretation: offloading factual values makes the modeling problem easier and frees capacity from memorizing long-tail facts.
  - **Factual precision**:
    - LMLM beats same-size standard models on FactScore, T-REx, and PopQA.
    - Example: LLaMA2-382M standard gets FactScore 14.0, T-REx 52.0, PopQA 22.7; LMLM gets 31.9, 58.1, 50.8.
    - The 382M LMLM approaches much larger off-the-shelf models on factual precision, especially compared with Pythia-1B and LLaMA2-7B in the paper's table.
  - **Unlearning**:
    - On TOFU, LMLM unlearning is implemented by deleting database entries for the forget set.
    - It reaches the desired forget-quality region while preserving model utility.
    - NPO-style weight unlearning improves forgetting but hurts model utility and retain-set answer quality.
  - **Evidence that facts are not stored internally**:
    - When database access is disabled, LMLM factual performance drops sharply.
    - Return-value token loss stays high under the masked objective, unlike standard SFT where loss on those tokens falls, consistent with memorization.
- **Takeaway**:
  - LMLM is not just RAG added after pretraining. It changes pretraining so the model is discouraged from internalizing certain facts in the first place.
  - The architecture separates:
    - **Internal memory**: language modeling, control flow, query formulation, local reasoning.
    - **External memory**: editable entity-level factual values.
  - This is a concrete systems answer to the FLM/IKP tension:
    - IKP says rare factual memory is capacity-limited.
    - LMLM says: then stop forcing all of it into weights; train the model to use an external store.
  - The big conceptual win is control: facts can be inspected, edited, deleted, and verified outside the neural weights.
- **Caveats / questions**:
  - Current scope is mostly entity-level atomic facts. It does not solve broader factual reasoning, procedures, causal knowledge, or multi-hop knowledge cleanly.
  - The annotation pipeline is expensive and depends on a strong teacher model plus filtering. This shifts complexity from model weights into data engineering.
  - Lookup quality depends on generated entity/relation strings and fuzzy embedding retrieval; failures can come from bad query generation, bad annotation, missing DB entries, or wrong nearest-neighbor matches.
  - The DB can contain noise because it is automatically extracted from the corpus.
  - Extra lookup tokens increase training and inference cost.
  - The approach assumes that the fact can be represented as a clean `(entity, relation, value)` triple. Many useful facts are not naturally this atomic.
  - Open question: how should a system decide which facts to externalize versus leave in weights? The paper's selective-offloading analysis suggests long-tail/specific facts benefit most.
  - Open question: how does this compose with conventional RAG? The authors argue the two are complementary: RAG retrieves broad document context, LMLM retrieves precise entity facts.
