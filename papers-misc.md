# Misc

- **Created**: 2026-02-20
- **Last Updated**: 2026-06-29
- **Status**: `In Progress`

---

- [x] [2024] Atari-GPT: Benchmarking Multimodal Large Language Models as Low-Level Policies in Atari Games - [paper](https://arxiv.org/abs/2408.15950)
- [x] [2025] OpenApps: Simulating Environment Variations to Measure UI-Agent Reliability - [paper](https://arxiv.org/abs/2511.20766)

---

## [2024] Atari-GPT: Benchmarking Multimodal Large Language Models as Low-Level Policies in Atari Games

- **Date**: 2026-02-20
- **Arxiv**: <https://arxiv.org/abs/2408.15950>
- **Paperpile**: <https://app.paperpile.com/view/?id=3ad1ee4d-d108-48e0-964e-a183da31d7d5>

---

- **Abstract**:
  - > Recent  advancements  in  large  language  models  (LLMs) have expanded their capabilities beyond traditional text-based tasks to multimodal domains, integrating visual, auditory, and textual data. While multimodal LLMs have been extensively explored  for  high-level  planning  in  domains  like  robotics and  games,  their  potential  as  low-level  controllers  remains largely untapped. In this paper, we introduce a novel bench- mark aimed at testing the emergent capabilities of multimodal LLMs as low-level policies in Atari games. Unlike traditional reinforcement  learning  (RL)  methods  that  require  training for each new environment and reward function specification, these LLMs utilize pre-existing multimodal knowledge to di- rectly  engage  with  game  environments.  Our  study  assesses the performances of multiple multimodal LLMs against tra- ditional RL agents, human players, and random agents, fo- cusing on their ability to understand and interact with com- plex visual scenes and formulate strategic responses. **Our results show that these multimodal LLMs are not yet capable of being zero-shot low-level policies. Furthermore, we see that this is, in part, due to their visual and spatial reasoning**. Ad- ditional results and videos are available on our project web- page: <https://dev1nw.github.io/atari-gpt/>.

---

## [2025] OpenApps: Simulating Environment Variations to Measure UI-Agent Reliability

- **Date**: 2026-06-29
- **Arxiv**: <https://arxiv.org/abs/2511.20766>
- **Authors**: Karen Ullrich, Jingtong Su, Claudia Shi, Arjun Subramonian, Amir Bar, Ivan Evtimov, Nikolaos Tsilivis, Randall Balestriero, Julia Kempe, Mark Ibrahim (Meta / FAIR)
- **Code/Website**: <https://facebookresearch.github.io/OpenApps/>

---

- **Abstract**:
  - > Reliability is key to realizing the promise of autonomous UI-Agents, multimodal agents that directly interact with apps in the same manner as humans, as users must be able to trust an agent to complete a given task. Current evaluations rely on fixed environments, often clones of existing apps, which are limited in that they can only shed light on whether or how often an agent can complete a task within a specific environment. When deployed however, agents are likely to encounter variations in app design and content that can affect an agent's ability to complete a task. To address this blind spot of measuring agent reliability across app variations, we develop OpenApps, a light-weight open-source ecosystem with six apps (messenger, calendar, maps, etc.) that are configurable in appearance and content. OpenApps requires just a single CPU to run, enabling easy generation and deployment of thousands of versions of each app. Specifically, we run more than 10,000 independent evaluations to study reliability across seven leading multimodal agents. We find that while standard reliability within a fixed app is relatively stable, reliability can vary drastically when measured across app variations. Task success rates for many agents can fluctuate by more than 50% across app variations. For example, Kimi-VL-3B's average success across all tasks fluctuates from 63% to just 4% across app versions. We also find agent behaviors such as looping or hallucinating actions can differ drastically depending on the environment configuration. These initial findings highlight the importance of measuring reliability along this new dimension of app variations.
- **Notes**:
  - Core point: a single fixed-app success number is a misleading way to report agent reliability. Real deployment means many app variants, so you should measure reliability *across* variations, not within one clone.
  - The contribution is mostly an eval artifact + methodology, not a new model or algorithm: OpenApps is a lightweight, single-CPU ecosystem of six everyday apps (messenger, calendar, maps, etc.) whose appearance and content can be procedurally varied into thousands of versions.
  - Headline finding: within a single app, success is fairly stable, but across variations it swings wildly — >50% for many agents (e.g. Kimi-VL-3B: 63% → 4% on the same tasks).
  - Failure modes (looping, hallucinated actions) are environment-dependent, i.e. they're triggered by surface design/content rather than being intrinsic to the agent.
  - Takeaway for evals: report reliability as a distribution over environment variations, not a point estimate — the variation axis is the thing fixed benchmarks miss.
