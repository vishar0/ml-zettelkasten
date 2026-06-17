# Generalist Agents

- **Created**: 2026-05-05
- **Last Updated**: 2026-05-07
- **Status**: `In Progress`
- **Related**:
  - [[papers-rl]]
  - [[papers-open-ended-learning]]
  - [[papers-world-models]]

---

## Reading List

### Core Lineage

- [[papers-generative-decision-making]] Decision Transformer: Reinforcement Learning via Sequence Modeling - [paper](https://arxiv.org/abs/2106.01345)
- [[papers-open-ended-learning]] XLand: Open-Ended Learning Leads to Generally Capable Agents - [paper](https://arxiv.org/abs/2107.12808), [blog](https://deepmind.google/discover/blog/generally-capable-agents-emerge-from-open-ended-play/)
- [x] [2022] Gato: A Generalist Agent - [paper](https://arxiv.org/abs/2205.06175), [blog](https://deepmind.google/blog/a-generalist-agent/)
- [x] [2024] SIMA: Scaling Instructable Agents Across Many Simulated Worlds - [paper](https://arxiv.org/abs/2404.10179), [blog](https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/)
- [x] [2025] SIMA 2: A Generalist Embodied Agent for Virtual Worlds - [paper](https://arxiv.org/abs/2512.04797), [blog](https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/)

### Minecraft / Open-World Agents

- [ ] [2022] Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos - [paper](https://arxiv.org/abs/2206.11795), [blog](https://openai.com/index/vpt/)
- [ ] [2022] MineDojo: Building Open-Ended Embodied Agents with Internet-Scale Knowledge - [paper](https://arxiv.org/abs/2206.08853), [website](https://minedojo.org/)
- [ ] [2023] Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents - [paper](https://arxiv.org/abs/2302.01560), [code](https://github.com/CraftJarvis/MC-Planner)
- [ ] [2023] Voyager: An Open-Ended Embodied Agent with Large Language Models - [paper](https://arxiv.org/abs/2305.16291), [website/code](https://voyager.minedojo.org/)
- [ ] [2023] STEVE-1: A Generative Model for Text-to-Behavior in Minecraft - [paper](https://arxiv.org/abs/2306.00937)
- [ ] [2023] JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models - [paper](https://arxiv.org/abs/2311.05997), [website](https://craftjarvis.org/)

### DeepMind Interactive Agents Team

- [ ] [2020] Imitating Interactive Intelligence - [paper](https://arxiv.org/abs/2012.05672)
- [ ] [2021] Creating Multimodal Interactive Agents with Imitation and Self-Supervised Learning - [paper](https://arxiv.org/abs/2112.03763)
- [ ] [2022] Improving Multimodal Interactive Agents with Reinforcement Learning from Human Feedback - [paper](https://arxiv.org/abs/2211.11602), [blog](https://deepmind.google/blog/building-interactive-agents-in-video-game-worlds/)

### Optional / Adjacent

- [ ] [2023] Generative Agents: Interactive Simulacra of Human Behavior - [paper](https://arxiv.org/abs/2304.03442) — simulation/social-agent architecture, not central to embodied control.

---

## [2022] Gato: A Generalist Agent

- **Date**: 2026-05-05
- **Arxiv**: <https://arxiv.org/abs/2205.06175>
- **Blog**: <https://deepmind.google/blog/a-generalist-agent/>
- **Paperpile**: <https://app.paperpile.com/view/?id=0bc59bb4-6186-4ae1-867c-f0abce0d94b4>

---

- **Abstract**:
  - > Inspired by progress in large-scale language modeling, we apply a similar approach towards building a single generalist agent beyond the realm of text outputs. The agent, which we refer to as Gato, works as a **multi-modal, multi-task, multi-embodiment generalist policy**. **The same network with the same weights can play Atari, caption images, chat, stack blocks with a real robot arm and much more, deciding based on its context whether to output text, joint torques, button presses, or other tokens**. In this report we describe the model and the data, and document the current capabilities of Gato.
- **Intro**:
  - > There are significant benefits to using a single neural sequence model across all tasks. It reduces the need for hand crafting policy models with appropriate inductive biases for each domain. It increases the amount and diversity of training data since the sequence model can ingest any data that can be serialized into a flat sequence. Furthermore, its performance continues to improve even at the frontier of data, compute and model scale (Kaplan et al., 2020; Hoffmann et al., 2022). Historically, generic models that are better at leveraging computation have also tended to overtake more specialized domain-specific approaches (Sutton, 2019), eventually.
  - Gato is a single transformer policy trained across text, vision, simulated control, Atari, DeepMind Lab, BabyAI, Procgen, Meta-World, and real robot block-stacking tasks.
  - The key claim is not that Gato is best at every task, but that a single set of weights can act across many modalities, tasks, and embodiments when all inputs and outputs are serialized as tokens.
  - It is intentionally close to the language-model recipe: train a large decoder-only sequence model on a broad token stream, then condition behavior on context.
  - The paper frames Gato as an operating point constrained by real-time robotics: they use a 1.18B parameter model because larger models were not yet practical for real robot control latency.
  - The agent is trained offline by supervised imitation/behavior cloning, not online RL. The authors explicitly leave offline/online RL versions as future work.
- **Model**:
  - **Tokenization**:
    - Text: SentencePiece with 32k subword vocabulary.
    - Images: non-overlapping $16 \times 16$ patches in raster order, with a small ResNet block used to embed patches.
    - Discrete values/actions: flattened integer sequences in the range $[0, 1024)$.
    - Continuous observations/actions: flattened floats, mu-law encoded, discretized into 1024 bins, then shifted into the token range $[32000, 33024)$.
  - **Sequence format**:
    - Text tokens keep raw order.
    - Image patches are raster ordered.
    - Agent timesteps are serialized as observation tokens, separator, then action tokens.
    - Agent episodes are serialized in temporal order.
  - **Architecture**:
    - Decoder-only transformer.
    - 1.18B parameters, 24 layers, embedding size 2048, feed-forward hidden size 8196.
    - Training context length is 1024 tokens.
  - **Loss**:
    - Autoregressive next-token prediction with masking.
    - Text tokens and logged agent actions are prediction targets.
    - Image tokens and non-text observations are inputs but not predicted; their loss contribution is masked out.
  - **Prompt conditioning**:
    - For 25% of training sequences, the model prepends a prompt from an episode generated by the same source agent on the same task.
    - Prompts are used instead of explicit task identifiers.
    - During evaluation, control tasks are prompted with a successful demonstration by default.
- **Deployment as a policy**:
  - The prompt/demonstration is tokenized into the initial context.
  - The environment emits an observation; Gato tokenizes and appends it.
  - Gato samples the action vector autoregressively, one token at a time.
  - The action tokens are decoded into the environment-specific action format, executed, and the loop repeats.
  - The model sees prior observations and actions inside a 1024-token window; Transformer-XL memory is used at deployment even though it was not used during training.
- **Data**:
  - Gato is trained on 604 tasks total.
  - Control data: 596 tasks, 63M episodes, roughly 1.5T tokens, 85.3% of training sample weight.
  - Vision/language data: 14.7% of sample weight.
  - Main control domains include DeepMind Lab, Atari, BabyAI, DM Control, Meta-World, Procgen, RGB Stacking in sim and real robot settings, Modular RL, MPG, and Playroom.
  - Control trajectories come from specialist near-SOTA RL agents and are filtered to episodes with returns at least 80% of expert return.
  - Vision/language datasets include MassiveText, M3W, ALIGN, MS-COCO captions, Conceptual Captions, LTIP, OKVQA, and VQAv2.
- **Results**:
  - In simulated control, the pretrained model gets above 50% of expert score on more than 450 out of 604 tasks.
  - On Atari, it reaches average-human-or-better score on 23 games and over twice human score on 11 games.
  - On BabyAI, it exceeds 80% expert score for nearly all levels; BossLevel reaches 75%.
  - On Meta-World, it exceeds 50% expert score on 44/45 trained tasks, 80% on 35 tasks, and 90% on 3 tasks.
  - On canonical DM Control from state, it exceeds 50% expert score on 21/30 tasks and 80% on 18 tasks.
  - On real robot RGB Stacking Skill Generalization, Gato averages 50.2% success across held-out object triplets, comparable to the BC-IMP baseline at 49%.
- **Scaling and adaptation**:
  - The paper compares 79M, 364M, and 1.18B parameter variants and finds consistent improvement with scale at fixed token count.
  - For held-out OOD tasks, the authors fine-tune rather than rely on in-context learning, because demonstrations often exceed the short 1024-token context.
  - Fine-tuning with all-data pretraining beats same-domain-only and scratch on several OOD tasks, but not uniformly; Atari Boxing shows no benefit from pretraining.
  - In robot stacking, the 1.18B model adapts better than smaller variants with limited fine-tuning data.
  - For a new real robot blue-on-green stacking task, fine-tuned Gato reaches 60% success, while a BC baseline trained from scratch reaches 0.5%.
- **Limitations**:
  - Gato depends on action-labeled control data, and there is no web-scale equivalent for control comparable to text/image corpora.
  - The 1024-token context is a major bottleneck, especially for image-based environments where one observation can consume hundreds of tokens.
  - Prompt-based in-context learning on new environments did not clearly improve over prompt-less evaluation in early experiments.
  - Pure behavior cloning inherits the quality and coverage of the data-generating experts; specialist Atari agents still outperform the generalist model.
  - The system can inherit harms from vision-language data and adds physical safety concerns when deployed on real robots.
- **Takeaway**:
  - Gato is best read as an existence proof for a unified action-generating sequence model across modalities and embodiments.
  - The important move is turning agent behavior into next-token prediction over observations, text, and actions, then relying on scale and data diversity rather than domain-specific policy architectures.
  - Its weaknesses point directly at the next problems for generalist agents: longer context, better control data, online/offline RL beyond imitation, and stronger out-of-distribution adaptation.

## [2024] SIMA: Scaling Instructable Agents Across Many Simulated Worlds

- **Date**: 2026-05-06
- **Arxiv**: <https://arxiv.org/abs/2404.10179>
- **Blog**: <https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/>

---

- **Abstract**:
  - SIMA studies generalist embodied agents that follow open-ended natural-language instructions in 3D virtual environments.
  - The agent interface is deliberately human-like: **image observations and language instructions in, keyboard-and-mouse actions out**.
  - The paper positions virtual worlds as a scalable training ground for embodied intelligence because they are visually rich, interactive, cheap to reset, and safer than physical robotics.
  - The long-term ambition is an instructable agent that can do anything a human can do in a simulated 3D environment, though this paper reports early progress on short-horizon tasks.
- **Thesis**:
  - SIMA shifts the generalist-agent question from "can one model serialize many modalities and actions?" to "can one embodied policy follow language instructions across many rich 3D worlds through a shared human interface?"
  - The main abstraction is not a universal token stream, as in Gato, but a universal **agent-environment interface**: screen pixels, natural language, keyboard, and mouse.
  - The paper is less about maximal task heterogeneity and more about **language-conditioned transfer across worlds**.
- **Design choices**:
  - Uses rich, visually complex, open-ended 3D games and research environments rather than small fixed RL benchmarks.
  - Treats environments as asynchronous: the game continues while the agent acts.
  - Uses the same keyboard-and-mouse controls available to humans, instead of privileged APIs or custom action spaces.
  - Focuses on following language instructions, not maximizing game score or producing plausible ambient behavior.
  - Uses free-form natural language instructions, rather than a fixed command grammar.
- **Environments**:
  - Commercial games include Goat Simulator 3, Hydroneer, No Man's Sky, Satisfactory, Teardown, Valheim, and Wobbly Life.
  - Research environments include Construction Lab, Playhouse, ProcTHOR, and WorldLab.
  - The paper reports quantitative results on a subset of environments where data collection and evaluation infrastructure were mature enough.
- **Data**:
  - Trained primarily with behavior cloning from human gameplay.
  - Data includes image observations, language instructions or dialogue, keyboard-and-mouse actions, and annotations of behavior or success.
  - Collection modes include single-player free play with post-hoc annotations and two-player setter-solver interactions where one human instructs another.
  - The authors filter and weight data across environments and collection sources to manage quality and balance.
- **Agent**:
  - Inputs are recent visual observations plus the natural-language instruction.
  - Outputs are keyboard-and-mouse actions, often as short action sequences rather than single isolated actions.
  - The architecture builds on prior DeepMind interactive-agent work, combining pretrained visual/language components with transformers trained on SIMA data.
  - Pretrained components include SPARC-style image-text representations and Phenaki-style video prediction representations.
  - The policy is trained with behavior cloning and an auxiliary goal-completion prediction objective.
  - At inference, the paper uses classifier-free guidance to strengthen language conditioning by comparing action logits with and without the instruction.
- **Evaluation**:
  - Evaluation is hard because commercial games do not provide a universal task-success API.
  - Some research environments provide explicit success/failure signals and distractor tests.
  - Some game evaluations use OCR or environment-specific task-completion cues.
  - Human evaluation is used where automatic success detection is not reliable.
  - A central evaluation concern is whether the agent is actually following the instruction, rather than merely exploiting common environmental affordances.
- **Results**:
  - SIMA demonstrates positive transfer across environments: the multi-environment agent generally beats environment-specialized agents in aggregate.
  - Removing pretrained visual/language representations hurts performance, suggesting that internet-scale pretraining helps ground behavior in these worlds.
  - Removing language performs poorly, supporting the claim that the tasks measure instruction following rather than just generic game navigation.
  - Zero-shot held-out-environment results are promising for generic skills, but still limited.
  - Performance remains far below human level on harder commercial-game tasks, especially those requiring precise spatial reasoning, long-horizon tool use, combat, or construction.
- **Relation to Gato**:
  - Gato is the broader sequence-modeling existence proof: many modalities, many embodiments, many domains, one tokenized transformer policy.
  - SIMA narrows the embodiment to a human computer-control interface, but makes the environments richer and the instruction-following problem more central.
  - Gato asks whether a single model can emit many kinds of action tokens. SIMA asks whether one embodied agent can use the same action interface to act across many 3D worlds.
  - Gato includes language and vision tasks, but its control tasks are mostly benchmark-style and prompt/demo conditioned. SIMA makes language the direct task interface.
  - Both are mostly imitation-learning systems, but SIMA puts more pressure on data collection, grounding, evaluation, and cross-environment transfer.
- **Takeaway**:
  - SIMA is the natural next slide after Gato if the talk wants to move from "generalist policy as token sequence model" to "generalist embodied agent as language-conditioned computer user."
  - The interesting conceptual jump is from modality unification to interface unification.
  - The paper is also a reminder that generalist agents are limited less by transformer mechanics alone and more by environment coverage, instruction data, success evaluation, and long-horizon grounding.

## [2025] SIMA 2: A Generalist Embodied Agent for Virtual Worlds

- **Date**: 2026-05-07
- **Arxiv**: <https://arxiv.org/abs/2512.04797>
- **Blog**: <https://deepmind.google/blog/sima-2-an-agent-that-plays-reasons-and-learns-with-you-in-virtual-3d-worlds/>

---

- **Abstract**:
  - > We introduce SIMA 2, a generalist embodied agent that understands and acts in a wide variety of 3D virtual worlds. Built upon a Gemini foundation model, SIMA 2 represents a significant step toward active, goal-directed interaction within an embodied environment. Unlike prior work (e.g., SIMA 1) limited to simple language commands, SIMA 2 acts as an interactive partner, capable of reasoning about high-level goals, conversing with the user, and handling complex instructions given through language and images.
  - > Across a diverse portfolio of games, SIMA 2 substantially closes the gap with human performance and demonstrates robust generalization to previously unseen environments, all while retaining the base model's core reasoning capabilities. Furthermore, we demonstrate a capacity for open-ended self-improvement: by leveraging Gemini to generate tasks and provide rewards, SIMA 2 can autonomously learn new skills from scratch in a new environment. This work validates a path toward creating versatile and continuously learning agents for both virtual and, eventually, physical worlds.
- **Thesis**:
  - SIMA 2 moves the SIMA line from a specialized language-conditioned action policy to a Gemini-based vision-language-action agent for virtual worlds.
  - The key jump from SIMA 1 is not just more environments or more human data. The central change is putting Gemini at the core so the agent can reason, dialogue, handle richer prompts, and still act through a human-like game interface.
  - In relation to Gato, SIMA 2 partially returns to the foundation-model/unified-interface framing: vision, language, reasoning, dialogue, and actions are closer to one model interface. The caveat is that actions are structured text parsed into keyboard/mouse commands, not Gato-style discretized action tokens.
- **Agent-environment interface**:
  - Input is a stream of 720p RGB video frames plus recent interaction history, natural-language inputs, and the agent's prior internal reasoning/responses.
  - Output is structured text that can contain internal reasoning, dialogue, and action chunks.
  - The action space emulates a standard human computer interface: 96 keyboard keys, mouse clicks, and discretized relative mouse movement.
  - The agent does not receive privileged game state. It acts from screen pixels and language, using keyboard/mouse controls.
  - This makes the interface comparable to SIMA 1, but the model interface is different: SIMA 1 used a policy head over actions, while SIMA 2 routes actions through Gemini's text-generation interface.
- **Architecture**:
  - SIMA 2 is a VLA model built from a Gemini Flash-Lite checkpoint fine-tuned on 3D virtual-world data.
  - The public paper does not specify model size, training budget, or exact action text schema.
  - Baseline Gemini models without embodied fine-tuning perform poorly at acting, despite prompt engineering; the paper reports 3.2% success for Flash-Lite and 7.0% for Pro on programmatic evaluations.
  - This supports the point that embodied competence is not simply an emergent property of internet-scale vision/language pretraining. It has to be explicitly trained.
- **Training data**:
  - The training mixture combines gameplay data with Gemini pretraining/non-gameplay data to retain the base model's reasoning, dialogue, vision understanding, and promptability.
  - Human data forms most of the gameplay training data by volume and includes RGB frames, text instructions/annotations, and keyboard/mouse actions.
  - Data collection includes single-person gameplay with post-hoc annotation and two-person setter-solver annotation, where one human instructs and another controls the game.
  - The pipeline includes quality filtering, trajectory preprocessing, converting gameplay into shorter instruction-aligned spans, and synthetic labeling from Gemini.
- **Bridge data**:
  - Human gameplay does not naturally contain internal reasoning and dialogue, so SIMA 2 adds Gemini-generated bridge data.
  - Bridge data interleaves task instructions, visual frames, actions, internal reasoning, and dialogue in a format consistent with the agent's eventual interface.
  - This is the mechanism for using language as the interface between Gemini's reasoning and embodied action.
  - It teaches the agent to connect high-level user intent and internal reasoning to low-level actions, rather than only imitating direct action labels.
- **Reinforcement learning**:
  - After SFT, SIMA 2 is further trained with online RL from verifiable rewards.
  - A verifiable task is an initial game state, a text instruction, and a verification function.
  - Rewards come from embodied task completion or correct answers to environment-grounded questions; some tasks add shaped rewards for instruction following and controllability.
  - This RL phase is limited to training environments and excludes held-out environments such as ASKA and MineDojo.
  - Important wording: this is not "self-play." The relevant phrase is self-generated experience or online RL from agent rollouts.
- **Self-improvement**:
  - The self-improvement loop uses Gemini as a task setter and Gemini as a reward model.
  - The task setter proposes achievable tasks from the current environment state and can be steered toward weak skills or likely learning progress.
  - The reward model scores video trajectories against the task, creating a dataset of scored self-generated experience.
  - Training on this self-generated experience improves the SIMA 2 agent.
  - In ASKA, the self-improved agent progresses farther than the initial SIMA 2 agent, despite ASKA being held out from the original SIMA 2 training.
  - In Genie 3, self-improvement on generated urban environments transfers to held-out natural environments, suggesting a loop between world models and improving agents.
- **Environments and evaluation**:
  - Training environments include Construction Lab, Playhouse, WorldLab, Goat Simulator 3, Hydroneer, No Man's Sky, Satisfactory, Space Engineers, Valheim, and Wobbly Life.
  - Held-out evaluations include ASKA and MineDojo quantitatively, and The Gunk plus Genie 3 qualitatively.
  - Evaluation uses ground-truth success where available, programmatic/OCR/pixel/action heuristics where possible, and human raters when automatic checks are unreliable.
  - The evaluation suite is more challenging than SIMA 1, with more programmatic tasks, stricter success persistence, constraints on post-completion actions, and sequential task chains.
- **Results**:
  - SIMA 2 substantially improves over SIMA 1 on training environments and across skill categories.
  - It nearly closes the gap to human performance in some categories, especially interaction and object management, but remains weaker on resource gathering and combat.
  - It generalizes to held-out environments: ASKA and MineDojo results show meaningful improvement over SIMA 1, and MineDojo performance covers 26 out of 50 task categories.
  - Qualitatively, it can progress through the early portion of The Gunk and navigate photorealistic Genie 3 environments, despite training only on research and video-game worlds.
  - Fine-tuning on embodied action data preserves much of Gemini's general capability, with only modest regressions reported on code/math/STEM benchmarks.
- **Relation to SIMA 1**:
  - SIMA 1 is a modular action policy: screen/language in, keyboard/mouse actions out.
  - SIMA 2 is a Gemini-based VLA: screen/language/history in, structured text out, then parsed into reasoning, dialogue, and keyboard/mouse action chunks.
  - SIMA 1 follows short, direct instructions. SIMA 2 can handle higher-level goals, dialogue, internal reasoning, multi-modal prompts, and more complex instructions.
  - SIMA 1 evaluates cross-world instruction-following; SIMA 2 adds self-improvement and stronger held-out generalization.
- **Relation to Gato**:
  - Gato is a token-in token-out generalist policy trained by behavior cloning over text, images, observations, and action tokens.
  - SIMA 2 is closer to modern multimodal LLM/VLA systems: it inherits a pretrained Gemini model and fine-tunes it for embodied action.
  - The commonality is the bet on a unified model interface that can absorb heterogeneous modalities.
  - The difference is action representation: Gato directly tokenizes actions; SIMA 2 emits structured text that is deterministically parsed into low-level keyboard/mouse commands.
- **Limitations**:
  - Many important details are sparse: model size, training budget, data scale, exact structured action schema, and RL algorithm/scoring details.
  - The self-improvement result is bounded, not strong recursive self-improvement. It depends on Gemini as task setter/reward model, curated environments, and a training loop built by the researchers.
  - Reward and verification remain bottlenecks. Where ground-truth or programmatic checks are unavailable, the setup relies on human raters or Gemini-scored trajectories.
  - Generalization is real but still within virtual 3D worlds and human-computer interfaces; physical transfer remains aspirational.
  - Long-horizon complex tasks, short interaction memory/context, precise low-level control, and robust visual understanding remain open challenges.
- **Takeaway**:
  - SIMA 2 is the strongest "Gato-like" idea in the talk after modern multimodal foundation models: instead of training a generalist token policy from scratch, start with Gemini and fine-tune it into an embodied VLA.
  - The most important concept is the loop: bootstrap from human data, use language as the bridge between reasoning and action, then improve from self-generated experience using task/reward models.
  - For research strategy, the paper pushes evals beyond "can the agent solve this task?" toward "does this task expose a bottleneck that survives foundation-model scale?"
