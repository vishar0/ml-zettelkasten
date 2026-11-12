# Robotics

- **Created**: 2025-11-10
- **Last Updated**: 2026-11-05
- **Status**: `In Progress`
- **Related**:
  - [[papers-rl]] — Policy learning from rewards and interaction.
  - [[papers-world-models]] — Learned dynamics for planning and policy improvement.

---

## Reading List

### Physical Intelligence

- Source: <https://www.pi.website/blog>, checked 2026-05-09.

- [ ] [2024] [PI] $\pi_0$: Our First Generalist Policy - [post](https://www.pi.website/blog/pi0), [paper](https://arxiv.org/abs/2410.24164)
- [ ] [2025] [PI] FAST: Efficient Robot Action Tokenization - [post](https://www.pi.website/research/fast), [paper](https://arxiv.org/abs/2501.09747), [huggingface](https://huggingface.co/physical-intelligence/fast)
- [ ] [2025] [PI] Open Sourcing $\pi_0$ - [post](https://www.pi.website/blog/openpi), [repo](https://github.com/Physical-Intelligence/openpi)
- [ ] [2025] [PI] Teaching Robots to Listen and Think Harder - [post](https://www.pi.website/research/hirobot), [paper](https://arxiv.org/abs/2502.19417)
- [ ] [2025] [PI] $\pi_{0.5}$: a VLA with Open-World Generalization - [post](https://www.pi.website/blog/pi05), [paper](https://www.pi.website/download/pi05.pdf)
- [ ] [2025] [PI] VLAs that Train Fast, Run Fast, and Generalize Better - [post](https://www.pi.website/research/knowledge_insulation), [paper](https://www.pi.website/download/pi05_KI.pdf)
- [ ] [2025] [PI] Real-Time Action Chunking with Large Models - [post](https://www.pi.website/research/real_time_chunking), [paper](https://arxiv.org/abs/2506.07339), [follow-up paper](https://arxiv.org/abs/2512.05964)
- [ ] [2025] [PI] $\pi^{*}_{0.6}$: a VLA that Learns from Experience - [post](https://www.pi.website/blog/pistar06), [paper](https://www.pi.website/download/pistar06.pdf)
- [ ] [2025] [PI] Emergence of Human to Robot Transfer in VLAs - [post](https://www.pi.website/research/human_to_robot), [paper](https://www.pi.website/download/human_to_robot.pdf)
- [ ] [2025] [PI] Moravec's Paradox and the Robot Olympics - [post](https://www.pi.website/blog/olympics)
- [ ] [2026] [PI] The Physical Intelligence Layer - [post](https://www.pi.website/blog/partner)
- [ ] [2026] [PI] VLAs with Long and Short-Term Memory - [post](https://www.pi.website/research/memory), [paper](https://www.pi.website/download/Mem.pdf)
- [ ] [2026] [PI] Precise Manipulation with Efficient Online RL - [post](https://www.pi.website/research/rlt), [paper](https://www.pi.website/download/rlt.pdf)
- [ ] [2026] [PI] $\pi_{0.7}$: a Steerable Model with Emergent Capabilities - [post](https://www.pi.website/blog/pi07), [paper](https://www.pi.website/download/pi07.pdf)

### Humanoid Motion Tracking and Differentiable Control

Train whole-body controllers to follow reference motions while obeying physical dynamics. These papers compare scaling model-free RL with learning through a differentiable dynamics model or the simulator itself.

- [ ] [2025] [NVIDIA] SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control - [paper](https://arxiv.org/abs/2511.07820), [project](https://nvlabs.github.io/GEAR-SONIC/) — scale PPO-based motion tracking across model capacity, motion data, and compute to learn one generalist humanoid controller with shared interfaces for downstream control.
- [ ] [2026] [QiayuanLiao] SuperMimic: Scaling Humanoid Whole-Body Control with One Differentiable World Model - [paper](https://arxiv.org/abs/2610.28480), [project](https://supermimic.github.io/) — fit shared robot dynamics from simulator transitions and train the tracking policy by backpropagating through short imagined rollouts, with a critic estimating longer-term rewards. See [[papers-world-models#Differentiable Dynamics for Control]] for the methodological connection.
- [ ] [2026] [WeylLu,YubeiChen] NablaMimic (∇Mimic): Ultra-Efficient Humanoid Control with Differentiable Simulation - [paper](https://sailab.ai/nablamimic/assets/nablamimic.pdf), [project](https://sailab.ai/nablamimic/) — train directly through simulator gradients using NablaEngine; median-relative per-sample clipping stabilizes contact gradients, and regression onto gradient-derived action targets improves the policy without a learned world model.

### RL Post-Training and Fleet Learning

Improve pretrained robot policies using deployment experience, combining offline data with autonomous trials and human interventions.

- [ ] [2026] [YiWang,JianlanLuo] Learning While Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies - [paper](https://arxiv.org/abs/2605.00416), [project](https://learning-while-deploying.github.io/) — offline-to-online RL post-training of a flow-based VLA policy from shared fleet experience; distributional implicit value learning estimates returns, and Q-learning via Adjoint Matching turns critic gradients into policy updates.

---

## [2024] [PI] $\pi_0$: Our First Generalist Policy

- **Date**: 2025-11-10
- **Paper**: <https://arxiv.org/abs/2410.24164>
- **Paperpile**: <https://app.paperpile.com/view/?id=125eea1a-fccc-4600-bc54-a1aa51524d09>
- **Blog**: <https://www.pi.website/blog/pi0>

---

- TODO
