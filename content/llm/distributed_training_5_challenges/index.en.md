---
title: "Distributed Training: Challenges"
date: 2026-04-14
tags: ["llm", "distributed-training"]
---

<!-- 我觉得还应该写的更细点。多一些技术点和我经历过的技术点-->

Large-scale training is not a single optimization problem. It is a coupled system spanning hardware reliability, network communication, numerical stability, data engineering, and post-training optimization. Each component is difficult in isolation; the practical challenge is that they must operate together for long-duration runs without throughput collapse or training divergence.

### Hardware

At small scale, hardware faults are infrequent enough to feel exceptional. At the scale of thousands or tens of thousands of GPUs, they become part of the steady-state operating environment. The system must therefore be designed under the assumption that some component will be degraded, disconnected, or silently producing incorrect results at any given time.

The most difficult failures are often not explicit crashes, but **Silent Data Corruption**. A partially degraded HBM region, an unstable link, or a kernel-level fault may occasionally inject a `NaN` or an anomalously large value into the computation. By the time monitoring reports a `NaN` loss, the root cause may be several layers removed from the visible failure. In practice, this is why production training stacks need aggressive fault detection, health checks, retry logic, and communication guards. Megatron-LM is a useful reference not only because of its parallelism support, but also because it contains a substantial amount of engineering for fault tolerance and async checkpointing at scale.

### Networking

Communication topology is equally important. MoE is a representative example: it scales parameter count efficiently, but introduces substantial `all-to-all` traffic. Sustaining high MFU requires combining tensor parallelism, pipeline parallelism, sequence parallelism, and expert parallelism in a topology-aware way. This depends on understanding locality across GPUs within a node, nodes within a switch domain, and links across racks or data centers, because the communication pattern that is acceptable at one level of the hierarchy may become a hard bottleneck at another.

### Numerical Stability

Distributed training is also constrained by numerical behavior. As model size increases, training increasingly relies on lower-precision formats to reduce memory footprint and improve throughput. The tradeoff is reduced numerical margin: gradients may overflow or underflow, and intermediate activations may drift into unstable ranges before the failure becomes externally visible.

This failure mode is one reason long training runs can remain apparently stable for a large fraction of training and then diverge abruptly. The loss curve may look well-behaved for hundreds of billions of tokens, after which some internal activations begin to spike and the run becomes unstable. Once that happens, debugging is difficult because the observed divergence may be the result of an interaction among low-precision arithmetic, fused kernels, optimizer state evolution, and distributed communication.

### Data

Data quality is another first-order constraint. For modern foundation models, the data pipeline is often comparable in complexity to the training pipeline itself. Web-scale raw data contains duplication, contamination, formatting noise, inconsistent distributions, and large volumes of low-value content. Scaling the model does not compensate for poor input quality; it often amplifies it.

In practice, this means the system depends on filtering, deduplication, quality scoring, mixture construction, and ongoing validation. The objective is not only to build a large corpus, but to maintain a distribution that is useful, stable, and consistent with the downstream training objective.

### Reinforcement Learning and Post-Training

Post-training adds another layer of systems complexity. In RLHF and related methods, optimization is no longer performed only on a static supervised dataset. The system must also run large-scale inference to generate rollouts, which are then scored and used for policy updates.

This creates both systems and optimization challenges. On the systems side, the policy model, value model, reward model, and reference model may all need to coexist within the same memory budget. On the optimization side, reinforcement learning is well known to be unstable. If the reward model contains exploitable flaws, the policy may optimize for reward without improving the intended behavior. This is the standard **reward hacking** failure mode.

Taken together, these issues explain why distributed training is not merely a question of fitting a model across more GPUs. The real problem is constructing a training system that remains efficient, numerically stable, and operationally robust at extreme scale.
