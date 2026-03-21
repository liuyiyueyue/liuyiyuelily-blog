---
title: "LLM Inference: Disaggregated Inference"
date: 2023-12-15
tags: ["llm", "inference", "optimization"]
---


### What Is Disaggregated Inference (DI)?

In LLM inference, the prefill and decode stages have different compute, memory, and bandwidth requirements. Prefill is typically compute-intensive, while decode is memory-bandwidth-intensive. Having both prefill and decode on the same node lead to interference. A prefill stage on a long prompt occupies the GPU and delays the decode stage for other requests. Hence, separating prefill and decode onto separate processes or hosts can improve performance. [^1]

### Architecture

The DI system includes a connector/scheduler that transfers the KV Cache between prefill nodes and decode nodes via NVLink/NVSwitch (intra-node) or GPUDirect RDMA (inter-node)

[^1]: vLLM Project. "Disaggregated Prefill V1". GitHub Pull Request #10502. <https://github.com/vllm-project/vllm/pull/10502>
