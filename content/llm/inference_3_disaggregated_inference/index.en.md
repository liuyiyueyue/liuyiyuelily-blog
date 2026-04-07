---
title: "LLM Inference: Disaggregated Inference"
date: 2023-12-15
tags: ["llm", "inference", "optimization"]
---


### What Is Disaggregated Inference (DI)?

In LLM inference, the prefill and decode stages have different compute, memory, and bandwidth requirements. Prefill is typically compute-intensive, while decode is memory-bandwidth-intensive. Having both prefill and decode on the same node lead to **interference**. A prefill stage on a long prompt occupies the GPU and delays the decode stage for other requests. Hence, separating prefill and decode onto separate processes or hosts can improve performance. [^1] [^2] [^3] [^4]

{{< figure src="./images/prefill_decode_interference.png" caption="Prefill and decode interference when both stages share the same GPU resources v.s. disaggregated inference." align="center" >}}

### Architecture

The DI system includes a connector/scheduler that transfers the KV Cache between prefill nodes and decode nodes via NVLink/NVSwitch (intra-node) or GPUDirect RDMA (inter-node)

### Heterogeneous DI

Cerebras and AWS describe disaggregated inference by splitting compute-bound prefill and memory-bound decode: AWS Trainium builds the KV cache during prefill, and Cerebras CS-3 handles decode for high token output speed. [^5] NVIDIA’s Vera Rubin + Groq 3 LPX design is different: prefill remains on GPUs, and decode itself is split so GPUs run attention while LPX accelerates latency-sensitive FFN/MoE work. [^6] In short, Cerebras/AWS separates phases of inference, while NVIDIA/Groq separates sub-operations within the decode loop.

[^1]: DistServe: Disaggregating Prefill and Decoding for Goodput-optimized
Large Language Model Serving. June 6, 2024. <https://arxiv.org/pdf/2401.09670>
[^2]: vLLM Project. "Disaggregated Prefill V1". GitHub Pull Request #10502. <https://github.com/vllm-project/vllm/pull/10502>
[^3]: Disaggregated Inference: 18 Months Later. November 3, 2025 https://haoailab.com/blogs/distserve-retro/
[^4]: Disaggregated Inference at Scale with PyTorch & vLLM. September 12, 2025 https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/
[^5]: James Wang. Cerebras is coming to AWS. March 13, 2026. <https://www.cerebras.ai/blog/cerebras-is-coming-to-aws>
[^6]: Kyle Aubrey and Farshad Ghodsian. Inside NVIDIA Groq 3 LPX: The Low-Latency Inference Accelerator for the NVIDIA Vera Rubin Platform. March 16, 2026. <https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/>
