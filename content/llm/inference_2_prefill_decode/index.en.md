---
title: "LLM Inference: Prefill and Decode"
date: 2023-12-14
tags: ["llm", "inference", "optimization"]
---


### What Are Prefill and Decode?

In LLM inference, prefill and decode are two distinct phases of text generation: 
- Prefill phase is the first forward pass through the model, where the **entire prompt** (all user input tokens) is processed. It builds the **initial KV cache**.
- Decode phase produces **one new token at a time** (or a few tokens in parallel if speculative/parallel decoding is used). At each step, it uses the KV from the prefill, only calculates the attention on the newest token, and extend the KV cache.

(The backward pass only appears in training, not inference...!)

### Prefill and Decode Have Different Performance Characteristics

- Prefill processes the entire prompt in parallel, so the GPU spends most of its time on large matrix multiplications. This makes it **compute-bound**.
- Decode generates one token at a time, so each step does relatively little compute but repeatedly reads model weights and KV cache from memory. This makes it **memory-bound**.

![Prefill and decode latency breakdown](prefill-decode-latency.jpg)

