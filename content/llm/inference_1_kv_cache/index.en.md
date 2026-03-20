---
title: "LLM Inference: KV Cache"
date: 2023-12-13
tags: ["llm", "inference", "optimization"]
---


### What is KV Cache?
In LLM **inference**, **KV cache** is used to improve performance by avoiding repeated computation of the attention **Key** and **Value** tensors for all previously generated tokens.

With KV Cache, when generating token `t+1`, the model only needs to compute `Q_{t+1}`, `K_{t+1}`, and `V_{t+1}`. The new `K_{t+1}` and `V_{t+1}` are then appended to the cache, and `Q_{t+1}` attends over the entire cached sequence of keys and values.

It is important to note that KV cache only applies to the **decoder**, together with **masked self-attention**. KV caching happens across multiple token-generation steps and only exists in the decoder, either in decoder-only models such as GPT or in the decoder portion of encoder-decoder models such as T5. Models like BERT are not generative, so they do not use KV cache.

### Visual Intuition
The diagrams below show the difference between regular masked self-attention and the KV-cache version used during autoregressive decoder stage. [^1]

Without KV Cache:
![Masked self-attention over the current decoder step](masked-self-attention-step.png)

![Masked self-attention over the full sequence](masked-self-attention-full.png)

With KV Cache:
![KV cache reusing previously computed keys and values](kv-cache-step.png)

### Why Is There a KV Cache but No Q Cache?
- In the decoder stage, each inference step only uses the **current query**. Once that step is finished, that query will not be reused in later steps, so there is no real benefit to caching `Q`.
- In contrast, each new decoder step needs access to the **current and all previous** keys and values. The `K` and `V` tensors computed in this step will be used again immediately in the next step, which is why caching them speeds up inference.

[^1]: LLM 推理优化之 KV Cache. SayHelloCode, Zhihu. <https://zhuanlan.zhihu.com/p/673923443>
