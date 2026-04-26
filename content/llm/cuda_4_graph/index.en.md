---
title: "CUDA Graph"
date: 2022-11-14
tags: ["llm", "cuda"]
---

In vLLM, kernel launch overhead is much lower during decode than during prefill. The main reason is that **decode can use CUDA Graphs more easily**, while prefill usually cannot. In this blog, we will discuss CUDA graph and its usages in inference.

### What is CUDA Graph?

Normally, the CPU launches GPU kernels one by one. Each launch only costs a few to a few dozen microseconds, but LLM inference may require dozens or even hundreds of launches per token. Those small costs add up, and the GPU can end up waiting on the CPU.

CUDA Graph reduces that overhead by **capturing a fixed sequence of kernel launches once and replaying it later as a single graph**. Replay is much cheaper than launching every kernel individually, so it can remove most launch overhead.[^1] [^2] [^3] Fireworks gives a useful systems view here: modern GPUs have become so fast that CPU dispatch overhead can become a bottleneck for deep learning workloads, especially LLM decode, and they report a `2.3x` speedup on one LLaMA v2 7B inference workload with CUDA Graphs.[^4]

{{< figure src="./images/cuda_graph.png" caption="CUDA Graph captures a fixed sequence of GPU work and replays it with much lower launch overhead." align="center" >}}

### Why It Works for Decode

Decode generates one token at a time. The input shape is typically:

```text
batch_size x 1 x hidden_dim
```

That middle dimension stays `1`, so the execution pattern is mostly stable. The same kernels run in the same order with the same shapes, and only the data changes. This is exactly the case where CUDA Graph works well.

In practice, systems such as vLLM can capture graphs for several fixed batch sizes and replay the closest one at runtime, sometimes with padding. This makes CUDA Graph effective in the decode path.

### Why Prefill Is Hard

Prefill is different because **the input sequence length is not fixed**. Once sequence length changes, tensor shapes change, attention shapes change, and kernel launch configurations can change too. At that point, it is no longer just different data on the same graph. The graph structure itself changes.

CUDA Graph requires capture and replay to use the same execution graph with static shapes. That is the most direct reason prefill usually does not use CUDA Graph.


[^1]: NVIDIA CUDA Graphs documentation, no publication date listed. <https://docs.nvidia.com/dl-cuda-graph/latest/>
[^2]: NVIDIA Developer Blog, "Getting Started with CUDA Graphs," posted September 5, 2019. <https://developer.nvidia.com/blog/cuda-graphs/>
[^3]: PyTorch Blog, "Accelerating PyTorch with CUDA Graphs," posted October 26, 2021, updated November 15, 2024. <https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/>
[^4]: Fireworks AI Blog, "Speed, Python: Pick Two. How CUDA Graphs Enable Fast Python Code for Deep Learning," published August 29, 2023. <https://fireworks.ai/blog/speed-python-pick-two-how-cuda-graphs-enable-fast-python-code-for-deep-learning>
