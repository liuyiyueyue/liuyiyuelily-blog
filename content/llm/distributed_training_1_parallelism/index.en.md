---
title: "Distributed Training: Parallelism Strategies"
date: 2025-12-01
tags: ["llm", "distributed-training", "parallelism"]
---


### Why Parallelism?
As training datasets grow, training takes longer.

At the same time, models are getting larger, and some no longer fit on a single device without running out of memory.

Distributed training addresses these scaling limits by partitioning computation across multiple devices or nodes.

This is usually done through sharding or parallelism strategies, which mainly fall into two categories: **data parallelism**, which partitions data, and **model parallelism**, which partitions the model.

This post gives a high-level overview of both. Later posts will examine each strategy in more detail.


### Common Parallelism Strategies
**Data parallelism (DP)** runs different subsets of a batch on different GPUs.
- Common approaches evolve from [parameter-server style **DP** to distributed style **DDP**](../distributed_trainining_2_ddp/), and then to memory-optimized methods such as **ZeRO and FSDP**
- DP sometimes also refers to the number of nodes, data-parallel degrees, or replica count


**Model parallelism** splits the model itself, including weights, optimizer states, and gradients, across GPUs.
- **Pipeline parallelism (PP)** runs different layers of the model on different GPUs. (GPipe)
- **Tensor parallelism (TP)** breaks up the math for a single operation such as a matrix multiplication within a layer. (GShard)
- **Mixture-of-Experts parallelism (EP)** processes each example with only a subset of the experts in each layer.
- **Sequence parallelism (SP)** splits along the sequence length.


### Multi-dimension Parallelism
All these parallelism strategies can be used together and form a multi-dimension parallelism. Here are some examples:

| Training Framework         | Dimensions                                    |
| -------------------------- | --------------------------------------------- |
| PyTorch DDP                | 1D (DP)                                       |
| Megatron-LM                | 2–3D (DP + TP + PP)                           |
| DeepSpeed                  | 2–4D (DP + TP + PP + ZeRO/optimizer sharding) |
| PaLM / GPT-4 class systems | 4–5D (DP + TP + PP + SP + EP)                 |


### Optimization Steps
A typical development workflow for distributed training looks like this:

1. First, optimize single-GPU performance and maximize memory utilization as much as possible.

2. If the training dataset grows larger, use data parallelism to accelerate training.

3. For larger models, if you run into out-of-memory issues, consider enabling `ZeRO` or `FSDP`.

4. If the model scale increases further, consider model parallel strategies such as pipeline parallelism or tensor parallelism.

5. For extremely large models, you may need to combine multiple distributed strategies, including pipeline parallelism, tensor parallelism, data parallelism, and `ZeRO`.
