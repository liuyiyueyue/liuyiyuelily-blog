---
title: "Distributed Training: ZeRO and FSDP"
date: 2025-12-15
tags: ["llm", "distributed-training", "zero", "fsdp"]
---

### Two Types of Memory

- We refer to the memory used by model parameters as **static memory**, meaning its lifetime is effectively unbounded and it must always occupy HBM. The optimizer states, model parameters, and usually gradients are static memory, since they must persist across training steps rather than being freed immediately after one operator finishes.
- We refer to activation memory as **dynamic memory**, meaning its lifetime is short and its memory can be reused. Activations have dynamic memory since they are created during forward pass and released or recomputed after backward.

{{< figure src="./images/static_dynamic_memory.jpg" caption="Static memory vs. dynamic memory in training." align="center" >}}


### ZeRo [^1]
In conventional data-parallel training, every machine still has to hold a full copy of the model state in memory, and that memory cost does not shrink as data parallelism scales out. As a result, memory often becomes the main bottleneck in data-parallel training. 

Microsoft proposed a highly influential algorithm called **Zero Redundancy Optimizer (ZeRO)** to address the memory limitations of data-parallel training, and later built the distributed training framework **DeepSpeed** on top of PyTorch around it. Its core idea is to partition model states evenly across GPUs, so the memory usage on each GPU becomes inversely proportional to the degree of data parallelism, while communication efficiency remains largely unaffected.

| ZeRO Stage | What’s Sharded |
| --- | --- |
| **Stage 1** | Optimizer states |
| **Stage 2** | + Gradients |
| **Stage 3** | + Model parameters |


### FSDP [^2] [^3]
Facebook introduced **FSDP (Fully Sharded Data Parallel)** as PyTorch’s counterpart to Microsoft’s **ZeRO** in DeepSpeed. FSDP can be viewed as an optimized version of **DDP** within PyTorch. It is still a form of data parallelism, but unlike DDP, FSDP uses **parameter sharding**. In other words, model parameters are partitioned across GPUs, whereas in DDP each GPU keeps a full copy of the parameters. This allows FSDP to achieve better training efficiency, both in terms of speed and GPU memory usage.

```python
fsdp_module = FullyShardedDataParallel(module)
```

{{< figure src="./images/fsdp-graph-2.png" caption="FSDP execution flow." align="center" >}}

**Constructor**

- Shard the model parameters and distribute the shards across ranks.

**Forward pass**

1. For each FSDP unit, run `all_gather` to collect parameter shards from all ranks, so each rank temporarily holds the full parameters for the current unit. This is also why FSDP still belongs to data parallelism: although the parameters are sharded for storage, computation still uses the original full parameters rather than computing directly on weight shards as in tensor parallelism.
2. Execute the forward computation.
3. Reshard the parameters, discard the portions that do not belong to the current rank, and free the corresponding memory.

**Backward pass**

1. For each FSDP unit, run `all_gather` again to collect parameter shards from all ranks.
2. Execute the backward computation.
3. Reshard the parameters again, discard the portions that do not belong to the current rank, and free the corresponding memory.
4. Run `reduce_scatter` to synchronize gradients across ranks.

**Optimizer updates**

- Each rank updates the local shard of gradients and parameters that it owns.

#### FSDP 2

略 ;-) 写累了

[^1]: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. arXiv, October 4, 2019. <https://arxiv.org/abs/1910.02054>
[^2]: Fully Sharded Data Parallel: Faster AI Training with Fewer GPUs. Meta Engineering, July 15, 2021. <https://engineering.fb.com/2021/07/15/open-source/fsdp/>
[^3]: PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel. arXiv, April 21, 2023. <https://arxiv.org/abs/2304.11277>
