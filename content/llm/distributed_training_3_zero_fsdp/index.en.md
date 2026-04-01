---
title: "Distributed Training: ZeRO and FSDP"
date: 2025-12-15
tags: ["llm", "distributed-training", "zero", "fsdp"]
math: true
---

### ZeRo [^1]
In conventional data-parallel training, every machine still has to hold a full copy of the model state in memory, and that memory cost does not shrink as data parallelism scales out. As a result, memory often becomes the main bottleneck in data-parallel training. 

Microsoft proposed a highly influential algorithm called **Zero Redundancy Optimizer (ZeRO)** to address the memory limitations of data-parallel training, and later built the distributed training framework **DeepSpeed** on top of PyTorch around it. Its core idea **ZeRO-DP** is to partition **model states** (optimizer states, gradients, and parameters) evenly across GPUs, so the memory usage on each GPU becomes inversely proportional to the degree of data parallelism, while communication efficiency remains largely unaffected. Besides model state, the remaining memory called **residual states** are used by activation, temporary buffers, and unusable fragmented memory. **ZeRO-R** is to optimize these residual memory. This section of the blog focuses on ZeRO-DP.

ZeRO-DP has three main optimization stages: $P_{OS}$ refers to ZeRO-1, $P_{OS+g}$ refers to ZeRO-2, and $P_{OS+g+p}$ refers to ZeRO-3.

| Name | Stage | Notation |What’s Sharded |
| --- | --- | --- | --- |
| ZeRO-1 | Stage 1 | $P_{OS}$     | Optimizer states |
| ZeRO-2 | Stage 2 | $P_{OS+g}$   | + Gradients |
| ZeRO-3 | Stage 3 | $P_{OS+g+p}$ | + Model parameters |

{{< figure src="./images/zero-dp.png" caption="ZeRO-DP optimization stages." align="center" >}}

**Memory Reduction**

In mixed-precision training, the model parameters are stored in `float16`, the model gradients are stored in `float16`, and the Adam states are stored in `float32`, including the master copy of the model parameters, the momentum, and the variance. Let the total number of model parameters be $\Phi$. Then the total memory required is

$$
2\Phi + 2\Phi + (4\Phi + 4\Phi + 4\Phi) = 4\Phi + 12\Phi = 16\Phi \text{ bytes}.
$$

This shows that the Adam states account for 75% of the memory footprint.

Let the data-parallel degree be $N_d$.

In ZeRO-1, only the Adam states are partitioned across GPUs, while the `float16` parameters and `float16` gradients are still replicated on every GPU. Therefore, the per-GPU memory usage is

$$
M_{\text{ZeRO-1}} = 2\Phi + 2\Phi + \frac{12\Phi}{N_d} = 4\Phi + \frac{12\Phi}{N_d}.
$$

In ZeRO-2, both the Adam states and the gradients are partitioned across GPUs, while the `float16` parameters are still replicated. Therefore, the per-GPU memory usage is

$$
M_{\text{ZeRO-2}} = 2\Phi + \frac{2\Phi}{N_d} + \frac{12\Phi}{N_d} = 2\Phi + \frac{14\Phi}{N_d}.
$$

In ZeRO-3, the Adam states, gradients, and model parameters are all partitioned across GPUs. Therefore, the per-GPU memory usage is

$$
M_{\text{ZeRO-3}} = \frac{2\Phi}{N_d} + \frac{2\Phi}{N_d} + \frac{12\Phi}{N_d} = \frac{16\Phi}{N_d}.
$$

Compared with conventional data parallelism, the per-GPU memory usage can therefore be reduced by up to $4\times$, $8\times$, and $N_d\times$ for ZeRO-1, ZeRO-2, and ZeRO-3, respectively.

**Communication Overhead**

TODO: "Communication Analysis of ZeRO-DP"



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
