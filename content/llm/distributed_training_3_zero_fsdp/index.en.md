---
title: "Distributed Training: Data Parallelism (ZeRO and FSDP)"
date: 2025-12-15
tags: ["llm", "distributed-training", "zero", "fsdp"]
math: true
---

Conventional data parallelism such as DDP replicates the full model state on every GPU, so memory does not decrease as the cluster scales out. ZeRO and FSDP address this by sharding optimizer states, gradients, and parameters across devices, allowing much larger models to be trained under the same memory budget.

这篇文章的重点有两个：
  1. 哪些 tensor 可以长期以分片形式保存，哪些 tensor 分片后需要在 forward、backward 和 optimizer step 中进行通信，什么时候进行通信，以及为什么这样的分片和通信方式不会破坏训练语义。
  2. 这种分片策略分别能节省多少 memory 和 communication。

### ZeRo [^1]
In conventional data-parallel training, every machine still has to hold a full copy of the model state in memory, and that memory cost does not shrink as data parallelism scales out. As a result, memory often becomes the main bottleneck in data-parallel training. 

Microsoft proposed a highly influential algorithm called **Zero Redundancy Optimizer (ZeRO)** to address the memory limitations of data-parallel training, and later built the distributed training framework **DeepSpeed** on top of PyTorch around it. Its core idea **ZeRO-DP** is to partition **model states (optimizer states, gradients, and parameters)** evenly across GPUs, so the memory usage on each GPU becomes inversely proportional to the degree of data parallelism, while communication efficiency remains largely unaffected. Besides model state, the remaining memory called **residual states** are used by activation, temporary buffers, and unusable fragmented memory. **ZeRO-R** is to optimize these residual memory. This section of the blog focuses on ZeRO-DP.

ZeRO-DP has three main optimization stages: $P_{OS}$ refers to ZeRO-1, $P_{OS+g}$ refers to ZeRO-2, and $P_{OS+g+p}$ refers to ZeRO-3.

| Name | Stage | Notation |What’s Sharded |
| --- | --- | --- | --- |
| ZeRO-1 | Stage 1 | $P_{OS}$     | Optimizer states |
| ZeRO-2 | Stage 2 | $P_{OS+g}$   | + Gradients |
| ZeRO-3 | Stage 3 | $P_{OS+g+p}$ | + Model parameters |

{{< figure src="./images/zero-dp.png" caption="ZeRO-DP optimization stages." align="center" >}}

**Training Step**

| Stage | Need parameter | Need gradient | Need optimizer states |
| --- | --- | --- | --- |
| Forward | Yes | No | No |
| Backward | Yes | Yes (layerwise) | No |
| Optimizer step | Yes | Yes | Yes |

In a standard training iteration:

- The **forward pass** uses the current parameters to compute activations.
- The **backward pass** again depends on those parameters and produces gradients.
- The **optimizer step** uses the parameters, gradients, and optimizer states to update the parameters.
- In the **next iteration**, the model enters the forward pass again using the updated parameters.

ZeRO-3's training iteration is:

- During the **forward pass**, each GPU persistently stores only its own parameter shard. When a layer is about to run, the ranks **all-gather** that layer's full **parameter** tensor, use it for computation, and then release the temporary full copy after the layer finishes.
-During the **backward pass**, the ranks again need the full **parameter** values for the current layer, so they perform another **all-gather**. They then compute gradients, after which the **gradients** are **reduce-scattered** across ranks so that each rank keeps only the gradient shard corresponding to its local parameter shard.
- During the **optimizer step**, each rank updates only its local parameter shard using its local shard of the parameters, gradients, and optimizer states.

This also explains why different tensors are communicated differently in ZeRO-3:

- **Optimizer states** do not need to be replicated across all ranks. For Adam, the momentum `m` and variance `v` remain sharded, and each rank updates only its own shard locally.
- **Gradients** need communication, because each rank computes gradients from a different mini-batch shard. ZeRO-3 therefore uses **reduce-scatter** so that each rank receives the reduced gradient shard it owns. In conventional DDP, this synchronization is usually done with **all-reduce**.
- **Parameters** need temporary **all-gather** communication in ZeRO-3, because the forward and backward computation for a layer requires the full weight tensor even though no rank stores that full tensor persistently.


**Memory Usage Reduction**

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

**Communication Volumn/ Data Movements**

In conventional data parallel training, gradients are synchronized across all data-parallel ranks at the end of backward propagation before the optimizer update. This is typically implemented with **all-reduce**, which can be viewed as a **reduce-scatter** followed by an **all-gather**. Let $Ψ$ denote the total number of gradient elements in the model. Under ring all-reduce, the per-rank communication is approximately $2Ψ$ elements.

ZeRO-2 keeps this communication volume roughly unchanged. Instead of all-reducing the full gradient tensor, it performs a **reduce-scatter** on gradients so that each rank receives only the reduced shard it owns, then applies the optimizer update locally, and finally runs an **all-gather** on the updated parameters. The total communication is therefore still about $Ψ + Ψ = 2Ψ$ elements per step.

ZeRO-3 adds extra communication because parameters are also sharded. It still needs about $Ψ$ elements for gradient **reduce-scatter** and about $Ψ$ elements for parameter **all-gather** after the optimizer step, but it must also gather model parameter shards for computation via an additional **all-gather** because no rank stores the full parameter set locally. Under this simplified accounting, the total communication becomes about $3Ψ$ elements per step.

See [Collective Operations](/llm/collective_operations/) for the detailed communication-volume breakdown and calculation of each operation.

The conclusion is that ZeRO-1 and ZeRO-2 have the same communication volume as conventional data parallelism, while ZeRO-3 increases communication volume.


### FSDP [^2] [^3]
Facebook introduced **FSDP (Fully Sharded Data Parallel)** as PyTorch’s counterpart to Microsoft’s **ZeRO** in DeepSpeed. FSDP can be viewed as an optimized version of **DDP** within PyTorch. It is still a form of data parallelism, but unlike DDP, FSDP uses **parameter sharding**. In other words, model parameters are partitioned across GPUs, whereas in DDP each GPU keeps a full copy of the parameters. This allows FSDP to achieve better training efficiency, both in terms of speed and GPU memory usage.

{{< figure src="./images/fsdp.png" caption="FSDP execution flow." align="center" >}}

**Constructor**

- Shard the model parameters and distribute the shards across ranks.

**Forward pass**

1. For each FSDP unit, run **all_gather** to collect **parameter** shards from all ranks, so each rank temporarily holds the full parameters for the current unit. This is also why FSDP still belongs to data parallelism: although the parameters are sharded for storage, computation still uses the original full parameters rather than computing directly on weight shards as in tensor parallelism.
2. Execute the forward computation.
3. Reshard the parameters, discard the portions that do not belong to the current rank, and free the corresponding memory.

**Backward pass**

1. For each FSDP unit, run **all_gather** again to collect **parameter** shards from all ranks.
2. Execute the backward computation.
3. Reshard the parameters again, discard the portions that do not belong to the current rank, and free the corresponding memory.
4. Run **reduce_scatter** to synchronize **gradients** across ranks.

**Optimizer updates**

- Each rank updates the local shard of gradients and parameters that it owns.

**Code**

```python
    # Optional: auto-wrap large layers
    # If a submodule has parameter count >= threshold, wrap it as its own FSDP unit
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1e6)

    model = FSDP(model, auto_wrap_policy=auto_wrap_policy)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

The full runnable example is in [`code/basic_fsdp1.py`](./code/basic_fsdp1.py). It initializes distributed execution, wraps the model with `FSDP`, and then runs a standard forward-backward-optimizer loop. The optional `auto_wrap_policy` demonstrates how larger submodules can be split into separate FSDP units instead of wrapping the entire model as one flat unit.


#### FSDP 2 [^4]

**fully_shard()**

FSDP2 replaces the wrapper-style `FSDP(...)` API in FSDP1 with the composable `fully_shard()` API. In practice, `fully_shard()` is typically applied bottom-up, often together with a device mesh:

```python
from torch.distributed.fsdp import fully_shard, FSDPModule
model = Transformer()

# apply fully_shard() on each layer first, then the root model
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)
```

**DTensor**

After `fully_shard(model)`, model parameters are represented as `DTensor`s, which encode sharded parameter layouts directly. In contrast, FSDP1 is mainly wrapper-based and historically relied on internal abstractions such as `FlatParameter` to manage sharded parameters. In other words, FSDP1 mostly manages sharding through the wrapper, while FSDP2 makes sharding part of the parameter representation itself.

**Streams**

Within all-gather, copy-in refers to copying and packing scattered sharded local parameters into one contiguous all-gather buffer within a single node so that NCCL can transfer them more efficiently. If the copy-in takes a long time, it's worth use a separate stream for it:

```
Layer N:   |<- Copy-in ->|<- All-Gather ->|<- Compute ->|
Layer N+1:               |<- Copy-in ->|<- All-Gather ->|<- Compute ->|
Layer N+2:                             |<- Copy-in ->|<- All-Gather ->|<- Compute ->|
```

For basic 1D FSDP2, the main communication path only needs the all-gather copy-in stream, the all-gather stream, and the reduce-scatter streams. The extra all-reduce stream exists because the runtime is shared with HSDP and custom all-reduce hooks. Below is a code snippet from pyTorch FSDP2 source code (https://github.com/pytorch/pytorch/blob/ba15482709c78c86a22524387f31bf6f13468dd2/torch/distributed/fsdp/_fully_shard/_fsdp_param_group.py#L72-L93):

```pytorch
def lazy_init(self, device: torch.device):
    ...
    self.all_gather_copy_in_stream = self.device_handle.Stream(priority=high_priority)
    self.all_gather_stream = self.device_handle.Stream(priority=high_priority)
    self.reduce_scatter_stream = self.device_handle.Stream(priority=high_priority)
    self.all_reduce_stream = self.device_handle.Stream()
    ...
```

[^1]: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. arXiv, October 4, 2019. <https://arxiv.org/abs/1910.02054>
[^2]: Fully Sharded Data Parallel: Faster AI Training with Fewer GPUs. Meta Engineering, July 15, 2021. <https://engineering.fb.com/2021/07/15/open-source/fsdp/>
[^3]: PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel. arXiv, April 21, 2023. <https://arxiv.org/abs/2304.11277>
[^4]: Getting Started with Fully Sharded Data Parallel (FSDP2). PyTorch Tutorials. <https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html>
