---
title: "Distributed Training: Data Parallelism (PyTorch DP vs DDP)"
date: 2025-12-07
tags: ["llm", "pytorch", "distributed-training"]
---


## nn.DataParallel (DP)

DP is a single-node, multi-GPU parallelization method (单机多卡).

#### How It Works

PyTorch's `DataParallel`, or `DP`, uses a parameter-server architecture and runs in a single process. During training, one GPU acts as the **server** and the remaining GPUs act as **workers**. Each GPU keeps a replica of the model for computation. The input data is first split across GPUs, each worker computes on its own shard, and the gradients are then gathered on the server GPU. The server updates the model parameters and synchronizes the updated model back to the other GPUs.

`DataParallel` is very easy to use. In many cases, it only requires one extra line in a single-GPU training script:

```python
model = nn.DataParallel(model)
```

#### Limitations

This approach has a clear drawback: the server GPU bears both heavy communication overhead and heavy computation cost. It must communicate with every other GPU, and it is responsible for gradient aggregation and parameter updates, which makes the overall training process inefficient. As the number of GPUs increases, the **communication overhead** also grows linearly.


## nn.DistributedDataParallel (DDP)

DDP supports multi-node, multi-GPU training and mixed precision (多机多卡).

#### How It Works

PyTorch's Distributed data parallelism (`DistributedDataParallel`, or `DDP`) uses a **Ring-AllReduce** architecture and runs with multiple processes. To train with DDP, we usually need to modify three parts of the code: the data loader, logging and printing, and evaluation logic. In practice, DDP code mainly requires attention in three areas: data partitioning, I/O behavior, and evaluation.

Its implementation is somewhat more complex, but one principle is enough to keep in mind: **each GPU corresponds to one process**, and data is not shared across processes unless we explicitly implement that behavior. PyTorch only provides the mechanisms for gradient synchronization and parameter updates. Everything else must be handled by the user.

#### Parameters

| Parameter | Meaning | How to inspect |
| --- | --- | --- |
| `group` | The process group used for distributed training. Each group can perform its own communication and gradient synchronization. | A group is usually created when initializing the distributed environment, or by creating custom groups with APIs such as `torch.distributed.new_group()`. |
| `world size` | The total number of processes participating in the current distributed training job. In single-node multi-GPU training, the world size usually equals the number of GPUs. In multi-node training, it is the total number of GPUs across all machines. | `torch.distributed.get_world_size()` |
| `rank` | The unique identifier of a process among all processes participating in distributed training. Rank usually starts from `0` and ends at `world_size - 1`. | `torch.distributed.get_rank()` |
| `local rank` | The relative identifier of the current process within its local node. For example, on a single machine with 4 GPUs, the local ranks are `0`, `1`, `2`, and `3`. This value is commonly used to decide which GPU a process should use. | `local rank` is not provided directly by the PyTorch distributed API. It is usually passed through environment variables set by the launcher or through arguments to the training script. |

#### Training Process

1. At the start of training, the dataset is evenly partitioned across GPUs. Each GPU independently performs the forward pass to compute predictions and the backward pass to compute gradients on its local data shard.

2. The gradients on all GPUs are synchronized to ensure consistent model updates. This synchronization is implemented through the Ring-AllReduce algorithm.

3. Once gradient synchronization is complete, each GPU updates the parameters of its local model replica using the aggregated gradients. Because all GPUs use the same synchronized gradients, all model replicas remain identical throughout training.

#### Ring-AllReduce Algorithm

Ring-All-Reduce uses a ring topology in which all GPUs are equal peers. Each GPU maintains a model replica and communicates only with its two neighboring GPUs in the ring. After `N` communication rounds, where `N` is the number of GPUs, every GPU has accumulated the sum of gradients from all GPUs.

#### Data Partition
How to evenly partition data across GPUs?

In PyTorch, this is typically done with `torch.utils.data.DataLoader` and `torch.utils.data.distributed.DistributedSampler`. `DistributedSampler` automatically partitions the dataset based on the dataset itself, the total number of processes (`world_size`), and the current process ID (`rank`), so that each process receives a unique and balanced subset of data.

#### I/O Operations
How to avoid duplicated I/O operations?

Because each process runs independently, I/O operations such as `print`, `save`, and `load` will execute in every GPU process unless handled explicitly. This often causes repeated console output, file write conflicts, and unnecessary resource usage. A common solution is to perform these operations only on the process with `rank == 0`, and synchronize with other processes only when needed.

#### Evaluation
How to gather data from all processes for evaluation?

The `torch.distributed.all_gather` function can collect evaluation results from all processes so that each process has access to the full evaluation set and can compute global metrics. If only aggregated values are needed, such as total loss or average accuracy, `torch.distributed.reduce` or `torch.distributed.all_reduce` is usually more efficient.


## Summary

**nn.DataParallel (DP)**
- DP uses a **parameter-server** style design.
- DP runs as a **single process with multiple threads**.
- The master GPU broadcasts parameters and gathers gradients from worker GPUs.
- This centralized coordination often becomes a **bottleneck in distributed training**.

**nn.DistributedDataParallel (DDP)**
- DDP uses decentralized gradient synchronization through **Ring AllReduce**.
- DDP runs **one process per GPU**, and each process maintains its own model replica and data shard.
- Gradients are synchronized after the backward pass through collective communication.
- DDP is usually much faster and more **scalable** than DP.


## Reference
- PyTorch Distributed: Experiences on Accelerating Data Parallel Training ([https://arxiv.org/abs/2006.15704](https://arxiv.org/abs/2006.15704)).
