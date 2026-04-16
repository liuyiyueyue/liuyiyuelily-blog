---
title: "Collective Operations"
date: 2023-05-14
tags: ["llm", "distributed-training", "nccl"]
---

Collective operations are the basic communication primitives used in distributed training. They define how tensors are reduced, replicated, partitioned, and exchanged across ranks. Below are some of the most common collective operations in distributed training [^1].

### Table of Contents

- [P2P](#p2p)
- [Reduce](#reduce)
- [Broadcast](#broadcast)
- [Scatter](#scatter)
- [Gather](#gather)
- [AllGather](#allgather)
- [ReduceScatter](#reducescatter)
- [AllReduce](#allreduce)
- [PyTorch Examples](#pytorch-examples)

### P2P

A `P2P` operation is point-to-point communication between exactly two ranks, typically implemented as `send` and `recv`. If one rank sends a tensor of size `n` bytes to another rank, the communication volume is `n` bytes.

### Reduce

A `Reduce` operation combines values from all ranks and writes the result to one destination rank.
If each rank holds `n` bytes and there are `p` ranks, the total communication volume is about `(p - 1) * n` bytes.

{{< figure src="./images/reduce.png" align="center" >}}

### Broadcast

A `Broadcast` operation copies data from one source rank to all ranks.
If the source tensor has size `n` bytes and there are `p` ranks, the total communication volume is about `(p - 1) * n` bytes.

{{< figure src="./images/broadcast.png" align="center" >}}

### Scatter

A `Scatter` operation splits data on one rank and sends one shard to each rank.
If the full tensor on the source rank has size `n` bytes and there are `p` ranks, the total communication volume is about `(p - 1) * n / p` bytes.

{{< figure src="./images/scatter.png" align="center" >}}

### Gather

A `Gather` operation collects shards from all ranks and places the combined result on one destination rank.
If the final gathered tensor has size `n` bytes and there are `p` ranks, the total communication volume is about `(p - 1) * n / p` bytes.

{{< figure src="./images/gather.png" align="center" >}}

### AllGather

An `AllGather` operation collects shards from all ranks and places the full concatenated result on every rank.
If each rank starts with `n` bytes and there are `p` ranks, each rank receives `(p - 1) * n` bytes, so the total communication volume is about `p * (p - 1) * n` bytes.

{{< figure src="./images/all-gather.png" align="center" >}}

### ReduceScatter

A `ReduceScatter` operation first reduces values across ranks and then scatters one reduced shard to each rank.
If the reduced output on each rank has size `n / p` bytes, the total communication volume is about `(p - 1) * n` bytes.

{{< figure src="./images/reduce-scatter.png" align="center" >}}

### AllReduce

An `AllReduce` operation reduces values across all ranks and returns the same reduced result to every rank.

{{< figure src="./images/all-reduce.png" align="center" >}}

**AllReduce = ReduceScatter + AllGather**

`AllReduce` can be implemented as a `ReduceScatter` followed by an `AllGather`.

{{< figure src="./images/all-reduce-composition.png" align="center" >}}

In FSDP, see [Distributed Training: ZeRO and FSDP](/llm/distributed_training_3_zero_fsdp/) for a concrete use of all-reduce-related collectives such as `all_gather` and `reduce_scatter`.

{{< figure src="./images/fsdp_all_reduce.png" caption="FSDP All Reduce." align="center" >}}

**Ring AllReduce**

Ring all-reduce is the standard implementation of `AllReduce` on many GPU systems [^2] [^3]. It decomposes `AllReduce` into a `ReduceScatter` phase followed by an `AllGather` phase, with ranks exchanging data in a ring topology. This gives each rank a communication volume of `2 * (p - 1) * n / p` bytes.

### PyTorch Examples

Here are PyTorch examples based on `torch.distributed` collective functions [^4] [^5]:

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def init_process(rank, world_size, fn, backend='nccl'):
    """
    Initialize the distributed environment for each process, enable communication between them, and call fn.
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    try:
        fn(rank, world_size)
    finally:
        dist.destroy_process_group()

def run(world_size, func):
    """
    Launch world_size processes and execute func.
    """
    ctx = mp.get_context("spawn")
    processes = []
    for rank in range(world_size):
        p = ctx.Process(target=init_process, args=(rank, world_size, func))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def p2p_block_func(rank, world_size):
    """
    Send the tensor on rank src to rank dst (blocking).
    """
    src_rank = 0
    dst_rank = 1
    group = dist.new_group(list(range(world_size)))
    # On rank src, this tensor is used for sending.
    # On rank dst, this tensor is used for receiving.
    tensor = torch.zeros(1).to(torch.device("cuda", rank))
    if rank == src_rank:
        tensor += 1
        # Send tensor([1.]).
        # group defines the set of processes visible to this operation; by default it is the entire world.
        dist.send(tensor=tensor, dst=dst_rank, group=group)
    elif rank == dst_rank:
        # The tensor on rank dst starts as tensor([0.]) and becomes tensor([1.]) after receiving.
        dist.recv(tensor=tensor, src=src_rank, group=group)
    print(f"P2P Block: Rank {rank} has data {tensor}")

def p2p_unblock_func(rank, world_size):
    """
    Send the tensor on rank src to rank dst (non-blocking).
    """
    src_rank = 0
    dst_rank = 1
    group = dist.new_group(list(range(world_size)))
    tensor = torch.zeros(1).to(torch.device("cuda", rank))
    if rank == src_rank:
        tensor += 1
        # Non-blocking send via isend.
        req = dist.isend(tensor=tensor, dst=dst_rank, group=group)
        print(f"P2P Unblock: Rank {rank} started sending")
    elif rank == dst_rank:
        # Non-blocking receive.
        req = dist.irecv(tensor=tensor, src=src_rank, group=group)
        print(f"P2P Unblock: Rank {rank} started receiving")
    req.wait() # Non-blocking send/recv must call wait() before reusing the tensor.
    print(f"P2P Unblock: Rank {rank} has data {tensor}")

def broadcast_func(rank, world_size):
    '''Broadcast tensor([1.]) from rank 0 to all ranks.'''
    src_rank = 0
    group = dist.new_group(list(range(world_size)))
    if rank == src_rank:
        # On rank src, initialize tensor([1.]).
        tensor = torch.zeros(1).to(torch.device("cuda", rank)) + 1
    else:
        # On non-src ranks, initialize tensor([0.]).
        tensor = torch.zeros(1).to(torch.device("cuda", rank))
    # On rank src, broadcast sends; on all other ranks, it receives.
    dist.broadcast(tensor=tensor, src=src_rank, group=group)
    print(f"Broadcast: Rank {rank} has data {tensor}")

def reduce_func(rank, world_size):
    '''Aggregate tensors from all ranks in the group and send the result to rank dst.'''
    dst_rank = 1
    group = dist.new_group(list(range(world_size)))
    tensor = torch.ones(1).to(torch.device("cuda", rank))
    # Every rank sends, but only dst receives the summed result.
    dist.reduce(tensor, dst=dst_rank, op=dist.ReduceOp.SUM, group=group)
    print(f"Reduce: Rank {rank} has data {tensor}")

def allreduce_func(rank, world_size):
    '''Aggregate tensors from all ranks in the group and send the result to every rank in the group.'''
    group = dist.new_group(list(range(world_size)))
    tensor = torch.ones(1).to(torch.device("cuda", rank))
    # tensor is used for both sending and receiving.
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print(f"Allreduce: Rank {rank} has data {tensor}")

def gather_func(rank, world_size):
    '''Collect tensors from all ranks in the group and send them to rank dst.'''
    dst_rank = 1
    group = dist.new_group(list(range(world_size)))
    # This tensor is used for sending.
    tensor = torch.zeros(1).to(torch.device("cuda", rank)) + rank
    gather_list = []
    if rank == dst_rank:
        # gather_list should contain world_size tensors to receive tensors sent from other ranks.
        gather_list = [torch.zeros(1).to(torch.device("cuda", dst_rank)) for _ in range(world_size)]
        # Only rank dst needs to provide gather_list.
        dist.gather(tensor, gather_list=gather_list, dst=dst_rank, group=group)
    else:
        # Non-dst ranks simply send their tensor.
        dist.gather(tensor, dst=dst_rank, group=group)
    print(f"Gather: Rank {rank} has data {gather_list}")

def allgather_func(rank, world_size):
    group = dist.new_group(list(range(world_size)))
    # This tensor is used for sending.
    tensor = torch.zeros(1).to(torch.device("cuda", rank)) + rank
    # gather_list is used to receive tensors sent from each rank.
    gather_list = [torch.zeros(1).to(torch.device("cuda", rank)) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor, group=group)
    # Every rank ends up with the same gather_list.
    print(f"Allgather: Rank {rank} has data {gather_list}")

def scatter_func(rank, world_size):
    src = 0
    group = dist.new_group(list(range(world_size)))
    # The tensor each rank uses for receiving.
    tensor = torch.empty(1).to(torch.device("cuda", rank))
    if rank == src:
        # On rank src, distribute tensors in tensor_list to different ranks.
        # tensor_list: [tensor([1.]), tensor([2.])]
        tensor_list = [torch.tensor([i + 1], dtype=torch.float32).to(torch.device("cuda", rank)) for i in range(world_size)]
        # Send tensor_list to all ranks.
        # Receive the portion that belongs to rank src.
        dist.scatter(tensor, scatter_list=tensor_list, src=0, group=group)
    else:
        # Receive the tensor assigned to the current rank.
        dist.scatter(tensor, scatter_list=[], src=0, group=group)
    # Each rank gets one tensor from tensor_list.
    print(f"Scatter: Rank {rank} has data {tensor}")

def reduce_scatter_func(rank, world_size):
    group = dist.new_group(list(range(world_size)))
    # Tensor used for receiving.
    tensor = torch.empty(1).to(torch.device("cuda", rank))
    # List of tensors used for sending.
    # On each rank, tensor_list = [tensor([0.]), tensor([1.])].
    tensor_list = [torch.Tensor([i]).to(torch.device("cuda", rank)) for i in range(world_size)]
    # Step 1: reduce produces the tensor list [tensor([0.]), tensor([2.])].
    # Step 2: the tensor list [tensor([0.]), tensor([2.])] is scattered across ranks.
    # Rank 0 gets tensor([0.]), and rank 1 gets tensor([2.]).
    dist.reduce_scatter(tensor, tensor_list, op=dist.ReduceOp.SUM, group=group)
    print(f"Reduce Scatter: Rank {rank} has data {tensor}")

if __name__ == "__main__":
    run(2, p2p_block_func)
    run(2, p2p_unblock_func)
    run(2, broadcast_func)
    run(2, reduce_func)
    run(2, allreduce_func)
    run(2, gather_func)
    run(2, allgather_func)
    run(2, scatter_func)
    run(2, reduce_scatter_func)
```

[^1]: Collective Operations. CS 168 Textbook. <https://textbook.cs168.io/beyond-client-server/collective-operations.html>
[^2]: Bringing HPC Techniques to Deep Learning. arXiv, February 20, 2017. <https://arxiv.org/abs/1702.06299>
[^3]: 小窗幽记xxy, "一文搞懂ring all reduce和bucketing机制." Zhihu. <https://zhuanlan.zhihu.com/p/504957661>
[^4]: 【深度学习】【分布式训练】Collective通信操作及Pytorch示例 - 白强伟的文章 - 知乎 <https://zhuanlan.zhihu.com/p/607605729>
[^5]: Distributed communication package - torch.distributed. PyTorch documentation. <https://docs.pytorch.org/docs/stable/distributed.html#collective-functions>
