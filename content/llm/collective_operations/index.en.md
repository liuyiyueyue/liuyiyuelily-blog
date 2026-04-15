---
title: "Collective Operations"
date: 2023-05-14
tags: ["llm", "distributed-training", "nccl"]
---

Collective operations are the basic communication primitives used in distributed training. They define how tensors are reduced, replicated, partitioned, and exchanged across ranks. Below are some of the most common collective operations in distributed training [^1].

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

In FSDP, 

{{< figure src="./images/fsdp_all_reduce.png" caption="FSDP All Reduce." align="center" >}}

**Ring AllReduce**

Ring all-reduce is the standard implementation of `AllReduce` on many GPU systems [^2] [^3]. It decomposes `AllReduce` into a `ReduceScatter` phase followed by an `AllGather` phase, with ranks exchanging data in a ring topology. This gives each rank a communication volume of `2 * (p - 1) * n / p` bytes.


[^1]: Collective Operations. CS 168 Textbook. <https://textbook.cs168.io/beyond-client-server/collective-operations.html>
[^2]: Bringing HPC Techniques to Deep Learning. arXiv, February 20, 2017. <https://arxiv.org/abs/1702.06299>
[^3]: 小窗幽记xxy, "一文搞懂ring all reduce和bucketing机制." Zhihu. <https://zhuanlan.zhihu.com/p/504957661>
