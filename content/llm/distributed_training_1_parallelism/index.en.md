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
- Common approaches evolve from [parameter-server style **DP** to distributed style **DDP**](../distributed_trainining_2_ddp/), and then to memory-optimized methods such as **ZeRO and FSDP** [^1] [^10] [^11].
- DP sometimes also refers to the number of nodes, data-parallel degrees, or replica count


**Model parallelism** splits the model itself, including weights, optimizer states, and gradients, across GPUs.
- **Pipeline parallelism (PP)** runs different layers of the model on different GPUs. Common approaches include PipeDream, PipeDream-2BW, HetPipe, GPipe, DAPPLE, and Chimera [^5] [^6] [^7] [^8] [^9] [^14].
- **Tensor parallelism (TP)** breaks up the math for a single operation such as a matrix multiplication within a layer. Common approaches include Megatron-LM, Mesh-TensorFlow, Colossal-AI, and GShard [^2] [^3] [^4] [^13].
- **Mixture-of-Experts parallelism (EP)** processes each example with only a subset of the experts in each layer [^12] [^13].
- **Sequence parallelism (SP)** splits along the sequence length.


### Multi-dimension Parallelism
All these parallelism strategies can be used together and form a multi-dimension parallelism. Here are some examples:

| Training Framework         | Dimensions                                    |
| -------------------------- | --------------------------------------------- |
| PyTorch DDP                | 1D (DP)                                       |
| Megatron-LM                | 2–3D (DP + TP + PP)                           |
| DeepSpeed                  | 2–4D (DP + TP + PP + ZeRO/optimizer sharding) |
| PaLM / GPT-4 class systems | 4–5D (DP + TP + PP + SP + EP)                 |


### Common Optimization Steps
A typical development workflow for distributed training looks like this:

1. First, optimize single-GPU performance and maximize memory utilization as much as possible.

2. If the training dataset grows larger, use data parallelism to accelerate training.

3. For larger models, if you run into out-of-memory issues, consider enabling `ZeRO` or `FSDP`.

4. If the model scale increases further, consider model parallel strategies such as pipeline parallelism or tensor parallelism.

5. For extremely large models, you may need to combine multiple distributed strategies, including pipeline parallelism, tensor parallelism, data parallelism, and `ZeRO`.

[^1]: PyTorch Distributed: Experiences on Accelerating Data Parallel Training. arXiv, June 28, 2020. <https://arxiv.org/abs/2006.15704>
[^2]: Efficient Large-Scale Language Model Training on GPU Clusters. arXiv, April 9, 2021. <https://arxiv.org/abs/2104.04473>
[^3]: Mesh-TensorFlow: Deep Learning for Supercomputers. arXiv, November 5, 2018. <https://arxiv.org/abs/1811.02084>
[^4]: Colossal-AI: A Unified Deep Learning System for Large-Scale Parallel Training. arXiv, October 28, 2021. <https://arxiv.org/abs/2110.14883>
[^5]: PipeDream: Fast and Efficient Pipeline Parallel DNN Training. arXiv, June 8, 2018. <https://arxiv.org/abs/1806.03377>
[^6]: Memory-Efficient Pipeline-Parallel DNN Training. arXiv, June 16, 2020. <https://arxiv.org/abs/2006.09503>
[^7]: HetPipe: Enabling Large DNN Training on (Whimpy) Heterogeneous GPU Clusters through Integration of Pipelined Model Parallelism and Data Parallelism. USENIX ATC 2020. <https://www.usenix.org/conference/atc20/presentation/park>
[^8]: GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. arXiv, November 16, 2018. <https://arxiv.org/abs/1811.06965>
[^9]: DAPPLE: A Pipelined Data Parallel Approach for Training Large Models. arXiv, July 2, 2020. <https://arxiv.org/abs/2007.01045>
[^10]: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. arXiv, October 4, 2019. <https://arxiv.org/abs/1910.02054>
[^11]: Fully Sharded Data Parallel: Faster AI Training with Fewer GPUs. Meta Engineering, July 15, 2021. <https://engineering.fb.com/2021/07/15/open-source/fsdp/>
[^12]: Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. arXiv, January 23, 2017. <https://arxiv.org/abs/1701.06538>
[^13]: GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. arXiv, June 30, 2020. <https://arxiv.org/abs/2006.16668>
[^14]: Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines. SC '21 / ACM, November 13, 2021. <https://dl.acm.org/doi/10.1145/3458817.3476145>
