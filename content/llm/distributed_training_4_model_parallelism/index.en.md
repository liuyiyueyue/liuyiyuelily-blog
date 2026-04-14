---
title: "Distributed Training: Model Parallelism"
date: 2025-12-17
tags: ["llm", "distributed-training", "model-parallelism"]
---

### Why Model Parallelism?

Data parallelism works well when the full model can still fit on each GPU. But once a model becomes too large, simply replicating it across devices is no longer practical.

Model parallelism solves this by partitioning the model itself across multiple GPUs. Instead of every rank storing all weights, each rank stores and computes only part of the model. The tradeoff is that memory usage goes down, but communication and scheduling complexity go up.

In practice, model parallelism usually appears in several forms: **tensor parallelism**, **pipeline parallelism**, **sequence parallelism**, **expert parallelism**, and **automatic parallelism**.

### Tensor Parallelism

**Tensor parallelism (TP)** splits the computation inside one layer across multiple GPUs. The classic example is a large matrix multiplication in a transformer block in the **Megatron-LM paper** by Nvidia [^1] [^2]. Megatron shards the weight matrix `W` across GPUs in two methods:
- **Column parallelism**: split the output dimension, so each GPU computes a subset of output columns.
- **Row parallelism**: split the input dimension, so each GPU computes a partial result and then sums across GPUs.

TP reduces peak GPU memory usage and speeds up computation, but the communication costs between GPUs are significant.

### Pipeline Parallelism

**Pipeline parallelism (PP)** splits the model by layers, partitioning several consecutive layers into one group, called a **stage**. Different stages are then assigned to different devices, so each GPU only computes a portion of the network’s layers.

For example, in a 12-layer transformer:

- GPU 0 may hold layers 1-3
- GPU 1 may hold layers 4-6
- GPU 2 may hold layers 7-9
- GPU 3 may hold layers 10-12

A naive implementation of pipeline parallelism leaves many **pipeline bubble**, i.e. some devices sit idle at the beginning and end of each step while waiting for other stages. Hence, research on pipeline parallelism mainly focuses on:
1. reducing pipeline bubbles to improve overall system throughput
2. lowering the memory overhead on each worker so the system can scale to larger models

**GPipe**

The amount of time each GPU spends working is closely tied to the batch size it processes. To reduce this waiting time, GPipe, proposed by Google, presents a simple idea to further split a batch into multiple smaller sub-batches, called **micro-batches** [^3].

{{< figure src="./images/gpipe_figure2_pipeline_parallelism.png" caption="Figure 2 from the GPipe paper: sequential execution, naive model parallelism, and pipeline parallelism with micro-batches." align="center" >}}

**PipeDream**

The PipeDream family was proposed by Microsoft's MSR Fiddle team [^4]. The core idea is to futher reduce pipeline bubbles. If multiple training iterations are in flight at the same time, each node can work on a different iteration at any given moment. This avoids waiting in place on strict step-by-step data dependencies and keeps the devices busy.

{{< figure src="./images/pipedream_model_parallel_4_machines.png" caption="Figure 3: Model parallel training with 4 machines. Numbers indicate minibatch ID. For simplicity, here we assume that forward and backward work in every stage takes one time unit, and communicating activations across machines has no overhead." align="center" >}}

{{< figure src="./images/pipedream_pipeline_4_machines.png" caption="Figure 8: An example pipeline with 4 machines, showing startup and steady states." align="center" >}}

PipeDream partitions the model into pipeline stages by balancing per-stage compute, memory, and communication, so no single stage becomes the throughput bottleneck (划分任务). To address numerical issues from stale weights, it uses techniques like weight stashing and 1F1B scheduling so each microbatch’s forward and backward passes stay more consistent (收敛性问题).


### Sequence Parallelism

Two papers both discuss sequence parallelism but with different goals and methods.

**Megatron-LM**

The first paper is Megatron-LM’s third paper, “Reducing Activation Recomputation in Large Transformer Models” [^5]. The motivation behind Megatron-LM’s sequence parallelism was to *distribute the memory that tensor parallelism could not shard anymore*.

{{< figure src="./images/sequence_parallel_figure5.png" caption="Figure 5 from the Megatron-LM paper." align="center" >}}

**ColossalAI**

The other paper is ColossalAI’s “Sequence Parallelism: Long Sequence Training from a System Perspective” [^6]. It mainly addresses limitations from long input sequence length, which scales quadratically with self-attention's memory usage. It splits the input sequence into multiple chunks and feed each chunk to a GPU. It proposes the the Ring Self-Attention (RSA) to compute the attention.


### Expert Parallelism

**MoE**

**Mixture of Experts** (MoE) is a neural network architecture that replaces one large feed-forward block with multiple specialized subnetworks called **experts**. Different experts learn to handle different kinds of tokens, since different tokens carry different meanings.

A **gating** or **routing** network routes each token to the most relevant expert or a small subset of experts. For each token, it assigns a probability to every expert, and we keep only the top-K experts with the highest probabilities. This can improve training effectiveness and increase model capacity without activating the whole model every time. There are topics like load balancing and token buffer, but we won't talk about them here.

Because each token uses only the top-K experts rather than all experts, an MoE layer is sparse. This also lets compute grow sublinearly as the model scales.

**Expert Parallelism**

Expert parallelism (EP) is mainly used with MoE models. In an MoE layer, a router sends each token to only a small subset of experts rather than activating the full dense feed-forward block [^7]. With EP, different experts are placed on different GPUs:

- GPU 0 and 1 hold experts 1-2
- GPU 2 and 3 hold experts 3-4

{{< figure src="./images/expert_parallel.png" caption="Expert parallelism." align="center" width="70%" >}}

During the forward pass, a token may be routed to any expert, so we must send it to the target GPU, run the computation there, and then bring it back. This process of “cross-GPU transfer plus return to the original GPU” is **EP all-to-all**, and it is the core of expert parallelism: 
1. how do we efficiently route input tokens to the devices hosting their selected experts during the all-to-all **Dispatch phase**? 
2. how do we send the expert outputs back to the original devices and aggregate the results during the all-to-all **Combine phase**?

{{< figure src="./images/all-to-all_expert_parallel.png" caption="All-to-all dispatch and combine. Each block represents one token." align="center" >}}

**GShard**

The GShard paper consists of two parts: one talks about the APIs ("GShard is a module composed of a set of lightweight annotation APIs and an extension to the XLA compiler."); the other discusses MoE. We focus on the second part here. GShard was the first work to extend the MoE idea to Transformers. Specifically, it replaced every other FFN layer in the Transformer encoder and decoder with a position-wise MoE layer, using a Top-2 gating network throughout [^7].

![GShard expert parallelism](images/gshard_expert_parallelism.png)

**DeepEP**

DeepSeek's DeepEP is a communication library designed specifically for MoE models and Expert Parallelism. It provides high-throughput, low-latency **all-to-all** CUDA kernels via NVLink and RDMA, aka MoE **dispatch** and **combine**. The library also supports low-precision operations, including FP8.

What suprises me is that the DeepEP's optimizations go very deep. This library uses **a custom PTX instruction**, `ld.global.nc.l1::no_allocate.l2::256b`, to improve global memory access by avoiding L1 cache allocation, reducing eviction, and taking advantage of 256-byte L2 cache transfers. DeepEP documents this optimization as an undefined-behavior technique and notes it can be disabled with DISABLE_AGGRESSIVE_PTX_INSTRS=1 on unsupported platforms [^13].

> PTX ("pee-tex") is NVIDIA's low-level parallel-thread execution virtual machine, exposing the GPU as a parallel compute device and serving as the intermediate instruction set generated by high-level compilers like CUDA C++. Those PTX instructions are then translated into SASS, the native assembly that actually runs on NVIDIA GPU hardware [^14].


### Automatic Parallelism

(这段儿写的不太走心。。。)

The goal of automatic parallelism (自动并行) is straightforward: given a model and the available hardware resources, the system should automatically choose a good parallelization strategy for efficient execution.

There are two common modes:

- **Semi-automatic**: users provide limited sharding hints for some tensors or operators, and the framework propagates them through the computation graph. Representative systems include **Mesh-TensorFlow**, **GShard**, and **GSPMD** [^12] [^7] [^8].
- **Fully automatic**: the framework searches or synthesizes the strategy for all tensors and operators. Representative systems include **FlexFlow**, **Unity**, and **Alpa** [^9] [^10] [^11].

**Mesh-TensorFlow.** Standard SPMD often means data parallelism by splitting the batch dimension. Mesh-TensorFlow generalizes this idea by allowing other tensor dimensions to be partitioned as well. Each operation is lowered into parallel computation plus collective communication, and users describe model dimensions and device layout with a DSL; the system then maps the program onto a TPU mesh automatically [^12].

**GSPMD.** GSPMD uses **tensor sharding annotations** as a unified abstraction for different parallelization strategies. In practice, it keeps the user-facing programming model close to single-device programming, while inferring the partitioning for the remaining operators from a small number of annotations. Its SPMD partitioner can also support pipeline-style partitioning through a lightweight wrapper layer [^8].

**FlexFlow.** FlexFlow defines the **SOAP** search space, covering parallelization across the Sample, Operator, Attribute, and Parameter dimensions. On top of that space, it provides a deep learning framework that searches for an efficient strategy for a given model and machine configuration [^9].


### Comparing the Strategies

| Strategy | Split Dimension | Main Benefit | Main Cost |
| --- | --- | --- | --- |
| **Tensor Parallelism** | Within a layer | Fits very wide layers | Frequent collectives inside each block |
| **Pipeline Parallelism** | Across layers | Fits deep models | Pipeline bubbles and stage balancing |
| **Sequence Parallelism** | Across sequence length | Lowers activation memory | Extra coordination with TP |
| **Expert Parallelism** | Across experts | Scales parameter count efficiently | Token routing and `all_to_all` |


[^1]: Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv, September 17, 2019. <https://arxiv.org/abs/1909.08053>
[^2]: Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. arXiv, April 9, 2021. <https://arxiv.org/abs/2104.04473>
[^3]: GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism. arXiv, November 16, 2018. <https://arxiv.org/abs/1811.06965>
[^4]: PipeDream: Fast and Efficient Pipeline Parallel DNN Training. arXiv, June 9, 2018. <https://arxiv.org/abs/1806.03377>
[^5]: Reducing Activation Recomputation in Large Transformer Models. arXiv, May 5, 2022. <https://arxiv.org/abs/2205.05198>
[^6]: Sequence Parallelism: Long Sequence Training from a System Perspective. arXiv, May 27, 2021. <https://arxiv.org/abs/2105.13120>
[^7]: GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. arXiv, June 30, 2020. <https://arxiv.org/abs/2006.16668>
[^8]: GSPMD: General and Scalable Parallelization for ML Computation Graphs. arXiv, May 10, 2021. <https://arxiv.org/abs/2105.04663>
[^9]: Beyond Data and Model Parallelism for Deep Neural Networks. arXiv, July 14, 2018. <https://arxiv.org/abs/1807.05358>
[^10]: Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization. OSDI 2022. <https://www.usenix.org/conference/osdi22/presentation/unger>
[^11]: Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning. arXiv, January 28, 2022. <https://arxiv.org/abs/2201.12023>
[^12]: Mesh-TensorFlow: Deep Learning for Supercomputers. arXiv, November 5, 2018. <https://arxiv.org/abs/1811.02084>
[^13]: DeepEP README, "Undefined-behavior PTX usage." GitHub. <https://github.com/deepseek-ai/DeepEP/tree/main?tab=readme-ov-file#undefined-behavior-ptx-usage>
[^14]: NVIDIA, "Parallel Thread Execution ISA." CUDA Toolkit Documentation. <https://docs.nvidia.com/cuda/archive/13.0.0/hopper-tuning-guide/parallel-thread-execution/index.html>. NVIDIA, "PTX and SASS Assembly Debugging." Nsight Visual Studio Edition User Guide. <https://docs.nvidia.com/nsight-visual-studio-edition/5.2/Content/PTX_SASS_Assembly_Debugging.htm>
