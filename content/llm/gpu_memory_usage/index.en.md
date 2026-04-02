---
title: "GPU Memory Usage"
date: 2026-03-28
tags: ["llm", "gpu", "training", "optimization"]
math: true
---

This post gives a compact breakdown of where GPU memory goes during training and inference. For the Transformer-specific discussion of how parameter memory and activation memory scale with model size, batch size, and sequence length, see a previous blog at [Transformer: Parameter Count, FLOPs, Memory Usage, and Training Time](/llm/attention_2_transformer_compute_memory/#memory-usage).

### Static and Dynamic Memory

We refer to the memory used by model parameters as **static memory**, meaning its lifetime is effectively unbounded and it must always occupy HBM. The optimizer states, model parameters, and usually gradients are static memory, since they must persist across training steps rather than being freed immediately after one operator finishes.

We refer to **activation** memory as **dynamic memory**, meaning its lifetime is short and its memory can be reused. Activations have dynamic memory since they are created during forward pass and released or recomputed after backward.

{{< figure src="/llm/distributed_training_3_zero_fsdp/images/static_dynamic_memory.jpg" caption="Static memory vs. dynamic memory in training." align="center" >}}


### Training

During neural network training, the main components of GPU memory usage usually fall into four categories: **model parameters**, intermediate **activations** produced during the forward pass, **gradients** computed during backpropagation, and **optimizer states**. For the memory footprint calculation during training, see the "Memory Reduction" section in the [Distributed Training: ZeRO and FSDP](/llm/distributed_training_3_zero_fsdp/) blog post.

### Inference

During neural network inference, there are no optimizer states or gradients, and there is no need to save intermediate activations. Without gradients, optimizer states, and intermediate activations, GPU memory usage during inference is much smaller than during training.

During inference, the main component of GPU memory usage is usually the model parameters. If `float16` is used for inference, the GPU memory occupied by the model parameters is approximately $2\Phi$ bytes.

If KV cache is used to accelerate inference, the KV cache also occupies GPU memory. See [How Big Is the KV Cache?](/llm/inference_1_kv_cache/#how-big-is-the-kv-cache). In addition, the input data also needs to be placed on the GPU, and there are some intermediate results as well. However, these intermediate results are released as soon as possible during inference, so this part of the memory usage is small and can be ignored.
