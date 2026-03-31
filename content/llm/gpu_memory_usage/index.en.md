---
title: "Memory Usage"
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

During neural network training, the main components of GPU memory usage usually fall into four categories: **model parameters**, intermediate **activations** produced during the forward pass, **gradients** computed during backpropagation, and **optimizer states**.

When training large models, it is common to use the AdamW optimizer together with mixed-precision training to accelerate computation. Based on this setup, we can analyze memory consumption as follows.

In one training iteration, each trainable model parameter corresponds to one gradient, and two optimizer states in AdamW: the first-order moment and the second-order moment.

Let the total number of model parameters be $\Phi$. Then the number of gradient elements is $\Phi$, and the number of AdamW optimizer-state elements is $2\Phi$. Each element of type `float16` occupies 2 bytes, and each element of type `float32` occupies 4 bytes.

In mixed-precision training, `float16` model parameters are used for the forward pass and backward pass. `float16` gradients are produced during backpropagation. During the optimizer update, `float32` optimizer states, `float32` gradients, and `float32` model parameters are used to update the parameters.

Therefore, for each trainable parameter, the memory cost is:

$$
(2+4) + (2+4) + (4+4) = 20 \text{ bytes}
$$

where:

- $(2+4)$ corresponds to the weights: `float16` weight + `float32` weight,
- $(2+4)$ corresponds to the gradient: `float16` gradient + `float32` gradient,
- $(4+4)$ corresponds to the two AdamW optimizer states, both stored in `float32`.

So, when training a large model with $\Phi$ parameters using AdamW and mixed-precision training, the total GPU memory occupied by model parameters, gradients, and optimizer states is $20\Phi \text{ bytes}$.

### Inference

During neural network inference, there are no optimizer states or gradients, and there is no need to save intermediate activations. Without gradients, optimizer states, and intermediate activations, GPU memory usage during inference is much smaller than during training.

During inference, the main component of GPU memory usage is usually the model parameters. If `float16` is used for inference, the GPU memory occupied by the model parameters is approximately $2\Phi$ bytes.

If KV cache is used to accelerate inference, the KV cache also occupies GPU memory. See [How Big Is the KV Cache?](/llm/inference_1_kv_cache/#how-big-is-the-kv-cache). In addition, the input data also needs to be placed on the GPU, and there are some intermediate results as well. However, these intermediate results are released as soon as possible during inference, so this part of the memory usage is small and can be ignored.
