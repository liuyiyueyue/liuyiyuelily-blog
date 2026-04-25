---
title: "Transformer: Parameter Count, FLOPs, Memory Usage, and Training Time"
date: 2023-07-13
tags: ["llm", "transformer"]
math: true
---

This post continues the [Transformer](/llm/attention_1_transformer/) article and focuses on model size and compute estimation.

### Parameter Count

A Transformer model consists of $l$ identical layers. Each layer has two parts: a self-attention block and an MLP block.

**The self-attention block** contains the weight matrices $W_Q$, $W_K$, $W_V$, and $W_O$, together with their biases. The four weight matrices satisfy $W_Q, W_K, W_V, W_O \in \mathbb{R}^{n \times n}$, and the four bias vectors satisfy $b_Q, b_K, b_V, b_O \in \mathbb{R}^{n}$. Therefore, the number of parameters in the self-attention block is 

{{< rawhtml >}}
$$
4n^2 + 4n
$$
{{< /rawhtml >}}

**The MLP block** consists of two linear layers. In general, the first linear layer maps the hidden dimension from $n$ to $4n$, and the second maps it back from $4n$ to $n$. For the first linear layer, $W_1 \in \mathbb{R}^{n \times 4n}$ and $b_1 \in \mathbb{R}^{4n}$. For the second linear layer, $W_2 \in \mathbb{R}^{4n \times n}$ and $b_2 \in \mathbb{R}^{n}$. So the total number of parameters in the MLP block is 

{{< rawhtml >}}
$$
8n^2 + 5n
$$
{{< /rawhtml >}}

The self-attention block and the MLP block each have one layer normalization. Each layer normalization has two trainable parameters: the scale parameter $\gamma$ and the shift parameter $\beta$, with $\gamma, \beta \in \mathbb{R}^{n}$. Therefore, the two layer normalizations contribute $4n$ parameters in total.

Hence, the total number of parameters in each Transformer layer is 

{{< rawhtml >}}
$$
(4n^2 + 4n) + (8n^2 + 5n) + 4n = 12n^2 + 13n
$$
{{< /rawhtml >}}

In addition, **the token embedding matrix** contributes a large number of parameters. Since the embedding dimension is usually equal to the hidden dimension $n$, the embedding matrix $E \in \mathbb{R}^{V \times n}$ has $Vn$ parameters, where $V$ is the vocabulary size.

The final output projection usually shares its weight matrix with the token embedding matrix, so it does not introduce an additional $Vn$ parameters in that case. For positional encoding, trainable positional embeddings introduce some additional parameters, but relatively few. If relative positional encoding is used, such as RoPE or ALiBi, then this part has no trainable parameters. We ignore positional-encoding parameters here.

Therefore, for a Transformer model with $l$ layers, **the total number of trainable parameters** (参数量) is 

{{< rawhtml >}}
$$
l(12n^2 + 13n) + Vn
$$
{{< /rawhtml >}}

When the hidden dimension $n$ is large, the linear terms can be neglected, so the total parameter count is **approximately**

{{< rawhtml >}}
$$
12ln^2
$$
{{< /rawhtml >}}

Using the approximation $P \approx 12ln^2$, we can estimate the parameter counts of several well-known Llama 3 models:

| Model | $l$ | $n$ | $12ln^2$ | Approx. |
| --- | ---: | ---: | ---: | ---: |
| 8B | 32 | 4096 | $6{,}442{,}450{,}944$ | $6.44\text{B}$ |
| 70B | 80 | 8192 | $64{,}424{,}509{,}440$ | $64.42\text{B}$ |
| 405B | 126 | 16384 | $405{,}874{,}409{,}472$ | $405.87\text{B}$ |

This is only a rough estimate. It is close for 405B, but it underestimates the 8B and 70B models because the formula ignores embeddings, output-layer choices, grouped-query attention details, and exact MLP dimensions.

### FLOPs

> FLOPs (计算量), or floating-point operations, measure computational cost.

> For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, the matrix multiplication $AB$ costs $2mnp$ FLOPs.

In one training iteration, let the input token shape be $[b, s]$, where $b$ is the batch size and $s$ is the sequence length. After embedding, the hidden states are $x \in \mathbb{R}^{b \times s \times h}$, where $h$ is the hidden size. 

For a **self-attention** block,

$$
Q = xW_Q,\quad K = xW_K,\quad V = xW_V
$$

$$
x_{\text{out}} = \operatorname{softmax}\left(\frac{QK^T}{\sqrt{h}}\right)V W_o + x
$$

1. Computing $Q$, $K$, and $V$: $x \in \mathbb{R}^{b \times s \times h}$, $W_Q \in \mathbb{R}^{h \times h}$, $W_K \in \mathbb{R}^{h \times h}$, and $W_V \in \mathbb{R}^{h \times h}$, so $Q \in \mathbb{R}^{b \times s \times h}$, $K \in \mathbb{R}^{b \times s \times h}$, and $V \in \mathbb{R}^{b \times s \times h}$. The FLOPs are:
   $$
   3 \cdot 2bsh^2 = 6bsh^2
   $$

2. Computing $QK^T$: $Q \in \mathbb{R}^{b \times n_{\mathrm{head}} \times s \times d_{\mathrm{head}}}$ and $K^T \in \mathbb{R}^{b \times n_{\mathrm{head}} \times d_{\mathrm{head}} \times s}$, so $QK^T \in \mathbb{R}^{b \times n_{\mathrm{head}} \times s \times s}$. So the FLOPs are:
   $$
   2bs^2h
   $$

3. Computing $\text{score} \cdot V$:
   $\text{score} \in \mathbb{R}^{b \times n_{\mathrm{head}} \times s \times s}$ and $V \in \mathbb{R}^{b \times n_{\mathrm{head}} \times s \times d_{\mathrm{head}}}$, so $\text{score} \cdot V \in \mathbb{R}^{b \times n_{\mathrm{head}} \times s \times d_{\mathrm{head}}}$. So the FLOPs are:
   $$
   2bs^2h
   $$

4. The output projection after attention:
   $x_{\text{attn}} \in \mathbb{R}^{b \times s \times h}$ and $W_o \in \mathbb{R}^{h \times h}$, so $x_{\text{attn}}W_o \in \mathbb{R}^{b \times s \times h}$. So the FLOPs are:
   $$
   2bsh^2
   $$

For the **MLP** block,

$$
x = f_{\text{gelu}}(x_{\text{out}}W_1)W_2 + x_{\text{out}}
$$

1. First linear layer:
   $x_{\text{out}} \in \mathbb{R}^{b \times s \times h}$ and $W_1 \in \mathbb{R}^{h \times 4h}$, so $x_{\text{out}}W_1 \in \mathbb{R}^{b \times s \times 4h}$. So the FLOPs are:
   $$
   8bsh^2
   $$
2. Second linear layer:
   $f_{\text{gelu}}(x_{\text{out}}W_1) \in \mathbb{R}^{b \times s \times 4h}$ and $W_2 \in \mathbb{R}^{4h \times h}$, so $f_{\text{gelu}}(x_{\text{out}}W_1)W_2 \in \mathbb{R}^{b \times s \times h}$. So the FLOPs are:
   $$
   8bsh^2
   $$

Adding them together, the FLOPs needed by **one Transformer layer** are approximately

$$
24bsh^2 + 4bs^2h
$$

At the very end of the Transformer architecture, there is the **final linear layer (logits)** that maps hidden states to vocabulary scores.
Given $h_{\text{out}} \in \mathbb{R}^{b \times s \times h} \quad$ and $W_{\text{vocab}} \in \mathbb{R}^{h \times V}$, we have $h_{\text{out}}W_{\text{vocab}} \in \mathbb{R}^{b \times s \times V}$. So the FLOPs are:

$$
2bshV
$$

Therefore, for a Transformer with $l$ layers and input shape $[b, s]$, the FLOPs of **one training iteration** are approximately

$$
l(24bsh^2 + 4bs^2h) + 2bshV
$$

Most people write it concisely **using big-O** below, ignoring the $b$ and $l$ constants. Again, $s$ is sequence length, and $h$ is hidden size. We can see that both the compute cost of the Transformer grow **quadratically with sequence length**.

$$
O(sh^2 + s^2h) \text{ or } O(s^2h) \text{ or } O(s^2) 
$$


### Activation Memory

Besides model parameters, gradients, and optimizer states, one of the largest contributors to GPU memory usage is the intermediate activations produced during the forward pass. These activations must be saved so they can be reused during backpropagation when computing gradients. Here, **activations** refer to all tensors that are computed during the forward pass and are needed again during the backward pass. This does not include model parameters or optimizer states, but it does include tensors such as the mask matrices required by dropout.

Throughout this section, activations are assumed to be stored in `fp16` or `bf16`, so each element takes 2 bytes. The only exception is the dropout mask, where each element takes 1 byte. Below, we give both the number of elements and the corresponding activation memory usage in bytes.

Let $a$ denote the number of attention heads, and let $d_{\mathrm{head}} = h / a$.

For a **self-attention** block,

1. For the projections $Q$, $K$, and $V$, the common input $x$ must be saved as an activation. Here
   $x \in \mathbb{R}^{b \times s \times h}$, so it has $bsh$ elements and uses $2bsh$ bytes.

2. For the matrix multiplication $QK^T$, the activations $Q$ and $K$ must be saved. Both satisfy
   $Q, K \in \mathbb{R}^{b \times s \times h}$, so together they have $2bsh$ elements and use $4bsh$ bytes.

3. For the softmax operation, its input must be saved. Then
   $Q \in \mathbb{R}^{b \times a \times s \times d_{\mathrm{head}}}$,
   $K^T \in \mathbb{R}^{b \times a \times d_{\mathrm{head}} \times s}$, and
   $QK^T \in \mathbb{R}^{b \times a \times s \times s}$, so the saved softmax input has $b a s^2$ elements and uses $2b a s^2$ bytes.

4. After softmax, dropout is applied. The dropout mask has the same shape as the attention score matrix:
   $M_{\mathrm{drop}} \in \mathbb{R}^{b \times a \times s \times s}$, so it has $b a s^2$ elements and uses $b a s^2$ bytes.

5. For the attention output
   $$
   \operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_{\mathrm{head}}}}\right)V,
   $$
   both the attention matrix and $V$ must be saved. Their shapes are
   $\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_{\mathrm{head}}}}\right) \in \mathbb{R}^{b \times a \times s \times s}$ and
   $V \in \mathbb{R}^{b \times a \times s \times d_{\mathrm{head}}}$, so together they have $b a s^2 + bsh$ elements and use $2b a s^2 + 2bsh$ bytes.

6. For the output projection and the final dropout, the projection input and the dropout mask must be saved. The projection input satisfies
   $x_{\mathrm{attn}} \in \mathbb{R}^{b \times s \times h}$, which contributes $bsh$ elements and $2bsh$ bytes, and the dropout mask contributes another $bsh$ elements and $bsh$ bytes.

Adding these terms together, the intermediate activation memory of the self-attention block is approximately

$$
11bsh + 5b a s^2
$$

For a **MLP** block:

1. The input to the first linear layer must be saved. Since $x \in \mathbb{R}^{b \times s \times h}$, this uses $2bsh$.
   That is, it has $bsh$ elements and uses $2bsh$ bytes.

2. The input to the activation function must be saved. Since $xW_1 \in \mathbb{R}^{b \times s \times 4h}$, this has $4bsh$ elements and uses $8bsh$ bytes.

3. The input to the second linear layer must be saved. Since $f(xW_1) \in \mathbb{R}^{b \times s \times 4h}$, this has $4bsh$ elements and uses $8bsh$ bytes.

4. The final dropout mask has the same shape as the MLP output, so it has $bsh$ elements and uses $bsh$ bytes.

Therefore, the MLP block saves about $10bsh$ elements, corresponding to activation memory

$$
19bsh
$$

In addition, the self-attention block and the MLP block each have a layer normalization. Each layer normalization must save its input, where $x \in \mathbb{R}^{b \times s \times h}$, so each contributes $bsh$ elements and uses $2bsh$ bytes. Together, the two layer normalizations contribute $2bsh$ elements and use $4bsh$ bytes.

In total, one Transformer layer saves about $18bsh + 3b a s^2$ elements, corresponding to activation memory

$$
34bsh + 5b a s^2
$$

Therefore, **for an $l$-layer Transformer model, the intermediate activation memory can be approximated as**

$$
l(34bsh + 5b a s^2)
$$

### Training

**Parameter Count and FLOPs**

When the hidden size $h$ is large and $h \gg s$, we can ignore the linear terms and approximate the compute in FLOPs as $24bsh^2 l$.

As noted in previous sections, a Transformer model with $l$ layers has around $12lh^2$ parameters, and the input contains $bs$ tokens, then:

$$
\frac{24bsh^2l}{12lh^2 \times bs} = 2 \ \text{FLOPs/token-parameter}
$$

This means that, in one forward pass, each token-parameter pair requires about 2 floating-point operations: one multiplication and one addition. One can also says that each parameter requires about 2 floating-point operations per token.

One training step includes both a forward pass and a backward pass. The backward pass costs about 2 times the forward pass. 
Therefore, in one training step, each token-parameter pair requires:

$$
2 \times (1 + 2) = 6 \ \text{FLOPs/token-parameter}
$$

We can then estimate the total training compute for GPT-3. For GPT-3 175B, each token-parameter pair uses 6 FLOPs, so the total compute is 6 times the parameter count times the total number of training tokens. GPT-3 has $174600\text{M}$ parameters and is trained on $300\text{B}$ tokens:

$$
6 \times (174600 \times 10^{6}) \times (300 \times 10^{9})
= 3.1428 \times 10^{23}\ \text{FLOPs}
$$

**Training Time**

The amount of compute required to train a Transformer model is determined by the model parameter count and the total number of training tokens. Once the GPU type is fixed, we can estimate the required training time. For a given amount of compute, training time, that is, the time GPUs need to execute that many FLOPs, depends not only on the GPU type but also on GPU utilization. When estimating end-to-end GPU utilization, we need to account for more than just forward and backward computation. We must also include the time spent on CPU data loading, optimizer updates, multi-GPU communication, and logging. In practice, GPU utilization is often in the range of 0.3 to 0.55.

As discussed above, in one forward pass, each token-parameter pair requires 2 floating-point operations. If activation recomputation is used to reduce activation memory, one additional forward pass is needed. Therefore, the coefficient for forward pass + backward pass + activation recomputation is $1 + 2 + 1 = 4$. In one training iteration with activation recomputation, each token-parameter pair requires:

$$
2 \times 4 = 8 \ \text{FLOPs/token-parameter}
$$

Given the number of training tokens and the hardware configuration, the training time of a Transformer model can be estimated as:

$$
\text{Training time} \approx \frac{8 \times \text{training tokens} \times \text{parameter count}}{\text{number of GPUs} \times \text{peak GPU FLOPs} \times \text{GPU utilization}}
$$

**Memory Usage**

In one training iteration, the GPU memory occupied by model parameters (or gradients) depends only on the number of model parameters and the parameter data type. It does **not** depend on the input size. The same is true for optimizer states: their memory usage depends on the optimizer type and the parameter count, but not on the input batch. Intermediate activations are different. Their memory usage grows with the input size, especially with batch size $b$ and sequence length $s$. As $b$ and $s$ increase, activation memory increases accordingly. This is why, when training runs into OOM issues, reducing the batch size often helps: it reduces activation memory, not the memory occupied by parameters, gradients, or optimizer states.

Take GPT-3 175B as a concrete example. Its model configuration is:

| Model | Parameters | Layers $l$ | Hidden size $h$ | Attention heads $a$ |
| --- | ---: | ---: | ---: | ---: |
| GPT-3 | 175B | 96 | 12288 | 96 |

Assume mixed-precision training, with both model parameters and activations stored in `fp16`, so each element takes 2 bytes.

GPT-3 has 175B parameters, so the parameter memory is approximately:

$$
175 \times 10^9 \times 2 \text{ bytes} = 350 \text{ GB}
$$

Now let the sequence length be $s = 2048$. Using the activation-memory estimate derived above,

$$
l(34bsh + 5b a s^2),
$$

we can compare the activation memory at different batch sizes:

- When $b = 1$, the activation memory is about $275 \text{ GB}$, which is about $0.79\times$ the parameter memory.
- When $b = 64$, the activation memory is about $17.6 \text{ TB}$, which is about $50\times$ the parameter memory.
- When $b = 128$, the activation memory is about $35.3 \text{ TB}$, which is about $101\times$ the parameter memory.

**As batch size increases, activation memory quickly grows far beyond parameter memory.** In practice, activation recomputation is commonly used to reduce this cost. In exchange for one extra forward-pass worth of compute, it reduces the amount of activation memory that must be stored. This is fundamentally a time-for-space tradeoff.

### Compute-Bound vs. Memory-Bound

The **roofline model** gives a simple way to think about GPU performance. It says that the throughput of an operation is limited by two ceilings: peak compute throughput and peak memory bandwidth. If an operation hits the compute ceiling, it is **compute-bound**. If it hits the bandwidth ceiling, it is **memory-bound**.

{{< figure src="./images/roofline.png" caption="Roofline model: throughput is limited by either compute throughput or memory bandwidth, depending on arithmetic intensity." align="center" >}}

The key quantity is **arithmetic intensity**, the number of floating-point operations performed per byte of data moved:

$$
\text{Arithmetic intensity} = \frac{\text{FLOPs}}{\text{bytes moved}}
$$

Operations with high arithmetic intensity tend to be compute-bound, while operations with low arithmetic intensity tend to be memory-bound.

Now, back to transformer.

With a large batch size, matrix multiplications such as the $QKV$ projections and the FFN layers are usually compute-bound. Their compute grows with batch size, while data movement grows more slowly, so their arithmetic intensity is relatively high. 

During decode stage, however, computation is usually memory-bound. The batch size is often very small, sometimes even 1, so matrix multiplications effectively become matrix-vector multiplications, which greatly lowers arithmetic intensity. In that regime, the bottleneck often becomes the memory bandwidth required to read model weights from HBM. 

Elementwise operations such as softmax and layer normalization are also usually memory-bound, because their arithmetic cost is small relative to the amount of data they move.
