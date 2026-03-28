---
title: "Transformer"
date: 2023-07-12
tags: ["llm", "transformer"]
math: true
---

## Overview

**Transformer** is a neural network architecture designed to process sequences. It was introduced as an alternative to RNN-based models, with the core idea that a model should not pass information only step by step through time. Instead, it should be able to directly look at the whole sequence and decide which parts matter most.

The key mechanism is **attention**, which aggregates information from a sequence by assigning different weights to different tokens. Because of this, a Transformer can model long-range dependencies more effectively than traditional recurrent models.

A standard Transformer contains an **encoder** and a **decoder**.

- The **encoder** is built from repeated blocks of multi-head self-attention, residual connections, feed-forward layers, and layer normalization.
- The **decoder** is built from masked multi-head self-attention, encoder-decoder attention, feed-forward layers, residual connections, and layer normalization.

Besides the encoder and decoder, the Transformer also includes embedding layers, positional encodings, and an output projection followed by a softmax layer.

![Transformer architecture](transformer.png)

In practice, modern large language models often use decoder-only variants, but the original encoder-decoder design is still the standard starting point for understanding the architecture.

## Transformer vs. RNN

Transformer and RNN both aim to model sequence data, and both use nonlinear layers such as MLPs to transform representations into richer semantic spaces. The main difference is **how they pass sequence information** (如何传递序列信息).

In an **RNN**, information is propagated recurrently:

- the hidden state at time step `t` is passed to time step `t + 1`
- each step depends on previous steps in order
- computation is naturally sequential

This design makes it difficult to **parallelize training** across tokens (并行计算能力). It also makes learning **long-range dependencies** harder (全局信息交互), because information has to travel through many recurrent steps.

In a **Transformer**, sequence information is propagated through attention:

- each token can directly attend to all relevant tokens in the sequence
- the model does not need to move information one step at a time
- training is much more parallelizable

In summary, the difference is not that one understands sequences and the other does not. Both do. The difference is how sequence information is transmitted:

- **RNN**: passes information forward through recurrent hidden states
- **Transformer**: aggregates information globally through attention


## Attention and Its Math

Attention measures how much one token should focus on another token. It can be understood as a weighted aggregation of sequence information, regardless of distance in the sequence.

At a high level:

- `weight = similarity(query, key)`
- `output = weighted sum of values`

### Self-Attention

In **self-attention**, `Q`, `K`, and `V` all come from the same input sequence:

- **Query (Q)**: what the current token is looking for
- **Key (K)**: what each token offers for matching
- **Value (V)**: the information carried by each token

So self-attention means that each token compares itself with all tokens in the same sequence and then gathers the most relevant information.

### Masked Attention

In **masked self-attention**, token `t` is not allowed to see tokens after `t`. This is required in autoregressive language models, where prediction at position `t` must not use future tokens.

### Multi-Head Attention

**Multi-head attention** means using several attention heads in parallel. Each head can learn a different type of relation, such as syntax, local dependency, or long-range dependency. This is loosely similar to how different CNN channels can capture different patterns.

![Scaled Dot-Product Attention. Multi-Head Attention](scaled_dot-product_attention_and_MHA.png)

### Scaled Dot-Product Attention

The Transformer paper defines scaled dot-product attention as:

{{< rawhtml >}}
$$
\operatorname{Attention}(Q, K, V)
=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$
{{< /rawhtml >}}

Parameters:
- $Q \in \mathbb{R}^{n \times d_k}$ is the query matrix, $K \in \mathbb{R}^{m \times d_k}$ is the key matrix, and $V \in \mathbb{R}^{m \times d_v}$ is the value matrix. In self-attention, $m = n$.
- $n$ is the number of query tokens, $m$ is the number of key/value tokens, and $d_k$ is the query/key feature dimension.

Breaking down the equation:
- The term $QK^\top \in \mathbb{R}^{n \times m}$ is the inner product (cosine) and measures **similarity** between tokens.
- $\sqrt{d_k}$ rescales the scores for numerical stability.
- $\frac{QK^\top}{\sqrt{d_k}}$ is called the **attention weight matrix** or **score matrix**.
- $softmax$ turns the scores into attention weights used to combine the values.

So we calculated the attetion using two matrix multiplications. This makes parallel execution easy. The images below illustrates these two matrix multiplications [^1]:

![Compute attention scores](scaled-dot-product-attention-step-1.png)

![Apply causal mask](scaled-dot-product-attention-step-2.png)

![Multiply by values](scaled-dot-product-attention-step-3.png)

![Attention mechanism](attention.png)


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

When the hidden dimension $n$ is large, the linear terms can be neglected, so the total parameter count is approximately

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




### FLOPs and Memory

If you want to analyze Transformer performance numerically, the first question is usually "how much compute and memory does one layer need?"

For a general matrix multiplication, two nested loops iterates over rows and columns.

`(M x N) * (N x K) ~= 2MNK` FLOPs

This leads to the standard roofline view:

- large GEMMs such as QKV projection and FFN are usually **compute-bound**
- decode-time attention is often **memory-bound** because batch size is small and the kernel keeps reading weights and KV cache
- softmax, layer norm, and other elementwise ops are also often **memory-bound**


## Layer Norm vs. Batch Norm

Both layer normalization and batch normalization are used to stabilize training, but they normalize over different dimensions.

**Batch normalization** computes statistics across the batch. Its behavior depends on the distribution of examples inside the mini-batch. This works well in many vision settings, but it is less suitable for sequence models when:

- sequence lengths vary a lot
- token distributions change across positions
- batch statistics become unstable or less meaningful

**Layer normalization** computes statistics within each individual token representation. It does not depend on other examples in the batch, which makes it more stable for variable-length sequence modeling.

This is why Transformers typically use **layer normalization instead of batch normalization**. For language tasks, each token representation should be normalized independently, without relying on the composition of the current mini-batch.


## Positional Encoding

Positional encoding provides the model with information about the positions of words in a sequence. Since the Transformer's self-attention mechanism does not naturally account for the order of elements in the sequence, positional encoding solves this by adding position information to each element's representation. In the original Transformer paper, positional encodings are defined using alternating sine and cosine functions.


## FFN

Feed-forward network (FFN) provides the model with nonlinear processing capacity. It operates independently on each position of the input, helping increase the model's complexity and expressive power.

## Complete Code

[RethinkFun/DeepLearning `chapter15/transformer.py`](https://github.com/RethinkFun/DeepLearning/blob/master/chapter15/transformer.py)

[^1]: Transformer模型详解（图解最完整版）. 初识CV, Zhihu. <https://zhuanlan.zhihu.com/p/338817680>
