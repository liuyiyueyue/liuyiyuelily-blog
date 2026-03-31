---
title: "Transformer: Basics"
date: 2023-07-12
tags: ["llm", "transformer"]
math: true
---

### Overview

**Transformer** is a neural network architecture designed to process sequences. It was introduced as an alternative to RNN-based models, with the core idea that a model should not pass information only step by step through time. Instead, it should be able to directly look at the whole sequence and decide which parts matter most.

The key mechanism is **attention**, which aggregates information from a sequence by assigning different weights to different tokens. Because of this, a Transformer can model long-range dependencies more effectively than traditional recurrent models.[^1]

A standard Transformer contains an **encoder** and a **decoder**.

- The **encoder** is built from repeated blocks of multi-head self-attention, residual connections, feed-forward layers, and layer normalization.
- The **decoder** is built from masked multi-head self-attention, encoder-decoder attention, feed-forward layers, residual connections, and layer normalization.

Besides the encoder and decoder, the Transformer also includes embedding layers, positional encodings, and an output projection followed by a softmax layer.

Below is the architecture of the standard Transformer [^2]:

![Transformer architecture](images/transformer.png)

In practice, modern large language models often use decoder-only variants, but the original encoder-decoder design is still the standard starting point for understanding the architecture.

### Transformer vs. RNN

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


### Attention and Its Math

Attention measures how much one token should focus on another token. It can be understood as a weighted aggregation of sequence information, regardless of distance in the sequence.

At a high level:

- `weight = similarity(query, key)`
- `output = weighted sum of values`

**Self-Attention**

In self-attention, `Q`, `K`, and `V` all come from the same input sequence:

- **Query (Q)**: what the current token is looking for
- **Key (K)**: what each token offers for matching
- **Value (V)**: the information carried by each token

So self-attention means that each token compares itself with all tokens in the same sequence and then gathers the most relevant information.

**Masked Attention**

In masked self-attention, token `t` is not allowed to see tokens after `t`. This is required in autoregressive language models, where prediction at position `t` must not use future tokens.

**Multi-Head Attention (MHA)**

Multi-head attention means using several attention heads in parallel. Each head can learn a different type of relation, such as syntax, local dependency, or long-range dependency. This is loosely similar to how different CNN channels can capture different patterns.

![Scaled Dot-Product Attention. Multi-Head Attention](images/scaled_dot-product_attention_and_MHA.png)

**Scaled Dot-Product Attention**

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

So we calculated the attetion using two matrix multiplications. This makes parallel execution easy. The images below illustrates these two matrix multiplications [^3]:

{{< figure src="images/scaled-dot-product-attention-step-1.png" alt="Compute attention scores" width="420" align="center" >}}

{{< figure src="images/scaled-dot-product-attention-step-2.png" alt="Apply causal mask" width="420" align="center" >}}

{{< figure src="images/scaled-dot-product-attention-step-3.png" alt="Multiply by values" width="420" align="center" >}}

![Attention mechanism](images/attention.png)


### Layer Norm vs. Batch Norm

Both layer normalization and batch normalization are used to stabilize training, but they normalize over different dimensions.

**Batch normalization** computes statistics across the batch. Its behavior depends on the distribution of examples inside the mini-batch. This works well in many vision settings, but it is less suitable for sequence models when:

- sequence lengths vary a lot
- token distributions change across positions
- batch statistics become unstable or less meaningful

**Layer normalization** computes statistics within each individual token representation. It does not depend on other examples in the batch, which makes it more stable for variable-length sequence modeling.

This is why Transformers typically use **layer normalization instead of batch normalization**. For language tasks, each token representation should be normalized independently, without relying on the composition of the current mini-batch.


### Positional Encoding

Positional encoding provides the model with information about the positions of words in a sequence. Since the Transformer's self-attention mechanism does not naturally account for the order of elements in the sequence, positional encoding solves this by adding position information to each element's representation. In the original Transformer paper, positional encodings are defined using alternating sine and cosine functions.


### MLP

In a Transformer block, attention and the **feed-forward network** (**FFN**, also called the **MLP** block) play different roles.

{{< figure src="images/ffn.png" alt="Attention and FFN" width="420" align="center" >}}

The FFN block typically consists of two linear layers with a nonlinear activation in between:

$$
x = f_{\text{gelu}}(x_{\text{out}}W_1)W_2 + x_{\text{out}}
$$

只用MLP为啥不行呢？MLP 通过全连接层实现全局特征之间的交互，但其计算开销过高，因此难以无限制地向更深层堆叠。Transformer 则通过 attention 机制实现全局特征之间的选择性交互，在保留全局信息建模能力的同时，提供了更高效的结构化建模方式。

### Complete Code

[RethinkFun/DeepLearning `chapter15/transformer.py`](https://github.com/RethinkFun/DeepLearning/blob/master/chapter15/transformer.py)

[^1]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. arXiv, June 12, 2017. <https://arxiv.org/abs/1706.03762>
[^2]: Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. Efficient Transformers: A Survey. arXiv, September 14, 2020. <https://arxiv.org/abs/2009.06732>
[^3]: Transformer模型详解（图解最完整版）. 初识CV, Zhihu. <https://zhuanlan.zhihu.com/p/338817680>
