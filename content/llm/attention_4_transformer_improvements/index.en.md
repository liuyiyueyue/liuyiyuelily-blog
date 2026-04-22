---
title: "Transformer: Improvements"
date: 2026-04-04
tags: ["llm", "transformer", "optimization"]
math: true
---

This post continues the Transformer series and focuses on the changes that made the original architecture work better for large language models. The core idea of attention is still the same, but several parts of the 2017 design are now commonly replaced:

- encoder-decoder --> decoder-only
- post-norm --> pre-norm
- LayerNorm --> RMSNorm
- absolute positional encoding --> RoPE
- MHA --> GQA or MQA
- GELU --> gated MLP such as SwiGLU

As a refresher, the below is the original transformer architecture:

{{< figure src="images/transformer-with-notes.png" alt="Transformer arch with notes" width="650" align="center" >}}

### Decoder-Only Instead of Encoder-Decoder

For autoregressive language modeling, the model only needs to predict the next token from previous tokens. That makes the encoder-decoder split unnecessary. Modern LLMs therefore usually keep only masked self-attention, an MLP block, residual connections, and normalization layers. This simplifies both training and inference. It also makes the model easier to scale, because every block has the same structure and all attention is causal self-attention.

### Pre-Norm Instead of Post-Norm

The original Transformer applies layer normalization after the residual addition. Many modern LLMs instead use **pre-norm**, where normalization is applied before attention and before the MLP:

$$
x = x + \operatorname{Attention}(\operatorname{Norm}(x))
$$

$$
x = x + \operatorname{MLP}(\operatorname{Norm}(x))
$$

This change improves optimization stability in deep networks. In practice, pre-norm makes gradient flow easier and allows models to scale to many more layers without becoming as difficult to train.

### RMSNorm Instead of LayerNorm

Many modern language models replace LayerNorm with **RMSNorm**. RMSNorm is a simplified variant that normalizes by the **root mean square** of the activations and does not subtract the mean. This reduces computation while usually preserving similar model quality. In practice, RMSNorm is often preferred because it can effectively mitigate the vanishing-gradient problem in deep networks while maintaining fast training convergence.

The corresponding formulas are:

{{< rawhtml >}}
$$
\hat{x}_{\mathrm{RMSNorm}} = \frac{x}{\operatorname{RMS}(x) + \epsilon},
\quad \text{where} \quad
\operatorname{RMS}(x) = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}
$$
{{< /rawhtml >}}

### RoPE Instead of Absolute Positional Encoding

The original Transformer adds absolute sinusoidal positional encodings to token embeddings. Many modern LLMs instead use **rotary positional embeddings** (**RoPE**).[^1]

In the original Transformer, the positional encoding vector is defined by alternating sine and cosine functions:

{{< rawhtml >}}
$$
\operatorname{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right)
$$
{{< /rawhtml >}}

{{< rawhtml >}}
$$
\operatorname{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right)
$$
{{< /rawhtml >}}

Here, $pos$ is the token position, $i$ indexes the feature dimension, and $d_{\mathrm{model}}$ is the hidden size. The final input representation is:

$$
x_{pos} = e_{pos} + \operatorname{PE}(pos)
$$

where $e_{pos}$ is the token embedding.

At a high level, RoPE has two practical advantages:

- it encodes **relative position information** naturally inside attention
- it usually extrapolates to longer sequence lengths better than fixed learned absolute embeddings

RoPE does not simply add a position vector to the hidden state. Instead, it rotates the query and key vectors in a position-dependent way before attention is computed. This means position information is injected directly into the attention dot product.

For each 2D pair of features, RoPE applies a rotation matrix that depends on the token position $m$:

{{< rawhtml >}}
$$
R(m\theta_i)=
\begin{bmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{bmatrix}
$$
{{< /rawhtml >}}

If $q_i$ and $k_i$ denote the $i$-th 2D blocks of the query and key, then after applying RoPE at positions $m$ and $n$:

$$
\tilde{q}_i = R(m\theta_i) q_i,\quad \tilde{k}_i = R(n\theta_i) k_i
$$

The attention score is then computed using the rotated vectors:

$$
\operatorname{score}(m,n) = \sum_i \tilde{q}_i^\top \tilde{k}_i
$$

The important property is that the inner product depends on the position difference $n-m$ [^2]:

$$
\tilde{q}_i^\top \tilde{k}_i = q_i^\top R((n-m)\theta_i) k_i
$$

{{< figure src="images/rope.png" alt="Rotary positional embedding illustration" width="750" align="center" >}}

So RoPE turns absolute positions into a form that naturally expresses **relative position** inside attention. That is the main reason it works so well for causal language models.

### GQA and MQA Instead of MHA

In standard multi-head attention (MHA), every head has its own query, key, and value projections. During inference, the KV cache therefore grows with the number of attention heads. This becomes expensive for large models and long sequences. So if the number of KV heads is smaller than the number of query heads, KV-cache memory becomes smaller:

```text
KV cache size = num_layers × seq_len × num_kv_heads × head_dim × 2 × dtype_size
```

Two common improvements are:

- **GQA**: several query heads share one key head and one value head per group [^4]. GQA keeps more modeling flexibility and is often a better quality-efficiency tradeoff
- **MQA**: many query heads share one key head and one value head [^3]. MQA gives the largest KV-cache reduction.

{{< figure src="images/mha_mqa_gqa.png" alt="Rotary positional embedding illustration" width="750" align="center" >}}


### GLU instead of GELU in FFN/MLP

The original Transformer uses a simple two-layer feed-forward network:

$$
\hat{x}_{\text{gelu}} = \operatorname{GELU}(xW_1)W_2
$$

Modern LLMs often replace it with a **gated MLP**, commonly **SwiGLU**, which is a hybrid activation function that combines the Swish activation function with a gated linear unit (GLU).

$$
\hat{x}_{\text{swiglu}} = (\operatorname{SiLU}(xW_1) \odot xW_2)W_3
$$

The gate lets the model control which information should pass through the MLP. In practice, gated MLPs usually improve model quality enough to justify the additional structure.

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.Wa = nn.Linear(d_model, hidden_dim)
        self.Wb = nn.Linear(d_model, hidden_dim)

    def forward(self, x):
        return self.Wa(x) * F.silu(self.Wb(x))


class FFN_SwiGLU(nn.Module):
    def __init__(self, d_model, mlp_ratio=3.0):  # common default in modern LLMs
        super().__init__()
        hidden_dim = int(mlp_ratio * d_model)

        self.fc = SwiGLU(d_model, hidden_dim)
        self.proj = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        return self.proj(self.fc(x))
```

[^1]: Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv, April 20, 2021. <https://arxiv.org/abs/2104.09864>
[^2]: Shreyash Shukla. RoPE (Rotary Positional Embeddings). <https://shreyashkar-ml.github.io/posts/rope/>
[^3]: Noam Shazeer. Fast Transformer Decoding: One Write-Head Is All You Need. arXiv, November 6, 2019. <https://arxiv.org/abs/1911.02150>
[^4]: Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. arXiv, May 22, 2023. <https://arxiv.org/abs/2305.13245>
