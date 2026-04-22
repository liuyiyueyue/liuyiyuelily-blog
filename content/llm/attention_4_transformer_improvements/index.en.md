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

{{< figure src="images/transformer-with-notes.png" alt="Transformer arch with notes" width="900" align="center" >}}

### Decoder-Only Instead of Encoder-Decoder

For autoregressive language modeling, the model only needs to predict the next token from previous tokens. That makes the encoder-decoder split unnecessary. Modern LLMs therefore usually keep only masked self-attention, an MLP block, residual connections, and normalization layers. This simplifies both training and inference. It also makes the model easier to scale, because every block has the same structure and all attention is causal self-attention.

### Pre-Norm Instead of Post-Norm

The original Transformer applies layer normalization after the residual addition:
{{< rawhtml >}}
$$
\begin{aligned}
x &= \operatorname{Norm}(x + \operatorname{Attention}(x)) \\
x &= \operatorname{Norm}(x + \operatorname{MLP}(x))
\end{aligned}
$$
{{< /rawhtml >}}

Many modern LLMs instead use **pre-norm**, where normalization is applied before attention and before the MLP:
{{< rawhtml >}}
$$
\begin{aligned}
x &= x + \operatorname{Attention}(\operatorname{Norm}(x)) \\
x &= x + \operatorname{MLP}(\operatorname{Norm}(x))
\end{aligned}
$$
{{< /rawhtml >}}

This change improves optimization stability in deep networks. In practice, pre-norm makes gradient flow easier and allows models to scale to many more layers without becoming as difficult to train.

Below is a PyTorch code snippet showing the pre-norm formulation. Note that the implementation in [Residual Connection](/llm/attention_1_transformer_basics/#residual-connection) section already uses pre-norm.
```python
class ResidualConnection(nn.Module):
    ...
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```

### RMSNorm Instead of LayerNorm

Many modern language models replace LayerNorm with **RMSNorm** (Root Mean Square Layer Normalization). RMSNorm is a simplified variant that normalizes by the **root mean square** of the activations and does not subtract the mean. The RMSNorm formula is:

{{< rawhtml >}}
$$
\hat{x}_{\mathrm{RMSNorm}} = \frac{x}{\operatorname{RMS}(x) + \epsilon},
\quad \text{where} \quad
\operatorname{RMS}(x) = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}
$$
{{< /rawhtml >}}

This reduces computation while usually preserving similar model quality. In practice, RMSNorm is often preferred because it can effectively mitigate the vanishing-gradient problem in deep networks while maintaining fast training convergence.

Below is a PyTorch code snippet for RMSNorm. This is the RMSNorm counterpart to the [Layer Norm](/llm/attention_1_transformer_basics/#batch-norm-and-layer-norm) example in the original Transformer basics post.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).sqrt()
        return self.weight * x / (rms + self.eps)
```

### RoPE Instead of Absolute Positional Encoding

The original Transformer adds absolute sinusoidal positional encodings to token embeddings. Many modern LLMs instead use **rotary positional embeddings** (**RoPE**).[^1]

In the original Transformer, the positional encoding vector is defined by alternating sine and cosine functions:

{{< rawhtml >}}
$$
\begin{aligned}
\operatorname{PE}(pos, 2i) &= \sin\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right) &= \sin(pos\,\theta_i) \\
\operatorname{PE}(pos, 2i+1) &= \cos\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right) &= \cos(pos\,\theta_i)
\end{aligned}

\qquad

\text{, where } \theta_i = 10000^{-2i/d_{\mathrm{model}}}
$$
{{< /rawhtml >}}


Here, $pos$ is the token position, $i$ indexes the feature dimension, and $d_{\mathrm{model}}$ is the hidden size. The final input representation is:

$$
x_{pos} = e_{pos} + \operatorname{PE}(pos)
$$

where $e_{pos}$ is the token embedding. This works, but the position signal is injected by addition at the input, rather than directly in the attention computation.

RoPE takes a different approach. Instead of adding a positional vector to the token embedding, it rotates the query and key vectors in a position-dependent way before computing attention. In practice, this gives two useful properties:

- position information is encoded directly inside the attention dot product
- the resulting attention depends naturally on relative position differences

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

The key property is that this inner product depends on the position difference $n-m$ rather than on two unrelated absolute positions [^2]:

$$
\tilde{q}_i^\top \tilde{k}_i = q_i^\top R((n-m)\theta_i) k_i
$$

{{< figure src="images/rope.png" alt="Rotary positional embedding illustration" width="750" align="center" >}}

The main implementation difference is that absolute positional encoding is added to the input embeddings, while RoPE is applied directly to the query and key vectors inside attention. As a result, absolute encoding injects position information indirectly through the input representation, whereas RoPE encodes position directly in the attention computation and naturally captures relative position differences.

Here is a PyTorch counterpart to the original [PositionalEncoding](/llm/attention_1_transformer_basics/#positional-encoding) module.

```python
class RoPE(nn.Module):
    """
    Rotary positional embedding for attention query and key tensors.
    """

    def __init__(self, dim: int, seq_len: int, base: float = 10000.0) -> None:
        """
        Precomputes the complex rotation factors for each position.
        cis(theta) = cos(theta) + i sin(theta)

        Args:
            dim (int): Per-head dimension. Must be even.
            seq_len (int): Maximum sequence length supported by the embedding.
            base (float): Base used to compute angular frequencies.
        """
        super().__init__()
        assert dim % 2 == 0

        position = torch.arange(seq_len, dtype=torch.float32)  # (seq_len,)
        theta = base ** (-torch.arange(0, dim, 2, dtype=torch.float32) / dim)  # (dim // 2,)
        freqs = torch.outer(position, theta)  # (seq_len, dim // 2)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # (seq_len, dim // 2)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RoPE to a tensor of shape (batch, n_head, seq_len, dim).

        Args:
            x (torch.Tensor): Query or key tensor.

        Returns:
            torch.Tensor: Tensor with rotary positional embedding applied.
        """
        dtype = x.dtype
        x = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = self.freqs_cis[:x.shape[-2]].view(1, 1, x.shape[-2], x.shape[-1])
        y = torch.view_as_real(x * freqs_cis).flatten(-2)
        return y.to(dtype)
```

DeepSeek V3 implements RoPE with `precompute_freqs_cis()` and `apply_rotary_emb()` [^5]. Unlike absolute positional encoding, RoPE is applied to the query and key tensors inside attention rather than added to the input embedding.

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
[^5]: DeepSeek-V3 GitHub repository. <https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py>
