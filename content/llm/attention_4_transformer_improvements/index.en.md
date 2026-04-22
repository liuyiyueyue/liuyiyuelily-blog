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
- MHA --> GQA, MQA, or MLA
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

DeepSeek V3 implements RoPE with `precompute_freqs_cis()` and `apply_rotary_emb()` [^5].

### GQA, MQA, and MLA Instead of MHA

In standard multi-head attention (MHA), every head has its own query, key, and value projections. During inference, the KV cache therefore grows with the number of attention heads. This becomes expensive for large models and long sequences. So if the number of KV heads is smaller than the number of query heads, KV-cache memory becomes smaller:

```text
KV cache size = num_layers × seq_len × num_kv_heads × head_dim × 2 × dtype_size
```

Common improvements are:

- **GQA** (Group-Query Attention): several query heads share one key head and one value head per group [^4]. GQA keeps more modeling flexibility and is often a better quality-efficiency tradeoff
- **MQA** (Multi-Query Attention): many query heads share one key head and one value head [^3]. MQA gives the largest KV-cache reduction.
- **MLA** (Multi-Head Latent Attention): introduced in DeepSeek-V2 [^5], it compresses attention information (key and value tensors) into a smaller latent representation and caches that compact state. The model then reconstructs the effective keys and values from the latent state during attention.

{{< figure src="images/mha_mqa_gqa_mla.png" alt="MHA, GQA, MQA, and MLA" width="1000" align="center" >}}

Intuitively, GQA and MQA reduce memory by **sharing** KV heads, while MLA reduces memory by **compressing** KV information into a latent space.

Below is an example of GQA implementation. You can compare it with the original [MultiHeadAttentionBlock](/llm/attention_1_transformer_basics/#multi-head-attention-mha) code. MQA is the special case where `n_kv_head = 1`:

```python
class GroupedQueryAttentionBlock(nn.Module):
    """
    Grouped-query self-attention block.

    Args:
        dim (int): Input and output dimension of each token representation.
        n_head (int): Number of query heads.
        n_kv_head (int): Number of shared key/value heads.
    """

    def __init__(self, dim: int, n_head: int, n_kv_head: int) -> None:
        """
        Initializes the grouped-query attention projections.

        Args:
            dim (int): Input and output dimension of each token representation.
            n_head (int): Number of query heads.
            n_kv_head (int): Number of shared key/value heads.
        """
        super().__init__()
        self.dim = dim
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        assert dim % n_head == 0
        assert n_head % n_kv_head == 0

        self.d_k = dim // n_head  # Per-query-head feature dimension.
        self.n_rep = n_head // n_kv_head  # Number of query heads per KV head.
        self.w_q = nn.Linear(dim, dim, bias=False)  # Wq
        self.w_k = nn.Linear(dim, n_kv_head * self.d_k, bias=False)  # Wk
        self.w_v = nn.Linear(dim, n_kv_head * self.d_k, bias=False)  # Wv
        self.w_o = nn.Linear(dim, dim, bias=False)  # Wo

    @staticmethod
    def repeat_kv(x, n_rep):
        """
        Repeats each KV head so query heads in the same group share it.

        Args:
            x (torch.Tensor): Tensor of shape (batch, n_kv_head, seq_len, d_k).
            n_rep (int): Number of query heads that share each KV head.

        Returns:
            torch.Tensor: Tensor of shape (batch, n_head, seq_len, d_k).
        """
        if n_rep == 1:
            return x
        return x.repeat_interleave(n_rep, dim=1)

    @staticmethod
    def attention(query, key, value, mask):
        """
        Computes scaled dot-product attention for all query heads.

        Args:
            query (torch.Tensor): Query tensor of shape (batch, n_head, seq_len, d_k).
            key (torch.Tensor): Key tensor of shape (batch, n_head, seq_len, d_k).
            value (torch.Tensor): Value tensor of shape (batch, n_head, seq_len, d_k).
            mask (torch.Tensor | None): Optional attention mask.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Attention output of shape
            (batch, n_head, seq_len, d_k) and attention scores of shape
            (batch, n_head, seq_len, seq_len).
        """
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)  # (batch, n_head, seq_len, seq_len)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, n_head, seq_len, seq_len)
        return (attention_scores @ value), attention_scores  # (batch, n_head, seq_len, d_k)

    def forward(self, q, k, v, mask):
        """
        Forward pass for grouped-query attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch, seq_len, dim).
            k (torch.Tensor): Key tensor of shape (batch, seq_len, dim).
            v (torch.Tensor): Value tensor of shape (batch, seq_len, dim).
            mask (torch.Tensor | None): Optional attention mask.

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, dim).
        """
        query = self.w_q(q)  # (batch, seq_len, dim) --> (batch, seq_len, dim)
        key = self.w_k(k)  # (batch, seq_len, dim) --> (batch, seq_len, n_kv_head * d_k)
        value = self.w_v(v)  # (batch, seq_len, dim) --> (batch, seq_len, n_kv_head * d_k)

        # Split query into n_head smaller subspaces, one for each query head.
        # (batch, seq_len, dim) --> (batch, seq_len, n_head, d_k) --> (batch, n_head, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.n_head, self.d_k).transpose(1, 2)

        # Split key/value into fewer shared KV heads.
        # (batch, seq_len, n_kv_head * d_k) --> (batch, seq_len, n_kv_head, d_k) --> (batch, n_kv_head, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.n_kv_head, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_kv_head, self.d_k).transpose(1, 2)

        # Repeat each KV head so all query heads in the same group share it.
        # (batch, n_kv_head, seq_len, d_k) --> (batch, n_head, seq_len, d_k)
        key = self.repeat_kv(key, self.n_rep)
        value = self.repeat_kv(value, self.n_rep)

        # Apply scaled dot-product attention independently in each query head.
        # Q K V are all (batch, n_head, seq_len, d_k) --> (batch, n_head, seq_len, d_k)
        x, self.attention_scores = GroupedQueryAttentionBlock.attention(query, key, value, mask)

        # Concatenate all query-head outputs back into the full model dimension.
        # (batch, n_head, seq_len, d_k) --> (batch, seq_len, n_head, d_k) --> (batch, seq_len, dim)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_head * self.d_k)

        # Final linear projection output after merging the heads.
        # (batch, seq_len, dim) --> (batch, seq_len, dim)
        return self.w_o(x)
```

Compared with MHA, the only structural change is that `K` and `V` are projected into fewer heads than `Q`, then shared across groups of query heads. If `n_kv_head = n_head`, this reduces to standard MHA. If `n_kv_head = 1`, it becomes MQA.

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
    """SwiGLU activation used in modern Transformer MLP blocks.

    Attributes:
        linear_gate (nn.Module): Linear layer that produces the gate values.
        linear_value (nn.Module): Linear layer that produces the value branch.
    """
    def __init__(self, dim: int, hidden_dim: int) -> None:
        """
        Initializes the SwiGLU activation.

        Args:
            dim (int): Input dimensionality.
            hidden_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.linear_gate = nn.Linear(dim, hidden_dim)
        self.linear_value = nn.Linear(dim, hidden_dim)

    def forward(self, x):
        """
        Forward pass for the SwiGLU activation.
        (batch, seq_len, dim) --> (batch, seq_len, hidden_dim)

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Gated hidden representation.
        """
        return F.silu(self.linear_gate(x)) * self.linear_value(x)


class FeedForwardSwiGLUBlock(nn.Module):
    """Two-layer feed-forward block with SwiGLU gating.

    Attributes:
        swiglu (nn.Module): Gated activation block.
        linear_out (nn.Module): Output projection back to model dimension.
    """
    def __init__(self, dim: int, hidden_dim: int) -> None:
        """
        Initializes the SwiGLU-based FFN layer.

        Args:
            dim (int): Input and output dimensionality.
            hidden_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.swiglu = SwiGLU(dim, hidden_dim)
        self.linear_out = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        """
        Forward pass for the SwiGLU-based FFN layer.
        (batch, seq_len, dim) --> (batch, seq_len, hidden_dim) --> (batch, seq_len, dim)

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.linear_out(self.swiglu(x))
```

[^1]: Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv, April 20, 2021. <https://arxiv.org/abs/2104.09864>
[^2]: Shreyash Shukla. RoPE (Rotary Positional Embeddings). <https://shreyashkar-ml.github.io/posts/rope/>
[^3]: Noam Shazeer. Fast Transformer Decoding: One Write-Head Is All You Need. arXiv, November 6, 2019. <https://arxiv.org/abs/1911.02150>
[^4]: Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit Sanghai. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. arXiv, May 22, 2023. <https://arxiv.org/abs/2305.13245>
[^5]: DeepSeek-AI. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arXiv, June 19, 2024. <https://arxiv.org/abs/2405.04434>
[^6]: DeepSeek-V3 GitHub repository. <https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py>
