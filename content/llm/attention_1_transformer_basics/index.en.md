---
title: "Transformer: Basics"
date: 2023-07-12
tags: ["llm", "transformer"]
math: true
---

### Table of Contents

- [Overview](#overview)
- [Encoder-Only, Decoder-Only, and Encoder-Decoder Tasks](#encoder-only-decoder-only-and-encoder-decoder-tasks)
- [Transformer vs. RNN](#transformer-vs-rnn)
- [Input Processing and Embeddings](#input-processing-and-embeddings)
- [Positional Encoding](#positional-encoding)
- [Attention and Its Math*](#attention-and-its-math)
- [MLP or FFN](#mlp-or-ffn)
- [Residual Connection](#residual-connection)
- [Batch Norm and Layer Norm](#batch-norm-and-layer-norm)
- [Complete PyTorch Code](#complete-pytorch-code)

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


### Encoder-Only, Decoder-Only, and Encoder-Decoder Tasks

Tasks that take an input sequence and generate a different output sequence usually need **both** encoder and decoder, such as machine translation, summarization, and many sequence-to-sequence generation problems. T5 and BART are encoder-decoder models.

Tasks that only need to **understand** an input sequence usually need only an **encoder**, such as classification, sentiment analysis, and token labeling. BERT is encoder-only model.

Tasks that only need to **generate** the next tokens autoregressively usually need only a **decoder**, such as language modeling, chat, and text completion. GPT, LLaMA, and Mistral are decoder-only models.

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

### Input Processing and Embeddings

Before tokens are turned into embeddings, the text is first converted into token ids by a tokenizer such as SentencePiece. In many Transformer training pipelines, the tokenizer also defines a few **special tokens** that control how sequences are batched and decoded:

- `PAD_ID`: the padding token id. It is used to extend shorter sequences to the same length within a batch, and attention masks usually hide these positions.
- `UNK_ID`: the unknown token id. It represents text that cannot be mapped cleanly to the tokenizer vocabulary.
- `BOS_ID`: the beginning-of-sequence token id. It marks where a sequence starts and is commonly added before the actual sentence.
- `EOS_ID`: the end-of-sequence token id. It marks where a sequence ends and tells the model when generation should stop.

For example, a tokenized sentence may look like:

```text
[BOS, token_1, token_2, token_3, EOS, PAD, PAD]
```

After that, the embedding layer maps each token id to a dense vector, and positional encoding is added so the model can use token order information.

### Positional Encoding

Positional encoding provides the model with information about the positions of words in a sequence. Since the Transformer's self-attention mechanism does not naturally account for the order of elements in the sequence, positional encoding solves this by adding position information to each element's representation. In the original Transformer paper, positional encodings are defined using **alternating sine and cosine functions**:

$$
\operatorname{PE}(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right)
$$

$$
\operatorname{PE}(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\mathrm{model}}}}\right)
$$

, where $pos$ is token position in the sequence and $i$ is the index of the sine-cosine pair inside the embedding dimension. Here is an illustration of how the positional encoding matrix is calculated [^5]:

{{< figure src="images/pos_encoding.png" alt="Positional Encoding" width="650" align="center" >}}

To apply the positional encoding to input embeddings:

$$
Y = X + PE
$$

Here is the Pytorch code for a simple positional encoding block:

```python
class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for Transformer inputs.
    """

    def __init__(self, dim: int, seq_len: int) -> None:
        """
        Initializes the positional encoding table.

        Args:
            dim (int): Dimension of the input embeddings.
            seq_len (int): Maximum sequence length supported by the encoding.
        """
        super().__init__()
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.pow(10000.0, -torch.arange(0, dim, 2, dtype=torch.float) / dim)  # (dim / 2)
        pe = torch.zeros(seq_len, dim)  # (seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(position / (10000 ** (2i / dim))
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(position / (10000 ** (2i / dim))
        pe = pe.unsqueeze(0)  # (1, seq_len, dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encodings to the input embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim).

        Returns:
            torch.Tensor: Tensor with positional encodings added, with the same shape as input.
        """
        return x + (self.pe[:, :x.shape[1], :])  # (batch, seq_len, dim)
```

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
- $n$ is the number of query tokens, $m$ is the number of key/value tokens, $d_k$ is the query/key feature dimension, and $d_v$ is the value dimension.
- $Q \in \mathbb{R}^{n \times d_k}$ is the query matrix, $K \in \mathbb{R}^{m \times d_k}$ is the key matrix, and $V \in \mathbb{R}^{m \times d_v}$ is the value matrix. In self-attention, $m = n$.

Breaking down the equation:
- The term $QK^\top \in \mathbb{R}^{n \times m}$ is the inner product (cosine) and measures **similarity** between tokens.
- $\sqrt{d_k}$ rescales the scores for numerical stability.
- $\frac{QK^\top}{\sqrt{d_k}}$ is called the **attention weight matrix** or **score matrix**.
- $softmax$ turns the scores into attention weights used to combine the values.

So we calculated the attetion using two matrix multiplications. This makes parallel execution easy. The images below illustrates these two matrix multiplications [^3]:

{{< figure src="images/scaled-dot-product-attention-step-1.png" alt="Compute attention scores" width="420" align="center" >}}

{{< figure src="images/scaled-dot-product-attention-step-2.png" alt="Apply causal mask" width="420" align="center" >}}

{{< figure src="images/scaled-dot-product-attention-step-3.png" alt="Multiply by values" width="420" align="center" >}}

![Scaled Dot-Product Attention](images/scale-dot-product-atetntion.png)


**Multi-Head Attention (MHA)**

Multi-head attention runs several attention heads in parallel. Instead of computing a single attention pattern, the model computes multiple attention patterns at the same time. Different heads can learn different relationships, such as local context, long-range dependencies, or syntactic structure.

![Scaled Dot-Product Attention. Multi-Head Attention](images/scaled_dot-product_attention_and_MHA.png)

Conceptually, each head has its own query, key, and value projections. In implementation, however, we usually do not build separate linear layers for every head. Instead, we use one large projection/matrix for `Q`, one for `K`, and one for `V`, then reshape the result into multiple heads. In other words, the projected features are *logically* partitioned by head, even though they are stored in one tensor. This lets all heads be computed with a small number of large matrix operations rather than many small ones, which is much more efficient. 我觉得可以粗略理解为：把投影后的 hidden dimension 分成 h 块，每一块对应一个 attention head。

If the embedding size is `d_model` and the number of heads is `h`, then each head typically uses:

$$
d_k = d_{model} / h
$$

You can learn more about the dimensions of each matrix in this blog post.[^8]


Here is the Pytorch code for a simple MHA block:

```python
class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-head self-attention block.

    Args:
        dim (int): Input and output dimension of each token representation.
        n_head (int): Number of attention heads.
    """

    def __init__(self, dim: int, n_head: int) -> None:
        """
        Initializes the multi-head attention projections.

        Args:
            dim (int): Input and output dimension of each token representation.
            n_head (int): Number of attention heads.
        """
        super().__init__()
        self.dim = dim
        self.n_head = n_head
        assert dim % n_head == 0

        self.d_k = dim // n_head  # Per-head feature dimension.
        self.w_q = nn.Linear(dim, dim, bias=False)  # Wq
        self.w_k = nn.Linear(dim, dim, bias=False)  # Wk
        self.w_v = nn.Linear(dim, dim, bias=False)  # Wv
        self.w_o = nn.Linear(dim, dim, bias=False)  # Wo

    @staticmethod
    def attention(query, key, value, mask):
        """
        Computes scaled dot-product attention for all heads.

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
        Forward pass for multi-head attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch, seq_len, dim).
            k (torch.Tensor): Key tensor of shape (batch, seq_len, dim).
            v (torch.Tensor): Value tensor of shape (batch, seq_len, dim).
            mask (torch.Tensor | None): Optional attention mask.

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, dim).
        """
        query = self.w_q(q)  # (batch, seq_len, dim) --> (batch, seq_len, dim)
        key = self.w_k(k)  # (batch, seq_len, dim) --> (batch, seq_len, dim)
        value = self.w_v(v)  # (batch, seq_len, dim) --> (batch, seq_len, dim)

        # Split dim into n_head smaller subspaces, one for each head.
        # (batch, seq_len, dim) --> (batch, seq_len, n_head, d_k) --> (batch, n_head, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.n_head, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.n_head, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.n_head, self.d_k).transpose(1, 2)

        # Apply scaled dot-product attention independently in each head.
        # Q K V are all (batch, n_head, seq_len, d_k) --> (batch, n_head, seq_len, d_k)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask)

        # Concatenate all head outputs back into the full model dimension.
        # (batch, n_head, seq_len, d_k) --> (batch, seq_len, n_head, d_k) --> (batch, seq_len, dim)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.n_head * self.d_k)

        # Final linear projection output after merging the heads.
        # (batch, seq_len, dim) --> (batch, seq_len, dim)
        return self.w_o(x)
```


### MLP or FFN

In a Transformer block, attention and the **feed-forward network** (**FFN**, also called the **MLP** block) play different roles.

{{< figure src="images/ffn.png" alt="Attention and FFN" width="420" align="center" >}}

The FFN block typically consists of two linear layers with a nonlinear activation in between:

$$
x = f_{\text{gelu}}(x_{\text{out}}W_1)W_2 + x_{\text{out}}
$$

只用MLP为啥不行呢？MLP 通过全连接层实现全局特征之间的交互，但其计算开销过高，因此难以无限制地向更深层堆叠。Transformer 则通过 attention 机制实现全局特征之间的选择性交互，在保留全局信息建模能力的同时，提供了更高效的结构化建模方式。

Here is the Pytorch code for a simple FFN block:

```python
class FeedForwardBlock(nn.Module):
    """Two-layer feed-forward block used in a Transformer.

    Attributes:
        linear_1 (nn.Module): Linear layer for input-to-hidden transformation.
        gelu (nn.GELU): Activation layer applied between the two linear projections.
        linear_2 (nn.Module): Linear layer for hidden-to-output transformation.
    """
    def __init__(self, dim: int, inter_dim: int) -> None:
        """
        Initializes the FFN layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.linear_1 = nn.Linear(dim, inter_dim)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(inter_dim, dim)

    def forward(self, x):
        """
        Forward pass for the FFN layer.
        (batch, seq_len, dim) --> (batch, seq_len, inter_dim) --> (batch, seq_len, dim)

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.linear_2(self.gelu(self.linear_1(x)))
```

### Residual Connection

Deep networks are hard to train. Gradients could vanish or opimization would degrade. Residual connections aim to mitigate vanishing gradients and degradation in deep models. They are like the skip connections, allowing the input flow directly to later layers, and each block only needs to learn a correction to the input, not a full new representation.

A residual connection means the layer does not only output its transformation, but also the learned changes: `output = input + learned_change` or in math notation: $$H(x) = x + F(x)$$

where:
- `x` is the original input
- `F(x)` is the change the layer learns
- `H(x)` is the final output after adding that change back to the input

So instead of asking the layer to learn the whole output from scratch, we ask it to learn only what should be added or adjusted. If the ideal output is almost the same as the input, then learning `F(x) = 0` is much easier than relearning `H(x) = x` from scratch.

The residual connections also make it easier for gradients to pass backward through many layers during training. Without the skip path, the gradient must pass only through the layer’s transformation `F(x)`, which can shrink or become unstable.


The encoder block applies one residual connection around attention, and another around the MLP/feed-forward part. The decoder has one extra residual connection because it includes the cross-attention step. The red arrows in the images below are residual connections [^6].

{{< figure src="images/residual_connection.png" alt="Residual connections" width="800" align="center" >}}

Here is the Pytorch code for a simple residual connection block:

```python
class ResidualConnection(nn.Module):
    """Residual wrapper with pre-norm and dropout for Transformer sublayers.

    Attributes:
        dropout (nn.Dropout): Dropout applied to the sublayer output before adding the residual.
        norm (LayerNorm): Layer normalization applied before the sublayer.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        """
        Initializes the residual connection block.

        Args:
            dim (int):  Dimensionality of the input and output.
            dropout (float): Dropout probability applied to the sublayer output.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(dim)

    def forward(self, x, sublayer):
        """
        Forward pass for the residual connection.
        (batch, seq_len, dim) --> norm --> sublayer --> dropout --> residual add

        Args:
            x (torch.Tensor):    Input tensor of shape (batch, seq_len, dim).
            sublayer (Callable): Sublayer function or module applied to the normalized input.

        Returns:
            torch.Tensor: Output tensor with the residual connection applied.
        """
        return x + self.dropout(sublayer(self.norm(x)))
```

### Batch Norm and Layer Norm

Both layer normalization and batch normalization are used to stabilize training, but they normalize over different dimensions.

$$
\hat{x} = \frac{x - \mu}{\sigma + \epsilon}
$$

- $\mu$: the mean of the values
- $\sigma$: the standard deviation of the values
- $\epsilon$: a small constant added for numerical stability to avoid division by zero.

**Batch normalization** computes statistics across the batch. Its behavior depends on the distribution of examples inside the mini-batch. This works well in many vision settings, but it is less suitable for sequence models when sequence lengths vary a lot, token distributions change across positions, or batch statistics become unstable or less meaningful. In the equation above, $\mu$ and $\sigma$ are computed across the batch dimension for each feature channel.

**Layer normalization** computes statistics within each individual token representation. It does not depend on other examples in the batch, which makes it more stable for variable-length sequence modeling.[^4] In the equation above, $\mu$ and $\sigma$ are computed from the features of one sample/token.

{{< figure src="images/layer_batch_norm.png" alt="layer norm v.s. batch norm" width="500" align="center" >}}

This is why Transformers use layer normalization instead of batch normalization. For language tasks, each token representation should be normalized independently, without relying on the composition of the current mini-batch.

In a **pre-norm** scenario, layer norm is applied at the start of each residual block, before the sublayer computation, e.g. `x + sublayer(self.norm(x))`, where sublayer is either an attention layer or a feed-forward layer.

Here is the Pytorch code for a simple layer norm block:

```python
class LayerNorm(nn.Module):
    """
    Layer Normalization.

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """

    def __init__(self, dim: int, eps: float = 10 ** -6) -> None:
        """
        Initializes the LayerNorm module.

        Args:
            dim (int): Dimension of the input tensor.
            eps (float): Epsilon value for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """
        Forward pass for LayerNorm.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, hidden_size).

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps)
```

###  Complete PyTorch Code

```python
class EncoderBlock(nn.Module):
    """
    Single Transformer encoder block.

    Args:
        dim (int): Feature dimension of each token representation.
        self_attention_block (MultiHeadAttentionBlock): Self-attention module.
        feed_forward_block (FeedForwardBlock): Position-wise feed-forward module.
        dropout (float): Dropout probability used in residual connections.
    """

    def __init__(self, dim: int, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        """
        Initializes the encoder block submodules.

        Args:
            dim (int): Feature dimension of each token representation.
            self_attention_block (MultiHeadAttentionBlock): Self-attention module.
            feed_forward_block (FeedForwardBlock): Position-wise feed-forward module.
            dropout (float): Dropout probability used in residual connections.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # One residual path wraps self-attention, and the other wraps the feed-forward block.
        self.residual_connections = nn.ModuleList([ResidualConnection(dim, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """
        Forward pass for one encoder block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim).
            src_mask (torch.Tensor | None): Optional source attention mask that prevents attending to padded source tokens.

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, dim).
        """
        # First residual connection applies self-attention with shared Q, K, and V.
        # src_mask keeps encoder tokens from attending to source-side padding.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # Second residual connection applies the position-wise feed-forward block.
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    Stacked Transformer encoder.

    Args:
        dim (int):              Feature dimension of each token representation.
        layers (nn.ModuleList): Encoder blocks applied in sequence.
    """

    def __init__(self, dim: int, layers: nn.ModuleList) -> None:
        """
        Initializes the encoder stack and final normalization layer.

        Args:
            dim (int):              Feature dimension of each token representation.
            layers (nn.ModuleList): Encoder blocks applied in sequence.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(dim)

    def forward(self, x, mask):
        """
        Forward pass for the full encoder stack.

        Args:
            x (torch.Tensor):         Input tensor of shape (batch, seq_len, dim).
            mask (torch.Tensor | None): Optional source attention mask that
                prevents attending to padded source tokens.

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, dim).
        """
        # Apply each encoder block in sequence.
        for layer in self.layers:
            x = layer(x, mask)
        # Apply the final layer normalization after the full encoder stack.
        return self.norm(x)


class DecoderBlock(nn.Module):
    """
    Single Transformer decoder block.

    Args:
        dim (int):                                  Feature dimension of each token representation.
        self_attention_block (MultiHeadAttentionBlock):  Masked self-attention module.
        cross_attention_block (MultiHeadAttentionBlock): Cross-attention module.
        feed_forward_block (FeedForwardBlock):           Position-wise feed-forward module.
        dropout (float):                                 Dropout probability used in residual connections.
    """

    def __init__(self, dim: int, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        """
        Initializes the decoder block submodules.

        Args:
            dim (int):                                  Feature dimension of each token representation.
            self_attention_block (MultiHeadAttentionBlock):  Masked self-attention module.
            cross_attention_block (MultiHeadAttentionBlock): Cross-attention module.
            feed_forward_block (FeedForwardBlock):           Position-wise feed-forward module.
            dropout (float):                                 Dropout probability used in residual connections.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        # Residual paths wrap masked self-attention, cross-attention, and feed-forward sublayers.
        self.residual_connections = nn.ModuleList([ResidualConnection(dim, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass for one decoder block.

        Args:
            x (torch.Tensor):               Input tensor of shape (batch, tgt_len, dim).
            encoder_output (torch.Tensor):  Encoder output of shape (batch, src_len, dim).
            src_mask (torch.Tensor | None): Source attention mask that prevents decoder cross-attention from reading padded encoder positions.
            tgt_mask (torch.Tensor | None): Target attention mask.

        Returns:
            torch.Tensor: Output tensor of shape (batch, tgt_len, dim).
        """
        # First residual connection applies masked self-attention within the decoder sequence.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # Second residual connection attends to encoder outputs using decoder states as queries.
        # src_mask ensures decoder queries ignore padding positions in encoder_output.
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # Third residual connection applies the position-wise feed-forward block.
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    """
    Stacked Transformer decoder.

    Args:
        dim (int):              Feature dimension of each token representation.
        layers (nn.ModuleList): Decoder blocks applied in sequence.
    """

    def __init__(self, dim: int, layers: nn.ModuleList) -> None:
        """
        Initializes the decoder stack and final normalization layer.

        Args:
            dim (int):              Feature dimension of each token representation.
            layers (nn.ModuleList): Decoder blocks applied in sequence.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(dim)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        Forward pass for the full decoder stack.

        Args:
            x (torch.Tensor):               Input tensor of shape (batch, tgt_len, dim).
            encoder_output (torch.Tensor):  Encoder output of shape (batch, src_len, dim).
            src_mask (torch.Tensor | None): Source attention mask that prevents decoder cross-attention from reading padded encoder positions.
            tgt_mask (torch.Tensor | None): Target attention mask.

        Returns:
            torch.Tensor: Output tensor of shape (batch, tgt_len, dim).
        """
        # Apply each decoder block in sequence.
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        # Apply the final layer normalization after the full decoder stack.
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    Output projection layer from hidden states to vocabulary logits.
    ProjectionLayer is the very last step, after the decoder has already produced hidden states.

    Args:
        d_model (int):    Hidden feature dimension of each token representation.
        vocab_size (int): Size of the output vocabulary.
    """

    def __init__(self, d_model, vocab_size) -> None:
        """
        Initializes the final linear projection layer.

        Args:
            d_model (int):    Hidden feature dimension of each token representation.
            vocab_size (int): Size of the output vocabulary.
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        Projects hidden states into vocabulary logits.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, vocab_size).
        """
        # Map each token representation to vocabulary-sized logits.
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):
    """
    Full encoder-decoder Transformer model.

    Args:
        encoder (Encoder):                     Transformer encoder stack.
        decoder (Decoder):                     Transformer decoder stack.
        src_embed (InputEmbeddings):           Source token embedding layer.
        tgt_embed (InputEmbeddings):           Target token embedding layer.
        src_pos (PositionalEncoding):          Source positional encoding layer.
        tgt_pos (PositionalEncoding):          Target positional encoding layer.
        projection_layer (ProjectionLayer):    Output projection to vocabulary logits.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        """
        Initializes the Transformer encoder, decoder, embeddings, and output projection.

        Args:
            encoder (Encoder):                     Transformer encoder stack.
            decoder (Decoder):                     Transformer decoder stack.
            src_embed (InputEmbeddings):           Source token embedding layer.
            tgt_embed (InputEmbeddings):           Target token embedding layer.
            src_pos (PositionalEncoding):          Source positional encoding layer.
            tgt_pos (PositionalEncoding):          Target positional encoding layer.
            projection_layer (ProjectionLayer):    Output projection to vocabulary logits.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        """
        Encodes a source sequence into contextualized hidden states.

        Args:
            src (torch.Tensor):              Source token ids of shape (batch, src_len).
            src_mask (torch.Tensor | None):  Source attention mask that prevents encoder attention from using padded source tokens.

        Returns:
            torch.Tensor: Encoder output of shape (batch, src_len, d_model).
        """
        # Embed source tokens and add positional information before the encoder stack.
        # (batch, src_len) --> (batch, src_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        # Pass src_mask through the encoder so every self-attention layer ignores source padding.
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        """
        Decodes a target sequence using encoder outputs as context.

        Args:
            encoder_output (torch.Tensor):   Encoder output of shape (batch, src_len, d_model).
            src_mask (torch.Tensor | None):  Source attention mask that prevents decoder cross-attention from using padded encoder positions.
            tgt (torch.Tensor):              Target token ids of shape (batch, tgt_len).
            tgt_mask (torch.Tensor | None):  Target attention mask.

        Returns:
            torch.Tensor: Decoder output of shape (batch, tgt_len, d_model).
        """
        # Embed target tokens and add positional information before the decoder stack.
        # (batch, tgt_len) --> (batch, tgt_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        # Pass src_mask into decoder cross-attention so target tokens only read real source tokens.
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        """
        Projects decoder hidden states into vocabulary logits.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch, seq_len, vocab_size).
        """
        # Apply the final vocabulary projection to each decoder position.
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.projection_layer(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512,
                      N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """
    Builds a full encoder-decoder Transformer model.

    Args:
        src_vocab_size (int): Source vocabulary size.
        tgt_vocab_size (int): Target vocabulary size.
        src_seq_len (int):    Maximum source sequence length.
        tgt_seq_len (int):    Maximum target sequence length.
        d_model (int):        Model dimension for token representations.
        N (int):              Number of encoder and decoder blocks.
        h (int):              Number of attention heads per multi-head attention block.
        dropout (float):      Dropout probability used throughout the model.
        d_ff (int):           Hidden dimension of the feed-forward blocks.

    Returns:
        Transformer: Fully constructed Transformer model with Xavier-initialized weights.
    """
    # Create source and target token embedding layers.
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create positional encoding layers for source and target sequences.
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Build the encoder stack from N identical encoder blocks.
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Build the decoder stack from N identical decoder blocks.
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Wrap the block lists into the full encoder and decoder modules.
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the output projection from decoder hidden states to target vocabulary logits.
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Assemble the full Transformer model.
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # Initialize weight matrices with Xavier uniform initialization for stable training.
    for p in transformer.parameters():
        if p.dim() > 1: # it is a weight matrix, not a bias vector
            nn.init.xavier_uniform_(p)

    return transformer
```


[^1]: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. arXiv, June 12, 2017. <https://arxiv.org/abs/1706.03762>
[^2]: Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. Efficient Transformers: A Survey. arXiv, September 14, 2020. <https://arxiv.org/abs/2009.06732>
[^3]: Transformer模型详解（图解最完整版）. 初识CV, Zhihu. <https://zhuanlan.zhihu.com/p/338817680>
[^4]: Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E. Hinton. Layer Normalization. arXiv, July 21, 2016. <https://arxiv.org/abs/1607.06450>
[^5]: Mehreen Saeed. A Gentle Introduction to Positional Encoding in Transformer Models, Part 1. Machine Learning Mastery, January 6, 2023. <https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/>
[^6]: M Javadnejadi. “Transformer is All You Need”. AI Advances, April 30, 2024. <https://ai.gopubby.com/transformer-is-all-you-need-fbb1d1e4d9b0>
[^7]: RethinkFun/DeepLearning chapter15 <https://github.com/RethinkFun/DeepLearning/blob/master/chapter15/transformer.py>
[^8]: Ketan Doshi. Transformers Explained Visually, Part 3: Multi-Head Attention Deep Dive. Towards Data Science. <https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853/>
