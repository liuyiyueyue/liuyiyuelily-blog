---
title: "FlashAttention"
date: 2023-12-10
tags: ["llm", "inference", "optimization"]
math: true
---

### What Is FlashAttention?

FlashAttention is a fast, memory-efficient, exact attention algorithm designed with IO awareness. [^1]

- **Fast**: speeds up attention computation.
- **Memory-efficient**: reduces GPU memory usage.
- **Exact**: preserves the exact attention result instead of using an approximation.
- **IO-aware**: improves performance by reducing data movement between GPU memory levels.

### Why Regular Attention Is Expensive

In a naive attention implementation, the matrices `Q`, `K`, and `V` are stored in high-bandwidth memory (HBM). Intermediate results are also repeatedly written to and read back from HBM:

1. Load `Q` and `K` from HBM into SRAM.
2. Compute `S = QK^T`.
3. Write `S` back to HBM.
4. Load `S` into SRAM.
5. Compute `P = softmax(S)`.
6. Write `P` back to HBM.
7. Load `P` and `V` from HBM into SRAM.
8. Compute `O = PV`.
9. Write `O` back to HBM.

The problem is that these reads and writes are expensive. Attention is often limited not just by computation, but by memory IO.

Also, the attention score matrix grows with sequence length as `N^2`, which makes the memory cost increasingly expensive for long contexts. See detailed calculation of compute and space complexity in

### How FlashAttention Helps

Earlier attention optimizations often focused on reducing computation. FlashAttention instead focuses on **reducing memory IO**. Its main goal is to **avoid materializing the full attention matrix in HBM**.

It does this in two main ways:

1. **Tiling and kernel fusion**: compute attention block by block and keep more intermediate results in fast on-chip memory, which reduces HBM traffic.
2. **Recomputation during backward pass**: in training, recompute some intermediate values instead of storing all of them, which saves memory.

As a result, FlashAttention improves both speed and memory efficiency, especially for long sequences.

### Softmax

**Safe Softmax**

For a vector $[x_1, x_2, \ldots, x_d]$, the standard softmax is

$$
\operatorname{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{d} e^{x_j}}
$$

Floating-point numbers in hardware have a limited dynamic range. For `float32` and `bfloat16`, when $x$ is large, $e^x$ overflows to `inf`. To avoid overflow and improve numerical stability, softmax is usually computed by subtracting the maximum value first. This is called **safe softmax**:

$$
m = \max_i(x_i), \qquad
\operatorname{softmax}(x_i) = \frac{e^{x_i - m}}{\sum_{j=1}^{d} e^{x_j - m}}
$$

**Tiling the Softmax**

The key difficulty is that softmax is a reduction over the whole row, so it does not look easy to compute block by block at first.

[I will write the math one day... I hate typing in latex]

This is the core idea behind tiled softmax in FlashAttention: each block keeps track of a running maximum and a running normalization term, and these can be merged across blocks exactly. That is why FlashAttention can compute attention in tiles while still producing the exact softmax result.


[^1]: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022. <https://proceedings.neurips.cc/paper_files/paper/2022/hash/7d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html>
