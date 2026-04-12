---
title: "CUDA Matrix-Vector Multiplication"
date: 2025-11-10
tags: ["llm", "cuda", "optimization"]
math: true
---

Matrix-vector multiplication (GEMV) is a foundational operation in linear algebra. Let matrix $A \in \mathbb{R}^{M \times N}$, vector $x \in \mathbb{R}^{N}$, and vector $y \in \mathbb{R}^{M}$. Then

$$
y = Ax
$$

### Naive Kernel

In the naive kernel implementation below, each thread handles a row of matrix $A$, an element of vector $x$, and an element of vector $y$. This kernel is simple and performs coalesced access on A (row-major). However, the vector $x$ is reloaded N times per thread, wasting memory bandwidth.

```c
__global__ void matvec1(float* A, float* x, float* y,
                       int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float sum = 0;
        for (int col = 0; col < N; col++)
            sum += A[row*N + col] * x[col];
        y[row] = sum;
    }
}
```

### Shared Memory

In the naive kernel implementation above, each thread loads the vector $x$ into its local memory. To optimize the kernel, each thread still handles one row of A. However, threads cooperatively load a tile of $x$ into shared memory. After synchronization, each thread computes a partial dot product using the cached tile. We iterate over tiles until covering $N$.

```c
__global__ void gemv_2(float* A, float* x, float* y, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_x[256];  // assume blockDim.x = 256

    float sum = 0.0f;

	// tile size equals to blockDim.x
    for (int tile = 0; tile < N; tile += blockDim.x) {

        // each thread loads one element of x into shared memory,
		// given 0 <= threadIdx.x < blockDim.x = 256
        int col = tile + threadIdx.x;
        if (col < N) {
            s_x[threadIdx.x] = x[col];
        } else {
            s_x[threadIdx.x] = 0.0f;
        }

        __syncthreads();  // ensure all x tile is loaded

        // compute partial dot product
        if (row < M) {
            for (int i = 0; i < blockDim.x && (tile + i) < N; i++) {
                sum += A[row * N + (tile + i)] * s_x[i];
            }
        }

        __syncthreads();  // before next tile overwrite
    }

    if (row < M) {
        y[row] = sum;
    }
}
```

{{< figure src="./images/gemv_shared_mem.png" align="center" >}}





### Memory Bound

For the naive GEMV kernel over the whole $M \times N$ matrix, there are $MN$ multiply operations and about $MN$ add operations, so the total work is approximately $2MN$ FLOPs.

The kernel reads all $MN$ floats from $A$. Although $x$ contains only $N$ floats, the kernel accesses those $N$ elements for each of the $M$ output rows, which contributes another $MN$ float accesses. Besides these reads, the kernel writes $M$ floats to $y$. Each float is 4 bytes, so the total memory traffic is $4(MN + MN + M)$ bytes.

Therefore, the arithmetic intensity is approximately

$$
\frac{2MN}{4(MN + MN + M)} \lesssim 0.25
$$

FLOPs/byte.

This is very low, so GEMV is strongly memory-bound.
