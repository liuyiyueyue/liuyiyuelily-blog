---
title: "CUDA Matrix-Vector Multiplication"
date: 2025-11-13
tags: ["llm", "cuda", "optimization"]
math: true
---

TODO:
1. 每个算法的FLOPs总数

Matrix-vector multiplication (GEMV) is a foundational operation in linear algebra. Let matrix $A \in \mathbb{R}^{M \times N}$, vector $x \in \mathbb{R}^{N}$, and vector $y \in \mathbb{R}^{M}$. Then

$$
y = Ax
$$

**Naive Kernel**

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

**Reuse Vector $x$**

Intead of reloading vector $x$ multiple times, we use the shared memory to store a tile of it. Each thread loads an element of $x$ into the shared tile $sx$

```text
x:  [ x0 x1 x2 x3 | x4 x5 x6 x7 | ... ]
        ↑ TILE 0        ↑ TILE 1
```

There is the code for the optimized kernel:

```c
#define TILE 256
// number of elements of x loaded into shared memory per iteration

__global__ void matvec_opt(float* A, float* x, float* y,
                           int M, int N) {

    // shared memory buffer to cache a tile of vector x
    __shared__ float sx[TILE];

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0;

    // loop over x (and columns of A) in tiles
    for (int t = 0; t < (N + TILE - 1)/TILE; t++) {

        // compute global index into x for this tile
        int idx = t*TILE + threadIdx.x;

        // load one element of x into shared memory if within bounds
        if (idx < N)
            sx[threadIdx.x] = x[idx];

        // ensure all threads have finished loading shared memory
        __syncthreads();

        if (row < M) {

            // iterate over elements in the shared memory tile
            for (int i = 0; i < TILE && t*TILE + i < N; i++) {

                // multiply A[row, col] with cached x[col] and accumulate
                sum += A[row*N + t*TILE + i] * sx[i];
            }
        }

        // ensure all threads are done using shared memory before overwrite
        __syncthreads();
    }

    // write final result to output if within bounds
    if (row < M)
        y[row] = sum;
}
```

**Memory Bound**

For the naive GEMV kernel over the whole $M \times N$ matrix, there are $MN$ multiply operations and about $MN$ add operations, so the total work is approximately $2MN$ FLOPs.

The kernel reads all $MN$ floats from $A$. Although $x$ contains only $N$ floats, the kernel accesses those $N$ elements for each of the $M$ output rows, which contributes another $MN$ float accesses. Besides these reads, the kernel writes $M$ floats to $y$. Each float is 4 bytes, so the total memory traffic is $4(MN + MN + M)$ bytes.

Therefore, the arithmetic intensity is approximately

$$
\frac{2MN}{4(MN + MN + M)} \lesssim 0.25
$$

FLOPs/byte.

This is very low, so GEMV is strongly memory-bound.
