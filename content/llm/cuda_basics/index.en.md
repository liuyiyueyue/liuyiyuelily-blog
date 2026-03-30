---
title: "CUDA Basics"
date: 2025-10-25
tags: ["llm", "cuda"]
---

### What is CUDA?

CUDA, short for Compute Unified Device Architecture, is NVIDIA’s programming model for heterogeneous computing on CPUs and GPUs. In CUDA, the CPU acts as the host that launches work, while the GPU acts as the device that executes many **threads** in parallel. The core idea is to divide a large computation into many small pieces so the GPU can process them concurrently.


### Memory Hierarchy Refresher

A useful mental model for CUDA memory is:

- registers: private storage for each thread
- shared memory: on-chip memory shared by threads in the same block
- L1 cache: on-chip cache associated with an SM
- L2 cache: cache shared across the whole GPU
- global memory: large device memory visible to all threads, typically backed by HBM

This hierarchy explains a lot of CUDA performance behavior. Good kernels maximize data reuse in registers, shared memory, and caches. Bad kernels repeatedly go out to HBM and waste the GPU's arithmetic throughput waiting on memory.


### SMs
At execution time, the GPU is organized around many **streaming multiprocessors (SMs)**. Each SM contains:

```text
Streaming Multiprocessors (SMs)
├── CUDA cores
├── Tensor Cores
├── Warp schedulers
├── Register file
├── Shared memory / L1 cache
└── Load/store + other functional units
```

{{< figure src="./images/blackwell_ultra_sm_architecture.png" caption="Blackwell Ultra SM architecture. [^1]" align="center" >}}

Conceptually, one SM can be viewed as four smaller execution partitions that share on-chip resources such as registers, shared memory, and cache. Each partition has its own warp scheduler and dispatch logic. Within a partition, the **warp scheduler** can often issue two instructions per cycle from the same warp: one compute instruction, such as INT32, FP32, or Tensor Core work, and one memory instruction, such as a load or store. This is why the scheduler is described as **dual-issue**. The Special Function Unit (SFU) sits alongside these pipelines and handles transcendental operations such as sine, cosine, reciprocal, and square root, but it is separate from the usual compute-plus-memory dual-issue pairing.


### Threads, Warps, Blocks, and Grids

CUDA organizes computation in a grid → block → warp → thread hierarchy:

{{< figure src="./images/cuda_program_model.png" caption="GPU hardware hierarchy and the CUDA execution model: SMs execute warps of 32 threads over a layered memory hierarchy." align="center" >}}

| Term | Meaning | Size | Syntax |
| --- | --- | --- | --- |
| **Thread** 线程 | Smallest unit of work | 1 |  |
| **Warp** 线程组 | 32 threads executing together | 32 threads |  |
| **Block** 线程块 | Group of threads/warps that share shared memory and can synchronize with each other. A block runs on one SM at a time, but one SM can host multiple blocks concurrently. | 1-1024 threads | `dim3 block(Bx, By, Bz);` |
| **Grid** 网格 | Collection of blocks launched by one kernel | Many blocks | `dim3 grid(Gx, Gy, Gz);` |

**Indexing**

In a typical 1D kernel, each thread computes its **global index** with:

```c
int global_index = blockIdx.x * blockDim.x + threadIdx.x;
```

In other words:

```text
global index = block index * threads per block + thread index within the block
```

For example, if `blockDim.x = 256`, `blockIdx.x = 3`, and `threadIdx.x = 10`, then the thread's global index is `3 * 256 + 10 = 778`.

**Number of Blocks and Threads Needed**

The choice of **`numBlocks` and `threadsPerBlock`** depends on both the total amount of work `N` and the GPU's hardware limits. A common pattern is:

```c
int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
```

Choosing `threadsPerBlock` is more hardware-sensitive. In practice, it is usually guided by warp size, the maximum threads per block, the number of SMs, occupancy, and the kernel's memory access pattern. For example, on Blackwell Ultra, the warp size is still 32 threads, the maximum block size is 1024 threads, and the full GPU can contain up to 160 SMs depending on SKU.[^1] In real kernels, developers often start with a multiple of 32 such as 128, 256, or 512 threads per block, then tune from there.


### Function Qualifiers

CUDA uses function qualifiers to specify where a function executes and where it can be called from.

| Qualifier | Executes On | Callable From | Return Value | Call Syntax | Common Role |
| --- | --- | --- | --- | --- | --- |
| `__host__` | CPU | CPU | Yes | normal `()` | Regular C/C++ host function. This is the default for ordinary functions. |
| `__device__` | GPU | GPU | Yes | normal `()` | Device-side helper function used inside kernels or other device code |
| `__global__` | GPU | CPU | No | `<<< >>>` | Kernel entry point launched from the host |

**`__host__` example**

```c
__host__ void printHello() {
    printf("Hello from CPU!\n");
}
```

**`__global__` example**

```c
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}
```

**`__device__` example**

```c
__device__ float square(float x) {
    return x * x;
}

__global__ void computeSquares(float *A, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        A[i] = square(A[i]);
}
```


### Launch a Kernel

Launching kernel means grid of threads is launched. All threads execute the same code, aka SPMD. The CUDA kernel syntax is `kernel_name<<<numBlocks, threadsPerBlock>>>(args...);`, where CUDA launches `numBlocks` * `threadsPerBlock` number of threads, and each thread runs one copy of `kernel_name()` in parallel. 

Below is a code snippet and illustration for vector addition. A complete vector-add example is here: [vec_add_kernel.cu](./code/vec_add_kernel.cu.txt).

```cpp
// compute vector sum C = A + B
// each thread performs one pair-wise addition
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { // check bounds
        C[i] = A[i] + B[i];
    }
}

int main() {
    ...
    vecAddKernel<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, n);
    ...
}
```

{{< figure src="./images/vec_add_kernel.png" caption="Threading in vecAddKernel() kernel." align="center" >}}


### CUDA Stream

**What is a CUDA Stream?**

A CUDA stream is a sequence of GPU commands that execute in order. Operations in the same stream run sequentially. Operations in different streams run concurrently, if hardware allows. Think of each stream as a queue of GPU work:

```text
Stream 0: kernel1 → kernel2 → memcpy → kernel3
Stream 1: kernelA → kernelB → memcpy
```

Default stream (`cudaStreamDefault`) is synchronizing (everything waits for it), while explicit streams (`cudaStreamCreate`) allow true concurrency.

**Why Use Streams?**

Streams are useful for running independent kernels concurrently, overlapping H2D or D2H transfers with kernel execution, and building pipelined execution across mini-batches.

**Sample Usage of Streams**

```cuda
cudaStream_t s1, s2;
cudaStreamCreate(&s1);
cudaStreamCreate(&s2);

// Kernels in s1 can run concurrently with kernels in s2.
myKernel<<<grid, block, 0, s1>>>(...);
myKernel<<<grid, block, 0, s2>>>(...);

cudaStreamDestroy(s1);
cudaStreamDestroy(s2);
```

**Stream Synchronization**

CUDA provides both stream- and event-based synchronization. `cudaStreamSynchronize(s1);` blocks the CPU until all previously issued GPU work in `s1` has completed, while `cudaEventSynchronize(ev);` waits for a specific event to be recorded. This is the mechanism used by systems such as PyTorch, cuBLAS, and FlashAttention to build execution pipelines.

### Summary

The four-layer architecture of CUDA
1. Thread execution layer (compute units)
    - Thread
        - The most basic execution unit.
        - One thread executes one kernel.
        - Has independent registers and local memory
        - `threadIdx(x, y, z)` indicates this thread's position within the block
    - Warp
        - The smallest execution unit
        - 32 threads form one warp.
        - lockstep: threads in a warp share one program counter and execute the same instruction
        - memory coalescing: contiguous accesses are merged into one transaction
        - optimization principles: multiples of 32 and avoiding branch divergence
2. Logical organization layer (software abstraction)
    - Block
        - Contains multiple warps, with up to 1024 threads
        - shared memory: threads inside a block can efficiently access `__shared__` memory
        - synchronization mechanism: `__syncthreads()` enables synchronization within a block
        - 1D/2D/3D structure: `blockDim`
        - independent scheduling: different blocks can execute out of order on different SMs
    - Grid
        - Contains multiple blocks
        - Corresponds to one kernel launch, i.e. `kernel<<<gridDim, blockDim>>>(args)`
        - 1D/2D/3D structure: `gridDim`
        - `blockIdx(x, y, z)` indicates this block's position in the grid
    - `tid = blockIdx.x * blockDim.x + threadIdx.x`
3. Concurrency control layer
    - Stream
        - Manages execution order and concurrency
        - asynchronous execution: async
        - concurrency mechanism: multiple streams can execute kernels and copies in parallel
        - within the same stream, operations execute in order
    - Event
        - Marker points. Used for coordination and timing between streams
4. Resource management
    - Context
        - A container for GPU resources. Manages all state and data.


[^1]: https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
