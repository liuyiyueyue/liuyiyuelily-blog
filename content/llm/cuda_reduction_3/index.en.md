---
title: "CUDA Reduction"
date: 2025-11-13
tags: ["llm", "cuda", "optimization"]
---

Reduction is one of the classic CUDA optimization exercises because it exposes several GPU performance bottlenecks in a compact example: warp divergence, shared-memory bank conflicts, idle threads, and synchronization overhead. This post walks through the usual progression from a naive reduction kernel to more optimized versions.[^1]

{{< figure src="./images/reduction.png" align="center" >}}

**Algorithm Overview**

Assume we are given an array of length `N` and need to compute the sum of it. We first split the array into `m` chunks. In the first stage, we launch `m` blocks, with each block reducing one chunk to a partial sum. In the second stage, a single block reduces those `m` partial sums to produce the final result. Since the second stage can reuse the same kernel as the first, we will not discuss it separately here. This post focuses only on optimization techniques for the first stage.

The kernel interface is:

```c
__global__ void reduce(T* input, T* output)
```

Here, input is the `input` array of length `N`, and `output` is the output array for the first stage, containing the `m` partial sums.

We need to define three parameters:

- `BlockNum`: the number of launched blocks, i.e. `m`, which determines how many chunks the array is divided into.
- `Thread_per_block`: the number of threads in each block. Common choices are `128`, `256`, `512`, and `1024`.
- `Num_per_block`: the number of elements reduced by each block.

These parameters satisfy `BlockNum * Num_per_block = N`.

**Baseline Kernel**

The baseline algorithm is straightforward and consists of three steps. First, load the data into shared memory. Second, perform the reduction inside shared memory. Third, write the final result back to global memory. The code is as follows:

```c
__global__ void reduce0(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i];
    __syncthreads(); // shared-memory load must be complete before reduction starts.

    // do reduction in shared mem
    for(unsigned int s=1; s<blockDim.x; s*=2){ // for each reduction step
        if(tid%(2*s) == 0){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads(); // each reduction step must be complete before the next step starts.
    }
    
    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}
```

This section focuses on explaining how the code maps to the hardware:

1. In the first step, we set `Num_per_block = Thread_per_block`. For example, with `256` threads per block, each block reduces `256` elements. If the input has `32M` elements, we launch `128K` blocks. Here, `tid` is the thread index within a block, and `i` is the index in the original array. In this stage, thread `tid` loads element `i` from global memory, passes it through a register, and stores it in shared memory.

2. In the second step, once all `256` values are in shared memory, the block performs reduction over multiple rounds: in round 1, threads with `tid % 2 == 0` add `sdata[tid + 1]` into `sdata[tid]`; in round 2, threads with `tid % 4 == 0` add `sdata[tid + 2]`; and so on until all values are accumulated into `sdata[0]`.

**Wrap-level Control-flow Optimization**

The main problem with the baseline `reduce0` kernel is **warp divergence**. Within a block, threads are expected to follow the same instruction stream, and in our case there are 256 / 32 = 8 warps per block. When an if/else branch appears, the warp must execute all branches, and only the results from the threads that satisfy the condition are kept. As shown above, each iteration introduces two branches, which significantly hurts performance.

> The smallest execution unit in hardware is a **warp**, which contains 32 threads. A thread is just one lane within a warp.

The improved kernel `reduce1` tries to keep as many threads as possible on the same branch:

```c
__global__ void reduce1(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=1; s<blockDim.x; s*=2){
        int index = 2*s*tid;
        if(index < blockDim.x){
            sdata[index]+=sdata[index+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}
```

For each warp, all threads take the same branch in a given iteration, so this pattern avoids warp divergence within that warp. In the first iteration, warps `0-3` satisfy `index < blockDim.x` while warps `4-7` do not; in the second iteration, only warps `0` and `1` remain active, etc.

**Avoid Bank-conflict**

The main problem in `reduce1` is bank conflict: in the first iteration, warp 0 accesses shared-memory locations with a stride of 2, so thread 0 reads `sdata[0]` and thread 16 reads `sdata[32]`, which map to the same bank. More generally, pairs such as `0` and `32`, `1` and `33`, and `2` and `34` fall into the same bank, so the warp's memory accesses are serialized instead of being served in parallel.

> Shared memory has **32 banks**, matching the warp size. The bank ID is roughly *shared-memory address index % 32*. If multiple threads access the same bank, a bank conflict occurs, the accesses are serialized, and performance drops sharply.

In reduction, the way to avoid bank conflicts is to reverse the for loop. Instead of increasing the stride from 1 up to 256, we decrease it from 128 down to 1. 

The pseudocode is as follows: 

```c
__global__ void reduce2(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*blockDim.x+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}
```

If we focus on warp 0, then in the first few iterations each active thread reads two values separated by the current stride, such as `(0,128)`, `(1,129)`, then `(0,64)`, `(1,65)`, and then `(0,32)`, `(1,33)`. These accesses stay aligned so that a warp reads one shared-memory row at a time without bank conflicts, and once the stride shrinks to `16`, only threads `0-15` remain active while threads `16-31` do no work.

**Prevent Idle Threads**

The main problem in `reduce2` is wasted threads. We launch `256` threads, but only `128` are active in the first iteration, only `64` in the second, and the number of working threads is cut in half each round.

```c
__global__ void reduce3(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    // each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x]; // each thread reads 2 numbers and performs an addition.
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}
```

Before reduce3, Num_per_block = 256, and the number of blocks is approximately N / 256. In reduce3, Num_per_block = 512, and the number of blocks is approximately N / 512. That means Num_per_block doubles, while the number of blocks is halved.

**Reduce Synchronization**

In reduce3 kernel, when the reduction reaches the final few iterations, only warp 0 in each block is still doing useful work, yet the threads are still forced to execute synchronization operations. This causes substantial overhead.

> Threads within a warp execute in **lockstep**, so no explicit synchronization is needed. At the hardware level, warp threads are issued and executed together.

When `s = 32`, only a single SIMD unit is still active, so `__syncthreads()` can be removed entirely at that point. Therefore, we unroll the last stage of the reduction to reduce synchronization overhead. The pseudocode is as follows:

```c
__device__ void warpReduce(volatile float* cache,int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}

__global__ void reduce4(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>32; s>>=1){
        if(tid < s){
            sdata[tid]+=sdata[tid+s];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid<32)warpReduce(sdata,tid);
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}
```

**Loop Unrolling**

At this point, the reduction is already highly efficient. Further optimization becomes quite difficult. To push performance to the limit, we can fully unrolling the for loop. There is still some benefit, but it is no longer especially significant. This is mainly due to the continued evolution of GPU hardware architectures, along with substantial improvements NVIDIA has made in the compiler. The pseudocode is as follows:

```c
template <unsigned int blockSize>
__global__ void reduce5(float *d_in,float *d_out){
    __shared__ float sdata[THREAD_PER_BLOCK];

    //each thread loads one element from global memory to shared mem
    unsigned int i=blockIdx.x*(blockDim.x*2)+threadIdx.x;
    unsigned int tid=threadIdx.x;
    sdata[tid]=d_in[i] + d_in[i+blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    if(blockSize>=512){
        if(tid<256){
            sdata[tid]+=sdata[tid+256];
        }
        __syncthreads();
    }
    if(blockSize>=256){
        if(tid<128){
            sdata[tid]+=sdata[tid+128];
        }
        __syncthreads();
    }
    if(blockSize>=128){
        if(tid<64){
            sdata[tid]+=sdata[tid+64];
        }
        __syncthreads();
    }
    
    // write result for this block to global mem
    if(tid<32)warpReduce<blockSize>(sdata,tid);
    if(tid==0)d_out[blockIdx.x]=sdata[tid];
}
```

**Shuffle**

Shuffle instructions are a set of warp-level instructions. Warp Shuffles are special CUDA instructions that let threads inside the same warp (32 threads) directly exchange values. No shared memory, no `__syncthreads()`. Their biggest advantage is improved programmability: in some scenarios, they let you avoid using shared memory altogether.

> Threads within a warp share the same execution units, and all registers are stored in a common register file.

```c
__device__ __forceinline__ float warpReduceSum(float sum) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    return sum;
}

__global__ void reduce7(const float *d_in, float *d_out, unsigned int n) {
    float sum = 0.0f;
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + tid;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;

    while (i < n) {
        sum += d_in[i];
        if (i + blockDim.x < n) {
            sum += d_in[i + blockDim.x];
        }
        i += gridSize;
    }

    __shared__ float warpSums[32];
    unsigned int lane = tid % warpSize;
    unsigned int warp = tid / warpSize;

    sum = warpReduceSum(sum);

    if (lane == 0) warpSums[warp] = sum;
    __syncthreads();

    sum = (tid < blockDim.x / warpSize) ? warpSums[lane] : 0.0f;

    if (warp == 0) sum = warpReduceSum(sum);

    if (tid == 0) d_out[blockIdx.x] = sum;
}
```

[^1]: Mark Harris. Optimizing Parallel Reduction in CUDA. NVIDIA. <https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf>
