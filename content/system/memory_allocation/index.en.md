---
title: "Memory Pool and Allocation"
date: 2026-04-01
tags: ["gpu", "memory"]
math: true
---

A memory pool preallocates a large chunk of memory and manages allocation and free internally, which avoids calling `malloc` and `free` too often. There are many interesting topics around it: how this is implemented inside Linux kernels, why data structures such as segment trees or red-black trees are used, how PyTorch and CUDA do it, how to handle concurrency, and how `jemalloc()` works?

There are two solutions to implement allocators of a memory pool: buddy and size-class slab allocators. A slab allocator is fast but has internal fragmentation, and is common in user-space allocators. A buddy allocator supports merging and usually has less external fragmentation, and is common in kernel page allocators.

### Slab Allocator

The slab allocator uses one pool per size class and maintains a fixed-size memory pool. On allocation, the requested size is rounded up to the nearest size class, such as 8, 16, 32, or 64 bytes. Its main advantage is `O(1)` allocation. In practice, multiple pools can be used for multiple size classes.

### Buddy Allocator

The buddy allocator organizes memory by orders of size `2^k` and maintains a flexible-size memory pool. On allocation, it finds the smallest order that can fit the request. If that order is unavailable, it splits a larger block. On free, it finds the buddy through XOR, merges if the buddy is also free, and keeps merging upward. One drawback is fragmentation. The merge algorithm determines whether the addresses are contiguous and how to merge them.

A sample implementation is Yunfeng's ~200-line buddy allocator code: https://github.com/cloudwu/buddy/blob/master/buddy.c

### Slab vs. Buddy

What is the difference between buddy and size-class slab allocators?

Slab allocation is usually faster, but it wastes space inside allocated blocks because requests are rounded up to size classes. Buddy allocation supports merging adjacent free blocks, so it usually has less external fragmentation and is common in kernel page allocators. Internal fragmentation means wasted space inside an allocated block, while external fragmentation means free memory exists but is broken into non-contiguous pieces that cannot satisfy a larger request.

### `jemalloc()`

`jemalloc()` is a general-purpose allocator that uses size classes for small allocations, arenas to reduce lock contention, and fast per-thread caches for recently freed objects. Large allocations are managed as page-sized extents that can be split, reused, and sometimes returned to the OS. In practice, it combines slab-style allocation for small objects with more flexible management for large objects, which helps balance speed, fragmentation, and concurrency.

### CUDA Memory Pool

The CUDA driver maintains a global GPU memory pool by default. When memory is freed asynchronously, it is returned to this pool and can be reused by later allocations. `cudaMallocAsync` and `cudaFreeAsync` are built on top of the CUDA memory pool. By reusing freed buffers instead of repeatedly requesting new memory from the OS, this pool improves allocation efficiency and helps reduce fragmentation over time. CUDA also exposes knobs such as `cudaMemPoolAttrReleaseThreshold`, `cudaMemPoolTrimTo`, and `cudaMemPoolSetAttribute` to control when cached memory is released and how pool behavior is configured. [^1]

### PyTorch Memory Pool

One should not rely on calling `cudaMalloc` and `cudaFree` for every allocation. `cudaMalloc` is slow and synchronous, and `cudaFree` may trigger a device-wide synchronization.
PyTorch has the `CUDACachingAllocator`, which is its GPU memory allocator (memory pool) that caches and reuses CUDA memory blocks instead of calling `cudaMalloc`/`cudaFree` for every tensor. [^2] [^3] [^4]

[^1]: NVIDIA. CUDA Runtime API, Stream Ordered Memory Allocator. <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html>
[^2]: PyTorch Dev Discuss. FSDP + CUDACachingAllocator: An Outsider Newb Perspective. <https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486>
[^3]: Zach DeVito. A guide to PyTorch's CUDA Caching Allocator. <https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html>
[^4]: PyTorch Memory Management. <https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management>
