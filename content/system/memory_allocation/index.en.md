---
title: "Memory Pool and Allocation"
date: 2026-04-01
tags: ["gpu", "memory"]
math: true
---

A memory pool preallocates a large chunk of memory and manages allocation and free internally, which avoids calling `malloc` and `free` too often. There are any interesting topics around it: how this is implemented inside Linux kernels, why data structures such as segment trees or red-black trees are used, how PyTorch and CUDA do it, how to handle concurrency, and how `jemalloc()` works?

There are two solutions to implement allocators of a fixed-size memory pool: buddy and size-class slab allocators. A slab allocator is fast but has internal fragmentation, and is common in user-space allocators. A buddy allocator supports merging and usually has less external fragmentation, and is common in kernel page allocators.

### Slab Allocator

The slab allocator uses one pool per size class. On allocation, the requested size is rounded up to the nearest size class, such as 8, 16, 32, or 64 bytes. Its main advantage is `O(1)` allocation. In practice, multiple pools can be used for multiple size classes.

### Flexible-Size Memory Pool: Buddy Allocator

The buddy allocator organizes memory by orders of size `2^k`. On allocation, it finds the smallest order that can fit the request. If that order is unavailable, it splits a larger block. On free, it finds the buddy through XOR, merges if the buddy is also free, and keeps merging upward. One drawback is fragmentation. The merge algorithm determines whether the addresses are contiguous and how to merge them.

A Sample implementation is Yunfeng's ~200-line buddy allocator code: https://github.com/cloudwu/buddy/blob/master/buddy.c


### How PyTorch Implements a Memory Pool

PyTorch uses `CUDACachingAllocator`.

The reason is that GPU code should not rely on calling `cudaMalloc` and `cudaFree` for every allocation. `cudaMalloc` is slow and synchronous, and `cudaFree` may trigger a device-wide synchronization.

PyTorch does not use a buddy allocator. Its CUDA allocator is a caching allocator with block splitting and coalescing. It also uses size-based binning, separates small and large pools, and applies a best-fit strategy to reduce fragmentation.

PyTorch 2.x also introduced `cudaMallocAsync` and the CUDA memory pool API for a more modern stream-ordered allocator.
