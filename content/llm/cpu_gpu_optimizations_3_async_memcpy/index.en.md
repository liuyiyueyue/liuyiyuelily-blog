---
title: "[3/4] CPU-GPU Optimization: Async Memcpy"
date: 2026-02-06
tags: ["llm", "optimization", "cuda", "async"]
---

In this blog, we will discuss techniques to squeeze the memcpy "bubbles" with kernel executions. 
This post focuses on how streams help overlap data movement with compute via `cudaMemcpyAsync`. For a basic introduction to CUDA streams themselves, see the [CUDA Stream section](/llm/cuda/#stream).


## Async Memcpy

In this section, we introduce `cudaMemcpyAsync`, the **non-blocking, asynchronous** counterpart to
`cudaMemcpy` described in previous blog (See [Pinned Memory + Memcpy](/llm/cpu_gpu_optimizations_2_cuda_memcpy/#2-pinned-memory--memcpy)). Unlike `cudaMemcpy`, `cudaMemcpyAsync` returns immediately;
the transfer is enqueued into a specified CUDA stream. This makes it possible to 
**overlap memcpy with kernel execution** or with other memcpy operations 
(when the hardware and stream setup allow it).

To use `cudaMemcpyAsync`, the **host memory must be pinned** via `cudaMallocHost` or `cudaHostAlloc`,
and the device memory must be a valid device pointer (typically allocated with `cudaMalloc`).

Below is an example of one stream. Within one stream, the (async) memcpys and kernels execute sequentially. 
That means the execution order is H2D → Kernel.

```cuda
size_t transfer_size = N * sizeof(float);

float *host_mem;
float *device_mem;
cudaMallocHost((void **)&host_mem, transfer_size);
cudaMalloc((void **)&device_mem, transfer_size);

cudaStream_t s;
cudaStreamCreate(&s);

cudaMemcpyAsync(device_mem, host_mem, transfer_size, cudaMemcpyHostToDevice, s);
kernel<<<grid, block, 0, s>>>(device_mem, device_mem);

cudaStreamSynchronize(s);
```


Below is an enhanced example of multiple streams. 
```cuda
size_t transfer_size = N * sizeof(float);
size_t transfer_size_per_stream = transfer_size / num_streams;
size_t transfer_element_per_stream = N / num_streams;

float *host_mem;
float *device_mem;
cudaMallocHost((void **)&host_mem, transfer_size);
cudaMalloc((void **)&device_mem, transfer_size);

cudaStream_t s[num_streams];
for (int i = 0; i < num_streams; i++) {
  cudaStreamCreate(&s[i]);
}

for (int i = 0; i < num_streams; i++) {
  int offset = i * transfer_element_per_stream;
  cudaMemcpyAsync(device_mem + offset, host_mem + offset, transfer_size_per_stream, cudaMemcpyHostToDevice, s[i]);
  kernel<<<grid, block, 0, s[i]>>>(device_mem + offset, transfer_element_per_stream);
}

for (int i = 0; i < num_streams; i++) {
  cudaStreamSynchronize(s[i]);
}
```

Although H2D transfers are issued from different CUDA streams, 
they share the same PCIe interconnect and are serviced by a limited number of copy engines. 
As a result, transfers in the same direction are typically serialized, 
while copy and compute can overlap, leading to a pipelined execution pattern.


Comparing serial version and multi stream version of the memcpy using a diagram below, 
we can see the concurrency reduced the total execution time and bubbles.
```
Time →

# serial version (one stream):
Copy Engine: |────────────────H2D────────────────|
GPU:                                             |───────────────Kernel───────────────|

# multi stream version (three streams)
Copy Engine: |────H2D────|────H2D────|────H2D────|
GPU:                     |───Kernel───|───Kernel───|───Kernel───|
```

Below are Nsight profile examples:

{{< figure src="./images/single_stream_report.png" caption="Figure 1: Nsight Systems profile with a single CUDA stream. Source code: [single_stream.txt](./code/single_stream.txt)." align="center" >}}

{{< figure src="./images/multi_stream_report.png" caption="Figure 2: Nsight Systems profile with multiple CUDA streams. Source code: [multi_stream.txt](./code/multi_stream.txt)." align="center" >}}



References:
1. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-and-overlapping-transfers-with-computation
