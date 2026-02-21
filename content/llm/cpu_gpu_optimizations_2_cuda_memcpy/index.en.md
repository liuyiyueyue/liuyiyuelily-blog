---
title: "[2/4] CPU-GPU Optimization: CUDA Memory Allocation and Memcpy Methods"
date: 2026-02-05
tags: ["llm", "optimization", "cuda"]
---

This is the second blog of the "CPU-GPU Optimization" series. Using the foundations 
built from the previous blog, we will discuss 4 types of memory allocation and transfer 
methods in CUDA. Then we will showcase a performance analysis with flamegraph profiles.


#### 1. Normal Memcpy

**Data moves slowly and may incur an extra CPU memcpy**

- `malloc` to allocate memory on CPU.
- `cudaMalloc` to allocate memory on GPU.
- `cudaMemcpy` to transfer data from the CPU memory to the GPU memory. This call hides a subtle performance trap. 
As described in the [Compare to "bounce-buffer"](/llm/cpu_gpu_optimizations_1_kernel/#compare-to-bounce-buffer) 
section of the previous blog, the kernel implicitly allocates a temporary buffer and results in **an extra CPU memcpy**.
- `free` to free the CPU memory.
- `cudaFree` to free the GPU memory.

The below example shows a host-to-device transfer:
```cuda
size_t transfer_size = N * sizeof(float);

float *host_mem = malloc(transfer_size);
float *device_mem;
cudaMalloc((void **)&device_mem, transfer_size);

cudaMemcpy(device_mem, host_mem, transfer_size, cudaMemcpyHostToDevice);

free(host_mem);
cudaFree(device_mem);
```

#### 2. Pinned Memory + Memcpy

**Faster transfers, most common and balanced**

- `cudaMallocHost` replaces `malloc` to **allocate pinned memory on CPU**.
- `cudaMalloc` stays exactly the same to allocate memory on GPU.
- `cudaMemcpy` stays exactly the same, but its internal execution mechanism is completely different.
- `cudaFreeHost` replaces `free` to free the pinned CPU memory.
- `cudaFree` stays exactly the same to free the GPU memory.

The below example shows a host-to-device transfer with pinned host memory:
```cuda
size_t transfer_size = N * sizeof(float);

float *host_mem;
cudaMallocHost((void **)&host_mem, transfer_size);
float *device_mem;
cudaMalloc((void **)&device_mem, transfer_size);

cudaMemcpy(device_mem, host_mem, transfer_size, cudaMemcpyHostToDevice);

cudaFreeHost(host_mem);
cudaFree(device_mem);
```

References: 
- CUDA Official Documentation: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#pinned-memory


#### 3. Zero-copy

**GPU stores no data, but remotely accesses and modifies CPU memory**

- `cudaHostAlloc` replaces `cudaMallocHost`. The `flags` parameter is the key, and we use `cudaHostAllocMapped`, which allocates **pinned** CPU memory 
and **maps this memory into GPU address space**.
- `cudaHostGetDevicePointer` obtains the GPU-mapped address of the host pinned memory.
- `cudaFreeHost` replaces `free`.
- Zero-copy uses **no device memory**; data always resides in **pinned host memory**. So there is no `cudaMemcpy` needed. 
The CUDA driver registers the host physical pages with the GPU MMU and installs GPU page table entries that map GPU virtual addresses to CPU physical addresses.
During kernel execution, the GPU directly reads and writes host memory **via PCIe** (or NVLink) memory transactions rather than DMA transfers.
- Zero-copy turns compute intensity into "memory-access intensity." Thus, it is rarely used for compute-intensive workloads such as ML and HPC.

The below example shows a GPU kernel directly accesses host memory without an explicit `cudaMemcpy`:
```cuda
__global__ void editData(float* data, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N)
    data[idx] += 2.0f;
}

int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
size_t transfer_size = N * sizeof(float);

float *host_mem;
cudaHostAlloc((void **)&host_mem, transfer_size, cudaHostAllocMapped);
float *device_ptr;
cudaHostGetDevicePointer((void**)&device_ptr, host_mem, 0);

editData<<<blocksPerGrid, threadsPerBlock>>>(device_ptr, N);
cudaDeviceSynchronize();

cudaFreeHost(host_mem);
```

References: 
- CUDA Official Documentation: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#zero-copy


#### 4. Unified Memory (UVM)

**A unified address space across CPU and all GPUs**

- `cudaMallocManaged` allocates a unified virtual address whose physical pages reside on either CPU or GPU memory.
When it is called, the kernel driver only reserves a virtual address (VA), but CPU or GPU physical adress (PA) pages are populated lazily on the first access.

- Different from zero-copy and pinned memory approaches, with UVM, CPU and all GPUs share the **same virtual address space**. 
At any moment, a VA maps to **one side's** physical pages: either CPU DRAM or GPU HBM. 

- With UVM, there is no need for a device pointer via `cudaHostGetDevicePointer`. 
Pages migrate on demand between CPU and GPU memory via **CPU/GPU page faults**, with residency managed by the CUDA driver.
When the CPU accesses a VA whose page is resident in GPU memory and not CPU-accessible, a CPU page fault occurs. 
The kernel driver migrates the page by DMA from GPU memory to host memory, update page tables, and resume execution with the page now resident on the CPU.
Vice versa, but it's a GPU page fault.

- UVM is supported since CUDA 6.0, starting with Kepler-generation GPUs.

- ML / HPC rarely use UVM as well.

The below example shows an example of UVM:
```cuda
float *mem;
cudaMallocManaged((void **)&mem, N * sizeof(float));
for (int i = 0; i < N; i++) {
  mem[i] = 1.0f;
}

editData<<<blocksPerGrid, threadsPerBlock>>>(mem, N);
cudaDeviceSynchronize();

cudaFree(mem);
```

References:
- CUDA Official Documentation: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#unified-virtual-addressing


#### Overhead of Pinning Memory

The previous post briefly mentions the cost introduced by pinning memory. In this section, we dive deep 
into it. 

Pinning memory is not free: `cudaMallocHost`/`cudaHostAlloc`/`cudaHostRegister` must lock physical pages and update page tables, which can take noticeable time compared with ordinary `malloc`.
This setup cost is usually paid on the CPU and can dominate workloads that transfer small buffers or frequently allocate/free pinned regions. Excessive pinned memory also reduces paging flexibility for the OS, which may hurt overall system responsiveness under memory pressure.

In practice, pin once and reuse buffers across many transfers so the higher one-time pinning cost is amortized by faster H2D/D2H bandwidth.

{{< figure src="./images/pageable_flamegraph.png" caption="Figure 1: Flamegraph for using malloc and pagable memory. Page faults (via `asm_exc_page_fault`) triggered by `malloc` and `host_mem[i] = 1.0f` take ~56% of total time. Source code: [pageable_flame.cu](./code/pageable_flame.txt)." >}}

{{< figure src="./images/pin_mem_flamegraph.png" caption="Figure 2: Flamegraph for using pinned memory. Pinning host memory (via `pin_user_pages`) takes ~90% of `cudaMallocHost` time. Source code: [pin_mem_flame.cu](./code/pin_mem_flame.txt)." >}}

{{< figure src="./images/zerocopy_flamegraph.png" caption="Figure 3: Flamegraph for using zero-copy. Pinning host memory (via `pin_user_pages`) takes ~90% of `cudaHostAlloc` time. Source code: [zerocopy_flame.cu](./code/zerocopy_flame.txt)." >}}
