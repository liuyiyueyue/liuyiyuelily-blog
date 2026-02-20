---
title: "[1/4] CPU-GPU Optimization: Pinned Memory in Linux Kernel"
date: 2026-01-25
tags: ["llm", "kernel", "optimization"]
---

This is the first post in the “CPU‑GPU Optimization” series. 
It lays the kernel‑level foundation for pinned memory, which later posts build on.

### What is pinned memory?

- Kernel pinned memory (also called **page-locked memory**) is memory in a computer system that the operating system marks as **non-swappable**—meaning it cannot be moved out of RAM into swap space (disk).


### Why it matters?

- **Normal memory** in an OS can be paged out to disk by the virtual memory manager. This is fine for general workloads but not for I/O devices that need **direct and fast access**.
- **Pinned memory** is “locked” into physical RAM so the OS guarantees it will always stay resident.
- This is crucial for **DMA (Direct Memory Access)**: when a device (like a GPU, NIC, or storage controller) directly reads/writes memory, the physical addresses must not change during the transfer.
- In GPU computing (CUDA, etc.), pinned memory allows **faster host–device transfers** because the GPU can DMA directly without needing a temporary bounce buffer (**zero-copy**), e.g., CUDA `cudaMallocHost` and `cudaHostAlloc`.


### Trade-offs

- Pinned memory reduces OS flexibility. The locked pages can’t be swapped, so excessive use can lower overall system performance.
- Allocation (pinning memory) is slower compared to pageable memory.


### Some terminologies

- “pageable”: a virtual memory page that can be **paged out** to disk (**swap**) and later **paged in**. It can change physical memory location.
- “pinned” or “non-pageable”: can’t be swapped or moved. e.g. pinned by kernel or allocated by kernel.
- “to fault pages”: to trigger page faults on virtual addresses so the kernel resolves them, bringing the pages into memory, creating PTEs, and making them resident/pinnable (or failing if the mapping is invalid).
- “GUP” stands for “Get User Pages.”
- “nr”: In Linux kernel naming conventions, `nr` means “number of”. nr_pages refers to the number of pages. (LOL why do I put it here..?)


### Kernel APIs

Linux kernel supports pin and unpin APIs starting on v5.6.

There are three APIs to pin memory so the kernel (or a device) can access them:
```
pin_user_pages()
pin_user_pages_fast()
pin_user_pages_remote()
```

Take pin_user_pages() as example:
```
long pin_user_pages(unsigned long start, unsigned long nr_pages,
		    unsigned int gup_flags, struct page **pages)
```
- `start`: The starting virtual address, aligned to the start of its page.
- `nr_pages`: Number of pages to pin (an integer count).
- `gup_flags`: Controls write permissions for the pinned pages. If this argument is zero the pages are pinned read-only (host to device copy); if non-zero (here FOLL_WRITE) they are pinned for writing (device to host copy).
- `pages`: Output array (struct page *[]) where pointers to the pinned pages are stored (one entry per page).
- Returned value: the number of pages actually pinned (>= 0) or a negative errno on error.

pin_user_pages_fast() works similarly, but it attempts to pin user pages by walking the page tables directly and without taking locks (the `mm->mmap` lock). When _fast path fails, slow path can be used.

Remote pinning is done via pin_user_pages_remote():
```
long pin_user_pages_remote(struct mm_struct *mm,
                           unsigned long start,
                           unsigned long nr_pages,
                           unsigned int gup_flags,
                           struct page **pages,
                           struct vm_area_struct **vmas);
```
It is a variant of pin_user_pages() that allows one to pin pages belonging to another process (and that process’s address space), not the current one. It requires a reference to that process’s `mm_struct`.
Sample usages of remote pinning are:
- In io_uring, workers running in kernel threads need to pin buffers from the submitter process of userspace.
- User posts a buffer but the kernel thread running the RDMA work queue is not “in” the user’s mm.


These are the APIs to unpin the memory:
```
unpin_user_pages()
unpin_user_pages_dirty_lock()
```

Take `unpin_user_pages()` as example:
```
void unpin_user_pages(struct page **pages, unsigned long npages)
```
This API releases the memory previously pinned by `pin_user_pages_*`. 

When the pinned pages have been modified, `unpin_user_pages_dirty_lock()` should be used.

One can OOM the host (run out of physical memory, not virtual) if one continuously calls the three pinning APIs without eventually unpinning the pages or limiting how many one pins.


### Compare to “bounce-buffer”

A bounce buffer is a piece of kernel-allocated staging memory that is
contiguous and DMA-accessible.

User-space pages may be discontiguous, pageable (unpinned), or otherwise
inaccessible to DMA due to device or IOMMU addressing constraints.
As a result, the kernel allocates a contiguous DMA-capable bounce buffer.

{{< figure src="./images/bounce_buffer.png" caption="Memcpy with Bounce Buffer" >}}

For a host-to-device transfer, the flow is:
1. CPU copies data from the user buffer into the kernel bounce buffer.
2. DMA engine reads from the bounce buffer and transfers data to the device.

The problem with using a bounce buffer is that host-to-device transfers
require an extra CPU memcpy:
- Copy 1: User buffer → kernel bounce buffer. This is done via a CPU memcpy.
- Copy 2: Kernel bounce buffer → device buffer. This is done via PCIe DMA.

Using pinned memory, we eliminate that extra CPU memcpy, but at the cost of the 
additional time required to pin memory, plus the risk of running out of pagable 
memory.

{{< figure src="./images/pinned_memory.png" caption="Memcpy with Pinned Memory" >}}

In the next blogs, we will dive deeper into the performance analysis of pinned-memory and zero-copy and discuss the cost of pinning memory.
