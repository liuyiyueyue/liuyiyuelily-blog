---
title: "GPU Architecture"
date: 2026-03-24
tags: ["llm", "gpu", "architecture", "cuda"]
---

### Grace-Blackwell as a Superchip

Recent NVIDIA systems stop treating the CPU and GPU as two loosely connected devices over PCIe. The Grace-Blackwell design instead packages one **Grace CPU** with two **Blackwell GPU dies** into a tightly coupled module, connected by **NVLink-C2C**. The point is not just faster I/O. It is to make CPU memory and GPU memory part of a coherent programming model with much lower communication overhead than a conventional host-device setup.

The first version of this idea was **Grace Hopper (GH200)**: one Grace CPU plus one Hopper GPU. The next step is **GB200**, which combines one Grace CPU with two Blackwell GPU dies on one module. In practice, the CPU sits in the middle of the package, with the two GPU dies around it, and the whole module exposes a shared memory space.

{{< figure src="./images/figure-2-1.png" caption="Grace-Blackwell superchip module: one Grace CPU centered between two Blackwell GPU dies on a single package." align="center" >}}

### Unified but Not Uniform Memory

The most important architectural idea is **cache-coherent unified memory**. Grace contributes roughly **480 GB of LPDDR5X**, while each Blackwell GPU provides roughly **192 GB of HBM3e** physically attached to the GPU package. Software can treat the CPU and GPU memory as one address space, and hardware coherence removes much of the old explicit-copy model.

But unified memory does **not** mean uniform performance.

- **HBM3e** is still the fastest memory tier for GPU compute.
- **LPDDR5X** is larger, but farther away from the GPU cores.
- GPU access to CPU memory over **NVLink-C2C** is much faster than going out to SSD, but still slower than hitting local HBM.

So the real benefit is not that CPU memory replaces HBM. The benefit is that it extends the effective memory pool for very large models. If the model and runtime state fit in the combined CPU+GPU memory footprint, the system can continue executing without immediately spilling to storage.

### Why Grace Exists

Grace is not just a host processor bolted onto GPUs. Its role is to keep the accelerator fed without becoming the bottleneck.

In older PCIe-attached systems, the CPU often acts as a slow control plane for a much faster GPU data plane. Grace changes that balance. It is designed so data movement, control, and memory capacity scale with the GPU complex, which matters for LLM training and inference where input pipelines, parameter movement, checkpointing, and orchestration can otherwise starve the GPUs.

### Blackwell as a Dual-Die GPU

Blackwell also changes the internal GPU packaging model. Instead of building one enormous monolithic die, NVIDIA splits the GPU into **two GPU dies** connected by an on-package **NV-HBI** link with roughly **10 TB/s** of die-to-die bandwidth.

This is a classic chiplet-style tradeoff:

- smaller dies are easier to manufacture at high yield
- the package can scale beyond monolithic reticle limits
- the interconnect must be fast enough that software can still treat the result as one GPU

That last point is critical. The CUDA stack presents the pair of dies as a **single logical GPU**, not as two independent devices. That abstraction only works because the die-to-die bandwidth is high enough to make the split mostly invisible at the programming model level.

### Memory Capacity and Cache

For the Blackwell B200, the nominal HBM3e capacity is **192 GB**, but the usable capacity is closer to **180 GB** after ECC and reserved overhead. That distinction matters for system planning: practitioners should size models and batch configurations against the usable memory, not the marketing number.

Blackwell also provides a large **L2 cache**: roughly **126 MB total**, or **63 MB per die**. That large shared cache is important because modern GPU performance is dominated not just by arithmetic throughput, but by whether data can be reused before going back to HBM.

### Tensor Cores and the Transformer Engine

The reason NVIDIA GPUs dominate transformer workloads is not just raw FLOPs. It is the combination of:

- **Tensor Cores** for high-throughput matrix multiply
- support for low-precision formats such as **FP8** and **4-bit**
- the **Transformer Engine (TE)** for mixed-precision execution

The Transformer Engine automates a key optimization: use higher precision where numerical stability matters, and lower precision where bandwidth and throughput dominate. In practice, users typically enable mixed precision at the framework level, while the hardware and runtime choose efficient precision paths for each layer.

This is exactly aligned with transformer workloads, where matrix multiply dominates runtime and where carefully managed low precision can preserve model quality while significantly increasing throughput.

### SMs, Warps, and the Execution Model

At execution time, the GPU is organized around many **streaming multiprocessors (SMs)**. Each SM contains arithmetic units, Tensor Cores, load/store units, and fast local storage. Threads execute in groups of **32** called **warps**, following the **SIMT** model: one instruction stream applied across many threads in lockstep.

A useful mental model is:

- registers: private to each thread
- shared memory: local to a thread block on an SM
- L1 cache: local to an SM
- L2 cache: shared across the whole GPU
- HBM: large off-chip device memory

This hierarchy explains a lot of CUDA performance behavior. Good kernels maximize data reuse in registers, shared memory, and caches. Bad kernels repeatedly go out to HBM and waste the GPU's arithmetic throughput waiting on memory.

{{< figure src="./images/figure-2-5.png" caption="GPU hardware hierarchy and the CUDA execution model: SMs execute warps of 32 threads over a layered memory hierarchy." align="center" >}}

### Summary

The Grace-Blackwell generation is not just a faster GPU. It is a packaging and memory-system redesign:

- **Grace + Blackwell** reduces the CPU-GPU boundary
- **dual-die Blackwell** extends GPU scale beyond monolithic limits
- **NVLink-C2C** and **NV-HBI** make the package behave like a coherent compute unit
- **HBM + LPDDR5X** gives a larger effective memory pool, but with clear performance tiers
- **Tensor Cores + Transformer Engine** map directly onto transformer math

For LLM systems, that means the unit of design is no longer just "a GPU." It is increasingly the **superchip**: a tightly integrated compute, memory, and interconnect package.
