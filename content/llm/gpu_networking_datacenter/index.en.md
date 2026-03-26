---
title: "GPU Networking and Datacenter Architecture"
date: 2026-03-25
tags: ["llm", "gpu", "networking", "datacenter"]
---

### From One GPU to One Rack

Once a model no longer fits on one GPU, system performance becomes a networking problem as much as a compute problem. NVIDIA's Grace-Blackwell systems push this idea to the rack level with **NVL72**, a design that packages **72 Blackwell GPUs** and **36 Grace CPUs** into one tightly connected system.

The key engineering goal is simple: keep as much communication as possible on the fastest fabric, and only leave that fabric when scale requires it. [^1]

### NVL72 Topology

An **NVL72** rack is built from **18 compute nodes**. Each node contains two GB200/GB300 superchips, for a total of:

- 4 Blackwell GPUs
- 2 Grace CPUs

Across the full rack, those GPUs are tied together by NVLink and NVSwitch, creating a high-bandwidth, low-latency network inside the rack.

This matters because large-model training and inference are dominated by communication patterns such as:

- tensor-parallel all-reduce
- expert-parallel all-to-all
- KV-cache movement
- parameter, gradient, and activation exchange

If those operations stay on the NVLink fabric, they are dramatically cheaper than sending traffic over a traditional cluster network.

### NVLink and NVSwitch

`NVLink` and `NVSwitch` solve two different problems. **NVLink** is the high-speed point-to-point interconnect that carries traffic between GPUs, or between CPU and GPU in designs such as `NVLink-C2C`. **NVSwitch** is the switching chip built on top of those NVLink connections, allowing many GPUs to communicate through a shared fabric instead of requiring direct wiring between every pair. In short, NVLink is the link, while NVSwitch is the network.

Each Blackwell GPU exposes 18 NVLink 5 ports. Those ports are not used as direct point-to-point cables between every GPU pair. Instead, GPUs connect into NVSwitch chips, which act like a specialized switch fabric for NVLink traffic.

The result is effectively a rack-scale nonblocking GPU interconnect:

- any GPU can reach any other GPU through the switch fabric
- bandwidth is much more uniform than ad hoc PCIe trees
- the rack behaves more like one large accelerator than like 72 isolated devices

This is why people sometimes say these systems "treat many GPUs as one." That is not literally true from a software abstraction standpoint, but for many collective operations the communication fabric is fast and regular enough that the rack can be programmed as a very tightly coupled parallel machine.

### Multi-GPU Programming Model

Inside the rack, one GPU can directly access another GPU's memory over NVLink using peer-to-peer mechanisms and PGAS-style programming models.

Across nodes, the model changes. Communication is typically built on **RDMA**, which allows direct memory access over the network without CPU copies. NVIDIA extends this with **GPUDirect RDMA**, which lets a NIC register GPU memory and perform DMA directly to and from that GPU memory.

That removes two common bottlenecks:

- no staging through host DRAM
- much less CPU involvement in the data path

Upper-layer libraries such as **NVSHMEM** build one-sided communication and remote atomic semantics on top of these RDMA transports.

The practical consequence is straightforward: if an algorithm can be structured to keep most of its traffic inside one NVL72 rack, it should. Inter-rack communication is still necessary at scale, but it is materially slower and should be treated as the expensive tier.

### In-Network Aggregation and SHARP

**SHARP** is NVIDIA's mechanism for doing some collective reduction work inside the network fabric rather than on the endpoints. Conceptually, instead of every GPU or host participating fully in a reduction tree, the switch fabric performs part of the aggregation on the data while it is in flight.

For collectives such as all-reduce, this can reduce:

- endpoint work
- network congestion
- end-to-end latency

So the right mental model is: **SHARP moves part of the collective algorithm into the network itself**.

### Beyond the Rack: NICs and DPUs

NVLink solves intra-rack GPU communication. It does not solve communication to:

- other NVL72 racks
- shared storage
- external services

That tier is handled by high-speed **InfiniBand** or **Ethernet** NICs, often paired with **BlueField DPUs**. A **DPU** is effectively a programmable network/storage offload processor. In these systems it handles line-rate packet processing, RDMA, and **NVMe over Fabrics** operations so the host CPUs do not spend cycles on packet handling, protocol processing, and interrupt overhead.

This changes the node architecture in an important way. The Grace CPU is not forced to sit in the middle of every storage or network transfer. The DPU and NIC can move data much closer to the actual data path, and **GPUDirect RDMA** can connect that path directly to GPU memory.

### Storage and Data Movement

At cluster scale, storage traffic also becomes a systems bottleneck. Training jobs need to ingest checkpoints, datasets, and optimizer state at very high rates. The combination of:

- **NICs** for external connectivity
- **DPUs** for protocol offload
- **GPUDirect RDMA** for direct GPU-memory access

reduces copies and CPU overhead in that path.

The architectural pattern is consistent across the stack:

- use **NVLink/NVSwitch** for the fastest local communication
- use **RDMA fabrics** for cross-rack and storage traffic
- use **DPUs** to offload infrastructure work from the CPU

### Power and Cooling Are First-Class Constraints

A fully populated **NVL72** rack can consume roughly **130 kW**:

- about **110 kW** from the 18 compute nodes
- about **20 kW** from NVSwitch trays, network gear, and cooling overhead

At that density, cooling is no longer a peripheral detail. It is a hard architectural constraint.

Traditional air cooling does not scale well to this power envelope, so NVL72 uses **liquid cooling**. Cold plates are attached directly to the superchips and NVSwitch components, and coolant loops carry heat out of the rack. This is not optional plumbing. Without aggressive thermal design, the rack could not sustain its target compute density.

### Summary

The main lesson is that modern AI infrastructure is hierarchical:

- inside a package: **NVLink-C2C** and coherent memory
- inside a rack: **NVLink + NVSwitch**
- across racks: **InfiniBand/Ethernet + RDMA**
- for offload: **NICs + DPUs**

The best software architecture follows the same hierarchy. Keep communication local when possible, treat inter-rack traffic as expensive, and design around the fact that power delivery and cooling are now core parts of system design, not just facility details.

[^1]: NVIDIA Technical Blog. "NVIDIA GB200 NVL72 Delivers Trillion-Parameter LLM Training and Real-Time Inference." <https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/>
