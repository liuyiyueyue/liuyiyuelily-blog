---
title: "[4/4] CPU-GPU Optimization: PyTorch"
date: 2026-02-19
tags: ["llm", "pytorch", "optimization", "cuda"]
---

This is the final post in the “CPU‑GPU Optimization” series. 
Here we use kernel‑ and CUDA‑level techniques from earlier posts to explain 
asynchronous execution in PyTorch and how to reduce CPU‑GPU synchronization costs.


## Asynchronous Execution

When a PyTorch program runs on the CPU, each line executes in program order. 
Below is a simple example:

```python
import torch

device = torch.device("cpu")

x = torch.randn(50_000_000, device=device)
y = torch.randn(50_000_000, device=device)
z = torch.randn(50_000_000, device=device)
```

{{< figure src="./1_cpu_execution_trace.png" caption="Figure 1: CPU execution timeline (sequential execution on main thread)<br><em>. Note: cudaDeviceSynchronize in this trace is introduced by the PyTorch Profiler instrumentation, not explicitly called in the example code.</em>" align="center" >}}


Now switch from CPU to GPU by using `device = torch.device("cuda")`, 
while keeping the rest of the code the same. The execution model changes completely. 
The CPU submits kernel and memcpy work to a CUDA stream and immediately continues. 
The GPU then fetches and executes operations in submission order. 
As a result, the Python code returns quickly even though the GPU is still busy. 
This collaboration between CPU and GPU is called **asynchronous execution**. 

```python
import torch

device = torch.device("cuda") # cpu replaced by gpu

x = torch.randn(50_000_000, device=device)
y = torch.randn(50_000_000, device=device)
z = torch.randn(50_000_000, device=device)
```

Below is a diagram: 

{{< figure src="./2_gpu_async_execution.png" caption="Figure 2: CPU-GPU asynchronous execution timeline<br><em>. Note: cudaDeviceSynchronize in this trace is also introduced by the PyTorch Profiler instrumentation, not explicitly called in the example code.</em>" align="center" >}}

In practice, if you want the CPU to wait for the GPU to finish, 
call `torch.cuda.synchronize`. For accurate timing, record timestamps 
only after a synchronize call, i.e. `time.time()` should follow 
`torch.cuda.synchronize`.
Here is a modified example with synchronization:

```python
import torch
import time

torch.cuda.synchronize()
t0 = time.time()

x = torch.randn(10_000_000, device=device)
for _ in range(100):
    y = x + 1
print("CPU finishes")

torch.cuda.synchronize()
t1 = time.time()
print("GPU finishes. Time:", t1 - t0)
```

{{< figure src="./3_synchronize_trace.png" caption="Figure 3: CPU waits for GPU completion after synchronization" align="center" >}}


During debugging, it’s fine to switch back to CPU execution or call `torch.cuda.synchronize`. 
In production, both reduce performance and should be avoided.


## Implicit Synchronization

`torch.cuda.synchronize` explicitly synchronizes CPU and GPU. 
However, some PyTorch operations are implicitly synchronous and can become performance 
bottlenecks. These operations often require the **CPU to know a data‑dependent result** 
(e.g., output shape, indices, or a scalar value) produced on the GPU before execution can continue.

The most common cases are **GPU tensor operations**, for example:

| Operation Name | Examples |
|---|---|
| indexing | `tensor.item()` and `tensor[0]` |
| copy to CPU | `tensor.cpu()` and `tensor.numpy()` |
| data-dependency | `print(tensor)` and `torch.nonzero(x)` — both operations must wait until the GPU kernel finishes |

Below is an example of how `torch.nonzero(z)` introduces implicit synchronization:
```python 
import torch

device = torch.device("cuda")

# Make workload large enough to show clear GPU time
N = 10_000_000
x = torch.randn(N, device=device)

torch.cuda.synchronize()

z = x
for _ in range(20): # all done asynchronously
	z = z * 1.000001

# ---- implicit synchronization ----
idx = torch.nonzero(z)

```

To execute `torch.nonzero(z)`, the GPU must finish all prior kernels that write to `z`, 
scan for non‑zero elements, and allocate an output tensor. The output tensor size is data‑dependent. 
Thus the CPU waits for these GPU operations to finish. 
There is no explicit `torch.cuda.synchronize` call, but this is still an implicit synchronization point.

{{< figure src="./4_implicit_sync.png" caption="Figure 4: Implicit synchronization in a data-dependent GPU operation" align="center" >}}


## Tensor Allocation and Memcpy

In PyTorch, by default, a CPU tensor uses pageable host memory, backed by `malloc` under the hood. 
To do a H2D memcpy, the CUDA runtime allocates a bounce buffer ([Compare to “bounce-buffer”](/llm/cpu_gpu_optimizations_1_kernel/#compare-to-bounce-buffer)).

Here is an example:
```python
x = torch.randn(10)  # Pageable host memory
y = x.to("cuda")     # Synchronous H2D transfer with bounce buffer, or `y = x.cuda()`
x_cpu = y.cpu()      # Synchronous D2H transfer (implicit sync)
```

With `pin_memory=True`, PyTorch calls `cudaMallocHost` and allocates pinned host memory.
With `non_blocking=True`, PyTorch calls `cudaMemcpyAsync`, submits work to the copy stream, and uses an event to synchronize with the compute stream. The CPU won’t wait for the H2D transfer to finish.
Here is an example:
```python
x = torch.randn(10, pin_memory=True)  # Pinned host memory
y = x.to("cuda", non_blocking=True)   # Asynchronous H2D transfer
x_cpu = y.cpu()                       # Synchronous D2H transfer (implicit sync)
```

To allocate a tensor on GPU, use the `device="cuda"` flag:
```python
x = torch.randn(10, device="cuda")
```

## Overlap Data Transfer and Computation (Double Buffering)

Now, using all the techniques above, we can achieve parallelism between data movement 
and GPU computation by assigning data transfers (e.g. `cudaMemcpyAsync`) 
and model execution kernels to different CUDA streams.

Concretely, we alternate submission of transfer tasks and compute tasks across two 
CUDA streams. As Stream A executes the kernels, Stream B does the data transfers 
for the next batch. Once they are done, the streams swap roles. 
This pipeline hides data transfer latency behind computation time.

This technique is similar to the double-buffering technique used in inference systems.


Conceptually, without any overlap, the total time is total copy time plus total 
compute time:

```
[ Copy0 ] → [ Compute0 ] → [ Copy1 ] → [ Compute1 ] → [ Copy2 ] → [ Compute2 ]
```

With double buffering, the total time is the max of total copy time vs. total 
compute time:


```
Compute:  [ Compute0 ]    [ Compute1 ]    [ Compute2 ]
Transfer:       [ Copy1 ]     [ Copy2 ]     [ Copy3 ]
```

Below are some examples. 

With double buffering:

```python
import torch

device = torch.device("cuda")

batch_size = 10_000_000
num_steps = 10

# Create two CUDA streams
compute_stream = torch.cuda.Stream()
transfer_stream = torch.cuda.Stream()

# Allocate pinned host buffers (required for async H2D overlap)
host_buffers = [
    torch.randn(batch_size, pin_memory=True),
    torch.randn(batch_size, pin_memory=True),
]

# Allocate device buffers
device_buffers = [
    torch.empty(batch_size, device=device),
    torch.empty(batch_size, device=device),
]

torch.cuda.synchronize()

for step in range(num_steps):

    buf_id = step % 2
    next_buf = (step + 1) % 2

    # 1. Launch compute on current buffer
    with torch.cuda.stream(compute_stream):
        if step > 0:
            x = device_buffers[buf_id]
            y = x * 2.0

    # 2. Launch transfer for next batch
    if step < num_steps - 1:
        with torch.cuda.stream(transfer_stream):
            device_buffers[next_buf].copy_(
                host_buffers[next_buf],
                non_blocking=True
            )

    # 3. Make compute wait for transfer of its buffer
    compute_stream.wait_stream(transfer_stream)

torch.cuda.synchronize()
```

{{< figure src="./5_double_buffer_trace.png" caption="Figure 5.1: Data-compute overlap with double buffering" align="center" >}}

Without double buffering and without pinned memory:

```python
import torch

device = torch.device("cuda")

batch_size = 10_000_000
num_steps = 10

host_buffer = torch.randn(batch_size) # Regular pageable host memory (NOT pinned)
device_buffer = torch.empty(batch_size, device=device) # Single device buffer

torch.cuda.synchronize()

for step in range(num_steps):

    # 1. Blocking H2D copy
    device_buffer.copy_(host_buffer)  # blocking copy

    # 2. GPU compute (default stream)
    y = device_buffer * 2.0

    # Optional: force full sync each iteration
    torch.cuda.synchronize()
```

{{< figure src="./5_no_double_buffer_trace.png" caption="Figure 5.2: No overlap" align="center" >}}
