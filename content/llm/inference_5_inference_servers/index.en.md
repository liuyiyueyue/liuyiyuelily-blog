---
title: "Inference Servers"
date: 2026-04-01
tags: ["llm", "inference", "serving", "system"]
---


How would we design an LLM inference serving system that supports multi-user concurrency, low latency, and high throughput? Such a system introduces a new set of systems challenges:
- How do we achieve low latency and high throughput when many user requests arrive concurrently?
- What should we do when the model is too large and the KV cache exceeds GPU memory capacity?
- How can we schedule GPU resources efficiently across many users and many models?

### vLLM

For simplicity, we use nano-vLLM as the example here [^1]. Its key components are:
1. `LLM Engine`: The top-level orchestration component of the inference service. It initializes the modules required for serving, exposes the request interface (generate), and coordinates the full request lifecycle, including input processing/encoding, scheduling, model execution, and output decoding.
2. `Scheduler`: Maintains pending requests through queues, organizes them, and dispatches the requests that need to be executed at each step.
3. `Model Runner`: Loads and runs the model, performing computation for each request. When `TP > 1`, the main process launches multiple `Model Runner` instances to jointly complete the forward pass.
4. `Block Manager`: Manages the GPU memory used for the KV cache based on PagedAttention.

**LLM Engine** 

LLM Engine is the core module of the inference framework. It creates one `Scheduler` instance and at least one `Model Runner` instance. Its main logic is encapsulated in the `generate` function, with data passed between modules through function arguments.

The process can be described as follows:

1. After receiving a user request, it uses the tokenizer to encode the prompt into token IDs and calls `add_request` to create a `Sequence` instance for each request.
2. It then calls `step`, which triggers the scheduler and lets it pass the pending request data to the `Model Runner`.
3. Finally, it decodes the token IDs into text and returns the result to the user, while the scheduler releases the corresponding resources.

Use tighter versions like these.

**Scheduler**

Scheduler is the component responsible for request scheduling and execution orchestration. It maintains two queues, `waiting` and `running`, and moves requests between them during execution. The default policy is `prefill`-first to reduce time to first token. During `decode`, if KV cache blocks are insufficient, it preempts later-admitted requests in `running` and moves them back to `waiting`. This design prioritizes admission latency and KV-cache feasibility over strict fairness. It also creates a `Block Manager` instance to manage KV cache blocks.

The process can be described as follows:

1. New requests are added to the `waiting` queue.
2. For runnable requests, it allocates the required KV cache blocks and records the mappings in the `block table`.
3. Scheduling proceeds in two stages, `prefill` and `decode`, with `prefill` prioritized by default.
4. After execution, it updates request states and releases KV cache resources for finished requests.
5. The generated results are returned to the upper-layer module for decoding.

**Model Runner**

Model Runner is the component responsible for model execution, including input preparation, forward passes, and sampling. When `TP > 1`, different ranks coordinate through `multiprocessing`, `SharedMemory`, and distributed communication. Rank 0 receives execution requests from the scheduler and shares the invocation data with other ranks. All ranks participate in the forward pass, but only rank 0 performs sampling and returns the final `token ids`. Its main execution entry point is `run()`.

The process can be described as follows:

1. At service startup, it loads model weights, performs warmup, allocates KV cache, and optionally captures CUDA Graphs.
2. Rank 0 receives runnable requests from the scheduler.
3. When `TP > 1`, rank 0 writes the invocation data into `SharedMemory`, and other ranks read it in `loop()` and execute the same method.
4. `run()` prepares the inputs for either `prefill` or `decode` and builds the required attention context.
5. `run_model()` performs the forward pass, and rank 0 samples from the resulting `logits` to produce `token ids`.

**Block Manager**

Block Manager manages KV cache blocks in GPU memory. It maintains the global block pool, tracks free and used blocks, and records each request’s block mappings in the `block table`. Following the PagedAttention design, it allocates KV cache in fixed-size blocks and supports prefix cache reuse through block hashing and reference counting.

The process can be described as follows:

1. When a request enters `prefill`, it checks whether enough KV cache blocks are available.
2. It allocates the required blocks and records them in the request’s `block table`; cached prefix blocks may be reused when available.
3. During `decode`, it checks whether the request can append new tokens and allocates a new block if needed.
4. It maintains reference counts for shared cached blocks.
5. When a request finishes or is preempted, it releases the corresponding blocks and returns unused blocks to the free pool.

[^1]: Nano-vLLM <https://github.com/GeeeekExplorer/nano-vllm>
