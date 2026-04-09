---
title: "LLM Inference: Optimization Techniques"
date: 2026-02-26
tags: ["llm", "inference", "optimization"]
---

A survey summarizes the inference optimization methods into three levels, i.e., data-level optimization, model-level
optimization and system-level optimization, illustrated below [^1].

{{< figure
    src="images/inference_methods.png"
    alt="Overview of inference serving methods"
    caption="Overview of inference serving methods."
    align="center"
>}}

We already covered some in other posts of the `LLM Inference:` series:

- [LLM Inference: Prefill and Decode](/llm/inference_1_prefill_decode/)
- [LLM Inference: KV Cache](/llm/inference_2_kv_cache/)

This post first introduces several common inference optimization techniques, then explains when to use them to improve performance and reduce memory usage.

## Inference Optimization Techniques

### Paged Attention (vLLM)

Inference servers handle live traffic rather than a static batch job, so requests can have very different token lengths. This creates a challenge for KV cache allocation. A naive approach is to reserve a fixed-size buffer for the maximum token length for every request. Since each request has a different input length, much of that allocated memory remains unused.

![Three types of memory wastes](images/pagedattention.png)

PagedAttention is a memory management system for KV cache designed to solve this problem [^2]. Instead of allocating one large contiguous buffer, it divides the KV cache into fixed-size blocks ("pages"). Each request's KV cache is then represented as a linked list of blocks. This reduces fragmentation and makes allocation more flexible and efficient.

This is analogous to OS virtual memory:

| OS concept                 | vLLM equivalent                           |
| -------------------------- | ----------------------------------------- |
| Virtual page               | Logical KV block                          |
| Physical page              | Physical KV cache block in GPU memory     |
| Page table                 | Block table mapping logical to physical blocks |
| Memory allocator           | KV cache manager / block allocator        |


### Continuous Batching

Traditional static batching has to wait until all requests in a batch finish before processing the next batch, which causes earlier-finished requests to sit idle.
Continuous batching is a special case of dynamic batching and one of the main reasons vLLM outperforms older serving engines. It can *add new requests to an active batch* while removing finished ones at the same time. With PagedAttention, each request's KV cache is managed independently, so adding or removing requests does not disrupt the memory layout [^3].

{{< figure
    src="images/continuous_batching.png"
    alt="Continuous batching"
    caption="Each sequence generates one token (blue) from the prompt tokens (yellow). After several iterations, the completed sequences each have different sizes because each emits its end-of-sequence token (red) at a different iteration. Once a sequence emits an end-of-sequence token, a new sequence is inserted in its place."
    align="center"
>}}


### Speculative Decoding

In normal decoding, each new token requires a full forward pass through the large LLM. This is expensive, especially for long generations. Speculative decoding changes the computation pattern itself. Instead of generating one token per forward pass, a small, fast model first predicts multiple tokens. The large model then verifies these predictions in a single pass. If the predictions are correct, one expensive memory load yields multiple tokens. This effectively increases tokens produced per unit of memory bandwidth, improving single-user latency without changing hardware [^4] [^5].


### Chunked prefill

When prompts are long, the prefill phase can monopolize GPU compute and delay decode-heavy traffic. Chunked prefill addresses this by splitting a long prefill into equal-sized chunks and scheduling those chunks alongside decode requests [^6].

The key idea is to form hybrid batches: one prefill chunk keeps the GPU compute-saturated, while decode requests piggyback in the remaining slots. This improves utilization, reduces pipeline bubbles, and lowers tail latency compared with running large prefills as a single monolithic step.

### Disaggregated Inference

In LLM inference, the prefill and decode stages have different compute, memory, and bandwidth requirements. Prefill is typically compute-intensive, while decode is more memory-bandwidth-intensive. Running both on the same node can cause interference: a long prefill can occupy the GPU and delay decode steps for other requests. Disaggregated inference separates prefill and decode onto different processes or hosts to reduce this interference and improve overall serving performance [^7] [^8] [^9] [^10].

{{< figure
    src="images/prefill_decode_interference.png"
    alt="Prefill and decode interference versus disaggregated inference"
    caption="Prefill and decode interference when both stages share the same GPU resources versus disaggregated inference."
    align="center"
>}}

The system usually includes a connector or scheduler that transfers KV cache from prefill nodes to decode nodes. Within a node this can use NVLink or NVSwitch; across nodes it can use GPUDirect RDMA.

#### Heterogeneous disaggregation

Different systems split inference at different boundaries. Cerebras and AWS describe disaggregated inference by separating compute-bound prefill from memory-bound decode: AWS Trainium builds the KV cache during prefill, and Cerebras CS-3 handles decode for high token output speed [^11]. NVIDIA's Vera Rubin plus Groq 3 LPX design is different: prefill stays on GPUs, and decode itself is further split so GPUs run attention while LPX accelerates latency-sensitive FFN and MoE work [^12]. In other words, Cerebras and AWS separate the major phases of inference, while NVIDIA and Groq separate sub-operations inside the decode loop.

## Root Cause Bottlenecks

### Performance Metrics

Inference performance is primarily determined by two interrelated and often competing metrics: **throughput** and **latency**.

1. **Throughput** (吞吐量) measures the total amount of work a system can process per unit time. In large model systems, it is typically expressed as **tokens per second (tokens/s)**. It reflects overall system capacity and directly impacts cost: the more tokens generated within the same time window, the lower the cost per token.

2. **Latency** (延迟) measures the response time for a single request and can be further decomposed into two key metrics. **Time to First Token (TTFT)** is the time from request arrival to the first generated token, determining perceived responsiveness. **Time per Output Token (TPOT)** is the average time to generate each subsequent token, determining the effective generation speed.

Throughput and latency are inherently in tension. Improving throughput (e.g., via batching) often increases latency, while optimizing latency may reduce system utilization and increase cost. The key mechanism to balance this tradeoff is **concurrency**, which, through scheduling and resource management, mediates between cost efficiency and service quality (SLA).

Besides, **fault tolerance** is another nontrivial serving challenge. In a distributed LLM system, failures can occur at the GPU, host, interconnect, or rack level during live traffic. Because modern serving stacks schedule work at mixed granularities such as requests, groups, and tokens, they must preserve runtime state and recover quickly. Memory management is part of the same problem: some models fit at load time but still hit OOM on long sequences without robust runtime safeguards.


### Memory Bandwidth Bottlenecks

Many people wonder: GPUs today have extremely high compute capability, so why is inference still slow? In most inference cases, the primary bottleneck is not compute (FLOPs), but **memory bandwidth**—how fast data can be read from GPU memory.

**1. Single Request**

Large language models use autoregressive decoding, generating one token at a time. For each token, the GPU must stream essentially the entire model weights from HBM into GPU. This creates a hard upper bound on generation speed:

```text
max token/s ≈ memory bandwidth / model size
```

For example, a 32B parameter model with 4-bit quantization occupies about 18GB in memory. On a GPU with 900 GB/s bandwidth, the theoretical limit is:

```text
900 / 18 ≈ 50 tokens per second
```

In reality, additional overheads such as KV cache access and other system overheads further reduce this number.

**2. Concurrent Requests**

Since single-request latency is bounded by bandwidth, we can improve utilization via **batching**. Instead of serving one request at a time, the system aggregates multiple requests and processes them together. The model weights are still loaded once per step, but now produce tokens for multiple sequences simultaneously. This does not reduce latency for an individual request. In fact, queuing may slightly increase it, but it significantly improves overall throughput and cost efficiency. See [Batching](/llm/inference_1_prefill_decode/#batching) and [Continuous Batching](/llm/inference_3_optimizations/#continuous-batching).

vLLM decides batching dynamically under three constraints: token budget, sequence limit, and KV-cache availability. A request is only admitted if the scheduler still has token/sequence budget and the KV cache manager can allocate enough KV blocks for it.

In the same example, the system still needs to load the 18 GB model only once, but it can compute the next token for several users at the same time. As another example, a WallstreetCN report estimates that a single H20 can support about 500 concurrent users of the full DeepSeek model on WeChat. At that scale, 100,000 to 200,000 H20s would support roughly 50 million to 100 million concurrent users [^13].

**3. Mixture of Experts (MoE)**

Another approach is to reduce the amount of data moved per step. Mixture of Experts (MoE) architectures activate only a subset of parameters for each token.

Instead of loading the full 32B model, only a fraction (e.g., 8B) is used per step. This reduces memory traffic proportionally. In the same example, the effective data movement drops to ~4.5GB, increasing the theoretical throughput to:

```text
900 / 4.5 ≈ 200 tokens per second
```

**4. Speculative Decoding**

See [Speculative Decoding](/llm/inference_3_optimizations/#speculative-decoding).

For example, the 18 GB model is loaded just once, to verify whether all tokens generated by the small model are correct at the same time. 

**5. Specialized Hardware**

A more radical direction is to eliminate the memory bottleneck at the hardware level.

One approach places the entire model in on-chip SRAM, avoiding external memory access entirely, like Groq and Cerebras. Since SRAM bandwidth is orders of magnitude higher than HBM, this can dramatically increase token generation speed, albeit at high cost and limited capacity. 

An even more extreme approach compiles model weights directly into hardware circuits, like Taalas. In this design, computation becomes signal propagation through fixed wiring, with no concept of memory reads. Fully pipelined execution allows extremely high throughput (on the order of tens of thousands of tokens per second) and low power consumption, but the chip becomes immutable after fabrication.

### Memory Capacity Bottlenecks

Though most LLM inference is bounded by memory bandwidth, some scenarios are bounded by memory capacity. Memory bandwidth limits speed. Memory capacity limits feasibility. If the KV cache for long sequences or many concurrent requests does not fit, you get OOM or have to evict/preempt requests.

Common strategies to reduce memory capacity overhead in LLM inference are mostly about shrinking or managing the **KV cache**, since that is usually the dominant runtime memory consumer. See background in [KV Cache](/llm/inference_1_kv_cache).

Beyond the KV cache, the bottleneck may also come from **model weights** or **activation memory**. If the main issue is model weights, the usual solutions are lower precision, quantization, or weight compression. If the model still does not fit on a single GPU, it must be sharded across multiple GPUs, for example with tensor parallelism or pipeline parallelism. If the main issue is activation memory, especially during prefill, the typical fixes are to limit prompt length, reduce prefill batch size, or use chunked prefill so that a very long prompt is processed in smaller pieces.

In summary, below are three categories of optimizations:

System-level optimizations:
1. Reduce concurrency or max context length. This is the simplest operational control when memory is the limiting factor.
2. [Paged/block-based KV allocation](#paged-attention-vllm). Systems like vLLM's PagedAttention allocate fixed-size blocks instead of large contiguous buffers, which reduces fragmentation and wasted space.
3. Prefix cache reuse. Reuse KV blocks for requests with the same prompt prefix, often with copy-on-write, such as SGLang.
4. [Chunked prefill](#chunked-prefill). Split very long prompts into smaller chunks to avoid large temporary peaks during prefill.
5. KV eviction/offloading. Move cold KV blocks to CPU memory or evict lower-priority requests when GPU memory is tight. This often appears together with [disaggregated inference](#disaggregated-inference) at the system level.

Model-level optimizations:
1. Use MQA/GQA. Multi-Query Attention or Grouped-Query Attention reduces KV size by sharing keys/values across heads.
2. Sliding-window or local attention. Keep only a recent window of tokens instead of the full history, when the model architecture supports it.

Numerical-level optimizations:
1. Quantize the KV cache. Store KV in lower precision such as `FP8` or `INT8` instead of `FP16`/`BF16`.


[^1]: Zixuan Zhou, Xuefei Ning, Ke Hong, Tianyu Fu, Jiaming Xu, Shiyao Li, Yuming Lou, Luning Wang, Zhihang Yuan, Xiuhong Li, Shengen Yan, Guohao Dai, Xiao-Ping Zhang, Huazhong Yang, Yuhan Dong, and Yu Wang. A Survey on Efficient Inference for Large Language Models. arXiv, April 22, 2024. <https://arxiv.org/abs/2404.14294>
[^2]: Efficient Memory Management for Large Language Model Serving with PagedAttention. arXiv, September 12, 2023. <https://arxiv.org/abs/2309.06180>
[^3]: How continuous batching enables 23x throughput in LLM inference while reducing p50 latency. June 22, 2023. <https://www.anyscale.com/blog/continuous-batching-llm-inference>
[^4]: Fast Inference from Transformers via Speculative Decoding. arXiv, November 30, 2022. <https://arxiv.org/abs/2211.17192>
[^5]: Accelerating Large Language Model Decoding with Speculative Sampling. arXiv, February 2, 2023. <https://arxiv.org/abs/2302.01318>
[^6]: Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, and Ramachandran Ramjee. SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills. arXiv, August 31, 2023. <https://arxiv.org/abs/2308.16369>
[^7]: DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. arXiv, June 6, 2024. <https://arxiv.org/pdf/2401.09670>
[^8]: vLLM Project. "Disaggregated Prefill V1". GitHub Pull Request #10502. <https://github.com/vllm-project/vllm/pull/10502>
[^9]: Disaggregated Inference: 18 Months Later. November 3, 2025. <https://haoailab.com/blogs/distserve-retro/>
[^10]: Disaggregated Inference at Scale with PyTorch & vLLM. September 12, 2025. <https://pytorch.org/blog/disaggregated-inference-at-scale-with-pytorch-vllm/>
[^11]: James Wang. Cerebras is coming to AWS. March 13, 2026. <https://www.cerebras.ai/blog/cerebras-is-coming-to-aws>
[^12]: Kyle Aubrey and Farshad Ghodsian. Inside NVIDIA Groq 3 LPX: The Low-Latency Inference Accelerator for the NVIDIA Vera Rubin Platform. March 16, 2026. <https://developer.nvidia.com/blog/inside-nvidia-groq-3-lpx-the-low-latency-inference-accelerator-for-the-nvidia-vera-rubin-platform/>
[^13]: WallstreetCN. "一台 H20 能带动 500 个微信用户？一文看懂 DeepSeek 是怎么炼成的". <https://wallstreetcn.com/articles/3741189>
