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

- [LLM Inference: KV Cache](/llm/inference_2_kv_cache/)
- [LLM Inference: Prefill and Decode](/llm/inference_1_prefill_decode/)

This post covers more optimization methods used in LLM inference.

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

(这章写的不走心啊！)

In normal decoding, each new token requires a full forward pass through the large LLM. This is expensive, especially for long generations. Speculative decoding uses a small, fast draft model to propose several tokens ahead, and then the large target model verifies them in parallel [^4] [^5].


### Chunked prefill
(这章也写的不走心啊！)

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
