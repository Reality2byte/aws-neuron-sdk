# Concepts & architecture

How vLLM Neuron works under the hood — parallelism strategies, plugin integration, speculative decoding internals, and the accuracy validation framework.

## Parallelism

| Topic | Description |
| --- | --- |
| [Data parallelism](parallelism/data_parallelism.md) | Data parallelism overview |
| [Expert parallelism](parallelism/expert_parallelism.md) | Expert parallelism for MoE |
| [Tensor parallelism](parallelism/tensor_parallelism.md) | Tensor parallelism overview |
| [Vision encoder parallelism](parallelism/vision_encoder_parallelism.md) | Independent TP/DP for vision encoders |

## Multimodal

| Topic | Description |
| --- | --- |
| [Block Packing Vision Attention](multimodal/block_packing_attention.md) | FFD block packing for multi-image attention efficiency |
| [On-Device Encoder Cache](multimodal/on_device_encoder_cache.md) | Block-based on-device cache for vision encoder outputs |
| [M-RoPE](multimodal/mrope.md) | Spatial position embeddings for VLMs |

## vLLM integration

| Topic | Description |
| --- | --- |
| [KV cache integration](vllm/vllm-integration-kv-cache.md) | KV cache integration points with vLLM |
| [Async scheduling and execution](vllm/async-scheduling-and-async-execution.md) | Async scheduling and execution design |
| [Metrics](vllm/metrics.md) | Production metrics design |
| [Neuron profiling](vllm/neuron-profiling.md) | Profiling integration |
| [Neuron scheduler](vllm/neuron-scheduler.md) | Holdback queue and admission control |
| [Prefix caching](vllm/prefix-caching.md) | Prefill segmentation and KV reuse |
| [Disaggregated inference](vllm/disaggregated-inference.md) | DI architecture, NIXL transport, hybrid TP |

:::{toctree}
:maxdepth: 1
:hidden:

parallelism/index
multimodal/index
vllm/index
:::
