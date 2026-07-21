# vLLM Neuron Plugin (Beta) Documentation

The vLLM Neuron plugin brings the full vLLM serving stack to AWS Trainium
accelerators. It supports continuous batching, speculative decoding
(EAGLE3), disaggregated inference, structured outputs, multimodal models, and
more — all accessible through the standard `vllm serve` command and
OpenAI-compatible API.

For a high-level overview of inference on Neuron and help choosing the right
inference solution, see
[Inference on Neuron](/libraries/vllm-neuron/neuron-inference-overview). The
source code for the vLLM Neuron plugin is hosted in the
[vLLM Neuron GitHub repository](https://github.com/vllm-project/vllm-neuron).

---

## Get started

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Setup guide
:link: getting-started/setup-guide
:link-type: doc

Install and configure vLLM Neuron on a Trainium instance.
:::

:::{grid-item-card} Online serving quickstart
:link: getting-started/quickstart-online-serving
:link-type: doc

Launch an OpenAI-compatible API server and send your first chat request.
:::

:::{grid-item-card} Offline serving quickstart
:link: getting-started/quickstart-offline-serving
:link-type: doc

High-throughput batch inference with the `vllm.LLM` Python API.
:::

:::{grid-item-card} Migration from NxD Inference
:link: getting-started/migration-nxdi-to-vllm-neuron
:link-type: doc

Migrate existing NxDI deployments to the vLLM Neuron plugin.
:::

::::

## Deploy & serve

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Features guide
:link: guides/features-guide
:link-type: doc

Configure and tune all serving features — bucketing, quantization, DI, speculation, and more.
:::

:::{grid-item-card} Configuration reference
:link: guides/reference-configuration
:link-type: doc

All Neuron-specific options in `additional_config` and environment variables.
:::

:::{grid-item-card} Profiling workloads
:link: guides/how-to-profile-workloads
:link-type: doc

Capture Neuron Runtime profiles via built-in profiler endpoints.
:::

:::{grid-item-card} Feature–model compatibility
:link: guides/reference-feature-model-compatibility
:link-type: doc

Supported feature and model combinations.
:::

::::

## Model Recipes

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Deploy GPT-OSS
:link: model-recipes/gpt-oss
:link-type: doc

Model recipe for GPT-OSS 20B and 120B (MoE) on Trn2/Trn3.
:::

:::{grid-item-card} Deploy Qwen3-VL
:link: model-recipes/qwen3-vl
:link-type: doc

Model recipe for Qwen3-VL 32B (multimodal) on Trn2/Trn3.
:::

::::

## Tutorials

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} GPT-OSS deployment tutorial
:link: tutorials/tutorial-gpt-oss
:link-type: doc

End-to-end deployment of GPT-OSS on Trn2/Trn3.
:::

:::{grid-item-card} Qwen3-VL multimodal tutorial
:link: tutorials/tutorial-qwen3-vl-32b
:link-type: doc

Deploy Qwen3-VL 32B for multimodal inference.
:::

:::{grid-item-card} Disaggregated inference: 1P1D and xPyD
:link: tutorials/tutorial-di-1p1d-xpyd
:link-type: doc

Configure disaggregated inference topologies.
:::

:::{grid-item-card} Prefix caching benchmark
:link: tutorials/tutorial-prefix-caching-gpt-oss-benchmarking
:link-type: doc

Measure TTFT improvement from prefix caching with GPT-OSS.
:::

::::

## Model development

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Onboard a new model
:link: model-dev/onboarding-models
:link-type: doc

Implement and register a new architecture with vLLM.
:::

:::{grid-item-card} CPU development workflow
:link: model-dev/cpu-development
:link-type: doc

Develop and test without Neuron hardware.
:::

:::{grid-item-card} NKI CPU simulator
:link: model-dev/nki_cpu_simulator
:link-type: doc

Validate NKI kernel correctness on CPU.
:::

:::{grid-item-card} Debugging accuracy issues
:link: model-dev/accuracy-debugging-guide
:link-type: doc

Methodology for isolating where accuracy drift is introduced.
:::

::::

## Concepts & architecture

### Parallelism

| Topic | Description |
|---|---|
| [Data parallelism](design/parallelism/data_parallelism.md) | Data parallelism overview |
| [Expert parallelism](design/parallelism/expert_parallelism.md) | Expert parallelism for MoE |
| [Tensor parallelism](design/parallelism/tensor_parallelism.md) | Tensor parallelism overview |
| [Vision encoder parallelism](design/parallelism/vision_encoder_parallelism.md) | Vision encoder parallelism for multimodal models |

### Multimodal

| Topic | Description |
|---|---|
| [M-RoPE](design/multimodal/mrope.md) | Spatial position embeddings for VLMs |
| [Block packing attention](design/multimodal/block_packing_attention.md) | Block packing attention for multimodal |
| [On-device encoder cache](design/multimodal/on_device_encoder_cache.md) | On-device encoder cache design |

### vLLM integration

| Topic | Description |
|---|---|
| [KV cache integration](design/vllm/vllm-integration-kv-cache.md) | KV cache integration points with vLLM |
| [Async scheduling and execution](design/vllm/async-scheduling-and-async-execution.md) | Async scheduling and execution design |
| [Metrics](design/vllm/metrics.md) | Production metrics design |
| [Neuron profiling](design/vllm/neuron-profiling.md) | Profiling integration |
| [Neuron scheduler](design/vllm/neuron-scheduler.md) | Holdback queue and admission control |
| [Prefix caching](design/vllm/prefix-caching.md) | Prefill segmentation and KV reuse |
| [Disaggregated inference](design/vllm/disaggregated-inference.md) | DI architecture, NIXL transport, hybrid TP |
