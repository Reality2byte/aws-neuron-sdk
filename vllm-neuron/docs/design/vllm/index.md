# vLLM integration

Design documentation for how vLLM Neuron integrates with the vLLM core
framework. For user-facing configuration, see the
[features guide](../../guides/features-guide.md) and
[configuration options](../../guides/reference-configuration.md).

| Topic | Description |
| --- | --- |
| [KV cache integration](vllm-integration-kv-cache.md) | KV cache integration points with vLLM |
| [Async scheduling and execution](async-scheduling-and-async-execution.md) | Async scheduling and execution design |
| [Metrics](metrics.md) | Production metrics design |
| [Neuron profiling](neuron-profiling.md) | Profiling integration |
| [Neuron scheduler](neuron-scheduler.md) | Holdback queue and admission control |
| [Prefix caching](prefix-caching.md) | Prefill segmentation and KV reuse |
| [Disaggregated inference](disaggregated-inference.md) | DI architecture, NIXL transport, hybrid TP |

:::{toctree}
:maxdepth: 1
:hidden:

vllm-integration-kv-cache
async-scheduling-and-async-execution
metrics
neuron-profiling
neuron-scheduler
prefix-caching
disaggregated-inference
:::
