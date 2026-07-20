# Feature and model compatibility matrix

<!-- meta: description: Lookup table for which features are supported with which models and on which instance types. -->
<!-- meta: date_updated: 2026-07-14 -->
<!-- Content type: reference-general -->
<!-- Jira: NDOC-183 -->

Use this page to check feature support before configuring a deployment.
For configuration details, see the [features guide](features-guide.md).

## Feature/model matrix

| Feature | [GPT-OSS](../model-recipes/gpt-oss.md) | [Qwen3-VL](../model-recipes/qwen3-vl.md) |
|---------|:---:|:---:|
| [Continuous batching](features-guide.md#continuous-batching) | ✅ | ✅ |
| [Segmented prefill](features-guide.md#segmented-prefill) | ✅ | ✅ |
| [Prefix caching (APC)](features-guide.md#prefix-caching) | ✅ | ✅ |
| [Speculative decoding (EAGLE3)](features-guide.md#speculative-decoding-eagle3) | ✅ | ❌ |
| [FP8 weight quantization (static)](features-guide.md#fp8-static-weight-quantization) | ❌ | ❌ |
| [MXFP8 weight quantization](features-guide.md#quantization) | ❌ | ✅ |
| [MXFP4 weight quantization](features-guide.md#mxfp4-weight-quantization-gpt-oss-on-trn3) | ✅ ¹ | ❌ |
| [KV cache FP8](features-guide.md#kv-cache-fp8-quantization) | ✅ | ❌ |
| [Multimodal (image input)](features-guide.md#multimodal-support) | ❌ | ✅ |
| [Disaggregated inference (1P1D / xPyD)](features-guide.md#disaggregated-inference) | ✅ | ❌ |
| [Structured outputs / tool calling](features-guide.md#structured-outputs-and-tool-calling) | ✅ | ✅ |
| [On-device sampling](features-guide.md#on-device-sampling) | ✅ | ✅ |
| [Tensor parallelism](features-guide.md#tensor-data-and-expert-parallelism) | ✅ | ✅ |
| [Data parallelism](features-guide.md#tensor-data-and-expert-parallelism) | ✅ | ✅ |
| [Expert parallelism](features-guide.md#tensor-data-and-expert-parallelism) | ✅ | N/A |
| [Vision encoder parallelism](features-guide.md#tensor-data-and-expert-parallelism) | N/A | ✅ |

¹ Trn3 only.

## Legend

- ✅ Supported and tested
- ❌ Not supported
- N/A Not applicable to this architecture

## Related

- [Features guide](features-guide.md) — configuration details for each feature
- [GPT-OSS recipe](../model-recipes/gpt-oss.md)
- [Qwen3-VL-32B recipe](../model-recipes/qwen3-vl.md)
