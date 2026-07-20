# Tutorial: Deploy Qwen3-VL-32B with vLLM Neuron

<!-- meta: description: End-to-end tutorial for deploying Qwen3-VL-32B with vLLM
on Neuron, covering environment setup, model download, online serving, and
offline inference for the multimodal Qwen3-VL-32B model on Trn2. -->
<!-- meta: keywords: vLLM, Neuron, Qwen3-VL, Qwen3-VL-32B, multimodal, VLM,
vision-language, tutorial, LLM serving, Trn2, Trainium -->
<!-- meta: date_updated: 2026-07-17 -->
<!-- Content type: procedural-tutorial -->
<!-- Jira: NDOC-183 -->

This tutorial walks through deploying [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) on a `trn2.48xlarge` instance using vLLM-Neuron. It covers environment setup, model download, online serving, and offline inference.

**Prerequisites:**

- A `trn2.48xlarge` instance (16 NeuronCores) with Neuron SDK `2.31.0` or later. See
  [setup guide](../getting-started/setup-guide.md).
- vLLM Neuron plugin `0.21.0` or above installed.
- Python 3.10+

## Step 1: Environment Setup

Verify Neuron devices are visible:

```bash
neuron-ls
# Expected: 16 NeuronCores listed for trn2.48xlarge
```

Set environment variables before running any inference script:

```bash
# Extend timeouts for large model compilation
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200
export VLLM_NEURON_COMPILATION_TIMEOUT=1200

# Required if your home directory is on NFS
export NEURON_CC_FLAGS="--temp-dir=/tmp/neuroncc_tmp"
mkdir -p /tmp/neuroncc_tmp
```

## Step 2: Download the Model

```bash
huggingface-cli download \
    Qwen/Qwen3-VL-32B-Instruct \
    --local-dir /path/to/Qwen3-VL-32B-Instruct
```

> **Tip:** On a `trn2` cluster, download to a shared filesystem instead of your home directory to avoid NFS write issues.

## Step 3: Online Serving

Start a vLLM OpenAI-compatible server:

```bash
vllm serve /path/to/Qwen3-VL-32B-Instruct \
    --served-model-name Qwen3-VL-32B-Instruct \
    --max-model-len 32768 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 8 \
    --tensor-parallel-size 16 \
    --additional-config '{
        "neuron_config": {
            "quantization": "bf16",
            "num_batched_tokens_buckets": [4096],
            "num_seqs_buckets": [8],
            "on_device_sampling_config": {"all_greedy": true}
        },
        "vision_neuron_config": {
            "num_vision_tokens_buckets": [2048],
            "vision_attention_block_size": 2048
        }
    }'
```

Once the server is up, send requests using the OpenAI Python SDK:

```python
import base64
from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

# Text-only
response = client.chat.completions.create(
    model="Qwen3-VL-32B-Instruct",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_tokens=50,
)
print(response.choices[0].message.content)

# Image + text
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Qwen3-VL-32B-Instruct",
    messages=[{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
        {"type": "text", "text": "Describe this image."},
    ]}],
    max_tokens=200,
)
print(response.choices[0].message.content)
```

## Step 4: Offline Inference

vLLM-Neuron compiles the model on the first run and caches the artifacts to `~/.cache/vllm/neuron/compile_cache`. Subsequent runs skip recompilation and load from cache.

### Configuration

The model has two components, each with its own config object passed via `additional_config` (see the [configuration options reference](../guides/reference-configuration.md)):

- **`neuron_config`**: text decoder settings (token/sequence bucket sizes, sampling, quantization).
- **`vision_neuron_config`**: vision encoder settings (vision token buckets, attention block size, and optional vision TP/DP split).

**Bucket sizes** control the discrete padded shapes compiled into each NEFF. Each bucket adds compile time; start with one and add more as needed. The two components bucket along different dimensions, so their buckets are configured separately:

- `num_batched_tokens_buckets` (in `neuron_config`): text-decoder buckets over the number of batched **text** tokens per forward pass. See [compilation options](../guides/reference-configuration.md#compilation-options).
- `num_vision_tokens_buckets` (in `vision_neuron_config`): vision-encoder buckets over the number of **vision** patches per encoder forward pass (raw `T*H*W` patches from `image_grid_thw`, before the 2x2 spatial merge; this is the count `select_vision_bucket` matches against). The scheduler may batch images from multiple requests into one forward pass, so size the buckets for the total images processed together, not a single request. These scale with image count and resolution:

| `num_vision_tokens_buckets` | Approximate capacity |
|-----------------------------|----------------------|
| `[2048]` | 1–2 images at 448×448 px |
| `[2048, 8192]` | Up to ~8 images |
| `[2048, 8192, 20480]` | Up to ~20 images |

### Run offline inference

The following script runs text-only, single-image, and multi-image inference:

```python
import os

os.environ["VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"] = "1200"
os.environ["VLLM_NEURON_COMPILATION_TIMEOUT"] = "1200"

from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset

MODEL_PATH = "/path/to/Qwen3-VL-32B-Instruct"

llm = LLM(
    model=MODEL_PATH,
    max_model_len=32768,
    max_num_batched_tokens=4096,
    max_num_seqs=8,
    tensor_parallel_size=16,
    additional_config={
        "neuron_config": {
            "quantization": "bf16",
            "num_batched_tokens_buckets": [4096],
            "num_seqs_buckets": [8],
            "on_device_sampling_config": {"all_greedy": True},
        },
        "vision_neuron_config": {
            "num_vision_tokens_buckets": [2048],
            "vision_attention_block_size": 2048,
        },
    },
)

processor = AutoProcessor.from_pretrained(MODEL_PATH)
sampling_params = SamplingParams(max_tokens=200, temperature=0.0)

# --- Text-only ---
outputs = llm.generate(["What is the capital of France?"], sampling_params)
print(outputs[0].outputs[0].text)

# --- Single image ---
image = ImageAsset("cherry_blossom").pil_image.resize((640, 320))
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "text", "text": "Describe this image."},
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": [image]}}], sampling_params)
print(outputs[0].outputs[0].text)

# --- Multi-image ---
images = [
    ImageAsset("stop_sign").pil_image.resize((448, 448)),
    ImageAsset("cherry_blossom").pil_image.resize((448, 448)),
]
messages = [{"role": "user", "content": [
    {"type": "image"},
    {"type": "image"},
    {"type": "text", "text": "Compare these two images."},
]}]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = llm.generate([{"prompt": prompt, "multi_modal_data": {"image": images}}], sampling_params)
print(outputs[0].outputs[0].text)
```

## Vision Parallelism (Optional)

By default the vision encoder runs as 16 independent DP replicas (TP1 DP16), one per NeuronCore, the recommended layout for high-throughput multi-image workloads. To shard the encoder weights across cores for a single large image instead, increase `tp_size`.

Set `tp_size` inside `vision_neuron_config` to change the split. DP is derived automatically as `world_size / tp_size`:

| `tp_size` | Vision TP | Vision DP | Best for |
|-----------|-----------|-----------|----------|
| `1` (default) | 1 | 16 | Multi-image, high throughput |
| `4` | 4 | 4 | Mixed workloads |
| `16` | 16 | 1 | Single-image, low latency |

Example — TP1 DP16:

```python
"vision_neuron_config": {
    "num_vision_tokens_buckets": [2048],
    "vision_attention_block_size": 2048,
    "tp_size": 1,  # DP = world_size / tp_size = 16
}
```

## Conclusion

You have successfully deployed Qwen3-VL-32B-Instruct on a `trn2.48xlarge` instance. The model inference on Trainium currently supports text-only, single-image, multi-image, and video inputs via both the offline `LLM` API and the OpenAI-compatible online serving endpoint. For accuracy validation results, see the [model card](../model-recipes/qwen3-vl.md).
