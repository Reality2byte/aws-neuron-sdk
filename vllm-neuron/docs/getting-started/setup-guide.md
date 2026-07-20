# How to set up vLLM Neuron

<!-- meta: description: Install and configure the vLLM Neuron plugin on AWS
Trainium or Inferentia instances so you can serve LLMs with vLLM Neuron. -->
<!-- meta: keywords: vLLM, Neuron, setup, install, vLLM Neuron plugin,
Trainium, Inferentia, DLAMI -->
<!-- meta: date_updated: 2026-06-11 -->
<!-- Content type: procedural-how-to -->
<!-- Jira: NDOC-179 -->

## Overview

This topic discusses how to install and configure the vLLM Neuron plugin on AWS
Trainium or Inferentia so you can serve large language models with vLLM on
Neuron. When you have completed it, you will have a working environment ready for
the online or offline serving quickstarts.

:::{note}
This guide is for the **vLLM Neuron** plugin introduced in Neuron 2.31. If you
are using vLLM with the NxD Inference library, see the
[vLLM + NxD Inference documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/libraries/nxd-inference/index.html).
For migration guidance from NxD Inference to vLLM Neuron, see
[migrate from NxD Inference to vLLM Neuron](migration-nxdi-to-vllm-neuron.md).
:::

## Prerequisites

- **Instance**: A supported Trainium EC2 instance (for example,
  `trn2.48xlarge`).
- **Neuron SDK version**: Neuron 2.31 or later.
- **Python**: Python 3.11 or later.
- **Hugging Face access (optional)**: An accepted model license and a Hugging
  Face token if you plan to pull gated models.

## Instructions

### Option A: Install from source

Clone the repository and install the plugin from source:

```bash
git clone https://github.com/vllm-project/vllm-neuron.git
cd vllm-neuron
pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .
```

This installs the vLLM Neuron plugin along with vLLM and all required Neuron
SDK packages.

### Option B: Use the Neuron DLAMI

Launch an instance with the Neuron Multi-Framework Deep Learning AMI. It ships
with the Neuron SDK, PyTorch NeuronX, and a pre-built virtual environment that
includes vLLM Neuron.

Launch an instance with the DLAMI by following the
[Neuron DLAMI setup instructions](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/deploy/environments/dlami.html).
After you connect, activate the vLLM virtual environment:

```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_21_0_1_0_0/bin/activate
```

### Option C: Use a container

Use the [vLLM Inference NeuronX DLC](https://github.com/aws-neuron/deep-learning-containers#vllm-inference-neuronx)
which bundles the SDK and all dependencies.

### Verify the installation

Confirm that vLLM discovers the Neuron platform plugin:

```bash
python -c "import vllm; from vllm.platforms import current_platform; print(current_platform.device_name)"
```

The output should be:

```text
neuron
```

Confirm the Neuron runtime is visible:

```bash
neuron-ls
```

You should see output listing the available NeuronCores on your instance.

### Configure environment variables

The following environment variables control common behaviors:

| Variable | Purpose | Default |
| --- | --- | --- |
| `VLLM_CACHE_ROOT` | Root directory for the Neuron compile cache. Set to local NVMe for best performance. | `~/.cache/vllm` |
| `VLLM_NEURON_LOG_LEVEL` | Plugin log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`). | `INFO` |
| `HF_TOKEN` | Hugging Face access token for gated model downloads. | (none) |

```bash
export VLLM_CACHE_ROOT=/local/cache/vllm
export HF_TOKEN=hf_your_token_here
```

## Confirm your work

Run a minimal `vllm serve` command to confirm the full stack is wired up. This
example uses gpt-oss-20b with a small context and a single compiled bucket to
keep compilation time short:

```bash
vllm serve openai/gpt-oss-20b \
    --tensor-parallel-size 8 \
    --max-num-seqs 1 \
    --max-model-len 1024 \
    --max-num-batched-tokens 512 \
    --hf-overrides '{"quantization_config": {}}' \
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [512], "num_seqs_buckets": [1]}}'
```

Wait for the server to print `Uvicorn running on ...`. Then, in a second terminal
on the same instance:

```bash
curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "openai/gpt-oss-20b",
      "messages": [{"role": "user", "content": "Hello, Neuron!"}],
      "max_tokens": 16
    }'
```

A JSON response containing generated text confirms that the plugin, Neuron
runtime, and model compilation are all working. Stop the server with `Ctrl+C`.

## Next steps

- [Quickstart: Online serving](quickstart-online-serving.md) — Launch an
  OpenAI-compatible API server.
- [Quickstart: Offline serving](quickstart-offline-serving.md) — Run batch
  inference with the Python API.
- [Features guide](../guides/features-guide.md) — Configure features like
  prefix caching, speculative decoding, data parallelism, and quantization.
- [Configuration options](../guides/reference-configuration.md) — Full
  parameter reference.

## Common issues

### Plugin fails to load the Neuron platform

- **Possible solution**: Confirm you activated the correct virtual environment and
  that `pip show vllm-neuron` reports version 0.21.0.1.0.0 or later. If you installed
  vLLM separately before the plugin, reinstall the plugin to ensure version
  alignment.

### `neuron-ls` shows no devices

- **Possible solution**: Verify you are on a Neuron-supported instance type
  (`trn2` or `trn3`). On a fresh instance, the Neuron runtime may need a
  reboot after driver installation. Check `dmesg | grep neuron` for driver
  messages.

### Compilation times out or is very slow on first run

- **Possible solution**: First-time model compilation on Neuron can take several
  minutes depending on model size. Subsequent runs use the compile cache. Ensure
  `VLLM_CACHE_ROOT` points to fast local storage (NVMe) rather than a network
  filesystem.

### Python version mismatch

- **Possible solution**: The vLLM Neuron plugin requires Python 3.11 or later. Run
  `python --version` to check. The DLAMI virtual environment ships with a
  compatible Python version.

## Related information

- [Quickstart: Online serving](quickstart-online-serving.md) — Launch an
  OpenAI-compatible API server with vLLM Neuron.
- [Quickstart: Offline serving](quickstart-offline-serving.md) — Run offline batch
  inference with vLLM Neuron.
- [Features guide](../guides/features-guide.md) — Detailed coverage of vLLM on
  Neuron features and configuration.
- [Neuron DLAMI options and setup](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/deploy/environments/dlami.html)
- [Neuron Deep Learning Containers](https://github.com/aws-neuron/deep-learning-containers#vllm-inference-neuronx)
  for Docker-based workflows.
