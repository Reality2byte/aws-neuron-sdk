# Parallelism

Design documentation for parallelism strategies. For configuration guidance,
see [Tensor, data, and expert parallelism](../../guides/features-guide.md#tensor-data-and-expert-parallelism)
in the features guide.

| Topic | Description |
| --- | --- |
| [Data parallelism](data_parallelism.md) | Data parallelism overview |
| [Expert parallelism](expert_parallelism.md) | Expert parallelism for MoE |
| [Tensor parallelism](tensor_parallelism.md) | Tensor parallelism overview |
| [Vision encoder parallelism](vision_encoder_parallelism.md) | Independent TP/DP for vision encoders |

:::{toctree}
:maxdepth: 1
:hidden:

data_parallelism
expert_parallelism
tensor_parallelism
vision_encoder_parallelism
:::
