# Model development

Onboard new model architectures, debug accuracy issues, and develop without hardware. For developers adding or validating models on vLLM Neuron.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Onboard a new model
:link: onboarding-models
:link-type: doc

Implement and register a new architecture with vLLM.
:::

:::{grid-item-card} CPU development workflow
:link: cpu-development
:link-type: doc

Develop and test without Neuron hardware.
:::

:::{grid-item-card} NKI CPU simulator
:link: nki_cpu_simulator
:link-type: doc

Validate NKI kernel correctness on CPU.
:::

:::{grid-item-card} Debugging accuracy issues
:link: accuracy-debugging-guide
:link-type: doc

Methodology for isolating where accuracy drift is introduced.
:::

::::

:::{toctree}
:maxdepth: 1
:hidden:

Onboarding a model <onboarding-models>
CPU development workflow <cpu-development>
NKI CPU simulator <nki_cpu_simulator>
Debugging accuracy issues <accuracy-debugging-guide>
:::
