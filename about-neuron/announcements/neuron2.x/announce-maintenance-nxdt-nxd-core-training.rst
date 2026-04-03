.. post:: March 31, 2026
    :language: en
    :tags: announce-maintenance-nxdt

.. _announce-maintenance-nxdt-nxd-core-training:

Announcing maintenance mode for NxDT and NxD Core Training APIs starting next release
-------------------------------------------------------------------------------------

Starting with Neuron 2.30.0, NxDT and NxD Core Training APIs are entering maintenance mode. Future releases will address critical security issues only and we will gradually end support.

How does this impact you?
~~~~~~~~~~~~~~~~~~~~~~~~~

Existing NxDT/NxD Core users should stay on Neuron 2.28 and PyTorch 2.9 until ready to migrate to native PyTorch on Neuron (starting PyTorch 2.10). Customers are recommended to use native PyTorch with standard distributed primitives (DTensor, FSDP, DDP) and TorchTitan starting with Neuron 2.30.0 and PyTorch 2.10. A migration guide will be published in a coming release.

See :doc:`/frameworks/torch/pytorch-native-overview` for more information.
