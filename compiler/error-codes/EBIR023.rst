.. _error-code-ebir023:

.. meta::
   :description: AWS Neuron SDK Graph Compiler error code documentation for error EBIR023.

NCC_EBIR023
===========

**Error message**: MLP kernel intermediate size exceeds the maximum supported value of 4096.

Consider tiling large intermediate tensors in your kernel to stay within the supported limit, or increase tensor parallelism to shard the intermediate dimension across more cores.
