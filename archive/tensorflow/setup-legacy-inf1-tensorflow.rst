.. meta::
   :description: Legacy TensorFlow installation guide for AWS Inferentia 1 (Inf1) instances
   :keywords: tensorflow, neuron, inf1, legacy, installation, tensorflow-neuron
   :framework: tensorflow
   :instance-types: inf1
   :status: legacy
   :content-type: legacy-guide
   :date-modified: 2026-03-30

TensorFlow on Inf1 (legacy)
=============================

.. warning::
   
   **Legacy hardware**: Inf1 instances use NeuronCore v1 with TensorFlow 2.x (``tensorflow-neuron``).
   
   For new projects, use **Inf2, Trn1, Trn2, or Trn3** with PyTorch 2.9+ or JAX 0.7+.
   See :ref:`setup-guide-index` for current setup options.

.. note::
   
   TensorFlow support for Inf2 has reached end of support as of Neuron SDK 2.29.
   See :ref:`announce-eos-tensorflow-inf2` for details.

Setup instructions
------------------

For complete Inf1 TensorFlow setup instructions, see the original setup guides:

- :doc:`/archive/tensorflow/tensorflow-neuron/setup/tensorflow-update` - TensorFlow Neuron setup and updates
- :doc:`/archive/tensorflow/tensorflow-neuron-inference` - Inference on Inf1

The setup guides cover:

- Ubuntu 20, Ubuntu 22, and Amazon Linux 2 installation
- DLAMI-based installation
- Manual pip installation
- TensorFlow 2.10.1, 2.9.3, and 2.8.4 versions

Verification
------------

After installation, verify with:

.. code-block:: python
   
   import tensorflow as tf
   import tensorflow_neuron
   
   print(f"TensorFlow version: {tf.__version__}")

.. code-block:: bash
   
   neuron-ls

Next steps
----------

- :doc:`/archive/tensorflow/tensorflow-neuron-inference` - Inference tutorials for Inf1
- :ref:`setup-guide-index` - Current setup options (Inf2, Trn1, Trn2, Trn3)
