
.. _pytorch-neuronx-ubuntu22-update:

.. Update PyTorch NeuronX to the latest release on Ubuntu 22.04

If you already have a previous Neuron release installed, select the PyTorch version tab below to get the update commands for your environment.


.. tab-set::

    .. tab-item:: PyTorch 2.9.0

        .. include:: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

        .. include:: /src/helperscripts/installationScripts/python_instructions.txt
            :start-line: 284
            :end-line: 285

    .. tab-item:: PyTorch 2.8.0

        .. include:: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

        .. note::
            PyTorch versions 2.7 and 2.8 are no longer supported on Neuron. If you are looking for setup instructions specific to PyTorch 2.7 and 2.8 on Amazon Linux 2023, Ubuntu 24.04, or Ubuntu 22.04, see `the Neuron release 2.28.0 version of the setup docs <https://awsdocs-neuron.readthedocs-hosted.com/en/v2.28.0/setup/neuron-setup/pytorch/neuronx/ubuntu/torch-neuronx-ubuntu22.html#setup-torch-neuronx-ubuntu22>`__.
