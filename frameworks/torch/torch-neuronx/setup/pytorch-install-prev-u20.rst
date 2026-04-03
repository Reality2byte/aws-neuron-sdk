
.. _pytorch-neuronx-install-prev-u20:

.. Install previous PyTorch NeuronX releases for Ubuntu 20.04

Use the tabs below to install a specific previous Neuron SDK release of PyTorch NeuronX on Ubuntu 20.04. Select the Neuron version you need.

.. tab-set::

    .. tab-item:: Neuron 2.21.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=2.1.2 --neuron-version=2.21.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=trn1 --ami=non-dlami

    .. tab-item:: Neuron 2.20.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=2.1.2 --neuron-version=2.20.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=trn1 --ami=non-dlami

    .. tab-item:: Neuron 2.19.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=2.1.2 --neuron-version=2.19.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=trn1 --ami=non-dlami
