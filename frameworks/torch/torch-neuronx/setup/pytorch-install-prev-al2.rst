.. _pytorch-neuronx-install-prev-al2:


.. meta::
   :description: Install previous PyTorch NeuronX releases on Amazon Linux 2
   :keywords: AWS Neuron, PyTorch, Trainium, Inferentia, setup, torch-neuronx, previous releases, Amazon Linux 2, AL2
   :date-modified: 2026-03-30


Install Previous PyTorch Neuron Releases for Amazon Linux (``torch-neuronx``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1



Use the tabs below to install a specific previous Neuron SDK release of PyTorch NeuronX on Amazon Linux 2. Select the Neuron version you need.


.. tab-set::

    .. tab-item:: Neuron 2.18.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=1.13.1 --neuron-version=2.18.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=trn1 --ami=non-dlami

    .. tab-item:: Neuron 2.17.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=1.13.1 --neuron-version=2.17.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=trn1 --ami=non-dlami

    .. tab-item:: Neuron 2.16.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=1.13.1 --neuron-version=2.16.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=trn1 --ami=non-dlami