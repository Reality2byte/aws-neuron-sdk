
.. _pytorch-neuron-install-prev-al2023:

.. Install previous PyTorch Neuron releases for Amazon Linux 2023 - archived

Use the tabs below to install a specific previous Neuron SDK release. Select the Neuron version you need.

.. tab-set::

    .. tab-item:: Neuron 2.21.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=1.13.1 --neuron-version=2.21.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=inf1 --ami=non-dlami

    .. tab-item:: Neuron 2.20.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=1.13.1 --neuron-version=2.20.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=inf1 --ami=non-dlami

    .. tab-item:: Neuron 2.19.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=1.13.1 --neuron-version=2.19.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=inf1 --ami=non-dlami
