
.. Install previous MXNet Neuron releases for Ubuntu 20.04 - archived

Use the tabs below to install a specific previous Neuron SDK release. Select the Neuron version you need.

.. tab-set::

    .. tab-item:: Neuron 2.20.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=mxnet --framework-version=1.8.0 --neuron-version=2.20.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

    .. tab-item:: Neuron 2.19.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=mxnet --framework-version=1.8.0 --neuron-version=2.19.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

    .. tab-item:: Neuron 2.18.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=mxnet --framework-version=1.8.0 --neuron-version=2.18.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami
