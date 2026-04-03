
.. _pytorch-neuron-al2023-update:

.. Update PyTorch Neuron (torch-neuron) on Amazon Linux 2023 - archived

If you already have a previous Neuron release installed, select the PyTorch version tab below to get the update commands.

.. tab-set::

    .. tab-item:: PyTorch 1.13.1

        .. include:: /setup/install-templates/inf1/note-setup-general.rst

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=pytorch --framework-version=1.13.1 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=inf1 --ami=non-dlami
