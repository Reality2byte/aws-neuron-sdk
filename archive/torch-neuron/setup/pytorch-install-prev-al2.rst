.. _pytorch-neuron-install-prev-al2:

.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-03-11

Install Previous PyTorch Neuron Releases for Amazon Linux (``torch-neuron``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   This document is archived. torch-neuron (Inf1) is no longer officially supported
   by the AWS Neuron SDK. It is provided for reference only. For current
   framework support, see :doc:`/frameworks/index`.


.. toctree::
   :maxdepth: 1


This section will assist you in installing previous Neuron releases.

.. tab-set::


    .. tab-item:: Neuron 2.18.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=1.13.1 --neuron-version=2.18.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami

    .. tab-item:: Neuron 2.17.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=1.13.1 --neuron-version=2.17.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami

    .. tab-item:: Neuron 2.16.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=1.13.1 --neuron-version=2.16.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=inf1 --ami=non-dlami
