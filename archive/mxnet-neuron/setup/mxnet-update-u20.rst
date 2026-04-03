
.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-03-11

.. mxnet-neuron-u20-update:

Update to latest MXNet Neuron  (``mxnet-neuron``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   This document is archived. MXNet is no longer officially supported
   by the AWS Neuron SDK. It is provided for reference only. For current
   framework support, see :doc:`/frameworks/index`.


If you already have a previous Neuron release installed, this section provide links that will assist you to update to latest Neuron release.


.. tab-set::

    .. tab-item:: MXNet 1.8.0

        .. include:: /setup/install-templates/inf1/note-setup-general.rst

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=mxnet --framework-version=1.8.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami


    .. tab-item:: MXNet 1.5.1

        .. include:: /setup/install-templates/inf1/note-setup-general.rst

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=compiler_framework --framework=mxnet --framework-version=1.5.1 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami
