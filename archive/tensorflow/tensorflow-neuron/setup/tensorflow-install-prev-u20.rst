.. _tensorflow-neuron-install-prev-u20:

.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-03-11

Install Previous Tensorflow Neuron Releases for Ubuntu (``tensorflow-neuron``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   This document is archived. TensorFlow is no longer officially supported
   by the AWS Neuron SDK. It is provided for reference only. For current
   framework support, see :doc:`/frameworks/index`.


.. toctree::
   :maxdepth: 1


This section will assist you in installing previous Neuron releases.

.. tab-set::

    .. tab-item:: Neuron 2.21.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=tensorflow --framework-version=2.10.1 --neuron-version=2.21.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami

    .. tab-item:: Neuron 2.20.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=tensorflow --framework-version=2.10.1 --neuron-version=2.20.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami
    
    .. tab-item:: Neuron 2.19.0

        .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=tensorflow --framework-version=2.10.1 --neuron-version=2.19.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami
