.. _install-neuronx-2.8.0-tensorflow:

.. meta::
   :noindex:
   :nofollow:
   :description: This content is archived and no longer maintained.
   :date-modified: 2026-03-11

Install Tensorflow Neuron (Neuron 2.8.0)
========================================

.. warning::

   This document is archived. TensorFlow is no longer officially supported
   by the AWS Neuron SDK. It is provided for reference only. For current
   framework support, see :doc:`/frameworks/index`.


.. tab-set::

    .. tab-item:: Tensorflow 2.10.0

        .. tab-set::

            .. tab-item:: Amazon Linux 2 AMI

                .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=tensorflow --framework-version=2.10.0 --neuron-version=2.8.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2 --instance=trn1 --ami=non-dlami

            .. tab-item:: Ubuntu 20 AMI

                .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=tensorflow --framework-version=2.10.0 --neuron-version=2.8.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=trn1 --ami=non-dlami
