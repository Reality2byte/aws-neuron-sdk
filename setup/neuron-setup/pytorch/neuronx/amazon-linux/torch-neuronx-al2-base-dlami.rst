.. _setup-torch-neuronx-al2-base-dlami:

.. card:: Select a Different Framework or Platform for Setup
    :link: setup-guide-index
    :link-type: ref
    :class-body: sphinx-design-class-title-small


PyTorch Neuron ("torch-neuronx") Setup on Amazon Linux 2 with DLAMI Base
=========================================================================

.. note::
    As of 2.20.0, Neuron Runtime no longer supports AL2. Upgrade to AL2023 following the :ref:`AL2 Migration guide <eos-al2>`

.. contents:: Table of contents
	:local:
	:depth: 2

.. include:: /setup/install-templates/al2-python.rst

Get Started with Latest Release of PyTorch Neuron (``torch-neuronx``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section provide links that will assist you to quickly start with a fresh installation of :ref:`pytorch-neuronx-main` for both Inference and Training.


.. dropdown::  Launch the Instance
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    * Please follow the instructions at `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ to launch an instance. When choosing the instance type at the EC2 console. please make sure to select the correct instance type.
    * To get more information about instance sizes and pricing see: `Trn1 web page <https://aws.amazon.com/ec2/instance-types/trn1/>`_, `Inf2 web page <https://aws.amazon.com/ec2/instance-types/inf2/>`_
    * Check for the latest version of the `DLAMI Base AMI <https://aws.amazon.com/releasenotes/aws-deep-learning-ami-base-neuron-amazon-linux-2/>`_ and copy the AMI name that starts with "Deep Learning Base Neuron AMI (Amazon Linux 2) <latest_date>" from "AMI Name:" section
    * Search for the copied AMI name in the AMI Search , you should see a matching AMI with the AMI name in Community AMIs. Select the AMI and use it to launch the instance.
    * When launching a Trn1, please adjust your primary EBS volume size to a minimum of 512GB.
    * After launching the instance, follow the instructions in `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_ to connect to the instance 

.. dropdown::  Install Drivers and Tools
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    .. include:: /src/helperscripts/installationScripts/python_instructions.txt
        :start-line: 2
        :end-line: 3


.. include:: /about-neuron/quick-start/tab-inference-torch-neuronx-al2.txt

.. include:: /frameworks/torch/torch-neuronx/setup/pytorch-update-al2.rst

.. include:: /frameworks/torch/torch-neuronx/setup/pytorch-install-prev-al2.rst