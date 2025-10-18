.. _setup-torch-neuronx-ubuntu20-dlami-pytorch:

.. card:: Select a Different Framework or Platform for Setup
    :link: setup-guide-index
    :link-type: ref
    :class-body: sphinx-design-class-title-small


PyTorch Neuron ("torch-neuronx") Setup on Ubuntu 20 with DLAMI Pytorch
======================================================================


.. contents:: Table of contents
	:local:
	:depth: 2


Get Started with Latest Release of PyTorch Neuron (``torch-neuronx``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section provide links that will assist you to quickly start with a fresh installation of :ref:`pytorch-neuronx-main` for both Inference and Training.


.. dropdown::  Launch the Instance
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    * Please follow the instructions at `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ to launch an instance. When choosing the instance type at the EC2 console. please make sure to select the correct instance type.
    * To get more information about instances sizes and pricing see: `Trn1 web page <https://aws.amazon.com/ec2/instance-types/trn1/>`_, `Inf2 web page <https://aws.amazon.com/ec2/instance-types/inf2/>`_
    * Check for the latest version of the `DLAMI Neuron Pytorch 1.13 AMI <https://aws.amazon.com/releasenotes/aws-deep-learning-ami-neuron-pytorch-1-13-ubuntu-20-04/>`_ and copy the AMI name that starts with "Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 20.04) <latest_date>" from "AMI Name:" section
    * Search for the copied AMI name in the AMI Search , you should see an exact matching AMI with the AMI name in Community AMIs. Select the AMI and use it to launch the instance.
    * When launching a Trn1, please adjust your primary EBS volume size to a minimum of 512GB.
    * After launching the instance, follow the instructions in `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_ to connect to the instance 

.. dropdown::  Update Neuron Drivers
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small

    .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=update --category=driver_runtime_tools --framework=pytorch --framework-version=1.13.0 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=trn1

.. dropdown::  Get Started With Pytorch DLAMI
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small

    .. include:: /src/helperscripts/installationScripts/python_instructions.txt
            :start-line: 53
            :end-line: 54

.. card:: Visit PyTorch Neuron(``torch-neuronx``) for Inference section
    :link: inference-torch-neuronx
    :link-type: ref
    :class-body: sphinx-design-class-title-small


.. card:: Visit PyTorch Neuron(``torch-neuronx``) for Training section
    :link: training-torch-neuronx
    :link-type: ref
    :class-body: sphinx-design-class-title-small

 
.. include:: /frameworks/torch/torch-neuronx/setup/pytorch-update-u20-dlami.rst

.. include:: /frameworks/torch/torch-neuronx/setup/pytorch-install-prev-u20.rst