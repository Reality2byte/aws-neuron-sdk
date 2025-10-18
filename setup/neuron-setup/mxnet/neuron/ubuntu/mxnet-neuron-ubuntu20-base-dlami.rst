.. _setup-mxnet-neuron-u20-base-dlami:

.. card:: Select a Different Framework or Platform for Setup
    :link: setup-guide-index
    :link-type: ref
    :class-body: sphinx-design-class-title-small


MXNet Neuron ("mxnet-neuron") Setup on Ubuntu 20
================================================


.. contents:: Table of contents
	:local:
	:depth: 2


Get Started with Latest Release of MXNet Neuron (``mxnet-neuron``)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section provide links that will assist you to quickly start with a fresh installation of :ref:`install-neuron-mxnet`.


.. dropdown::  Launch the Instance
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    * Please follow the instructions at `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_ to launch an instance. When choosing the instance type at the EC2 console. please make sure to select the correct instance type.
    * To get more information about instance sizes and pricing see: `Inf1 web page <https://aws.amazon.com/ec2/instance-types/inf1/>`_
    * Check for the latest version of the `DLAMI Base AMI <https://aws.amazon.com/releasenotes/aws-deep-learning-ami-base-neuron-ubuntu-20-04/>`_ and copy the AMI name that starts with "Deep Learning Base Neuron AMI (Ubuntu 20.04) <latest_date>" from "AMI Name:" section
    * Search for the copied AMI name in the AMI Search , you should see a matching AMI with the AMI name in Community AMIs. Select the AMI and use it to launch the instance.
    * After launching the instance, follow the instructions in `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_ to connect to the instance

.. dropdown::  Install Drivers and Tools
    :class-title: sphinx-design-class-title-small
    :class-body: sphinx-design-class-body-small
    :animate: fade-in

    .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --framework=pytorch --framework-version=1.13.1 --file=src/helperscripts/n2-manifest.json --os=ubuntu20 --instance=inf1 --ami=non-dlami --category=driver_runtime_tools

.. include:: /about-neuron/quick-start/tab-inference-mxnet-neuron-u20.txt

.. include:: /frameworks/mxnet-neuron/setup/mxnet-update-u20.rst

.. include:: /frameworks/mxnet-neuron/setup/mxnet-install-prev-u20.rst