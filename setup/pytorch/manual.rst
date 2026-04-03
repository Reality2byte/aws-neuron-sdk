.. meta::
   :description: Manual installation of PyTorch Neuron on Inf2, Trn1, Trn2, Trn3 instances
   :keywords: pytorch, neuron, manual, installation, pip
   :framework: pytorch
   :installation-method: manual
   :instance-types: inf2, trn1, trn2, trn3
   :os: ubuntu-24.04, ubuntu-22.04, al2023
   :python-versions: 3.10, 3.11, 3.12
   :content-type: installation-guide
   :estimated-time: 15 minutes
   :date-modified: 2026-03-30

Install PyTorch via manual installation
========================================

Install PyTorch with Neuron support on a bare OS AMI or existing system.

⏱️ **Estimated time**: 15 minutes

.. note::
   For a faster setup, consider using the :doc:`DLAMI-based installation <dlami>` instead.

.. include:: /frameworks/torch/torch-neuronx/setup/note-setup-general.rst

----

Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Requirement
     - Details
   * - Instance Type
     - Inf2, Trn1, Trn2, or Trn3
   * - Operating System
     - Ubuntu 24.04, Ubuntu 22.04, or Amazon Linux 2023
   * - Python Version
     - Python 3.10, 3.11, or 3.12
   * - AWS Account
     - With EC2 permissions
   * - SSH Key Pair
     - For instance access

Installation steps
------------------

.. tab-set::

   .. tab-item:: Ubuntu 24.04
      :sync: ubuntu-24-04

      **Step 1: Launch instance**

      * Follow the instructions to `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_.
      * Select Ubuntu Server 24 AMI.
      * For Trn1, adjust your primary EBS volume size to a minimum of 512GB.
      * `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_.

      **Step 2: Install drivers and tools**

      .. include:: /src/helperscripts/installationScripts/python_instructions.txt
          :start-line: 299
          :end-line: 300

      **Step 3: Install EFA** (Trn1/Trn1n/Trn2/Trn3 only)

      .. include:: /src/helperscripts/installationScripts/python_instructions.txt
          :start-line: 290
          :end-line: 293

      **Step 4: Install PyTorch and Neuron packages**

      .. tab-set::

          .. tab-item:: PyTorch 2.9.0

              .. include:: /src/helperscripts/installationScripts/python_instructions.txt
                  :start-line: 296
                  :end-line: 297

          .. tab-item:: PyTorch 2.8.0

              .. include:: /src/helperscripts/installationScripts/python_instructions.txt
                  :start-line: 305
                  :end-line: 306

      **Step 5: Verify installation**

      .. code-block:: bash

         python3 -c "import torch; import torch_neuronx; print(f'PyTorch {torch.__version__}, torch-neuronx {torch_neuronx.__version__}')"
         neuron-ls

      You should see output similar to this (the versions, instance IDs, and details should match your expected ones, not the ones in this example):
      
      **Expected output**:
      
      .. code-block:: text
         
         PyTorch 2.9.0+cpu, torch-neuronx 2.9.0.1.0
         
         +--------+--------+--------+-----------+
         | DEVICE | CORES  | MEMORY | CONNECTED |
         +--------+--------+--------+-----------+
         | 0      | 2      | 32 GB  | Yes       |
         | 1      | 2      | 32 GB  | Yes       |
         +--------+--------+--------+-----------+

   .. tab-item:: Ubuntu 22.04
      :sync: ubuntu-22-04

      **Step 1: Launch instance**

      * Follow the instructions to `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_.
      * Select Ubuntu Server 22 AMI.
      * For Trn1, adjust your primary EBS volume size to a minimum of 512GB.
      * `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_.

      **Step 2: Install drivers and tools**

      .. include:: /src/helperscripts/installationScripts/python_instructions.txt
          :start-line: 242
          :end-line: 243

      **Step 3: Install EFA** (Trn1/Trn1n/Trn2/Trn3 only)

      .. include:: /src/helperscripts/installationScripts/python_instructions.txt
          :start-line: 248
          :end-line: 249

      **Step 4: Install PyTorch and Neuron packages**

      .. tab-set::

          .. tab-item:: PyTorch 2.9.0

              .. include:: /src/helperscripts/installationScripts/python_instructions.txt
                  :start-line: 287
                  :end-line: 288

          .. tab-item:: PyTorch 2.8.0

              .. include:: /src/helperscripts/installationScripts/python_instructions.txt
                  :start-line: 281
                  :end-line: 282

      **Step 5: Verify installation**

      .. code-block:: bash

         python3 -c "import torch; import torch_neuronx; print(f'PyTorch {torch.__version__}, torch-neuronx {torch_neuronx.__version__}')"
         neuron-ls

      You should see output similar to this (the versions, instance IDs, and details should match your expected ones, not the ones in this example):
      
      **Expected output**:
      
      .. code-block:: text
         
         PyTorch 2.9.0+cpu, torch-neuronx 2.9.0.1.0
         
         +--------+--------+--------+-----------+
         | DEVICE | CORES  | MEMORY | CONNECTED |
         +--------+--------+--------+-----------+
         | 0      | 2      | 32 GB  | Yes       |
         | 1      | 2      | 32 GB  | Yes       |
         +--------+--------+--------+-----------+

   .. tab-item:: Amazon Linux 2023
      :sync: al2023

      **Step 1: Launch instance**

      * Follow the instructions to `launch an Amazon EC2 Instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html#ec2-launch-instance>`_.
      * Select Amazon Linux 2023 AMI.
      * For Trn1, adjust your primary EBS volume size to a minimum of 512GB.
      * `Connect to your instance <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html>`_.

      **Step 2: Install drivers and tools**

      .. include:: /src/helperscripts/installationScripts/python_instructions.txt
          :start-line: 239
          :end-line: 240

      **Step 3: Install EFA** (Trn1/Trn1n/Trn2/Trn3 only)

      .. include:: /src/helperscripts/installationScripts/python_instructions.txt
          :start-line: 245
          :end-line: 246

      **Step 4: Install PyTorch and Neuron packages**

      .. tab-set::

          .. tab-item:: PyTorch 2.8.0

              .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=2.8.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=trn1 --ami=non-dlami

          .. tab-item:: PyTorch 2.7.0

              .. program-output:: python3 src/helperscripts/n2-helper.py --install-type=install --category=compiler_framework --framework=pytorch --framework-version=2.7.0 --file=src/helperscripts/n2-manifest.json --os=amazonlinux2023 --instance=trn1 --ami=non-dlami

      .. note::
         PyTorch 2.9 is not yet available on Amazon Linux 2023.
         Use Ubuntu 24.04 for PyTorch 2.9 support.

      **Step 5: Verify installation**

      .. code-block:: bash

         python3 -c "import torch; import torch_neuronx; print(f'PyTorch {torch.__version__}, torch-neuronx {torch_neuronx.__version__}')"
         neuron-ls

      You should see output similar to this (the versions, instance IDs, and details should match your expected ones, not the ones in this example):

      .. code-block::

         PyTorch version: 2.9.1+cu128, torch-neuronx version: 2.9.0.2.13.23887+8e870898
         $ neuron-ls
         instance-type: trn1.2xlarge
         instance-id: i-0bea223b1afb7e159
         +--------+--------+----------+--------+--------------+----------+------+
         | NEURON | NEURON |  NEURON  | NEURON |     PCI      |   CPU    | NUMA |
         | DEVICE | CORES  | CORE IDS | MEMORY |     BDF      | AFFINITY | NODE |
         +--------+--------+----------+--------+--------------+----------+------+
         | 0      | 2      | 0-1      | 32 GB  | 0000:00:1e.0 | 0-7      | -1   |
         +--------+--------+----------+--------+--------------+----------+------+


.. tip:: **vLLM for LLM inference**

   After completing the manual installation, you can add vLLM for inference serving
   using the ``vllm-neuron`` plugin:

   .. code-block:: bash

      git clone https://github.com/vllm-project/vllm-neuron.git
      cd vllm-neuron
      pip install --extra-index-url=https://pip.repos.neuron.amazonaws.com -e .

   Or use the pre-configured vLLM DLC image for a containerized deployment.
   See :doc:`/libraries/nxd-inference/vllm/index` for all deployment options.

Update an existing installation
--------------------------------

To update PyTorch versions or Neuron drivers on an existing manual installation, see
:doc:`update-manual`.

Next steps
----------

- :doc:`/frameworks/torch/training-torch-neuronx` - Training on Trn1/Trn2
- :doc:`/frameworks/torch/inference-torch-neuronx` - Inference on Inf2/Trn1/Trn2
- :doc:`/tools/profiler/neuron-profile-user-guide` - Profile your workloads
- :doc:`/tools/neuron-sys-tools/neuron-top-user-guide` - Monitor system resources

Advanced
--------

- :doc:`/frameworks/torch/torch-neuronx/setup/pytorch-neuronx-install-cxx11` - Build torch-xla from source with CXX11 ABI

Additional resources
--------------------

- :doc:`dlami` - Use pre-configured DLAMI instead
- :doc:`dlc` - Use pre-configured Docker containers
- :doc:`/containers/index` - Container-based deployment
- :doc:`../troubleshooting` - Common issues and solutions
- :doc:`/release-notes/index` - Version compatibility information
