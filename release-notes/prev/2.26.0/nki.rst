.. _neuron-2-26-0-nki:

.. meta::
   :description: The official release notes for the AWS Neuron Kernel Interface (NKI) component, version 2.26.0. Release date: 9/18/2025.

AWS Neuron SDK 2.26.0: Neuron Kernel Interface (NKI) release notes
===================================================================

**Date of release**:  September 18, 2025

.. contents:: In this release
   :local:
   :depth: 2

* Go back to the :ref:`AWS Neuron 2.26.0 release notes home <neuron-2-26-0-whatsnew>`

Improvements
------------

New nki.language APIs
^^^^^^^^^^^^^^^^^^^^^

* gelu_apprx_sigmoid - Gaussian Error Linear Unit activation function with sigmoid approximation.

Updated nki.language APIs
^^^^^^^^^^^^^^^^^^^^^^^^^

* tile_size.total_available_sbuf_size constant - Added a new field, ``total_available_sbuf_size``, that contains the returned total available SBUF size.

New nki.isa APIs
^^^^^^^^^^^^^^^^

* select_reduce - Selectively copy elements with maximum reduction.
* sequence_bounds - Compute sequence bounds of segment IDs.
* dma_transpose - Enhanced with:

  * ``axes`` parameter to define 4D transpose for supported cases
  * ``dge_mode`` parameter to specify Descriptor Generation Engine (DGE)

* activation - Supports the new ``nl.gelu_apprx_sigmoid`` nki.language operation.

Improvements and fixes
^^^^^^^^^^^^^^^^^^^^^^

* **nki.language.store()** - Supports PSUM buffer with extra additional copy inserted.

Documentation and tutorial updates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* Added documentation and example for dma_transpose API
* Improved simulate_kernel example
* Updated tutorial code to use ``nl.fp32.min`` instead of a magic number

Previous release notes
----------------------

* :ref:`nki_rn`

