.. _nki-isa:

nki.isa
========

.. currentmodule:: nki.isa

The ``nki.isa`` module provides low-level instructions that map directly to the NeuronDevice instruction set architecture. These APIs give you fine-grained control over compute engines, data movement, and memory operations.

Matrix Operations
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - :doc:`nc_matmul <generated/nki.isa.nc_matmul>`
     - Matrix multiplication on the Tensor Engine
   * - :doc:`nc_matmul_mx <generated/nki.isa.nc_matmul_mx>`
     - Matrix multiplication with MX (microscaling) format support
   * - :doc:`nc_transpose <generated/nki.isa.nc_transpose>`
     - Transpose a tile on the Tensor Engine

Activation and Element-wise Operations
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - :doc:`activation <generated/nki.isa.activation>`
     - Apply activation functions (exp, gelu, sigmoid, etc.)
   * - :doc:`activation_reduce <generated/nki.isa.activation_reduce>`
     - Apply activation with reduction
   * - :doc:`exponential <generated/nki.isa.exponential>`
     - Dedicated exponential with max subtraction (Trn3 only)
   * - :doc:`reciprocal <generated/nki.isa.reciprocal>`
     - Compute element-wise reciprocal
   * - :doc:`quantize_mx <generated/nki.isa.quantize_mx>`
     - Quantize tensors to MX (microscaling) format

Tensor Arithmetic
-----------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - :doc:`tensor_tensor <generated/nki.isa.tensor_tensor>`
     - Element-wise operation on two tensors
   * - :doc:`tensor_tensor_scan <generated/nki.isa.tensor_tensor_scan>`
     - Element-wise operation with scan (prefix sum)
   * - :doc:`scalar_tensor_tensor <generated/nki.isa.scalar_tensor_tensor>`
     - Scalar-tensor-tensor fused operation
   * - :doc:`tensor_scalar <generated/nki.isa.tensor_scalar>`
     - Element-wise operation between a tensor and a scalar
   * - :doc:`tensor_scalar_reduce <generated/nki.isa.tensor_scalar_reduce>`
     - Tensor-scalar operation with reduction
   * - :doc:`tensor_scalar_cumulative <generated/nki.isa.tensor_scalar_cumulative>`
     - Tensor-scalar operation with cumulative reduction

Reduction Operations
--------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - :doc:`tensor_reduce <generated/nki.isa.tensor_reduce>`
     - Reduce a tensor along an axis (sum, max, min, etc.)
   * - :doc:`tensor_partition_reduce <generated/nki.isa.tensor_partition_reduce>`
     - Reduce along the partition dimension

Data Movement
-------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - :doc:`tensor_copy <generated/nki.isa.tensor_copy>`
     - Copy a tensor between on-chip buffers
   * - :doc:`tensor_copy_predicated <generated/nki.isa.tensor_copy_predicated>`
     - Conditionally copy tensor elements based on a predicate
   * - :doc:`dma_copy <generated/nki.isa.dma_copy>`
     - DMA transfer between HBM and on-chip memory
   * - :doc:`dma_transpose <generated/nki.isa.dma_transpose>`
     - DMA transfer with transpose
   * - :doc:`dma_compute <generated/nki.isa.dma_compute>`
     - DMA transfer with compute (reduce, scatter)
   * - :doc:`memset <generated/nki.isa.memset>`
     - Set all elements of a tensor to a constant value

Selection and Masking
---------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - :doc:`iota <generated/nki.isa.iota>`
     - Generate index patterns for masking and selection
   * - :doc:`dropout <generated/nki.isa.dropout>`
     - Apply dropout mask to a tensor
   * - :doc:`affine_select <generated/nki.isa.affine_select>`
     - Select elements using an affine index pattern
   * - :doc:`range_select <generated/nki.isa.range_select>`
     - Select elements within a value range
   * - :doc:`select_reduce <generated/nki.isa.select_reduce>`
     - Select elements with max reduction
   * - :doc:`sequence_bounds <generated/nki.isa.sequence_bounds>`
     - Compute sequence bounds from segment IDs

Batch Normalization
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - :doc:`bn_stats <generated/nki.isa.bn_stats>`
     - Compute batch normalization statistics (mean, variance)
   * - :doc:`bn_aggr <generated/nki.isa.bn_aggr>`
     - Aggregate batch normalization statistics

Gather and Shuffle
------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - :doc:`local_gather <generated/nki.isa.local_gather>`
     - Gather elements from a tensor using indices
   * - :doc:`nc_n_gather <generated/nki.isa.nc_n_gather>`
     - Gather with flattened free partition (up to full SBUF)
   * - :doc:`max8 <generated/nki.isa.max8>`
     - Find top-8 maximum values
   * - :doc:`nc_find_index8 <generated/nki.isa.nc_find_index8>`
     - Find indices of top-8 values
   * - :doc:`nc_match_replace8 <generated/nki.isa.nc_match_replace8>`
     - Match and replace up to 8 values
   * - :doc:`nc_stream_shuffle <generated/nki.isa.nc_stream_shuffle>`
     - Shuffle elements across streams

Register Operations
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - :doc:`register_alloc <generated/nki.isa.register_alloc>`
     - Allocate a GpSimd register
   * - :doc:`register_load <generated/nki.isa.register_load>`
     - Load a value into a GpSimd register
   * - :doc:`register_move <generated/nki.isa.register_move>`
     - Move a value between GpSimd registers
   * - :doc:`register_store <generated/nki.isa.register_store>`
     - Store a GpSimd register value to memory

Synchronization and Communication
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - :doc:`core_barrier <generated/nki.isa.core_barrier>`
     - Insert a barrier to synchronize engine execution
   * - :doc:`sendrecv <generated/nki.isa.sendrecv>`
     - Send and receive data between engines

Random Number Generation
------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - :doc:`rng <generated/nki.isa.rng>`
     - Generate random numbers (legacy, NeuronCore v2/v3)
   * - :doc:`rand2 <generated/nki.isa.rand2>`
     - Generate random numbers (NeuronCore v4+)
   * - :doc:`rand_set_state <generated/nki.isa.rand_set_state>`
     - Set the random number generator state
   * - :doc:`rand_get_state <generated/nki.isa.rand_get_state>`
     - Get the current random number generator state
   * - :doc:`set_rng_seed <generated/nki.isa.set_rng_seed>`
     - Set the random number generator seed

Utility
-------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - :doc:`nonzero_with_count <generated/nki.isa.nonzero_with_count>`
     - Return nonzero element indices and their count

Configuration Enums
-------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Enum
     - Description
   * - :doc:`engine <generated/nki.isa.engine>`
     - Compute engine selection (Tensor, Vector, Scalar, GpSimd)
   * - :doc:`dma_engine <generated/nki.isa.dma_engine>`
     - DMA transfer engine selection
   * - :doc:`reduce_cmd <generated/nki.isa.reduce_cmd>`
     - Reduction command for accumulator control
   * - :doc:`dge_mode <generated/nki.isa.dge_mode>`
     - Descriptor Generation Engine mode
   * - :doc:`oob_mode <generated/nki.isa.oob_mode>`
     - Out-of-bounds access handling mode
   * - :doc:`matmul_perf_mode <generated/nki.isa.matmul_perf_mode>`
     - Matrix multiplication performance mode

Target
------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - API
     - Description
   * - :doc:`nc_version <generated/nki.isa.nc_version>`
     - NeuronCore version enum
   * - :doc:`get_nc_version <generated/nki.isa.get_nc_version>`
     - Get the target NeuronCore version

Constants
---------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Constant
     - Description
   * - :doc:`VirtualRegister <generated/nki.isa.VirtualRegister>`
     - Represents a GpSimd virtual register

.. toctree::
   :hidden:
   :maxdepth: 1

   generated/nki.isa.nc_matmul
   generated/nki.isa.nc_matmul_mx
   generated/nki.isa.nc_transpose
   generated/nki.isa.activation
   generated/nki.isa.activation_reduce
   generated/nki.isa.exponential
   generated/nki.isa.reciprocal
   generated/nki.isa.quantize_mx
   generated/nki.isa.tensor_tensor
   generated/nki.isa.tensor_tensor_scan
   generated/nki.isa.scalar_tensor_tensor
   generated/nki.isa.tensor_scalar
   generated/nki.isa.tensor_scalar_reduce
   generated/nki.isa.tensor_scalar_cumulative
   generated/nki.isa.tensor_reduce
   generated/nki.isa.tensor_partition_reduce
   generated/nki.isa.tensor_copy
   generated/nki.isa.tensor_copy_predicated
   generated/nki.isa.dma_copy
   generated/nki.isa.dma_transpose
   generated/nki.isa.dma_compute
   generated/nki.isa.memset
   generated/nki.isa.iota
   generated/nki.isa.dropout
   generated/nki.isa.affine_select
   generated/nki.isa.range_select
   generated/nki.isa.select_reduce
   generated/nki.isa.sequence_bounds
   generated/nki.isa.bn_stats
   generated/nki.isa.bn_aggr
   generated/nki.isa.local_gather
   generated/nki.isa.nc_n_gather
   generated/nki.isa.max8
   generated/nki.isa.nc_find_index8
   generated/nki.isa.nc_match_replace8
   generated/nki.isa.nc_stream_shuffle
   generated/nki.isa.register_alloc
   generated/nki.isa.register_load
   generated/nki.isa.register_move
   generated/nki.isa.register_store
   generated/nki.isa.core_barrier
   generated/nki.isa.sendrecv
   generated/nki.isa.rng
   generated/nki.isa.rand2
   generated/nki.isa.rand_set_state
   generated/nki.isa.rand_get_state
   generated/nki.isa.set_rng_seed
   generated/nki.isa.nonzero_with_count
   generated/nki.isa.engine
   generated/nki.isa.dma_engine
   generated/nki.isa.reduce_cmd
   generated/nki.isa.dge_mode
   generated/nki.isa.oob_mode
   generated/nki.isa.matmul_perf_mode
   generated/nki.isa.nc_version
   generated/nki.isa.get_nc_version
   generated/nki.isa.VirtualRegister
