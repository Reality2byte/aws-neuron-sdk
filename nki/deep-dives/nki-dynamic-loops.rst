.. meta::
   :description: Deep dive into nki.language.dynamic_range for dynamic loop iteration with runtime bounds on AWS Neuron hardware.
   :keywords: NKI, dynamic_range, hardware loop, runtime bounds, VirtualRegister, AWS Neuron, Trainium
   :date-modified: 03/31/2026

.. _nki-dynamic-loops:

==================
NKI Dynamic Loops
==================

This document covers the `dynamic_range` NKI language API and describes how it
can be used to create on-chip (a.k.a. dynamic) loops.

To begin, let's look at the `dynamic_range` function which is defined below.

.. py:function:: nki.language.dynamic_range(start, stop=None, step=1)
   :noindex:

   Create a sequence for **dynamic** loop iteration with runtime bounds.

   :param start: Start value (inclusive), or stop if ``stop`` is ``None``. Can be a ``VirtualRegister``.
   :param stop: Stop value (exclusive). Can be a ``VirtualRegister``.
   :param step: Step size. Must be a compile-time positive ``int`` (not a ``VirtualRegister``).
   :return: An iterator yielding integer values from *start* to *stop*.

The other NKI range iterators (``affine_range``, ``sequential_range``,
``static_range``) all require compile-time constant bounds. However, some
kernels need trip counts determined at execution time on the NeuronCore---for
example, when the number of tiles to process is loaded from a tensor or
computed on device. The ``nl.dynamic_range`` iterator supports this use case.

When the compiler encounters a ``dynamic_range`` loop it emits a **hardware
loop instruction** on the device. The loop body is not unrolled; instead, a
single copy of the body is generated and the hardware iterates over it at
runtime.


.. contents:: On this page
   :local:
   :depth: 2

Parameter Constraints
-----------------------

``start`` / ``stop``
   Can be Python ``int`` literals **or** ``VirtualRegister`` objects (runtime
   values computed on device). When only one positional argument is given it is
   treated as ``stop`` and ``start`` defaults to ``0``, matching the Python
   ``range()`` convention.

``step``
   **Must** be a compile-time positive integer. Passing a ``VirtualRegister``
   raises an ``AssertionError``:
   The step must be known at compile time because the hardware loop instruction
   encodes the step as an immediate operand.

Comparison with Other Range Iterators
---------------------------------------

NKI provides four range iterators. The table below summarises their key
differences:

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 20 30

   * - Iterator
     - Bounds
     - Unrolled?
     - Generated Code
     - Primary Use Case
   * - ``static_range``
     - Compile-time ``int``
     - Yes (at compile time)
     - Fully unrolled---no loop instruction
     - Default choice---supersedes ``sequential_range`` and ``affine_range``.
   * - ``sequential_range``
     - Compile-time ``int``
     - Yes (at compile time)
     - Fully unrolled---no loop instruction
     - Deprecated, formerly for iterations with loop-carried dependencies. Prefer ``static_range`` instead.
   * - ``affine_range``
     - Compile-time ``int``
     - Yes (at compile time)
     - Fully unrolled---no loop instruction
     - Deprecated, formerly for parallel iterations with no loop-carried dependency. Prefer ``static_range`` instead.
   * - ``dynamic_range``
     - Runtime ``VirtualRegister`` or ``int``
     - **No**
     - **Hardware loop instruction**
     - Trip count unknown at compile time

There are three key distinctions worth calling out:

- ``static_range``, ``affine_range``, and ``sequential_range`` require all bounds to be
  compile-time integers. The compiler keeps them as loops internally but may
  unroll them in the backend. ``dynamic_range`` bounds can be
  runtime values and the loop is **never** unrolled.
- ``static_range``, ``affine_range``, and ``sequential_range`` fully unrolls at compile time, which can dramatically increase
  compilation time, ``dynamic_range`` avoids this entirely.

Hardware Lowering
-------------------

The compiler lowers ``dynamic_range`` loops to hardware loop instructions on
the NeuronCore. Because the loop exists as a single hardware instruction with a body:

- The compiled artifact size does **not** grow with the trip count.
- The loop variable is a device register, not a Python ``int``. You cannot use
  it in host-side Python expressions (e.g., ``if i == 0:``). Use NKI
  device-side operations for any conditional logic that depends on the loop
  variable.

Register Allocation Implications
----------------------------------

Inside a ``dynamic_range`` loop the compiler must keep all live tensors in
on-chip memory (SBUF/PSUM) for the **entire duration** of the loop, because
the hardware re-executes the same body on each iteration. This means:

- Tensors allocated inside the loop body are allocated once and reused across
  iterations.
- Keeping the loop body small and limiting the number of live tiles reduces
  memory pressure.

In contrast, ``static_range`` unrolls each iteration independently, giving the
compiler full freedom to schedule instructions across the flattened instruction
stream. However, this does not solve the issue when the trip count is unknown
at compile time---which is precisely when ``dynamic_range`` is needed.

Interaction with ``no_reorder``
---------------------------------

``dynamic_range`` loops are **not supported** inside a ``nl.no_reorder()``
block. The ``no_reorder`` directive forces strict program-order execution,
which requires the compiler to fully unroll the block---and that conflicts with
the hardware loop mechanism.

.. code-block:: python

   # ✗ This is NOT supported and will error
   with nl.no_reorder():
       for i in nl.dynamic_range(n):
           ...

``affine_range``, ``sequential_range``, and ``static_range`` are all permitted
inside ``no_reorder`` blocks because they are resolved or managed at compile
time.

There are two ways to work around this. Either move the ``dynamic_range`` loop
outside the ``no_reorder`` block, or place the ``no_reorder`` block inside the
loop body:

.. code-block:: python

   # ✓ no_reorder inside the dynamic loop body
   for i in nl.dynamic_range(n):
       with nl.no_reorder():
           ...

Using ``while`` with a ``VirtualRegister``
--------------------------------------------

As an alternative to ``dynamic_range``, you can use a standard ``while`` loop
with a ``VirtualRegister`` as the condition. The loop terminates when the
register holds the value ``0``.

.. code-block:: python

   import nki.language as nl
   import nki.isa as nisa

   reg = nisa.register_alloc(1)
   while reg:
       # perform work ...

       # update condition from an SBUF tensor
       nisa.register_load(reg, cond_tensor)


When to Use ``dynamic_range``
-------------------------------

Use ``dynamic_range`` when:

- The number of iterations is **not known at compile time**---for example, it
  depends on a value loaded from a tensor or computed on device.
- The trip count is **large** and unrolling (``static_range``, ``affine_range``, or ``sequential_range``) would cause
  excessive compilation time or code size.

Prefer other iterators when:

- Bounds are compile-time constants and iterations are independent, contain loop-carried dependencies, or need full unrolling → 
  ``static_range``, ``affine_range``, or ``sequential_range``.

Examples
----------

Basic usage with a constant bound
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import nki.language as nl

   for _ in nl.dynamic_range(1):
       tile = nl.load(input_tensor[0:128, 0:512])
       result = nl.multiply(tile, tile)
       nl.store(out_tensor[0:128, 0:512], result)

Even with a constant bound, this generates a hardware loop instruction rather than unrolling.

Runtime trip count from a VirtualRegister
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import nki.language as nl
   import nki.isa as nisa

   num_tiles = nisa.register_alloc(4)
   for i in nl.dynamic_range(num_tiles):
       tile = nl.load(input_tensor[i * 128:(i + 1) * 128, 0:512])
       result = nl.multiply(tile, 2.0)
       nl.store(out_tensor[i * 128:(i + 1) * 128, 0:512], result)


Specifying start, stop, and step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import nki.language as nl

   # Loop from `begin` to `end` with step 2
   # begin and end are VirtualRegisters; step must be a compile-time int
   begin = nisa.register_alloc(0)
   end = nisa.register_alloc(4)
   for i in nl.dynamic_range(begin, end, step=2):
       ...
