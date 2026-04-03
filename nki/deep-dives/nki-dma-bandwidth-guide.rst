.. meta::
   :description: Guidelines for maximizing DMA bandwidth by using large contiguous payloads in NKI.
   :keywords: NKI, DMA, bandwidth, payload size, AWS Neuron, Trainium
   :date-modified: 03/31/2026

.. _nki-dma-bandwidth-guide:

=====================================================
Guideline to Avoid Under-Utilizing DMA Bandwidth
=====================================================

We have seen a common misconception in kernel reviews: that CRC-based HBM
address hashing removes the need for large contiguous DMA payloads. While this
isn't a problem per se with the hashing itself, the misconception leads to
kernels with small, fragmented transfers that underperform badly. This document
clarifies what CRC hashing actually does, what it does *not* do, and why large
payloads (≥4 KiB) are still required to saturate HBM bandwidth.

.. contents:: On this page
   :local:
   :depth: 2


What CRC-Based Hashing Does
------------------------------

HBM is organized into multiple independent channels and banks. When many DMA
transfers target the same channel, that channel becomes a bottleneck while
others sit idle.

CRC-based address hashing applies a CRC function to the physical address to
determine which HBM channel and bank services each request. This spreads
traffic more uniformly across all channels compared to simple modular
(bit-slice) interleaving, which can create hot-spots when access strides align
with the channel count.

In essence, CRC hashing solves the *channel utilization* problem: higher
effective HBM channel utilization and more consistent bandwidth across diverse
access patterns.


What CRC-Based Hashing Does NOT Do
-------------------------------------

CRC hashing operates on the **address-to-channel mapping**. It has no effect
on the per-transfer payload size seen by the DMA engines.

Importantly, CRC hashing does **not**:

- Reduce the number of DMA packets required for a given transfer.
- Remove the need for large contiguous payloads per DMA operation.
- Eliminate DMA packets-per-second (PPS) bottlenecks caused by small
  transfers.

Put another way: channel utilization and per-engine throughput are independent
concerns. CRC hashing addresses the first; payload size addresses the second.


Why Large Contiguous DMA Payloads Are Still Required
------------------------------------------------------

The fundamental problem is per-packet overhead. Each NeuronCore has 16 DMA
engines, and every DMA transfer incurs descriptor setup, synchronization, and
semaphore-to-start latency (~1300 ns cross-engine). When payloads are small,
the engines spend more time on overhead than on data movement, and the DMA
packet rate---not HBM bandwidth---becomes the limiting factor.

Large contiguous payloads (≥4 KiB per partition) amortize this fixed overhead
and allow each engine to sustain its peak throughput:

=====  ==============  ==============  ==============
 Gen   BW / Engine     Engines / NC    Aggregate BW
=====  ==============  ==============  ==============
TRN1   17 B/ns         16              272 GB/s
TRN2   23 B/ns         16              368 GB/s
TRN3   33 B/ns         16              528 GB/s
=====  ==============  ==============  ==============

With small payloads the engines cannot fill their pipelines, and achieved
bandwidth drops well below these peaks regardless of how well CRC hashing
distributes traffic across channels.


Bandwidth vs. Payload Size
----------------------------

The relationship between DMA payload size and achieved bandwidth follows a
saturation curve:

- **< 256 B per partition:** Severely overhead-bound. Achieved bandwidth is a
  small fraction of peak.
- **256 B -- 2 KiB per partition:** Improving but still below peak. Per-packet
  overhead is a significant fraction of transfer time.
- **≥ 2 KiB per partition (minimum recommended):** Approaches peak bandwidth.
  The kernel efficiency guide recommends at least 2 KiB of contiguous data per
  partition for all data types.
- **≥ 4 KiB per partition (target for full saturation):** Fully amortizes
  per-packet overhead and saturates the DMA engines.

.. list-table:: Minimum free-dimension sizes for 2 KiB per partition
   :header-rows: 1

   * - Data Type
     - Minimum Free Dimension
     - Bytes per Partition
   * - float32
     - 512 elements
     - 2 048
   * - bfloat16 / float16
     - 1 024 elements
     - 2 048
   * - float8
     - 2 048 elements
     - 2 048


Practical Guidance
--------------------

- **Maximize the free dimension** of every DMA tile. Target ≥4 KiB per
  partition for peak throughput.
- **Coalesce transfers.** One large DMA covering multiple logical sub-tiles
  is faster than many small DMAs to adjacent addresses.
- **Do not rely on CRC hashing alone** to solve bandwidth problems caused by
  small or fragmented transfers. Channel utilization and per-engine throughput
  are independent concerns.
- **Use full partitions (P=128).** Fewer partitions means fewer engines
  utilized, compounding the effect of small payloads.
