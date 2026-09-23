.. HeavyDB Runtime

Memory and Execution Runtime
============================

This page summarizes how HeavyDB stages data in memory and admits query work. It
complements :doc:`../data_model/memory_layout` and :doc:`../execution/overview`.

Memory Hierarchy
----------------

``MemoryLevel`` (``DataMgr/MemoryLevel.h``) orders storage as **DISK → CPU → GPU**.
``DataMgr::populateMgrs`` wires buffer managers:

.. code-block:: text

   bufferMgrs_[DISK]   → PersistentStorageMgr
                            ├─ GlobalFileMgr / CachingGlobalFileMgr (native)
                            ├─ ForeignStorageMgr / CachingForeignStorageMgr (FSI)
                            └─ ForeignStorageCache (shared disk cache object)
   bufferMgrs_[CPU]    → CpuBufferMgr (parent = PersistentStorageMgr)
   bufferMgrs_[GPU]    → GpuCudaBufferMgr per device (parent = CpuBufferMgr)

On a CPU or GPU cache miss, ``BufferMgr::fetchBuffer`` delegates to ``parent_mgr_`` until
disk or foreign storage satisfies the read. Foreign routing is described in
:doc:`../foreign_storage/index`.

Slab and Page Defaults
----------------------

Defaults are defined in ``Shared/SystemParameters.h`` and applied in ``DataMgr.cpp``:

.. list-table::
   :header-rows: 1

   * - Parameter
     - Default
   * - ``buffer_page_size``
     - 512 bytes
   * - ``min_cpu_slab_size`` / ``min_gpu_slab_size``
     - 256 MB
   * - ``default_cpu_slab_size`` / ``default_gpu_slab_size``
     - 4 GB
   * - ``max_cpu_slab_size`` / ``max_gpu_slab_size``
     - 4 GB (clamped to pool size)
   * - ``cpu_buffer_mem_bytes`` / ``gpu_buffer_mem_bytes``
     - 0 = auto-size pools

Auto-sizing behavior:

* **CPU pool** — 80% of system RAM (page-aligned) when bytes are unset
* **GPU pool** (per device) — device global memory minus reserved GPU memory (default
  reserved **768 MB**)

Slab sizes are clamped to the corresponding pool via ``get_slab_size()``. CLI flags
``--cpu-buffer-mem-bytes``, ``--gpu-buffer-mem-bytes``, and min/max/default slab options
override these defaults (see :doc:`../configuration/index`).

Disk Cache (Execution Layer)
----------------------------

On-disk cache for foreign and/or mutable native tables uses ``DiskCacheConfig`` and
``CachingFileMgr`` (LRU eviction). Levels: ``foreign_tables`` (FSI only, CLI default when
FSI enabled), ``local_tables``, ``all``, or ``none``. Default path:
``{data_dir}/disk_cache``; default size limit **100 GB**.

This cache is distinct from in-memory CPU/GPU slabs but participates in the same fetch
chain through ``PersistentStorageMgr``.

Executor Resource Manager
-------------------------

``g_enable_executor_resource_mgr`` defaults to **true** (``QueryEngine/Execute.cpp``).
``ExecutorResourceMgr`` queues and grants per-query resources:

* CPU and GPU **compute slots**
* CPU and GPU **result memory**
* CPU and GPU **buffer-pool memory** (pinned vs pageable subtypes)

Pools are initialized in ``DBHandler::init_executor_resource_mgr`` from thread counts,
GPU count, and buffer-pool sizes (with a 0.95 fudge factor on buffer pools). Each query
builds a ``RequestInfo`` in ``Executor::launchKernelsViaResourceMgr``; an
``ExecutorResourceHandle`` releases grants when the query step completes.

When the resource manager is disabled, kernels launch under a global ``kernel_mutex_``
(``launchKernelsLocked``).

Parallel executors: ``SystemParameters::num_executors`` defaults to **4**; each may
hold its own resource-manager instance.

Key Source Paths
----------------

* ``DataMgr/DataMgr.{h,cpp}``, ``DataMgr/BufferMgr/BufferMgr.{h,cpp}``
* ``DataMgr/PersistentStorageMgr/``, ``DataMgr/FileMgr/CachingFileMgr.h``
* ``QueryEngine/ExecutorResourceMgr/``
* ``QueryEngine/Execute.cpp`` — admission and kernel launch
* ``ThriftHandler/DBHandler.cpp`` — ``init_executor_resource_mgr``
