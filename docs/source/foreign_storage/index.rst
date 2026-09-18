.. HeavyDB Foreign Storage

Foreign Storage Interface (FSI)
===============================

HeavyDB can query data that lives outside native on-disk table files through the
**Foreign Storage Interface (FSI)**. Foreign tables use the same chunk and buffer APIs
as native tables; routing in ``PersistentStorageMgr`` sends foreign-table keys to
``ForeignStorageMgr`` (or ``CachingForeignStorageMgr`` when disk cache is enabled for FSI).

Placement in DataMgr
--------------------

Native tables are served by ``GlobalFileMgr`` (or ``CachingGlobalFileMgr`` when mutable
native disk cache is enabled). Foreign tables are served by:

* ``ForeignStorageMgr`` — on-demand fetch through a per-table ``ForeignDataWrapper``
* ``CachingForeignStorageMgr`` — adds FSI disk cache via ``ForeignStorageCache``

See also :doc:`../data_model/physical_layout` for native file layout and
:doc:`../runtime/index` for the full memory hierarchy.

Catalog Objects
---------------

FSI metadata lives in the system catalog (SQLite) and in-memory catalog structures:

.. list-table::
   :header-rows: 1

   * - SQLite table
     - Purpose
   * - ``omnisci_foreign_servers``
     - Server name, data-wrapper type, owner, options
   * - ``omnisci_foreign_tables``
     - Links a table id to a server; refresh timestamps and table options
   * - ``omnisci_user_mappings``
     - Per-user credentials for a foreign server (encrypted at rest)

Core C++ types: ``foreign_storage::ForeignServer``, ``ForeignTable``, and
``UserMapping`` under ``Catalog/``.

DDL Flow
--------

FSI DDL is gated by ``g_enable_fsi`` (default **true**). Commands are executed through
``Catalog/DdlCommandExecutor.cpp``:

* **CREATE FOREIGN SERVER** — validates wrapper type and server options via
  ``ForeignDataWrapperFactory``
* **CREATE FOREIGN TABLE** — binds columns to a server; sets ``StorageType::FOREIGN_TABLE``
* **CREATE USER MAPPING** — stores encrypted connection options per user
* **ALTER FOREIGN TABLE** / **REFRESH FOREIGN TABLE** — option changes and manual refresh

Privilege checks use the same object model as native DDL (``CREATE_SERVER``,
``CREATE_TABLE``, ``SERVER_USAGE``, and related grants).

Data Wrappers
-------------

Each foreign table has a lazy-created ``ForeignDataWrapper`` in
``ForeignStorageMgr::data_wrapper_map_``. The wrapper contract includes:

* ``populateChunkMetadata`` — discover fragments and encodings
* ``populateChunkBuffers`` — fill required (and optional prefetch) column chunks
* Option validation for server, table, user mapping, and schema

``ForeignDataWrapperFactory`` selects an implementation from the server's
``data_wrapper_type`` string. Built-in types include file wrappers
(``DELIMITED_FILE``, ``PARQUET_FILE``, ``REGEX_PARSED_FILE``, ``RASTER_FILE``),
internal system-table wrappers (``INTERNAL_*``), and optional ``ODBC`` /
S3-related wrappers when compiled in.

Chunk Fetch Lifecycle
---------------------

On first query touch, ``ForeignStorageMgr`` creates the wrapper if needed, then:

1. ``populateChunkMetadata`` builds ``ChunkMetadata`` for the table
2. ``fetchBuffer`` calls ``populateChunkBuffers`` for the requested fragment/column
3. ``ChunkSizeValidator`` and fragmenter metadata are updated from encoder output

One wrapper instance exists per ``{db_id, table_id}`` key.

Disk Cache
----------

When ``--disk-cache-level`` includes foreign tables (CLI default ``foreign_tables`` when
FSI is enabled), ``CachingForeignStorageMgr`` persists chunk data and wrapper state under
``{data_dir}/disk_cache`` via ``CachingFileMgr`` with LRU eviction. Default cache size
limit is **100 GB** (``DiskCacheConfig`` in ``DataMgr/FileMgr/CachingFileMgr.h``).

Refresh
-------

Foreign tables support manual and scheduled refresh (``REFRESH_TIMING_TYPE``,
``REFRESH_UPDATE_TYPE``, interval options on ``ForeignTable``). **REFRESH FOREIGN TABLE**
clears external caches, in-memory chunks, and optionally evicts disk-cache entries before
re-populating metadata. When ``g_enable_foreign_table_scheduled_refresh`` is true (default),
``ForeignTableRefreshScheduler`` polls every **60 seconds** for tables due for refresh.

Key Source Paths
----------------

* ``DataMgr/ForeignStorage/ForeignStorageMgr.{h,cpp}``
* ``DataMgr/ForeignStorage/CachingForeignStorageMgr.{h,cpp}``
* ``DataMgr/ForeignStorage/ForeignDataWrapper.h``
* ``DataMgr/PersistentStorageMgr/PersistentStorageMgr.cpp``
* ``Catalog/ForeignTable.h``, ``Catalog/DdlCommandExecutor.cpp``
* ``ThriftHandler/ForeignTableRefreshScheduler.cpp``
