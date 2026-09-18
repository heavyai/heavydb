.. HeavyDB Configuration

Server Configuration Overview
=============================

HeavyDB is configured primarily through **command-line flags** parsed in
``ThriftHandler/CommandLineOptions.cpp``. An optional **configuration file** (``--config``)
supplies the same options in INI form; values from the file are merged after CLI parsing.

Configuration File
------------------

The sample unit file ``systemd/heavy.conf.in`` shows common defaults:

.. code-block:: ini

   port = 6274
   http-port = 6278
   calcite-port = 6279
   data = @HEAVYAI_STORAGE@
   null-div-by-zero = true

   [web]
   port = 6273
   frontend = @HEAVYAI_PATH@/frontend

The ``[web]`` section configures the optional web/Immerse frontend and is stripped before
Boost program-options validates the main server section (``sanitize_config_file`` in
``CommandLineOptions.cpp``).

Default Network Ports
---------------------

.. list-table::
   :header-rows: 1

   * - Service
     - Default port
   * - Thrift TCP (binary)
     - 6274
   * - HTTP/JSON Thrift
     - 6278
   * - Calcite Java parser
     - 6279
   * - Web frontend (``[web]`` section)
     - 6273

Additional listeners (for example HTTP-binary Thrift on **6276**, ``--http-binary-port``)
are documented in :doc:`../data_model/api`.

Startup Sequence
----------------

At a high level (``HeavyDB.cpp`` → ``DBHandler::initialize``):

1. Parse CLI and optional ``--config`` file; validate data directory and lock file
2. Construct ``DataMgr`` and buffer pools; initialize Calcite and extension whitelists
3. ``SysCatalog::init`` loads catalog state (including RLS ``policies`` rows)
4. Optional subsystems: UDF compiler, rendering, foreign-table refresh scheduler
5. Start Thrift listeners (TCP, HTTP JSON, HTTP binary) and run optional warmup queries

The ``[web]`` block is consumed by ``heavy_web_server``, not ``heavydb``. Many tuning
flags appear only under ``--dev-options``; see ``CommandLineOptions.cpp`` for the full
surface area rather than duplicating it here.

Feature and Subsystem Flags
---------------------------

High-impact toggles (defaults shown in source where notable):

.. list-table::
   :header-rows: 1

   * - Area
     - CLI / flag
     - Notes
   * - Foreign storage
     - ``--enable-fsi``
     - ``g_enable_fsi`` default **true**
   * - Disk cache
     - ``--disk-cache-level``, ``--disk-cache-path``, ``--disk-cache-size``
     - Default level ``foreign_tables`` when FSI enabled
   * - Resource manager
     - ``--enable-executor-resource-mgr``
     - Default **on**
   * - Buffer pools
     - ``--cpu-buffer-mem-bytes``, ``--gpu-buffer-mem-bytes``, slab min/max/default
     - See :doc:`../runtime/index`
   * - Foreign refresh
     - (internal scheduler)
     - ``g_enable_foreign_table_scheduled_refresh`` default **true**
   * - Read-only / multi-instance
     - ``--read-only``, ``--multi-instance``
     - Mutually exclusive

Build-time CMake options (GEOS, LDAP, SAML, FSI ODBC, system table functions, ML
backends) determine which features exist in a given binary; see :doc:`../quickstart/build`.

Where to Look in Code
---------------------

* ``ThriftHandler/CommandLineOptions.cpp`` — full option definitions and parsing order
* ``Shared/SystemParameters.h`` — default numeric parameters
* ``systemd/heavy.conf.in`` — packaged defaults
* ``HeavyDB.cpp`` — server startup wiring (FSI refresh scheduler, DataMgr, executors)

This developer guide intentionally does not duplicate every CLI flag. For operational
runbooks and installer-specific paths, see repository ``scripts/``, ``docker/``, and
``systemd/`` trees.
