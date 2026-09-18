.. HeavyDB Table Functions

Table Functions
===============

HeavyDB supports **user-defined table functions (UDTFs)** that appear in the ``FROM``
clause and produce row sets like tables. All registered table functions—including
**system table functions** shipped with the server—use the same ``TableFunction`` /
``TableFunctionsFactory`` type; “system TF” denotes load-time registration, not a
separate SQL object class.

Registration and Build Flags
----------------------------

System table functions are registered at startup. The body of
``TableFunctionsFactory::init()`` is **generated** by
``QueryEngine/scripts/generate_TableFunctionsFactory_init.py`` into
``TableFunctionsFactory_init.cpp``. CMake options such as ``ENABLE_SYSTEM_TFS``,
``ENABLE_ML_ONEDAL_TFS``, and related flags control which implementations are compiled
and linked.

Runtime table functions can also be registered from extension modules; see
``Executor::register_runtime_table_functions`` in ``QueryEngine/Execute.h``.

Server flags (defaults in ``QueryEngine/Execute.cpp``):

* ``--enable-table-functions`` (default **on**)
* ``--enable-ml-functions`` (default **on**; gates ML-related TFs separately)
* ``--enable-dev-table-functions`` (default **off**) — exposes non-whitelisted load-time TFs

Use **SHOW TABLE FUNCTIONS** / **SHOW TABLE FUNCTION DETAILS** for introspection
(``ShowTableFunctionsCommand`` in ``Catalog/DdlCommandExecutor.cpp``).

Function Naming
---------------

Each ``TableFunction`` record maps a C++ implementation name to a SQL-visible name.
Implementation names follow the pattern:

.. code-block:: text

   <sql_name>__<optional_cpu_or_gpu>_<overload_suffix>

The portion before ``__`` is the name used in SQL. Multiple overloads may differ by
argument types or by ``cpu_`` / ``gpu_`` execution context.

Output buffer sizing is controlled by a **sizer** parameter (row multiplier, constant
size, or table-function-specified resize). See ``TableFunctionsFactory.h`` for the full
``TableFunctionOutputBufferSizeType`` enumeration.

Query Planning and Execution
----------------------------

Table functions enter the relational plan as ``RelTableFunction`` nodes (documented in
:doc:`../execution/optimizer`). Execution path:

1. Calcite parses ``SELECT ... FROM table_function(...)`` into relational algebra
2. HeavyDB-specific optimization produces ``RelTableFunction`` in the DAG
3. ``RelAlgExecutor::executeTableFunction`` dispatches to ``Executor::executeTableFunction``
4. ``bind_table_function()`` selects a CPU or GPU implementation; GPU miss may fall back
   to CPU (``QueryMustRunOnCpu`` in ``RelAlgExecutor::createTableFunctionWorkUnit``)
5. ``Executor::executeTableFunction`` compiles and runs the chosen implementation
   (optional preflight on CPU; dynamic output sizing may force CPU)

Table functions are whitelisted for Calcite alongside scalar extension functions; see
``ExtensionFunctionsWhitelist.cpp``.

System Table Functions
----------------------

Beyond user extensions, HeavyDB ships many **system table functions** for introspection
and utilities (memory stats, storage stats, logs, ML helpers, geospatial helpers, and
more). These share the same factory and execution path as UDTFs. Internal foreign-table
wrappers (``INTERNAL_*`` data wrapper types) expose some system information through FSI;
see :doc:`../foreign_storage/index`.

Key Source Paths
----------------

* ``QueryEngine/TableFunctions/TableFunctionsFactory.{h,cpp}``
* ``QueryEngine/RelAlgExecutor.cpp`` (table-function execution)
* ``QueryEngine/Execute.cpp`` / ``Execute.h``
* ``QueryEngine/ExtensionFunctionsWhitelist.cpp``
* ``CMakeLists.txt`` — ``ENABLE_SYSTEM_TFS*`` options
