.. HeavyDB ML Models

Machine Learning Models
=======================

HeavyDB supports in-database **CREATE MODEL** workflows for training and evaluating
models from SQL. Models are held in an in-memory registry; metadata is exposed through
system tables and optional table functions.

In-Memory Model Store
---------------------

Trained model objects live in the global ``g_ml_models`` map
(``QueryEngine/TableFunctions/SystemFunctions/ML/MLModel.h``). This store is **not
persisted** across server restarts—model weights exist only in process memory.

The ``ml_models`` system table (via ``INTERNAL_ML_MODEL_METADATA`` foreign wrapper)
mirrors **metadata** from ``g_ml_models`` at query time; it is not authoritative storage
and is empty after restart until models are recreated.

Supported ``CREATE MODEL ... OF TYPE`` values (``MLModelType.h``) include
``LINEAR_REG``, ``DECISION_TREE_REG``, ``GBT_REG``, ``RANDOM_FOREST_REG``, and ``PCA``.
Training uses ``{TYPE}_FIT`` table functions; batch prediction uses ``ML_PREDICT`` or
``PCA_PROJECT`` in expressions (``MLPredictCodegen.cpp``). **EVALUATE MODEL** applies
to regression types only (not PCA).

DDL and Privileges
------------------

Model DDL is handled in ``Parser/ParserNode.cpp`` and enforces read-write mode:

* **CREATE MODEL** — trains a model from a query (type such as ``LINEAR_REG``)
* **SHOW MODELS** / model detail commands — list registered models
* **EVALUATE MODEL** — score held-out data
* **DROP MODEL** — remove from ``g_ml_models``

Requires ``--enable-table-functions`` and ``--enable-ml-functions``. Optional
``--restrict-ml-model-metadata-to-superusers`` limits SHOW commands to superusers.

``CREATE MODEL`` requires appropriate table privileges on underlying training data
(see ``Tests/DBObjectPrivilegesTest.cpp``).

Execution Integration
---------------------

At query time, ML inference hooks into the codegen path:

* ``QueryEngine/MLPredictCodegen.cpp`` resolves model names through ``g_ml_models``
* ML-related **table functions** under ``QueryEngine/TableFunctions/SystemFunctions/ML/``
  complement DDL (build flags such as ``ENABLE_ML_ONEDAL_TFS`` gate optional backends)

Calcite and the extension whitelist expose ML table functions when compiled in, same as
other system TFs (see :doc:`../table_functions/index`).

Operational Notes
-----------------

* Restarting HeavyDB clears in-memory model weights; re-run **CREATE MODEL** or restore
  from external pipelines as needed
* Model types and supported options are defined in parser and ML subsystem headers under
  ``ML/`` and ``QueryEngine/TableFunctions/SystemFunctions/ML/``
* For end-user SQL syntax, see the external HeavyDB documentation at ``docs.nvidia.com/heavyai``

Key Source Paths
----------------

* ``QueryEngine/TableFunctions/SystemFunctions/ML/MLModel.h`` — ``g_ml_models``
* ``QueryEngine/TableFunctions/SystemFunctions/MLTableFunctions.cpp``
* ``QueryEngine/MLPredictCodegen.cpp``
* ``Parser/ParserNode.cpp`` — CREATE/SHOW/EVALUATE/DROP MODEL
* ``Catalog/DdlCommandExecutor.cpp`` — related privilege paths
