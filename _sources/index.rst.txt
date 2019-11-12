.. OmniSciDB documentation master file, created by
   sphinx-quickstart on Sun Feb 10 09:03:31 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================================
OmniSciDB Developer Documentation
==================================

OmniSciDB is an open source SQL-based, relational, columnar database engine
that leverages the full performance and parallelism of modern hardware
(both CPUs and GPUs) to enable querying of multi-billion row datasets
in milliseconds, without the need for indexing, pre-aggregation, or
downsampling.

OmniSciDB can be run on hybrid CPU/GPU systems, as well as on CPU-only systems
featuring X86, Power, and ARM (experimental support) architectures. To achieve
maximum performance, OmniSciDB features multi-tiered caching of data between
storage, CPU memory, and GPU memory, as well as an innovative Just-In-Time
(JIT) query compilation framework built around LLVM.

This developer documentation provides an in-depth discussion of the OmniSciDB
internals, and details the data model and query execution flows.

.. toctree::
    :caption: System Architecture
    :maxdepth: 2
    :numbered:
    :glob:

    overview/index

    quickstart/getting_started
    
    catalog/index

    data_model/index

    flow/data.rst

    calcite/*

    execution/index

.. toctree::
    :caption: API Reference
    :glob:

    api/*

.. toctree::
    :caption: Additional Resources

    Doxygen <https://doxygen.omnisci.com>
    GitHub Repository <https://github.com/omnisci/omniscidb>
    glossary/index

.. toctree::
    :caption: Detailed Class Information
    :maxdepth: 2

    components/logger
    components/query_state



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
