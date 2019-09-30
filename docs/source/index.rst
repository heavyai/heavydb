.. OmniSciDB documentation master file, created by
   sphinx-quickstart on Sun Feb 10 09:03:31 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==================================
OmniSciDB at 30,000 feet
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



High Level Diagram
==================

.. image:: ./img/platform_overview.png

Query Execution
==========================
The `Query Execution` section provides a high-level overview
of how a query is executed by the OmniSci server.

At a high-level, all SQL queries made to the server pass through the
Thrift_ `sql_execute` endpoint. The query string is passed to Apache Calcite_ 
for parsing and cost-based optimization, yielding an optimized relational 
algebra tree. This relational algebra tree is then passed through OmniSci-specific 
optimization passes and translated into an OmniSCi-specific abstract syntax tree (AST). 
The AST provides all the information necessary to generate native machine code for 
query execution on the target device. Execution then occurs in parallel on the target 
device, with device results being aggregated and reduced into a final `ResultSet`
for each query step.

The sections following provide in-depth details on each of the
stages involved in executing a query.

.. _Thrift: https://thrift.apache.org/
.. _Calcite: https://calcite.apache.org/
.. _Bison: https://www.gnu.org/software/bison/

Simple Execution Model
======================

.. uml::
   :align: center

    @startuml
   
    start
   
    :Parse and Validate SQL;
   
    :Generate Optimized 
     Relational Algebra Sequence;
   
    :Prepare Execution Environment;
    
    repeat
        fork
            :Data Ownership, 
             Identification, 
             Load (as required);
            :Execute Query Kernel 
             on Target Device;
        fork again
            :Data Ownership, 
             Identification, 
             Load (as required);
            :Execute Query Kernel 
             on Target Device;
        fork again
            :Data Ownership, 
             Identification, 
             Load (as required);
            :Execute Query Kernel 
             on Target Device;
        end fork      
        :Reduce Result;

    while (Query Completed?)

    :Return Result;
    
    stop

    @enduml


.. toctree::
    :caption: System Architecture
    :maxdepth: 2
    :numbered:
    :glob:
    
    catalog/index

    data_model/index

    flow/data.rst

    calcite/*

    execution/index

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
