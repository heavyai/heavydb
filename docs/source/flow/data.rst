.. OmniSciDB Query Execution

==================================
Data Flow
==================================

A SQL query is essentially a series of transformations on some input data. While the input to the query engine is a SQL query, the query engine (and more so, the ``Executor``) can be thought of a black box which builds a transformation, loads data, applies the transformation, and returns the result. In the following section, we provide a summary of how data flows through the system, from physical storage on disk through the memory hierarchy to the output.

Input Data
==========

All requests for input data start from the ``Executor``. The executor loads chunks required for a query to device memory by making a call into the standalone ``ColumnFetcher`` class. The ``ColumnFetcher`` makes calls directly into the buffer manager hierarchy and storage layer to ensure requested chunks are available in the appropriate memory level for the device. 

The following schematic illustrates the process for requesting inputs for a query step, from storage through to the execution device (in this example, a GPU). 

.. uml::
    :align: center

    @startuml
    "Executor::fetchChunks" -> "Column Fetcher": Request Chunks for Query

    alt input is physical table

    "Column Fetcher" -> Chunk: Build Input Chunk

    Chunk -> "Data Mgr": Request Chunk Buffer 

    "Data Mgr" -> "GPU Buffer Mgr": Request Chunk Buffer 

    "GPU Buffer Mgr" -> "CPU Buffer Mgr": Request Chunk Buffer

    "CPU Buffer Mgr" -> "File Mgr": Request Chunk Buffer

    else input is intermediate result

    "Column Fetcher" -> "Data Mgr": Convert Temporary Buffer to Input and Load to GPU

    end 
    
    @enduml

For a **physical table**, input data is loaded from the storage layer (see :doc:`../data_model/physical_layout` and :doc:`../data_model/columnar_layout`) via the buffer manager hierarchy (see :doc:`../data_model/memory_layout`). If the data is already present on GPU, the request terminates with the `GPU Buffer Mgr`. If not, the request passes on to the parent manager until it reaches storage. 

For **intermediate results**, input data is loaded directly from a per-query temporary tables map and transferred to the GPU directly via the `Data Mgr`.

.. TODO: Add information about temporary tables implementation once it is available for public access.

Output Data
===========

Query step outputs are managed by the ``ResultSet`` (see :doc:`../execution/results`). However, some intermediate buffers may still be required after a query completes. The ``RowSetMemoryOwner`` is a helper structure for managing the lifetime of all outputs related to a given query. The ``ResultSet`` holds a shared pointer to the ``RowSetMemoryOwner``, ensuring that the data held by a ``ResultSet`` is always valid until the ``ResultSet`` class is destroyed. 
