.. OmniSciDB Data Model

==================================
Memory  Layout
==================================

The management of CPU memory is performed by code in the DataMgr portion of the repository.  To minimise requests for memory allocation the data manager requests CPU memory in 4 GByte blocks or 'slabs' (not GPU memory is allocated in 2 GByte blocks).  The allocation of the memory once the slab is allocated is managed by the Data Manager.  Internally 'slabs' are divided into 512byte pages and allocations within a 'slab' are aligned to 512byte boundaries.

The data manager is designed to load a list of chunks (pages within files) specified by the `Query Engine` into memory.  In preparing the list of chunks to load the `Query Engine` performs  optimizations such fragment skipping.  Chunks of data must traverse the memory hierarchy from disk to cpu memory to gpu memory.  The database system is designed to minimise data conversion through this traversal.

Meta data including access counters are kept about slabs to provide a mechanism for release/'swapped' memory when the system is heavily loaded.  It also possible to 'pin' a memory slab into memory, preventing it from being evicted.

Via `omnisql` command `\memory_summary` the database provides methods to examine both CPU and GPU memory.

