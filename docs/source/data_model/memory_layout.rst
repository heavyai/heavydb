.. OmniSciDB Data Model

==================================
Memory  Layout
==================================

Once data is loaded from disk (see :doc:`./physical_layout`), the higher level memory hierarchies in the `BufferMgr` take over. Data must pass between hierarchies; i.e. data must travel from disk to CPU to GPU, and can not be loaded directly from disk into GPU (note that this design directly mimics the way data is loaded to modern GPUs over the PCI-E bus). Additionally, not all data from CPU is required to be loaded to GPU for a given query. 

Data buffers in the in-memory levels of the BufferMgr hierarchy are backed by `slabs` and managed by the `DataMgr`. CPU Memory is allocated in 4GB slabs, while GPU memory allocations are currently limited to 2GB slabs. When a request comes in to allocate memory (e.g. for an input chunk, query parameter, or output buffer) the `DataMgr` will search for the first open space in the slab that can fulfill the rqeuest. If the current slab is full, a new slab is allocated. Note that this means utilities like `htop` or `nvtop` may show OmniSciDB exhausting all available GPU memory, when in reality memory is still available to the DB from the slab allocations. The `omnisql` commands `\memory_summary`, `\memory_gpu` and `\memory_cpu` show information about allocated slabs and current slab occupany. 

Internally, slabs are divided into 512 byte pages. Allocations within a slab are aligned to a 512 byte boundary; therefore, all top-level OmniSciDB allocations are aligned to 512 bytes.

Buffers within a slab contain access counters to track the last access to the buffer and allow for a last-recently used caching mechanism to evict data from a slab if it is no longer needed. Buffers may also be **pinned* to a slab, preventing them from being evicted.
