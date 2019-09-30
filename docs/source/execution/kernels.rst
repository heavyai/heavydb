.. OmniSciDB Query Execution

==================
Execution Kernels
==================

Query execution begins with ``Executor::dispatchFragments`` and ends with reducing results from each device and returning a ``ResultSet`` back to the ``RelAlgExecutor``. Each device (GPU or CPU thread) has a dedicated CPU thread. All devices initialize state and execute queries in parallel. On CPU, this means the execution within a single device is not parallel. On GPU, execution within a device also occurs in parallel. 

Input data is assigned to the relevant device in a pre-processing step. Input fragments are typically assigned in round-robin order, unless the input data is sharded. For sharded input data, all shards of the same `key` are assigned to the same device. Input assignment is managed by the ``QueryFragmentDescriptor``.

The execution process consists of the following main steps (each run concurrently per execution device):

1. Fetch all assigned input chunks (assigned input fragments across columns required for the query).
2. Prepare output buffers (allocation and initialization if necessary).
3. Execute the generated code (i.e. launch kernels on the device).
4. Prepare ``ResultSet`` and return (reducing if necessary).

Execution is managed by the ``ExecutionDispatch`` class (a singleton) which manages the execution process. Each device has its own ``QueryExecutionContext``, which owns and manages the state for the duration of the :term:`kernel` execution on the device. 

.. image:: ../img/dispatch_fragments.png

Query Fragment Descriptor
----------------------------------

The basic unit of work in OmniSciDB is a fragment. The ``QueryFragmentDescriptor`` class maintains useful information about fragments that are involved with execution of a particular work unit; most importantly, the fragment descriptor partitions fragments among all available devices. 

Dispatch Fragments
----------------------------------

As discussed above, the ``QueryFragmentDescriptor`` assigns fragments to devices (i.e., kernels). Using this information, the ``Executor`` concurrently dispatches an execution procedure per device. 

All CPU queries are executed in `kernel per fragment` mode, meaning each CPU kernel executes over a single fragment.

GPU queries can execute in either `kernel per fragment` mode or `multi-fragment kernel` mode, where a single GPU kernel executes over multiple input fragments. Since GPU execution supports intra-kernel parallelism, multi-fragment kernels are typically more efficient in GPU execution mode. 

Query Execution Context
----------------------------------

The ``QueryExecutionContext`` object is created for each device and manages the following high level tasks:

1. Prepares for kernel execution (setup output buffers, parameters, etc)
2. Launches the execution kernel on the device
3. Handles errors returned by the kernel (if any)
4. Transfers the output buffers from the device back to the host (for GPU execution)

While the same execution context is created for CPU and GPU execution, the exact procedure for each mode is slightly different. 

CPU execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. uml::
    :align: center

    @startuml
    (*) --> "Initialization"

    --> "CPU Output Buffer Allocation / Initialization"

    if "is Group By / Projection" then
    -->[true] "Execute Plan With Group By"
    --> "Launch CPU Code"
    else
    -> [false] "Execute Plan Without Group By" 
    --> "Launch CPU Code"
    endif

    --> "Reduce inter-device Results"

    -right-> (*)

    @enduml

GPU execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. uml::
    :align: center

    @startuml
    (*) --> "Initialization"

    --> "CPU Output Buffer Allocation / Initialization"
    --> "GPU Output Buffer Allocation / Initialization"

    if "is Group By / Projection" then
    -->[true] "Execute Plan With Group By"
    --> "Launch GPU Code"
    else
    -> [false] "Execute Plan Without Group By" 
    --> "Launch GPU Code"
    endif

    --> "Prepare Kernel Params"
    --> "Launch Cuda Kernel"
    --> "Copy Back Output Buffer"
    --> "Reduce inter-device Results"

    -right-> (*)

    @enduml



.. note::

    Some queries will allocate more than one output buffer on the GPU to reduce thread contention during parallel intra-fragment execution. For each allocated output buffer on the GPU, a match output buffer on CPU is also allocated to support copying results back from the GPU once execution finishes.

All arguments for the GPU kernel must be allocated in GPU memory and copied to the device. The GPU kernel launch function takes a pointer to the GPU generated code (in device memory) and a pointer to the kernel parameters buffer (also in device memory).

Kernel launches on GPU are asynchronous; that is, ``cuLaunchKernel`` returns immediately after the kernel successfully starts on the device. The next call to the nVidia CUDA driver API is blocking. Immediately after the kernel is launched, an attempt is made to copy the error codes buffer back using the CUDA driver API. This call is blocking; therefore, if the kernel generates an error during execution, we will detect it only after the entire kernel finishes. 

