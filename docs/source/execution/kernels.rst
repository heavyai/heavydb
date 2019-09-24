.. OmniSciDB Query Execution

==================================
Execution Kernels
==================================
From the moment the query is compiled and the generated code is available (per execution step),
the execution process is started and it continues until all required results from all devices are 
available on the CPU host. In our execution mode, each available device (GPU/CPU) is represented by a single CPU thread
and that thread is responsible to make sure the execution process for that particular device 
finished successfully and the results were collected.

Overall, the execution process is consisted of the following main steps, each being done concurrently per execution device:

1. Assigning fragments to the device
2. Fetching all required input data (copying them to the device if necessary)
3. Preparing output buffers (allocations and initializations)
4. Executing the generated code (launching kernels)

Item 1 is done by the ``QueryFragmentDescriptor`` class. Item 2 is done in `Executor::dispatchFragments`. 
Items 3 and 4 are initiated by the ``dispatchFragments`` function through ``ExecutionDispatch`` class's member functions, 
but most of the work is actually done by the ``QueryExecutionContext`` class.
There are plenty of optimization passes around these simplified steps, 
which might lead to bypassing some of them, but the current itemized list captures the high level idea.
Next, we will discuss each of these components more thoroughly. 

.. image:: ../img/dispatch_fragments.png

Query Fragment Descriptor
----------------------------------
The basic unit of work in OmniSciDB is a fragment. The main goal of the ``QueryFragmentDescriptor`` class is  
to maintain useful information about fragments that are involved with execution of a particular work unit. 
Among this class's responsibilities is the ability to partition fragments into different groups
so that each is assigned to an available device (or kernel). 

Query Execution Context
----------------------------------
The ``QueryExecutionContext`` object is created for each device and is in charge of the following high level tasks:

1. Preparing the requirements for kernel execution
2. Launching the execution kernel on the device
3. Error handling for the launched kernel 
4. Transferring the output buffers from the device back to the host

Whether this class is created for GPU execution or not, 
there might be a different set of actions required. 
As a result of this and for the sake of clarity, we discuss each case separately.

CPU execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Upon construction of this class, and depending on the query type 
(e.g., whether the query is a group by or not), 
the output buffer is allocated and initialized accordingly (through a unique object of ``QueryMemoryInitializer``).
Different aggregate targets may require different initial values.
After this stage, it is possible to launch the CPU code through ``launchCpuCode``. 
By doing so, the compiled code is executed on the CPU. 
After execution, error codes are readily verified, and then results are ready to be collected. 

GPU execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Similar to CPU execution, upon construction of this object a proper set of output buffers 
are allocated and initialized on the GPU device.
It is possible to allocate more than one output buffer on the GPU. This decision is made based on some low-level 
optimizations and depending on the query's type and related details about the metadata involved. 
Regardless of the number of output buffers, there should be a matching number of buffers allocated on the CPU too, 
so that GPU results can be directly transferred back to the host. 

All arguments for the GPU kernel should also be properly allocated and transferred to the GPU device 
using the ``prepareKernelParams`` function.

Once all required output buffers are ready, the compiled code for the GPU can be launched.
Proper error handling is done immediately after the kernel launch. 
The error code buffer is transferred back to the CPU and gets verified. 
Once a successful execution is verified, all output buffers are directly transferred back 
to the CPU to their corresponding buffers on the CPU.

Dispatch fragments
----------------------------------
As discussed above, the ``QueryFragmentDescriptor`` assigns fragments to devices (i.e., kernels). 
By using this information, the ``Executor`` concurrently dispatches an execution procedure per device 
(to execute a single or multiple fragments).
For CPU execution, this means assigning a CPU thread to each available fragment. 
For GPU execution, this means assigning a CPU thread to each group of fragments per device.
These thread concurrently execute the code on all available resources until all executions 
and data transfers are successfully finished.

The execution procedure mentioned above is done through the ``ExecutionDispatch`` class. 
Upon each usage of its ``run`` member function, 
it proceeds with execution of the assigned fragments on a particular device.
To do so, among many other things, it fetches proper input data to be available on each device, and also creates and owns 
an instance of ``QueryExecutionContext`` per device (which itself handles required output buffer allocations, etc.).