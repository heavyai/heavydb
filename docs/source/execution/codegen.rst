.. HeavyDB Query Execution

==================================
Code Generation
==================================

HeavyDB generates native code using the `LLVM <http://llvm.org>`_ library. Code generation primarily makes calls into the `LLVM IR Builder` to build up LLVM IR according to the provided AST (see :doc:`./scheduler`). 

Code generation for a query is managed by the ``Executor`` object assigned to the query. Code generation state is stored within the ``QueryCompilationDescriptor``. The compilation descriptor also initiates code generation via its ``compile`` method. 

The outer-most function for a :term:`kernel` is pre-defined in ``RuntimeFunctions.cpp``. The generated code typically consists of two functions; the ``query_func`` and the ``row_func``. The ``query_func`` loops over all input rows, while the ``row_func`` contains most of the logic for processing inputs, running expressions, and writing outputs. 

Query Templates
===============

The ``query_func`` is built and then bound to a query function template depending on query execution dispatch mode (see :doc:`./kernels` for more on kernel execution modes). Query templates, specified in ``RuntimeFunctions.cpp``, provide the top-level kernel function declaration and basic logic (i.e. iterate over input fragments for a `multi-fragment kernel`). The ``query_func`` is bound to the ``query_stub`` function, replacing its default implementation in the first step of code generation. 

``CodeGenerator``
=================

The ``CodeGenerator`` converts analyzer expressions into `LLVM IR`. The ``CodeGenerator`` constructor takes the code generation state and query plan state from the ``Executor`` and makes calls into the `LLVM IR Builder`, using the context and module attached to the code generation state. 

The ``ScalarCodeGenerator`` is derived from the ``CodeGenerator`` for unit
testing. It omits full query-plan state but still uses an ``Executor`` for
device initialization when generating native CPU or GPU code for analyzer
expressions.


Runtime Functions and Extension Functions
=========================================

All query kernels are generated from LLVM IR using the ``CodeGenerator``. However, for certain helper functions it is more convenient to implement required functionality in C. For example, the aggregate function for computing `SUM` of a not null column requires null sentinel awareness to ensure `null` values of the argument type are not accidentally added to the running value. The 32-bit integer implementation of this function is shown below:

.. code-block:: c

    extern "C" ALWAYS_INLINE int32_t agg_sum_int32_skip_val(int32_t* agg,
                                                            const int32_t val,
                                                            const int32_t skip_val) {
        const auto old = *agg;
        if (val != skip_val) {
            if (old != skip_val) {
            return agg_sum_int32(agg, val);
            } else {
            *agg = val;
            }
        }
        return old;
    }


System functions like the example above are referred to as `Runtime Functions`.
SQL-callable functions are `Extension Functions`. Their declarations are
organized across ``QueryEngine/ExtensionFunctions.hpp`` and the headers it
includes, such as ``ExtensionFunctionsArray.hpp``, ``ExtensionFunctionsGeo.hpp``,
``ExtensionFunctionsH3.hpp``, and ``ExtensionFunctionsText.hpp``. Calcite exposes
the extension-function whitelist at startup; HeavyDB loads it into the query
engine. Runtime functions are helper functions used by the :term:`kernel`.

The `ALWAYS_INLINE` decorator tells the LLVM compiler to inline the runtime functions, so the net effect is the same as generating the function using the LLVM IR Builder. 

LLVM Optimization Passes
========================

Optimization of generated code is primarily managed by the LLVM Pass Manager. The ``optimize_ir`` free function runs multiple LLVM passes over the IR (e.g. `instruction combining`, `instruction simplification`, etc). By using the pass manager, HeavyDB can chose LLVM passes which will maximize query performance based on both query parameters and the target device for which code is being compiled. 

HeavyDB also marks uncalled runtime and extension functions with internal linkage
so they can be pruned from the module before JIT compilation.


Native Codegen
==============

Once the complete set of `LLVM IR` has been assembled for a query, generation of machine code can begin. HeavyDB refers to machine code as `native code` (i.e. native machine code for a particular device). Below we describe `CPU Native Code Generation` and `GPU Native Code Generation`.

CPU Native Code Generation
--------------------------

CPU code generation uses the `LLVM MCJIT <https://llvm.org/docs/MCJITDesignAndImplementation.html>`_ to generate native code for the CPU. The code generator performs the following steps when generating native CPU code: 

1. Optimizes input IR using the techniques described above.
2. Initializes the LLVM MCJIT Backend.
3. Takes ownership of the LLVM Module.
4. Creates an ``llvm::EngineBuilder`` wrapped in an ``ExecutionEngineWrapper``
   inside a ``CpuCompilationContext``.
5. Generates native code by calling ``finalizeObject()`` on the underlying
   ``llvm::ExecutionEngine``.

Native code generated for CPU can be called by getting the function pointer from the execution engine and calling the function.

GPU Native Code Generation
--------------------------

GPU code generation uses LLVM to generate `nVidia PTX <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html>`_ and then converts the PTX to machine code using the nVidia CUDA driver API. The following intermediate steps are performed during this process:

1. Updates LLVM Module target details to target nVidia GPU 
2. Optimizes input IR using the techniques described above.
3. Generates PTX using the LLVM Pass Manager.
4. Converts the PTX to a `cubin` binary machine code file using the nVidia CUDA driver API.
5. Copies the `cubin` binary to the relevant GPUs (typically all available GPUs)
6. Stores a function pointer to the copied binary in GPU memory, to be passed to the nVidia CUDA driver API for kernel launch.

Code Cache
----------

Both CPU and GPU generated code is cached in a code cache per query. The cache
uses an LRU eviction mechanism to ensure large numbers of queries do not fill up
CPU or GPU memory. The cache key combines serialized LLVM representations of
``query_func`` and ``row_func``, and may also include the executor address and
filter function when present.


Troubleshooting
===============

Log Files
---------

Sometimes it can be invaluable to see the text IR, PTX, and/or assembly code generated by the JIT compiler in HeavyDB.

The SQL commands ``EXPLAIN`` and ``EXPLAIN OPTIMIZED`` are available to show LLVM IR code for a query.

The logging system can produce log files ending in ``.IR``, ``.PTX``, and ``.ASM`` when enabled with the ``--log-channels`` option. See also: :doc:`../components/logger`

Automatic LLVM IR Metadata
--------------------------

Finding the C++ code that generated each line of LLVM IR can be hard. Take this simple example of LLVM IR code:

``br label %singleton_true_``

Code generation spans many C++ files. Consecutive lines of IR can be generated
by different source files. Where did this branch instruction come from?

LLVM provides instruction metadata that can connect IR instructions to the C++
code that generated them. In debug builds, HeavyDB automatically adds source
metadata to generated instructions:

``br label %singleton_true_, !IRCodegen.cpp !22``

``!22 = !{!"HEAVY.AI Debugging Info: compileWorkUnit near NativeCodegen.cpp line #2908, codegenJoinLoops near IRCodegen.cpp line #1222"}``

A "footnote" at the bottom shows more location detail if needed, the ``!22``. The detailed footnote contains an approximation of the C++ call stack at the time the line of IR code was generated.

A more complex example:

.. code-block:: llvm

  %16 = call i32 @row_func(...), !NativeCodegen.cpp !20
  %17 = call i32 @record_error_code(i32 %16, i32* %error_code), !NativeCodegen.cpp !21

  !20 = !{!"HEAVY.AI Debugging Info: compileWorkUnit near NativeCodegen.cpp line #2908"}
  !21 = !{!"HEAVY.AI Debugging Info: compileWorkUnit near NativeCodegen.cpp line #2908, codegenJoinLoops near IRCodegen.cpp line #1222, codegen near LoopControlFlow/JoinLoop.cpp, createErrorCheckControlFlow near NativeCodegen.cpp line #2129"}

LLVM encodes metadata very efficiently, using a string dictionary to reduce strings like ``IRCodegen.cpp`` to a single integer stored on the ``llvm::Instruction``.
