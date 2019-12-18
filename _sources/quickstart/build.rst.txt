.. OmniSciDB Quickstart

Build OmniSciDB
===============

.. note::

    Before you begin building, install the appropriate :doc:``./deps``.


OmniSciDB uses `CMake <https://cmake.org/>`_ for its build system. The following commands will build a simple, ``CUDA`` enabled build using 4 CPU threads:

.. code-block:: shell

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=debug ..
    make -j 4

The following ``cmake``/``ccmake`` options can enable/disable different features:

* ``-DCMAKE_BUILD_TYPE=release`` - Build type and compiler options to use.
*                                Options are ``Debug``, ``Release``, ``RelWithDebInfo``, ``MinSizeRel``, and unset.
* ``-DENABLE_ASAN=off`` - Enable address sanitizer. Default is ``off``.
* ``-DENABLE_AWS_S3=on`` - Enable AWS S3 support, if available. Default is ``on``.
* ``-DENABLE_CALCITE_DELETE_PATH=on`` - Enable Calcite Delete Path. Default is ``on``.
* ``-DENABLE_CALCITE_UPDATE_PATH=on`` - Enable Calcite Update Path. Default is ``on``.
* ``-DENABLE_CUDA=off`` - Disable CUDA. Default is ``on``.
* ``-DENABLE_CUDA_KERNEL_DEBUG=off`` - Enable debugging symbols for CUDA kernels. Will dramatically reduce kernel performance. Default is ``off``.
* ``-DENABLE_DECODERS_BOUNDS_CHECKING=off`` - Enable bounds checking for column decoding. Default is ``off``.
* ``-DENABLE_FOLLY=on`` - Use Folly. Default is ``on``.
* ``-DENABLE_IWYU=off`` - Enable include-what-you-use. Default is ``off``.
* ``-DENABLE_JIT_DEBUG=off`` - Enable debugging symbols for the JIT. Default is ``off``.
* ``-DENABLE_PROFILER=off`` - Enable google perftools. Default is ``off``.
* ``-DENABLE_STANDALONE_CALCITE=off`` - Require standalone Calcite server. Default is ``off``.
* ``-DENABLE_TESTS=on`` - Build unit tests. Default is ``on``.
* ``-DENABLE_TSAN=off`` - Enable thread sanitizer. Default is ``off``.
* ``-DENABLE_CODE_COVERAGE=off`` - Enable code coverage symbols (clang only). Default is ``off``.
* ``-DENALBE_JAVA_REMOTE_DEBUG=on`` - Enable Java Remote Debug. Default is ``off``.
* ``-DMAPD_DOCS_DOWNLOAD=on`` - Download the latest master build of the documentation / ``docs.mapd.com``. Default is ``off``. **Note:** this is a >50MB download.
* ``-DPREFER_STATIC_LIBS=off`` - Static link dependencies, if available. Default is ``off``.
