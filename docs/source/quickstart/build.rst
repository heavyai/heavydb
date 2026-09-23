.. HeavyDB Quickstart

Build HeavyDB
=============

Primary path: ``dev-tools/dev.sh``
----------------------------------

Local builds reuse the same ``scripts/ci/*.sh`` helpers as GitHub Actions, so a
dev-script build is the same process CI runs—just on your machine.

Prerequisites are in :doc:`./deps`. From the repository root:

.. code-block:: bash

    # Engine + Immerse + webserver + geos-dsos (no HeavyIQ)
    dev-tools/dev.sh build

    # Full product including HeavyIQ, optionally pack a product Docker image
    dev-tools/dev.sh build all --docker

    # Database engine only (uses already-built component artifacts if present)
    dev-tools/dev.sh build-heavydb

.. note::

   **Blank Immerse on :6273.** ``build-heavydb`` with no component artifacts is
   a **database-only** tree. SQL on 6274/6278 works; the web UI on 6273 will be
   blank or missing until Immerse and ``heavy_web_server`` are present. Use
   ``build`` / ``build all``, or pass ``--immerse`` / ``--webserver`` (or
   ``*-file``) into ``build-heavydb``. See :doc:`./start`.

Common options (see ``dev-tools/dev.sh build --help``):

.. code-block:: text

    --distro=ubuntu22.04|rockylinux8   # default ubuntu22.04
    --compiler=gcc|clang
    --cuda | --cpu
    --static | --shared
    --build-type=Release|Debug|RelWithDebInfo
    --nproc=<n>
    --docker          # after cpack, also build a product image
    --clean           # wipe output dirs (forces fresh cmake)
    --no-package      # skip cpack
    --run-tests=none|sanity|all

Pin component sources on a full build with ``--ref=``, ``--immerse-ref=``,
``--heavyiq-ref=``, ``--webserver-ref=``, and ``--repos-dir=``.

Output layout
~~~~~~~~~~~~~

=============================== ========================================
Item                            Default path
=============================== ========================================
Component artifacts             ``build/<distro>/components/``
HeavyDB build + CPack tarball   ``build/<distro>/``
Build logs                      ``build/<distro>/logs/``
=============================== ========================================

One ``build/`` tree covers every distro you have built (already gitignored).

CI-style and single-component builds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Named CI config (same scripts as Actions)
    dev-tools/dev.sh build ci tsan

    # One frontend/backend piece
    dev-tools/dev.sh build-component immerse

    dev-tools/dev.sh list   # authoritative config / test-variant names

Building the **deps image itself** (``build deps`` / ``build-deps-image``) is a
multi-hour job—only needed when ``scripts/mapd-deps-*.sh`` or the build
Dockerfiles change.

Secondary path: manual cmake inside the deps container
------------------------------------------------------

Use this when you are iterating on the engine and want to drive cmake / ninja
yourself, still on the CI-matched dependency stack.

1. Open a shell in the deps container:

   .. code-block:: bash

       dev-tools/dev.sh shell
       # alias: dev-tools/dev.sh enter-deps

2. Inside the container (``mapd-deps`` already sourced), configure and build
   under your usual build directory, for example:

   .. code-block:: bash

       mkdir -p build && cd build
       cmake -DCMAKE_BUILD_TYPE=Release ..
       cmake --build . -j"$(nproc)"

   Or use ninja if you prefer a Ninja generator.

Host cmake against a sourced ``mapd-deps.sh`` is the same idea without Docker;
keep the container path as the default so your toolchain matches CI.

CMake options (when driving cmake yourself)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

HeavyDB requires CMake **3.25** or newer. Frequently used cache options:

================================== ======= ==============================================
Option                             Default Description
================================== ======= ==============================================
``ENABLE_CUDA``                    ``on``  GPU execution support.
``ENABLE_TESTS``                   ``on``  Unit / integration tests.
``ENABLE_RENDERING``               ``off`` Backend rendering.
``PREFER_STATIC_LIBS``             varies  ``on`` on RedHat-derived hosts; ``off``
                                           elsewhere.
``MAPD_IMMERSE_DOWNLOAD``          ``on``  Download Immerse + web server for packaging.
``HEAVYIQ_DOWNLOAD``               ``on``  Download HeavyIQ for packaging.
================================== ======= ==============================================

Several ``ON``-by-default options disable themselves when their dependency is
missing—read the CMake summary. For packaging-only backend trees:

.. code-block:: bash

    cmake -DMAPD_IMMERSE_DOWNLOAD=off -DHEAVYIQ_DOWNLOAD=off ..

Testing after a build
---------------------

.. code-block:: bash

    # Sanity suite against ./build
    dev-tools/dev.sh test
    # or: dev-tools/dev.sh sanity-tests

    # Reproduce a CI failure in the exact GHCR image CI produced
    dev-tools/dev.sh test in-image debug-static tsan

On a PR, ``/test`` (and subsets like ``/test multi-render``) dispatch the
required-check workflow—handy when you only changed one area and do not want
the full gate.
