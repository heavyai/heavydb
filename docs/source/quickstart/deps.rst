.. HeavyDB Quickstart

###################
Server Dependencies
###################

The supported developer path uses the **deps container** from GHCR, driven by
``dev-tools/dev.sh``. You do not install compilers or third-party libraries on
the host for that workflow—only Docker and GitHub access.

For the full command reference, see ``dev-tools/README.md`` and
``dev-tools/dev.sh <subcommand> --help``. This page covers prerequisites and
environment variables that trip people up in practice.

Prerequisites
=============

* **Docker** installed and running on the host
* **GitHub CLI** authenticated (``gh auth login``)
* **GitHub packages scope** (once per machine):

  .. code-block:: bash

      gh auth refresh -h github.com -s read:packages

* **GPU (optional):** ``nvidia-container-toolkit``. The dev script detects GPU
  via ``nvidia-smi``. Without it, containers run **CPU-only**—easy to miss when
  a CUDA build mysteriously falls back.

Node.js and Go are downloaded on demand into
``<repo-parent>/.heavyai-dev/tools/``. Override the cache with
``HEAVYAI_DEV_CACHE`` if you share tools across worktrees or want them on a
faster disk:

.. code-block:: bash

    export HEAVYAI_DEV_CACHE=/data/heavyai-cache

Environment Variables
=====================

None of these are required for a bare ``dev-tools/dev.sh build-heavydb`` (engine
only). They matter for full product / Immerse builds.

================================== =============================================
Variable                           Purpose
================================== =============================================
``MAPD_DEPS_SH``                   Path to ``mapd-deps.sh`` when working
                                   **outside** the usual deps image layout
                                   (default
                                   ``/usr/local/mapd-deps/mapd-deps.sh``).
``HEAVYAI_DEV_CACHE``              Cache for Node/Go and npm/Go module caches.
``MAPBOX_TOKEN``                   Optional. Omit and Mapbox tiles will not work
                                   in the built frontend (warning only).
``GOOGLE_API_KEY``                 Optional. Omit and geocoding features will not
                                   work in the built frontend.
================================== =============================================

Example — Immerse-capable session:

.. code-block:: bash

    export MAPBOX_TOKEN=pk.eyJ1...   # optional
    export GOOGLE_API_KEY=AIza...    # optional

Host mapd-deps (legacy / secondary)
===================================

If you are **not** using the deps container, install the prebuilt archive with
``scripts/mapd-deps-prebuilt.sh`` (Ubuntu ``--static`` or ``--shared``; Rocky
Linux static), then:

.. code-block:: bash

    source /usr/local/mapd-deps/mapd-deps.sh

Prefer ``dev-tools/dev.sh shell`` (or ``enter-deps``) over a host install when
you can—the container matches CI.
