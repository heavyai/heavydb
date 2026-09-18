.. HeavyDB Quickstart

###############
Getting Started
###############

These pages get a developer from a clean checkout to a running HeavyDB using
the **dev-tools** workflow (same scripts CI uses). The primary path is
Docker-based; a secondary path is ``dev-tools/dev.sh shell`` plus manual
cmake/ninja inside the deps container.

.. toctree::
    :maxdepth: 1

    deps
    build
    start

Start with :doc:`./deps`, then :doc:`./build`, then :doc:`./start`.

For exhaustive CI workflow, secret, and publishing detail, use the internal
developer runbook and ``dev-tools/README.md``—those topics are intentionally
not mirrored here.
