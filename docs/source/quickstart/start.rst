.. HeavyDB Quickstart

#############
Start HeavyDB
#############

After a :doc:`./build`, artifacts live under ``build/<distro>/`` (dev-script) or
your cmake build directory (manual path).

What you need for a working UI
==============================

A usable Immerse session on port **6273** needs all three:

* ``heavydb`` (and friends: ``initheavy``, ``heavysql``)
* ``frontend/`` — Immerse static assets
* ``bin/heavy_web_server`` — serves Immerse

Engine-only trees answer SQL on **6274** / **6278**; **6273** stays blank or
missing until the frontend and web server are present. That is the usual
"I built heavydb but the browser is empty" failure mode.

Using ``startheavy``
====================

From a build directory that contains ``bin/`` (for a cmake build directly under
the repo, that is often ``build/``; for ``dev-tools`` output, use
``build/<distro>/``):

.. code-block:: bash

    ../startheavy
    # or, from repo root pointing at the build tree you care about:
    #   ./startheavy --data ...   # see startheavy --help / scripts/innerstartheavy

``startheavy`` will:

* initialize the ``storage`` directory with ``initheavy`` when needed;
* start ``heavydb``;
* start ``heavy_web_server`` when both the binary and a ``frontend`` directory
  exist; and
* start HeavyIQ when its directory is present.

Useful flags: ``--data PATH``, ``--config PATH``, ``--base-port PORT``,
``--non-interactive``. Extra arguments are forwarded to ``heavydb``.

Default ports
=============

============= ============================
Service       Default port
============= ============================
Thrift TCP    6274
HTTP/JSON     6278
HTTP/binary   6276
Calcite       6279
Web / Immerse 6273
============= ============================

Manual start
============

From the build directory:

.. code-block:: bash

    mkdir -p storage
    ./bin/initheavy -f --data storage
    ./bin/heavydb storage

In another terminal:

.. code-block:: bash

    ./bin/heavysql -p HyperInteractive

Default development user is ``admin``; password ``HyperInteractive``.

If the web server was built:

.. code-block:: bash

    ./bin/heavy_web_server

Then open http://localhost:6273 when a frontend is present.

Product Docker image
====================

If you built with ``dev-tools/dev.sh build all --docker`` (or equivalent), run
the resulting product image with your usual Docker GPU/port mappings. Prefer
that path when you want the packaged layout rather than a raw build tree.
