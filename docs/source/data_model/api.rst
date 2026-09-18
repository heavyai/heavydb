.. HeavyDB Data Model

============
External API
============

HeavyDB exposes its database API through
`Apache Thrift <https://thrift.apache.org/>`_. The service definition is
``heavy.thrift`` at the repository root. That file includes shared data types
from ``common.thrift``, completion hints, serialized result sets, and extension
function definitions.

The CMake build generates C++ bindings in ``build/gen-cpp`` and links them into
the ``mapd_thrift`` library. Other languages generate equivalent client and
server classes from the same service definition.

Server Implementation
=====================

The generated ``HeavyIf`` interface is implemented by ``DBHandler`` in
``ThriftHandler/DBHandler.h`` and ``ThriftHandler/DBHandler.cpp``.
``HeavyDB.cpp`` creates a single Thrift processor around that handler and
serves it over each enabled transport. As a result, all transports expose the
same service methods.

Most database operations are session based. A client calls ``connect`` with
database credentials and receives a ``TSessionId``. It passes that identifier
to subsequent methods and calls ``disconnect`` when finished.

Transports and Ports
====================

HeavyDB starts three Thrift listeners by default:

* Buffered TCP with the Thrift binary protocol uses port 6274 and is
  configured with ``--port``.
* HTTP with the Thrift JSON protocol uses port 6278 and is configured with
  ``--http-port``.
* HTTP with the Thrift binary protocol uses port 6276 and is configured with
  ``--http-binary-port``.

The HTTP/binary listener can be disabled with
``--enable-http-binary-server=false``. Supplying both ``--ssl-cert`` and
``--ssl-private-key`` replaces the plain sockets with TLS sockets for the
enabled listeners; clients must then use TLS or HTTPS as appropriate.

Clients
-------

The repository contains several clients and examples built on this API:

* ``heavysql`` under ``SQLFrontend``;
* the JDBC driver and SQLImporter under ``java``;
* KafkaImporter and StreamImporter under ``ImportExport``; and
* small insert examples under ``SampleCode``.

Each client supports the transports implemented by that client; the presence
of a server listener does not imply that every client exposes a switch for it.
Consult the client's ``--help`` output or its own documentation.

Internal Thrift Services
========================

HeavyDB also uses separate Thrift services between internal processes. In
particular, HeavyDB is the client of the Java Calcite server defined by
``java/thrift/calciteserver.thrift``. This service is independent of the
external ``Heavy`` service described above. See :doc:`../calcite/calcite_parser`
for the query-planning flow.
