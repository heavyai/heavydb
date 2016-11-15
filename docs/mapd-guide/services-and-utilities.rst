MapD Services and Utilities
===========================

Assuming ``$MAPD_PATH`` is the directory where MapD software is
installed, make sure that ``$MAPD_PATH/bin`` is in ``PATH``:

::

    export PATH=$MAPD_PATH:$PATH

Services
~~~~~~~~

``mapd_server``
---------------

::

    mapd_server $MAPD_DATA [--cpu|--gpu]
                           [--config </path/to/mapd.conf>]
                           [{-p|--port} <port number>]
                           [--http-port <port number>]
                           [--ldap-uri <string>]
                           [--ldap-ou-dc <string>
                           [--flush-log]
                           [--disable-rendering]
                           [--num-gpus <number>]
                           [--start-gpu <number>]
                           [--version|-v]
                           [--help|-h]

This command starts the MapD server process. ``$MAPD_DATA`` must match
that in the ``initdb`` command when it was run. The options are:

-  ``[--cpu|--gpu]``: Execute queries on CPU-only or on both GPU and
   CPU. The default is ``--gpu``.
-  ``[--config]``: Path to mapd.conf.
-  ``[{-p|--port} <port number>]``: Specify the port for MapD's
   binary-over-TCP protocol. The default is port 9091.
-  ``[{--http-port} <port number>]``: Specify the port for MapD's
   JSON-over-HTTP protocol. The default is port 9090.
-  ``[--ldap-uri <string>]``: LDAP URI for authentication, for example
   ldap://ldap.mycompany.com
-  ``[--ldap-ou-dc <string]``: LDAP organization unit and domain,
   defaults to ou=users,dc=mapd,dc=com.
-  ``[--flush-log]``: Flush log files to disk. Useful for ``tail -F`` on
   log files.
-  ``[--disable-rendering]``: Disables server-based rendering. Defaults
   to FALSE.
-  ``[--num-gpus <number>]``: Number of GPUs to use, defaults to number
   detected on hardware.
-  ``[--start-gpu <number>]``: When not using all detected GPUs, start
   using GPUs at this number, defaults to zero.
-  ``[--version|-v]``: Prints version number.
-  ``[--help|-h]``: Print this help text.

``mapd_web_server``
-------------------

::

    mapd_web_server [{--port} <port number>]
                    [{--proxy-backend} <bool>]
                    [{--backend-url} <backend URL>]
                    [{--frontend} <path/to/frontend>]
                    [{--enable-https} <bool>]
                    [{--cert} <cert.pem>]
                    [{--key} <key.pem>]
                    [{--tmpdir} </path/to/tmp>]

This command starts the MapD web server. This server provides access to
MapD's visualization frontend and allows the frontend to communicate
with the MapD Server. HTTPS certificates and keys may be generated via
the provided ``generate_cert`` utility, or provided by your Certificate
Authority. The options are:

-  ``[{--port} <port number>]``: Specify the port the web server listens
   on. The default is port 9092.
-  ``[{--proxy-backend} <bool>]``: Specify whether to act as a proxy to
   the backend. The default is ``true``.
-  ``[{--backend-url} <backend URL>]``: Specify the URL to the backend
   HTTP server. The default is ``http://localhost:9090``.
-  ``[{--frontend} <path/to/frontend>]``: Specify the path to the
   frontend directory. The default is ``frontend``.
-  ``[{--enable-https} <bool>]``: Enable HTTPS for serving the frontend.
   The default is ``false``.
-  ``[{--cert} <cert.pem>]``: Path to the HTTPS certificate file. The
   default is ``cert.pem``.
-  ``[{--key} <key.pem>]``: Path to the HTTPS key file. The default is
   ``key.pem``.
-  ``[{--tmpdir} </path/to/tmp>]``: Path to custom temporary directory.
   The default is ``/tmp/``.

The temporary directory is used as a staging location for file uploads.
It is sometimes desirable to place this directory on the same file
system as the MapD data directory. If not specified on the command line,
``mapd_web_server`` also respects the standard ``TMPDIR`` environment
variable as well as a specific ``MAPD_TMPDIR`` environment variable, the
latter of which takes precedence. Defaults to the system default
``/tmp/`` if neither the command line argument nor at least one of the
environment variables are specified.

Utilities
~~~~~~~~~

``initdb``
----------

The very first step before using MapD is to initialize the MapD data
directory via ``initdb``:

::

    initdb [-f] $MAPD_DATA

This creates three subdirectories:

-  ``mapd_catalogs``: stores MapD catalogs
-  ``mapd_data``: stores MapD data
-  ``mapd_log``: contains all MapD log files. MapD uses
   `glog <https://code.google.com/p/google-glog/>`__ for logging.

The ``-f`` flag forces ``initdb`` to overwrite existing data and
catalogs in the specified directory.

``generate_cert``
-----------------

::

    generate_cert [{-ca} <bool>]
                  [{-duration} <duration>]
                  [{-ecdsa-curve} <string>]
                  [{-host} <host1,host2>]
                  [{-rsa-bits} <int>]
                  [{-start-date} <string>]

This command generates certificates and private keys for an HTTPS
server. The options are:

-  ``[{-ca} <bool>]``: Whether this certificate should be its own
   Certificate Authority. The default is ``false``.
-  ``[{-duration} <duration>]``: Duration that certificate is valid for.
   The default is ``8760h0m0s``.
-  ``[{-ecdsa-curve} <string>]``: ECDSA curve to use to generate a key.
   Valid values are ``P224``, ``P256``, ``P384``, ``P521``.
-  ``[{-host} <string>]``: Comma-separated hostnames and IPs to generate
   a certificate for.
-  ``[{-rsa-bits} <int>]``: Size of RSA key to generate. Ignored if
   --ecdsa-curve is set. The default is ``2048``.
-  ``[{-start-date} <string>]``: Start date formatted as
   ``Jan 1 15:04:05 2011``

``mapdql``
----------

::

    mapdql [<database>]
           [{--user|-u} <user>]
           [{--passwd|-p} <password>]
           [--port <port number>]
           [{-s|--server} <server host>]
           [--http]

``mapdql`` is the client-side SQL console. All SQL statements can be
submitted to the MapD Server and the results are returned and displayed.
The options are:

-  ``[<database>]``: Database to connect to. Not connected to any
   database if omitted.
-  ``[{--user|-u} <user>]``: User to connect as. Not connected to MapD
   Server if omitted.
-  ``[{--passwd|-p} <password>]``: User password.
-  ``[--port <port number>]``: Port number of MapD Server. The default
   is port 9091.
-  ``[{--server|-s} <server host>]``: MapD Server hostname in DNS name
   or IP address. The default is localhost.
-  ``[--http]``: Use the Thrift HTTP transport instead of the default TCP
   transport. Must set ``--port`` to ``mapd_web_server``'s port (default 9092).

In addition to SQL statements ``mapdql`` also accepts the following list
of backslash commands:

-  ``\h``: List all available backslash commands.
-  ``\u``: List all users.
-  ``\l``: List all databases.
-  ``\t``: List all tables.
-  ``\d <table>``: List all columns of table.
-  ``\c <database> <user> <password>``: Connect to a new database.
-  ``\gpu``: Switch to GPU mode in the current session.
-  ``\cpu``: Switch to CPU mode in the current session.
-  ``\timing``: Print timing information.
-  ``\notiming``: Do not print timing information.
-  ``\version``: Print MapD Server version.
-  ``\memory_summary``: Print memory usage summary.
-  ``\copy <file path> <table>``: Copy data from file on client side to
   table. The file is assumed to be in CSV format unless the file name
   ends with ``.tsv``.
-  ``\copygeo <file path> <table>``: Experimental support for copying a
   server side shapefile to a new table. Coordinates are assumed to be
   in the EPSG:4326 / WGS 84 / latitude+longitude projection.
-  ``\q``: Quit.

``mapdql`` automatically attempts to reconnect to ``mapd_server`` in
case it restarts due to crashes or human intervention. There is no need
to restart or reconnect.
