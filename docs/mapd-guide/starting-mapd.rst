Starting and Stopping MapD Core Services
========================================

MapD Core consists of two system services: ``mapd_server`` and
``mapd_web_server``. These services may be started individually using
``systemd`` or run via the interactive script ``startmapd``. For
permanent installations, it is recommended that you use ``systemd`` to
manage the MapD Core services.

MapD Core Via ``startmapd``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

MapD Core may be run via the ``startmapd`` script provided in
``$MAPD_PATH/startmapd``. This script handles creating the ``data``
directory if it does not exist, inserting a sample dataset if desired,
and starting both ``mapd_server`` and ``mapd_web_server``.

For backend rendering support, please start Xorg and set the ``DISPLAY``
environment variable before running ``startmapd``:

::

    sudo X :1 &
    export DISPLAY=:1

Starting MapD Core Via ``startmapd``
------------------------------------

To use ``startmapd`` to start MapD Core, run:

::

    $MAPD_PATH/startmapd --config /path/to/mapd.conf

if using a configuration file, or

::

    $MAPD_PATH/startmapd --data $MAPD_DATA

to explicitly specify the ``$MAPD_DATA`` directory.

Stopping MapD Core Via ``startmapd``
------------------------------------

To stop an instance of MapD Core that was started with the ``startmapd``
script, simply kill the ``startmapd`` process via ``CTRL-C`` or
``pkill startmapd``. You can also use ``pkill mapd`` to ensure all
processes have been killed.

MapD Core Via ``systemd``
~~~~~~~~~~~~~~~~~~~~~~~~~

For permanent installations of MapD Core, it is recommended that you use
``systemd`` to manage the MapD Core services. ``systemd`` automatically
handles tasks such as log management, starting the services on restart,
and restarting the services in case they die. Instructions for
configuring your system to start MapD Core via ``systemd`` are in the
`Initial Setup <#initial-setup>`__ section below.

Initial Setup
-------------

The provided ``install_mapd_systemd.sh`` script will ask a few questions
about your environment and then install the ``systemd`` service files
into the correct location.

::

    cd $MAPD_PATH/systemd
    ./install_mapd_systemd.sh

This script will ask for the location of the following directories:

-  ``MAPD_PATH``: path to the MapD Core installation directory
-  ``MAPD_STORAGE``: path to the storage directory for MapD Core data and
   configuration files
-  ``MAPD_USER``: user to run MapD Core as. User must exist prior to running
   the script.
-  ``MAPD_GROUP``: group to run MapD Core as. Group must exist prior to
   running the script.
-  ``MAPD_LIBJVM_DIR``: path to the ``libjvm`` library directory, as
   determined in the *Common Dependencies* section above.

For backend rendering-enabled builds, the ``install_mapd_systemd.sh``
script also installs a service named ``mapd_xorg``. This service is
configured to start ``Xorg`` on display ``:1``, which the
``mapd_server`` service is configured to use. Before proceeding, please
start the the ``mapd_xorg`` service before ``mapd_server`` if you wish
to utilize backend rendering:

::

    sudo systemctl start mapd_xorg
    sudo systemctl enable mapd_xorg # start mapd_xorg on startup

Starting MapD Core Via ``systemd``
----------------------------------

To manually start MapD Core via ``systemd``, run:

::

    sudo systemctl start mapd_server
    sudo systemctl start mapd_web_server

Stopping MapD Core Via ``systemd``
----------------------------------

To manually stop MapD Core via ``systemd``, run:

::

    sudo systemctl stop mapd_server
    sudo systemctl stop mapd_web_server

Enabling MapD Core on Startup
-----------------------------

To enable the MapD Core services to be started on restart, run:

::

    sudo systemctl enable mapd_server
    sudo systemctl enable mapd_web_server
