Configuration
=============

Before starting MapD, the ``data`` directory must be initialized. To do
so, first create an empty directory at the desired path (``$MAPD_DATA``)
and change the owner to the user that the server will run as
(``$MAPD_USER``):

::

    sudo mkdir -p $MAPD_DATA
    sudo chown -R $MAPD_USER $MAPD_DATA

where ``$MAPD_USER`` is the system user account that the server will run
as, such as ``mapd``, and ``$MAPD_DATA`` is the desired path to the MapD
``data`` directory, such as ``/var/lib/mapd/data``.

Finally, run ``$MAPD_PATH/bin/initdb`` with the data directory path as
the argument:

::

    $MAPD_PATH/bin/initdb $MAPD_DATA

Configuration file
~~~~~~~~~~~~~~~~~~

MapD supports storing options in a configuration file. This is useful
if, for example, you need to run the MapD database and/or web servers on
different ports than the default. An example configuration file is
provided under ``$MAPD_PATH/mapd.conf.sample``.

To use options provided in this file, provide the path the the config
file to the ``--config`` flag of ``startmapd`` or ``mapd_server`` and
``mapd_web_server``. For example:

::

    $MAPD_PATH/startmapd --config $MAPD_DATA/mapd.conf
