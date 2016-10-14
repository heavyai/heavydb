Terminology
===========

Environment variables:

-  ``$MAPD_PATH``: MapD install directory, e.g. ``/opt/mapd/mapd2``
-  ``$MAPD_DATA``: MapD data directory, e.g. ``/var/lib/mapd/data``

Programs and scripts:

-  ``mapd_server``: MapD database server. Located at
   ``$MAPD_PATH/bin/mapd_server``.
-  ``mapd_web_server``: Web server which hosts the web-based frontend
   and provides database access over HTTP(S). Located at
   ``$MAPD_PATH/bin/mapd_web_server``.
-  ``initdb``: Initializes the MapD data directory. Located at
   ``$MAPD_PATH/bin/initdb``.
-  ``mapdql``: Command line-based program that gives direct access to
   the database. Located at ``$MAPD_PATH/bin/mapdql``.
-  ``startmapd``: All-in-one script that will initialize a MapD data
   directory at ``$MAPD_PATH/data``, offer to load a sample dataset, and
   then start the MapD server and web server. Located at
   ``$MAPD_PATH/startmapd``.

Other

-  ``systemd``: init system used by most major Linux distributions.
   Sample ``systemd`` target files for starting MapD are provided in
   ``$MAPD_PATH/systemd``.
