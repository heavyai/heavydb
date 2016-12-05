Users and Databases
===================

Users and databases can only be manipulated when connected to the MapD
Core system database ``mapd`` as a super user. MapD Core ships with a default
super user named ``mapd`` with default password ``HyperInteractive``.

``CREATE USER``
~~~~~~~~~~~~~~~

::

    CREATE USER <name> (<property> = value, ...);

Example:

::

    CREATE USER jason (password = 'MapDRocks!', is_super = 'true');

``DROP USER``
~~~~~~~~~~~~~

::

    DROP USER <name>;

Example:

::

    DROP USER jason;

``ALTER USER``
~~~~~~~~~~~~~~

::

    ALTER USER <name> (<property> = value, ...);

Example:

::

    ALTER USER mapd (password = 'MapDIsFast!');
    ALTER USER jason (is_super = 'false', password = 'SilkySmooth');

``CREATE DATABASE``
~~~~~~~~~~~~~~~~~~~

::

    CREATE DATABASE <name> (<property> = value, ...);

Example:

::

    CREATE DATABASE test (owner = 'jason');

``DROP DATABASE``
~~~~~~~~~~~~~~~~~

::

    DROP DATABASE <name>;

Example:

::

    DROP DATABASE test;

Basic Database Security Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system db is **mapd** The superuser is **mapd**

There are two user: **Michael** and **Nagesh**

There are two Databases: **db1** and **db2**

Only user **Michael** can see **db1**

Only user **Nagesh** can see **db2**

::

    admin@hal:~$ bin/mapdql mapd -u mapd -p HyperInteractive
    mapd> create user Michael (password = 'Michael');
    mapd> create user Nagesh (password = 'Nagesh');
    mapd> create database db1 (owner = 'Michael');
    mapd> create database db2 (owner = 'Nagesh');
    mapd> \q
    User mapd disconnected from database mapd
    admin@hal:~$ bin/mapdql db1 -u Nagesh -p Nagesh
    User Nagesh is not authorized to access database db1
    mapd> \q
    admin@hal:~$ bin/mapdql db2 -u Nagesh -p Nagesh
    User Nagesh connected to database db2
    mapd> \q
    User Nagesh disconnected from database db2
    admin@hal:~$ bin/mapdql db1 -u Michael -p Michael
    User Michael connected to database db1
    mapd> \q
    User Michael disconnected from database db1
    admin@hal:~$ bin/mapdql db2 -u Michael -p Michael
    User Michael is not authorized to access database db2
    mapd>
