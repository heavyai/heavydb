OmniSciDB stores database, table, and user information in a centralized `Catalog`. Each database has its own `Catalog`, with the top-level *System Catalog* (`SysCatalog`) managing catalogs across databases. Each catalog (including the system catalog) is backed by a `SQLite` database in the `mapd_catalogs` directory (a top-level directory in the OmniSciDB data directory). In addition to managing catalogs for individual databases, the `SysCatalog` manages users, roles, and higher-level object permissions.

Both SysCatalog and Catalog classes are defined in files of the same name in the `Catalog` source directory. All catalog-related code lives inside the ``Catalog_Namespace``.

.. image:: ../img/catalog/catalog_overview.png
