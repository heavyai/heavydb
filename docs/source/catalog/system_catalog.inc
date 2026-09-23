``SysCatalog`` is a singleton class responsible for managing top-level HeavyDB server metadata, including:

- Users
- Roles
- Databases
- Permissions

Data is persisted using a *SQLite* database at `<path_to_db/catalogs/system_catalog>`. When the `SysCatalog` is created, the contents of the SQLite database is read and stored in memory. Subsequent reads only reference in-memory data structures. All writes are immediately flushed to SQLite.

Below is a UML representation of the schema of the SQLite database backing the system catalog:

.. image:: ../img/catalog/syscat_schema.png
    :height: 400


****************************************
Users
****************************************

User accounts in HeavyDB are global (that is, one user account can have access to multiple databases). The users login information is stored in the ``mapd_users`` system catalog table. When a user is authenticated, a ``UserMetadata`` struct is created to provide an in-memory representation of the user.

****************************************
Roles
****************************************

Roles are used to grant permissions to groups of users. Roles are stored in the ``mapd_roles`` table. Each user also has a role corresponding to the name of the user, e.g. adding a user `Max` will also add a corresponding role called `Max` automatically. 

****************************************
Databases 
****************************************

The ``mapd_databases`` table functions as a lookup table for mapping a database name (a string) to the internal database identifier (an integer). 

****************************************
Permissions
****************************************

All system-level object privilege grants are stored and accessed from the
``mapd_object_permissions`` table. An object permission can be assigned to a
role and specifies the access that role has to parts of the system. See
`Roles <https://docs.nvidia.com/heavyai/installation-and-configuration/security/roles>`_
for the user-facing privilege commands.

The ``granteeMap_`` in the ``SysCatalog`` class stores ``Grantee`` objects, representing object level permissions grants. The ``granteeMap_`` is built at startup.


The ``objectDescriptorMap_`` in the ``SysCatalog`` class stores
``ObjectRoleDescriptor`` objects that relate roles to specific privilege grants.
Like the ``granteeMap_``, it is built at startup. Together, the two maps are
referenced when determining whether a user can perform an action in the
database.
