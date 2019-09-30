``SysCatalog`` is a singleton class responsible for managing top-level OmniSciDB server metadata, including:

- Users
- Roles
- Databases
- Permissions

Data is persisted using a *SQLite* database at `<path_to_db/mapd_catalogs/omnisci_system_catalog>`. When the `SysCatalog` is created, the contents of the SQLite database is read and stored in memory. Subsequent reads only reference in-memory data structures. All writes are immediately flushed to SQLite.

Below is a UML representation of the schema of the SQLite database backing the system catalog:

.. image:: ../../img/catalog/syscat_schema.png
    :height: 400


****************************************
Users
****************************************

User accounts in OmniSciDB are global (that is, one user account can have access to multiple databases). The users login information is stored in the ``mapd_users`` system catalog table. When a user is authenticated, a ``UserMetadata`` struct is created to provide an in-memory representation of the user.

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

All system-level object privilege grants are stored and accessed from the ``mapd_object_permissions`` table. An object permission can be assigned to a role, and specifies what level of access a role has to various parts of the system. See `DB Object Privileges <https://docs.omnisci.com/latest/5_roles.html#ddl-roles-and-privileges>`_ for more information about interfacing with OmniSciDB's security model.

The ``granteeMap_`` in the ``SysCatalog`` class stores ``Grantee`` objects, representing object level permissions grants. The ``granteeMap_`` is built at startup.


The ``objectDescriptorMap_`` in the ``SysCatalog`` class stores ``ObjectRoleDescriptor``s, relating a role to a specific privilege grant. The  ``objectDescriptorMap_`` is also built when  Together, the ``granteeMap_`` and ``objectDescriptorMap_`` are referenced when determining if a user has access to perform an action in the database.
