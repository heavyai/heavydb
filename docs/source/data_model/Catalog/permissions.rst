All privileges granted on the system level are stored and accessed from the ``mapd_object_permissions`` table. ``objectPermissions`` specify what level of access a User/Role has on an object.
Below are all the permissions which can be granted on an object:

.. code-block:: cpp

        static const AccessPrivileges NONE;
        // database permissions
        static const AccessPrivileges ALL_DATABASE;
        static const AccessPrivileges VIEW_SQL_EDITOR;
        static const AccessPrivileges ACCESS;
        // table permissions
        static const AccessPrivileges ALL_TABLE_MIGRATE;
        static const AccessPrivileges ALL_TABLE;
        static const AccessPrivileges CREATE_TABLE;
        static const AccessPrivileges DROP_TABLE;
        static const AccessPrivileges SELECT_FROM_TABLE;
        static const AccessPrivileges INSERT_INTO_TABLE;
        static const AccessPrivileges UPDATE_IN_TABLE;
        static const AccessPrivileges DELETE_FROM_TABLE;
        static const AccessPrivileges TRUNCATE_TABLE;
        static const AccessPrivileges ALTER_TABLE;
        // dashboard permissions
        static const AccessPrivileges ALL_DASHBOARD_MIGRATE;
        static const AccessPrivileges ALL_DASHBOARD;
        static const AccessPrivileges CREATE_DASHBOARD;
        static const AccessPrivileges VIEW_DASHBOARD;
        static const AccessPrivileges EDIT_DASHBOARD;
        static const AccessPrivileges DELETE_DASHBOARD;
        // view permissions
        static const AccessPrivileges ALL_VIEW_MIGRATE;
        static const AccessPrivileges ALL_VIEW;
        static const AccessPrivileges CREATE_VIEW;
        static const AccessPrivileges DROP_VIEW;
        static const AccessPrivileges SELECT_FROM_VIEW;
        static const AccessPrivileges INSERT_INTO_VIEW;
        static const AccessPrivileges UPDATE_IN_VIEW;
        static const AccessPrivileges DELETE_FROM_VIEW;
        static const AccessPrivileges TRUNCATE_VIEW;

``granteeMap_``, (a member variable of class SysCatalog) which consists of ``Grantee`` objects holds in-memory representation of the Grantees. ``User`` and ``Role`` classes extend Grantee to form respective classes.  ``granteeMap_`` is to check permissions of a user/role on an object.

.. code-block:: cpp

    class Grantee {
        ....
    };

    class User : public Grantee {
        ....
    };

    class Role : public Grantee {
        ....
    };



``objectDescriptorMap_`` (a member variable of the class SysCatalog) which consists of ``ObjectRoleDescriptor`` is also build from ``mapd_object_permissions`` which holds reference to UserRole and Role types [Note, if ``roleType`` is ``0`` then object is a role, else it is user].
Below is the structure of ``ObjectRoleDescriptor``:

.. code-block:: cpp

    struct ObjectRoleDescriptor {
        std::string roleName;
        bool roleType;
        int32_t objectType;
        int32_t dbId;
        int objectId;
        AccessPrivileges privs;
        int32_t objectOwnerId;
        std::string objectName;
    };