.. HeavyDB Row-Level Security

Row-Level Security (RLS)
========================

HeavyDB implements row-level security through **policies** attached to grantees
(users or roles). This is **not** PostgreSQL-style expression-based RLS: each policy is
a **column equality allow-list**. Rows are visible when their values match an allowed
literal for that column (multiple ``VALUES`` on one policy are **OR**'d; multiple
policies on a table—including inherited role policies and policies on **different
columns**—combine as **OR** across restrictions on that scan). For example, policies
on ``owner`` and ``region`` yield ``owner = … OR region = …``, not an AND of the two
column filters.

Tables without a policy for the current user are **not** filtered by RLS (opt-in model,
assuming normal ``SELECT`` privilege). **Superusers bypass RLS entirely**
(``SysCatalog::getRestrictions`` returns empty for superusers).

Policy Storage
--------------

Policies are stored in the system catalog SQLite table ``policies``:

.. code-block:: text

   role_name, db_id, table_id, column_id, value

Each ``CREATE POLICY`` value in the ``VALUES (...)`` clause becomes one row. The in-memory
``Grantee`` / ``Role`` graph mirrors restrictions for fast lookup
(``Catalog/Grantee.cpp``).

DDL Commands
------------

Superusers only (``session_ptr_->get_currentUser().isSuper``):

* **CREATE POLICY ON COLUMN** *table.column* **TO** *grantee* **VALUES (** *literals* **)**
* **SHOW [EFFECTIVE] POLICIES** *grantee* — ``EFFECTIVE`` merges restrictions from granted roles
* **DROP POLICY ON COLUMN** *table.column* **FROM** *grantee*

``CREATE OR REPLACE POLICY`` is **not** supported; drop and recreate instead.

Examples from ``Tests/ShowCommandsDdlTest.cpp``:

.. code-block:: sql

   CREATE POLICY ON COLUMN t1.owner TO u1 VALUES ('user1');
   CREATE POLICY ON COLUMN t2.make TO r1 VALUES ('Audi');
   SHOW POLICIES r1;

Constraints enforced at DDL time include: grantee must exist, cannot target a superuser,
one policy per grantee per column (duplicate column policies throw), and valid
table/column names in the current database.

Effective Restrictions
----------------------

``SysCatalog::getRestrictions(user_name, effective)`` collects policies for the
session user. When ``effective`` is true, restrictions from all assigned roles are
merged (union of allowed values per ``db_id/table_id/column_id`` key).

Query-Time Enforcement
----------------------

During SQL planning, ``Calcite::processImpl`` (``Calcite/Calcite.cpp``) loads
restrictions for the connected user and passes them to Calcite as ``TRestriction``
objects (database, table, column, allowed values). Java Calcite applies
``InjectFilterRule``, which adds ``column = literal`` filters on **logical table scans**
only—filters are not injected on every relational shape (for example some view or
subquery paths may differ from reader expectations). All matching equalities from
every applicable ``Restriction`` are collected into a single ``OR`` list
(cross-column policies are therefore OR'd together, as in the ``owner`` /
``region`` example above).

When any restriction applies, the server logs that row-level security filtering is
active for the query.

Legacy SAML column filters (deprecated key in ``Restriction``) are still merged for
backward compatibility but should not be used for new deployments.

Related Security Features
-------------------------

Role-based object privileges (tables, databases, servers) are separate from RLS and
are documented under :doc:`../catalog/index`. The ``--disable-column-level-security`` flag
controls a different mechanism and should not be conflated with SQL policies. SAML
authentication can supply legacy restrictions; SQL **CREATE POLICY** is the supported
mechanism for column filters.

Key Source Paths
----------------

* ``Catalog/SysCatalog.cpp`` — ``createPolicy``, ``dropPolicy``, ``getRestrictions``
* ``Catalog/DdlCommandExecutor.cpp`` — CREATE/SHOW/DROP POLICY commands
* ``Catalog/Grantee.cpp`` — restriction inheritance
* ``Calcite/Calcite.cpp`` — ``TRestriction`` handoff to Calcite
* ``Tests/ShowCommandsDdlTest.cpp`` — behavioral tests
