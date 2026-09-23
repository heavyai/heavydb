.. HeavyDB Data Model

=======================================
DUMP TABLE and RESTORE TABLE Statements
=======================================

``DUMP TABLE`` archives the native data files, string dictionaries, schema, and
catalog metadata for a disk-backed table. ``RESTORE TABLE`` creates a new table
from such an archive.

Syntax
======

.. code-block:: sql

  DUMP TABLE table_name TO 'archive_path'
    [WITH (compression = 'gzip' | 'lz4' | 'none')];

  RESTORE TABLE table_name FROM 'archive_path'
    [WITH (compression = 'gzip' | 'lz4' | 'none')];

If ``compression`` is omitted, HeavyDB selects gzip when the corresponding
program is available, then lz4, and otherwise creates an uncompressed tar
archive. Restore must use the same compression mode as dump. If both commands
omit the option, this requires the same compressor availability in both
environments.

Restrictions
============

``DUMP TABLE`` supports persistent, disk-backed user tables. It does not
support system tables, foreign tables, views, or temporary tables. The archive
path must not already exist, and the user needs both ``SELECT`` privilege on
the table and ``CREATE TABLE`` privilege.

``RESTORE TABLE`` requires a target table name that does not already exist and
creates that table from the archived schema. Restoring over an existing table
is not supported. The user needs ``CREATE TABLE`` privilege.

Restore does not reshard or refragment data. In particular, the archived and
restored table definitions must have the same shard count, and their column
types and encodings must be compatible.

File Format
===========

The archive reuses HeavyDB's native data and dictionary files. Current
versioned archives also contain:

- ``_table.dumpversion`` — the archive format version.
- ``metadata.json`` — the table schema, epoch, table and column comments,
  column identifiers, and dictionary-directory metadata.

Older archives used ``_table.sql``, ``_table.oldinfo``, and ``_table.epoch``;
the restore implementation retains compatibility with those formats.

All page versions present in the native files are included in the archive.
``OPTIMIZE TABLE table_name`` recomputes chunk metadata. Adding
``WITH (VACUUM = 'true')`` also removes rows marked as deleted, but no
``OPTIMIZE TABLE`` option trims historical file pages.


DUMP TABLE
==========

For example:

.. code-block:: sql

  CREATE TABLE t (
    i INTEGER,
    s TEXT,
    d TEXT,
    SHARED DICTIONARY (d) REFERENCES t(s),
    SHARD KEY (i)
  ) WITH (FRAGMENT_SIZE = 10, SHARD_COUNT = 2);

  DUMP TABLE t TO '/tmp/t.tgz' WITH (compression = 'gzip');

The archive contains ``_table.dumpversion``, ``metadata.json``, the physical
``table_<database_id>_<table_id>`` directories for each shard, and any
``DB_<database_id>_DICT_<dictionary_id>`` directories used by dictionary
encoded columns. Native data files currently use the ``.data`` extension.


RESTORE TABLE
=============

To restore the example under a new name:

.. code-block:: sql

  RESTORE TABLE restored_t FROM '/tmp/t.tgz'
    WITH (compression = 'gzip');

HeavyDB reads the archived schema and creates the destination table. It then:

1. Validates the table options, shard count, column types, and encodings.
2. Extracts the archive into a temporary directory.
3. Maps archived column identifiers and dictionary directories to their new
   catalog identifiers, updating chunk headers when required.
4. Moves the table and dictionary directories into the current data directory.
5. Restores the archived table epoch and, for current archive versions, table
   and column comments.

The operation uses temporary backup directories while moving files. If an
exception occurs after destination files have been moved aside, HeavyDB
restores those files before returning the error.
