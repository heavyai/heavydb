A chunk represents a vertical partition (column) of a fragment. A set of chunks makes up a fragment. A chunk is the smallest unit of data OmniSci works with. Different chunks in a given fragment will contain the same number of items, but depending on the data type of the column the size of the chunks (in bytes) will differ.  A chunk has a maximum size of 1GByte.

Each chunk contains metadata about the data in the chunk. This information is updated any time a change is made to the chunk.

.. code-block:: cpp

    struct ChunkStats {
        Datum min;
        Datum max;
        bool has_nulls;
    };

    struct ChunkMetadata {
        SQLTypeInfo sqlType;
        size_t numBytes;
        size_t numElements;
        ChunkStats chunkStats;
    }




Example for chunk physical representations
--------------------------------------------

The example is based on a table **t**, defined as the code snippet below, with the default fragment size of 32M rows and loaded with 8M records.

.. code-block:: sql
    
    create table t(
        c1 SMALLINT,
        c2 INTEGER
    )

The table will contain a single fragment with the following two chunks;

    chunk for **c1** will be 8M x 2 bytes (Smallint size) = **16Mbytes** in actual chunk physical size

    chunk for **c2** will be 8M x 4 bytes (Int size) = **32Mbytes** in actual chunk physical size.

Table **t** contains a single fragment, defined solely by the number of rows in the table. As each individual chunk is less than the maxium size of 1GByte they do not cause the fragment to be split.

Adding another column **c3** as Date will add the following chunk to the table;

    chunk for **c3** would be 8M x 8 bytes (Date size) = **64MBytes** in actual physical size

As this chunk is again under the maximum of 1GByte, its addition still does not impact the number of fragments in the table, which stays at one. However, if a column of a variable length type was added to the table (such as non text encoded string) and the import process caused that new chunk to grow beyond 1GByte, this would cause the single fragment to split into multiples.


ChunkKey
----------------

A ChunkKey is a vector of integer values that refer to a DB object or a piece of memory set aside by the executor. If the first value in the chunk key is negative it is an **ephemeral** chunk that is being used for storage of a transient object at the 'logical' layer. If the first value of the chunk key is positive it is referring to a **chunk** in the DB, it can be referenced logically at different levels.


.. table:: CHUNK KEY MAPPING
    :widths: 12 12 13 15 13 35

    +------------+------------+------------+------------+------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | CK[0]      | CK[1]      | CK[2]      | CK[3]      | CK[4]      |                                                                                                                                                                                                 |
    +============+============+============+============+============+=================================================================================================================================================================================================+
    | -1         | Some number|            |            |            | Ephemeral temp chunk                                                                                                                                                                            |
    +------------+------------+------------+------------+------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | db > 0     |            |            |            |            | Referencing a DB id of db                                                                                                                                                                       |
    +------------+------------+------------+------------+------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | db > 0     | tb > 0     |            |            |            | Referencing a DB id of db and table table id tb                                                                                                                                                 |
    +------------+------------+------------+------------+------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | db > 0     | tb > 0     | col > 0    |            |            | Referencing a DB id of db and table table id tb and column col                                                                                                                                  |
    +------------+------------+------------+------------+------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | db > 0     | tb > 0     | col > 0    | frag >= 0  |            | Referencing a DB id of db and table table id tb and column col.  Which fragment this chunk                                                                                                      |
    +------------+------------+------------+------------+------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | db > 0     | tb > 0     | col > 0    | frag >= 0  | var > 0    | Referencing a DB id of db and table table id tb and column col.  Which fragment this chunk belongs to.  Var is used for variable length columns to indicate if it is data chunk of offset chunk |
    +------------+------------+------------+------------+------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

From a DataMgr perspective the **normal** ChunkKeys will be of the last two forms

``db, tb, col, frag``

Or

``db, tb, col, frag, var``

As these uniquely identify the smallest unit of data that the MapD core system will operate on.  The `omnisql` processor provides a number of utilities to examine the ChunkKeys used in a database.
