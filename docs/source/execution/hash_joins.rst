==========
Hash Joins
==========

A hash join is a technique used by HeavyDB to accelerate a SQL join query.

============
Introduction
============

A SQL join is a part of a query that combines rows from two tables. A join clause of the form ``<table1> JOIN <table2> ON <qualifier>`` can be used in place of a table name to temporarily combine the two tables for all pairs of rows where the expression (the qualifier) is true.

Loop Joins
----------

A simple way for the HeavyDB backend to execute a join, but also a slow way, is to use two nested loops. The outer loop scans the first of the two tables in the JOIN clause, while the inner loop scans the second of the two tables, and each inner loop iteration checks the join qualifier for the two rows currently being scanned.

A join can produce up to MxN combined rows where M and N are the sizes of the two tables. With a loop join, the join qualifier must be checked MxN times using this technique, and the inner table must be rescanned many times, one full scan for each row in the outer table. (Complexity is O(n^2), quadratic).

Hash Joins
----------

A faster way to execute a join is to eliminate the inner loop from a loop join and replace it with a hash table lookup. Some part of the row from the outer loop is used as a key into the hash table to find a list of all of the inner table's rows that are known to match the outer table's row.

While it takes time and memory for the hash table to be generated, this up-front investment can pay off big by avoiding the need to scan the inner table repeatedly as a loop join would have required. The inner table is only scanned once to build the hash table. (Complexity is O(n), linear).

A failed hash join can sometimes automatically fall back to a loop join, such as when there isn't enough memory for the hash table to be built.

=================
Hash Join Buffers
=================

HeavyDB can choose between different hash-table implementations when executing
a join. Each hash table stores its sections in a contiguous buffer.

A hash join buffer can have up to four sections which are located consecutively in memory:

1) Keys
2) Offsets
3) Counts
4) Payloads

Keys Section
------------

The Keys section of a hash join buffer, if present, is an array containing the hashable keys from the key/value pairs to be stored in the hash table.

Keys can have multiple components, depending on the kind of hash join in use.

A sentinel value (typically a huge value near max int) is used to indicate an empty location in the Keys section.

Offsets Section
---------------

The Offsets section of a hash join buffer, if present, is an array containing integer indexes into the Payloads section.

The Offsets section is a parallel array with the Keys section and/or with the Counts section, meaning that if there is information stored in those other sections at some location, then the matching location in the Offsets section will contain an integer offset.

Empty locations in the Offsets section are filled with -1.

Counts Section
--------------

The Counts section of a hash join buffer, if present, is an array containing the integer sizes of the subarrays stored in the Payloads section.

The Counts section is a parallel array with the Keys section and/or with the Offsets section, meaning that if there is information stored in those other sections at some location, then the matching location in the Counts section will contain an integer count.

Empty locations in the Counts section are filled with zeros.

Payloads Section
----------------

The Payloads section of a hash join buffer is an array of subarrays, with each subarray containing one or more row ID integer references for rows in the one of the join tables. The Payloads section is always present in a hash join buffer in some form although it can be interleaved with the Keys section.

The location of each Payloads subarray is stored in the Offsets section. The length of each Payloads subarray is stored in the Counts section.

===================
Kinds of Hash Joins
===================

A HeavyDB hash join buffer can have either a one-to-one layout or a one-to-many layout.

A one-to-one layout is the least-complicated and fastest kind of hash join buffer. The Offsets and Counts sections are not required for a one-to-one layout because there is always exactly one payload row ID stored per key.

A one-to-many hash join buffer will have at least the Offsets, Counts, and Payloads sections. Offsets and Counts are both required to define the subarrays present in a one-to-many Payloads section.

In some cases the Keys section can be omitted from the hash join buffer giving perfect hashing, where integer keys are directly mapped to locations in the other sections.

HeavyDB selects among these implementations:

* ``PerfectJoinHashTable`` maps a bounded integer-like key range directly to
  buffer slots. It supports one-to-one and one-to-many layouts.
* ``BaselineJoinHashTable`` stores explicit keys and handles composite keys or
  key ranges that are unsuitable for perfect hashing. It also supports both
  layouts.
* ``BoundingBoxIntersectJoinHashTable`` builds a one-to-many spatial hash for
  bounding-box intersection.
* ``RangeJoinHashTable`` extends the bounding-box implementation for supported
  range predicates.

=============================
Inspecting a Hash Join Buffer
=============================

The ``HashJoin`` subclasses provide ``toString()`` for decoding a small buffer
into a human-readable representation. Verbose level 2 logs this representation
automatically when the buffer is no larger than 1,000 bytes.

Buffers may be built in parallel, so equivalent hash tables do not always have
identical byte layouts. Tests should use ``toSet()`` to decode the entries into
a set and compare their logical contents.

Perfect Hashing Examples
------------------------

Consider a join where the inner table has one row for each key:

.. code-block:: sql

  CREATE TABLE outer_t (a INTEGER);
  CREATE TABLE inner_t (b INTEGER);

  INSERT INTO outer_t VALUES (1), (1), (2), (3), (4);
  INSERT INTO inner_t VALUES (0), (1), (3);

  SELECT * FROM outer_t JOIN inner_t ON a = b;

The unique inner keys permit a one-to-one perfect hash table. A decoded buffer
has a payload slot for each value in the selected key range; ``*`` marks an
empty slot:

.. code-block:: text

  | perfect one-to-one | payloads 0 1 * 2 |

Adding a duplicate inner key changes the required layout:

.. code-block:: sql

  INSERT INTO inner_t VALUES (3);

The one-to-many layout uses offsets and counts to locate all payload row IDs
for a key:

.. code-block:: text

  | perfect one-to-many |
  | offsets 0 1 * 2 | counts 1 1 * 2 | payloads 0 1 2 3 |

Baseline Hashing Example
------------------------

A composite equality condition requires explicit tuple keys:

.. code-block:: sql

  CREATE TABLE outer_pair (a1 INTEGER, a2 INTEGER);
  CREATE TABLE inner_pair (b1 INTEGER, b2 INTEGER);

  SELECT *
  FROM outer_pair
  JOIN inner_pair ON outer_pair.a1 = inner_pair.b1
                 AND outer_pair.a2 = inner_pair.b2;

HeavyDB coalesces the compatible equality predicates into one composite key
and builds a ``BaselineJoinHashTable``. A one-to-one baseline layout can
interleave each payload row ID with its key. If an inner key is duplicated,
the implementation switches to one-to-many and uses all four sections.

Bounding-Box Intersection
-------------------------

Supported geospatial joins use ``BoundingBoxIntersectJoinHashTable`` rather
than comparing every geometry pair. The implementation chooses spatial bucket
sizes, inserts each inner geometry's bounds into the buckets it overlaps, and
stores the matching inner row IDs in a one-to-many payload section. An outer
geometry probes its overlapping buckets; the original spatial predicate then
determines the exact matches.

The same inner row can occupy several spatial buckets, so explicit keys,
offsets, counts, and payloads are required. Bucket sizing and table-size limits
are controlled by the bounding-box intersection options in
``ThriftHandler/CommandLineOptions.cpp``.

==========================
Equijoins vs Non-Equijoins
==========================

Equality and bitwise-equality predicates over supported scalar or
dictionary-encoded string columns are candidates for perfect or baseline hash
joins. Bounding-box and supported range predicates use their specialized hash
tables.

Other non-equijoins require nested-loop execution. Loop-join fallback depends
on the query shape, inner-table size, and the ``allow-loop-joins`` setting; a
failed hash-table build therefore cannot always fall back automatically.

==========
Coalescing
==========

When a join has multiple compatible equality predicates over the same pair of
tables, HeavyDB can combine them into one tuple equality. Numeric, Boolean, and
dictionary-encoded string columns are supported, up to the runtime limit of
eight conditions. The tuple is used as a composite baseline-hash key, avoiding
separate hash-table probes for each predicate.
