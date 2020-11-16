==========
Hash Joins
==========

A hash join is a technique used by OmniSciDB to accelerate a SQL join query.

============
Introduction
============

A SQL join is a part of a query that combines rows from two tables. A join clause of the form ``<table1> JOIN <table2> ON <qualifier>`` can be used in place of a table name to temporarily combine the two tables for all pairs of rows where the expression (the qualifier) is true.

Loop Joins
----------

A simple way for the OmniSciDB backend to execute a join, but also a slow way, is to use two nested loops. The outer loop scans the first of the two tables in the JOIN clause, while the inner loop scans the second of the two tables, and each inner loop iteration checks the join qualifier for the two rows currently being scanned.

A join can produce up to MxN combined rows where M and N are the sizes of the two tables. With a loop join, the join qualifier must be checked MxN times using this technique, and the inner table must be rescanned many times, one full scan for each row in the outer table. (Complexity is O(n^2), quadratic).

Hash Joins
----------

A faster way to execute a join is to eliminate the inner loop from a loop join and replace it with a hash table lookup. Some part of the row from the outer loop is used as a key into the hash table to find a list of all of the inner table's rows that are known to match the outer table's row.

While it takes time and memory for the hash table to be generated, this up-front investment can pay off big by avoiding the need to scan the inner table repeatedly as a loop join would have required. The inner table is only scanned once to build the hash table. (Complexity is O(n), linear).

A failed hash join can sometimes automatically fall back to a loop join, such as when there isn't enough memory for the hash table to be built.

=================
Hash Join Buffers
=================

OmniSciDB can choose between different kinds of hash tables (described later) when executing a SQL join query, but each hash join clause will have a single buffer allocated in memory, except that multiple buffers will sometimes be coalesced into a single buffer. (Coalescing is described later.)

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

An OmniSciDB hash join buffer can have either a one-to-one layout or a one-to-many layout.

A one-to-one layout is the least-complicated and fastest kind of hash join buffer. The Offsets and Counts sections are not required for a one-to-one layout because there is always exactly one payload row ID stored per key.

A one-to-many hash join buffer will have at least the Offsets, Counts, and Payloads sections. Offsets and Counts are both required to define the subarrays present in a one-to-many Payloads section.

In some cases the Keys section can be omitted from the hash join buffer giving perfect hashing, where integer keys are directly mapped to locations in the other sections.

OmniSciDB automatically selects from one of three C++ classes when building a hash join buffer:

============================ ============================= ========================
C++ Class Name               Layouts                       Selected For
============================ ============================= ========================
JoinHashTable                One-To-One or One-To-Many     Perfect hashing
BaselineJoinHashTable        One-To-One or One-To-Many     Keyed hashing
OverlapsJoinHashTable        only One-To-Many              Geospatial hashing
============================ ============================= ========================

=============================
Inspecting a Hash Join Buffer
=============================

For learning about and/or for debugging a hash join buffer, OmniSciDB provides a toString() function for decoding the buffer into a human-readable string.

Be aware that a hash join buffer often may be built using multiple threads, possibly causing the exact layout of a buffer to vary for the same SQL across different builds. (See later section about comparing buffers for more info.)

One-To-One JoinHashTable Example
--------------------------------

This SQL causes OmniSciDB to use one-to-one perfect hashing shown by the very simple hash join buffer containing only a small Payloads section. The second table is selected for hashing because it has the lowest cardinality and because it has no duplicate records.

    SQL:
      create table table1 (a integer);
      create table table2 (b integer);

      insert into table1 values (1);
      insert into table1 values (1);
      insert into table1 values (2);
      insert into table1 values (3);
      insert into table1 values (4);

      insert into table2 values (0);
      insert into table2 values (1);
      insert into table2 values (3);

      select * from table1 join table2 on a = b;

    C++ toString():
      | payloads 0 1 * 2 |

One-To-Many JoinHashTable Example
---------------------------------

This SQL is nearly identical to the previous example, except that a duplicate record has been added to the second table, causing one-to-many perfect hashing to be selected instead of one-to-one perfect hashing. The one-to-many hashing requires Offsets and Counts sections to be built into the hash join buffer in addition to the Payloads section, and the Offsets section acts as the hash table instead of the Payloads section.

    SQL:
      create table table1 (a integer);
      create table table2 (b integer);

      insert into table1 values (1);
      insert into table1 values (1);
      insert into table1 values (2);
      insert into table1 values (3);
      insert into table1 values (4);

      insert into table2 values (0);
      insert into table2 values (1);
      insert into table2 values (3);
      insert into table2 values (3);

      select * from table1 join table2 on a = b;

    C++ toString():
      | offsets 0 1 * 2 | counts 1 1 * 2 | payloads 0 1 2 3 |

One-To-One BaselineJoinHashTable Example
----------------------------------------

Adding a second column to one of the tables, then including that column in the join qualifier (the ``ON`` expression) prevents perfect hashing from being used and requires a Keys section to be built into the hash buffer. As an optimization that is possible with one-to-one hashing, the payloads are interleaved into the keys as if each payload row ID was an additional key component.

    SQL:
      create table table1 (a1 integer, a2 integer);
      create table table2 (b integer);

      insert into table1 values (1, 11);
      insert into table1 values (2, 12);
      insert into table1 values (3, 13);
      insert into table1 values (4, 14);

      insert into table2 values (0);
      insert into table2 values (1);
      insert into table2 values (3);

      select * from table1 join table2 on a1 = b and a2-10 = b;

    C++ toString():
      | keys * (1,1,1) (3,3,2) (0,0,0) * * |

One-To-Many BaselineJoinHashTable Example
-----------------------------------------

Adding a duplicate record to the previous example turns the hash join into a one-to-many lookup, requiring all four buffer sections to be built.

    SQL:
      create table table1 (a1 integer, a2 integer);
      create table table2 (b integer);

      insert into table1 values (1, 11);
      insert into table1 values (2, 12);
      insert into table1 values (3, 13);
      insert into table1 values (4, 14);

      insert into table2 values (0);
      insert into table2 values (1);
      insert into table2 values (3);
      insert into table2 values (3);

      select * from table1 join table2 on a1 = b and a2-10 = b;

    C++ toString():
      | keys * (1,1) (3,3) (0,0) * * | offsets * 0 1 3 * * | counts * 1 2 1 * * | payloads 1 2 3 0 |

One-To-Many OverlapsJoinHashTable Example
-----------------------------------------

TODO

===========================
Comparing Hash Join Buffers
===========================

To help support unit testing, a hash join buffer can be decoded into a std::set by using the toSet() member function. Two of these sets can be compared for equality to determine if the hash join buffers are logically equal, even when the exact layouts of the buffers may differ in memory, such as when trivial layout differences occur due to multiple threads being used to build a single hash join buffer.

==========================
Equijoins vs Non-Equijoins
==========================

TODO

==========
Coalescing
==========

TODO
