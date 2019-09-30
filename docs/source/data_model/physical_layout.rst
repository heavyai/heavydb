.. OmniSciDB Data Model

==================================
Physical Data Layout
==================================
OmniSciDB includes a full-featured storage layer to manage the persistence and modification of table data stored on disk.

Data on disk is organized into metadata pages and data multipages. The `BufferMgr` class manages data in each level of the memory hierarchy, with data on disk considered the "lowest" level. Specifically, the `FileMgr` mananges loading data from disk and flushing data back to disk during inserts, updates, and deletes. Initially, a single `GlobalFileMgr` is created to serve as the entry point for all file management. In turn, the `GlobalFileMgr` has a child file manager for each table in the current database (see diagram :ref:`file-manager-structure`).

.. figure:: /img/DataMgr.png
   :name: file-manager-structure
   :align: center

   File Manager Object Hierarchy

Directory Structure
===================

The OmniSciDB data directory contains a `mapd_data` folder which stores the physical data pages for each table. Everytime a table is created, a new folder is created in `mapd_data` identified with the **table_id** and **database_id** uniquely representing each table in the system. The directory name takes the following form:

``mapd_data/table_<db_id>_<table_id>``

E.g. for table 1, db 1:

``mapd_data/table_1_1``

Within the data directory, data is stored in multipage files which vary in number, size, and makeup depending on the width, row count, and insert / update / delete activity for the table.

Epoch
-----

OmniSciDB implements recovery and rollback via an `epoch`. The `epoch` is a monotonically incrementing integer starting from 0. As changes are made to a table, the epoch is incremented. Each change creates a new data page. The header for each data page contains to the `epoch` for that change. `Epoch` values are incremented at the start of any job which modifies data on disk (i.e. adds data pages). Sometimes, multiple pages will be written for the same `epoch` value (e.g. with bulk inserts). Once the work is considered complete, the incremented `epoch` is committed and flushed to the `epoch` file in the data directory via calling `checkpoint` in the storage layer. If a job fails before checkpointing, the previous `epoch` value is used and pages with `epoch` values higher than the last committed value are ignored and overwritten. 

Data Multipages
================

Table data is stored in **data multipages** in the data directory. The naming format for a data multipage file is ``<file_number>.<page_size>.mapd``. Consider a file with the filename ``0.2097152.mapd``.  This is file number **0** and it has a *page_size* of **2097152 bytes** (the default page size).

Each multipage file consists of **256** pages. Thus, a file with the defualt page size will be **512MB (2M page_size x 256 pages)** on disk. When a new file is created the entire file is written and zeroed, regardless of how many records are actually stored.

Internally each page consists of a header and the raw, serialized data. The header and data formats for meta data files is the same as the format for data files; only the payload differs. The diagram below (:ref:`internal-file-format`) illustrates the internal format of a data file. Note that the **DB** and **Table** IDs of the :ref:`chunk-key-label` may be overloaded, as the `DB` and `Table` information is specified by the `GlobalFileMgr` during load. 

A 'page' directly corresponds to an in-memory `Chunk` (see :ref:`chunk-label`).

Example Data Page:
------------------
Consider the following table:

.. code-block:: sql

    CREATE TABLE t ( c1 SMALLINT, c2 INTEGER );

The `create` command will create a new directory for the table and populate it with a data file containing 256 pages. Three of the pages will contain data and a valid epoch:  one for each column and one for the 'hidden' delete column.

.. figure:: /img/datapage.png
   :name: internal-file-format
   :align: center

   Data File Internal Format

Metadata Pages
===============

Table metadata is stored in a metadata multipage file (or multiple files). Metadata pages contain metadata information for each data page in the data multipage files. By default, these files have a `page_size` of **4096** and will appear in the data directory using the same naming format for data multipages, e.g. ``<file_number>.4096.mapd``. Each file is 16 MB on disk (4096 bytes x 4096 pages).

Metadata pages include a header much like a datafile, but with a fixed *page_id* of **-1** for each page. The page ID of **-1** identifies a metadata page. Chunk metadata is stored in the metadata pages, and a new metadata page is written out for a chunk each time the chunk contents change; the current metadata page for a chunk is the one with the highest valid epoch.
