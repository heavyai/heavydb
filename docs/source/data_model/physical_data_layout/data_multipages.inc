Table data can be stored in one to many data files on disk depending on the tables size. The naming format for a data file is ``<file_number>.<page_size>.mapd``; for example ``0.2097152.mapd``.  This is file number **0** and it has a *page_size* of **2097152 bytes** (the default page size).

Each file has **256** pages per file. A default *page_size* file will be **512M (2M page_size x 256 pages)** on disk. When a new file is created the entire file is written and zeroed, regardless of how many records are actually stored.

Internally each page consists of a header records and data.  The header and data formats for meta data files is the same as the format for data files. The diagram below (:ref:`internal-file-format`) illustrates the internal format of a data file.

The 'page' is the subunit of a file that corresponds to a 'chunk' in memory.  A file representing a table defined by  the command `create table t(c1 SMALLINT c2 INTEGER)` which contains 1 row, will contain 256 pages of which 3 will have had data written to them; one for each column and one for the 'hidden' delete column.


.. figure:: /img/datapage.png
   :name: internal-file-format
   :align: center

   Data File Internal Format

