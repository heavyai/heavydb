.. OmniSciDB Data Model

========================================
DUMP TABLE and RESTORE TABLE Statements
========================================

**DUMP TABLE** statement archives data files and string dictionary files of a table to a gzip-compressed tar file.

**RESTORE TABLE** statement restores or migrates data files and string dictionary files of a table from a gzip-compressed tar file.

Syntax
==================

DUMP TABLE *table* TO *'tgz_file_path`*;

RESTORE TABLE *table* FROM *'tgz_file_path`*;

Note: When table *table* does not exist in current database, RESTORE TABLE creates a new table named *table* and migrates the table files in *tgz_file_path* to the table. 


File Format
==================

Inside the tar archive, **native** Omnisci file formats are reused instead of inventing new ones for dumping table data and dictionary files. The native formats are proven simple and solid and so is existing code that accesses files of the formats.

Files in native OmniSci formats are easy to accommodate onto target database where they are to be restored or migrated. In rare cases when rows archived in files need to be refragmented or redistributed (e.g, need to be resharded) on target database, there is no problem on accessing the rows belonging to individual fragments stored in data files, thanks to the existing code mentioned above.

Note:

- Using native Omnisci file formats means that DUMP TABLE archives metadata and data pages of **all versions** by default. Archives may consume more disk storage space than necessary for latest pages. To minimize storage usage and speed up later table restore, if a table has experienced a signficant number of UPDATE/DELETE/ INSERT of small batches of data thus leaving excessive amount of old pages in data files, it is recommended to optimize the table (in place) beforehand, for example by running **OPTIMIZE TABLE** statement with (VACUUM='**PAGES**') option to trim old pages in the data files (when the option becomes available). Methods that trim pages not insitu would require extra storage to buffer pages before they are archived.
- Currently there is no support of resharding or refragmentization, for example on migrating a 3-shard table to become 4-shard. It is doable in next releases if necessary.


DUMP TABLE
==================

Besides data files and dictionary files of a table, DUMP TABLE creates and includes the following files in the tar file:

- **_table.sql** - contains table schema in a SQL **CREATE TABLE** statement which will be used to create a new table when migrating the table to another database using **RESTORE TABLE** statement.
- **_table.oldinfo** - contains table information that is used to migrate the table. The information consists of:

  - column Ids of the dumped table (aka. "source" table)
  - dictionary file paths which assist in rebuilding dictionary references on table migration
  
- **_table.epoch** - contains current epoch of the table.

File **_table.oldinfo** contains a list of space delimited strings, each of which corresponds to a column in the table and is formatted as *column_name*:*column_id*:*dictionary_path*. For none text or none encoded text column, *dictionary_path* is a empty string.

For example, for a table *t* which was created using the following SQL statement::

  CREATE TABLE t(i int,s text,d text,f text, SHARED DICTIONARY (d) REFERENCES t(s), 
  SHARED DICTIONARY (f) REFERENCES s(s), SHARD KEY(i)) WITH (FRAGMENT_SIZE=10, SHARD_COUNT=2);
  
executing the following DUMP TABLE statement::

  DUMP TABLE t TO '/tmp/Orz_.tgz';
  
creates the tar file **/tmp/Orz.tgz** consisting of the following files::

	$tar xvfz /tmp/Orz.tgz 
	_table.sql
	_table.oldinfo
	_table.epoch
	table_1_5/
	table_1_5/0.2097152.mapd
	table_1_5/epoch
	table_1_5/1.4096.mapd
	table_1_6/
	table_1_6/0.2097152.mapd
	table_1_6/epoch
	table_1_6/1.4096.mapd
	DB_1_DICT_2/
	DB_1_DICT_2/DictOffsets
	DB_1_DICT_2/DictPayload
	DB_1_DICT_1/
	DB_1_DICT_1/DictOffsets
	DB_1_DICT_1/DictPayload

	$ cat _table.sql
	CREATE TABLE @T (i INTEGER, j INTEGER, s TEXT ENCODING DICT(32), d TEXT, f TEXT ENCODING DICT(32), SHARED DICTIONARY (d) REFERENCES @T(s), SHARD KEY(i)) WITH (FRAGMENT_SIZE=10, MAX_CHUNK_SIZE=1073741824, PAGE_SIZE=2097152, MAX_ROWS=4611686018427387904, VACUUM='DELAYED', SHARD_COUNT=2); 
	
	$ cat _table.oldinfo
	i:1: j:2: s:3:DB_1_DICT_2 d:4:DB_1_DICT_2 f:5:DB_1_DICT_1 rowid:6: $deleted$:7: 
	
	$ cat _table.epoch
	50


RESTORE TABLE
==================

On executing a **RESTORE TABLE** statement, OmniSci core will accommodate those data and dict files archived in the input tar file to base directory of current database. 

When restoring a table (ie. the table exists in current database) data and dict files in input tar file directly replace the counterparts in base directory of current database. No adjustment to the files are necessary.

**Note:** To facilitate rollback in case of any exception during the course of processing the statement, existing data and dict files are backed up to a temporary directory and are restored when any exception happens.

Omnisci core processing **table migration** (ie. the table doesn't exists) in current database, handling is similar to the way it processes table recovery, except that the numbering (ie. table id) of data and dict files in current database can differ from that of those files in the tar file, and except in the case when schema of the table had been altered through any **ALTER TABLE ADD/DELETE COLUMN** statement before it was archived, and hence column IDs can differ between source and target or destination tables. 

For the former exception above, core renames input data and dict files accordingly. For the latter, core will scan all data files of the table and adjust column ids in related chunk headers.

Overall, core follows these steps to process RESTORE TABLE statements:

- reads in source table schema (ref. file **_table.sql**)
- (table migration only) creates the table in current database
- grabs WRITE lock on the table and READ lock on catalog
- checks schema compatibility between source and target tables
- builds a map of column IDs between source and target tables to check whether source table had been altered and for later adjustment of chunk headers in case the table had been altered. (ref. file **_table.oldinfo**)
- builds a map of dict file paths between source and target tables for later rename of source dict files. (ref. file **_table.oldinfo**)
- untars the tar file to a temporary directory
- adjusts chunk headers if source table had been altered 
- backs up (move only; not copy) existing data and dict files of the table to another temporary directory
- renames data files of the table
- renames dict files of the table
- set table epoch of the target table to that of the source table (ref. file **_table.epoch**)  

In case of runtime exception during processing, existing data and dict files are restored. 
