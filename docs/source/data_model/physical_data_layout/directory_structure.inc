Each table has its own data directory under the MapD Data directory.  The directory takes the form

``mapd_data/table_<db_id>_<table_id>``

An example directory would be

``mapd_data/table_1_1``

This directory would contain the files for the table identified by **database id 1 table id 1** in the catalog.  This directory will also contain meta data files and the epoch file for the table.

Epoch
-----

MapD implements recovery and rollback via the notion of an epoch. The epoch is a monotonically incrementing number, as changes are made to any file the epoch of the change is noted in the header of the changed page. When all the work is done to update the file layer a `checkpoint` is called and the current epoch is updated in its own epoch file.

When the database starts up the epoch file is read and that is considered the DB’s current epoch. When the headers of any pages are read the epoch must be less than the DB’s epoch. If not the page is considered not to have been check pointed so is `unsafe` and so is discarded.

The epoch for the table is stored in a single file called **epoch** in the table directory.
