Migration is performed in Omnisci whenever it is necessary to update or revert existing data in SQLite database. 
Each SQLite database has a ``mapd_version_history`` table which is used internally to verify if migration was performed or needs to be performed.
``mapd_record_ownership_marker`` table holds the DB owner privileges from older databases.