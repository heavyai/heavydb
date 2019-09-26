A fragment represents a horizontal partition of a set of rows in a table.  A set of fragments makes up a table (see Diagram below). Fragment size by default is 32M records, but can be configured as part of the CREATE TABLE stmt.

[Note: Fragments are made up of smaller units (chunks) which have a maximum size in bytes; in some instance their size can reduce the actual number of rows in a given fragment]


.. image:: /img/fragment.png

