A fragment describes a horizontal partition of a set of rows in a table. A set
of fragments makes up a table (see the diagram below). By default, each
fragment consists of 32 million records. Fragment size is configurable when
the table is created (see
`CREATE TABLE <https://docs.nvidia.com/heavyai/sql/data-definition-ddl/tables#create-table>`_).


.. image:: /img/fragment.png

