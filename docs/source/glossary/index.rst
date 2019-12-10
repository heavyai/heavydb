.. OmniSciDB Glossary

=================
Glossary of Terms
=================

.. glossary::

    kernel
        A public function which describes one or more transformations on a set of data.  

    expression
        Typically refers to a SQL expression; takes as input one or more columns and, through a series of operations, produces either a scalar value or a column value. 

    target
        An expression whose value is returned to the user as part of a projection. 

    fragment
        A row-wise striping across all columns in a table in a fixed unit (32,000,000 rows by default). See :doc:`../data_model/columnar_layout` for more information. 
    
    chunk
        The intersection of a :term:`fragment` and a column. A single chunk is a series of rows from one column in a table and is uniquely identified by the `chunk key`, a tuple consisting of database ID, table ID, column ID, and fragment ID. 
