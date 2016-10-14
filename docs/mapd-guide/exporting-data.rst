Exporting Data
==============

``COPY TO``
~~~~~~~~~~~

::

    COPY ( <SELECT statement> ) TO '<file path>' [WITH (<property> = value, ...)];

``<file path>`` must be a path on the server. This command exports the
results of any SELECT statement to the file. There is a special mode
when ``<file path>`` is empty. In that case, the server automatically
generates a file in ``<MapD Directory>/mapd_export`` that is the client
session id with the suffix ``.txt``.

``<property>`` in the optional WITH clause can be:

-  ``delimiter``: a single-character string for the delimiter between
   column values. The default is ``","``, i.e., as a CSV file.
-  ``nulls``: a string pattern indicating a field is NULL. The default
   is ``\N``.
-  ``escape``: a single-character string for escaping quotes. The
   default is the quote character itself.
-  ``quoted``: ``'true'`` or ``'false'`` indicating whether all the
   column values should be output in quotes. The default is ``'false'``.
-  ``quote``: a single-character string for quoting a column value. The
   default quote character is double quote ``"``.
-  ``line_delimiter`` a single-character string for terminating each
   line. The default is ``"\n"``.
-  ``header``: ``'true'`` or ``'false'`` indicating whether to output a
   header line for all the column names. The default is ``'true'``.

Example:

::

    COPY (SELECT * FROM tweets) TO '/tmp/tweets.csv';
    COPY (SELECT * tweets ORDER BY tweet_time LIMIT 10000) TO
      '/tmp/tweets.tsv' WITH (delimiter = '\t', quoted = 'true', header = 'false');
