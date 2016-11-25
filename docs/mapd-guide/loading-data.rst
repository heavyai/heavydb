Loading Data
============

``COPY FROM``
~~~~~~~~~~~~~

::

    COPY <table> FROM '<file pattern>' [WITH (<property> = value, ...)];

``<file pattern>`` must be local on the server, the file pattern can contain
wildcards if there are multiple files to be loaded.
There is a way to import client-side files (``\copy`` command in mapdql)
but it will be significantly slower. For large files, it is recommended
to first scp the file to the server and then issue the COPY command.

``<property>`` in the optional WITH clause can be:

-  ``delimiter``: a single-character string for the delimiter between
   input fields. The default is ``","``, i.e., as a CSV file.
-  ``nulls``: a string pattern indicating a field is NULL. By default,
   an empty string or ``\N`` means NULL.
-  ``header``: can be either ``'true'`` or ``'false'`` indicating
   whether the input file has a header line in Line 1 that should be
   skipped. The default is ``'true'``.
-  ``escape``: a single-character string for escaping quotes. The
   default is the quote character itself.
-  ``quoted``: ``'true'`` or ``'false'`` indicating whether the input
   file contains quoted fields. The default is ``'true'``.
-  ``quote``: a single-character string for quoting a field. The default
   quote character is double quote ``"``. All characters are inside
   quotes are imported as is except for line delimiters.
-  ``line_delimiter`` a single-character string for terminating each
   line. The default is ``"\n"``.
-  ``array``: a two character string consisting of the start and end characters
   surrounding an array. The default is ``{}``. For example, data to be inserted
   into a table with a string array in the second column (e.g. ``BOOLEAN,
   STRING[], INTEGER``) may be written as ``true,{value1,value2,value3},3``.
-  ``array_delimiter``: a single-character string for the delimiter between
   input values contained within an array. The default is ``","``.
-  ``threads`` number of threads for doing the data importing. The
   default is the number of CPU cores on the system.

Note: by default the CSV parser assumes one row per line. To import a
file with multiple lines in a single field, specify ``threads = 1`` in
the ``WITH`` clause.

Example:

::

    COPY tweets from '/tmp/tweets.csv' WITH (nulls = 'NA');
    COPY tweets from '/tmp/tweets.tsv' WITH (delimiter = '\t', quoted = 'false');

``SQL Importer``
~~~~~~~~~~~~~~~~

::

    java -cp [MapD JDBC driver]:[3rd party JDBC driver]
    com.mapd.utility.SQLImporter -t [MapD table name] -su [external source user]
    -sp [external source password] -c "jdbc:[external
    source]://server:port;DatabaseName=some_database" -ss "[select statement]"

::

	usage: SQLImporter
	-b,--bufferSize <arg>      transfer buffer size
	-c,--jdbcConnect <arg>     JDBC Connection string
 	-d,--driver <arg>          JDBC driver class
 	-db,--database <arg>       MapD Database
 	-f,--fragmentSize <arg>    table fragment size
 	-p,--passwd <arg>          MapD Password
    	--port <arg>               MapD Port
 	-r <arg>                   Row Load Limit
 	-s,--server <arg>          MapD Server
 	-sp,--sourcePasswd <arg>   Source Password
 	-ss,--sqlStmt <arg>        SQL Select statement
 	-su,--sourceUser <arg>     Source User
 	-t,--targetTable <arg>     MapD Target Table
 	-tr,--truncate             Truncate table if it exists
 	-u,--user <arg>            MapD User



SQL Importer executes a select statement on another database via JDBC
and brings the result set into MapD.

If the table doesn't exist it will create the table in MapD.

If the truncate flag is set it will truncate the contents in the file.

If the file exists and truncate is not set it will fail if the table
does not match the SELECT statements metadata.

It is recommended to use a service account with read-only permissions
when accessing data from a remote database.

MySQL Example:

::

	java -cp mapd-1.0-SNAPSHOT-jar-with-dependencies.jar:
	mysql/mysql-connector-java-5.1.38/mysql-connector-java-5.1.38-bin.jar 
	com.mapd.utility.SQLImporter -t test1 -sp mypassword -su myuser 
	-c jdbc:mysql://localhost -ss "select * from employees.employees"

SQLServer Example:

::

    java -cp
    /path/to/mapd/bin/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar:/path/to/sqljdbc4.jar
    com.mapd.utility.SQLImporter -d com.microsoft.sqlserver.jdbc.SQLServerDriver -t
    mapd_target_table -su source_user -sp source_pwd -c
    "jdbc:sqlserver://server:port;DatabaseName=some_database" -ss "select top 10 *
    from dbo.some_table"

PostgreSQL Example:

::

    java -cp
    /p/to/mapd/bin/mapd-1.0-SNAPSHOT-jar-with-dependencies.jar:
    /p/to/postgresql-9.4.1208.jre6.jar
    com.mapd.utility.SQLImporter -t mapd_target_table -su source_user -sp
    source_pwd -c "jdbc:postgresql://server/database" -ss "select * from some_table
    where transaction_date > '2014-01-01'"

``StreamInsert``
~~~~~~~~~~~~~~~~

::

    <data stream> | StreamInsert <table> <mapd database> --host <localhost> --port 9091
    -u <mapd_user> -p <mapd_pwd> --delim '\t' --batch 1000

Stream data into MapD by attaching the StreamInsert program onto the end
of a data stream. The data stream could be another program printing to
standard out, a Kafka endpoint, or any other real-time stream output.
Users may specify the appropriate batch size according to the expected
stream rates and desired insert frequency. The MapD target table must
already exist before attempting to stream data into the table.

Example:

::

    cat file.tsv | /path/to/mapd/SampleCode/StreamInsert stream_example mapd --host localhost
    --port 9091 -u mapd -p MapDRocks!  --delim '\t' --batch 1000

``HDFS``
~~~~~~~~

Consume a CSV or Parquet file residing in HDFS into MapD

Copy the MapD JDBC driver into the sqoop lib, normally
/usr/lib/sqoop/lib/

Example:

::

    sqoop-export --table alltypes --export-dir /user/cloudera/ \
      --connect "jdbc:mapd:192.168.122.1:9091:mapd" \
      --driver com.mapd.jdbc.MapDDriver --username mapd \
      --password HyperInteractive --direct --batch
