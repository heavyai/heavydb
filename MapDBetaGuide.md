#MapD Quick Guide (Release 0.1 BETA)
##Getting Started
Assuming `$MAPDHOME` is the directory where MapD software is installed, make sure that `$MAPDHOME/bin` is in PATH.

###initdb
The very first step before using MapD is to run initdb:
```
initdb [-f] <MapD Directory>
```
initializes the MapD directory. It creates three subdirectories:

* mapd_catalogs: stores MapD catalogs
* mapd_data: stores MapD data
* mapd_log: contains all MapD log files. MapD uses [glog](https://code.google.com/p/google-glog/) for logging.

The `-f` flag forces `initdb` to overwrite existing data and catalogs in the specified directory.

###mapd_server

```
mapd_server <MapD directory> [--cpu|--gpu|--hybrid]
                             [{-p|--port} <port number>]
                             [--flush-log]
                             [--version|-v]
```
This command starts the MapD Server process. `<MapD directory>` must match that in the `initdb` command when it was run. The options are:

* `[--cpu|--gpu|--hybrid]`: Execute queries on CPU, GPU or both. The default is GPU.
* `[{-p|--port} <port number>]`: Specify the port number mapd_server listens on. The default is port 9091.
* `[--flush-log]`: Flush log files to disk. Useful for `tail -f` on log files.
* `[--version|-v]`: Prints version number.

`mapd_server` automatically re-spawns itself in case of unexpected termination.  To force termination of `mapd_server` kill -9 all `mapd_server` processes.
###mapd_http_server

```
mapd_http_server [{-p|--port} <port number>]
                 [{-s|--server} <port number>]
                 [--version|-v]
```
This command starts the MapD HTTP proxy server.  It must be started on the same computer after `mapd_server` is started.  MapD's visualization frontend communicates to the MapD Server via the MapD HTTP proxy server. The options are:

* `[{-p|--port} <port number>]`: Specify the port the HTTP proxy server listens on. The default is port 9090.
* `[{-s|--server} <port number>]`: Specify the port number that mapd_server listens on. The default is port 9091.
* `[--version|-v]`: Prints version number.

###mapdql

```
mapdql [<database>]
       [{--user|-u} <user>]
       [{--passwd|-p} <password>]
       [--port <port number>]
       [{-s|--server} <server host>]
```
`mapdql` is the client-side SQL console. All SQL statements can be submitted to the MapD Server and the results are returned and displayed. The options are:

* `[<database>]`: Database to connect to. Not connected to any database if omitted.
* `[{--user|-u} <user>]`: User to connect as. Not connected to MapD Server if omitted.
* `[{--passwd|-p} <password>]`: User password. *Will not be in clear-text in production release*.
* `[--port <port number>]`: Port number of MapD Server. The default is port 9091.
* `[{--server|-s} <server host>]`: MapD Server hostname in DNS name or IP address. The default is localhost.

In addition to SQL statements `mapdql` also accepts the following list of backslash commands:

* `\h`: List all available backslash commands.
* `\u`: List all users.
* `\l`: List all databases.
* `\t`: List all tables.
* `\d <table>`: List all columns of table.
* `\c <database> <user> <password>`: Connect to a new database.
* `\gpu`: Switch to GPU mode.
* `\cpu`: Switch to CPU mode.
* `\hybrid`: Switch to Hybrid mode.
* `\timing`: Print timing information.
* `\notiming`: Do not print timing information.
* `\version`: Print MapD Server version.
* `\copy <file path> <table>`: Copy data from file on client side to table. The file is assumed to be in CSV format unless the file name ends with `.tsv`.
* `\q`: Quit.

`mapdql` automatically attempts to reconnect to `mapd_server` in case it restarts due to crashes or human intervention.  There is no need to restart or reconnect.
##Users and Databases

Users and databases can only be manipulated when connected to database mapd as a super user.

###CREATE USER

```
CREATE USER <name> (<property> = value, ...);
```
Example:
```
CREATE USER jason (password = 'MapDRocks!', is_super = 'true');
```
###DROP USER
```
DROP USER <name>;
```
Example:
```
DROP USER jason;
```
###ALTER USER
```
ALTER USER <name> (<property> = value, ...);
```
Example:
```
ALTER USER jason (is_super = 'false', password = 'SilkySmooth');
```
###CREATE DATABASE
```
CREATE DATABASE <name> (<property> = value, ...);
```
Example:
```
CREATE DATABASE test (owner = 'jason');
```
###DROP DATABASE
```
DROP DATABASE <name>;
```
Example:
```
DROP DATABASE test;
```

##Tables

###CREATE TABLE

```
CREATE TABLE [IF NOT EXISTS] <table>
  (<column> <type> [NOT NULL] [ENCODING <encoding spec>], ...)
  [WITH (<property> = value, ...)];
```

`<type>` supported include:

* BOOLEAN
* SMALLINT
* INT[EGER]
* BIGINT
* FLOAT | REAL
* DOUBLE [PRECISION]
* [VAR]CHAR(length)
* TEXT
* TIME
* TIMESTAMP
* DATE

`<encoding spec>` supported include:

* DICT: Dictionary encoding on string columns.
* FIXED(bits): Fixed length encoding of integer or timestamp columns.

The `<property>` in the optional WITH clause can be

* `fragment_size`: number of rows per fragment which is a unit of the table for query processing. It defaults to 8 million rows and is not expected to be changed.
* `page_size`: number of bytes for an I/O page. This defaults to 1MB and does not need to be changed.

Example:
```
CREATE TABLE IF NOT EXISTS tweets (
  tweet_id BIGINT NOT NULL, 
  tweet_time TIMESTAMP NOT NULL ENCODING FIXED(32), 
  lat REAL, 
  lon REAL, 
  sender_id BIGINT NOT NULL,
  sender_name TEXT NOT NULL ENCODING DICT, 
  location TEXT ENCODING DICT, 
  source TEXT ENCODING DICT, 
  reply_to_user_id BIGINT, 
  reply_to_tweet_id BIGINT, 
  lang TEXT ENCODING DICT, 
  followers INT, 
  followees INT, 
  tweet_count INT, 
  join_time TIMESTAMP ENCODING FIXED(32), 
  tweet_text TEXT, 
  state TEXT ENCODING DICT, 
  county TEXT ENCODING DICT, 
  place_name TEXT, 
  state_abbr TEXT ENCODING DICT, 
  county_state TEXT ENCODING DICT, 
  origin TEXT ENCODING DICT);
```
###DROP TABLE
```
DROP TABLE [IF EXISTS] <table>;
```
Example:
```
DROP TABLE IF EXISTS tweets;
```
###COPY
```
COPY <table> FROM '<file path>' [WITH (<property> = value, ...)];
```
`<file path>` must be a path on the server. There is a way to import client-side files (`\copy` command in mapdql) but it will be significantly slower. For large files, it is recommended to first scp the file to the server and then issue the COPY command.

`<property>` in the optional WITH clause can be:

* `delimiter`: a single-character string for the delimiter between input fields. The default is `","`, i.e., as a CSV file.
* `nulls`: a string pattern indicating a field is NULL. By default, an empty string means NULL.
* `header`: can be either `'true'` or `'false'` indicating whether the input file has a header line in Line 1 that should be skipped.
* `escape`: a single-character string for escaping quotes. The default is the quote character itself.
* `quoted`: `'true'` or `'false'` indicating whether the input file contains quoted fields.  The default is `'false'`.
* `quote`: a single-character string for quoting a field. The default quote character is double quote `"`. All characters are inside quotes are imported as is except for line delimiters.
* `line_delimiter` a single-character string for terminating each line. The default is `"\n"`.
* `threads` number of threads for doing the data importing.  The default is the number of CPU cores on the system.

Example:
```
COPY tweets from '/tmp/tweets.csv' WITH (nulls = 'NA');
COPY tweets from '/tmp/tweets.tsv' WITH (delimiter = '\t', quoted = 'false');
```
##DML
###INSERT
```
INSERT INTO <table> VALUES (value, ...);
```
This statement is used for one row at a time ad-hoc inserts and should not be used for inserting a large number of rows. The COPY command should be used instead which is far more efficient.
Example:
```
CREATE TABLE foo (a INT, b FLOAT, c TEXT, d TIMESTAMP);
INSERT INTO foo VALUES (NULL, 3.1415, 'xyz', '2015-05-11 211720`);
```
###SELECT
```
SELECT [ALL|DISTINCT] <expr> [AS [<alias>]], ... FROM <table>
  [WHERE <expr>]
  [GROUP BY <expr>, ...]
  [HAVING <expr>]
  [ORDER BY <expr>, ...]
  [LIMIT {<number>|ALL} [OFFSET <number> [ROWS]]];
```
It supports all the common SELECT features except for the following temporary limitations:

* Only a single table is allowed in FROM clause.
* Subqueries are not supported.

##Client Interfaces

MapD uses [Apache Thrift](https://thrift.apache.org) to generate client-side interfaces.  The *interface definitions* are in `$MAPDHOME/mapd.thrift`.  See Apache Thrift documentation on how to generate client-side interfaces for different programming languages with Thrift.  Also see `$MAPDHOME/samples` for sample client code.
