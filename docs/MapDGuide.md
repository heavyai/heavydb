---
title: MapD Guide | Release 0.1
author: MapD Technologies, Inc.
logo: mapd-logo.pdf
---

# Dependencies
MapD is distributed as a group of mostly statically-linked executables, which minimizes the number of dependencies required. The following are the minimum requirements for running MapD.

Operating Systems

* CentOS/RHEL 7.0 or later. CentOS/RHEL 6.x builds may be provided upon request, but are not supported.
* Ubuntu 15.04 or later.
* Mac OS X 10.9 or later, on 2013 or later hardware. Builds which support 2012 hardware may be provided upon request, but are not supported.

Libraries

* CUDA 7.0 or later. Basic installation instructions are provided below.

# Terminology

Environment variables:

* `$MAPD_PATH`: MapD install directory, e.g. `/opt/mapd/mapd2`
* `$MAPD_DATA`: MapD data directory, e.g. `/var/lib/mapd/data`

Programs and scripts:

* `mapd_server`: MapD database server. Located at `$MAPD_PATH/bin/mapd_server`.
* `mapd_web_server`: Web server which hosts the web-based frontend and provides database access over HTTP(S). Located at `$MAPD_PATH/bin/mapd_web_server`.
* `initdb`: Initializes the MapD data directory. Located at `$MAPD_PATH/bin/initdb`.
* `mapdql`: Command line-based program that gives direct access to the database. Located at `$MAPD_PATH/bin/mapdql`.
* `startmapd`: All-in-one script that will initialize a MapD data directory at `$MAPD_PATH/data`, offer to load a sample dataset, and then start the MapD server and web server. Located at `$MAPD_PATH/startmapd`.

Other

* `systemd`: init system used by most major Linux distributions. Sample `systemd` target files for starting MapD are provided in `$MAPD_PATH/systemd`.

# Installation

## CUDA Installation
CUDA-enabled installations of MapD depends on `libcuda` which is provided by the NVIDIA GPU drivers and NVIDIA CUDA Toolkit, respectively. As of January 2016, both CUDA 7.0 and CUDA 7.5 are supported by MapD.

The NVIDIA CUDA Toolkit, which includes the NVIDIA GPU drivers, is available from the [NVIDIA CUDA Zone](https://developer.nvidia.com/cuda-downloads).

The installation notes below are just a summary of what is required to install the CUDA Toolkit. Please see the [CUDA Quick Start Guide](http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Quick_Start_Guide.pdf) for full instructions.

Before proceeding, please make sure your system is completely up-to-date and you have restarted to activate the latest kernel, etc.

### CentOS / Red Hat Enterprise Linux (RHEL)
Please download the RPM package provided by NVIDIA from the [NVIDIA CUDA Zone](https://developer.nvidia.com/cuda-downloads).

RHEL-based distributions require Dynamic Kernel Module Support (DKMS) in order to build the GPU driver kernel modules, which is provided by the Extra Packages for Enterprise Linux (EPEL) repository. See the [EPEL website](https://fedoraproject.org/wiki/EPEL) for complete instructions for enabling this repository.

1. Enable EPEL
```
sudo yum install epel-release
```

2. Install GCC and Linux headers
```
sudo yum groupinstall "Development Tools"
sudo yum install linux-headers
```

3. Install the CUDA repository, update local repository cache, and then install the CUDA Toolkit and GPU drivers
```
sudo rpm --install cuda-repo-<distro>-<version>.<architecture>.rpm
sudo yum clean expire-cache
sudo yum install cuda
```

### Ubuntu / Debian
Please download the DEB package provided by NVIDIA from the [NVIDIA CUDA Zone](https://developer.nvidia.com/cuda-downloads).

Install the CUDA repository, update local repository cache, and then install the CUDA Toolkit and GPU drivers
```
sudo dpkg --install cuda-repo-<distro>-<version>.<architecture>.deb
sudo apt-get update
sudo apt-get install cuda
```

### Mac OS X
Please download the DMG package provided by NVIDIA from the [NVIDIA CUDA Zone](https://developer.nvidia.com/cuda-downloads).

The DMG package will walk you through all required steps to install CUDA.

### Environment Variables
MapD depends on `libcuda`, which must be available in your environment in order to run MapD. The NVIDIA GPU drivers usually make `libcuda` available by default by installing it to a system-wide `lib` directory such as `/usr/lib64` (on CentOS/RHEL) or `/usr/lib/x86_64-linux-gnu` (on Ubuntu).

### Verifying Installation
After installing CUDA and setting up the environment variables, please restart your machine to activate the GPU drivers.

On Linux, you can verify installation of the GPU drivers by running `nvidia-smi`.

## MapD Installation
MapD is distributed as a self-extracting archive, that is, a shell script which contains the complete contents of a .tar.gz file. Other package types are available upon request.

To install, move the archive to the desired installation directory (`$MAPD_PATH`) and run:
```
sh mapd2-<date>-<hash>-<platform>-<architecture>.sh
```
replacing `mapd2-<date>-<hash>-<platform>-<architecture>.sh` with the name of the archive provided to you.

The installer will then present the EULA and, if accepted, ask for the installation path. 

### `systemd`
For Linux, the MapD archive includes `systemd` target files which allows `systemd` to manage MapD as a service on your server. The provided `install_mapd_systemd.sh` script will ask a few questions about your environment and then install the target files into the correct location.

```
cd $MAPD_PATH/systemd
./install_mapd_systemd.sh
```

# Configuration
Before starting MapD, the `data` directory must be initialized. To do so, create an empty directory at the desired path (`/var/lib/mapd/data`) and run `$MAPD_PATH/bin/initdb` with that path as the argument. For example:

```
sudo mkdir -p /var/lib/mapd/data
sudo $MAPD_PATH/bin/initdb /var/lib/mapd/data
```

Finally, make sure this directory is owned by the user that will be running MapD (i.e. `mapd`):
```
sudo chown -R mapd /var/lib/mapd
```

You can now test your installation of MapD with the `startmapd` script:
```
$MAPD_PATH/startmapd --data $MAPD_DATA
```

## Configuration file
MapD also supports storing options in a configuration file. This is useful if, for example, you need to run the MapD database and/or web servers on different ports than the default. An example configuration file is provided under `$MAPD_PATH/mapd.conf.sample`.

To use options provided in this file, provide the path the the config file to the `--config` flag of `startmapd` or `mapd_server` and `mapd_web_server`. For example:
```
$MAPD_PATH/startmapd --config $MAPD_DATA/mapd.conf
```

# Starting and Stopping MapD Services
MapD consists of two system services: `mapd_server` and `mapd_web_server`. These services may be started individually or run via the interactive script `startmapd`. For permanent installations, it is recommended that you use `systemd` to manage the MapD services.

## MapD Via `startmapd`
MapD may be run via the `startmapd` script provided in `$MAPD_PATH/startmapd`. This script handles creating the `data` directory if it does not exist, inserting a sample dataset if desired, and starting both `mapd_server` and `mapd_web_server`.

### Starting MapD Via `startmapd`
To use `startmapd` to start MapD, run:
```
$MAPD_PATH/startmapd --config /path/to/mapd.conf
```
if using a configuration file, or
```
$MAPD_PATH/startmapd --data $MAPD_DATA
```
to explicitly specify the `$MAPD_DATA` directory.

### Stopping MapD Via `startmapd`
To stop an instance of MapD that was started with the `startmapd` script, simply kill the `startmapd` process via `CTRL-C` or `pkill startmapd`. You can also use `pkill mapd` to ensure all processes have been killed.

## MapD Via `systemd`
For permenant installations of MapD, it is recommended that you use `systemd` to manage the MapD services. `systemd` automatically handles tasks such as log management, starting the services on restart, and restarting the services in case they die. It is assumed that you have followed the instructions above for installing the `systemd` service unit files for MapD.

### Starting MapD Via `systemd`
To manually start MapD via `systemd`, run:
```
systemctl start mapd_server
systemctl start mapd_web_server
```

### Stopping MapD Via `systemd`
To manually stop MapD via `systemd`, run:
```
systemctl stop mapd_server
systemctl stop mapd_web_server
```

### Enabling MapD on Startup
To enable the MapD services to be started on restart, run:
```
systemctl enable mapd_server
systemctl enable mapd_web_server
```

## MapD Service Details
Assuming `$MAPD_PATH` is the directory where MapD software is installed, make sure that `$MAPD_PATH/bin` is in `PATH`.

### `initdb`
The very first step before using MapD is to run initdb:
```
initdb [-f] $MAPD_DATA
```
initializes the MapD data directory. It creates three subdirectories:

* `mapd_catalogs`: stores MapD catalogs
* `mapd_data`: stores MapD data
* `mapd_log`: contains all MapD log files. MapD uses [glog](https://code.google.com/p/google-glog/) for logging.

The `-f` flag forces `initdb` to overwrite existing data and catalogs in the specified directory.

### `mapd_server`

```
mapd_server $MAPD_DATA [--cpu|--gpu|--hybrid]
                       [{-p|--port} <port number>]
                       [--flush-log]
                       [--version|-v]
```
This command starts the MapD Server process. `$MAPD_DATA` must match that in the `initdb` command when it was run. The options are:

* `[--cpu|--gpu|--hybrid]`: Execute queries on CPU, GPU or both. The default is GPU.
* `[{-p|--port} <port number>]`: Specify the port number mapd_server listens on. The default is port 9091.
* `[{--http-port} <port number>]`: Specify the port the HTTP server listens on. The default is port 9090.
* `[--flush-log]`: Flush log files to disk. Useful for `tail -f` on log files.
* `[--version|-v]`: Prints version number.

`mapd_server` automatically re-spawns itself in case of unexpected termination.  To force termination of `mapd_server` kill -9 **all** `mapd_server` processes.

### `mapd_web_server`

```
mapd_web_server [{--port} <port number>]
                [{--proxy-backend} <bool>]
                [{--backend-url} <backend URL>]
                [{--frontend} <path/to/frontend>]
```
This command starts the MapD web server.  This server provides access to MapD's visualization frontend and allows the frontend to communicate with the MapD Server. The options are:

* `[{--port} <port number>]`: Specify the port the web server listens on. The default is port 9092.
* `[{--proxy-backend} <bool>]`: Specify whether to act as a proxy to the backend. The default is `true`.
* `[{--backend-url} <backend URL>]`: Specify the URL to the backend HTTP server. The default is `http://localhost:9090`.
* `[{--frontend} <path/to/frontend>]`: Specify the path to the frontend directory. The default is `frontend`.

### `mapdql`

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
* `\gpu`: Switch to GPU mode in the current session.
* `\cpu`: Switch to CPU mode in the current session.
* `\hybrid`: Switch to Hybrid mode in the current session.
* `\timing`: Print timing information.
* `\notiming`: Do not print timing information.
* `\version`: Print MapD Server version.
* `\copy <file path> <table>`: Copy data from file on client side to table. The file is assumed to be in CSV format unless the file name ends with `.tsv`.
* `\q`: Quit.

`mapdql` automatically attempts to reconnect to `mapd_server` in case it restarts due to crashes or human intervention.  There is no need to restart or reconnect.

# Users and Databases

Users and databases can only be manipulated when connected to the MapD system database ``mapd`` as a super user.  MapD ships with a default super user named ``mapd`` with default password ``HyperInteractive``.

## `CREATE USER`

```
CREATE USER <name> (<property> = value, ...);
```
Example:
```
CREATE USER jason (password = 'MapDRocks!', is_super = 'true');
```

## `DROP USER`
```
DROP USER <name>;
```
Example:
```
DROP USER jason;
```

## `ALTER USER`
```
ALTER USER <name> (<property> = value, ...);
```
Example:
```
ALTER USER mapd (password = 'MapDIsFast!');
ALTER USER jason (is_super = 'false', password = 'SilkySmooth');
```

## `CREATE DATABASE`
```
CREATE DATABASE <name> (<property> = value, ...);
```
Example:
```
CREATE DATABASE test (owner = 'jason');
```

## `DROP DATABASE`
```
DROP DATABASE <name>;
```
Example:
```
DROP DATABASE test;
```

## Basic Database Security Example
The system db is **mapd**
The superuser is **mapd**

There are two user: **Michael** and **Nagesh**

There are two Databases: **db1** and **db2**

Only user **Michael** can see **db1**

Only user **Nagesh** can see **db2**
```
admin@hal:~$ bin/mapdql mapd -u mapd -p HyperInteractive
mapd> create user Michael (password = 'Michael');
mapd> create user Nagesh (password = 'Nagesh');
mapd> create database db1 (owner = 'Michael');
mapd> create database db2 (owner = 'Nagesh');
mapd> \q
User mapd disconnected from database mapd
admin@hal:~$ bin/mapdql db1 -u Nagesh -p Nagesh
User Nagesh is not authorized to access database db1
mapd> \q
admin@hal:~$ bin/mapdql db2 -u Nagesh -p Nagesh
User Nagesh connected to database db2
mapd> \q
User Nagesh disconnected from database db2
admin@hal:~$ bin/mapdql db1 -u Michael -p Michael
User Michael connected to database db1
mapd> \q
User Michael disconnected from database db1
admin@hal:~$ bin/mapdql db2 -u Michael -p Michael
User Michael is not authorized to access database db2
mapd>
```

# Tables

## `CREATE TABLE`

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

## `DROP TABLE`
```
DROP TABLE [IF EXISTS] <table>;
```
Example:
```
DROP TABLE IF EXISTS tweets;
```

## `COPY FROM`
```
COPY <table> FROM '<file path>' [WITH (<property> = value, ...)];
```
`<file path>` must be a path on the server. There is a way to import client-side files (`\copy` command in mapdql) but it will be significantly slower. For large files, it is recommended to first scp the file to the server and then issue the COPY command.

`<property>` in the optional WITH clause can be:

* `delimiter`: a single-character string for the delimiter between input fields. The default is `","`, i.e., as a CSV file.
* `nulls`: a string pattern indicating a field is NULL. By default, an empty string or `\N` means NULL.
* `header`: can be either `'true'` or `'false'` indicating whether the input file has a header line in Line 1 that should be skipped.  The default is `'true'`.
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

## `COPY TO`
```
COPY ( <SELECT statement> ) TO '<file path>' [WITH (<property> = value, ...)];
```
`<file path>` must be a path on the server.  This command exports the results of any SELECT statement to the file.  There is a special mode when `<file path>` is empty.  In that case, the server automatically generates a file in `<MapD Directory>/mapd_export` that is the client session id with the suffix `.txt`.

`<property>` in the optional WITH clause can be:

* `delimiter`: a single-character string for the delimiter between column values. The default is `","`, i.e., as a CSV file.
* `nulls`: a string pattern indicating a field is NULL. The default is `\N`.
* `escape`: a single-character string for escaping quotes. The default is the quote character itself.
* `quoted`: `'true'` or `'false'` indicating whether all the column values should be output in quotes.  The default is `'false'`.
* `quote`: a single-character string for quoting a column value. The default quote character is double quote `"`.
* `line_delimiter` a single-character string for terminating each line. The default is `"\n"`.
* `header`: `'true'` or `'false'` indicating whether to output a header line for all the column names.  The default is `'true'`.

Example:
```
COPY (SELECT * FROM tweets) TO '/tmp/tweets.csv';
COPY (SELECT * tweets ORDER BY tweet_time LIMIT 10000) TO
  '/tmp/tweets.tsv' WITH (delimiter = '\t', quoted = 'true', header = 'false');
```

# DML

## `INSERT`
```
INSERT INTO <table> VALUES (value, ...);
```
This statement is used for one row at a time ad-hoc inserts and should not be used for inserting a large number of rows. The COPY command should be used instead which is far more efficient.
Example:
```
CREATE TABLE foo (a INT, b FLOAT, c TEXT, d TIMESTAMP);
INSERT INTO foo VALUES (NULL, 3.1415, 'xyz', '2015-05-11 211720`);
```

## `SELECT`
```
SELECT [ALL|DISTINCT] <expr> [AS [<alias>]], ... FROM <table> [,<table>]
  [WHERE <expr>]
  [GROUP BY <expr>, ...]
  [HAVING <expr>]
  [ORDER BY <expr>, ...]
  [LIMIT {<number>|ALL} [OFFSET <number> [ROWS]]];
```
It supports all the common SELECT features except for the following temporary limitations:

* Only equi join between two tables is currently supported.
* Subqueries are not supported.

## Function Support
```
COUNT
MIN
MAX
SUM
NOW
EXTRACT
DATE_TRUNC
CAST
LENGTH
CHAR_LENGTH
```

## Array Support
```
SELECT <ArrayCol>[n] ...
```
Query array elements n of column `ArrayCol`
```
SELECT UNNEST(<ArrayCol>) ...
```
 Flatten entire array `ArrayCol`

# Client Interfaces

MapD uses [Apache Thrift](https://thrift.apache.org) to generate client-side interfaces.  The *interface definitions* are in `$MAPD_PATH/mapd.thrift`.  See Apache Thrift documentation on how to generate client-side interfaces for different programming languages with Thrift.  Also see `$MAPD_PATH/samples` for sample client code.
