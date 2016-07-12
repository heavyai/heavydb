# MapD Platform
The latest version of the MapD Platform is 1.1.9.

#### **1.2.0** - Released July 11, 2016

##### New
- Changed SQL parser to Apache Calcite
   -Subquery support
   -Further join support (e.g. left outer join)
   -Case insensitivity for column and table names
- New core execution engine, Relational Algebra Virtual Machine ("RAVM"), gives more flexibility allowing execution of arbitrarily complex queries
- Added additional formats for date import
- MapD Immerse v.2 technical preview
   - Newly organized user interface allows the more flexible and intuitive creation of charts by allowing user to specify either data or chart type, rather than forcing chart type first
   - User can quickly toggle between chart types to easily preview how data would appear
   - To access the Immerse Technical Preview Dashboards page, go to `http://<your server>:9092/v2`. Note, change the port number `:9092` to match your configuration
   - Immerse v.2 technical preview is an unstable preview release.  A subset of major known bugs is [here](https://docs.google.com/document/d/1sigSA4IhQTulibtDcxlALaCNEiAqEkPNjR7rkK-BXDo)

##### Fixed
- Fixed a problem with count distinct and group by queries which caused a crash
- Fixed a problem with count on float which caused a crash
- Fixed a problem with projection queries which caused a crash in limited cases
- Fixed a problem where tables created via MapD web-based table importer were not consistent with tables built via SQL CREATE
   - Disallowed additional SQL keywords from column names

##### Removed
-Loss of Group By ordinals (would restore pending Calcite support)

##### Dependencies
-Now requiring Java Runtime Environment (JRE) version 1.6 or higher

#### **1.1.9** - Released June 27, 2016

##### New
- Improved logging and system process management
  - Deprecated `--disable-fork` flag in `mapd_server`. Please remove this flag from any config files.
  - Removed `fork()` from `mapd_server`. Automatic restart should now be handled by an external process, such as `systemd`.
  - Added graceful shutdown to `mapd_web_server` so that `systemd` more accurately reports its status
  - Modified `mapd_server` service file so that `systemd` more accurately reports its status
  - Improved logging of various mapd_server operations
- Improved memory handling to better maximize GPU RAM usage

##### Fixed
- Fixed a bug that prevented queries from running which were joining an empty table
- Fixed a subtle stroke/line visual defect when polygons are rendered on the backend

#### **1.1.8** — Released June 21, 2016

##### New
- Added `\copygeo` command to support ingesting shapefiles
- Added backend API for rendering polygons

##### Fixed
- Improved performance of `CASE` queries that don't have an `ELSE` clause
- Fixed a crash that would occur when certain large output results were generated
- Improved performance of queries, such as `SELECT * FROM table_name LIMIT 5`
- Fixed a bug that would sometimes omit results from queries with `AVG` where `NULL`s were present

#### **1.1.7** — Released June 13, 2016

##### Fixed
- Fixed bug where certain long-running queries would needlessly block others
- Immerse: fixed a problem where embedding apostrophes or % in filters or custom filters could cause errors
- Immerse: added MapDCon example for Node.js

#### **1.1.6** — Released May 31, 2016

##### New
- Added Apache Sqoop support to the MapD JDBC driver. Please contact us at `support@mapd.com` to obtain the JDBC driver.
- Improved performance when grouping on `date_trunc` with additional columns

##### Fixed
- Fixed a bug that would appear when calculated fields tried to divide by zero
- Fixed bug with CASE expressions
- Fixed bug where COPY statement blocks execution of other queries

#### **1.1.5** — Released May 23, 2016

##### New
- Improved error logging to reveal the root kernel launch error for group by queries
- Added a new API endpoint `sql_validate` to the API

##### Fixed
- Fixed a bug that calculated incorrect results on` COUNT(CASE....)` style conditional counting queries
- Fixed a memory usage and performance bug which was causing some `render` API calls to timeout

#### **1.1.4** — Released May 16, 2016

##### New
- Improved memory fragmentation handling by adding support for huge pages.
- Improved performance when joining large tables to small tables.
- Improved join on dictionary strings performance.

##### Fixed
- Fixed out-of-bound access in VRAM when out-of-slot exception raised
- Fixed issue with queries returning empty result sets
- More conservative tuple threshold for compaction, fixing count overflow on large tables
- Reduced memory fragmentation for long-running servers

#### **1.1.3** — Released May 9, 2016

##### New
- Added a new chart type: _Number Chart_. The _Number Chart_ shows a single value, making it simpler to point out important averages, totals, etc.
- Added a `--quiet` flag to `mapdql` to supress it's informational messages from appearing in `STDOUT`
- Added frontend-rendered choropleth overlays to _Point Map_ charts
- Added a watchdog capability to catch SQL queries that are poorly formulated
- Improved the Database Engine log messages to improve readability, and consistency
- Improved the `render()` API to work with more column types. You can now color output by values taken from your boolean and decimal columns

##### Fixed
- Fixed a bug that caused _Bar Charts_ to jump around when users clicked on certain rows in long multi-page chart instances
- Fixed a bug where the CSV import logic prevented some quoted empty strings from being handled properly
- Fixed a bug where the CSV import logic rejected rows with empty strings in the last position
- Fixed a bug where the import logic wouldn't properly handle string arrays with embedded `NULL` elements
- Fixed a bug where the SQL `AVG()` function would introduce rounding errors under some circumstances
- Fixed a bug where SQL statements with `JOIN` and `HAVING` clauses wouldn't execute
