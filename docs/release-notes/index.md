# MapD Platform
The latest version of the MapD Platform is 1.2.8.

#### **1.2.8** - Released October 3, 2016

#####New
- Text columns now default to dictionary encoding. If old unencoded behavior required then `TEXT ENCODING NONE` should be used in create table statement.  NOTE: This will not affect existing tables but any new tables created will be affected.
- Now able to color by boolean

#####Fixed
- Issue for some `CASE` statements involving nullability
- Issue with sort on very high cardinality column
- Now throwing exception on overflow for arithmetic operations
- Allow hash joins rather than loop joins in queries with `ORDER BY`
- Issue when trying to `GROUP BY` array column
- Issue with `OR` statements involving NULLs
- Issue in comparing decimal column with integer literal
- Issue for any string literal containing the term `all` or other SQL tokens
- Now throwing exception for tables with very high number of columns

#### **1.2.7** - Released September 12, 2016

#####New
- Add support in JDBC driver for implicit type casting of expressions to double/string, not requiring explicit CAST operator


#### **1.2.6** - Released September 6, 2016

#####New
- Support for POSIX regular expressions, boolean match
- Performance improvement for some `GROUP BY` `ORDER BY` queries with a `LIMIT`
- Added NVARCHAR support to SQLImporter
- Added function distance_in_meters
- Now supporting sub-pixel morphological anti-aliasing, for better line anti-aliasing

#####Fixed
- Problem when coloring by string with null value
- Failure to update pointmap color when range of the scale changes
- Parsing problem with SQL text containing “all” or “any”

#### **1.2.5** - Released August 23, 2016

#####New
- Improvement in memory efficiency for `GROUP BY` unnested string arrays
- Added fragment size option to SQL Importer
- Optimization to leverage hardware-accelerated FP64 atomics on Pascal architecture
- Improved stability and performance for high cardinality group by queries

#####Fixed
- Issue with multi-key `GROUP BY` on empty table
- Regression with coloring by string on backend rendered images
- Issue on certain hardware where backend rendered pointmap images draw to a corner/side

#### **1.2.4** - Released August 15, 2016

#####New
- `EXTRACT` week support
- `TRUNCATE` support for non-decimal numeric types
- `CAST` from timestamp to date
- Partial `INTERVAL` support
- Performance improvement for `GROUP BY` date
- Additional performance optimizations for subqueries
- `LOG10` support
- Backend rendering now supports all quantitative scales in vega specification, including pow, sqrt, and log

#####Fixed
- Fixed issue with Postgres importer reporting boolean as bit
- Fixed occasional slowdown for render queries on servers with many GPUs
- Fixed issue affecting non-`GPOUP BY` queries on an empty table
- Fixed issue when selecting `MIN` or `MAX` from empty table
- Fixed issue for `IN` subqueries when inner query result is above a certain size
- Fixed issue with performance for “top n” queries

#### **1.2.3** - Released August 1, 2016

#####New
- Now allow using aliases in `FROM` and `WHERE` clauses

#####Fixed
- Made loading from cold cache (disk) faster
- Fixed memory leaks around unsupported queries
- Fixed problem when recreating a previously dropped table
- Fixed problem when parsing CSVs with inconsistent number of columns

#### **1.2.2** - Released July 25, 2016

#####New
- Added math functions (`ACOS`,`ASIN`,`ATAN`,`ATAN2`,`COS`,`COT`,`SIN`,`TAN`,`ABS`,`CEIL`,`DEGREES`,`EXP`,`FLOOR`,`LN`,`LOG`,`MOD`,`PI`,`POWER`,`RADIANS`,`ROUND`,`SIGN`)
- Improved performance for top k IN subqueries
- Added partial support for NOT IN subqueries
- Added automatic reprojection of lat/long to mercator for mapping display

#####Fixed
- Fixed an issue for CAST from a literal decimal
- Fixed CAST of NULL to a numeric type
- Fixed unary minus operator for nullable inputs

#### **1.2.1** - Released July 18, 2016

##### New
- Backend rendered images can now be colored along a spectrum between two colors, based on an accumulated measure (e.g. accumulated red or blue datapoints can result in purple)
- Added `DROP` and `ALTER` table support for Apache Calcite

##### Fixed
- Added a more robust conversion of decimal literals to float, retaining more precision
- Fixed an issue for `CASE` expressions which return booleans

#### **1.2.0** - Released July 11, 2016

##### New
- Changed SQL parser to Apache Calcite
  - Subquery support
  - Further join support (e.g. left outer join)
  - Case insensitivity for column and table names
- New core execution engine, Relational Algebra Virtual Machine ("RAVM"), gives more flexibility allowing execution of arbitrarily complex queries
- Added additional formats for date import
- MapD Immerse v.2 technical preview
  - Redesigned user interface allows more powerful chart creation and intuitive data exploration
  - To access the Immerse Technical Preview Dashboards page, go to `http://<server>:<port>/v2/`
  - Immerse v.2 technical preview is an unstable preview release.  A subset of major known bugs is [here](https://docs.google.com/document/d/1sigSA4IhQTulibtDcxlALaCNEiAqEkPNjR7rkK-BXDo)

##### Fixed
- Fixed a problem with count distinct and group by queries
- Fixed a problem with count on float
- Fixed a problem with projection queries in limited cases
- Fixed a problem where tables created via MapD web-based table importer were not consistent with tables built via SQL CREATE
- Disallowed use of reserved SQL keywords as column names

##### Removed
- Loss of Group By ordinals (would restore pending Calcite support)

##### Dependencies
- Now requiring Java Runtime Environment (JRE) version 1.6 or higher

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
