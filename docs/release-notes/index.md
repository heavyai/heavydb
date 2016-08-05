    commit 5642b74232314a851eee21b9411b6c9176946253
    Author: norairk <norair@mapd.com>
    Date:   Tue Jul 19 18:10:39 2016 -0700

        Revert "Run time improvement for loading data from cold cache"

        This reverts commit c98f51edd50e99fbf1d1c11ad6ac58ea1db3bbee.

    commit 0b61ca3138f9affcf99f044bcac930efd33edf3a
    Author: Minggang Yu <miyu@mapd.com>
    Date:   Thu Aug 11 12:57:01 2016 -0700

        Compact prepended index buffer and key buffer for Thrust sort

    commit 71b59a5b8e5833a8ab7f8967298c191dfe0764aa
    Author: Minggang Yu <miyu@mapd.com>
    Date:   Wed Aug 10 15:06:44 2016 -0700

        Allow 32 bit compaction to kick in for 4B rows tables

    commit 065ae55a6e1ab73188a1f90bc70841f85cd4250c
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Thu Aug 11 13:46:13 2016 -0700

        Set the async execution policy in result set reduction

        The constructor we were using made both the async and deferred policies
        acceptable and the implementation chose deferred (which is single-threaded).
        This was one of several reasons for https://github.com/map-d/mapd2/issues/349.

    commit a4b7e51c3b185f7192fc1255d5c30866798ff9b5
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Wed Aug 10 17:52:37 2016 -0700

        Avoid redundant memory copy when SMs share output buffer

        Should help big group by + top queries.

    commit fb969fde2ae7d6074a553767bc33e702ae8b4205
    Author: Christopher Root <chris@mapd.com>
    Date:   Wed Aug 10 16:15:42 2016 -0700

        Improve cleanup when throwing opengl out-of-memory error

    commit 3a98e6dfefd5115cf25175d2926e04e04ad69b40
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Wed Aug 10 12:15:48 2016 -0700

        New result set baseline sort 3/3

    commit 6b73da9e42fdac99aed53e7ab3fa0056bb811f6d
    Author: Minggang Yu <miyu@mapd.com>
    Date:   Tue Aug 9 18:12:45 2016 -0700

        Fix a typo in computing column buffer size that causes CPU buffer overrun.

        Fixes #341.

    commit 8e8a9ca955966aed1d2ed845d7bc7482fbe65cf4
    Author: Christopher Root <chris@mapd.com>
    Date:   Tue Aug 9 15:25:11 2016 -0700

        Adding support for a 'nullValue' in scales to map null values in data to some reasonable output

    commit 67ca24b0dc378e493bf529ac3cdfb4fdf9ef32b5
    Author: Michael Thomson <michael@mapd.com>
    Date:   Tue Aug 9 09:33:38 2016 -0700

        Correct usgae message and status message

    commit fee8ff6c7f12bd1752a108d55fc14815d43c7862
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Mon Aug 8 18:06:42 2016 -0700

        Factor out pointer computation in getTargetValueFromBufferColwise

    commit 04a7a0a71f7aa7369d95945bdbfb9188d251c322
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Mon Aug 8 17:15:50 2016 -0700

        New result set baseline sort 2/?

    commit 30406351f02c8e8fd1c55419bf551d0c37e13a68
    Author: Christopher Root <chris@mapd.com>
    Date:   Mon Aug 8 16:43:47 2016 -0700

        Fix sign-flip bug in scale ops with near-limit vals

    commit 112c4fd367e7345c89b8c41d5e3395c0860eb46e
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Mon Aug 8 16:20:15 2016 -0700

        Remove redundant check in isSelectStar

    commit 67bd77953e6757b38205546a56953fd7387e09b7
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Mon Aug 8 15:25:28 2016 -0700

        Don't remove rowid from * expansion for subquery results

        Fixes #346.

    commit 2e2a0d1060772cad371771ba6d47d14ac5d6af48
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Mon Aug 8 12:04:44 2016 -0700

        New result set baseline sort 1/?

        Mostly a port from ResultRows but completely in-place except for
        the indices buffer to represent the permutation.

    commit 3248ee05fc30db418a21821a805752dda358710f
    Author: Andrew Seidl <dev@aas.io>
    Date:   Mon Aug 8 12:00:19 2016 -0700

        Hardcode Thrift threadpool size to 4

        Threadpool size appears to be correlated to the libnvidia-glcore
        crashes. Hard code to 4 for now.

        cc: #331

    commit cf065b07e275b56e52c2bbc6a28012af7cc33c3a
    Author: Alex Şuhan <alex.suhan@gmail.com>
    Date:   Sun Aug 7 12:14:30 2016 -0700

        Relax incorrect check in inline_null_val

        Fixes #344.

    commit 324bdf2efd046b6103547b72c229cee6c5e5ca96
    Author: Michael Thomson <michael@mapd.com>
    Date:   Sat Aug 6 17:26:57 2016 -0700

        Add session restart retry facilty to StreamInsert

        added params
        --retry_count
        --retry_wait

        Help:
        ./StreamInsert --help
        Usage: <table name> <database name> {-u|--user} <user> {-p|--passwd} <password> [{--host} <hostname>][--port <port number>][--delim <delimiter>][--null <null string>][--line <line delimiter>][--batch <batch size>][{-t|--transform} transformation ...][--retry_count <num_of_retries>] [--retry_wait <wait in secs.][--print_error][--print_transform]

        Example:
        ./StreamInsert --batch 200 --retry_count 50 --retry_wait 3 --database mapd --passwd HyperInteractive --table stest --user mapd < stestin

        Examle output with a Db stop and restart during stream load

        109600 Rows Inserted, 0 rows skipped.
        109800 Rows Inserted, 0 rows skipped.
        110000 Rows Inserted, 0 rows skipped.
        110200 Rows Inserted, 0 rows skipped.
        110400 Rows Inserted, 0 rows skipped.
        110600 Rows Inserted, 0 rows skipped.
        Exception trying to insert data No more data to read.
          Waiting  5 secs to retry Inserts , will try 10 times more
        Thrift error: No more data to read.
        Thrift: Sat Aug  6 17:13:38 2016 TSocket::open() connect() <Host: localhost Port: 9091>Connection refused
        Thrift error: connect() failed: Connection refused
        Exception trying to insert data Called write on non-open socket
          Waiting  5 secs to retry Inserts , will try 9 times more
        Thrift error: Called write on non-open socket
        Thrift: Sat Aug  6 17:13:43 2016 TSocket::open() connect() <Host: localhost Port: 9091>Connection refused
        Thrift error: connect() failed: Connection refused
        Exception trying to insert data Called write on non-open socket
          Waiting  5 secs to retry Inserts , will try 8 times more
        Thrift error: Called write on non-open socket
        Thrift: Sat Aug  6 17:13:48 2016 TSocket::open() connect() <Host: localhost Port: 9091>Connection refused
        Thrift error: connect() failed: Connection refused
        Exception trying to insert data Called write on non-open socket
          Waiting  5 secs to retry Inserts , will try 7 times more
        Thrift error: Called write on non-open socket
        Thrift: Sat Aug  6 17:13:53 2016 TSocket::open() connect() <Host: localhost Port: 9091>Connection refused
        Thrift error: connect() failed: Connection refused
        Exception trying to insert data Called write on non-open socket
          Waiting  5 secs to retry Inserts , will try 6 times more
        Thrift error: Called write on non-open socket
        110800 Rows Inserted, 0 rows skipped.
        111000 Rows Inserted, 0 rows skipped.
        111200 Rows Inserted, 0 rows skipped.
        111400 Rows Inserted, 0 rows skipped.

    commit 9ccb340bac060d0f2731a712736a5911bf2a9afd
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Fri Aug 5 17:31:06 2016 -0700

        Support for LOG10

    commit f502513834f53303e226859fb9744e4c7e3ce1cf
    Author: Andrew Seidl <dev@aas.io>
    Date:   Fri Aug 5 09:15:51 2016 -0700

        systemd: always restart

    commit d59771ca78471173c4e5e4b45dc8c6ee2d3f5228
    Author: Alex Şuhan <alex.suhan@gmail.com>
    Date:   Thu Aug 4 21:18:25 2016 -0700

        Revert "Fix non-group queries on empty table"

        This reverts commit 9bf5328b3212eb4c3d90f39388f6b76e37bed66d.

    commit 909d8ed8cc2275e405c5e4716800802bbda9a7ca
    Author: Alex Şuhan <alex.suhan@gmail.com>
    Date:   Thu Aug 4 19:34:06 2016 -0700

        Fix leaks in legacy parser

    commit 9bf5328b3212eb4c3d90f39388f6b76e37bed66d
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Thu Aug 4 15:15:42 2016 -0700

        Fix non-group queries on empty table

        Fixes #9.

    commit 067ca99cfc9e706fbe508407fcae1cf36919f524
    Author: Alex Şuhan <alex.suhan@gmail.com>
    Date:   Thu Aug 4 23:29:27 2016 -0700

        Fix non-group queries on empty table, take 2

        Fix the behavior when fragments are skipped as well. Also, isolate
        this path from the regular path.

        Fixes #9.

    commit dfe4a64ea4c2988997d21679189a8fc42f4d2132
    Author: Christopher Root <chris@mapd.com>
    Date:   Thu Aug 4 16:01:16 2016 -0700

        Adding non-linear pow/sqrt/log quantitative scale support

    commit 822e0e035c802ba748bfa26cac4eb303f8f9cace
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Thu Aug 4 11:40:54 2016 -0700

        Add bucket information to the range of a date column

        We know the minimum gap between consecutive days, we can make group by date fast.

    commit 5f1bd1f0f943b137671b621dd338a37f501dd99c
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Wed Aug 3 15:33:35 2016 -0700

        Support for interval type 3/?

        Rewrite some DATETIME_PLUS idioms to DATE_TRUNC, which we already support.

    commit 990878cac3af03cbf5727b53ac780c43f0f707be
    Author: Alex Şuhan <alex.suhan@gmail.com>
    Date:   Wed Aug 3 19:44:01 2016 -0700

        Add time interval support to scalar_datum_to_string

    commit 48445a98d5db8530ed1c82d998224d26218a400d
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Wed Aug 3 15:53:27 2016 -0700

        Add visitor methods for functions and datediff

    commit b8b688429edc37dfd048637d0623a0511a62e57a
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Wed Aug 3 12:23:26 2016 -0700

        Support for interval type 2/?

        Arithmetic support for time intervals.

    commit d226cc083a09da99c134946660e3e06a51094f3b
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Tue Aug 2 17:16:54 2016 -0700

        Support for interval type 1/?

        This patch only adds literal and comparison support.

    commit c6613916d14f2920f666157bd8b4643293d599ed
    Author: Andrew Seidl <dev@aas.io>
    Date:   Tue Aug 2 14:03:08 2016 -0700

        Switch to TThreadPoolServer

        Reuse threads to help minimize OpenGL context warmup time.

        Resolves #301

    commit 21e0dae2b7094717544de5a357ad29b03e27a11d
    Author: Christopher Root <chris@mapd.com>
    Date:   Wed Aug 3 07:56:37 2016 -0700

        Scale domain/range data can now be coerced to all be the same type - i.e. now supports a domain of [0, 1.01]

    commit c498f1f278c8ee1bf8509a15e01e06d7d7958fb9
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Tue Aug 2 16:57:53 2016 -0700

        Add interval type to the SQLTypes enum

    commit 0036065eee4f52e45aae1b74d5f6b269c2dd84d9
    Author: Minggang Yu <miyu@mapd.com>
    Date:   Sun Jul 24 12:42:10 2016 -0700

        Allow more sequences to be coalesced

        cc #235

    commit 02261bca9181945cfefca576df09faad82d0a5d4
    Author: Minggang Yu <miyu@mapd.com>
    Date:   Sun Jul 24 00:45:17 2016 -0700

        Remove useless ForLoop info

    commit 5f2c67ef2a1701e268dc53c235b08175537c3532
    Author: Minggang Yu <miyu@mapd.com>
    Date:   Fri Jul 22 17:50:48 2016 -0700

        Fix DFA to transit along du-chain instead of adjacency in vector when detecting compound pattern

        Fixes #235.

    commit 711254c2de88b715b60b9b948fa19bf11902d69a
    Author: Minggang Yu <miyu@mapd.com>
    Date:   Fri Jul 22 18:20:46 2016 -0700

        Fix the order of releasing node ownership to eliminate more correctly

    commit ff4c78235a4ff3a85668a019e64a855feecd45a9
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Tue Aug 2 13:50:04 2016 -0700

        Drop subquery_test at the end of execution tests

    commit aa2b032cf55533d2f55620443ba1777c1e86875b
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Mon Aug 1 18:06:14 2016 -0700

        Make TRUNCATE covariant

    commit 32382ebcf9265a8e0766031707d3eb5533087cbf
    Author: Alex Şuhan <alex.suhan@gmail.com>
    Date:   Mon Aug 1 22:02:06 2016 -0700

        Support for literal timestamp cast to date

    commit 7e3bcd754a998264a46d82c56cfbc9053fbc4dea
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Mon Aug 1 12:53:54 2016 -0700

        Allow cast from timestamp to date

        Fixes #332.

    commit d06f26f9c9f3754ad09a54b28809cbcc4d7e583b
    Author: Michael Thomson <michael@mapd.com>
    Date:   Sun Jul 31 18:32:01 2016 -0700

        Repair my break to ISODOW, add further tests

    commit f4d0d6ca183a1d08ac311cfbbcf54352b6483719
    Author: Michael Thomson <michael@mapd.com>
    Date:   Sun Jul 31 11:35:47 2016 -0700

        Update Docs for WEEK and numeric TRUNCATE

    commit 6c0dbc141109521f0e09f01dd26497eb7ecc3ffe
    Author: Michael Thomson <michael@mapd.com>
    Date:   Sat Jul 30 04:22:57 2016 -0700

        Further function support for Tableau

        Further changes for tableau Support
        add EXTRACT WEEK function returns 1 -53 based on caledndar week not ISO WEEK currently
        Hacky function handling in jdbc
        need to visit the CAST rewrite regex a it will break hand written queries probably

    commit 7a7d5cbe01e3880470f783b85beefba1555f4419
    Author: Alex Şuhan <alex.suhan@gmail.com>
    Date:   Sat Jul 30 11:22:15 2016 -0700

        Throw exception for unsupported type on deserialization

    commit b8d0c29f0bc3f5fbe3875e8dc34d0fb6cf8e22cb
    Author: Michael Thomson <michael@mapd.com>
    Date:   Fri Jul 29 22:39:15 2016 -0700

        SQLImported deal with postgress reporting booelan as bit

    commit 816f610c6462427337e595a23f7c3080acf41a6d
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Fri Jul 29 14:34:50 2016 -0700

        Add dictionary encoded strings to ResultSet 2/2

    commit 4a586dacb36d96ba5e5be20543c205baa8963618
    Author: Alex Şuhan <alex@map-d.com>
    Date:   Fri Jul 29 12:18:17 2016 -0700

        Add dictionary encoded strings to ResultSet 1/?



# MapD Platform
The latest version of the MapD Platform is 1.2.4.

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
