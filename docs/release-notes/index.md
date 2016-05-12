# MapD Platform
The latest version of the MapD Platform is 1.1.4.


#### **1.1.4** — Released May 16, 2016

#### New
- Improved memory fragmentation handling by adding support for huge pages.
- Improved performance when joining large tables to small tables.
- Improved join on dictionary strings performance.

#### Fixed
- Fixed out-of-bound access in VRAM when out-of-slot exception raised.
- Fixed issue with queries returning empty result sets.
- More conservative tuple threshold for compaction, fixing count overflow on large tables.
- Reduced memory fragmentation for long-running servers.


#### **1.1.3** — Released May 9, 2016

#### New
- Added a new chart type: _Number Chart_. The _Number Chart_ shows a single value, making it simpler to point out important averages, totals, etc.
- Added a `--quiet` flag to `mapdql` to supress it's informational messages from appearing in `STDOUT`.
- Added frontend-rendered choropleth overlays to _Point Map_ charts.
- Added a watchdog capability to catch SQL queries that are poorly formulated.
- Improved the Database Engine log messages to improve readability, and consistency.
- Improved the `render()` API to work with more column types. You can now color output by values taken from your boolean and decimal columns.

#### Fixed
- Fixed a bug that caused _Bar Charts_ to jump around when users clicked on certain rows in long multi-page chart instances.
- Fixed a bug where the CSV import logic prevented some quoted empty strings from being handled properly.
- Fixed a bug where the CSV import logic rejected rows with empty strings in the last position.
- Fixed a bug where the import logic wouldn't properly handle string arrays with embedded `NULL` elements.
- Fixed a bug where the SQL `AVG()` function would introduce rounding errors under some circumstances.
- Fixed a bug where SQL statements with `JOIN` and `HAVING` clauses wouldn't execute.
