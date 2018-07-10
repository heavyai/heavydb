The MapD Core database is being intensively developed and is evolving quickly. In order to help the MapD developer community understand our near term priorities, this document shares work targeted for the next 3-6 months. Many of these items are the result of community requests.

We welcome and encourage developer contributions to the project. Please see the [contribution guidelines](https://github.com/mapd/mapd-core#contributing) and [GitHub issues](https://github.com/mapd/mapd-core/issues).

## Database

#### Completed
- Top k sort on GPU. Ability to efficiently order by a projected column (Completed 3.2.2)
- JOIN improvements. Additional join types and performance improvements (Completed 3.2)
- JOIN improvements.  Table orderings and optimizations for larger numbers of tables (Ongoing improvements)
- TRUNCATE. Efficiently remove contents of table or part of table (Completed v3.2)
- Batched UPDATE and DELETE (Completed 4.0)
- Interval type, and TIMESTAMP_ADD, TIMESTAMP_DIFF (Completed v3.2)
- Table Sharding. Table sharding types in addition to the existing round robin (Completed v3.2)
- Table-level security, GRANT, REVOKE. Ability to grant and revoke table access to users/roles (Testing in 3.2.3, released in 4.0)

#### Upcoming

- Use of shared memory to accelerate certain group by operations
- More efficient memory layout for certain group by operations
- More efficient range joins
- Better usage of both GPU and CPU, depending on memory requirements of queries
- Create table as select (CTAS) improvements (performance, and handling variable length data)
- Update improvements, including handling variable length data, update via subquery, and more efficient columnar output
- Delete improvements, including immediate/delayed vacuum capability
- More granular timestamp support (to nanoseconds)
- Row-level User Defined Functions (UDFs)

## GIS

#### Completed
- Geo Types. Point, Line, Polygon (Completed 4.0)
- Basic Geo functions (ST_Contains, ST_Distance) (Completed 4.0)

#### Upcoming
- Additional OGC geospatial types: Multi(Point|Line|Polygon)
- Additional OGC geospatial operators (ST_Intersects, ST_Within, ST_Area, ST_Perimiter, etc)
- Support for geometric constructors (ST_MakePoint, etc)
- Accelerated geospatial joins (with dynamic spatial hashing, not relying purely on brute force loop joins)
- Fixed length arrays for POINT datatype to conserve memory
- Geodatabase/geopackage import
- WMS support

## Data Science/[GPU Data Frame (GDF)](http://gpuopenanalytics.com/#/)/[Apache Arrow](https://arrow.apache.org/)

#### Completed
- Data frame as input to MapD. Ability to insert into a table from a data frame (Completed 3.2.4)
- [PyMapD DB-API Python client](https://github.com/mapd/pymapd)
- [Ibis backend for MapD](https://github.com/ibis-project/ibis)

#### Upcoming
- Increase efficiency of Arrow serialization
