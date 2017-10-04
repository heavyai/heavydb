The MapD Core database is being intensively developed and is evolving quickly. In order to help the MapD developer community understand our near term priorities, this document shares our near-term internal work. The work below is targeted for the next few months. Many of these items are the result of community requests.

We welcome and encourage developer contributions to the project. Please see the [contribution guidelines](https://github.com/mapd/mapd-core#contributing) and [GitHub issues](https://github.com/mapd/mapd-core/issues).

## Database
- Top k sort on GPU. Ability to efficiently order by a projected column. (Completed 3.2.2)
- JOIN improvements. Additional join types and performance improvements. (Completed 3.2)
- JOIN improvements.  Table orderings and optimizations for larger numbers of tables.
- TRUNCATE. Efficiently remove contents of table or part of table. (Completed v3.2)
- Batched UPDATE and DELETE.
- Interval type, and TIMESTAMP_ADD, TIMESTAMP_DIFF. (Completed v3.2)
- Table Sharding. Table sharding types in addition to the existing round robin. (Completed v3.2)
- Table-level security, GRANT, REVOKE. Ability to grant and revoke table access to users/roles. (Testing in 3.2.3)
- User Defined Functions (UDF)
- Rendering for graphs and lines

## GIS
- Geo Types. Point, Line, Polygon.
- Basic Geo functions (st_contains, st_distance)

## GOAI - GPU data frame
- Multi-GPU. Return data frame on multiple GPUs without a final reduction phase.
- Data frame as input to MapD. Ability to insert into a table from a data frame.
- Collaborative memory management. Ability to coordinate memory management with other GPU components.
