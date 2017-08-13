The MapD Core database is being intensively developed and is evolving quickly. In order to help the MapD developer community understand our near term priorities, this document shares our near-term internal work. The work below is targeted for the next few months. Many of these items are the result of community requests.

We welcome and encourage developer contributions to the project. Please see the [contribution guidelines](https://github.com/mapd/mapd-core#contributing) and [GitHub issues](https://github.com/mapd/mapd-core/issues).

## Database
- Top k sort on GPU. Ability to efficiently order by a projected column.
- JOIN improvements. Additional join types and performance improvements.
- TRUNCATE. Efficiently remove contents of table or part of table.
- Batched UPDATE and DELETE.
- Interval type, and TIMESTAMP_ADD, TIMESTAMP_DIFF.
- Table Sharding. Table sharding types in addition to the existing round robin.
- Table-level security, GRANT, REVOKE. Ability to grant and revoke table access to users/roles.
- User Defined Functions (UDF)

## GIS
- Geo Types. Point, Line, Polygon.
- Basic Geo functions (st_contains, st_distance)

## GOAI - GPU data frame
- Multi-GPU. Return data frame on multiple GPUs without a final reduction phase.
- Data frame as input to MapD. Ability to insert into a table from a data frame.
- Collaborative memory management. Ability to coordinate memory management with other GPU components.
