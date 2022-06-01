OmniSciDB is being intensively developed and is evolving quickly. In order to help the MapD developer community understand our near term priorities, this document shares work targeted for the next 3-6 months. Many of these items are the result of community requests.

We welcome and encourage developer contributions to the project. Please see the [contribution guidelines](https://github.com/mapd/mapd-core#contributing) and [GitHub issues](https://github.com/mapd/mapd-core/issues).

## Database

#### Completed
- Update via subquery (Completed 5.0)
- Binary/dump restore of tables (Completed 5.0)
- In-memory temporary tables (Completed 5.1)
- Support for SQL VALUES syntax, allowing literals to be used inline in SQL queries (Completed 5.1.2)
- More performant multi-fragment joins (Completed 5.2)
- `ALTER TABLE DROP COLUMN` (Completed 5.2)
- `SQL SHOW` Commands (Completed 5.2)
- Import of compressed parquet files (Completed 5.2)
- Initial support for UNION ALL (Completed 5.3)
- Initial support of multiple executors for improved concurrency (Completed 5.3+)
- Query hint framework, initially allowing `/*+ cpu_mode */ hint (Completed 5.3.1)
- Support for implicit casting for `INSERT AS SELECT` queries to match existing table types when possible
- Concurrent `UPDATE` and `SELECT` on the same table (Completed 5.5)
- Allow none-encoded inputs into user defined functions (Completed 5.5)
- Initial foreign Server/Table support for CSV and Parquet (Completed 5.4-5.5) 
- Query interrupt improvements (Completed 5.5 and ongoing)


#### Upcoming
- `APPROX_MEDIAN` and `APPROX_PERCENTILE` operators
- Additional string function support
- Queryable system metadata tables
- Accelerated range joins
- Query interrupt improvements
- Query/subquery result set recycling for greater performance

## Data Science/[GPU Data Frame (GDF)](http://gpuopenanalytics.com/#/)/[Apache Arrow](https://arrow.apache.org/)

#### Completed
- [PyMapD DB-API Python client](https://github.com/mapd/pymapd)
- [Ibis backend for MapD](https://github.com/ibis-project/ibis)
- Support for Arrow result sets over the wire (in addition to existing in-situ Arrow egress) (Completed 5.5)
- Basic user-defined row and table functions (Completed 5.0, with ongoing improvements)
- User-defined table function (Completed UDTF) improvements (multiple column/query inputs, composability, lazy linking, function redefinition) (Completed 5.4-5.5) 
- `CREATE DATAFRAME` temporary table creation from csv via Arrow (Completed 5.4 and ongoing)

#### Upcoming
- Further increase efficiency of Arrow serialization
- Additional UDF/UDTF improvements (dictionary-encoded text column support, variadic types, performance on large inputs, semantics)
- Experimental ML operators built on UDTFs
