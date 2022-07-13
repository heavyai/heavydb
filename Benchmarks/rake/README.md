# Benchmark and Correctness Testing between HeavyDB and PostgreSQL databases

### Requirements:

* `ruby` and the `rake` gem.
* Vanilla TPC zip files from the [TPC download page](http://tpc.org/tpc_documents_current_versions/current_specifications5.asp) (currently only TCP-DS is supported but more are coming.)
* A HeavyAI `heavysql` client that connects to a running HeavyDB server with option `--allowed-import-paths` which whitelists the path to the `Rakefile`.
* A PostgreSQL `psql` client that connects to a running PostgreSQL server that is able to run admin commands via `psql -U postgres`. Alternatively, if the 99 TPC-DS output files `query*-pg.txt` exist in the output directory, which can be imported from a separate run, then `psql` won't be needed when run with the `SKIP_PG=1` option (see below).

### Why is this written as a ruby Rakefile?

Like a Makefile, a Rakefile is simply a collection of tasks that are dependent upon one another. Running benchmarks includes a number of interdependent steps, such as
* Decompressing the TPC source files.
* Creating patch files for them (they don't typically compile without changes).
* Building the query and data-generation tools.
* Generating the tables and data.
* Creating the databases, tables and importing the data into each database.
* Patching the SQL query templates to make them compatible with our databases.
* Generating the SQL queries.
* Running the queries and collecting the output.
* Comparing the output from PostgreSQL and HeavyDB, accounting for the idiosyncrasies, and generating json reports on their benchmarks and correctness.

If something goes wrong, it is too time-consuming to redo everything if this were performed by one monolithic script. Instead, a Rakefile compartmentalizes these steps, and skips steps that were already completed, much as a Makefile does.

### How do I run this?

1. Verify above Requirements are met.
2. Place the files `Rakefile` and `TPC-DS_Tools_v3.2.0.zip` in the current directory.
3. Run `rake tpcds:compare HEAVYSQL="/path/to/bin/heavysql -u username -p password"` with correct path, username and password.  Similarly, you may also set `SCALE` and `RNGSEED`.
4. Optional: If the PostgreSQL results `query*-pg.txt` are already present in the output directory (see next step), and compressed lz4 tables generated from a previous `DUMP` command were created, then you may skip running the PostgreSQL queries entirely by appending `SKIP_PG=1` to the above `rake` command.
5. The output files are placed in a directory `TPC-DS_Tools_v3.2.0_1GB`. For each query, 4 files are generated, in additional to a final `report.html` file gathering all results together. Taking query 1 as an example:
   1. **query1.sql** &mdash; The SQL query that was run on each database.
   2. **query1-hv.txt** &mdash; The output from HeavyDB.
   3. **query1-pg.txt** &mdash; The output from PostgreSQL.
   4. **query1.json** &mdash; A json file reporting on the results. For example:<br/>
      `{"success":true,"message":"100 rows in 1188.0 ms.","nrows":100,"time_ms":1188.0}`

### Configuration

See the top of the `Rakefile` for additional environment variables that may be set. In particular:
* `SCALE` &mdash; The scale/size of the database in gigabytes. Default: 1.
* `RNGSEED` &mdash; The random number generator seed for random values placed in the queries.
* `SKIP_PG` &mdash; Set to any non-blank value to skip PostgreSQL queries. Requires all of the the 99 `query*-pg.txt` files to already exist in the output directory.
