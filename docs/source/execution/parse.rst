.. OmniSciDB Query Execution

==================================
Parser / Planner
==================================
The first phase of query engine execution entails parsing SQL statements to a structure that can be easily
understood and processed by downstream query engine components. This is done by using
`Apache Calcite <https://calcite.apache.org/>`_ (referred to as *Calcite* for the remainder of this document),
which is an open source framework for building data management systems. OmniSci DB specifically uses the
relational algebra construction aspect of Calcite to transform user provided SQL statements to relational
algebra expression trees, represented in JSON format. Calcite is run as a sidecar Thrift service in a child
process of the OmniSci DB server.

A `relational algebra <https://calcite.apache.org/docs/algebra.html>`_ expression tree can be thought of as a data
structure that breaks down a SQL statement into simple discrete steps that may depend on each other. For example,
assume there exists a *Users* table that contains the first and last name of users of a given application, and the
following query is run to fetch user information ``SELECT first_name, last_name from Users;``. Output from Calcite
will look like the following:

.. code-block:: JSON

  {
    "rels": [
      {
        "id": "0",
        "relOp": "EnumerableTableScan",
        "fieldNames": [
          "first_name",
          "last_name",
          "rowid"
        ],
        "table": [
          "omnisci",
          "Users"
        ],
        "inputs": []
      },
      {
        "id": "1",
        "relOp": "LogicalProject",
        "fields": [
          "first_name",
          "last_name"
        ],
        "exprs": [
          {
            "input": 0
          },
          {
            "input": 1
          }
        ]
      }
    ]
  }

The above JSON object/relational algebra expression tree can be read as a directive to perform the following
sequence of steps:

* Scan/read all rows (ref: *EnumerableTableScan*) of the *omnisci.Users* table, which has the *first_name*,
  *last_name*, and *rowid* (automatically created) columns.

* Project/choose (ref: *LogicalProject*) column names *first_name* and *last_name*, which corresponds to columns
  with indices 0 and 1 of the provided input (input is a row that is scanned/read, per previous
  *EnumerableTableScan* step)

Query Planning
--------------
There are a lot of use cases where the same query can be executed in different ways or different steps. Each of
these execution option is referred to as a query plan. Calcite attempts to select a query plan with the fastest
execution.

Code References
---------------
Calcite Thrift service startup and API access business logic can be found in the
`Calcite.cpp <https://github.com/omnisci/omniscidb/blob/master/Calcite/Calcite.cpp>`_ file. Calcite server
business logic can be found in the `/java/calcite <https://github.com/omnisci/omniscidb/tree/master/java/calcite>`_
sub-module.