.. OmniSciDB Query Execution

===========================
DAG Execution / Translation
===========================

Once the relational algebra tree provided by Calcite has been interpreted (see :doc:`../calcite/calcite_parser`) and optimized (see :doc:`./optimizer`) the tree is ordered and executed. Each remaining node in the tree (except for ``RelTableScan`` and ``RelJoin``) forms a query step. Each step is executed in order, with the result from all previous steps available for subsequent steps. The relational algebra tree is a directed, ascyclic graph dictating the execution order of the query steps.

Determining Execution Order
===========================

The DAG is built from the relational algebra tree by adding edges between nodes in the tree according to the inputs to each node, starting with the last node in the tree (i.e. the last query step to be executed). 

Once the DAG has been built, an execution ordering is determined using a topological sort over the DAG. The sort returns an ordered list of vertices according to the dependencies described by the DAG. For query execution, we walk the DAG according to the ordering. 


Consider the following example: 

.. code-block:: sql 

  SELECT t1.x FROM dead_cols_test t1 JOIN (SELECT * FROM dead_cols_test) t2 ON t1.x = t2.x;

with the following Calcite plan:

.. code-block::
  :linenos:

  LogicalProject(x=[$0])
    LogicalJoin(condition=[=($0, $3)], joinType=[inner])
      EnumerableTableScan(table=[[mapd, dead_cols_test]])
      LogicalProject(x=[$0], y=[$1], rowid=[$2])
        EnumerableTableScan(table=[[mapd, dead_cols_test]])

The DAG for this query is as follows:

.. graphviz::

   digraph {
      "Project [1]" -> "Join [2]";
      "Join [2]" -> "Scan [3]";
      "Join [2]" -> "Project [4]";
      "Project [4]" -> "Scan [5]";
   }

(with the number in paranthesis corresponding to the line number in the calcite plan)

The topological for the above graph produces the following ordering:

.. graphviz::

   digraph {
      "Project [1]" -> "Join [2]"[label="4"];
      "Join [2]" -> "Scan [3]"[label="3"];
      "Join [2]" -> "Project [4]"[label="2"];
      "Project [4]" -> "Scan [5]"[label="1"];
   }

.. note::

  The ordering will be applied per vertex, but for illustration purposes we have placed the ordering on the edges. The vertex ordering can be deduced from the above figure by moving the edge order up to the higher vertex in the graph (e.g. the first step to be executed will be `Project [4]`).

Finally, note that `Scan` and `Join` nodes are not executed, but are automatically rolled into the next node during work unit generation.

``RaExecutionDescriptor``
=============================

After the ordering has been determined, query steps are wrapped in ``RaExecutionDescriptor`` objects. The ``RaExecutionDescriptor`` stores the relational algebra node for the query step, along with the query step result and any related output metadata. It is important that these objects do not go out of scope until the final ``ResultSet`` is returned to the client, as intermediate results may be required by subsequent query steps. 

Query Step Translation
======================

Each query step is packaged into a work unit for code generation and execution. The act of packaging a query step into code generation is called `translation` and is managed by the ``RelAlgTranslator``. The translator converts a set of `Rex` expressions into an abstract syntax tree (AST) representation, which maps directly to the generated code for the :term:`kernel`. 

The translated AST is stored in multiple vectors which logically separate the projected SQL expressions from group by targets, filters, etc. The ``RelAlgExecutionUnit`` stores analyzer expressions in the following members:

- ``target_exprs``: Projected output expressions for the query step.
- ``groupby_exprs``: Columns being grouped. Note that all projection queries are considered group by queries with the group key being the identity function.
- ``quals``: Filter expressions.
- ``simple_quals``: Filter expressions involving a literal value (e.g. `WHERE x = 10`). These are separated for purposes of `fragment skipping`.
- ``join_quals``: Join expressions. 
- ``sort_info``: Columns used for sorting, along with related sort info (`limit`, `offset`, etc). 

.. note:: 

  The ``quals``, ``simple_quals``, and ``join_quals`` vectors together make up the set of all filter expressions. That is, a filter :term:`expression` comparing with a literal will be in ``simple_quals`` only, and will not be duplicated in the ``quals`` vector. 

The ``RelAlgExecutionUnit`` is the primary member of the ``WorkUnit`` and contains all the information required to generate code for the query. 


Query Step Execution
====================

After translation, the `work unit` is passed to the ``Executor`` for native code generation and kernel execution. The ``Executor`` returns a ``ResultSet`` pointer. The ``ResultSet`` pointer is stored in the ``ExecutionDescriptor`` for the current step, and is also stored in the global temporary tables map. Intermediate results are referenced by negating the node ID of their parent query step. 

Scalar Subqueries
-----------------

Scalar subqueries are subqueries which return a single literal value, e.g.:

.. code-block:: sql 

  SELECT x FROM test WHERE x = (SELECT y FROM test2);

Scalar subqueries are identified during interpretation and split out prior to execution of the first query step. The subqueries are then executed as individual queries. The ``ResultSet`` for scalar subquery execution is expected to be a single row with a single column. During translation, a ``RexSubQuery`` expression is replaced with the result from the subquery, represented by a literal analyzer expression. The ``subqueries_`` member of the ``RelAlgExecutor`` manages scalar subquery results for use in future steps. 
