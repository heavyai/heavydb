DML
===

``INSERT``
~~~~~~~~~~

::

    INSERT INTO <table> VALUES (value, ...);

This statement is used for one row at a time ad-hoc inserts and should
not be used for inserting a large number of rows. The COPY command
should be used instead which is far more efficient. Example:

::

    CREATE TABLE foo (a INT, b FLOAT, c TEXT, d TIMESTAMP);
    INSERT INTO foo VALUES (NULL, 3.1415, 'xyz', '2015-05-11 211720`);

``SELECT``
~~~~~~~~~~

::

    [ WITH <alias> AS <query>,... ]
    SELECT [ALL|DISTINCT] <expr> [AS [<alias>]], ...
      FROM <table> [ <alias> ], ...
      [WHERE <expr>]
      [GROUP BY <expr>, ...]
      [HAVING <expr>]
      [ORDER BY <expr> [ ASC | DESC ] , ...]
      [LIMIT {<number>|ALL} [OFFSET <number> [ROWS]]];

``EXPLAIN``
~~~~~~~~~~~
::

   EXPLAIN <STMT>;

Shows the generated IR code identifying whether it will be executed on GPU or CPU, mostly useful for MapD own internal debug.

::

   EXPLAIN CALCITE <STMT>;

Returns a Relational Algebra tree describing the high level plan that will be followed to execute the statement


	
``Table Expression and Join Support``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    <table> , <table> WHERE <column> = <column>
    <table> [ LEFT ] JOIN <table> ON <column> = <column>
    <table> NATURAL JOIN <table> ON <column>

**Usage Notes**

- Tables may be joined on one column only.
- If join column names or aliases are not unique, they must be prefixed by
  their table name.
- For NATURAL join, the respective tables must each have an identically named
  column.
- Data types of join columns must be SMALLINT, INTEGER, BIGINT, or TEXT/VARCHAR
  ENCODING DICT.
- Data types of join columns must match exactly, for example a SMALLINT column
  cannot be joined to a BIGINT column.
- For all but the first table list in the from-list, the data values in the
  join column must be unique. In data warehouse terms, list the "fact" table
  first, followed by any number of "dimension" tables.
- For all but the first table in the from-list, the number of rows in the table
  must be smaller than **fragment\_size**.
- ORDER BY sort order defaults to ASC.

Logical Operator Support
~~~~~~~~~~~~~~~~~~~~~~~~

+------------+-----------------+
| Operator   | Description     |
+============+=================+
| AND        | logical AND     |
+------------+-----------------+
| NOT        | negates value   |
+------------+-----------------+
| OR         | logical OR      |
+------------+-----------------+

Comparison Operator Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------------+---------------------------------+
| Operator                      | Description                     |
+===============================+=================================+
| =                             | equals                          |
+-------------------------------+---------------------------------+
| <>                            | not equals                      |
+-------------------------------+---------------------------------+
| >                             | greater than                    |
+-------------------------------+---------------------------------+
| >=                            | greater than or equal to        |
+-------------------------------+---------------------------------+
| <                             | less than                       |
+-------------------------------+---------------------------------+
| <=                            | less than or equal to           |
+-------------------------------+---------------------------------+
| BETWEEN **x** AND **y**       | is a value within a range       |
+-------------------------------+---------------------------------+
| NOT BETWEEN **x** AND **y**   | is a value not within a range   |
+-------------------------------+---------------------------------+
| IS NULL                       | is a value null                 |
+-------------------------------+---------------------------------+
| IS NOT NULL                   | is a value not null             |
+-------------------------------+---------------------------------+

Mathematical Function Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------+--------------------------------+
| Function             | Description                    |
+======================+================================+
| ABS(\ **x**)         | returns the absolute value of  |
|                      | **x**                          |
+----------------------+--------------------------------+
| CEIL(\ **x**)        | returns the smallest integer   |
|                      | not less than the argument     |
+----------------------+--------------------------------+
| DEGREES(\ **x**)     | converts radians to degrees    |
+----------------------+--------------------------------+
| EXP(\ **x**)         | returns the value of e to the  |
|                      | power of **x**                 |
+----------------------+--------------------------------+
| FLOOR(\ **x**)       | returns the largest integer    |
|                      | not greater than the argument  |
+----------------------+--------------------------------+
| LN(\ **x**)          | returns the natural logarithm  |
|                      | of **x**                       |
+----------------------+--------------------------------+
| LOG(\ **x**, **y**)  | returns the logarithm of **y** |
|                      | to the base **x**              |
+----------------------+--------------------------------+
| MOD(\ **x**, **y**)  | returns the remainder of **x** |
|                      | divided by **y**               |
+----------------------+--------------------------------+
| PI()                 | returns the value of pi        |
+----------------------+--------------------------------+
| POWER(\ **x**,       | returns the value of **x**     |
| **y**)               | raised to the power of **y**   |
+----------------------+--------------------------------+
| RADIANS(\ **x**)     | converts degrees to radians    |
+----------------------+--------------------------------+
| ROUND(\ **x**,       | rounds **x** to **y** decimal  |
| **y**)               | places                         |
+----------------------+--------------------------------+
| SIGN(\ **x**)        | returns the sign of **x** as   |
|                      | -1, 0, 1 if **x** is negative, |
|                      | zero, or positive              |
+----------------------+--------------------------------+
| TRUNCATE(\ **x**,    | truncates **x** to **y**       |
| **y**)               | decimal places                 |
+----------------------+--------------------------------+

Trigonometric Function Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------+----------------------------------------------+
| Function                | Description                                  |
+=========================+==============================================+
| ACOS(\ **x**)           | returns the arc cosine of **x**              |
+-------------------------+----------------------------------------------+
| ASIN(\ **x**)           | returns the arc sine of **x**                |
+-------------------------+----------------------------------------------+
| ATAN(\ **x**)           | returns the arc tangent of **x**             |
+-------------------------+----------------------------------------------+
| ATAN2(\ **x**, **y**)   | returns the arc tangent of **x** and **y**   |
+-------------------------+----------------------------------------------+
| COS(\ **x**)            | returns the cosine of **x**                  |
+-------------------------+----------------------------------------------+
| COT(\ **x**)            | returns the cotangent of **x**               |
+-------------------------+----------------------------------------------+
| SIN(\ **x**)            | returns the sine of **x**                    |
+-------------------------+----------------------------------------------+
| TAN(\ **x**)            | returns the tangent of **x**                 |
+-------------------------+----------------------------------------------+

Geometric Function Support
~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------------------------------------------+-----------------------+
| Function                                           | Description           |
+====================================================+=======================+
| DISTANCE\_IN\_METERS(\ **fromLon**, **fromLat**,   | calculate distance in |
| **toLon**, **toLat**)                              | meters between two    |
|                                                    | WGS-84 positions      |
+----------------------------------------------------+-----------------------+

String Function Support
~~~~~~~~~~~~~~~~~~~~~~~

+---------------------------+------------------------------------------------+
| Function                  | Description                                    |
+===========================+================================================+
| CHAR\_LENGTH(\ **str**)   | returns the number of characters in a string   |
+---------------------------+------------------------------------------------+
| LENGTH(\ **str**)         | returns the length of a string in bytes        |
+---------------------------+------------------------------------------------+

Pattern Matching Support
~~~~~~~~~~~~~~~~~~~~~~~~

+------------------------------------+------------------------+---------------------+
| Name                               | Example                | Description         |
+====================================+========================+=====================+
| **str** LIKE **pattern**           | ``'ab' LIKE 'ab'``     | returns true if the |
|                                    |                        | string matches the  |
|                                    |                        | pattern             |
+------------------------------------+------------------------+---------------------+
| **str** NOT LIKE **pattern**       | ``'ab' NOT LIKE 'cd'`` | returns true if the |
|                                    |                        | string does not     |
|                                    |                        | match the pattern   |
+------------------------------------+------------------------+---------------------+
| **str** ILIKE **pattern**          | ``'AB' ILIKE 'ab'``    | case-insensitive    |
|                                    |                        | LIKE                |
+------------------------------------+------------------------+---------------------+
| **str** REGEXP **POSIX pattern**   | ``'^[a-z]+r$'``        | lower case string   |
|                                    |                        | ending with r       |
+------------------------------------+------------------------+---------------------+
| REGEXP\_LIKE ( **str** , **POSIX   | ``'^[hc]at'``          | cat or hat          |
| pattern** )                        |                        |                     |
+------------------------------------+------------------------+---------------------+

Wildcard characters supported by ``LIKE`` and ``ILIKE``:

``%`` matches any number of characters, including zero characters

``_`` matches exactly one character

Date/Time Function Support
~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------------------+--------------------------------------+
| Function                            | Description                          |
+=====================================+======================================+
| DATE\_TRUNC(\ **date\_part**,       | truncates the **timestamp** to the   |
| **timestamp**)                      | specified **date\_part**             |
+-------------------------------------+--------------------------------------+
| EXTRACT(\ **date\_part** FROM       | returns the specified **date\_part** |
| **timestamp**)                      | from provided **timestamp**          |
+-------------------------------------+--------------------------------------+
| NOW()                               | returns the current timestamp        |
+-------------------------------------+--------------------------------------+

Supported **date\_part** types:

::

    DATE_TRUNC [YEAR, QUARTER, MONTH, DAY, HOUR, MINUTE, SECOND,
                MILLENNIUM, CENTURY, DECADE, WEEK, QUARTERDAY]
    EXTRACT [YEAR, QUARTER, MONTH, DAY, HOUR, MINUTE, SECOND,
             DOW, ISODOW, DOY, EPOCH, QUARTERDAY, WEEK]

Accepted date, time, and timestamp formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------+--------------------+----------------------------+
| Datatype    | Formats            | Examples                   |
+=============+====================+============================+
| DATE        | YYYY-MM-DD         | 2013-10-31                 |
+-------------+--------------------+----------------------------+
| DATE        | MM/DD/YYYY         | 10/31/2013                 |
+-------------+--------------------+----------------------------+
| DATE        | DD-MON-YY          | 31-Oct-13                  |
+-------------+--------------------+----------------------------+
| DATE        | DD/Mon/YYYY        | 31/Oct/2013                |
+-------------+--------------------+----------------------------+
| TIME        | HHMMSS             | 234901                     |
+-------------+--------------------+----------------------------+
| TIME        | HH:MM:SS           | 23:49:01                   |
+-------------+--------------------+----------------------------+
| TIMESTAMP   | DATE TIME          | 31-Oct-13 23:49:01         |
+-------------+--------------------+----------------------------+
| TIMESTAMP   | DATETTIME          | 31-Oct-13T23:49:01         |
+-------------+--------------------+----------------------------+
| TIMESTAMP   | DATE:TIME          | 11/31/2013:234901          |
+-------------+--------------------+----------------------------+
| TIMESTAMP   | DATE TIME ZONE     | 31-Oct-13 11:30:25 -0800   |
+-------------+--------------------+----------------------------+
| TIMESTAMP   | DATE HH.MM.SS PM   | 31-Oct-13 11.30.25pm       |
+-------------+--------------------+----------------------------+
| TIMESTAMP   | DATE HH:MM:SS PM   | 31-Oct-13 11:30:25pm       |
+-------------+--------------------+----------------------------+
| TIMESTAMP   |                    | 1383262225                 |
+-------------+--------------------+----------------------------+

**Usage Notes**

- For two-digit years, years 69-99 are assumed to be previous century (e.g.
  1969), and 0-68 are assumed to be current century (016).
- For four-digit years, negative years (e.g. BC) are not supported.
- Hours are expressed in 24-hour format.
- When time components are separated by colons, they maybe be written as one or
  two digits.
- Months are case insensitive, and can be spelled out or abbreviated to three
  characters.
- For timestamps, decimal seconds are ignored. Time zone offsets are written as
  +/-HHMM.
- For timestamps, a numeric string is converted to +/- seconds since January 1,
  1970.
- On output, dates are formatted as YYYY-MM-DD. Times are formatted as HH:MM:SS.

Aggregate Function Support
~~~~~~~~~~~~~~~~~~~~~~~~~~

+----------------+----------------------------------------------------+
| Function       | Description                                        |
+================+====================================================+
| AVG(\ **x**)   | returns the average value of **x**                 |
+----------------+----------------------------------------------------+
| COUNT()        | returns the count of the number of rows returned   |
+----------------+----------------------------------------------------+
| MAX(\ **x**)   | returns the maximum value of **x**                 |
+----------------+----------------------------------------------------+
| MIN(\ **x**)   | returns the minimum value of **x**                 |
+----------------+----------------------------------------------------+
| SUM(\ **x**)   | returns the sum of the values of **x**             |
+----------------+----------------------------------------------------+

Conditional Expression Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+-------------------------------------------+------------------------------------------------+
| Expression                                | Description                                    |
+===========================================+================================================+
| CASE WHEN **condition** THEN **result**   | Case operator                                  |
+-------------------------------------------+------------------------------------------------+
| COALESCE(\ **val1**, **val2**, ..)        | returns the first non-null value in the list   |
+-------------------------------------------+------------------------------------------------+

Subquery Expression Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

+------------+------------------------------------------+-----------------------+
| Expression | Example                                  | Description           |
+============+==========================================+=======================+
| EXISTS     | EXISTS (**subquery**)                    | evaluates whether the |
|            |                                          | subquery returns rows |
+------------+------------------------------------------+-----------------------+
| IN         | **expr** IN (**subquery** or **list of   | evaluates whether     |
|            | values**)                                | **expr** equals any   |
|            |                                          | value of the IN list  |
+------------+------------------------------------------+-----------------------+
| NOT IN     | **expr** NOT IN (**subquery** or **list  | evaluates whether     |
|            | of values**)                             | **expr** does not     |
|            |                                          | equal any value of    |
|            |                                          | the IN list           |
+------------+------------------------------------------+-----------------------+

**Usage notes**

- A subquery may be used anywhere an expression may be used, subject to any
  run-time constraints of that expression. For example, a subquery in a CASE
  statement must return exactly one row, but a subquery may return multiple
  values to an IN expression.
- A subquery may be used anywhere a table is allowed (e.g. **FROM subquery**),
  making use of aliases to name any reference to the table and columns returned
  by the subquery.

Type Cast Support
~~~~~~~~~~~~~~~~~

+---------------------+------------------------+--------------------------------+
| Expression          | Example                | Description                    |
+=====================+========================+================================+
| CAST(\ **expr** AS  | CAST(1.25 AS FLOAT)    | converts an expression to      |
| **type**)           |                        | another data type              |
+---------------------+------------------------+--------------------------------+

Array Support
~~~~~~~~~~~~~

+-------------------------------------+-------------------------------------------------+
| Expression                          | Description                                     |
+=====================================+=================================================+
| ``SELECT <ArrayCol>[n] ...``        | Query array elements n of column ``ArrayCol``   |
+-------------------------------------+-------------------------------------------------+
| ``SELECT UNNEST(<ArrayCol>) ...``   | Flatten entire array ``ArrayCol``               |
+-------------------------------------+-------------------------------------------------+
