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

    SELECT [ALL|DISTINCT] <expr> [AS [<alias>]], ... FROM <table> [,<table>]
      [WHERE <expr>]
      [GROUP BY <expr>, ...]
      [HAVING <expr>]
      [ORDER BY <expr>, ...]
      [LIMIT {<number>|ALL} [OFFSET <number> [ROWS]]];

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

Trignometric Function Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

+------------------------------------+-------------------+---------------------+
| Name                               | Example           | Description         |
+====================================+===================+=====================+
| **str** LIKE **pattern**           | ``'ab' LIKE 'ab'` | returns true if the |
|                                    | `                 | string matches the  |
|                                    |                   | pattern             |
+------------------------------------+-------------------+---------------------+
| **str** NOT LIKE **pattern**       | ``'ab' NOT LIKE ' | returns true if the |
|                                    | cd'``             | string does not     |
|                                    |                   | match the pattern   |
+------------------------------------+-------------------+---------------------+
| **str** ILIKE **pattern**          | ``'AB' ILIKE 'ab' | case-insensitive    |
|                                    | ``                | LIKE                |
+------------------------------------+-------------------+---------------------+
| **str** REGEXP **POSIX pattern**   | ``'^[a-z]+r$'``   | lower case string   |
|                                    |                   | ending with r       |
+------------------------------------+-------------------+---------------------+
| REGEXP\_LIKE ( **str** , **POSIX   | ``'^[hc]at'``     | cat or hat          |
| pattern** )                        |                   |                     |
+------------------------------------+-------------------+---------------------+

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
                MILLENIUM, CENTURY, DECADE, WEEK, QUARTERDAY]
    EXTRACT [YEAR, QUARTER, MONTH, DAY, HOUR, MINUTE, SECOND,
             DOW, ISODOW, DOY, EPOCH, QUARTERDAY, WEEK]

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

+-----------+------------------------------------------+-----------------------+
| Expressio | Example                                  | Description           |
| n         |                                          |                       |
+===========+==========================================+=======================+
| EXISTS    | EXISTS (**subquery**)                    | evaluates whether the |
|           |                                          | subquery returns rows |
+-----------+------------------------------------------+-----------------------+
| IN        | **expr** IN (**subquery** or **list of   | evaluates whether     |
|           | values**)                                | **expr** equals any   |
|           |                                          | value of the IN list  |
+-----------+------------------------------------------+-----------------------+
| NOT IN    | **expr** NOT IN (**subquery** or **list  | evaluates whether     |
|           | of values**)                             | **expr** does not     |
|           |                                          | equal any value of    |
|           |                                          | the IN list           |
+-----------+------------------------------------------+-----------------------+

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
