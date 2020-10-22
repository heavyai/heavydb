import omnisci.thrift.ttypes as T
from .exceptions import _translate_exception
from ._parsers import _bind_parameters, _extract_description, _extract_col_vals


class Cursor:
    """A database cursor."""

    def __init__(self, connection):
        # XXX: supposed to share state between cursors of the same connection
        self.connection = connection
        self.rowcount = -1
        self._description = None
        self._arraysize = 1
        self._result = None
        self._result_set = None
        if connection:
            connection.register_runtime_udfs()

    def __iter__(self):
        if self.result_set is None:
            return iter([])
        return self.result_set

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def description(self):
        """
        Read-only sequence describing columns of the result set.
        Each column is an instance of `Description` describing

        - name
        - type_code
        - display_size
        - internal_size
        - precision
        - scale
        - null_ok

        We only use name, type_code, and null_ok; The rest are always ``None``
        """
        return self._description

    @property
    def result_set(self):
        return self._result_set

    @property
    def arraysize(self):
        """The number of rows to fetch at a time with `fetchmany`. Default 1.

        See Also
        --------
        fetchmany
        """
        return self._arraysize

    @arraysize.setter
    def arraysize(self, value):
        """Number of items to fetch with :func:`fetchmany`."""
        if not isinstance(value, int):
            raise TypeError(
                "Value must be an integer, got {} instead".format(type(value))
            )
        self._arraysize = value

    def close(self):
        """Close this cursor."""
        # TODO
        pass

    def execute(self, operation, parameters=None):
        """Execute a SQL statement.

        Parameters
        ----------
        operation: str
            A SQL query
        parameters: dict
            Parameters to substitute into ``operation``.

        Returns
        -------
        self : Cursor

        Examples
        --------
        >>> c = conn.cursor()
        >>> c.execute("select symbol, qty from stocks")
        >>> list(c)
        [('RHAT', 100.0), ('IBM', 1000.0), ('MSFT', 1000.0), ('IBM', 500.0)]

        Passing in ``parameters``:

        >>> c.execute("select symbol qty from stocks where qty <= :max_qty",
        ...           parameters={"max_qty": 500})
        [('RHAT', 100.0), ('IBM', 500.0)]
        """

        # https://github.com/omnisci/pymapd/issues/263
        operation = operation.strip()

        if parameters is not None:
            operation = str(_bind_parameters(operation, parameters))
        self.rowcount = -1
        try:
            result = self.connection._client.sql_execute(
                self.connection._session,
                operation,
                column_format=True,
                nonce=None,
                first_n=-1,
                at_most_n=-1,
            )
        except T.TOmniSciException as e:
            raise _translate_exception(e) from e
        self._description = _extract_description(result.row_set.row_desc)

        try:
            self.rowcount = len(result.row_set.columns[0].nulls)
        except IndexError:
            pass

        self._result_set = make_row_results_set(result)
        self._result = result
        return self

    def executemany(self, operation, parameters):
        """Execute a SQL statement for many sets of parameters.

        Parameters
        ----------
        operation: str
        parameters: list of dict

        Returns
        -------
        results: list of lists
        """
        results = [
            list(self.execute(operation, params)) for params in parameters
        ]
        return results

    def fetchone(self):
        """Fetch a single row from the results set"""
        try:
            return next(self.result_set)
        except StopIteration:
            return None

    def fetchmany(self, size=None):
        """Fetch ``size`` rows from the results set."""
        if size is None:
            size = self.arraysize
        results = [self.fetchone() for _ in range(size)]
        return [x for x in results if x is not None]

    def fetchall(self):
        return list(self)

    def setinputsizes(self, sizes):
        pass

    def setoutputsizes(self, size, column=None):
        pass


# -----------------------------------------------------------------------------
# Result Sets
# -----------------------------------------------------------------------------


def make_row_results_set(data):
    """
    Build a results set of python objects.

    Parameters
    ----------
    data: QueryResultSet

    Returns
    -------
    results: Iterator[tuple]
    """

    if data.row_set.columns:
        nrows = len(data.row_set.columns[0].nulls)
        ncols = len(data.row_set.row_desc)
        columns = [
            _extract_col_vals(desc, col)
            for desc, col in zip(data.row_set.row_desc, data.row_set.columns)
        ]
        for i in range(nrows):
            yield tuple(columns[j][i] for j in range(ncols))
