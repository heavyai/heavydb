"""
Tests that rely on a server running
"""
import base64
import json
import datetime
import os

import pytest
from omnisci import connect, ProgrammingError, DatabaseError
from omnisci.cursor import Cursor
from omnisci._parsers import Description, ColumnDetails
from omnisci.thrift.ttypes import TOmniSciException

# XXX: Make it hashable to silence warnings; see if this can be done upstream
# This isn't a huge deal, but our testing context mangers for asserting
# exceptions need hashability
TOmniSciException.__hash__ = lambda x: id(x)
omniscihost = os.environ.get('OMNISCI_HOST', 'localhost')


@pytest.mark.usefixtures("omnisci_server")
class TestIntegration:
    def test_connect_binary(self):
        con = connect(
            user="admin",
            password='HyperInteractive',
            host=omniscihost,
            port=6274,
            protocol='binary',
            dbname='omnisci',
        )
        assert con is not None

    def test_connect_http(self):
        con = connect(
            user="admin",
            password='HyperInteractive',
            host=omniscihost,
            port=6278,
            protocol='http',
            dbname='omnisci',
        )
        assert con is not None

    def test_connect_uri(self):
        uri = (
            'omnisci://admin:HyperInteractive@{0}:6274/omnisci?'
            'protocol=binary'.format(omniscihost)
        )
        con = connect(uri=uri)
        assert con._user == 'admin'
        assert con._password == 'HyperInteractive'
        assert con._host == omniscihost
        assert con._port == 6274
        assert con._dbname == 'omnisci'
        assert con._protocol == 'binary'

    def test_connect_uri_and_others_raises(self):
        uri = (
            'omnisci://admin:HyperInteractive@{0}:6274/omnisci?'
            'protocol=binary'.format(omniscihost)
        )
        with pytest.raises(TypeError):
            connect(username='omnisci', uri=uri)

    def test_invalid_sql(self, con):
        with pytest.raises(ProgrammingError) as r:
            con.cursor().execute("this is invalid;")
        r.match("SQL Error:")

    def test_nonexistant_table(self, con):
        with pytest.raises(DatabaseError) as r:
            con.cursor().execute("select it from fake_table;")
        r.match("Table 'FAKE_TABLE' does not exist|Object 'fake_table' not")

    def test_connection_execute(self, con):
        result = con.execute("drop table if exists FOO;")
        result = con.execute("create table FOO (a int);")
        assert isinstance(result, Cursor)
        con.execute("drop table if exists FOO;")

    def test_select_sets_description(self, con):

        c = con.cursor()
        c.execute('drop table if exists stocks;')
        create = (
            'create table stocks (date_ text, trans text, symbol text, '
            'qty int, price float, vol float);'
        )
        c.execute(create)
        i1 = "INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14,1.1);"  # noqa
        i2 = "INSERT INTO stocks VALUES ('2006-01-05','BUY','GOOG',100,12.14,1.2);"  # noqa

        c.execute(i1)
        c.execute(i2)

        c.execute("select * from stocks")
        expected = [
            Description('date_', 6, None, None, None, None, True),
            Description('trans', 6, None, None, None, None, True),
            Description('symbol', 6, None, None, None, None, True),
            Description('qty', 1, None, None, None, None, True),
            Description('price', 3, None, None, None, None, True),
            Description('vol', 3, None, None, None, None, True),
        ]
        assert c.description == expected
        c.execute('drop table if exists stocks;')

    def test_select_parametrized(self, con):

        c = con.cursor()
        c.execute('drop table if exists stocks;')
        create = (
            'create table stocks (date_ text, trans text, symbol text, '
            'qty int, price float, vol float);'
        )
        c.execute(create)
        i1 = "INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14,1.1);"  # noqa
        i2 = "INSERT INTO stocks VALUES ('2006-01-05','BUY','GOOG',100,12.14,1.2);"  # noqa

        c.execute(i1)
        c.execute(i2)

        c.execute(
            'select symbol, qty from stocks where symbol = :symbol',
            {'symbol': 'GOOG'},
        )
        result = list(c)
        expected = [
            ('GOOG', 100),
        ]  # noqa
        assert result == expected
        c.execute('drop table if exists stocks;')

    def test_executemany_parametrized(self, con):

        c = con.cursor()
        c.execute('drop table if exists stocks;')
        create = (
            'create table stocks (date_ text, trans text, symbol text, '
            'qty int, price float, vol float);'
        )
        c.execute(create)
        i1 = "INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14,1.1);"  # noqa
        i2 = "INSERT INTO stocks VALUES ('2006-01-05','BUY','GOOG',100,12.14,1.2);"  # noqa

        c.execute(i1)
        c.execute(i2)

        parameters = [{'symbol': 'GOOG'}, {'symbol': "RHAT"}]
        expected = [[('GOOG', 100)], [('RHAT', 100)]]
        query = 'select symbol, qty from stocks where symbol = :symbol'
        c = con.cursor()
        result = c.executemany(query, parameters)
        assert result == expected
        c.execute('drop table if exists stocks;')

    def test_executemany_parametrized_insert(self, con):

        c = con.cursor()
        c.execute('drop table if exists stocks;')
        create = (
            'create table stocks (date_ text, trans text, symbol text, '
            'qty int, price float, vol float);'
        )
        c.execute(create)
        i1 = "INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14,1.1);"  # noqa
        i2 = "INSERT INTO stocks VALUES ('2006-01-05','BUY','GOOG',100,12.14,1.2);"  # noqa

        c.execute(i1)
        c.execute(i2)

        c = con.cursor()
        c.execute("drop table if exists stocks2;")
        # Create table
        c.execute('CREATE TABLE stocks2 (symbol text, qty int);')
        params = [{"symbol": "GOOG", "qty": 10}, {"symbol": "AAPL", "qty": 20}]
        query = "INSERT INTO stocks2 VALUES (:symbol, :qty);"
        result = c.executemany(query, params)
        assert result == [[], []]  # TODO: not sure if this is standard
        c.execute("drop table stocks2;")
        c.execute('drop table if exists stocks;')

    def test_fetchone(self, con):

        c = con.cursor()
        c.execute('drop table if exists stocks;')
        create = (
            'create table stocks (date_ text, trans text, symbol text, '
            'qty int, price float, vol float);'
        )
        c.execute(create)
        i1 = "INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14,1.1);"  # noqa
        i2 = "INSERT INTO stocks VALUES ('2006-01-05','BUY','GOOG',100,12.14,1.2);"  # noqa

        c.execute(i1)
        c.execute(i2)

        c.execute("select symbol, qty from stocks")
        result = c.fetchone()
        expected = ('RHAT', 100)
        assert result == expected
        c.execute('drop table if exists stocks;')

    def test_fetchmany(self, con):

        c = con.cursor()
        c.execute('drop table if exists stocks;')
        create = (
            'create table stocks (date_ text, trans text, symbol text, '
            'qty int, price float, vol float);'
        )
        c.execute(create)
        i1 = "INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14,1.1);"  # noqa
        i2 = "INSERT INTO stocks VALUES ('2006-01-05','BUY','GOOG',100,12.14,1.2);"  # noqa

        c.execute(i1)
        c.execute(i2)

        c.execute("select symbol, qty from stocks")
        result = c.fetchmany()
        expected = [('RHAT', 100)]
        assert result == expected

        c.execute("select symbol, qty from stocks")
        result = c.fetchmany(size=10)
        expected = [('RHAT', 100), ('GOOG', 100)]
        assert result == expected
        c.execute('drop table if exists stocks;')

    def test_select_dates(self, con):

        c = con.cursor()
        c.execute('drop table if exists dates;')
        c.execute(
            'create table dates (date_ DATE, datetime_ TIMESTAMP, '
            'time_ TIME);'
        )
        i1 = (
            "INSERT INTO dates VALUES ('2006-01-05','2006-01-01T12:00:00',"
            "'12:00:00');"
        )
        i2 = (
            "INSERT INTO dates VALUES ('1901-12-14','1901-12-13T20:45:53',"
            "'23:59:00');"
        )
        c.execute(i1)
        c.execute(i2)

        result = list(c.execute("select * from dates"))
        expected = [
            (
                datetime.date(2006, 1, 5),
                datetime.datetime(2006, 1, 1, 12),
                datetime.time(12),
            ),
            (
                datetime.date(1901, 12, 14),
                datetime.datetime(1901, 12, 13, 20, 45, 53),
                datetime.time(23, 59),
            ),
        ]
        assert result == expected
        c.execute('drop table if exists dates;')


class TestExtras:
    def test_sql_validate(self, con):
        from omnisci.common.ttypes import TTypeInfo

        c = con.cursor()
        c.execute('drop table if exists stocks;')
        create = (
            'create table stocks (date_ text, trans text, symbol text, '
            'qty int, price float, vol float);'
        )
        c.execute(create)

        q = "select * from stocks"
        results = con._client.sql_validate(con._session, q)
        col_names = sorted([r.col_name for r in results])
        col_types = [r.col_type for r in results]

        expected_col_names = [
            'date_',
            'price',
            'qty',
            'symbol',
            'trans',
            'vol',
        ]

        expected_types = [
            TTypeInfo(
                type=6,
                encoding=4,
                nullable=True,
                is_array=False,
                precision=0,
                scale=0,
                comp_param=32,
                size=-1,
            ),
            TTypeInfo(
                type=6,
                encoding=4,
                nullable=True,
                is_array=False,
                precision=0,
                scale=0,
                comp_param=32,
                size=-1,
            ),
            TTypeInfo(
                type=6,
                encoding=4,
                nullable=True,
                is_array=False,
                precision=0,
                scale=0,
                comp_param=32,
                size=-1,
            ),
            TTypeInfo(
                type=1,
                encoding=0,
                nullable=True,
                is_array=False,
                precision=0,
                scale=0,
                comp_param=0,
                size=-1,
            ),
            TTypeInfo(
                type=3,
                encoding=0,
                nullable=True,
                is_array=False,
                precision=0,
                scale=0,
                comp_param=0,
                size=-1,
            ),
            TTypeInfo(
                type=3,
                encoding=0,
                nullable=True,
                is_array=False,
                precision=0,
                scale=0,
                comp_param=0,
                size=-1,
            ),
        ]

        assert col_types == expected_types
        assert col_names == expected_col_names
