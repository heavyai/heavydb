import os
import pytest
from heavydb.thrift.ttypes import TColumnType
from heavydb.common.ttypes import TTypeInfo
from heavydb import OperationalError, connect
from heavydb.connection import _parse_uri, ConnectionInfo
from heavydb.exceptions import Error
from heavydb._parsers import ColumnDetails, _extract_column_details

heavydb_host = os.environ.get('HEAVYDB_HOST', 'localhost')


@pytest.mark.usefixtures("heavydb_server")
class TestConnect:
    def test_host_specified(self):
        with pytest.raises(TypeError):
            connect(user='foo')

    def test_raises_right_exception(self):
        with pytest.raises(OperationalError):
            connect(host=heavydb_host, protocol='binary', port=1234)

    def test_close(self):
        conn = connect(
            user='admin',
            password='HyperInteractive',
            host=heavydb_host,
            dbname='heavyai',
        )
        assert conn.closed == 0
        conn.close()
        assert conn.closed == 1

    def test_commit_noop(self, con):
        result = con.commit()  # it worked
        assert result is None

    def test_bad_protocol(self, mock_client):
        with pytest.raises(ValueError) as m:
            connect(
                user='user',
                host=heavydb_host,
                dbname='dbname',
                protocol='fake-proto',
            )
        assert m.match('fake-proto')

    def test_session_logon_success(self):
        conn = connect(
            user='admin',
            password='HyperInteractive',
            host=heavydb_host,
            dbname='heavyai',
        )
        sessionid = conn._session
        connnew = connect(sessionid=sessionid, host=heavydb_host)
        assert connnew._session == sessionid

    def test_session_logon_failure(self):
        sessionid = 'ILoveDancingOnTables'
        with pytest.raises(Error):
            connect(sessionid=sessionid, host=heavydb_host, protocol='binary')

    def test_bad_binary_encryption_params(self):
        with pytest.raises(TypeError):
            connect(
                user='admin',
                host=heavydb_host,
                dbname='heavyai',
                protocol='http',
                validate=False,
            )


class TestURI:
    def test_parse_uri(self):
        uri = (
            'heavydb://admin:HyperInteractive@{0}:6274/heavyai?'
            'protocol=binary'.format(heavydb_host)
        )
        result = _parse_uri(uri)
        expected = ConnectionInfo(
            "admin",
            "HyperInteractive",
            heavydb_host,
            6274,
            "heavyai",
            "binary",
            None,
            None,
        )
        assert result == expected

    def test_both_raises(self):
        uri = (
            'heavydb://admin:HyperInteractive@{0}:6274/heavyai?'
            'protocol=binary'.format(heavydb_host)
        )
        with pytest.raises(TypeError):
            connect(uri=uri, user='my user')


class TestExtras:
    def test_extract_row_details(self):
        data = [
            TColumnType(
                col_name='date_',
                col_type=TTypeInfo(
                    type=6,
                    encoding=4,
                    nullable=True,
                    is_array=False,
                    precision=0,
                    scale=0,
                    comp_param=32,
                ),
                is_reserved_keyword=False,
                src_name='',
            ),
            TColumnType(
                col_name='trans',
                col_type=TTypeInfo(
                    type=6,
                    encoding=4,
                    nullable=True,
                    is_array=False,
                    precision=0,
                    scale=0,
                    comp_param=32,
                ),
                is_reserved_keyword=False,
                src_name='',
            ),
            TColumnType(
                col_name='symbol',
                col_type=TTypeInfo(
                    type=6,
                    encoding=4,
                    nullable=True,
                    is_array=False,
                    precision=0,
                    scale=0,
                    comp_param=32,
                ),
                is_reserved_keyword=False,
                src_name='',
            ),
            TColumnType(
                col_name='qty',
                col_type=TTypeInfo(
                    type=1,
                    encoding=0,
                    nullable=True,
                    is_array=False,
                    precision=0,
                    scale=0,
                    comp_param=0,
                ),
                is_reserved_keyword=False,
                src_name='',
            ),
            TColumnType(
                col_name='price',
                col_type=TTypeInfo(
                    type=3,
                    encoding=0,
                    nullable=True,
                    is_array=False,
                    precision=0,
                    scale=0,
                    comp_param=0,
                ),
                is_reserved_keyword=False,
                src_name='',
            ),
            TColumnType(
                col_name='vol',
                col_type=TTypeInfo(
                    type=3,
                    encoding=0,
                    nullable=True,
                    is_array=False,
                    precision=0,
                    scale=0,
                    comp_param=0,
                ),
                is_reserved_keyword=False,
                src_name='',
            ),
        ]
        result = _extract_column_details(data)

        expected = [
            ColumnDetails(
                name='date_',
                type='STR',
                nullable=True,
                precision=0,
                scale=0,
                comp_param=32,
                encoding='DICT',
                is_array=False,
            ),
            ColumnDetails(
                name='trans',
                type='STR',
                nullable=True,
                precision=0,
                scale=0,
                comp_param=32,
                encoding='DICT',
                is_array=False,
            ),
            ColumnDetails(
                name='symbol',
                type='STR',
                nullable=True,
                precision=0,
                scale=0,
                comp_param=32,
                encoding='DICT',
                is_array=False,
            ),
            ColumnDetails(
                name='qty',
                type='INT',
                nullable=True,
                precision=0,
                scale=0,
                comp_param=0,
                encoding='NONE',
                is_array=False,
            ),
            ColumnDetails(
                name='price',
                type='FLOAT',
                nullable=True,
                precision=0,
                scale=0,
                comp_param=0,
                encoding='NONE',
                is_array=False,
            ),
            ColumnDetails(
                name='vol',
                type='FLOAT',
                nullable=True,
                precision=0,
                scale=0,
                comp_param=0,
                encoding='NONE',
                is_array=False,
            ),
        ]
        assert result == expected
