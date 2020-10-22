import os
import pytest
from omnisci.thrift.ttypes import TColumnType
from omnisci.common.ttypes import TTypeInfo
from omnisci import OperationalError, connect
from omnisci.connection import _parse_uri, ConnectionInfo
from omnisci.exceptions import Error
from omnisci._parsers import ColumnDetails, _extract_column_details

omniscihost = os.environ.get('OMNISCI_HOST', 'localhost')

@pytest.mark.usefixtures("omnisci_server")
class TestConnect:
    def test_host_specified(self):
        with pytest.raises(TypeError):
            connect(user='foo')

    def test_raises_right_exception(self):
        with pytest.raises(OperationalError):
            connect(host=omniscihost, protocol='binary', port=1234)

    def test_close(self):
        conn = connect(
            user='admin',
            password='HyperInteractive',
            host=omniscihost,
            dbname='omnisci',
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
                host=omniscihost,
                dbname='dbname',
                protocol='fake-proto',
            )
        assert m.match('fake-proto')

    def test_session_logon_success(self):
        conn = connect(
            user='admin',
            password='HyperInteractive',
            host=omniscihost,
            dbname='omnisci',
        )
        sessionid = conn._session
        connnew = connect(sessionid=sessionid, host=omniscihost)
        assert connnew._session == sessionid

    def test_session_logon_failure(self):
        sessionid = 'ILoveDancingOnTables'
        with pytest.raises(Error):
            connect(sessionid=sessionid, host=omniscihost, protocol='binary')

    def test_bad_binary_encryption_params(self):
        with pytest.raises(TypeError):
            connect(
                user='admin',
                host=omniscihost,
                dbname='omnisci',
                protocol='http',
                validate=False,
            )


class TestURI:
    def test_parse_uri(self):
        uri = (
            'omnisci://admin:HyperInteractive@{0}:6274/omnisci?'
            'protocol=binary'.format(omniscihost)
        )
        result = _parse_uri(uri)
        expected = ConnectionInfo(
            "admin",
            "HyperInteractive",
            omniscihost,
            6274,
            "omnisci",
            "binary",
            None,
            None,
        )
        assert result == expected

    def test_both_raises(self):
        uri = (
            'omnisci://admin:HyperInteractive@{0}:6274/omnisci?'
            'protocol=binary'.format(omniscihost)
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
