import os
import subprocess
import time
from uuid import uuid4

import pytest
from thrift.transport import TSocket, TTransport
from thrift.transport.TSocket import TTransportException
from omnisci import connect
import random
import string

omniscihost = os.environ.get('OMNISCI_HOST', 'localhost')

def _check_open():
    """
    Test to see if OmniSci running on localhost and socket open
    """
    socket = TSocket.TSocket(omniscihost, 6274)
    transport = TTransport.TBufferedTransport(socket)

    try:
        transport.open()
        return True
    except TTransportException:
        return False


@pytest.fixture(scope='session')
def omnisci_server():
    """Ensure an omnisci server is running, optionally starting one if none"""
    if _check_open():
        # already running before pytest started
        pass
    else:
        # not yet running...
        subprocess.check_output(
            [
                'docker',
                'run',
                '-d',
                '--ipc=host',
                '-v',
                '/dev:/dev',
                '-p',
                '6274:6274',
                '-p',
                '9092:9092',
                '--name=mapd',
                'omnisci/core-os-cpu:latest',
            ]
        )
        # yield and stop afterwards?
        assert _check_open()
        # Takes some time to start up. Unfortunately even trying to connect
        # will cause it to hang.
        time.sleep(5)


@pytest.fixture(scope='session')
def con(omnisci_server):
    """
    Fixture to provide Connection for tests run against live OmniSci instance
    """
    return connect(
        user="admin",
        password='HyperInteractive',
        host=omniscihost,
        port=6274,
        protocol='binary',
        dbname='omnisci',
    )


@pytest.fixture
def mock_client(mocker):
    """A magicmock for omnisci.connection.Client"""
    return mocker.patch("omnisci.connection.Client")


def no_gpu():
    """Check for the required GPU dependencies"""
    try:
        from numba import cuda
        import cudf  # noqa

        try:
            cuda.select_device(0)
        except cuda.cudadrv.error.CudaDriverError:
            return True
    except ImportError:
        return True
    return False


def gen_string():
    """Generate a random string sequence for use in _tests_table_no_nulls"""
    return ''.join(
        [
            random.choice(string.ascii_letters + string.digits)
            for n in range(10)
        ]
    )


@pytest.fixture
def tmp_table(con) -> str:
    table_name = 'table_{}'.format(uuid4().hex)
    con.execute("drop table if exists {};".format(table_name))

    try:
        yield table_name
    finally:
        con.execute("drop table if exists {};".format(table_name))
