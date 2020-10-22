"""

Connect to an OmniSci database.
"""
from collections import namedtuple
import base64
from sqlalchemy.engine.url import make_url
from thrift.protocol import TBinaryProtocol, TJSONProtocol
from thrift.transport import TSocket, TSSLSocket, THttpClient, TTransport
from thrift.transport.TSocket import TTransportException
from omnisci.thrift.OmniSci import Client
from omnisci.thrift.ttypes import TOmniSciException

from .cursor import Cursor
from .exceptions import _translate_exception, OperationalError

from ._parsers import _extract_column_details

from ._samlutils import get_saml_response

from packaging.version import Version


ConnectionInfo = namedtuple(
    "ConnectionInfo",
    [
        'user',
        'password',
        'host',
        'port',
        'dbname',
        'protocol',
        'bin_cert_validate',
        'bin_ca_certs',
    ],
)


def connect(
    uri=None,
    user=None,
    password=None,
    host=None,
    port=6274,
    dbname=None,
    protocol='binary',
    sessionid=None,
    bin_cert_validate=None,
    bin_ca_certs=None,
    idpurl=None,
    idpformusernamefield='username',
    idpformpasswordfield='password',
    idpsslverify=True,
):
    """
    Create a new Connection.

    Parameters
    ----------
    uri: str
    user: str
    password: str
    host: str
    port: int
    dbname: str
    protocol: {'binary', 'http', 'https'}
    sessionid: str
    bin_cert_validate: bool, optional, binary encrypted connection only
        Whether to continue if there is any certificate error
    bin_ca_certs: str, optional, binary encrypted connection only
        Path to the CA certificate file
    idpurl : str
        EXPERIMENTAL Enable SAML authentication by providing
        the logon page of the SAML Identity Provider.
    idpformusernamefield: str
        The HTML form ID for the username, defaults to 'username'.
    idpformpasswordfield: str
        The HTML form ID for the password, defaults to 'password'.
    idpsslverify: str
        Enable / disable certificate checking, defaults to True.

    Returns
    -------
    conn: Connection

    Examples
    --------
    You can either pass a string ``uri``, all the individual components,
    or an existing sessionid excluding user, password, and database

    >>> connect('mapd://admin:HyperInteractive@localhost:6274/omnisci?'
    ...         'protocol=binary')
    Connection(mapd://mapd:***@localhost:6274/mapd?protocol=binary)

    >>> connect(user='admin', password='HyperInteractive', host='localhost',
    ...         port=6274, dbname='omnisci')

    >>> connect(user='admin', password='HyperInteractive', host='localhost',
    ...         port=443, idpurl='https://sso.localhost/logon',
                protocol='https')

    >>> connect(sessionid='XihlkjhdasfsadSDoasdllMweieisdpo', host='localhost',
    ...         port=6273, protocol='http')

    """
    return Connection(
        uri=uri,
        user=user,
        password=password,
        host=host,
        port=port,
        dbname=dbname,
        protocol=protocol,
        sessionid=sessionid,
        bin_cert_validate=bin_cert_validate,
        bin_ca_certs=bin_ca_certs,
        idpurl=idpurl,
        idpformusernamefield=idpformusernamefield,
        idpformpasswordfield=idpformpasswordfield,
        idpsslverify=idpsslverify,
    )


def _parse_uri(uri):
    """
    Parse connection string

    Parameters
    ----------
    uri: str
        a URI containing connection information

    Returns
    -------
    info: ConnectionInfo

    Notes
    ------
    The URI may include information on

    - user
    - password
    - host
    - port
    - dbname
    - protocol
    - bin_cert_validate
    - bin_ca_certs
    """
    url = make_url(uri)
    user = url.username
    password = url.password
    host = url.host
    port = url.port
    dbname = url.database
    protocol = url.query.get('protocol', 'binary')
    bin_cert_validate = url.query.get('bin_cert_validate', None)
    bin_ca_certs = url.query.get('bin_ca_certs', None)

    return ConnectionInfo(
        user,
        password,
        host,
        port,
        dbname,
        protocol,
        bin_cert_validate,
        bin_ca_certs,
    )


class Connection:
    """Connect to your OmniSci database."""

    def __init__(
        self,
        uri=None,
        user=None,
        password=None,
        host=None,
        port=6274,
        dbname=None,
        protocol='binary',
        sessionid=None,
        bin_cert_validate=None,
        bin_ca_certs=None,
        idpurl=None,
        idpformusernamefield='username',
        idpformpasswordfield='password',
        idpsslverify=True,
    ):

        self.sessionid = None
        self._closed = 0
        if sessionid is not None:
            if any([user, password, uri, dbname, idpurl]):
                raise TypeError(
                    "Cannot specify sessionid with user, password,"
                    " dbname, uri, or idpurl"
                )
        if uri is not None:
            if not all(
                [
                    user is None,
                    password is None,
                    host is None,
                    port == 6274,
                    dbname is None,
                    protocol == 'binary',
                    bin_cert_validate is None,
                    bin_ca_certs is None,
                    idpurl is None,
                ]
            ):
                raise TypeError("Cannot specify both URI and other arguments")
            (
                user,
                password,
                host,
                port,
                dbname,
                protocol,
                bin_cert_validate,
                bin_ca_certs,
            ) = _parse_uri(uri)
        if host is None:
            raise TypeError("`host` parameter is required.")
        if protocol != 'binary' and not all(
            [bin_cert_validate is None, bin_ca_certs is None]
        ):
            raise TypeError(
                "Cannot specify bin_cert_validate or bin_ca_certs,"
                " without binary protocol"
            )
        if protocol in ("http", "https"):
            if not host.startswith(protocol):
                # the THttpClient expects http[s]://localhost
                host = '{0}://{1}'.format(protocol, host)
            transport = THttpClient.THttpClient("{}:{}".format(host, port))
            proto = TJSONProtocol.TJSONProtocol(transport)
            socket = None
        elif protocol == "binary":
            if any([bin_cert_validate is not None, bin_ca_certs]):
                socket = TSSLSocket.TSSLSocket(
                    host,
                    port,
                    validate=(bin_cert_validate),
                    ca_certs=bin_ca_certs,
                )
            else:
                socket = TSocket.TSocket(host, port)
            transport = TTransport.TBufferedTransport(socket)
            proto = TBinaryProtocol.TBinaryProtocolAccelerated(transport)
        else:
            raise ValueError(
                "`protocol` should be one of"
                " ['http', 'https', 'binary'],"
                " got {} instead".format(protocol),
            )
        self._user = user
        self._password = password
        self._host = host
        self._port = port
        self._dbname = dbname
        self._transport = transport
        self._protocol = protocol
        self._socket = socket
        self._tdf = None
        self._rbc = None
        try:
            self._transport.open()
        except TTransportException as e:
            if e.NOT_OPEN:
                err = OperationalError("Could not connect to database")
                raise err from e
            else:
                raise
        self._client = Client(proto)
        try:
            # If a sessionid was passed, we should validate it
            if sessionid:
                self._session = sessionid
                self._client.get_tables(self._session)
                self.sessionid = sessionid
            else:
                if idpurl:
                    self._user = ''
                    self._password = get_saml_response(
                        username=user,
                        password=password,
                        idpurl=idpurl,
                        userformfield=idpformusernamefield,
                        passwordformfield=idpformpasswordfield,
                        sslverify=idpsslverify,
                    )
                    self._dbname = ''
                    self._idpsslverify = idpsslverify
                    user = self._user
                    password = self._password
                    dbname = self._dbname

                self._session = self._client.connect(user, password, dbname)
        except TOmniSciException as e:
            raise _translate_exception(e) from e
        except TTransportException:
            raise ValueError(
                f"Connection failed with port {port} and "
                f"protocol '{protocol}'. Try port 6274 for "
                "protocol == binary or 6273, 6278 or 443 for "
                "http[s]"
            )

        # if OmniSci version <4.6, raise RuntimeError, as data import can be
        # incorrect for columnar date loads
        # Caused by https://github.com/omnisci/pymapd/pull/188
        semver = self._client.get_version()
        if Version(semver.split("-")[0]) < Version("4.6"):
            raise RuntimeError(
                f"Version {semver} of OmniSci detected. "
                "Please use pymapd <0.11. See release notes "
                "for more details."
            )

    def __repr__(self):
        tpl = (
            'Connection(omnisci://{user}:***@{host}:{port}/{dbname}?'
            'protocol={protocol})'
        )
        return tpl.format(
            user=self._user,
            host=self._host,
            port=self._port,
            dbname=self._dbname,
            protocol=self._protocol,
        )

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def closed(self):
        return self._closed

    def close(self):
        """Disconnect from the database unless created with sessionid"""
        if not self.sessionid and not self._closed:
            try:
                self._client.disconnect(self._session)
            except (TOmniSciException, AttributeError, TypeError):
                pass
        self._closed = 1
        self._rbc = None

    def commit(self):
        """This is a noop, as OmniSci does not provide transactions.

        Implemented to comply with the DBI specification.
        """
        return None

    def execute(self, operation, parameters=None):
        """Execute a SQL statement

        Parameters
        ----------
        operation: str
            A SQL statement to exucute

        Returns
        -------
        c: Cursor
        """
        c = Cursor(self)
        return c.execute(operation, parameters=parameters)

    def cursor(self):
        """Create a new :class:`Cursor` object attached to this connection."""
        return Cursor(self)

    def __call__(self, *args, **kwargs):
        """Runtime UDF decorator.

        The connection object can be applied to a Python function as
        decorator that will add the function to bending registration
        list.
        """
        try:
            from rbc.omniscidb import RemoteOmnisci
        except ImportError:
            raise ImportError("The 'rbc' package is required for `__call__`")
        if self._rbc is None:
            self._rbc = RemoteOmnisci(
                user=self._user,
                password=self._password,
                host=self._host,
                port=self._port,
                dbname=self._dbname,
            )
            self._rbc._session_id = self.sessionid
        return self._rbc(*args, **kwargs)

    def register_runtime_udfs(self):
        """Register any bending Runtime UDF functions in OmniSci server.

        If no Runtime UDFs have been defined, the call to this method
        is noop.
        """
        if self._rbc is not None:
            self._rbc.register()
