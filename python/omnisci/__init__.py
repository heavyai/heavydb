from .version import get_package_version
__version__ = get_package_version()
del get_package_version

apilevel = "2.0"
threadsafety = 2
paramstyle = "named"

from .connection import connect, Connection  # noqa

from .cursor import Cursor  # noqa

from .exceptions import (  # noqa
    Warning,
    Error,
    InterfaceError,
    DatabaseError,
    DataError,
    OperationalError,
    IntegrityError,
    InternalError,
    ProgrammingError,
    NotSupportedError,
)


from .dtypes import (  # noqa
    Binary,
    Date,
    Time,
    Timestamp,
    BINARY,
    STRING,
    NUMBER,
    DATETIME,
    ROWID,
    DateFromTicks,
    TimeFromTicks,
    TimestampFromTicks,
)
