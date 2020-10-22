from pkg_resources import get_distribution, DistributionNotFound

# module constants
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

# clean up
del get_distribution, DistributionNotFound

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
