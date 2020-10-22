"""
Define exceptions as specified by the DB API 2.0 spec.

Includes some helper methods for translating thrift
exceptions to the ones defined here.
"""
from omnisci.thrift.ttypes import TOmniSciException


class Warning(Exception):
    """Emitted for important warnings, e.g. data truncatiions"""


class Error(Exception):
    """Base class for all pymapd errors."""


class InterfaceError(Error):
    """Raised whenever you use pymapd interface incorrectly."""


class DatabaseError(Error):
    """Raised when the database encounters an error."""


class DataError(DatabaseError):
    """Raised for data processing errors like division by zero, etc."""


class OperationalError(DatabaseError):
    """Raised for non-programmer related database errors, e.g.
    an unexpected disconnect.
    """


class IntegrityError(DatabaseError):
    """Raised when the relational integrity of the database is affected."""


class InternalError(DatabaseError):
    """Raised for errors internal to the database, e.g. and invalid cursor."""


class ProgrammingError(DatabaseError):
    """Raised for programming errors, e.g. syntax errors, table already
    exists.
    """


class NotSupportedError(DatabaseError):
    """Raised when an API not supported by the database is used."""


def _translate_exception(e):
    # type: (Exception) -> Exception
    """Translate a thrift-land exception to a DB-API 2.0
    exception.
    """
    # TODO: see if there's a way to get error codes, rather than relying msgs
    if not isinstance(e, TOmniSciException):
        return e
    if 'Validate failed' in e.error_msg or 'Parse failed' in e.error_msg:
        err = ProgrammingError
    elif 'Exception occurred' in e.error_msg:
        err = DatabaseError
    else:
        err = Error
    return err(e.error_msg)
