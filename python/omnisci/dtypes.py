"""
https://www.python.org/dev/peps/pep-0249/#type-objects
"""
import datetime
import time
from omnisci.common.ttypes import TDatumType

T = TDatumType


class DataType:
    def __init__(self, matches):
        self._matches = set(matches)

    def __eq__(self, other):
        return other in self._matches

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash(tuple(self._matches))


Binary = memoryview
Date = datetime.date
Time = datetime.time
Timestamp = datetime.datetime

BINARY = DataType([])
STRING = DataType(
    [
        T.STR,
        T.POINT,
        T.LINESTRING,
        T.POLYGON,
        T.MULTIPOLYGON,
        T.GEOMETRY,
    ]
)
NUMBER = DataType(
    [
        T.SMALLINT,
        T.INT,
        T.BIGINT,
        T.FLOAT,
        T.DECIMAL,
        T.DOUBLE,
        T.BOOL,
        T.TINYINT,
    ]
)
DATETIME = DataType([T.DATE, T.TIME, T.TIMESTAMP])
ROWID = DataType([])


def DateFromTicks(ticks):
    return Date(*time.localtime(ticks)[:3])


def TimeFromTicks(ticks):
    return Time(*time.localtime(ticks)[3:6])


def TimestampFromTicks(ticks):
    return Timestamp(*time.localtime(ticks)[:6])
